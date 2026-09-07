"""Private shared runner used by the individual terrain-model scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from compare import available_methods, estimator_for
from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from datasets.supervision_masks import canonical_positive_mask_from_batch
from metrics.aggregator import EvaluationAggregator
from utils.config import jsonable_config, load_config
from utils.logging import write_rows
from utils.misc import atomic_write_json
from utils.raster_io import write_geotiff


FLOOD_SUPPORT = "valid_depth_mask"


def _dataset(config: Mapping[str, Any], split: str) -> FloodDepthDataset:
    contract = DatasetContract.load(config["dataset"]["contract"])
    return FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], split,
        band_spec=resolve_band_spec(config, contract),
        input_spec=ModelInputSpec.from_config(config),
        minimum_event_band_fraction=float(config["dataset"].get("minimum_event_band_fraction", 1.0)),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )


def _metadata_item(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping[key]
    if isinstance(value, (list, tuple)):
        return value[0]
    if isinstance(value, torch.Tensor):
        return value[0].item()
    return value


def _validate_config(config: Mapping[str, Any], method: str) -> None:
    compare = config.get("compare")
    if not isinstance(compare, Mapping):
        raise KeyError("Comparison configuration requires a <compare> section")
    if compare.get("method") != method:
        raise ValueError(
            f"Configuration is for {compare.get('method')!r}, not {method!r}"
        )
    if compare.get("flood_support") != FLOOD_SUPPORT:
        raise ValueError(f"Only {FLOOD_SUPPORT!r} is supported as flood range")
    if method not in available_methods():
        raise ValueError(f"Unknown comparison model {method!r}")


def run_model(
    config: Mapping[str, Any], split: str, method: str, output: Path,
    max_batches: int | None, save_predictions: bool,
) -> dict[str, Any]:
    """Evaluate one named terrain model using ``valid_depth_mask`` directly."""

    _validate_config(config, method)
    estimator = estimator_for(method)
    dataset = _dataset(config, split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), dataset.contract)
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    aggregator = EvaluationAggregator(depth_bins, primary_depth_bins=normalizer.train_depth_bins)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "resolved_config.json", jsonable_config(config))
    for batch_index, batch in enumerate(tqdm(loader, desc=f"{method} {split}")):
        if max_batches is not None and batch_index >= max_batches:
            break
        metadata = batch["metadata"]
        valid = canonical_positive_mask_from_batch(batch)[0, 0].numpy() > 0.5
        terrain = batch["terrain_raw"][0, 0].numpy().astype(np.float64)
        target = batch["label"][0, 0].numpy()
        output_valid = batch["validity"]["output_valid"][0, 0].numpy() > 0.5
        support = batch["masks"]["valid_depth_mask"][0, 0].numpy() > 0.5
        sample_id = str(_metadata_item(metadata, "sample_id"))
        event_id = str(_metadata_item(metadata, "source_event_id"))
        label_path = Path(str(_metadata_item(metadata, "label_path")))
        prediction = estimator(support, terrain)
        aggregator.add(sample_id, event_id, prediction, target, np.ones_like(prediction), valid)
        if save_predictions:
            with rasterio.open(label_path) as reference:
                write_geotiff(
                    output / "predicted_depth_m" / f"{sample_id}.tif", prediction,
                    crs=reference.crs, transform=reference.transform,
                    valid_mask=output_valid, descriptions=("predicted_depth_m",),
                )
    summary, samples, events, bins = aggregator.summarize()
    summary = {"method": method, "flood_support": FLOOD_SUPPORT, **summary}
    write_rows(output / "metrics_by_sample.csv", samples)
    write_rows(output / "metrics_by_event.csv", events)
    write_rows(output / "metrics_by_train_depth_bin.csv", bins)
    atomic_write_json(output / "summary.json", summary)
    return summary


def main_for_model(method: str, default_config: Path) -> int:
    """Run one named model from its model-specific executable wrapper."""

    parser = argparse.ArgumentParser(
        description=f"Evaluate {method} with valid_depth_mask as the fixed flood range."
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    summary = run_model(
        load_config(args.config), args.split, method, args.output.resolve(),
        args.max_batches, args.save_predictions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

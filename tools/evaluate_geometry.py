#!/usr/bin/env python3
"""Evaluate three terrain-depth methods using one frozen predicted flood extent."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
from tqdm import tqdm

from compare.geometry import GEOMETRY_METHODS, run_geometry_method
from datasets.contract import sha256_file
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from metrics.aggregator import EvaluationAggregator
from utils.config import jsonable_config, load_config
from utils.logging import setup_logging, write_rows
from utils.misc import atomic_write_json
from utils.raster_io import write_geotiff


PROTOCOL_NOTE = (
    "All geometry methods reuse one frozen AI4G-MobileNetV2-U-Net IoU extent. "
    "valid_depth_mask is used only as the positive-depth evaluation mask, never "
    "as a geometry-method input."
)


def _load_extent_product(
    extent_root: Path, dataset: FloodDepthDataset, config: dict[str, Any], split: str
) -> tuple[Path, dict[str, Any]]:
    root = extent_root.expanduser().resolve(strict=True)
    product_path = root / "extent_product.json"
    if not product_path.is_file():
        raise FileNotFoundError(f"Missing frozen extent manifest: {product_path}")
    import json

    product = json.loads(product_path.read_text(encoding="utf-8"))
    expected_fingerprint = {
        "contract_sha256": dataset.contract.hash,
        "manifest_sha256": sha256_file(dataset.contract.manifest_path),
        "normalization_sha256": sha256_file(Path(config["dataset"]["train_stats"])),
    }
    if product.get("dataset_fingerprint") != expected_fingerprint:
        raise RuntimeError("Predicted-extent product does not match the active dataset")
    if product.get("prediction_uses_valid_depth_mask") is not False:
        raise RuntimeError("Extent product does not certify label-independent inference")
    if split not in product.get("splits", {}):
        raise RuntimeError(f"Extent product has no frozen {split!r} predictions")
    return root, product


def _read_predicted_extent(
    extent_root: Path, split: str, sample: dict[str, Any]
) -> np.ndarray:
    sample_id = str(sample["metadata"]["sample_id"])
    path = extent_root / split / sample_id / "flood_extent.tif"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predicted flood extent for {sample_id}: {path}")
    with rasterio.open(path) as source:
        if source.count != 1 or (source.height, source.width) != (
            int(sample["metadata"]["height"]),
            int(sample["metadata"]["width"]),
        ):
            raise RuntimeError(f"Predicted extent grid shape mismatch: {path}")
        if source.crs is None or source.crs.to_string() != str(sample["metadata"]["crs"]):
            raise RuntimeError(f"Predicted extent CRS mismatch: {path}")
        expected_transform = np.asarray(sample["metadata"]["transform"][:6], dtype=float)
        if not np.allclose(np.asarray(tuple(source.transform)[:6]), expected_transform):
            raise RuntimeError(f"Predicted extent transform mismatch: {path}")
        values = source.read(1, masked=True).filled(0.0)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Predicted extent contains non-finite values: {path}")
    return values > 0.5


def _terrain_band_indices(dataset: FloodDepthDataset) -> tuple[int, int]:
    descriptions = list(dataset.contract.group("terrain")["band_descriptions"])
    try:
        return descriptions.index("elevation_m_DSM"), descriptions.index("slope_deg")
    except ValueError as exc:
        raise RuntimeError(
            "Geometry evaluator requires terrain bands elevation_m_DSM and slope_deg; "
            f"found {descriptions}"
        ) from exc


def _method_parameters(config: dict[str, Any], method: str) -> dict[str, Any]:
    parameters = config["geometry"].get("parameters", {})
    selected = parameters.get(method, {})
    if not isinstance(selected, dict):
        raise ValueError(f"geometry.parameters.{method} must be a mapping")
    return dict(selected)


def _selected_methods(
    config: dict[str, Any], requested: Iterable[str] | None
) -> list[str]:
    methods = list(requested) if requested is not None else list(config["geometry"]["methods"])
    if not methods:
        raise ValueError("At least one geometry method is required")
    if len(set(methods)) != len(methods):
        raise ValueError(f"Geometry method list contains duplicates: {methods}")
    unknown = [method for method in methods if method not in GEOMETRY_METHODS]
    if unknown:
        raise ValueError(
            f"Unknown geometry methods {unknown}; available={sorted(GEOMETRY_METHODS)}"
        )
    return methods


def _strip_unavailable_uncertainty(summary: dict[str, Any]) -> None:
    for key in [key for key in summary if key.startswith("uncertainty_")]:
        del summary[key]
    summary["uncertainty_available"] = False


def _save_prediction(
    output_dir: Path,
    sample: dict[str, Any],
    prediction: np.ndarray,
    water_surface: np.ndarray,
    support: np.ndarray,
    output_valid: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    sample_id = str(sample["metadata"]["sample_id"])
    sample_dir = output_dir / "predictions" / sample_id
    crs = str(sample["metadata"]["crs"])
    transform = sample["metadata"]["transform"]
    write_geotiff(
        sample_dir / "predicted_depth_m.tif",
        prediction,
        crs=crs,
        transform=transform,
        valid_mask=output_valid,
        descriptions=["predicted_depth_m_shared_predicted_extent"],
    )
    write_geotiff(
        sample_dir / "water_surface_elevation_m.tif",
        water_surface,
        crs=crs,
        transform=transform,
        valid_mask=output_valid,
        descriptions=["water_surface_elevation_m_shared_predicted_extent"],
    )
    write_geotiff(
        sample_dir / "input_predicted_flood_extent.tif",
        support.astype(np.float32),
        crs=crs,
        transform=transform,
        valid_mask=np.ones_like(support, dtype=bool),
        descriptions=["input_predicted_flood_extent"],
    )
    atomic_write_json(sample_dir / "metrics.json", metrics)


def run_geometry_evaluation(
    config_path: Path,
    split: str,
    output_dir: Path,
    extent_root: Path,
    *,
    methods: Iterable[str] | None = None,
    save_predictions: bool = False,
    max_samples: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run frozen geometry methods with one independently frozen extent product."""

    if split not in {"val", "test"}:
        raise ValueError("Geometry evaluation split must be val or test")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")

    config = load_config(config_path)
    geometry_config = config.get("geometry")
    if not isinstance(geometry_config, dict):
        raise ValueError("Configuration has no <geometry> section")
    if geometry_config.get("extent_source") != "ai4g_mobilenet_v2_unet_iou_frozen_prediction":
        raise ValueError(
            "Geometry config must declare the frozen AI4G IoU prediction product"
        )
    selected_methods = _selected_methods(config, methods)
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], split,
        input_spec=ModelInputSpec.from_config(config),
    )
    extent_root, extent_product = _load_extent_product(extent_root, dataset, config, split)
    dsm_index, slope_index = _terrain_band_indices(dataset)
    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), dataset.contract)
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    aggregators = {
        method: EvaluationAggregator(depth_bins, normalizer.train_depth_bins)
        for method in selected_methods
    }
    method_sample_rows: dict[str, list[dict[str, Any]]] = {
        method: [] for method in selected_methods
    }
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(len(dataset), max_samples or len(dataset))

    iterator = tqdm(range(sample_count), desc=f"geometry-{split}", disable=not progress)
    for sample_index in iterator:
        sample = dataset[sample_index]
        terrain = sample["terrain_raw"].numpy()
        dsm = terrain[dsm_index]
        slope_degrees = terrain[slope_index]
        extent = _read_predicted_extent(extent_root, split, sample)
        positive_mask = sample["masks"]["valid_depth_mask"][0].numpy() > 0.5
        dem_valid = sample["validity"]["dem_valid"][0].numpy() > 0.5
        target = sample["label"].numpy()
        sample_id = str(sample["metadata"]["sample_id"])
        event_id = str(sample["metadata"]["source_event_id"])
        resolution = tuple(float(value) for value in sample["metadata"]["resolution"])
        for method in selected_methods:
            # Only geometry and the frozen predicted support cross this boundary.
            # The target-depth tensor is deliberately absent from the call.
            result = run_geometry_method(
                method,
                dsm=dsm,
                slope_degrees=slope_degrees,
                extent=extent,
                dem_valid=dem_valid,
                pixel_size=resolution,
                parameters=_method_parameters(config, method),
            )
            if not np.array_equal(result.support, extent):
                raise RuntimeError(
                    f"{method} changed the frozen predicted support for {sample_id}"
                )
            row = aggregators[method].add(
                sample_id,
                event_id,
                result.depth[None],
                target,
                np.ones_like(target, dtype=np.float32),
                positive_mask[None],
                result.support.astype(np.float32)[None],
            )
            row.update(
                {
                    "comparison_class": "shared_predicted_extent_geometry",
                    "extent_source": "ai4g_mobilenet_v2_unet_iou_frozen_prediction",
                    "known_positive_extent_recall": float(
                        np.mean(extent[positive_mask])
                    ) if np.any(positive_mask) else float("nan"),
                    **{
                        f"geometry_{key}": value
                        for key, value in result.diagnostics.items()
                    },
                }
            )
            method_sample_rows[method].append(row)
            if save_predictions:
                _save_prediction(
                    output_dir / method,
                    sample,
                    result.depth,
                    result.water_surface_elevation,
                    result.support,
                    sample["validity"]["output_valid"][0].numpy() > 0.5,
                    row,
                )

    method_summaries: dict[str, dict[str, Any]] = {}
    for method in selected_methods:
        summary, sample_rows, event_rows, bin_rows = aggregators[method].summarize()
        _strip_unavailable_uncertainty(summary)
        summary.update(
            {
                "method": method,
                "comparison_class": "shared_predicted_extent_geometry",
                "split": split,
                "sample_count": sample_count,
                "primary_metric": "pixel_micro_mae",
                "extent_source": "ai4g_mobilenet_v2_unet_iou_frozen_prediction",
                "uses_label_derived_extent": False,
                "uses_target_depth_values_for_prediction": False,
                "deployable_without_depth_label": True,
                "extent_checkpoint_sha256": extent_product["checkpoint_sha256"],
                "terrain_source": "elevation_m_DSM_and_slope_deg",
                "terrain_void_fill": "nearest_valid_terrain_pixel",
                "parameters": _method_parameters(config, method),
                "protocol_note": PROTOCOL_NOTE,
                "terrain_void_pixels_imputed_inside_extent": int(
                    sum(
                        int(row["geometry_terrain_void_pixels_imputed_inside_extent"])
                        for row in method_sample_rows[method]
                    )
                ),
            }
        )
        method_dir = output_dir / method
        atomic_write_json(method_dir / "summary.json", summary)
        write_rows(method_dir / "metrics_by_sample.csv", sample_rows)
        write_rows(method_dir / "metrics_by_event.csv", event_rows)
        write_rows(method_dir / "metrics_by_train_depth_bin.csv", bin_rows)
        method_summaries[method] = summary

    report: dict[str, Any] = {
        "run_name": config.get("run_name", "geometry_shared_predicted_extent"),
        "split": split,
        "partial_run": sample_count != len(dataset),
        "evaluated_samples": sample_count,
        "total_split_samples": len(dataset),
        "primary_metric": "pixel_micro_mae",
        "protocol": {
            "comparison_class": "shared_predicted_extent_geometry",
            "extent_source": "ai4g_mobilenet_v2_unet_iou_frozen_prediction",
            "uses_label_derived_extent": False,
            "uses_target_depth_values_for_prediction": False,
            "shared_by_all_methods": True,
            "note": PROTOCOL_NOTE,
        },
        "extent_product": {
            "root": str(extent_root),
            "manifest_sha256": sha256_file(extent_root / "extent_product.json"),
            "checkpoint_sha256": extent_product["checkpoint_sha256"],
            "model": extent_product["model"],
            "split_summary": extent_product["splits"][split],
        },
        "dataset_fingerprint": {
            "contract_sha256": dataset.contract.hash,
            "normalization_sha256": sha256_file(Path(config["dataset"]["train_stats"])),
        },
        "reproducibility_fingerprint": {
            "configuration_sha256": sha256_file(config_path),
            "geometry_implementation_sha256": sha256_file(
                PROJECT_ROOT / "compare/geometry/terrain_extent.py"
            ),
            "evaluator_sha256": sha256_file(Path(__file__)),
        },
        "methods": method_summaries,
    }
    atomic_write_json(output_dir / "summary.json", report)
    atomic_write_json(output_dir / "resolved_config.json", jsonable_config(config))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--extent-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", choices=sorted(GEOMETRY_METHODS))
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    output = args.output or Path("runs/geometry_predicted_extent") / args.split
    report = run_geometry_evaluation(
        args.config,
        args.split,
        output,
        args.extent_root,
        methods=args.methods,
        save_predictions=args.save_predictions,
        max_samples=args.max_samples,
    )
    compact = {
        method: {
            "pixel_micro_mae": summary["pixel_micro_mae"],
            "pixel_micro_rmse": summary["pixel_micro_rmse"],
            "pixel_micro_pixels": summary["pixel_micro_pixels"],
        }
        for method, summary in report["methods"].items()
    }
    print(compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

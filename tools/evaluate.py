#!/usr/bin/env python3
"""Evaluate a registered model on val or test with partial-positive metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.preprocessing import RELIABILITY_NAMES, RobustNormalizer, resolve_depth_stratification_bins, reliability_spec_for_mode
from datasets.model_input_spec import ModelInputSpec
from datasets.reliability_spec import ReliabilitySpec


def _batch_reliability_names(batch: dict[str, Any]) -> tuple[str, ...]:
    values = batch.get("reliability_names")
    if isinstance(values, tuple) and values and all(isinstance(item, str) for item in values):
        return values
    if isinstance(values, list) and values and all(isinstance(item, str) for item in values):
        return tuple(values)
    if isinstance(values, (list, tuple)) and values and all(
        isinstance(item, (list, tuple)) and item and isinstance(item[0], str)
        for item in values
    ):
        return tuple(str(item[0]) for item in values)
    metadata = batch.get("metadata")
    if isinstance(metadata, dict):
        return _batch_reliability_names({"reliability_names": metadata.get("reliability_names")})
    return RELIABILITY_NAMES
from losses.composite_loss import CompositeFloodDepthLoss
from metrics.aggregator import EvaluationAggregator
from metrics.physical_metrics import (
    local_wse_laplacian,
    local_wse_laplacian_reference_error,
    prediction_continuity_high_relief,
    reference_gated_wse_gradient_mae,
    terrain_order_violation_metrics,
)
from utils.checkpoint import load_checkpoint
from utils.config import jsonable_config, load_config
from utils.logging import setup_logging, write_rows
from utils.misc import atomic_write_json, move_to_device
from utils.raster_io import write_geotiff
from utils.registry import build_model
from utils.visualization import save_prediction_panel
from datasets.contract import DatasetContract, sha256_file
from datasets.band_selection import resolve_band_spec


def dataset_fingerprint(config: dict[str, Any]) -> dict[str, str]:
    contract = DatasetContract.load(config["dataset"]["contract"])
    fingerprint = {
        "contract_sha256": contract.hash,
        "manifest_sha256": sha256_file(contract.manifest_path),
        "normalization_sha256": sha256_file(Path(config["dataset"]["train_stats"])),
    }
    input_spec = ModelInputSpec.from_config(config)
    if "input_mode" in config.get("dataset", {}) or str(config["model"]["name"]) == "pa_hydrokan_s1_v14":
        fingerprint["model_input_spec_sha256"] = input_spec.sha256
        fingerprint["active_groups_sha256"] = input_spec.active_groups_sha256
        fingerprint["reliability_spec_sha256"] = ReliabilitySpec.from_mode(input_spec.mode).sha256
    return fingerprint


def embed_source_fingerprints(config: dict[str, Any]) -> dict[str, Any]:
    """Place audited key-source hashes into every saved resolved configuration."""

    contract = DatasetContract.load(config["dataset"]["contract"])
    config["dataset"]["source_file_sha256"] = dict(contract.payload["key_file_sha256"])
    config["dataset"]["contract_sha256"] = contract.hash
    config["dataset"]["normalization_sha256"] = sha256_file(
        Path(config["dataset"]["train_stats"])
    )
    config["dataset"]["resolved_model_bands"] = resolve_band_spec(
        config, contract
    ).as_dict()
    input_spec = ModelInputSpec.from_config(config)
    config["dataset"]["resolved_model_input_spec"] = input_spec.as_dict()
    config["dataset"]["resolved_reliability_schema"] = list(
        reliability_spec_for_mode(input_spec.mode).names
    )
    return config


def metadata_item(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        selected = value[index].detach().cpu()
        return selected.item() if selected.ndim == 0 else selected.tolist()
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, torch.Tensor) for item in value):
            return tuple(metadata_item(item, index) for item in value)
        return value[index]
    return value


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_depth_bins: list[float],
    *,
    primary_depth_bins: list[float] | None = None,
    criterion: CompositeFloodDepthLoss | None = None,
    epoch: int = 0,
    max_batches: int | None = None,
    output_dir: Path | None = None,
    save_predictions: bool = False,
    progress: bool = True,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    input_spec: ModelInputSpec | None = None,
    validity_mask: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    aggregator = EvaluationAggregator(train_depth_bins, primary_depth_bins)
    loss_values: list[float] = []
    component_values: dict[str, list[float]] = {}
    event_depth_scales: list[float] = []
    support_branch_seen = False
    iterator = tqdm(loader, desc="evaluate", leave=False, disable=not progress)
    for batch_index, cpu_batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_to_device(cpu_batch, device)
        reliability_names = _batch_reliability_names(cpu_batch)
        reliability_index = {name: index for index, name in enumerate(reliability_names)}
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled and device.type == "cuda",
            dtype=amp_dtype,
        ):
            outputs = model(prepare_model_inputs(batch, input_spec))
            if criterion is not None:
                loss, components = criterion(outputs, batch, epoch)
                loss_values.append(float(loss.detach().cpu()))
                for name, value in components.items():
                    component_values.setdefault(name, []).append(float(value.detach().cpu()))
        batch_size = outputs["depth"].shape[0]
        for sample_index in range(batch_size):
            prediction = outputs["depth"][sample_index].detach().float().cpu().numpy()
            conditional_depth = (
                outputs.get("conditional_depth", outputs["positive_depth"])[sample_index]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            support_weighted_depth = (
                outputs.get("expected_depth", outputs["depth"])[sample_index]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            scale = outputs["uncertainty_scale"][sample_index].detach().float().cpu().numpy()
            support_tensor = outputs.get("support_probability")
            support_branch_seen = support_branch_seen or support_tensor is not None
            support = (
                support_tensor[sample_index].detach().float().cpu().numpy()
                if support_tensor is not None else None
            )
            target = batch["label"][sample_index].detach().float().cpu().numpy()
            positive_mask = (
                batch["masks"]["valid_depth_mask"][sample_index].detach().cpu().numpy() > 0.5
            )
            metric_valid_mask = positive_mask
            if validity_mask is not None:
                if validity_mask == "common_s1":
                    metric_valid_mask = positive_mask & (
                        batch["validity"]["s1_event_support"][sample_index].detach().cpu().numpy() > 0.5
                    ) & (
                        batch["validity"]["dem_valid"][sample_index].detach().cpu().numpy() > 0.5
                    )
                elif validity_mask not in batch["validity"]:
                    raise KeyError(f"Unknown evaluation validity mask: {validity_mask!r}")
                else:
                    metric_valid_mask = positive_mask & (
                        batch["validity"][validity_mask][sample_index].detach().cpu().numpy() > 0.5
                    )
            sample_id = str(metadata_item(cpu_batch["metadata"]["sample_id"], sample_index))
            event_id = str(
                metadata_item(cpu_batch["metadata"]["source_event_id"], sample_index)
            )
            day_index = reliability_index.get("absolute_normalized_sensor_day_difference")
            sensor_day = (
                batch["reliability"][sample_index, day_index:day_index + 1].detach().cpu().numpy()
                if day_index is not None else np.zeros_like(target)
            )
            observation_indices = [reliability_index[name] for name in ("s1_event_observation_count_z", "s2_pre_clear_observation_count_z", "s2_event_clear_observation_count_z") if name in reliability_index]
            observation = batch["reliability"][sample_index, observation_indices].mean(dim=0, keepdim=True).detach().cpu().numpy() if observation_indices else np.zeros_like(sensor_day)
            row = aggregator.add(
                sample_id,
                event_id,
                prediction,
                target,
                scale,
                metric_valid_mask,
                support,
                sensor_day,
                observation,
            )
            if "event_depth_scale" in outputs:
                event_depth_scale = float(
                    outputs["event_depth_scale"][sample_index].detach().float().cpu()
                )
                row["predicted_event_depth_scale"] = event_depth_scale
                event_depth_scales.append(event_depth_scale)
            z_hyd = outputs["physical_features"]["z_hyd"][sample_index].detach().float().cpu().numpy()
            local_relief = (
                outputs["physical_features"]["local_relief"][sample_index]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            output_valid_np = (
                batch["validity"]["output_valid"][sample_index].detach().cpu().numpy() > 0.5
            )
            positive_output_valid = metric_valid_mask & output_valid_np
            row["evaluation_valid_pixel_fraction"] = float(np.mean(metric_valid_mask))
            day_difference_np = sensor_day
            row["positive_region_wse_laplacian"] = local_wse_laplacian(
                prediction, z_hyd, positive_output_valid
            )
            row["reference_positive_region_wse_laplacian"] = local_wse_laplacian(
                target, z_hyd, positive_output_valid
            )
            row["positive_region_wse_laplacian_reference_error"] = (
                local_wse_laplacian_reference_error(
                    prediction, target, z_hyd, positive_output_valid
                )
            )
            physics_config = criterion.config if criterion is not None else {}
            row["reference_gated_wse_gradient_mae"] = (
                reference_gated_wse_gradient_mae(
                    prediction,
                    target,
                    z_hyd,
                    positive_output_valid,
                    day_difference_np,
                    float(physics_config.get("wse_time_sigma", 0.25)),
                    float(physics_config.get("wse_reference_sigma_m", 0.12)),
                    float(physics_config.get("wse_terrain_sigma_m", 0.75)),
                )
            )
            terrain_order = terrain_order_violation_metrics(
                prediction,
                z_hyd,
                positive_output_valid,
                day_difference_np,
                float(physics_config.get("wse_time_sigma", 0.25)),
                float(physics_config.get("terrain_order_min_step_m", 0.02)),
                float(physics_config.get("terrain_order_max_step_m", 0.75)),
            )
            row["terrain_order_violation_mae"] = terrain_order["mae"]
            row["terrain_order_violation_fraction"] = terrain_order["fraction"]
            row["high_relief_prediction_continuity"] = prediction_continuity_high_relief(
                prediction, local_relief, output_valid_np
            )
            if save_predictions and output_dir is not None:
                sample_dir = output_dir / sample_id
                crs = str(metadata_item(cpu_batch["metadata"]["crs"], sample_index))
                transform = metadata_item(cpu_batch["metadata"]["transform"], sample_index)
                output_valid = (
                    batch["validity"]["output_valid"][sample_index].detach().cpu().numpy()
                    > 0.5
                )
                write_geotiff(
                    sample_dir / "predicted_depth_m.tif",
                    prediction,
                    crs=crs,
                    transform=transform,
                    valid_mask=output_valid,
                    descriptions=["predicted_depth_m"],
                )
                write_geotiff(
                    sample_dir / "conditional_depth_m.tif",
                    conditional_depth,
                    crs=crs,
                    transform=transform,
                    valid_mask=output_valid,
                    descriptions=["conditional_depth_m"],
                )
                if support is not None:
                    write_geotiff(
                        sample_dir / "support_weighted_depth_m.tif",
                        support_weighted_depth,
                        crs=crs,
                        transform=transform,
                        valid_mask=output_valid,
                        descriptions=["support_weighted_depth_m"],
                    )
                    write_geotiff(
                        sample_dir / "support_probability.tif",
                        support,
                        crs=crs,
                        transform=transform,
                        valid_mask=output_valid,
                        descriptions=["support_probability"],
                    )
                write_geotiff(
                    sample_dir / "uncertainty_scale_m.tif",
                    scale,
                    crs=crs,
                    transform=transform,
                    valid_mask=output_valid,
                    descriptions=["uncertainty_scale_m"],
                )
                save_prediction_panel(
                    sample_dir / "prediction_panel.png",
                    s1_change=cpu_batch["s1_change"][sample_index, 0].numpy(),
                    s2_change=(cpu_batch["s2_change"][sample_index, 0].numpy() if "s2_change" in cpu_batch else None),
                    dsm=cpu_batch["terrain_raw"][sample_index, 0].numpy(),
                    target=target[0],
                    prediction=prediction[0],
                    uncertainty=scale[0],
                    valid_label=positive_mask[0],
                )
                atomic_write_json(sample_dir / "metrics.json", row)
    summary, sample_rows, event_rows, bin_rows = aggregator.summarize()
    unwrapped = model.module if hasattr(model, "module") else model
    summary["depth_output_semantics"] = str(
        getattr(
            getattr(unwrapped, "heads", None),
            "depth_output_semantics",
            "unknown",
        )
    )
    summary["evaluation_validity_mask"] = validity_mask or "label_valid_mask"
    summary["support_probability_reported"] = support_branch_seen
    if event_depth_scales:
        scale_array = np.asarray(event_depth_scales, dtype=np.float64)
        summary["event_depth_scale_mean"] = float(scale_array.mean())
        summary["event_depth_scale_std"] = float(scale_array.std())
        summary["event_depth_scale_min"] = float(scale_array.min())
        summary["event_depth_scale_max"] = float(scale_array.max())
    if loss_values:
        summary["objective_mean"] = float(np.mean(loss_values))
        for name, values in component_values.items():
            summary[f"objective_{name}_mean"] = float(np.mean(values))
    return summary, sample_rows, event_rows, bin_rows


def run_evaluation(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    device_name: str,
    output_dir: Path,
    save_predictions: bool,
    max_batches: int | None = None,
    weights: str = "raw",
    validity_mask: str | None = None,
    output_semantics_override: str | None = None,
) -> dict[str, Any]:
    config = embed_source_fingerprints(load_config(config_path))
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    input_spec = ModelInputSpec.from_config(config)
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], split,
        band_spec=band_spec,
        input_spec=input_spec,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        expected_fingerprint=dataset_fingerprint(config),
        map_location=device,
    )
    if weights == "ema":
        ema_state = checkpoint.get("ema_model")
        if ema_state is None:
            raise ValueError("--weights ema requested but checkpoint has no EMA state")
        model.load_state_dict(ema_state, strict=True)
    elif weights != "raw":
        raise ValueError("weights must be raw or ema")
    if output_semantics_override is not None:
        setter = getattr(getattr(model, "heads", None), "set_depth_output_semantics", None)
        if not callable(setter):
            raise ValueError("The selected model does not support output semantics override")
        setter(output_semantics_override)
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), dataset.contract)
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    prior_config = config["dataset"]["positive_prior"]
    prior = normalizer.positive_prior if prior_config["mode"] == "auto" else float(prior_config["value"])
    criterion = CompositeFloodDepthLoss(
        config["loss"], prior, depth_bins, normalizer.train_depth_bins,
        normalizer.train_depth_bin_counts,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, samples, events, bins = evaluate_loader(
        model,
        loader,
        device,
        depth_bins,
        primary_depth_bins=normalizer.train_depth_bins,
        criterion=criterion,
        epoch=checkpoint_epoch,
        max_batches=max_batches,
        output_dir=output_dir,
        save_predictions=save_predictions,
        amp_enabled=bool(config["training"].get("amp", False)),
        amp_dtype=(
            torch.bfloat16
            if str(config["training"].get("amp_dtype", "auto")) in {"auto", "bfloat16"}
            and device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        ),
        input_spec=input_spec,
        validity_mask=validity_mask,
    )
    summary["checkpoint_epoch"] = checkpoint_epoch
    summary["weights"] = weights
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "resolved_config.json", jsonable_config(config))
    write_rows(output_dir / "metrics_by_sample.csv", samples)
    write_rows(output_dir / "metrics_by_event.csv", events)
    write_rows(output_dir / "metrics_by_train_depth_bin.csv", bins)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--validity-mask", choices=("output_valid", "common_s1", "s1_event_support"))
    parser.add_argument("--output-semantics", choices=("conditional_positive_v2", "probability_weighted_v1"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    output = args.output or Path("runs/evaluate") / f"{args.split}_{args.checkpoint.stem}"
    summary = run_evaluation(
        args.config,
        args.checkpoint,
        args.split,
        args.device,
        output.resolve(),
        args.save_predictions,
        args.max_batches,
        args.weights,
        args.validity_mask,
        args.output_semantics,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

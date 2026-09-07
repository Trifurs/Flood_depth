"""Shared training and evaluation mechanics for named learned comparators.

Public entry points in this repository are model-named wrappers.  This private
module only keeps their data loading, masked objective, checkpoints, and metric
schema consistent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.preprocessing import resolve_depth_stratification_bins
from datasets.supervision_masks import (
    CANONICAL_POSITIVE_MASK,
    canonical_positive_mask_from_batch,
    supervision_mask_counts,
)
from metrics.aggregator import EvaluationAggregator
from compare.common._depth_regression import prepare_comparison_tensor
from compare.common.comparison_factory import build_comparison_model
from tools.evaluate_pa_hydrokan import dataset_fingerprint, embed_source_fingerprints, metadata_item
from tools.train_pa_hydrokan import create_dataloaders
from utils.amp import resolve_amp
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import jsonable_config, load_config
from utils.logging import append_csv, setup_logging, write_rows
from utils.misc import atomic_write_json, move_to_device
from utils.optim import build_scheduler
from utils.seed import seed_everything


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def _validated_config(config_path: Path, expected_model: str) -> dict[str, Any]:
    config = embed_source_fingerprints(load_config(config_path))
    model = config.get("model", {})
    if str(model.get("name")) != expected_model:
        raise ValueError(
            f"This entry point is reserved for {expected_model!r}; config requests {model.get('name')!r}"
        )
    if str(model.get("flood_support")) != "valid_depth_mask":
        raise ValueError("Learned comparators must use direct valid_depth_mask flood support")
    if str(model.get("depth_output_semantics", "conditional_positive")) != "conditional_positive":
        raise ValueError("Learned comparators require conditional_positive depth semantics")
    return config


def _apply_cli(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.device is not None:
        config["device"] = args.device
    if getattr(args, "epochs", None) is not None:
        config["training"]["epochs"] = int(args.epochs)
    if getattr(args, "batch_size", None) is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if getattr(args, "num_workers", None) is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        config["training"]["num_workers"] = int(args.num_workers)
    if getattr(args, "seed", None) is not None:
        config["seed"] = int(args.seed)
    if getattr(args, "no_amp", False):
        config["training"]["amp"] = False
    config["training"]["persistent_workers"] = bool(
        config["training"].get("persistent_workers", False)
    ) and int(config["training"]["num_workers"]) > 0
    return config


def _masked_objective(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any], loss_config: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    positive = canonical_positive_mask_from_batch(batch)
    if not bool(torch.any(positive)):
        raise RuntimeError("A learned comparison batch contains no valid positive-depth pixels")
    prediction = outputs["conditional_depth"]
    target = batch["label"]
    beta = float(loss_config.get("depth_huber_beta_m", 0.5))
    linear = F.smooth_l1_loss(prediction[positive], target[positive], beta=beta)
    log = F.smooth_l1_loss(
        torch.log1p(prediction[positive]), torch.log1p(target[positive]), beta=beta
    )
    log_weight = float(loss_config.get("lambda_log", 0.05))
    total = linear + log_weight * log
    return total, {"depth": linear, "log_depth": log, "total": total}


def _batch_metadata(batch: Mapping[str, Any], index: int) -> tuple[str, str]:
    metadata = batch.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return str(index), "unknown"
    return (
        str(metadata_item(metadata.get("sample_id", (str(index),)), index)),
        str(metadata_item(metadata.get("source_event_id", ("unknown",)), index)),
    )


@torch.no_grad()
def evaluate_comparison_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Mapping[str, Any],
    depth_bins: list[float],
    primary_bins: list[float],
    *,
    max_batches: int | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    aggregator = EvaluationAggregator(depth_bins, primary_bins)
    objectives: list[float] = []
    mask_totals = {
        "valid_depth_mask_pixels": 0,
        "output_valid_pixels": 0,
        "positive_supervision_pixels": 0,
        "positive_excluded_by_output_valid_pixels": 0,
    }
    schema = str(config["model"]["input_schema"])
    for batch_index, cpu_batch in enumerate(tqdm(loader, desc="evaluate", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_to_device(cpu_batch, device, non_blocking=device.type == "cuda")
        for name, value in supervision_mask_counts(batch).items():
            if name in mask_totals:
                mask_totals[name] += int(value.detach().cpu())
        inputs, flood_range = prepare_comparison_tensor(batch, schema)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(inputs, flood_range)
            objective, _ = _masked_objective(outputs, batch, config["loss"])
        objectives.append(float(objective.detach().cpu()))
        valid = canonical_positive_mask_from_batch(batch).detach().cpu().numpy()
        for sample_index in range(outputs["depth"].shape[0]):
            sample_id, event_id = _batch_metadata(cpu_batch, sample_index)
            aggregator.add(
                sample_id,
                event_id,
                outputs["depth"][sample_index].detach().float().cpu().numpy(),
                batch["label"][sample_index].detach().float().cpu().numpy(),
                outputs["uncertainty_scale"][sample_index].detach().float().cpu().numpy(),
                valid[sample_index],
            )
    summary, samples, events, bins = aggregator.summarize()
    summary.update(mask_totals)
    summary.update({
        "method": str(config["model"]["name"]),
        "flood_support": "valid_depth_mask",
        "input_schema": schema,
        "evaluation_mask": CANONICAL_POSITIVE_MASK,
        "positive_excluded_by_output_valid_fraction": float(
            mask_totals["positive_excluded_by_output_valid_pixels"]
        ) / max(float(mask_totals["valid_depth_mask_pixels"]), 1.0),
        "objective_mean": float(np.mean(objectives)) if objectives else float("nan"),
    })
    return summary, samples, events, bins


def _write_evaluation(
    output: Path,
    summary: Mapping[str, Any],
    samples: list[dict[str, Any]],
    events: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", dict(summary))
    atomic_write_json(output / "resolved_config.json", jsonable_config(config))
    write_rows(output / "metrics_by_sample.csv", samples)
    write_rows(output / "metrics_by_event.csv", events)
    write_rows(output / "metrics_by_train_depth_bin.csv", bins)


def _parameter_payload(model: torch.nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": str(config["model"]["name"]),
        "total_parameters": int(sum(item.numel() for item in model.parameters())),
        "trainable_parameters": int(sum(item.numel() for item in model.parameters() if item.requires_grad)),
        "input_schema": str(config["model"]["input_schema"]),
        "flood_support": "valid_depth_mask",
        "architecture": dict(config["model"]),
    }


def run_training(args: argparse.Namespace, expected_model: str) -> Path:
    config = _apply_cli(_validated_config(args.config, expected_model), args)
    device = _device(str(config["device"]))
    seed_everything(int(config["seed"]), bool(config["deterministic"]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.output.resolve()
        if args.output is not None
        else Path(config["runs_root"]) / "train" / f"{config['run_name']}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    train_loader, val_loader, train_dataset, _ = create_dataloaders(config)
    model = build_comparison_model(config).to(device)
    parameters = _parameter_payload(model, config)
    atomic_write_json(run_dir / "parameters.json", parameters)
    atomic_write_json(run_dir / "resolved_config.json", jsonable_config(config))
    optimizer_name = str(config["optimizer"]["name"]).lower()
    if optimizer_name != "adamw":
        raise ValueError("Learned comparison models currently support optimizer.name=adamw")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    max_train = args.max_train_batches
    steps_per_epoch = math.ceil(min(len(train_loader), max_train or len(train_loader)))
    scheduler = build_scheduler(
        optimizer,
        config,
        total_steps=max(1, steps_per_epoch * epochs),
        warmup_steps=int(config["scheduler"]["warmup_epochs"]) * steps_per_epoch,
    )
    amp_enabled, amp_dtype, scaler_enabled = resolve_amp(
        device, bool(config["training"].get("amp", False)), str(config["training"].get("amp_dtype", "auto"))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    fingerprint = dataset_fingerprint(config)
    depth_bins = resolve_depth_stratification_bins(config["loss"], train_dataset.normalizer)
    primary_bins = train_dataset.normalizer.train_depth_bins
    monitor = str(config["training"].get("best_metric", "pixel_micro_mae"))
    if monitor not in {"pixel_micro_mae", "event_macro_mae", "event_hierarchical_composite_mae"}:
        raise ValueError(f"Unsupported learned-comparator best metric {monitor!r}")
    best = float("inf")
    patience = 0
    best_epoch = -1
    max_val = args.max_val_batches
    for epoch in range(epochs):
        model.train()
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        total_loss, batches = 0.0, 0
        for batch_index, cpu_batch in enumerate(tqdm(train_loader, desc=f"train {epoch + 1}", leave=False)):
            if max_train is not None and batch_index >= max_train:
                break
            batch = move_to_device(cpu_batch, device, non_blocking=device.type == "cuda")
            inputs, flood_range = prepare_comparison_tensor(batch, str(config["model"]["input_schema"]))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(inputs, flood_range)
                loss, components = _masked_objective(outputs, batch, config["loss"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        if batches == 0:
            raise RuntimeError("No comparison training batches were executed")
        summary, samples, events, bins = evaluate_comparison_loader(
            model, val_loader, device, config, depth_bins, primary_bins,
            max_batches=max_val, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        )
        summary["epoch"] = epoch
        summary["total_parameters"] = parameters["total_parameters"]
        summary["trainable_parameters"] = parameters["trainable_parameters"]
        _write_evaluation(run_dir / "validation", summary, samples, events, bins, config)
        row = {
            "epoch": epoch,
            "train_total_loss": total_loss / batches,
            "validation_objective": float(summary["objective_mean"]),
            "pixel_micro_mae": float(summary["pixel_micro_mae"]),
            "event_macro_mae": float(summary["event_macro_mae"]),
            "event_hierarchical_composite_mae": float(summary["event_hierarchical_composite_mae"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        append_csv(run_dir / "metrics.csv", row)
        candidate = float(summary[monitor])
        save_checkpoint(
            run_dir / "last_raw.pth", model, optimizer, scheduler, scaler, epoch, best,
            jsonable_config(config), fingerprint,
            extra={"best_metric_name": monitor, "validation_summary": dict(summary)},
            training_context={"epochs": epochs, "sampler": type(train_loader.sampler).__name__},
            global_step=(epoch + 1) * steps_per_epoch,
        )
        if candidate < best:
            best, best_epoch, patience = candidate, epoch, 0
            save_checkpoint(
                run_dir / "best_raw.pth", model, optimizer, scheduler, scaler, epoch, best,
                jsonable_config(config), fingerprint,
                extra={"best_metric_name": monitor, "validation_summary": dict(summary)},
                training_context={"epochs": epochs, "sampler": type(train_loader.sampler).__name__},
                global_step=(epoch + 1) * steps_per_epoch,
            )
        else:
            patience += 1
        if epoch + 1 >= int(config["training"].get("minimum_epochs", 1)) and patience >= int(config["training"].get("early_stop_patience", epochs)):
            break
    atomic_write_json(run_dir / "training_summary.json", {
        **parameters,
        "best_metric_name": monitor,
        "best_metric": best,
        "best_epoch": best_epoch,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return run_dir


def run_evaluation(args: argparse.Namespace, expected_model: str) -> dict[str, Any]:
    config = _validated_config(args.config, expected_model)
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        config["training"]["num_workers"] = int(args.num_workers)
        config["training"]["persistent_workers"] = False
    device = _device(args.device)
    _, loader, train_dataset, _ = create_dataloaders(config)
    model = build_comparison_model(config).to(device)
    checkpoint = load_checkpoint(
        args.checkpoint, model, expected_fingerprint=dataset_fingerprint(config), map_location=device
    )
    depth_bins = resolve_depth_stratification_bins(config["loss"], train_dataset.normalizer)
    amp_enabled, amp_dtype, _ = resolve_amp(
        device, bool(config["training"].get("amp", False)), str(config["training"].get("amp_dtype", "auto"))
    )
    # create_dataloaders supplies val.  For test, construct the matching loader
    # by reusing its audited dataset settings without training augmentation.
    if args.split == "test":
        from datasets.band_selection import resolve_band_spec
        from datasets.contract import DatasetContract
        from datasets.flooddepth_dataset import FloodDepthDataset
        from datasets.model_input_spec import ModelInputSpec

        contract = DatasetContract.load(config["dataset"]["contract"])
        test_dataset = FloodDepthDataset(
            config["dataset"]["contract"], config["dataset"]["train_stats"], "test",
            band_spec=resolve_band_spec(config, contract),
            input_spec=ModelInputSpec.from_config(config),
            minimum_event_band_fraction=float(config["dataset"].get("minimum_event_band_fraction", 1.0)),
            s1_qa_names=config["dataset"].get("model_s1_qa_names"),
        )
        loader = DataLoader(
            test_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
            persistent_workers=bool(config["training"].get("persistent_workers", False)),
        )
    summary, samples, events, bins = evaluate_comparison_loader(
        model, loader, device, config, depth_bins, train_dataset.normalizer.train_depth_bins,
        max_batches=args.max_batches, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
    )
    summary["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    summary.update(_parameter_payload(model, config))
    output = args.output.resolve() if args.output else Path(config["runs_root"]) / "evaluate" / f"{expected_model}_{args.split}_{args.checkpoint.stem}"
    _write_evaluation(output, summary, samples, events, bins, config)
    return summary


def train_main(expected_model: str, default_config: str) -> int:
    parser = argparse.ArgumentParser(description=f"Train {expected_model}.")
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(f"training output: {run_training(args, expected_model)}")
    return 0


def evaluation_main(expected_model: str, default_config: str) -> int:
    parser = argparse.ArgumentParser(description=f"Evaluate {expected_model}.")
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(run_evaluation(args, expected_model))
    return 0

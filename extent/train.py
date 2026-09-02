#!/usr/bin/env python3
"""Train the independent AI4G-style binary flood-extent model with Soft-IoU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.contract import DatasetContract, sha256_file
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.transforms import SynchronousAugment
from extent.ai4g_mobilenet_unet import AI4GFloodExtentNet
from extent.losses import masked_soft_iou_loss
from extent.protocol import build_ai4g_change_features, postprocess_extent
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import jsonable_config, load_config
from utils.logging import append_csv, setup_logging
from utils.misc import atomic_write_json, move_to_device
from utils.seed import seed_everything, seed_worker


FEATURE_KEYS = (
    "vv_water_threshold_db",
    "vh_water_threshold_db",
    "minimum_drop_db",
    "vv_valid_floor_db",
    "vh_valid_floor_db",
)


def dataset_fingerprint(config: dict[str, Any]) -> dict[str, str]:
    contract = DatasetContract.load(config["dataset"]["contract"])
    return {
        "contract_sha256": contract.hash,
        "manifest_sha256": sha256_file(contract.manifest_path),
        "normalization_sha256": sha256_file(Path(config["dataset"]["train_stats"])),
    }


def build_model(config: dict[str, Any]) -> AI4GFloodExtentNet:
    channels = tuple(int(value) for value in config["extent_model"]["decoder_channels"])
    return AI4GFloodExtentNet(channels)


def feature_parameters(config: dict[str, Any]) -> dict[str, float]:
    return {key: float(config["extent_model"][key]) for key in FEATURE_KEYS}


def _binary_target(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    target = batch["masks"]["valid_depth_mask"] > 0.5
    valid = batch["validity"]["output_valid"] > 0.5
    return target, valid


def _forward_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    change, _ = build_ai4g_change_features(batch, **feature_parameters(config))
    logits = model(change)
    target, valid = _binary_target(batch)
    loss, soft_iou = masked_soft_iou_loss(logits, target, valid)
    return loss, {"iou_loss": loss, "soft_iou": soft_iou}, logits, target


def create_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, FloodDepthDataset]:
    augmentation = config["dataset"]["augmentation"]
    transform = SynchronousAugment(
        float(augmentation["horizontal_flip_probability"]),
        float(augmentation["vertical_flip_probability"]),
        float(augmentation["rotate90_probability"]),
        0.0,
    )
    train_dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        "train",
        transform=transform,
    )
    val_dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], "val"
    )
    workers = int(config["training"]["num_workers"])
    generator = torch.Generator().manual_seed(int(config["seed"]))
    common = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": workers,
        "persistent_workers": bool(config["training"]["persistent_workers"]) and workers > 0,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, train_dataset


def _scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    batches = 0
    optimizer.zero_grad(set_to_none=True)
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    iterator = tqdm(loader, desc="extent-train", leave=False)
    for batch_index, cpu_batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_to_device(cpu_batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss, components, _, _ = _forward_loss(model, batch, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite extent loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= scale_before:
            scheduler.step()
        batches += 1
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().cpu())
        iterator.set_postfix(iou=f"{float(components['soft_iou'].detach()):.4f}")
    if batches == 0:
        raise RuntimeError("No extent training batches were executed")
    return {name: value / batches for name, value in sums.items()}


@torch.no_grad()
def evaluate_extent(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    max_batches: int | None = None,
    progress: bool = True,
) -> dict[str, float | int]:
    model.eval()
    soft_intersection = soft_union = 0.0
    raw_tp = raw_fp = raw_fn = 0
    buffered_tp = buffered_fp = buffered_fn = 0
    predicted_pixels = raw_pixels = eligible_pixels = positive_pixels = 0
    iterator = tqdm(loader, desc="extent-val", leave=False, disable=not progress)
    for batch_index, cpu_batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_to_device(cpu_batch, device)
        change, _ = build_ai4g_change_features(batch, **feature_parameters(config))
        logits = model(change)
        probabilities = torch.sigmoid(logits)
        target, valid = _binary_target(batch)
        truth = target & valid
        soft_intersection += float((probabilities * truth * valid).sum().cpu())
        soft_union += float(
            ((probabilities + truth - probabilities * truth) * valid).sum().cpu()
        )
        raw, buffered = postprocess_extent(
            probabilities,
            batch["validity"]["output_valid"],
            probability_threshold=float(config["extent_model"]["probability_threshold"]),
            buffer_pixels=int(config["extent_model"]["buffer_pixels"]),
        )
        output_valid = valid
        raw_tp += int((raw & truth).sum())
        raw_fp += int((raw & ~truth & output_valid).sum())
        raw_fn += int((~raw & truth).sum())
        buffered_tp += int((buffered & truth).sum())
        buffered_fp += int((buffered & ~truth & output_valid).sum())
        buffered_fn += int((~buffered & truth).sum())
        raw_pixels += int(raw.sum())
        predicted_pixels += int(buffered.sum())
        eligible_pixels += int(output_valid.sum())
        positive_pixels += int(truth.sum())
    if positive_pixels == 0 or eligible_pixels == 0:
        raise RuntimeError("Extent validation lacks positive or eligible pixels")

    def metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
        iou = tp / max(1, tp + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        }

    raw_metrics = metrics(raw_tp, raw_fp, raw_fn)
    buffered_metrics = metrics(buffered_tp, buffered_fp, buffered_fn)
    return {
        "soft_iou": soft_intersection / max(1e-12, soft_union),
        **{f"raw_{key}": value for key, value in raw_metrics.items()},
        **{f"buffered_{key}": value for key, value in buffered_metrics.items()},
        "raw_predicted_area_fraction": raw_pixels / eligible_pixels,
        "buffered_predicted_area_fraction": predicted_pixels / eligible_pixels,
        "positive_pixels": positive_pixels,
        "eligible_pixels": eligible_pixels,
    }


def run_training(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seed_everything(int(config["seed"]), bool(config["deterministic"]))
    output = (args.output or Path("runs/extent/train") / config["run_name"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup_logging(output / "train.log")
    train_loader, val_loader, _ = create_loaders(config)
    model = build_model(config).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    effective_batches = min(len(train_loader), args.max_train_batches or len(train_loader))
    total_steps = max(1, epochs * effective_batches)
    warmup_steps = int(config["scheduler"]["warmup_epochs"]) * effective_batches
    scheduler = _scheduler(
        optimizer,
        total_steps,
        warmup_steps,
        float(config["scheduler"]["minimum_learning_rate"])
        / float(config["optimizer"]["learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["training"]["amp"]) and device.type == "cuda")
    fingerprint = dataset_fingerprint(config)
    start_epoch = 0
    best_metric = float("-inf")
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            expected_fingerprint=fingerprint,
            restore_rng=True,
            map_location=device,
            adopt_checkpoint_output_semantics=False,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
    atomic_write_json(output / "resolved_config.json", jsonable_config(config))
    atomic_write_json(output / "dataset_fingerprint.json", fingerprint)
    atomic_write_json(
        output / "model_summary.json",
        {
            "name": config["extent_model"]["name"],
            "paper": config["extent_model"]["paper"],
            "parameters": total_parameters,
            "supervision": "binary: valid_depth_mask is the flood label",
            "objective": "pure masked Soft-IoU loss",
            "binary_extent_metrics_available": True,
        },
    )
    atomic_write_json(
        output / "environment.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
        },
    )
    patience = 0
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            args.max_train_batches,
        )
        val_metrics = evaluate_extent(
            model,
            val_loader,
            device,
            config,
            args.max_val_batches,
        )
        metric = float(val_metrics["raw_iou"])
        improved = metric > best_metric
        if improved:
            best_metric = metric
            patience = 0
        else:
            patience += 1
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_val_iou": best_metric,
        }
        append_csv(output / "metrics_by_epoch.csv", row)
        checkpoint_args = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            resolved_config=jsonable_config(config),
            dataset_fingerprint=fingerprint,
            extra={
                "best_metric_name": "val_raw_iou",
                "parameters": total_parameters,
                "validation": val_metrics,
            },
        )
        save_checkpoint(output / "last.pth", **checkpoint_args)
        if improved:
            save_checkpoint(output / "best.pth", **checkpoint_args)
        print(
            f"extent epoch={epoch + 1}/{epochs} train_soft_iou={train_metrics['soft_iou']:.6f} "
            f"val_iou={metric:.6f} recall={val_metrics['raw_recall']:.4f} "
            f"area={val_metrics['buffered_predicted_area_fraction']:.4f}"
        )
        if patience >= int(config["training"]["early_stop_patience"]):
            break
    atomic_write_json(
        output / "training_runtime.json",
        {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "best_val_iou": best_metric,
        },
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    print(f"extent training output: {run_training(parse_args())}")

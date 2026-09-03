#!/usr/bin/env python3
"""Real-data train/val/checkpoint/inference smoke validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from datasets.flooddepth_dataset import prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from datasets.preprocessing import resolve_depth_stratification_bins
from losses.composite_loss import CompositeFloodDepthLoss
from tools.evaluate import dataset_fingerprint, embed_source_fingerprints, evaluate_loader
from tools.train import create_dataloaders
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import jsonable_config, load_config
from utils.logging import setup_logging
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model
from utils.seed import seed_everything
from utils.amp import resolve_amp


def run_smoke(
    config_path: Path,
    device_name: str = "auto",
    train_batches: int = 2,
    val_batches: int = 1,
    output_root: Path | None = None,
    batch_size: int = 1,
) -> dict:
    config = embed_source_fingerprints(load_config(config_path))
    config["training"]["batch_size"] = int(batch_size)
    config["training"]["num_workers"] = 0
    config["training"]["persistent_workers"] = False
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
    )
    seed_everything(int(config["seed"]), bool(config["deterministic"]))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (output_root or Path("runs")).resolve()
    train_dir = root / "train" / f"smoke_{stamp}"
    infer_dir = root / "infer" / f"smoke_{stamp}"
    train_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(train_dir / "smoke.log")
    train_loader, val_loader, train_dataset, _ = create_dataloaders(config)
    input_spec = ModelInputSpec.from_config(config)
    depth_bins = resolve_depth_stratification_bins(
        config["loss"], train_dataset.normalizer
    )
    model = build_model(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, train_batches)
    )
    amp_enabled, amp_dtype, scaler_enabled = resolve_amp(
        device, bool(config["training"]["amp"]),
        str(config["training"].get("amp_dtype", "float16")),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    criterion = CompositeFloodDepthLoss(
        config["loss"],
        train_dataset.normalizer.positive_prior,
        depth_bins,
        train_dataset.normalizer.train_depth_bins,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    successful_optimizer_steps = 0
    model.train()
    for batch_index, cpu_batch in enumerate(train_loader):
        if batch_index >= train_batches:
            break
        batch = move_to_device(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(prepare_model_inputs(batch, input_spec))
            loss, _ = criterion(outputs, batch, epoch=0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Smoke loss is not finite at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            scheduler.step()
            successful_optimizer_steps += 1
        losses.append(float(loss.detach().cpu()))
    if len(losses) != train_batches:
        raise RuntimeError(f"Requested {train_batches} train batches, ran {len(losses)}")
    if successful_optimizer_steps < 1:
        raise RuntimeError("AMP skipped every optimizer step during smoke validation")

    fingerprint = dataset_fingerprint(config)
    atomic_write_json(train_dir / "resolved_config.json", jsonable_config(config))
    atomic_write_json(train_dir / "dataset_fingerprint.json", fingerprint)
    checkpoint_path = train_dir / "last.pth"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=0,
        best_metric=float("inf"),
        resolved_config=jsonable_config(config),
        dataset_fingerprint=fingerprint,
        extra={"smoke": True, "train_batches": train_batches},
    )
    reloaded = build_model(config).to(device)
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-4)
    reloaded_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        reloaded_optimizer, T_max=max(1, train_batches)
    )
    reloaded_scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    checkpoint = load_checkpoint(
        checkpoint_path,
        reloaded,
        reloaded_optimizer,
        reloaded_scheduler,
        reloaded_scaler,
        expected_fingerprint=fingerprint,
        map_location=device,
    )
    summary, samples, _, _ = evaluate_loader(
        reloaded,
        val_loader,
        device,
        depth_bins,
        primary_depth_bins=train_dataset.normalizer.train_depth_bins,
        criterion=criterion,
        max_batches=val_batches,
        output_dir=infer_dir,
        save_predictions=True,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        input_spec=input_spec,
    )
    if not samples:
        raise RuntimeError("Smoke validation produced no sample metrics")
    sample_id = samples[0]["sample_id"]
    prediction_path = infer_dir / sample_id / "predicted_depth_m.tif"
    if not prediction_path.is_file():
        raise RuntimeError(f"Smoke GeoTIFF was not exported: {prediction_path}")
    report = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameter_count": parameter_count,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
        "train_batches": train_batches,
        "val_batches": val_batches,
        "train_losses": losses,
        "successful_optimizer_steps": successful_optimizer_steps,
        "checkpoint_epoch_loaded": int(checkpoint["epoch"]),
        "validation_pixel_micro_mae": summary["pixel_micro_mae"],
        "validation_pixel_micro_rmse": summary["pixel_micro_rmse"],
        "validation_event_macro_mae": summary["event_macro_mae"],
        "validation_event_depth_bin_macro_mae": summary[
            "event_depth_bin_macro_mae"
        ],
        "validation_event_depth_hierarchical_macro_mae": summary[
            "event_depth_hierarchical_macro_mae"
        ],
        "validation_event_hierarchical_composite_mae": summary[
            "event_hierarchical_composite_mae"
        ],
        "checkpoint": str(checkpoint_path),
        "prediction_geotiff": str(prediction_path),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }
    atomic_write_json(train_dir / "smoke_report.json", report)
    print(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_smoke(
        args.config,
        args.device,
        args.train_batches,
        args.val_batches,
        args.output_root,
        args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a registered flood-depth model with strict reproducibility."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.contract import DatasetContract, sha256_file
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from datasets.samplers import (
    DistributedEventBalancedSampler,
    DistributedEventEpochSampler,
    EventEpochSampler,
    make_event_balanced_sampler,
)
from datasets.transforms import SynchronousAugment
from losses.composite_loss import CompositeFloodDepthLoss
from tools.evaluate import dataset_fingerprint, embed_source_fingerprints, evaluate_loader
from utils.checkpoint import (
    checkpoint_depth_output_semantics,
    load_checkpoint,
    save_checkpoint,
)
from utils.config import jsonable_config, load_config
from utils.distributed import cleanup_distributed, initialize_distributed, is_main_process
from utils.logging import append_csv, setup_logging
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model
from utils.seed import seed_everything, seed_worker


LOGGER = logging.getLogger("train")


def resolve_cli(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.no_amp:
        config["training"]["amp"] = False
    if args.seed is not None:
        config["seed"] = args.seed
    config["training"]["persistent_workers"] = (
        bool(config["training"]["persistent_workers"])
        and int(config["training"]["num_workers"]) > 0
    )
    return config


def create_dataloaders(
    config: dict[str, Any], rank: int = 0, world_size: int = 1
) -> tuple[DataLoader, DataLoader, FloodDepthDataset, FloodDepthDataset]:
    augmentation = config["dataset"]["augmentation"]
    transform = SynchronousAugment(
        float(augmentation["horizontal_flip_probability"]),
        float(augmentation["vertical_flip_probability"]),
        float(augmentation["rotate90_probability"]),
        float(augmentation["modality_dropout_probability"]),
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
    replacement = bool(config["dataset"]["sampling"].get("replacement", False))
    if world_size > 1:
        sampler: Any = (
            DistributedEventBalancedSampler(
                train_dataset.event_ids, world_size, rank, int(config["seed"])
            )
            if replacement
            else DistributedEventEpochSampler(
                train_dataset.event_ids, world_size, rank, int(config["seed"])
            )
        )
    else:
        sampler = (
            make_event_balanced_sampler(train_dataset.event_ids, int(config["seed"]))
            if replacement
            else EventEpochSampler(train_dataset.event_ids, int(config["seed"]))
        )
    workers = int(config["training"]["num_workers"])
    generator = torch.Generator().manual_seed(int(config["seed"]) + rank)
    common = {
        "num_workers": workers,
        "persistent_workers": bool(config["training"]["persistent_workers"]) if workers > 0 else False,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    criterion: CompositeFloodDepthLoss,
    device: torch.device,
    epoch: int,
    accumulation_steps: int,
    grad_clip: float,
    amp_enabled: bool,
    max_batches: int | None,
    run_dir: Path,
    rank: int,
) -> dict[str, float]:
    model.train()
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    batches = 0
    iterator = tqdm(loader, desc=f"train {epoch + 1}", leave=False, disable=rank != 0)
    for batch_index, cpu_batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_to_device(cpu_batch, device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(prepare_model_inputs(batch))
            loss, components = criterion(outputs, batch, epoch)
            scaled_loss = loss / accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, batch={batch_index}")
        scaler.scale(scaled_loss).backward()
        final_batch = batch_index + 1 == len(loader) or (
            max_batches is not None and batch_index + 1 >= max_batches
        )
        if (batch_index + 1) % accumulation_steps == 0 or final_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= scale_before:
                scheduler.step()
        batches += 1
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().cpu())
        if rank == 0:
            iterator.set_postfix(loss=f"{float(loss.detach()):.4f}")
            append_csv(
                run_dir / "train_steps.csv",
                {
                    "epoch": epoch,
                    "batch": batch_index,
                    "loss": float(loss.detach().cpu()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
            )
    if batches == 0:
        raise RuntimeError("No train batches were executed")
    return {name: value / batches for name, value in sums.items()}


def environment_payload(device: torch.device) -> dict[str, Any]:
    return {
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
    }


def run_training(args: argparse.Namespace) -> Path:
    config = resolve_cli(embed_source_fingerprints(load_config(args.config)), args)
    device, rank, world_size, local_rank = initialize_distributed(str(config["device"]))
    seed_everything(int(config["seed"]) + rank, bool(config["deterministic"]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.resume.resolve().parent
        if args.resume is not None
        else Path(config["runs_root"]) / "train" / f"{config['run_name']}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log" if rank == 0 else None)
    LOGGER.info("Resolved config: %s", jsonable_config(config))
    train_loader, val_loader, train_dataset, _ = create_dataloaders(config, rank, world_size)
    normalizer = train_dataset.normalizer
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    prior_config = config["dataset"]["positive_prior"]
    prior = normalizer.positive_prior if prior_config["mode"] == "auto" else float(prior_config["value"])
    minimum, maximum = float(prior_config["minimum"]), float(prior_config["maximum"])
    prior = float(np.clip(prior, minimum, maximum))
    LOGGER.info("nnPU positive prior=%f method=%s", prior, prior_config["mode"])
    LOGGER.info("train-only depth stratification edges (m)=%s", depth_bins)

    model = build_model(config).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters >= 25_000_000:
        raise RuntimeError(f"Model exceeds the 25M parameter target: {total_parameters}")
    LOGGER.info("parameters total=%d trainable=%d", total_parameters, trainable_parameters)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    effective_train_batches = min(
        len(train_loader), args.max_train_batches or len(train_loader)
    )
    steps_per_epoch = math.ceil(effective_train_batches / accumulation)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(config["scheduler"]["warmup_epochs"]) * steps_per_epoch
    minimum_ratio = float(config["scheduler"]["minimum_learning_rate"]) / float(
        config["optimizer"]["learning_rate"]
    )
    scheduler = cosine_warmup_scheduler(optimizer, total_steps, warmup_steps, minimum_ratio)
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = CompositeFloodDepthLoss(
        config["loss"], prior, depth_bins, normalizer.train_depth_bins
    )
    fingerprint = dataset_fingerprint(config)
    monitor = str(config["training"]["best_metric"])
    start_epoch, best_metric = 0, float("inf")
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            expected_fingerprint=fingerprint,
            allow_fingerprint_mismatch=args.allow_fingerprint_mismatch,
            restore_rng=True,
            map_location=device,
        )
        checkpoint_monitor = str(
            checkpoint.get("extra", {}).get("best_metric_name", "event_macro_mae")
        )
        if checkpoint_monitor != monitor:
            raise RuntimeError(
                "Resume checkpoint monitor differs from the active configuration: "
                f"{checkpoint_monitor!r} != {monitor!r}"
            )
        checkpoint_semantics = checkpoint_depth_output_semantics(checkpoint)
        configured_semantics = str(
            config["model"].get("depth_output_semantics", "probability_weighted_v1")
        )
        if checkpoint_semantics != configured_semantics:
            raise RuntimeError(
                "Resume checkpoint depth semantics differs from the active configuration: "
                f"{checkpoint_semantics!r} != {configured_semantics!r}"
            )
        checkpoint_config = checkpoint.get("resolved_config", {})
        checkpoint_loss = (
            checkpoint_config.get("loss", {})
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        checkpoint_depth_bins = resolve_depth_stratification_bins(
            checkpoint_loss if isinstance(checkpoint_loss, Mapping) else {}, normalizer
        )
        if checkpoint_depth_bins != depth_bins:
            raise RuntimeError(
                "Resume checkpoint depth strata differ from the active configuration: "
                f"{checkpoint_depth_bins} != {depth_bins}"
            )
        active_loss = jsonable_config(config["loss"])
        saved_loss = jsonable_config(dict(checkpoint_loss))
        if saved_loss != active_loss:
            raise RuntimeError(
                "Resume checkpoint loss configuration differs from the active "
                "configuration. Start a new run for a changed objective."
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        LOGGER.info("Resumed %s at epoch %d", args.resume, start_epoch)

    writer = SummaryWriter(run_dir / "tensorboard") if rank == 0 and config["logging"]["tensorboard"] else None
    if rank == 0:
        atomic_write_json(run_dir / "resolved_config.json", jsonable_config(config))
        atomic_write_json(run_dir / "environment.json", environment_payload(device))
        atomic_write_json(run_dir / "dataset_fingerprint.json", fingerprint)
        atomic_write_json(
            run_dir / "model_summary.json",
            {
                "name": str(config["model"]["name"]),
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "positive_prior": prior,
                "depth_output_semantics": config["model"].get(
                    "depth_output_semantics", "probability_weighted_v1"
                ),
                "best_metric": monitor,
                "depth_stratification_edges_m": depth_bins,
                "primary_depth_stratification_edges_m": normalizer.train_depth_bins,
            },
        )
    patience = 0
    start_time = time.perf_counter()
    try:
        for epoch in range(start_epoch, epochs):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                criterion,
                device,
                epoch,
                accumulation,
                float(config["training"]["grad_clip_norm"]),
                amp_enabled,
                args.max_train_batches,
                run_dir,
                rank,
            )
            val_summary, _, _, _ = evaluate_loader(
                model,
                val_loader,
                device,
                depth_bins,
                primary_depth_bins=normalizer.train_depth_bins,
                criterion=criterion,
                epoch=epoch,
                max_batches=args.max_val_batches,
                progress=rank == 0,
            )
            if monitor not in val_summary:
                raise KeyError(
                    f"Configured best metric {monitor!r} is absent from validation summary"
                )
            metric = float(val_summary[monitor])
            improved = metric < best_metric
            if improved:
                best_metric, patience = metric, 0
            else:
                patience += 1
            if rank == 0:
                row = {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in val_summary.items()},
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    f"best_{monitor}": best_metric,
                }
                append_csv(run_dir / "metrics_by_epoch.csv", row)
                if writer is not None:
                    for key, value in row.items():
                        if key != "epoch" and isinstance(value, (int, float)) and np.isfinite(value):
                            writer.add_scalar(key, value, epoch)
                common = dict(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    resolved_config=jsonable_config(config),
                    dataset_fingerprint=fingerprint,
                    extra={
                        "total_parameters": total_parameters,
                        "positive_prior": prior,
                        "best_metric_name": monitor,
                        "depth_stratification_edges_m": depth_bins,
                        "primary_depth_stratification_edges_m": normalizer.train_depth_bins,
                    },
                )
                save_checkpoint(run_dir / "last.pth", **common)
                if improved:
                    save_checkpoint(run_dir / "best.pth", **common)
                LOGGER.info(
                    "epoch=%d train_loss=%.5f val_%s=%.5f best=%.5f",
                    epoch,
                    train_metrics["total"],
                    monitor,
                    metric,
                    best_metric,
                )
            if patience >= int(config["training"]["early_stop_patience"]):
                LOGGER.info("Early stopping at epoch %d", epoch)
                break
    finally:
        if writer is not None:
            writer.close()
        if rank == 0:
            elapsed = time.perf_counter() - start_time
            atomic_write_json(
                run_dir / "training_runtime.json",
                {
                    "elapsed_seconds": elapsed,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0,
                },
            )
        cleanup_distributed()
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-fingerprint-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = run_training(args)
    if is_main_process():
        print(f"training output: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

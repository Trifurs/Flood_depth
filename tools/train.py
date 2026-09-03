#!/usr/bin/env python3
"""Train a registered flood-depth model with strict reproducibility."""

from __future__ import annotations

import argparse
import csv
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
from datasets.band_selection import resolve_band_spec
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from datasets.reliability_spec import ReliabilitySpec
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from datasets.samplers import (
    BalancedRemainderBatchSampler,
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
    training_identity_sha256,
)
from utils.config import jsonable_config, load_config
from utils.distributed import (
    broadcast_object,
    cleanup_distributed,
    initialize_distributed,
    is_main_process,
    reduce_weighted_metrics,
)
from utils.logging import append_csv, setup_logging
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model
from utils.seed import seed_everything, seed_worker
from utils.amp import resolve_amp
from utils.ema import ModelEMA
from utils.optim import build_optimizer, build_scheduler


LOGGER = logging.getLogger("train")


def infer_legacy_patience(
    metrics_path: Path,
    checkpoint_epoch: int,
    monitor: str,
    weights: str,
    min_delta: float,
) -> int:
    """Recover an early-stop counter from legacy epoch CSV when possible."""

    if not metrics_path.exists():
        return 0
    metric_column = (
        f"val_ema_{monitor}" if weights == "ema" else f"val_{monitor}"
    )
    best = float("inf")
    last_improvement = -1
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            if epoch > checkpoint_epoch or not row.get(metric_column):
                continue
            value = float(row[metric_column])
            if value < best - min_delta:
                best = value
                last_improvement = epoch
    return max(0, checkpoint_epoch - last_improvement) if last_improvement >= 0 else 0


def accumulation_window_sizes(total_batches: int, accumulation_steps: int) -> list[int]:
    if total_batches < 0 or accumulation_steps <= 0:
        raise ValueError("Invalid accumulation dimensions")
    return [
        min(accumulation_steps, total_batches - start)
        for start in range(0, total_batches, accumulation_steps)
    ]


def normalize_accumulated_gradients(model: torch.nn.Module, sample_count: int) -> None:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)


def resolve_cli(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resume = getattr(args, "resume", None)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if resume is not None and init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
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


def make_training_context(
    config: Mapping[str, Any], loader: DataLoader, epochs: int, accumulation: int,
    max_train_batches: int | None,
) -> dict[str, Any]:
    effective_batches = min(len(loader), max_train_batches or len(loader))
    steps_per_epoch = math.ceil(effective_batches / accumulation)
    return {
        "epochs": int(epochs),
        "batch_size": int(config["training"]["batch_size"]),
        "gradient_accumulation_steps": int(accumulation),
        "sampler": type(getattr(loader, "sampler", None)).__name__,
        "batch_sampler": type(getattr(loader, "batch_sampler", None)).__name__,
        "sampling": jsonable_config(config["dataset"]["sampling"]),
        "augmentation": jsonable_config(config["dataset"]["augmentation"]),
        "seed": int(config["seed"]),
        "effective_train_batches": int(effective_batches),
        "steps_per_epoch": int(steps_per_epoch),
        "planned_total_optimizer_steps": int(epochs * steps_per_epoch),
        "warmup_steps": int(config["scheduler"]["warmup_epochs"]) * steps_per_epoch,
    }


def create_dataloaders(
    config: dict[str, Any], rank: int = 0, world_size: int = 1
) -> tuple[DataLoader, DataLoader, FloodDepthDataset, FloodDepthDataset]:
    augmentation = config["dataset"]["augmentation"]
    input_spec = ModelInputSpec.from_config(config)
    transform = SynchronousAugment(
        float(augmentation["horizontal_flip_probability"]),
        float(augmentation["vertical_flip_probability"]),
        float(augmentation["rotate90_probability"]),
        augmentation.get("modality_dropout_probability"),
        augmentation.get("feature_dropout_probability"),
        augmentation.get("sensor_missing_simulation_probability"),
        input_mode=input_spec.mode,
    )
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    train_dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        "train",
        transform=transform,
        band_spec=band_spec,
        input_spec=input_spec,
    )
    val_dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], "val",
        band_spec=band_spec,
        input_spec=input_spec,
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
    if workers > 0:
        common["prefetch_factor"] = int(config["training"].get("prefetch_factor", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BalancedRemainderBatchSampler(
            sampler,
            int(config["training"]["batch_size"]),
            bool(config["training"].get("drop_last", False)),
        ),
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
    amp_dtype: torch.dtype,
    max_batches: int | None,
    run_dir: Path,
    rank: int,
    ema: ModelEMA | None = None,
    log_every_steps: int = 10,
    csv_enabled: bool = True,
    non_blocking: bool = True,
    input_spec: ModelInputSpec | None = None,
) -> dict[str, float]:
    model.train()
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    batches = 0
    samples = 0
    iterator = tqdm(loader, desc=f"train {epoch + 1}", leave=False, disable=rank != 0)
    effective_batches = min(len(loader), max_batches or len(loader))
    accumulated_samples = 0
    optimizer_steps = 0
    skipped_steps = 0
    interval_start = time.perf_counter()
    interval_samples = 0
    interval_data_time = 0.0
    interval_compute_time = 0.0
    last_batch_end = interval_start
    for batch_index, cpu_batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch_received = time.perf_counter()
        interval_data_time += batch_received - last_batch_end
        compute_start = batch_received
        batch = move_to_device(cpu_batch, device, non_blocking=non_blocking)
        batch_size = int(cpu_batch["label"].shape[0])
        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
        ):
            outputs = model(prepare_model_inputs(batch, input_spec))
            loss, components = criterion(outputs, batch, epoch)
        nonfinite = [name for name, value in components.items()
                     if not torch.isfinite(value.detach()).all()]
        if not torch.isfinite(loss).all() or nonfinite:
            def value_range(value: Any) -> list[float | None]:
                if not isinstance(value, torch.Tensor):
                    return [None, None]
                finite = value.detach().float()[torch.isfinite(value.detach().float())]
                if finite.numel() == 0:
                    return [None, None]
                return [float(finite.min().cpu()), float(finite.max().cpu())]
            positive_mask = batch["masks"]["valid_depth_mask"] > 0.5
            unlabeled_mask = (
                (batch["validity"]["output_valid"] > 0.5) & ~positive_mask
                & ~(batch["masks"]["permanent_water_mask"] > 0.5)
                & ~(batch["masks"]["extreme_high_mask"] > 0.5)
            )
            graph_payload = {
                key: float(value.detach().float().mean().cpu())
                for key, value in outputs.get("graph_diagnostics", {}).items()
                if isinstance(value, torch.Tensor)
            }
            payload = {
                "epoch": epoch,
                "batch": batch_index,
                "sample_id": str(cpu_batch.get("metadata", {}).get("sample_id", "unknown")),
                "nonfinite_components": nonfinite,
                "loss": float(loss.detach().float().cpu()) if torch.isfinite(loss).all() else None,
                "prediction_range": value_range(outputs.get("depth")),
                "target_range": value_range(batch.get("label")),
                "uncertainty_range": value_range(outputs.get("uncertainty_scale")),
                "positive_pixels": int(positive_mask.sum().item()),
                "unlabeled_pixels": int(unlabeled_mask.sum().item()),
                "graph_diagnostics": graph_payload,
                "amp_scale": float(scaler.get_scale()),
            }
            if rank == 0:
                with (run_dir / "nonfinite_losses.jsonl").open("a", encoding="utf-8") as handle:
                    import json
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, batch={batch_index}; components={nonfinite}")
        # Accumulate sums over samples, then normalize the complete (including
        # short final) window once before clipping.  This prevents a singleton or
        # max-batches remainder from receiving a full-sized update.
        scaler.scale(loss * batch_size).backward()
        accumulated_samples += batch_size
        final_batch = batch_index + 1 >= effective_batches
        if (batch_index + 1) % accumulation_steps == 0 or final_batch:
            scaler.unscale_(optimizer)
            normalize_accumulated_gradients(model, accumulated_samples)
            raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            clipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            successful_step = (not scaler.is_enabled()) or scaler.get_scale() >= scale_before
            if successful_step:
                if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
                if ema is not None:
                    ema.update(model)
                optimizer_steps += 1
            else:
                skipped_steps += 1
            accumulated_samples = 0
        else:
            raw_grad_norm = loss.new_tensor(float("nan"))
            clipped_grad_norm = loss.new_tensor(float("nan"))
        batches += 1
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * batch_size
        samples += batch_size
        interval_samples += batch_size
        interval_compute_time += time.perf_counter() - compute_start
        if rank == 0:
            iterator.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if rank == 0 and csv_enabled and (
            batch_index % max(1, log_every_steps) == 0 or final_batch
        ):
            now = time.perf_counter()
            graph = outputs.get("graph_diagnostics", {})
            modality_weights = outputs.get("modality_weights", [])
            modality_mean = torch.stack([value.mean() for value in modality_weights]).mean() if modality_weights else loss.new_tensor(float("nan"))
            s1_weight_mean = torch.stack([value[:, 0:1].mean() for value in modality_weights]).mean() if modality_weights else loss.new_tensor(float("nan"))
            s2_weight_mean = torch.stack([value[:, 1:2].mean() for value in modality_weights]).mean() if modality_weights else loss.new_tensor(float("nan"))
            uncertainty = outputs["uncertainty_scale"].detach().float()
            elapsed = max(now - interval_start, 1e-9)
            terrain_gate = outputs.get("terrain_gates", [])
            terrain_gate_mean = (
                torch.stack([value.mean() for value in terrain_gate]).mean()
                if terrain_gate else loss.new_tensor(float("nan"))
            )
            fusion_entropy = outputs.get("fusion_entropy", [])
            fusion_entropy_mean = (
                torch.stack([value.mean() for value in fusion_entropy]).mean()
                if fusion_entropy else loss.new_tensor(float("nan"))
            )
            if isinstance(batch.get("validity"), Mapping):
                if "s2_valid" in batch["validity"]:
                    both_valid = (
                        torch.minimum(batch["validity"]["s1_valid"], batch["validity"]["s2_valid"]) > 0.5
                    ).float().mean()
                else:
                    both_valid = (batch["validity"]["s1_event_support"] > 0.5).float().mean()
            else:
                both_valid = loss.new_tensor(float("nan"))
            step_row = {
                "epoch": epoch,
                "batch": batch_index,
                "loss": float(loss.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "raw_gradient_norm": float(raw_grad_norm.detach().cpu()),
                "clipped_gradient_norm": float(clipped_grad_norm.detach().cpu()),
                "amp_scale": float(scaler.get_scale()),
                "optimizer_steps": optimizer_steps,
                "skipped_steps": skipped_steps,
                "step_time_seconds": elapsed,
                "samples_per_second": interval_samples / elapsed,
                "interval_samples": interval_samples,
                "data_time_seconds": interval_data_time,
                "compute_time_seconds": interval_compute_time,
                "graph_gate_mean": float(graph.get("gate_mean", loss.new_tensor(float("nan"))).detach().cpu()),
                "graph_gamma_mean": float(graph.get("gamma_mean", loss.new_tensor(float("nan"))).detach().cpu()),
                "terrain_gate_mean": float(terrain_gate_mean.detach().cpu()),
                "uncertainty_scale_mean": float(uncertainty.mean().cpu()),
                "uncertainty_scale_p90": float(torch.quantile(uncertainty.flatten(), 0.9).cpu()),
                "gpu_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                "gpu_reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else 0,
            }
            if modality_weights:
                step_row.update({
                    "modality_weight_mean": float(modality_mean.detach().cpu()),
                    "s1_weight_mean": float(s1_weight_mean.detach().cpu()),
                    "s2_weight_mean": float(s2_weight_mean.detach().cpu()),
                    "fusion_entropy_mean": float(fusion_entropy_mean.detach().cpu()),
                    "both_sensor_valid_ratio": float(both_valid.detach().cpu()),
                })
            append_csv(run_dir / "train_steps.csv", step_row)
            interval_start = now
            interval_samples = 0
            interval_data_time = 0.0
            interval_compute_time = 0.0
        last_batch_end = time.perf_counter()
    if batches == 0:
        raise RuntimeError("No train batches were executed")
    result = {name: value / samples for name, value in sums.items()}
    result.update({"optimizer_steps": float(optimizer_steps), "amp_skipped_steps": float(skipped_steps)})
    averaged = reduce_weighted_metrics(result, samples, device)
    # Optimizer-step counters are identical across correctly sharded DDP ranks;
    # retain the per-rank value rather than treating them as sample averages.
    averaged["optimizer_steps"] = float(optimizer_steps)
    averaged["amp_skipped_steps"] = float(skipped_steps)
    return averaged


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
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        selected_run_dir = (
            args.resume.resolve().parent
            if args.resume is not None
            else args.output.resolve() if args.output is not None
            else Path(config["runs_root"]) / "train" / f"{config['run_name']}_{timestamp}"
        )
        run_dir_value: str | None = str(selected_run_dir)
    else:
        run_dir_value = None
    run_dir = Path(broadcast_object(run_dir_value, source=0))
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log" if rank == 0 else None)
    LOGGER.info("Resolved config: %s", jsonable_config(config))
    train_loader, val_loader, train_dataset, _ = create_dataloaders(config, rank, world_size)
    input_spec = train_dataset.input_spec
    normalizer = train_dataset.normalizer
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    prior_config = config["dataset"]["positive_prior"]
    prior = normalizer.positive_prior if prior_config["mode"] == "auto" else float(prior_config["value"])
    minimum, maximum = float(prior_config["minimum"]), float(prior_config["maximum"])
    prior = float(np.clip(prior, minimum, maximum))
    LOGGER.info("nnPU positive prior=%f method=%s", prior, prior_config["mode"])
    LOGGER.info("train-only depth stratification edges (m)=%s", depth_bins)

    model = build_model(config).to(device)
    parent_checkpoint = None
    if args.init_checkpoint is not None:
        parent_checkpoint = args.init_checkpoint.resolve()
        init_payload = load_checkpoint(
            parent_checkpoint, model, map_location=device,
            adopt_checkpoint_output_semantics=True,
        )
        if args.init_weights == "ema" and init_payload.get("ema_model") is not None:
            model.load_state_dict(init_payload["ema_model"], strict=True)
        elif args.init_weights == "ema":
            LOGGER.warning(
                "init checkpoint has no EMA state; initializing the new stage from raw weights"
            )
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
    optimizer = build_optimizer(model, config)
    epochs = int(config["training"]["epochs"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    effective_train_batches = min(
        len(train_loader), args.max_train_batches or len(train_loader)
    )
    steps_per_epoch = math.ceil(effective_train_batches / accumulation)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(config["scheduler"]["warmup_epochs"]) * steps_per_epoch
    training_context = make_training_context(
        config, train_loader, epochs, accumulation, args.max_train_batches
    )
    scheduler = build_scheduler(optimizer, config, total_steps, warmup_steps)
    amp_enabled, amp_dtype, scaler_enabled = resolve_amp(
        device, bool(config["training"]["amp"]),
        str(config["training"].get("amp_dtype", "float16")),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    ema = ModelEMA(
        model, float(config["training"].get("ema_decay", 0.999)),
        int(config["training"].get("ema_warmup_steps", 0)),
    ) if bool(config["training"].get("ema_enabled", False)) else None
    criterion = CompositeFloodDepthLoss(
        config["loss"], prior, depth_bins, normalizer.train_depth_bins,
        normalizer.train_depth_bin_counts,
    )
    fingerprint = dataset_fingerprint(config)
    monitor = str(config["training"]["best_metric"])
    start_epoch, best_metric, patience = 0, float("inf"), 0
    global_step = 0
    best_raw_metric, best_ema_metric = float("inf"), float("inf")
    if args.resume is not None:
        current_identity = training_identity_sha256(
            jsonable_config(config), fingerprint, version=3,
            training_context=training_context,
        )
        legacy_identity = training_identity_sha256(
            jsonable_config(config), fingerprint, version=2
        )
        legacy_v1_identity = training_identity_sha256(
            jsonable_config(config), fingerprint, version=1
        )
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
            expected_training_identity_sha256=current_identity,
            expected_legacy_training_identity_sha256=legacy_identity,
            expected_legacy_v1_training_identity_sha256=legacy_v1_identity,
        )
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
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
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint["best_metric"])
        saved_extra = checkpoint.get("extra", {})
        best_raw_metric = float(saved_extra.get("best_raw_metric", best_metric))
        best_ema_metric = float(saved_extra.get("best_ema_metric", best_metric))
        patience = int(checkpoint.get("extra", {}).get("early_stop_patience", -1))
        if patience < 0:
            patience = infer_legacy_patience(
                run_dir / "metrics_by_epoch.csv",
                int(checkpoint["epoch"]),
                monitor,
                str(config["training"].get("best_weights", "raw")),
                float(config["training"].get("min_delta", 0.0)),
            )
        if world_size > 1:
            derived_seed = int(config["seed"]) + rank + 1_000_003 * start_epoch
            seed_everything(derived_seed, bool(config["deterministic"]))
            train_loader.generator.manual_seed(derived_seed)
            val_loader.generator.manual_seed(derived_seed + 1)
        LOGGER.info("Resumed %s at epoch %d", args.resume, start_epoch)

    writer = SummaryWriter(run_dir / "tensorboard") if rank == 0 and config["logging"]["tensorboard"] else None
    if rank == 0:
        atomic_write_json(run_dir / "resolved_config.json", jsonable_config(config))
        atomic_write_json(run_dir / "environment.json", environment_payload(device))
        atomic_write_json(run_dir / "dataset_fingerprint.json", fingerprint)
        if parent_checkpoint is not None:
            atomic_write_json(
                run_dir / "run_metadata.json",
                {"stage": "init_checkpoint", "parent_checkpoint": str(parent_checkpoint),
                 "init_weights": args.init_weights},
            )
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
                "resolved_model_bands": config["dataset"].get("resolved_model_bands"),
                "amp_dtype": str(amp_dtype),
            },
        )
    start_time = time.perf_counter()
    try:
        for epoch in range(start_epoch, epochs):
            if (
                epoch >= int(config["training"].get("minimum_epochs", 0))
                and patience >= int(config["training"]["early_stop_patience"])
            ):
                LOGGER.info(
                    "Resume checkpoint already satisfies early stopping at epoch %d",
                    epoch - 1,
                )
                break
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
                amp_dtype,
                args.max_train_batches,
                run_dir,
                rank,
                ema,
                int(config["logging"]["log_every_steps"]),
                bool(config["logging"]["csv"]),
                bool(config["training"].get("non_blocking", device.type == "cuda")),
                input_spec,
            )
            global_step += int(train_metrics.get("optimizer_steps", 0.0))
            validation_interval = max(
                1, int(config["training"].get("validation_interval", 1))
            )
            should_validate = (
                (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
            )
            if not should_validate:
                if rank == 0:
                    if bool(config["checkpoint"]["save_last"]):
                        save_checkpoint(
                            run_dir / "last.pth",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch,
                            best_metric,
                            jsonable_config(config),
                            fingerprint,
                            extra={
                                "total_parameters": total_parameters,
                                "positive_prior": prior,
                                "best_metric_name": monitor,
                                "depth_stratification_edges_m": depth_bins,
                                "primary_depth_stratification_edges_m": normalizer.train_depth_bins,
                                "early_stop_patience": patience,
                                "best_raw_metric": best_raw_metric,
                                "best_ema_metric": best_ema_metric,
                            },
                            ema=ema,
                            training_context=training_context,
                            global_step=global_step,
                        )
                    LOGGER.info(
                        "epoch=%d train_loss=%.5f validation=skipped interval=%d",
                        epoch,
                        train_metrics["total"],
                        validation_interval,
                    )
                continue
            val_summary = None
            ema_summary = None
            if rank == 0:
                evaluation_model = model.module if hasattr(model, "module") else model
                val_summary, _, _, _ = evaluate_loader(
                    evaluation_model,
                    val_loader,
                    device,
                    depth_bins,
                    primary_depth_bins=normalizer.train_depth_bins,
                    criterion=criterion,
                    epoch=epoch,
                    max_batches=args.max_val_batches,
                    progress=True,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    input_spec=input_spec,
                )
                if ema is not None:
                    with ema.swap_in(model):
                        ema_summary, _, _, _ = evaluate_loader(
                            evaluation_model, val_loader, device, depth_bins,
                            primary_depth_bins=normalizer.train_depth_bins,
                            criterion=criterion, epoch=epoch,
                            max_batches=args.max_val_batches, progress=True,
                            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                            input_spec=input_spec,
                        )
            val_summary = broadcast_object(val_summary, source=0)
            ema_summary = broadcast_object(ema_summary, source=0)
            selected_summary = (
                ema_summary if str(config["training"].get("best_weights", "raw")) == "ema" and ema_summary is not None
                else val_summary
            )
            if monitor not in selected_summary:
                raise KeyError(
                    f"Configured best metric {monitor!r} is absent from validation summary"
                )
            raw_metric = float(val_summary[monitor])
            ema_metric = float(ema_summary[monitor]) if ema_summary is not None else None
            raw_improved = raw_metric < best_raw_metric - float(config["training"].get("min_delta", 0.0))
            ema_improved = (
                ema_metric is not None
                and ema_metric < best_ema_metric - float(config["training"].get("min_delta", 0.0))
            )
            if raw_improved:
                best_raw_metric = raw_metric
            if ema_improved and ema_metric is not None:
                best_ema_metric = ema_metric
            metric = float(selected_summary[monitor])
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metric)
            improved = metric < best_metric - float(config["training"].get("min_delta", 0.0))
            if improved:
                best_metric, patience = metric, 0
            else:
                patience += 1
            if rank == 0:
                row = {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in val_summary.items()},
                    **({f"val_ema_{key}": value for key, value in ema_summary.items()} if ema_summary else {}),
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
                        "early_stop_patience": patience,
                        "best_raw_metric": best_raw_metric,
                        "best_ema_metric": best_ema_metric,
                    },
                )
                if bool(config["checkpoint"]["save_last"]):
                    save_checkpoint(
                        run_dir / "last.pth", ema=ema, training_context=training_context,
                        global_step=global_step, **common
                    )
                if improved:
                    if bool(config["checkpoint"]["save_best"]):
                        save_checkpoint(
                            run_dir / "best.pth", ema=ema, training_context=training_context,
                            global_step=global_step, **common
                        )
                if raw_improved and bool(config["checkpoint"]["save_best"]):
                    raw_common = dict(common)
                    raw_common["best_metric"] = best_raw_metric
                    save_checkpoint(
                        run_dir / "best_raw.pth",
                        ema=ema, training_context=training_context,
                        global_step=global_step, **raw_common
                    )
                if ema_improved and bool(config["checkpoint"]["save_best"]):
                    ema_common = dict(common)
                    ema_common["best_metric"] = best_ema_metric
                    save_checkpoint(
                        run_dir / "best_ema.pth",
                        ema=ema, training_context=training_context,
                        global_step=global_step, **ema_common
                    )
                LOGGER.info(
                    "epoch=%d train_loss=%.5f val_%s=%.5f best=%.5f",
                    epoch,
                    train_metrics["total"],
                    monitor,
                    metric,
                    best_metric,
                )
            if (
                epoch + 1 >= int(config["training"].get("minimum_epochs", 0))
                and patience >= int(config["training"]["early_stop_patience"])
            ):
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
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--init-weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-fingerprint-mismatch", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = run_training(args)
    if is_main_process():
        print(f"training output: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

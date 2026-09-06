#!/usr/bin/env python3
"""Validate an active V15.2 weak-physics loss on one real S1+terrain batch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from losses.composite_loss import CompositeFloodDepthLoss
from tools.evaluate import dataset_fingerprint, embed_source_fingerprints, frozen_depth_balance_for_config
from tools.train import _physics_output_gradient_diagnostics
from utils.amp import resolve_amp
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.epoch < 0 or args.batch_size <= 0:
        raise ValueError("epoch must be nonnegative and batch-size must be positive")

    config = embed_source_fingerprints(load_config(args.config))
    input_spec = ModelInputSpec.from_config(config)
    if not input_spec.is_s1_only:
        raise ValueError("physics probe is strictly Sentinel-1 plus terrain only")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    contract = DatasetContract.load(config["dataset"]["contract"])
    dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        args.split,
        band_spec=resolve_band_spec(config, contract),
        input_spec=input_spec,
        minimum_event_band_fraction=float(config["dataset"].get("minimum_event_band_fraction", 1.0)),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))
    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), contract)
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    prior_config = config["dataset"]["positive_prior"]
    prior = (
        normalizer.positive_prior
        if prior_config["mode"] == "auto"
        else float(prior_config["value"])
    )
    criterion = CompositeFloodDepthLoss(
        config["loss"],
        prior,
        depth_bins,
        normalizer.train_depth_bins,
        normalizer.train_depth_bin_counts,
        frozen_depth_balance_for_config(config),
    )
    if criterion.physics_weight(args.epoch) <= 0.0:
        raise ValueError("physics probe epoch does not activate a nonzero physics weight")
    model = build_model(config).to(device).train()
    checkpoint = load_checkpoint(
        args.checkpoint,
        model,
        expected_fingerprint=dataset_fingerprint(config),
        map_location=device,
    )
    amp_enabled, amp_dtype, _ = resolve_amp(
        device,
        bool(config["training"].get("amp", False)),
        str(config["training"].get("amp_dtype", "float16")),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch = move_to_device(batch, device, non_blocking=device.type == "cuda")
    with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
        outputs = model(prepare_model_inputs(batch, input_spec))
        loss, components = criterion(outputs, batch, args.epoch)
    gradient_diagnostics = _physics_output_gradient_diagnostics(outputs, components)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite_parameter_gradients = bool(gradients) and all(
        bool(torch.isfinite(value).all()) for value in gradients
    )
    payload: dict[str, Any] = {
        "scope": "single real batch; no optimizer step; no test split",
        "split": args.split,
        "sentinel2_groups_requested": [],
        "device": str(device),
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "physics_epoch": args.epoch,
        "physics_effective_weight": float(components["physics_effective_weight"].detach().cpu()),
        "physics_mode": str(config["loss"]["physics_mode"]),
        "physics_loss": float(components["physics"].detach().cpu()),
        "physics_active_pair_fraction": float(components["physics_active_pair_fraction"].detach().cpu()),
        "physics_active_pair_count": float(components["physics_active_pair_count"].detach().cpu()),
        "physics_candidate_pair_count": float(components["physics_candidate_pair_count"].detach().cpu()),
        "physics_mean_violation_m": float(components["physics_mean_violation_m"].detach().cpu()),
        "physics_p90_violation_m": float(components["physics_p90_violation_m"].detach().cpu()),
        "physics_mean_sar_compatibility": float(components["physics_mean_sar_compatibility"].detach().cpu()),
        "physics_mean_barrier_weight": float(components["physics_mean_barrier_weight"].detach().cpu()),
        "physics_mean_complexity_weight": float(components["physics_mean_complexity_weight"].detach().cpu()),
        "gradient_diagnostics": gradient_diagnostics,
        "parameter_gradients_finite": finite_parameter_gradients,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }
    atomic_write_json(args.output, payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

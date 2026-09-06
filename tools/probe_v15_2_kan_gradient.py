#!/usr/bin/env python3
"""Record one reproducible train-batch KAN gradient probe for V15.2.

This is an audit utility, not a training or model-selection entry point.  It
uses a single training batch, never opens the test split, and records the
gradient of the loaded checkpoint under the configured objective.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.flooddepth_dataset import prepare_model_inputs
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from losses.composite_loss import CompositeFloodDepthLoss
from tools.evaluate import embed_source_fingerprints, frozen_depth_balance_for_config
from tools.train import create_dataloaders
from utils.amp import resolve_amp
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model


def _norm(parameter: torch.nn.Parameter | None) -> float:
    if parameter is None or parameter.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(parameter.grad.detach().float()).cpu())


def _all_finite(parameter: torch.nn.Parameter | None) -> bool:
    return bool(
        parameter is not None
        and parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--objective-epoch",
        type=int,
        default=24,
        help="Epoch used solely to instantiate the configured scheduled objective.",
    )
    args = parser.parse_args()
    if args.objective_epoch < 0:
        raise ValueError("objective epoch must be nonnegative")

    config = embed_source_fingerprints(load_config(args.config))
    config["training"]["num_workers"] = 0
    config["training"]["persistent_workers"] = False
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    train_loader, _, train_dataset, _ = create_dataloaders(config)
    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), train_dataset.contract)
    bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    prior_config = config["dataset"]["positive_prior"]
    prior = (
        normalizer.positive_prior
        if prior_config["mode"] == "auto"
        else float(prior_config["value"])
    )
    criterion = CompositeFloodDepthLoss(
        config["loss"],
        prior,
        bins,
        normalizer.train_depth_bins,
        normalizer.train_depth_bin_counts,
        frozen_depth_balance_for_config(config),
    )
    model = build_model(config).to(device).train()
    load_checkpoint(args.checkpoint, model, map_location=device)
    batch = move_to_device(next(iter(train_loader)), device)
    amp_enabled, amp_dtype, _ = resolve_amp(
        device,
        bool(config["training"].get("amp", False)),
        str(config["training"].get("amp_dtype", "float16")),
    )
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
        outputs = model(prepare_model_inputs(batch))
        loss, terms = criterion(outputs, batch, args.objective_epoch)
    loss.backward()
    graph = model.graph
    spline = graph.edge_kan.spline_coefficients
    base = graph.edge_kan.base_weight
    gamma = graph.raw_gamma
    payload: dict[str, Any] = {
        "scope": "single training batch gradient audit; no optimizer step or test split",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "amp_dtype": str(amp_dtype) if amp_enabled else None,
        "objective_epoch": int(args.objective_epoch),
        "objective_total": float(loss.detach().float().cpu()),
        "physics_effective_weight": float(terms["physics_effective_weight"].detach().cpu()),
        "edge_kan_spline_gradient_norm": _norm(spline),
        "edge_kan_spline_gradient_finite": _all_finite(spline),
        "edge_kan_base_gradient_norm": _norm(base),
        "edge_kan_base_gradient_finite": _all_finite(base),
        "graph_raw_gamma_gradient_norm": _norm(gamma),
        "graph_raw_gamma_gradient_finite": _all_finite(gamma),
    }
    payload["edge_kan_spline_gradient_nonzero"] = bool(
        payload["edge_kan_spline_gradient_norm"] > 0.0
    )
    atomic_write_json(args.output, payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

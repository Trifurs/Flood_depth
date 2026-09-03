#!/usr/bin/env python3
"""Collect one-batch Hydro-v14 graph, gate, and physical-feature diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.flooddepth_dataset import prepare_model_inputs
from tools.evaluate import embed_source_fingerprints
from tools.train import create_dataloaders
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model


def _stats(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().flatten()
    return {
        "mean": float(flat.mean()), "std": float(flat.std(unbiased=False)),
        "p05": float(torch.quantile(flat, 0.05)),
        "p50": float(torch.quantile(flat, 0.50)),
        "p95": float(torch.quantile(flat, 0.95)),
        "minimum": float(flat.min()), "maximum": float(flat.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = embed_source_fingerprints(load_config(args.config))
    config["model"]["diagnostic_mode"] = True
    config["training"]["num_workers"] = 0
    config["training"]["persistent_workers"] = False
    device = torch.device(args.device)
    train_loader, val_loader, _, _ = create_dataloaders(config)
    loader = val_loader if args.split == "val" else train_loader
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    batch = move_to_device(next(iter(loader)), device)
    with torch.no_grad():
        outputs = model(prepare_model_inputs(batch))
    graph = outputs["graph_diagnostics"]
    graph_scalars = {}
    graph_distributions = {}
    for key, value in graph.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim == 0 or value.numel() <= 16:
            graph_scalars[key] = value.detach().float().cpu().tolist()
        else:
            graph_distributions[key] = _stats(value)
    gate_rows = []
    for stage, item in enumerate(outputs.get("decoder_gates", ())):
        gate_rows.append({"stage": stage, "sensor": _stats(item["sensor"]), "terrain": _stats(item["terrain"])})
    physical_names = ("physics_elevation", "z_ground_proxy", "z_relative", "z_barrier", "local_relief", "dz_dx", "dz_dy", "derived_slope_m_per_m")
    payload = {
        "schema_version": "hydrov14.diagnostics.v1", "config": str(args.config),
        "checkpoint": str(args.checkpoint), "split": args.split, "device": str(device),
        "batch_size": int(batch["label"].shape[0]), "graph_scalars": graph_scalars,
        "graph_distributions": graph_distributions, "decoder_gates": gate_rows,
        "physical_features": {name: _stats(outputs["physical_features"][name]) for name in physical_names},
        "notes": ["Graph diagnostics are static topographic descriptors plus latent aggregation diagnostics; they are not hydraulic fluxes or conservation residuals.", "DSM outputs are reported as DSM/ground-like proxies, not bare-earth DTM."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

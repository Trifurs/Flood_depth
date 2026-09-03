#!/usr/bin/env python3
"""Build train-only robust statistics for HydroEdgeKAN edge features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.flooddepth_dataset import prepare_model_inputs
from models.terrain_graph_kan import DIRECTIONS, _roll_with_boundary_mask, _masked_pool
from tools.evaluate import embed_source_fingerprints
from tools.train import create_dataloaders
from utils.config import load_config
from utils.misc import move_to_device
from utils.registry import build_model


FEATURES = ("signed_dz", "edge_slope", "relative_height_or_barrier", "local_relief", "neighbour_distance")


def _update(store, values):
    values = values.detach().float().flatten()
    values = values[torch.isfinite(values)]
    if values.numel():
        store.extend(values.cpu().tolist())


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=Path("configs/pa_hydrokan/subset150_v13_corrected_final.xml")); parser.add_argument("--output", type=Path, default=Path("artifacts/optimization/hydrov13_2/graph_edge_train_stats.json")); parser.add_argument("--max-batches", type=int, default=0); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); config = embed_source_fingerprints(load_config(args.config)); config["training"]["num_workers"] = 0; config["training"]["persistent_workers"] = False
    device = torch.device(args.device); loader, _, _, _ = create_dataloaders(config)
    model = build_model(config).to(device).eval(); values = {name: [] for name in FEATURES}; count = 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_to_device(cpu_batch, device); inp = prepare_model_inputs(batch)
            _, physical = model.terrain(inp["terrain"], inp["terrain_raw"], inp["dem_valid"])
            size = (physical["z_hyd"].shape[-2] // 8, physical["z_hyd"].shape[-1] // 8)
            size = (max(1, size[0]), max(1, size[1]))
            dem = F.adaptive_avg_pool2d(physical["dem_valid"], size)
            z = _masked_pool(physical["z_hyd"], physical["dem_valid"], size)
            barrier = _masked_pool(physical["z_barrier"], physical["dem_valid"], size)
            relief = _masked_pool(physical["local_relief"], physical["dem_valid"], size)
            node_valid = (dem > 0.5)
            for dy, dx in DIRECTIONS:
                nz, boundary = _roll_with_boundary_mask(z, dy, dx)
                nb, _ = _roll_with_boundary_mask(barrier, dy, dx)
                nr, _ = _roll_with_boundary_mask(relief, dy, dx)
                nv, _ = _roll_with_boundary_mask(node_valid.float(), dy, dx)
                valid = node_valid * nv.bool() * boundary.bool()
                distance = float((dx * dx + dy * dy) ** 0.5 * 8.0 * float(config["model"].get("terrain_pixel_size_m", 20.0)))
                for name, tensor in (("signed_dz", (nz - z)), ("edge_slope", (nz - z) / distance), ("relative_height_or_barrier", 0.5 * (barrier + nb)), ("local_relief", 0.5 * (relief + nr)), ("neighbour_distance", torch.full_like(z, distance))):
                    _update(values[name], tensor[valid])
            count += 1
            if args.max_batches and count >= args.max_batches: break
    result = {"config": str(args.config), "split": "train", "batches": count, "features": {}, "mapping": "robust_linear_then_clamp(-4,4); HydroEdgeKAN applies one bounded mapping"}
    for name, data in values.items():
        x = torch.tensor(data, dtype=torch.float64)
        q = torch.quantile(x, torch.tensor([.01, .05, .95, .99], dtype=x.dtype)) if x.numel() else torch.full((4,), float("nan"), dtype=x.dtype)
        median = float(torch.median(x)) if x.numel() else float("nan")
        q05, q95 = float(q[1]), float(q[2]); scale = max((q95 - q05) / 2.0, 1e-6)
        result["features"][name] = {"count": int(x.numel()), "median": median, "iqr": float(torch.quantile(x, torch.tensor(.75, dtype=x.dtype)) - torch.quantile(x, torch.tensor(.25, dtype=x.dtype))) if x.numel() else float("nan"), "p01": float(q[0]), "p05": q05, "p95": q95, "p99": float(q[3]), "recommended_center": median, "recommended_scale": scale}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Build train-only graph-edge calibration statistics for v13/v14 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.flooddepth_dataset import prepare_model_inputs
from models.terrain_graph_kan import DIRECTIONS, _masked_pool, _roll_with_boundary_mask
from tools.evaluate import embed_source_fingerprints
from tools.train import create_dataloaders
from utils.config import load_config
from utils.misc import move_to_device
from utils.registry import build_model


V14_FEATURES = ("absolute_edge_slope", "path_barrier_proxy", "local_relief_pair")
V13_FEATURES = ("signed_dz", "edge_slope", "relative_height_or_barrier", "local_relief", "neighbour_distance")


def _append(store: dict[str, list[float]], name: str, values: torch.Tensor, valid: torch.Tensor) -> None:
    selected = values.detach().float()[valid.detach().bool()]
    selected = selected[torch.isfinite(selected)]
    if selected.numel():
        store[name].extend(selected.cpu().tolist())


def _quantiles(values: list[float]) -> dict[str, object]:
    if not values:
        return {"count": 0}
    total_count = len(values)
    if total_count > 2_000_000:
        stride = int(np.ceil(total_count / 2_000_000))
        values = values[::stride]
    array = np.asarray(values, dtype=np.float64)
    q = np.quantile(array, [.01, .05, .10, .25, .50, .75, .90, .95, .99])
    return {
        "count": total_count, "quantile_sample_count": int(array.size), "p01": float(q[0]), "p05": float(q[1]), "p10": float(q[2]),
        "p25": float(q[3]), "p50": float(q[4]), "p75": float(q[5]), "p90": float(q[6]),
        "p95": float(q[7]), "p99": float(q[8]), "mean": float(array.mean()),
        "std": float(array.std()), "recommended_center": float(q[4]),
        "recommended_scale": max(float((q[7] - q[1]) / 2.0), 1e-6),
        "quantile_breakpoints": [float(value) for value in q],
    }


@torch.no_grad()
def _collect_v14(model, loader, device, max_batches: int, store: dict[str, list[float]]) -> int:
    for batch_index, cpu_batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = move_to_device(cpu_batch, device)
        inputs = prepare_model_inputs(batch)
        _, physical = model.terrain(inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"])
        size = (max(1, inputs["terrain"].shape[-2] // model.graph.graph_scale), max(1, inputs["terrain"].shape[-1] // model.graph.graph_scale))
        _, raw, valid_edge, *_ = model.graph._descriptors(physical, size, torch.maximum(inputs["s1_valid"], inputs["s2_valid"]))
        for index, name in enumerate(V14_FEATURES):
            _append(store, name, raw[:, :, index], valid_edge[:, :, 0])
    return min(len(loader), max_batches) if max_batches else len(loader)


@torch.no_grad()
def _collect_v13(model, loader, device, max_batches: int, store: dict[str, list[float]]) -> int:
    for batch_index, cpu_batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = move_to_device(cpu_batch, device)
        inputs = prepare_model_inputs(batch)
        _, physical = model.terrain(inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"])
        size = (max(1, inputs["terrain"].shape[-2] // model.graph.graph_scale), max(1, inputs["terrain"].shape[-1] // model.graph.graph_scale))
        dem = F.adaptive_avg_pool2d(physical["dem_valid"], size)
        z = _masked_pool(physical["z_hyd"], physical["dem_valid"], size)
        barrier = _masked_pool(physical["z_barrier"], physical["dem_valid"], size)
        relief = _masked_pool(physical["local_relief"], physical["dem_valid"], size)
        node_valid = dem > 0.5
        for dy, dx in DIRECTIONS:
            nz, boundary = _roll_with_boundary_mask(z, dy, dx)
            nb, _ = _roll_with_boundary_mask(barrier, dy, dx)
            nr, _ = _roll_with_boundary_mask(relief, dy, dx)
            nv, _ = _roll_with_boundary_mask(node_valid.float(), dy, dx)
            valid = node_valid * nv.bool() * boundary.bool()
            distance = float((dx * dx + dy * dy) ** 0.5 * model.graph.graph_pixel_size_m)
            for name, tensor in (("signed_dz", nz - z), ("edge_slope", (nz - z) / max(distance, 1e-6)), ("relative_height_or_barrier", 0.5 * (barrier + nb)), ("local_relief", 0.5 * (relief + nr)), ("neighbour_distance", torch.full_like(z, distance))):
                _append(store, name, tensor, valid)
    return min(len(loader), max_batches) if max_batches else len(loader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = embed_source_fingerprints(load_config(args.config))
    config["training"]["num_workers"] = 0; config["training"]["persistent_workers"] = False
    device = torch.device(args.device)
    train_loader, val_loader, _, _ = create_dataloaders(config)
    model = build_model(config).to(device).eval()
    v14 = str(config["model"]["name"]) == "pa_hydrokan_v14"
    names = V14_FEATURES if v14 else V13_FEATURES
    train_values, val_values = {name: [] for name in names}, {name: [] for name in names}
    collector = _collect_v14 if v14 else _collect_v13
    train_batches = collector(model, train_loader, device, args.max_batches, train_values)
    val_batches = collector(model, val_loader, device, args.max_batches, val_values)
    result = {"schema_version": "hydrov14.graph_edge_stats.v1", "config": str(args.config), "train_split": "train", "validation_split": "val", "graph_scale": int(config["model"].get("graph_scale", 8)), "mapping": "fixed train quantile breakpoints are exported; runtime descriptors remain deterministic and never read val/test", "train_batches": train_batches, "val_batches": val_batches, "features": {}}
    for name in names:
        train_summary, val_summary = _quantiles(train_values[name]), _quantiles(val_values[name])
        if train_values[name] and val_values[name]:
            train_array, val_array = np.asarray(train_values[name], dtype=np.float64), np.asarray(val_values[name], dtype=np.float64)
            q = [.05, .50, .95]
            shift = [float(value) for value in (np.quantile(val_array, q) - np.quantile(train_array, q))]
        else:
            shift = [None, None, None]
        result["features"][name] = {"train": train_summary, "val": val_summary, "val_minus_train_p05_p50_p95": shift}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

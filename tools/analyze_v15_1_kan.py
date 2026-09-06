#!/usr/bin/env python3
"""Export compact validation diagnostics for a PA-HydroKAN-S1-V15 checkpoint.

This analyzer never opens the test split.  It enables descriptor emission only
for the requested validation passes, aggregates immediately, and writes compact
JSON/CSV artifacts rather than retaining raster-sized diagnostic tensors.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract, sha256_file
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from tools.evaluate import dataset_fingerprint, embed_source_fingerprints
from utils.amp import resolve_amp
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.logging import write_rows
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model


def _percentiles(values: list[np.ndarray]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    merged = np.concatenate(values)
    return {
        "mean": float(merged.mean()),
        "p05": float(np.quantile(merged, 0.05)),
        "p50": float(np.quantile(merged, 0.50)),
        "p95": float(np.quantile(merged, 0.95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--curve-points", type=int, default=65)
    args = parser.parse_args()
    if args.max_batches <= 0 or args.curve_points < 5:
        raise ValueError("max-batches must be positive and curve-points must be at least five")

    config = embed_source_fingerprints(load_config(args.config))
    if str(config["model"]["name"]) not in {
        "pa_hydrokan_s1_v15_1",
        "pa_hydrokan_s1_v15_2",
    }:
        raise ValueError("analyze_v15_1_kan.py requires a V15.1/V15.2 S1 model")
    if str(config["model"].get("variant")) not in {"kan", "simple"}:
        raise ValueError("KAN diagnostics require the V15 kan or simple variant")
    config["model"]["diagnostics_enabled"] = True
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
        if args.device == "auto"
        else args.device
    )
    contract = DatasetContract.load(config["dataset"]["contract"])
    input_spec = ModelInputSpec.from_config(config)
    dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        "val",
        band_spec=resolve_band_spec(config, contract),
        input_spec=input_spec,
        minimum_event_band_fraction=float(config["dataset"].get("minimum_event_band_fraction", 1.0)),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )
    model = build_model(config).to(device).eval()
    load_checkpoint(
        args.checkpoint,
        model,
        expected_fingerprint=dataset_fingerprint(config),
        map_location=device,
    )
    graph = model.graph
    feature_names = tuple(graph.edge_feature_names)
    occupancy = np.zeros((len(feature_names), int(graph.edge_kan.grid_size)), dtype=np.int64)
    boundary_counts = np.zeros(len(feature_names), dtype=np.int64)
    valid_counts = np.zeros(len(feature_names), dtype=np.int64)
    scalar_values: dict[str, list[float]] = defaultdict(list)
    gate_values: list[np.ndarray] = []
    amp_enabled, amp_dtype, _ = resolve_amp(
        device,
        bool(config["training"].get("amp", False)),
        str(config["training"].get("amp_dtype", "float16")),
    )
    batches = 0
    with torch.no_grad():
        for cpu_batch in loader:
            if batches >= args.max_batches:
                break
            batch = move_to_device(cpu_batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(prepare_model_inputs(batch, input_spec))
            diagnostics = outputs["graph_diagnostics"]
            descriptors = diagnostics["edge_descriptors"].float()
            valid_edges = diagnostics["valid_edges"] > 0.5
            expanded_valid = valid_edges.expand(-1, -1, len(feature_names), -1, -1)
            for index in range(len(feature_names)):
                selected = descriptors[:, :, index][expanded_valid[:, :, index]]
                if selected.numel():
                    values = selected.detach().cpu().numpy()
                    occupancy[index] += np.histogram(
                        values, bins=np.linspace(-1.0, 1.0, graph.edge_kan.grid_size + 1)
                    )[0]
                    boundary_counts[index] += int(np.count_nonzero(np.abs(values) >= 0.98))
                    valid_counts[index] += int(values.size)
            gates = diagnostics["final_graph_gates"]
            gates_valid = gates[(valid_edges.expand_as(gates))]
            if gates_valid.numel():
                gate_values.append(gates_valid.detach().float().cpu().numpy())
            for name, value in diagnostics.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    scalar_values[name].append(float(value.detach().float().cpu()))
            batches += 1

    probe = torch.zeros(args.curve_points, len(feature_names), device=device)
    points = torch.linspace(-1.0, 1.0, args.curve_points, device=device)
    curve_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        base, spline = graph.edge_kan.featurewise_contributions(probe)
        for feature_index, feature_name in enumerate(feature_names):
            probes = probe.clone()
            probes[:, feature_index] = points
            base_feature, spline_feature = graph.edge_kan.featurewise_contributions(probes)
            for head in range(graph.heads):
                for point_index, point in enumerate(points):
                    curve_rows.append(
                        {
                            "feature": feature_name,
                            "head": head,
                            "bounded_input": float(point.cpu()),
                            "base_contribution": float(base_feature[point_index, feature_index, head].cpu()),
                            "spline_contribution": float(spline_feature[point_index, feature_index, head].cpu()),
                            "total_feature_contribution": float(
                                (base_feature[point_index, feature_index, head]
                                 + spline_feature[point_index, feature_index, head]).cpu()
                            ),
                        }
                    )
    occupancy_rows = [
        {
            "feature": name,
            "valid_edge_values": int(valid_counts[index]),
            "boundary_saturation_fraction": float(
                boundary_counts[index] / max(valid_counts[index], 1)
            ),
            "knot_interval_occupancy": occupancy[index].tolist(),
        }
        for index, name in enumerate(feature_names)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "kan_curves.csv", curve_rows)
    write_rows(args.output / "kan_validation_occupancy.csv", occupancy_rows)
    summary = {
        "scope": "validation split only",
        "batches": batches,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "graph_identity": model.graph_identity(),
        "feature_names": list(feature_names),
        "scalar_means": {
            name: float(np.mean(values)) for name, values in scalar_values.items()
        },
        "final_graph_gate_distribution": _percentiles(gate_values),
        "occupancy": occupancy_rows,
        "curves_path": str((args.output / "kan_curves.csv").resolve()),
    }
    atomic_write_json(args.output / "kan_diagnostics.json", summary)
    print({"output": str(args.output), "batches": batches})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

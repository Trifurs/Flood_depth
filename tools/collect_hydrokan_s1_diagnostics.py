#!/usr/bin/env python3
"""Collect finite KAN and S1 encoder diagnostics from one checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.flooddepth_dataset import prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from tools.evaluate import dataset_fingerprint, embed_source_fingerprints
from tools.train import create_dataloaders
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = embed_source_fingerprints(load_config(args.config))
    config["training"]["num_workers"] = 0
    config["training"]["persistent_workers"] = False
    input_spec = ModelInputSpec.from_config(config)
    _, loader, _, _ = create_dataloaders(config)
    batch = move_to_device(next(iter(loader)), torch.device("cpu"))
    model = build_model(config).cpu()
    load_checkpoint(
        args.checkpoint,
        model,
        expected_fingerprint=dataset_fingerprint(config),
        map_location="cpu",
    )
    model.eval()
    with torch.no_grad():
        outputs = model(prepare_model_inputs(batch, input_spec))
    graph = outputs["graph_diagnostics"]
    sar = outputs["sar_diagnostics"]
    finite_scalars = {}
    for name, value in {**graph, **sar}.items():
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            finite_scalars[name] = float(value.detach().float())
    payload = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "input_spec": input_spec.as_dict(),
        "depth_finite": bool(torch.isfinite(outputs["depth"]).all()),
        "uncertainty_finite": bool(torch.isfinite(outputs["uncertainty_scale"]).all()),
        "diagnostics_finite": all(torch.isfinite(torch.tensor(value)) for value in finite_scalars.values()),
        "graph_scalar_diagnostics": finite_scalars,
        "sar_diagnostics": {
            "internal_weight_mean": finite_scalars.get("internal_weight_mean"),
            "external_weight_mean": finite_scalars.get("external_weight_mean"),
            "quality_mean": finite_scalars.get("quality_mean"),
            "angle_film_amplitude_mean": float(torch.stack(sar["angle_film_amplitude"]).mean()),
        },
        "kan": {
            "graph_scale": int(model.graph.graph_scale),
            "heads": int(model.graph.heads),
            "edge_feature_names": list(model.graph.edge_feature_names),
            "coefficient_magnitude": finite_scalars.get("kan_coefficient_magnitude"),
            "coefficient_smoothness": finite_scalars.get("kan_coefficient_smoothness"),
        },
    }
    atomic_write_json(args.output, payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a strict v15 initialization checkpoint from compatible v14 S1 weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.pa_hydrokan_s1_v14 import build_pa_hydrokan_s1_v14
from models.pa_hydrokan_s1_v15 import build_pa_hydrokan_s1_v15
from tools.evaluate import dataset_fingerprint, embed_source_fingerprints
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.misc import atomic_write_json


def _copy_padded(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor | None:
    if destination.ndim != source.ndim or destination.shape[0] != source.shape[0]:
        return None
    if destination.shape[2:] != source.shape[2:] or destination.shape[1] < source.shape[1]:
        return None
    result = torch.zeros_like(destination)
    result[:, : source.shape[1]] = source.to(dtype=result.dtype)
    return result


def transfer(v14_checkpoint: Path, v14_config_path: Path, v15_config_path: Path, output: Path) -> dict[str, Any]:
    v14_config = embed_source_fingerprints(load_config(v14_config_path))
    v15_config = embed_source_fingerprints(load_config(v15_config_path))
    source = build_pa_hydrokan_s1_v14(v14_config)
    target = build_pa_hydrokan_s1_v15(v15_config)
    load_checkpoint(v14_checkpoint, source, map_location="cpu", allow_fingerprint_mismatch=False)
    source_state = source.state_dict()
    target_state = target.state_dict()
    transferred: list[str] = []
    padded: list[str] = []
    for name, value in list(target_state.items()):
        if name in source_state and source_state[name].shape == value.shape:
            target_state[name] = source_state[name].detach().clone()
            transferred.append(name)
    for target_name, source_name in (
        ("sar_encoder.state_mix", "sar_encoder.state"),
        ("sar_encoder.change_mix", "sar_encoder.internal"),
    ):
        target_modules = getattr(target.sar_encoder, target_name.split(".")[-1])
        source_modules = getattr(source.sar_encoder, source_name.split(".")[-1])
        for index in range(len(target_modules)):
            target_prefix = f"{target_name}.{index}.0.weight"
            source_prefix = f"{source_name}.{index}.0.weight"
            padded_weight = _copy_padded(target_state[target_prefix], source_state[source_prefix])
            if padded_weight is not None:
                target_state[target_prefix] = padded_weight
                padded.append(target_prefix)
    target.load_state_dict(target_state, strict=True)
    source_hash = str(v14_checkpoint.resolve())
    save_checkpoint(
        output,
        target,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=0,
        best_metric=float("inf"),
        resolved_config=v15_config,
        dataset_fingerprint=dataset_fingerprint(v15_config),
        extra={
            "initialization": "compatible_s1_v14_to_s1_v15",
            "source_checkpoint": source_hash,
            "transferred_parameter_tensors": len(transferred),
            "padded_parameter_tensors": len(padded),
        },
    )
    summary = {
        "source_checkpoint": source_hash,
        "target_config": str(v15_config_path.resolve()),
        "output": str(output.resolve()),
        "transferred_parameter_tensors": len(transferred),
        "padded_parameter_tensors": len(padded),
        "target_parameter_tensors": len(target_state),
    }
    atomic_write_json(output.with_suffix(".json"), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v14-checkpoint", type=Path, required=True)
    parser.add_argument("--v14-config", type=Path, required=True)
    parser.add_argument("--v15-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(transfer(args.v14_checkpoint, args.v14_config, args.v15_config, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

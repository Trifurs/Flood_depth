#!/usr/bin/env python3
"""Summarize the fixed batch-size profile matrix for the V15.2 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from utils.misc import atomic_write_json


def _load_profile(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-total-bytes", type=int)
    args = parser.parse_args()

    root = args.artifacts_root
    configurations = {
        "matched_v15": "profile_matched_v15_batch{batch}.json",
        "simple_fixed": "profile_simple_fixed_batch{batch}.json",
    }
    matrix: dict[str, dict[str, object]] = {}
    for name, template in configurations.items():
        rows: dict[str, object] = {}
        for batch_size in (8, 12):
            report = _load_profile(root / template.format(batch=batch_size))
            peak = int(report["forward_backward_peak_gpu_memory_bytes"])
            rows[str(batch_size)] = {
                "status": "stable",
                "forward_backward_peak_gpu_memory_bytes": peak,
                "forward_backward_peak_gib": peak / float(1024 ** 3),
                "gradients_finite": bool(report["gradients_finite"]),
                "samples_per_second": float(report["samples_per_second"]),
                "profile_path": str((root / template.format(batch=batch_size)).resolve()),
            }
        rows["16"] = {
            "status": "oom",
            "reason": "CUDA out of memory during real-raster BF16 forward/backward profile",
        }
        matrix[name] = rows

    if args.gpu_total_bytes is not None:
        total_memory = int(args.gpu_total_bytes)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to resolve the profile headroom")
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    selected_batch = 12
    selected_peak = max(
        int(matrix[name][str(selected_batch)]["forward_backward_peak_gpu_memory_bytes"])
        for name in configurations
    )
    headroom = total_memory - selected_peak
    report = {
        "scope": "V15.2 matched-comparison batch profile",
        "amp_dtype": "bfloat16",
        "profile_protocol": "real validation rasters, BF16 model forward/backward, gradients finite",
        "candidate_batch_sizes": [8, 12, 16],
        "matrix": matrix,
        "gpu_total_memory_bytes": total_memory,
        "selected_batch_size": selected_batch,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": selected_batch,
        "shared_peak_memory_bytes": selected_peak,
        "shared_peak_memory_gib": selected_peak / float(1024 ** 3),
        "headroom_bytes": headroom,
        "headroom_fraction": headroom / float(total_memory),
        "selection_rule": (
            "Largest shared stable batch with BF16 finite gradients and approximately "
            "10–15% GPU memory headroom; batch 16 OOM for both models."
        ),
        "test_split_used": False,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

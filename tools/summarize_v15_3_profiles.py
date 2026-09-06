#!/usr/bin/env python3
"""Materialize the V15.3 real-raster batch/profile selection record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.misc import atomic_write_json


GPU_TOTAL_MEMORY_BYTES = 33_635_434_496


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile(path: Path, status: str = "stable") -> dict[str, Any]:
    data = _read(path)
    return {
        "status": status,
        "profile_path": str(path.resolve()),
        "forward_backward_peak_gpu_memory_bytes": int(
            data["forward_backward_peak_gpu_memory_bytes"]
        ),
        "forward_backward_peak_gib": int(
            data["forward_backward_peak_gpu_memory_bytes"]
        )
        / float(1024**3),
        "samples_per_second": float(data["samples_per_second"]),
        "forward_backward_seconds": float(data["forward_backward_seconds"]),
        "gradients_finite": bool(data["gradients_finite"]),
        "parameters": int(data["parameters"]),
        "amp_dtype": str(data["amp_dtype"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/optimization/hydrokan_s1_v15_3"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.artifacts_root
    matrix: dict[str, dict[str, Any]] = {
        "matched_current_best": {},
        "v15_3_stability_graph": {},
    }
    for label, prefix in (
        ("matched_current_best", "profile_matched_current_best"),
        ("v15_3_stability_graph", "profile_v15_3_stability_graph"),
    ):
        for batch in (8, 12):
            matrix[label][str(batch)] = _profile(
                root / f"{prefix}_batch{batch}.json"
            )
        matrix[label]["16"] = {
            "status": "oom",
            "reason": "CUDA out of memory during real-raster BF16 forward/backward profile",
        }
    shared_peak = max(
        matrix["matched_current_best"]["12"][
            "forward_backward_peak_gpu_memory_bytes"
        ],
        matrix["v15_3_stability_graph"]["12"][
            "forward_backward_peak_gpu_memory_bytes"
        ],
    )
    selection = {
        "name": "V15.3 batch profile selection",
        "scope": "real validation rasters; BF16 forward/backward; no test split",
        "candidate_batch_sizes": [8, 12, 16],
        "amp_dtype": "bfloat16",
        "gradient_accumulation_steps": 1,
        "gpu_total_memory_bytes": GPU_TOTAL_MEMORY_BYTES,
        "gpu_total_memory_gib": GPU_TOTAL_MEMORY_BYTES / float(1024**3),
        "selected_batch_size": 12,
        "effective_batch_size": 12,
        "shared_peak_memory_bytes": shared_peak,
        "shared_peak_memory_gib": shared_peak / float(1024**3),
        "headroom_bytes": GPU_TOTAL_MEMORY_BYTES - shared_peak,
        "headroom_fraction": (GPU_TOTAL_MEMORY_BYTES - shared_peak)
        / float(GPU_TOTAL_MEMORY_BYTES),
        "matrix": matrix,
        "selection_rule": (
            "Largest shared stable batch with finite BF16 gradients and approximately "
            "10-15% GPU memory headroom; batch 16 OOM for both routes."
        ),
        "test_split_used": False,
    }
    output = args.output or root / "batch_profile_selection.json"
    atomic_write_json(output, selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

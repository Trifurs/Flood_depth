#!/usr/bin/env python3
"""Audit the V15.2 record and establish the V15.3 current-best baseline.

The audit is validation-only.  It never opens the test split or modifies data,
checkpoints, or historical artifacts.  The script writes the two compact JSON
artifacts required before V15.3 training starts.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _gpu_info() -> dict[str, Any]:
    query = _command([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.used",
        "--format=csv,noheader",
    ])
    devices = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ] if torch.cuda.is_available() else []
    return {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_build": torch.version.cuda,
        "torch_device_count": int(torch.cuda.device_count()),
        "torch_devices": devices,
        "bf16_supported": [
            bool(torch.cuda.is_bf16_supported(index))
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "nvidia_smi": query,
    }


def _metric_summary(block: dict[str, Any]) -> dict[str, Any]:
    return {
        name: block[name]
        for name in (
            "mae", "rmse", "p90_absolute_error", "bias",
            "deep_mae", "deep_bias",
        )
        if name in block
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pytest-result", default="not rerun by audit script")
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    source_root = PROJECT_ROOT / "artifacts/optimization/hydrokan_s1_v15_2"
    decision = _read_json(source_root / "final_decision.json")
    seed_summary = _read_json(source_root / "final_seed_summary.json")
    verification = _read_json(source_root / "verification_summary.json")
    aggregate = seed_summary["raw_aggregate"]
    baseline = aggregate["matched_v15"]
    candidate = aggregate["candidate"]
    if float(baseline["mae"]["mean"]) >= float(candidate["mae"]["mean"]):
        raise RuntimeError("The expected current-best selection is not Matched V15")

    baseline_config = PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_final_matched_v15.xml"
    baseline_root = PROJECT_ROOT / "runs/optimization/hydrokan_s1_v15_2/final_matched_v15"
    seeds = [int(value) for value in seed_summary["seeds"]]
    rows = [
        row for row in seed_summary["rows"]
        if row["variant"] == "matched_v15"
    ]
    current_best = {
        "name": "Matched Current-Best V15",
        "model_name": "pa_hydrokan_s1_v15",
        "selection_metric": "raw pixel_micro_mae mean across three seeds",
        "selection_reason": (
            "V15.2 strict validation record has lower mean raw MAE than the "
            "Graph+KAN+weak-physics candidate; no historical unmatched score was used."
        ),
        "config": str(baseline_config),
        "run_root": str(baseline_root),
        "representative_seed": seeds[0],
        "checkpoints": {
            str(seed): str(baseline_root / f"seed_{seed}" / "best_raw.pth")
            for seed in seeds
        },
        "seeds": seeds,
        "raw_aggregate": baseline,
        "per_seed_raw": [
            {
                "seed": row["seed"],
                "checkpoint_epoch": row["raw_checkpoint_epoch"],
                "metrics": {
                    key.removeprefix("raw_"): row[key]
                    for key in row
                    if key.startswith("raw_")
                    and key not in {"raw_summary_path"}
                },
                "checkpoint": str(baseline_root / f"seed_{row['seed']}" / "best_raw.pth"),
            }
            for row in rows
        ],
        "protocol": verification["protocol"],
        "input_spec": verification["s1_only"]["matched_v15"],
        "test_split_used": False,
        "source_artifacts": {
            "decision": str(source_root / "final_decision.json"),
            "seed_summary": str(source_root / "final_seed_summary.json"),
            "verification": str(source_root / "verification_summary.json"),
        },
    }
    prechange = {
        "scope": "V15.3 pre-change environment and current-best audit",
        "git": {
            "head": _command(["git", "rev-parse", "HEAD"]),
            "working_tree": _command(["git", "status", "--short"]),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
        },
        "gpu": _gpu_info(),
        "pytest": {
            "command": "conda run --no-capture-output -n flood-depth python -m pytest -q",
            "result": args.pytest_result,
        },
        "dataset": {
            "root": "/home/whu/桌面/myData/Flood_depth/subset1000",
            "input_mode": "s1_terrain",
            "test_split_used": False,
            "s2_groups_inactive": True,
            "contract": "/home/whu/桌面/myCode/Flood_depth/artifacts/dataset_audit/subset1000_contract.json",
        },
        "current_best": current_best,
        "historical_context": {
            "v15_2_artifacts": str(source_root),
            "v15_2_report": str(PROJECT_ROOT / "docs/HYDROKAN_S1_V15_2_ENGINEERING_REPORT.md"),
            "historical_unmatched_scores_are_not_used_as_current_best": True,
        },
        "v15_3_targets": {
            "runs_root": str(PROJECT_ROOT / "runs/optimization/hydrokan_s1_v15_3"),
            "artifacts_root": str(output_root),
            "report": str(PROJECT_ROOT / "docs/HYDROKAN_S1_V15_3_ENGINEERING_REPORT.md"),
        },
    }
    (output_root / "current_best_baseline.json").write_text(
        json.dumps(current_best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "prechange_status.json").write_text(
        json.dumps(prechange, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "current_best": current_best["name"],
        "mean_raw_mae": baseline["mae"]["mean"],
        "candidate_mean_raw_mae": candidate["mae"]["mean"],
        "output_root": str(output_root.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

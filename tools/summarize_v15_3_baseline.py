#!/usr/bin/env python3
"""Create the independent V15.3 matched-current-best baseline artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.misc import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_view(summary: dict[str, Any]) -> dict[str, Any]:
    names = (
        "checkpoint_epoch",
        "pixel_micro_mae",
        "pixel_micro_rmse",
        "pixel_micro_p90_absolute_error",
        "pixel_micro_bias",
        "pixel_micro_median_absolute_error",
        "pixel_micro_pixels",
        "event_depth_hierarchical_macro_mae",
        "event_depth_hierarchical_macro_bias",
        "event_hierarchical_composite_mae",
        "evaluation_validity_mask",
        "depth_output_semantics",
        "weights",
    )
    return {name: summary[name] for name in names}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    config = _read(run / "resolved_config.json")
    model = _read(run / "model_summary.json")
    runtime = _read(run / "training_runtime.json")
    raw = _read(run / "eval_raw/summary.json")
    ema = _read(run / "eval_ema/summary.json")
    history = _history(run / "metrics_by_epoch.csv")
    best = min(history, key=lambda row: float(row["val_pixel_micro_mae"]))
    best_epoch = int(best["epoch"])
    if best_epoch != int(raw["checkpoint_epoch"]):
        raise RuntimeError("raw checkpoint epoch does not match the minimum history MAE")
    profile = _read(args.artifacts_root / "profile_matched_current_best_batch12.json")
    batch = _read(args.artifacts_root / "batch_profile_selection.json")
    prechange = _read(args.artifacts_root / "prechange_status.json")
    raw_mae = float(raw["pixel_micro_mae"])
    history_mae = float(best["val_pixel_micro_mae"])
    if abs(raw_mae - history_mae) > 2.0e-6:
        raise RuntimeError("independent raw validation differs materially from history")
    report = {
        "name": "Matched Current-Best V15.3 baseline",
        "scope": "strict V15.3 matched 45-epoch validation baseline; S1-only",
        "test_split_used": False,
        "current_best_before_change": prechange["current_best"],
        "selection": {
            "checkpoint": str((run / "best_raw.pth").resolve()),
            "weights": "raw",
            "selection_metric": "pixel_micro_mae",
            "best_epoch_zero_based": best_epoch,
            "selection_rule": "minimum raw canonical validation MAE within the fixed 45-epoch budget",
        },
        "protocol": {
            "seed": int(config["seed"]),
            "epochs_maximum": int(config["training"]["epochs"]),
            "epochs_minimum": int(config["training"]["minimum_epochs"]),
            "batch_size": int(config["training"]["batch_size"]),
            "gradient_accumulation_steps": int(config["training"]["gradient_accumulation_steps"]),
            "effective_batch_size": int(config["training"]["batch_size"])
            * int(config["training"]["gradient_accumulation_steps"]),
            "amp_enabled": bool(config["training"]["amp"]),
            "amp_dtype": str(config["training"]["amp_dtype"]),
            "optimizer": config["optimizer"],
            "scheduler": config["scheduler"],
            "sampler": config["dataset"]["sampling"],
            "augmentation": config["dataset"]["augmentation"],
            "s1_only": True,
            "test_split_used": False,
        },
        "raw_validation": _metric_view(raw),
        "ema_validation": _metric_view(ema),
        "model": {
            "name": model["name"],
            "parameters": int(model["total_parameters"]),
            "trainable_parameters": int(model["trainable_parameters"]),
            "graph_identity": model["graph_identity"],
        },
        "efficiency": {
            "profile_path": str((args.artifacts_root / "profile_matched_current_best_batch12.json").resolve()),
            "forward_samples_per_second": float(profile["samples_per_second"]),
            "forward_backward_samples_per_second": int(config["training"]["batch_size"])
            / float(profile["forward_backward_seconds"]),
            "profile_forward_backward_peak_gpu_memory_bytes": int(profile["forward_backward_peak_gpu_memory_bytes"]),
            "training_peak_gpu_memory_bytes": int(runtime["peak_gpu_memory_bytes"]),
            "training_elapsed_seconds": float(runtime["elapsed_seconds"]),
            "gpu_total_memory_bytes": int(batch["gpu_total_memory_bytes"]),
            "training_headroom_fraction": (
                int(batch["gpu_total_memory_bytes"]) - int(runtime["peak_gpu_memory_bytes"])
            )
            / float(batch["gpu_total_memory_bytes"]),
        },
        "consistency_checks": {
            "history_best_epoch_zero_based": best_epoch,
            "history_best_pixel_micro_mae": history_mae,
            "independent_raw_recheck_pixel_micro_mae": raw_mae,
            "absolute_difference": abs(raw_mae - history_mae),
            "within_tolerance": abs(raw_mae - history_mae) <= 2.0e-6,
            "raw_beats_ema_on_primary_metric": raw_mae < float(ema["pixel_micro_mae"]),
        },
        "paths": {
            "run_dir": str(run.resolve()),
            "raw_evaluation": str((run / "eval_raw").resolve()),
            "ema_evaluation": str((run / "eval_ema").resolve()),
        },
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

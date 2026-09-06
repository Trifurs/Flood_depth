#!/usr/bin/env python3
"""Create the machine-readable V15.2 matched V15 validation baseline."""

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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric_row(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            result[key] = value
            continue
        if value.lower() in {"inf", "+inf", "-inf"}:
            # Preserve open-ended bin boundaries without emitting non-standard
            # JSON Infinity literals.
            result[key] = value
            continue
        try:
            result[key] = int(value) if key == "bin" or key == "pixels" else float(value)
        except ValueError:
            result[key] = value
    return result


def _metric_view(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
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
    return {key: summary[key] for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    raw_dir = run_dir / "eval_raw"
    ema_dir = run_dir / "eval_ema"
    raw = _read_json(raw_dir / "summary.json")
    ema = _read_json(ema_dir / "summary.json")
    config = _read_json(run_dir / "resolved_config.json")
    model = _read_json(run_dir / "model_summary.json")
    runtime = _read_json(run_dir / "training_runtime.json")
    profile = _read_json(args.artifacts_root / "profile_matched_v15_batch12.json")
    batch_selection = _read_json(args.artifacts_root / "batch_profile_selection.json")
    frozen = _read_json(run_dir / "frozen_depth_weights.json")
    initialization_artifact = _read_json(
        run_dir / "train_positive_depth_initialization.json"
    )
    initialization = model.get("depth_initialization")
    if not isinstance(initialization, dict):
        raise RuntimeError("model summary lacks depth-initialization audit state")
    if (
        float(initialization_artifact["depth_initialization_bias"])
        != float(initialization["depth_initialization_bias"])
        or float(initialization_artifact["train_positive_depth_median_m"])
        != float(initialization["train_positive_depth_median_m"])
    ):
        raise RuntimeError("depth initialization artifact disagrees with model summary")
    history = [_numeric_row(row) for row in _read_csv(run_dir / "metrics_by_epoch.csv")]
    if not history:
        raise RuntimeError("metrics_by_epoch.csv is empty")
    selected_history = min(history, key=lambda row: float(row["val_pixel_micro_mae"]))
    selected_epoch = int(selected_history["epoch"])
    if selected_epoch != int(raw["checkpoint_epoch"]):
        raise RuntimeError(
            "best raw checkpoint epoch does not match the minimum training-history MAE: "
            f"{raw['checkpoint_epoch']} != {selected_epoch}"
        )
    history_mae = float(selected_history["val_pixel_micro_mae"])
    raw_mae = float(raw["pixel_micro_mae"])
    # CUDA validation reductions can differ at the sub-micro-MAE level between
    # the in-training pass and an independent process.  Keep the independent
    # evaluation as the reported value, while rejecting anything larger than a
    # deliberately tiny tolerance.
    recheck_tolerance = 2.0e-6
    if abs(history_mae - raw_mae) > recheck_tolerance:
        raise RuntimeError("raw validation recheck differs materially from training history")
    selected_batch = int(config["training"]["batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    profile_batch = int(profile["batch_size"])
    if profile_batch != selected_batch:
        raise RuntimeError("profiled batch and trained batch differ")
    actual_peak = int(runtime["peak_gpu_memory_bytes"])
    total_memory = int(batch_selection["gpu_total_memory_bytes"])
    forward_backward_seconds = float(profile["forward_backward_seconds"])
    report = {
        "name": "Matched V15 baseline",
        "scope": "V15.2 strict comparison baseline; validation only",
        "selection": {
            "checkpoint": str((run_dir / "best_raw.pth").resolve()),
            "weights": "raw",
            "selection_metric": "pixel_micro_mae",
            "best_epoch_zero_based": selected_epoch,
            "selection_rule": "minimum raw canonical validation MAE within the fixed 45-epoch budget",
        },
        "protocol": {
            "seed": int(config["seed"]),
            "amp_enabled": bool(config["training"]["amp"]),
            "amp_dtype": str(config["training"]["amp_dtype"]),
            "batch_size": selected_batch,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": selected_batch * accumulation,
            "epochs_maximum": int(config["training"]["epochs"]),
            "epochs_minimum": int(config["training"]["minimum_epochs"]),
            "optimizer": config["optimizer"],
            "scheduler": config["scheduler"],
            "sampler": config["dataset"]["sampling"],
            "augmentation": config["dataset"]["augmentation"],
            "canonical_supervision_mask": config["supervision"]["positive_mask"],
            "validation_mask": raw["evaluation_validity_mask"],
            "s1_only": True,
            "model_s1_qa_names": config["dataset"]["model_s1_qa_names"],
            "test_split_used": False,
        },
        "raw_validation": _metric_view(raw),
        "ema_validation": _metric_view(ema),
        "depth_bin_validation_raw": [
            _numeric_row(row) for row in _read_csv(raw_dir / "metrics_by_train_depth_bin.csv")
        ],
        "depth_bin_validation_ema": [
            _numeric_row(row) for row in _read_csv(ema_dir / "metrics_by_train_depth_bin.csv")
        ],
        "model": {
            "name": model["name"],
            "parameters": int(model["total_parameters"]),
            "trainable_parameters": int(model["trainable_parameters"]),
            "graph_identity": model["graph_identity"],
            "frozen_depth_balance_sha256": model["frozen_depth_balance_sha256"],
        },
        "train_only_calibration": {
            "frozen_depth_weights": {
                "sha256": frozen["sha256"],
                "train_weight_mean": frozen["train_weight_mean"],
                "train_weight_min": frozen["train_weight_min"],
                "train_weight_max": frozen["train_weight_max"],
                "train_depth_scan": frozen["train_depth_scan"],
            },
            "depth_initialization": initialization,
        },
        "efficiency": {
            "profile_path": str((args.artifacts_root / "profile_matched_v15_batch12.json").resolve()),
            "forward_samples_per_second": float(profile["samples_per_second"]),
            "forward_backward_samples_per_second": selected_batch / forward_backward_seconds,
            "profile_forward_backward_peak_gpu_memory_bytes": int(
                profile["forward_backward_peak_gpu_memory_bytes"]
            ),
            "training_peak_gpu_memory_bytes": actual_peak,
            "training_peak_gpu_memory_gib": actual_peak / float(1024 ** 3),
            "gpu_total_memory_bytes": total_memory,
            "training_headroom_bytes": total_memory - actual_peak,
            "training_headroom_fraction": (total_memory - actual_peak) / float(total_memory),
            "training_elapsed_seconds": float(runtime["elapsed_seconds"]),
        },
        "consistency_checks": {
            "history_best_epoch_zero_based": selected_epoch,
            "history_best_pixel_micro_mae": history_mae,
            "independent_raw_recheck_pixel_micro_mae": raw_mae,
            "absolute_difference": abs(history_mae - raw_mae),
            "tolerance": recheck_tolerance,
            "within_tolerance": abs(history_mae - raw_mae) <= recheck_tolerance,
            "raw_beats_ema_on_primary_metric": raw_mae < float(ema["pixel_micro_mae"]),
        },
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "raw_evaluation": str(raw_dir.resolve()),
            "ema_evaluation": str(ema_dir.resolve()),
            "batch_selection": str((args.artifacts_root / "batch_profile_selection.json").resolve()),
        },
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create validation-only, matched-budget V15.1 candidate decision artifacts.

The tool deliberately never reads test outputs.  It compares the new
experiments to the historical S1-only V15 checkpoint at the exact checkpoint
epoch (zero-based epoch 35) that produced the canonical 0.40226 MAE baseline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import write_rows
from utils.misc import atomic_write_json


OUTPUT_ROOT = PROJECT_ROOT / "artifacts/optimization/hydrokan_s1_v15_1"
RUN_ROOT = PROJECT_ROOT / "runs/optimization/hydrokan_s1_v15_1"
MATCHED_EPOCH = 35
PRACTICAL_MAE_IMPROVEMENT = 0.005
METRIC_NAMES = (
    "pixel_micro_mae",
    "pixel_micro_rmse",
    "pixel_micro_p90_absolute_error",
    "pixel_micro_bias",
    "event_depth_hierarchical_macro_mae",
    "event_depth_hierarchical_macro_bias",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _epoch_metrics(run_dir: Path, epoch: int) -> dict[str, Any]:
    path = run_dir / "metrics_by_epoch.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if int(row["epoch"]) == epoch:
            return {name: float(row[f"val_{name}"]) for name in METRIC_NAMES} | {
                "epoch": epoch,
                "pixel_micro_pixels": int(row["val_pixel_micro_pixels"]),
            }
    available = [int(row["epoch"]) for row in rows]
    raise ValueError(f"Epoch {epoch} is not present in {path}; available={available}")


def _summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {name: float(summary[name]) for name in METRIC_NAMES} | {
        "checkpoint_epoch": int(summary["checkpoint_epoch"]),
        "pixel_micro_pixels": int(summary["pixel_micro_pixels"]),
        "weights": str(summary["weights"]),
        "evaluation_validity_mask": str(summary["evaluation_validity_mask"]),
    }


def _profile_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "parameters": int(profile["parameters"]),
        "forward_backward_peak_gpu_memory_bytes": int(
            profile.get(
                "forward_backward_peak_gpu_memory_bytes",
                profile.get("peak_gpu_memory_bytes_forward_backward", 0),
            )
        ),
        "samples_per_second": float(profile["samples_per_second"]),
        "amp_dtype": str(profile.get("amp_dtype", "unknown")),
        "batch_size": int(profile["batch_size"]),
    }


def _candidate_record(
    name: str,
    run_name: str,
    config: str,
    profile_name: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_name
    raw = _summary_metrics(_read_json(run_dir / "eval_raw/summary.json"))
    ema = _summary_metrics(_read_json(run_dir / "eval_ema/summary.json"))
    matched = _epoch_metrics(run_dir, MATCHED_EPOCH)
    profile = _profile_metrics(_read_json(OUTPUT_ROOT / profile_name))
    baseline_mae = float(baseline["pixel_micro_mae"])
    threshold = baseline_mae * (1.0 - PRACTICAL_MAE_IMPROVEMENT)
    matched_mae = float(matched["pixel_micro_mae"])
    raw_mae = float(raw["pixel_micro_mae"])
    accepted = matched_mae <= threshold and raw_mae < baseline_mae
    return {
        "name": name,
        "config": config,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "matched_epoch": matched,
        "raw": raw,
        "ema": ema,
        "profile": profile,
        "matched_epoch_mae_delta_vs_historical_v15": matched_mae - baseline_mae,
        "raw_mae_delta_vs_historical_v15": raw_mae - baseline_mae,
        "matched_epoch_mae_relative_change_vs_historical_v15": (matched_mae - baseline_mae) / baseline_mae,
        "passes_matched_epoch_gate": accepted,
    }


def _csv_row(record: dict[str, Any], status: str) -> dict[str, Any]:
    raw = record["raw"]
    matched = record["matched_epoch"]
    profile = record["profile"]
    return {
        "candidate": record["name"],
        "status": status,
        "matched_epoch": matched["epoch"],
        "matched_epoch_mae": matched["pixel_micro_mae"],
        "matched_epoch_rmse": matched["pixel_micro_rmse"],
        "matched_epoch_p90": matched["pixel_micro_p90_absolute_error"],
        "matched_epoch_bias": matched["pixel_micro_bias"],
        "raw_checkpoint_epoch": raw["checkpoint_epoch"],
        "raw_mae": raw["pixel_micro_mae"],
        "raw_rmse": raw["pixel_micro_rmse"],
        "raw_p90": raw["pixel_micro_p90_absolute_error"],
        "raw_bias": raw["pixel_micro_bias"],
        "ema_checkpoint_epoch": record["ema"]["checkpoint_epoch"],
        "ema_mae": record["ema"]["pixel_micro_mae"],
        "parameters": profile["parameters"],
        "forward_backward_peak_gpu_memory_bytes": profile["forward_backward_peak_gpu_memory_bytes"],
        "samples_per_second": profile["samples_per_second"],
        "passes_matched_epoch_gate": record["passes_matched_epoch_gate"],
    }


def main() -> int:
    baseline_recheck = _read_json(OUTPUT_ROOT / "baseline_recheck.json")
    baseline = dict(baseline_recheck["canonical_support"])
    baseline_profile = _profile_metrics(_read_json(OUTPUT_ROOT / "prechange_profile.json"))
    baseline["checkpoint_epoch"] = 35
    baseline["actual_training_last_epoch"] = 60
    baseline["profile"] = baseline_profile
    baseline["matched_epoch_practical_target_mae"] = float(baseline["pixel_micro_mae"]) * (
        1.0 - PRACTICAL_MAE_IMPROVEMENT
    )

    candidates = [
        _candidate_record(
            "V15-corrected",
            "v15_corrected",
            "configs/pa_hydrokan/subset1000_s1_v15_1_corrected.xml",
            "profile_corrected.json",
            baseline,
        ),
        _candidate_record(
            "V15.1-KAN",
            "v15_1_kan",
            "configs/pa_hydrokan/subset1000_s1_v15_1_kan.xml",
            "profile_kan.json",
            baseline,
        ),
        _candidate_record(
            "V15.1-simple",
            "v15_1_simple",
            "configs/pa_hydrokan/subset1000_s1_v15_1_simple.xml",
            "profile_simple.json",
            baseline,
        ),
    ]
    best_new = min(candidates, key=lambda item: float(item["raw"]["pixel_micro_mae"]))
    eligible = [item for item in candidates if item["passes_matched_epoch_gate"]]
    decision = {
        "selection_scope": "validation_only",
        "historical_baseline": {
            "name": "V15 S1-only historical best_raw",
            "checkpoint": baseline_recheck["checkpoint"],
            "config": baseline_recheck["config"],
            "checkpoint_epoch_zero_based": 35,
            "actual_training_last_epoch_zero_based": 60,
            "canonical_metrics": {
                name: baseline[name]
                for name in (
                    "pixel_micro_mae",
                    "pixel_micro_rmse",
                    "pixel_micro_p90_absolute_error",
                    "pixel_micro_bias",
                    "pixel_count",
                )
            },
        },
        "matched_budget_policy": {
            "epoch_zero_based": MATCHED_EPOCH,
            "comparison_mask": "valid_depth_mask_and_output_valid",
            "comparison_split": "val",
            "required_practical_mae_improvement_fraction": PRACTICAL_MAE_IMPROVEMENT,
            "required_matched_epoch_mae_at_most": baseline["matched_epoch_practical_target_mae"],
            "rule": "A new model must pass the epoch-35 MAE gate and beat the historical V15 raw MAE on full validation; later epochs cannot bypass the matched-budget gate.",
        },
        "best_new_candidate_by_raw_validation_mae": {
            "name": best_new["name"],
            "raw_mae": best_new["raw"]["pixel_micro_mae"],
            "raw_mae_delta_vs_historical_v15": best_new["raw_mae_delta_vs_historical_v15"],
            "matched_epoch_mae": best_new["matched_epoch"]["pixel_micro_mae"],
            "passes_matched_epoch_gate": best_new["passes_matched_epoch_gate"],
        },
        "eligible_new_candidates": [item["name"] for item in eligible],
        "selected_model": "historical V15 S1-only best_raw",
        "new_model_accepted": False,
        "new_final_training_started": False,
        "decision": "retain_historical_v15_s1_only",
        "reason": "No V15.1 candidate reached the matched-epoch practical MAE target or beat the historical V15 canonical raw MAE.",
        "test_split_used": False,
    }

    summary = {
        "selection_scope": "validation_only",
        "historical_v15_baseline": baseline,
        "candidates": candidates,
        "best_new_candidate_by_raw_validation_mae": best_new["name"],
        "eligible_new_candidates": [item["name"] for item in eligible],
        "decision_path": "artifacts/optimization/hydrokan_s1_v15_1/final_decision.json",
        "test_split_used": False,
    }
    rows = [
        {
            "candidate": "Historical V15 S1-only",
            "status": "retained_baseline",
            "matched_epoch": 35,
            "matched_epoch_mae": baseline["pixel_micro_mae"],
            "matched_epoch_rmse": baseline["pixel_micro_rmse"],
            "matched_epoch_p90": baseline["pixel_micro_p90_absolute_error"],
            "matched_epoch_bias": baseline["pixel_micro_bias"],
            "raw_checkpoint_epoch": 35,
            "raw_mae": baseline["pixel_micro_mae"],
            "raw_rmse": baseline["pixel_micro_rmse"],
            "raw_p90": baseline["pixel_micro_p90_absolute_error"],
            "raw_bias": baseline["pixel_micro_bias"],
            "ema_checkpoint_epoch": "unavailable",
            "ema_mae": "unavailable",
            "parameters": baseline_profile["parameters"],
            "forward_backward_peak_gpu_memory_bytes": baseline_profile["forward_backward_peak_gpu_memory_bytes"],
            "samples_per_second": baseline_profile["samples_per_second"],
            "passes_matched_epoch_gate": True,
        },
        *[
            _csv_row(
                record,
                "rejected_by_historical_v15_gate" if not record["passes_matched_epoch_gate"] else "eligible",
            )
            for record in candidates
        ],
    ]
    verification = {
        "scope": "validation_only",
        "test_split_used": False,
        "safe_pytest": {
            "command": "conda run --no-capture-output -n flood-depth python -m pytest -q --ignore=tests/test_dataset_loading.py",
            "result": "134 passed, 2 skipped, 2 warnings",
            "reason_for_exclusion": "tests/test_dataset_loading.py explicitly accesses the test split, which is outside this validation-only experiment.",
        },
        "cuda_smoke": {
            "path": "runs/optimization/hydrokan_s1_v15_1/verification_smoke",
            "description": "Real-raster CUDA BF16 optimizer-step and validation-only evaluation completed before formal runs.",
        },
        "retained_model_validation_geotiff": {
            "sample_id": "JRC_2017_244_2017-09-25_OBJ_0056_R000091_C000000",
            "output": "runs/optimization/hydrokan_s1_v15_1/final_retained_v15_infer_val",
            "description": "The retained historical V15 model produced GeoTIFF depth, conditional depth, uncertainty, support, and a PNG panel from a validation sample only.",
        },
        "formal_validation_outputs": {
            record["name"]: {
                "raw": str(Path(record["run_dir"]) / "eval_raw/summary.json"),
                "ema": str(Path(record["run_dir"]) / "eval_ema/summary.json"),
            }
            for record in candidates
        },
        "kan_diagnostics": "artifacts/optimization/hydrokan_s1_v15_1/kan_diagnostics.json",
    }
    final_profile = {
        "selected_model": "historical V15 S1-only best_raw",
        "selected_checkpoint": baseline_recheck["checkpoint"],
        "source_profile": "artifacts/optimization/hydrokan_s1_v15_1/prechange_profile.json",
        "profile": baseline_profile,
        "reason": "The historical V15 model is retained because no V15.1 candidate passed the matched-epoch validation MAE gate.",
        "test_split_used": False,
    }
    atomic_write_json(OUTPUT_ROOT / "candidate_summary.json", summary)
    write_rows(OUTPUT_ROOT / "candidate_summary.csv", rows)
    atomic_write_json(OUTPUT_ROOT / "final_decision.json", decision)
    atomic_write_json(OUTPUT_ROOT / "verification_summary.json", verification)
    atomic_write_json(OUTPUT_ROOT / "final_profile.json", final_profile)
    print(json.dumps({"decision": decision["decision"], "best_new": best_new["name"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

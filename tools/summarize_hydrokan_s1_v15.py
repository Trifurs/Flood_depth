#!/usr/bin/env python3
"""Summarize comparable S1-v15 evaluations without mixing validity masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.misc import atomic_write_json


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(root: Path, evaluation: str, model: str, split: str) -> dict | None:
    path = root / evaluation / "summary.json"
    if not path.is_file():
        return None
    summary = _read(path)
    return {
        "model": model,
        "evaluation": evaluation,
        "split": split,
        "validity_mask": summary.get("evaluation_validity_mask"),
        "depth_output_semantics": summary.get("depth_output_semantics"),
        "event_hierarchical_composite_mae": summary.get("event_hierarchical_composite_mae"),
        "pixel_micro_mae": summary.get("pixel_micro_mae"),
        "pixel_micro_bias": summary.get("pixel_micro_bias"),
        "pixel_micro_pixels": summary.get("pixel_micro_pixels"),
        "checkpoint_epoch": summary.get("checkpoint_epoch"),
        "support_probability_reported": summary.get("support_probability_reported", False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/optimization/hydrokan_s1_v15"),
    )
    args = parser.parse_args()
    root = args.output_root
    specifications = [
        # Keep the short CPU/control run for provenance, but never use it as
        # the primary baseline for the full-budget GPU comparison.
        ("v14_control_val_native_s1_support", "v14_control_short", "val"),
        ("v14_gpu_fair_val_output_valid", "v14_gpu_fair", "val"),
        ("full_transfer_val_output_valid", "v15_full_transfer", "val"),
        ("eventscale_control_val_output_valid", "v15_eventscale", "val"),
        ("gpu_precision_val_output_valid", "v15_gpu_precision", "val"),
        ("v14_control_test_output_valid", "v14_control_short", "test"),
        ("v14_gpu_fair_test_output_valid", "v14_gpu_fair", "test"),
        ("full_transfer_test_output_valid", "v15_full_transfer", "test"),
        ("gpu_precision_test_output_valid", "v15_gpu_precision", "test"),
    ]
    rows = [
        row for evaluation, model, split in specifications
        if (row := _row(root, evaluation, model, split)) is not None
    ]
    baselines = {
        split: next(
            (row for row in rows if row["model"] == "v14_gpu_fair" and row["split"] == split),
            None,
        )
        for split in {row["split"] for row in rows}
    }
    for row in rows:
        baseline = baselines.get(row["split"])
        value = row["event_hierarchical_composite_mae"]
        baseline_value = baseline["event_hierarchical_composite_mae"] if baseline else None
        row["event_metric_relative_improvement_vs_v14"] = (
            (baseline_value - value) / baseline_value
            if value is not None and baseline_value not in (None, 0.0)
            else None
        )
    candidates = [
        row for row in rows
        if not row["model"].startswith("v14_")
        and row["validity_mask"] == "output_valid"
        and row["depth_output_semantics"] == "conditional_positive_v2"
        and row["event_hierarchical_composite_mae"] is not None
    ]
    best_by_split = {
        split: min(
            (row for row in candidates if row["split"] == split),
            key=lambda row: row["event_hierarchical_composite_mae"],
            default=None,
        )
        for split in {row["split"] for row in candidates}
    }
    best_val = best_by_split.get("val")
    best_test = best_by_split.get("test")
    baseline_val = baselines.get("val")
    baseline_test = baselines.get("test")
    decision = {
        "status": "single_seed_gpu_gain_requires_multiseed_confirmation",
        "selection_metric": "event_hierarchical_composite_mae",
        "significant_gain_threshold": 0.05,
        "best_validation_candidate": best_val["evaluation"] if best_val else None,
        "best_test_candidate": best_test["evaluation"] if best_test else None,
        "validation_relative_improvement_vs_v14": best_val.get("event_metric_relative_improvement_vs_v14") if best_val else None,
        "test_relative_improvement_vs_v14": best_test.get("event_metric_relative_improvement_vs_v14") if best_test else None,
        "validation_baseline": baseline_val["event_hierarchical_composite_mae"] if baseline_val else None,
        "test_baseline": baseline_test["event_hierarchical_composite_mae"] if baseline_test else None,
        "interpretation": (
            "The v15 refactor is runnable and optically/S2-free. Against the full-data v14 GPU baseline with the same "
            "batch/AMP/early-stopping budget, the single-seed GPU precision run improves the selected held-out event "
            "metric; multi-seed confirmation is still required for a publication claim."
        ),
    }
    payload = {
        "scope": "S1-only optical-free model; all listed comparisons use output_valid and conditional_positive_v2",
        "executed_candidates": rows,
        "decision": decision,
    }
    atomic_write_json(root / "candidate_summary.json", payload)
    with (root / "candidate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(root / "final_decision.json", decision)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

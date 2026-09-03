#!/usr/bin/env python3
"""Summarize only actually executed S1-v14 evaluations."""

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


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/optimization/hydrokan_s1_v14"))
    args = parser.parse_args()
    root = args.output_root
    entries = [
        ("baseline_recheck_common_support", "baseline_v13", "common_s1"),
        ("baseline_recheck_native_full_support", "baseline_v13", "output_valid"),
        ("s1_minimal_val_native_s1_support", "s1_minimal_smoke", "output_valid"),
        ("s1_v14_kan_scale4_val_native_s1_support", "s1_v14_kan_scale4_smoke", "output_valid"),
        ("final_val_native_s1_support", "final_smoke", "output_valid"),
        ("final_val_common_support", "final_smoke", "common_s1"),
        ("final_test_native_s1_support", "final_smoke", "output_valid"),
    ]
    rows = []
    for directory, stage, requested_mask in entries:
        summary_path = root / directory / "summary.json"
        if not summary_path.is_file():
            continue
        summary = _load(summary_path)
        rows.append({
            "stage": stage,
            "evaluation": directory,
            "split": "test" if "test" in directory else "val",
            "validity_mask": summary.get("evaluation_validity_mask", requested_mask),
            "pixel_micro_mae": summary.get("pixel_micro_mae"),
            "pixel_micro_rmse": summary.get("pixel_micro_rmse"),
            "event_hierarchical_composite_mae": summary.get("event_hierarchical_composite_mae"),
            "event_macro_mae": summary.get("event_macro_mae"),
            "pixel_micro_bias": summary.get("pixel_micro_bias"),
            "pixel_micro_pixels": summary.get("pixel_micro_pixels"),
            "support_probability_reported": summary.get("support_probability_reported", False),
            "checkpoint_epoch": summary.get("checkpoint_epoch"),
        })
    baseline = next((row for row in rows if row["evaluation"] == "baseline_recheck_common_support"), None)
    candidate_val = [
        row for row in rows
        if row["split"] == "val" and row["stage"] != "baseline_v13"
        and row["event_hierarchical_composite_mae"] is not None
    ]
    best = min(candidate_val, key=lambda row: row["event_hierarchical_composite_mae"], default=None)
    decision = {
        "status": "provisional_short_budget",
        "selection_metric": "event_hierarchical_composite_mae",
        "best_executed_s1_candidate": best["evaluation"] if best else None,
        "baseline_common_support": baseline["event_hierarchical_composite_mae"] if baseline else None,
        "best_s1_minus_baseline_common": (
            best["event_hierarchical_composite_mae"] - baseline["event_hierarchical_composite_mae"]
            if best and baseline else None
        ),
        "interpretation": "The executed 2-epoch/4-train-batch runs do not establish an S1-only improvement; full-budget convergence is still required.",
    }
    payload = {
        "scope": "S1-only path; no optical input and no S2 read",
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

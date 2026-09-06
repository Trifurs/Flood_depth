#!/usr/bin/env python3
"""Attribute V15.2 seed instability before changing the V15.3 model.

The report is intentionally based on validation-selected raw checkpoints and
the training history that produced them.  It does not read the test split or
select a new checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.misc import atomic_write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in {None, "", "nan", "NaN"}:
        return default
    result = float(value)
    return result if result == result else default


def _history_at_epoch(history: list[dict[str, str]], epoch: int) -> dict[str, Any]:
    if not history:
        return {}
    selected = min(history, key=lambda row: abs(int(row["epoch"]) - int(epoch)))
    fields = (
        "epoch",
        "val_pixel_micro_mae",
        "train_total",
        "train_physics_effective_weight",
        "train_physics_active_pair_fraction",
        "train_physics_mean_violation_m",
        "train_physics_gradient_norm",
        "train_physics_depth_gradient_cosine_similarity",
        "train_topographic_kan_logit_mean",
        "train_final_graph_gate_mean",
        "train_graph_gamma_mean",
        "train_graph_update_input_rms_ratio",
        "train_spline_base_rms_ratio",
        "train_knot_boundary_saturation_fraction",
        "train_change_gate_mean",
        "train_terrain_residual_input_rms_ratio",
    )
    result: dict[str, Any] = {}
    for key in fields:
        if key == "epoch":
            result[key] = int(selected[key])
        else:
            result[key] = _float(selected, key)
    return result


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    values = [float(row[key]) for row in rows]
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/optimization/hydrokan_s1_v15_2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/optimization/hydrokan_s1_v15_3"),
    )
    args = parser.parse_args()
    summary = _read_json(args.source_root / "final_seed_summary.json")
    rows = list(summary["rows"])
    variants = sorted({str(row["variant"]) for row in rows})
    report: dict[str, Any] = {
        "name": "V15.3 pre-change seed instability attribution",
        "scope": "V15.2 validation-only histories; raw checkpoint is primary",
        "source": str((args.source_root / "final_seed_summary.json").resolve()),
        "test_split_used": False,
        "selection_metric": "raw pixel_micro_mae",
        "variants": {},
    }
    for variant in variants:
        variant_rows = [row for row in rows if str(row["variant"]) == variant]
        seed_rows = []
        for row in sorted(variant_rows, key=lambda item: int(item["seed"])):
            run_dir = Path(str(row["run_dir"]))
            checkpoint_epoch = int(row["raw_checkpoint_epoch"])
            history_path = run_dir / "metrics_by_epoch.csv"
            history = _read_history(history_path)
            seed_rows.append(
                {
                    "seed": int(row["seed"]),
                    "raw_mae": float(row["raw_mae"]),
                    "raw_rmse": float(row["raw_rmse"]),
                    "raw_p90_absolute_error": float(row["raw_p90_absolute_error"]),
                    "raw_bias": float(row["raw_bias"]),
                    "raw_checkpoint_epoch": checkpoint_epoch,
                    "training_epochs_completed": int(row["training_epochs_completed"]),
                    "training_elapsed_seconds": float(row["training_elapsed_seconds"]),
                    "history_at_raw_checkpoint": _history_at_epoch(history, checkpoint_epoch),
                }
            )
        report["variants"][variant] = {
            "seeds": seed_rows,
            "raw_mae": _aggregate(seed_rows, "raw_mae"),
            "raw_rmse": _aggregate(seed_rows, "raw_rmse"),
            "raw_p90_absolute_error": _aggregate(seed_rows, "raw_p90_absolute_error"),
            "raw_bias": _aggregate(seed_rows, "raw_bias"),
        }

    candidate = report["variants"].get("candidate", {})
    candidate_seeds = candidate.get("seeds", [])
    if candidate_seeds:
        bad = max(candidate_seeds, key=lambda row: float(row["raw_mae"]))
        candidate_mean = float(candidate["raw_mae"]["mean"])
        report["bad_seed"] = {
            "variant": "candidate",
            "seed": int(bad["seed"]),
            "reason": "largest raw validation MAE among the matched three candidate seeds",
            "raw_mae": float(bad["raw_mae"]),
            "mean_raw_mae": candidate_mean,
            "excess_over_candidate_mean": float(bad["raw_mae"]) - candidate_mean,
            "history_at_raw_checkpoint": bad["history_at_raw_checkpoint"],
            "attribution_limits": [
                "CSV diagnostics identify correlated training-state symptoms, not a causal proof.",
                "No test split was inspected and no seed was removed from final comparisons.",
            ],
        }
    if "matched_v15" in report["variants"] and "candidate" in report["variants"]:
        report["comparison"] = {
            "candidate_minus_matched_v15_mean_raw_mae": (
                report["variants"]["candidate"]["raw_mae"]["mean"]
                - report["variants"]["matched_v15"]["raw_mae"]["mean"]
            ),
            "candidate_to_matched_v15_std_ratio": (
                report["variants"]["candidate"]["raw_mae"]["sample_std"]
                / max(report["variants"]["matched_v15"]["raw_mae"]["sample_std"], 1.0e-12)
            ),
            "interpretation": (
                "The candidate has both higher mean error and larger seed spread in the matched V15.2 run; "
                "V15.3 therefore prioritizes neutral gates, bounded graph messages, and stable optimization "
                "before expanding the search."
            ),
        }
    output = args.output_root / "bad_seed_diagnostics.json"
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

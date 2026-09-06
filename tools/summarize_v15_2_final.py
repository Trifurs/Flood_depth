#!/usr/bin/env python3
"""Summarize the pre-registered V15.2 final seed comparison.

The script only consumes validation artifacts written by ``tools/evaluate.py``.
It deliberately does not inspect the test split, choose a favourable seed, or
pool bootstrap units across independently trained seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import write_rows
from utils.misc import atomic_write_json


RAW_METRICS: dict[str, str] = {
    "mae": "pixel_micro_mae",
    "rmse": "pixel_micro_rmse",
    "p90_absolute_error": "pixel_micro_p90_absolute_error",
    "bias": "pixel_micro_bias",
}
DEEP_LOWER_BOUNDARY_M = 0.5
VARIANTS = ("candidate", "matched_v15")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required final-comparison artifact is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_deep_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing depth-bin validation artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row for row in rows
        if math.isclose(float(row["lower_train_boundary_m"]), DEEP_LOWER_BOUNDARY_M)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one >= {DEEP_LOWER_BOUNDARY_M:g} m depth bin in {path}; "
            f"found {len(matches)}"
        )
    row = matches[0]
    return {
        "deep_mae": float(row["mae"]),
        "deep_rmse": float(row["rmse"]),
        "deep_p90_absolute_error": float(row["p90_absolute_error"]),
        "deep_bias": float(row["bias"]),
        "deep_pixels": int(row["pixels"]),
    }


def _last_training_epoch(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Missing training history: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No epoch rows in {path}")
    return int(rows[-1]["epoch"])


def _metric_block(summary: Mapping[str, Any], depth: Mapping[str, float]) -> dict[str, float]:
    return {
        **{name: float(summary[key]) for name, key in RAW_METRICS.items()},
        **{name: float(value) for name, value in depth.items() if name != "deep_pixels"},
    }


def _seed_row(variant: str, root: Path, seed: int) -> dict[str, Any]:
    run_dir = root / f"seed_{seed}"
    raw_dir = run_dir / "eval_raw"
    ema_dir = run_dir / "eval_ema"
    raw_summary = _read_json(raw_dir / "summary.json")
    ema_summary = _read_json(ema_dir / "summary.json")
    raw_depth = _read_deep_metrics(raw_dir / "metrics_by_train_depth_bin.csv")
    ema_depth = _read_deep_metrics(ema_dir / "metrics_by_train_depth_bin.csv")
    model_summary = _read_json(run_dir / "model_summary.json")
    runtime = _read_json(run_dir / "training_runtime.json")
    row: dict[str, Any] = {
        "variant": variant,
        "seed": int(seed),
        "run_dir": str(run_dir.resolve()),
        "raw_checkpoint_epoch": int(raw_summary["checkpoint_epoch"]),
        "ema_checkpoint_epoch": int(ema_summary["checkpoint_epoch"]),
        "last_training_epoch": _last_training_epoch(run_dir / "metrics_by_epoch.csv"),
        "training_epochs_completed": _last_training_epoch(run_dir / "metrics_by_epoch.csv") + 1,
        "parameters": int(model_summary["total_parameters"]),
        "training_elapsed_seconds": float(runtime["elapsed_seconds"]),
        "training_peak_gpu_memory_bytes": int(runtime["peak_gpu_memory_bytes"]),
        "raw_summary_path": str((raw_dir / "summary.json").resolve()),
        "ema_summary_path": str((ema_dir / "summary.json").resolve()),
    }
    for prefix, summary, depth in (
        ("raw", raw_summary, raw_depth),
        ("ema", ema_summary, ema_depth),
    ):
        for name, value in _metric_block(summary, depth).items():
            row[f"{prefix}_{name}"] = value
    return row


def _distribution(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("Cannot summarize an empty metric distribution")
    return {
        "mean": float(statistics.fmean(data)),
        "sample_std": float(statistics.stdev(data)) if len(data) > 1 else 0.0,
        "median": float(statistics.median(data)),
        "minimum": float(min(data)),
        "maximum": float(max(data)),
        "n": len(data),
    }


def _aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    metrics = (
        "mae", "rmse", "p90_absolute_error", "bias",
        "deep_mae", "deep_rmse", "deep_p90_absolute_error", "deep_bias",
    )
    return {
        metric: _distribution(row[f"{prefix}_{metric}"] for row in rows)
        for metric in metrics
    }


def _paired_payload(paired_root: Path, seeds: Iterable[int]) -> dict[str, Any]:
    units: dict[str, dict[str, Any]] = {}
    for unit in ("sample", "event"):
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            path = paired_root / f"seed_{seed}" / unit / "paired_bootstrap.json"
            per_seed[str(seed)] = _read_json(path)
        mae_deltas = [
            float(payload["metrics"]["mae"]["observed_delta_candidate_minus_baseline"])
            for payload in per_seed.values()
        ]
        win_rates = [float(payload["candidate_mae_win_rate"]) for payload in per_seed.values()]
        units[unit] = {
            "aggregation_note": (
                "Intervals are computed separately for each independently trained "
                "seed pair and are intentionally not pooled as if seeds were new "
                "validation samples."
            ),
            "mean_observed_delta_mae_candidate_minus_baseline_across_seeds": float(
                statistics.fmean(mae_deltas)
            ),
            "mean_candidate_mae_win_rate_across_seeds": float(statistics.fmean(win_rates)),
            "per_seed": per_seed,
        }
    return {
        "scope": "paired validation analysis; no test split",
        "bootstrap_units": units,
        "test_split_used": False,
    }


def _selection(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    candidate_mae = float(candidate["mae"]["mean"])
    baseline_mae = float(baseline["mae"]["mean"])
    mae_delta = candidate_mae - baseline_mae
    relative = mae_delta / baseline_mae
    candidate_p90 = float(candidate["p90_absolute_error"]["mean"])
    baseline_p90 = float(baseline["p90_absolute_error"]["mean"])
    candidate_deep = float(candidate["deep_mae"]["mean"])
    baseline_deep = float(baseline["deep_mae"]["mean"])
    candidate_bias = float(candidate["bias"]["mean"])
    baseline_bias = float(baseline["bias"]["mean"])
    return {
        "primary_metric": "pixel_micro_mae",
        "candidate_minus_matched_v15_mean_mae": mae_delta,
        "relative_mae_change_candidate_minus_matched_v15": relative,
        "relative_mae_improvement_candidate_vs_matched_v15": -relative,
        "candidate_beats_matched_v15_on_mean_mae": candidate_mae < baseline_mae,
        "preferred_relative_mae_improvement_at_least_0_5pct": -relative >= 0.005,
        "requires_two_additional_seeds_under_0_3pct_rule": abs(relative) < 0.003,
        "candidate_minus_matched_v15_mean_p90": candidate_p90 - baseline_p90,
        "candidate_p90_within_1pct_of_matched_v15": candidate_p90 <= baseline_p90 * 1.01,
        "candidate_minus_matched_v15_mean_deep_mae": candidate_deep - baseline_deep,
        "candidate_minus_matched_v15_mean_bias": candidate_bias - baseline_bias,
        "decision": (
            "accept_candidate_as_accuracy_replacement"
            if candidate_mae < baseline_mae
            else "reject_candidate_as_accuracy_replacement"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--paired-root", type=Path,
        help="Optional root containing seed_<N>/{sample,event}/paired_bootstrap.json.",
    )
    args = parser.parse_args()
    seeds = tuple(int(seed) for seed in args.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Final comparison seeds must be unique")
    if len(seeds) < 3:
        raise ValueError("The final comparison requires at least three seeds")

    rows_by_variant = {
        "candidate": [_seed_row("candidate", args.candidate_root, seed) for seed in seeds],
        "matched_v15": [_seed_row("matched_v15", args.baseline_root, seed) for seed in seeds],
    }
    raw_aggregate = {
        name: _aggregate(rows, "raw") for name, rows in rows_by_variant.items()
    }
    ema_aggregate = {
        name: _aggregate(rows, "ema") for name, rows in rows_by_variant.items()
    }
    decision = _selection(raw_aggregate["candidate"], raw_aggregate["matched_v15"])
    paired = _paired_payload(args.paired_root, seeds) if args.paired_root is not None else None

    args.output_root.mkdir(parents=True, exist_ok=True)
    ordered_rows = [
        row for variant in VARIANTS for row in rows_by_variant[variant]
    ]
    write_rows(args.output_root / "seed_summary.csv", ordered_rows)
    payload = {
        "scope": "strict final validation-only comparison; raw checkpoint is primary",
        "test_split_used": False,
        "sentinel2_groups_requested": [],
        "seeds": list(seeds),
        "rows": ordered_rows,
        "raw_aggregate": raw_aggregate,
        "ema_aggregate": ema_aggregate,
        "selection": decision,
        "paired_bootstrap": paired,
    }
    atomic_write_json(args.output_root / "final_seed_summary.json", payload)
    if paired is not None:
        atomic_write_json(args.output_root / "paired_bootstrap.json", paired)
    atomic_write_json(
        args.output_root / "final_decision.json",
        {
            "scope": payload["scope"],
            "test_split_used": False,
            "seeds": list(seeds),
            "raw_aggregate": raw_aggregate,
            "selection": decision,
            "paired_bootstrap_path": str((args.output_root / "paired_bootstrap.json").resolve())
            if paired is not None else None,
        },
    )
    print(json.dumps({"seeds": list(seeds), "selection": decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

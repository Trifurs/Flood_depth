#!/usr/bin/env python3
"""Paired validation bootstrap at the sample or event unit, never pixel unit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.logging import write_rows
from utils.misc import atomic_write_json


METRICS = ("mae", "rmse", "p90_absolute_error", "bias")


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "source_event_id", "pixels", *METRICS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks required sample-metric columns")
        rows: dict[str, dict[str, Any]] = {}
        for raw in reader:
            sample_id = str(raw["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {path}")
            values = {
                "sample_id": sample_id,
                "source_event_id": str(raw["source_event_id"]),
                "pixels": int(raw["pixels"]),
                **{name: float(raw[name]) for name in METRICS},
            }
            if values["pixels"] <= 0 or not all(
                np.isfinite(values[name]) for name in METRICS
            ):
                raise ValueError(f"invalid finite metric/pixel count for {sample_id!r}")
            rows[sample_id] = values
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _aligned_rows(
    baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    missing_baseline = sorted(set(candidate).difference(baseline))
    missing_candidate = sorted(set(baseline).difference(candidate))
    if missing_baseline or missing_candidate:
        raise ValueError(
            "sample identifiers differ between paired inputs; "
            f"missing_in_baseline={missing_baseline[:5]}, "
            f"missing_in_candidate={missing_candidate[:5]}"
        )
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(baseline):
        left, right = baseline[sample_id], candidate[sample_id]
        if left["source_event_id"] != right["source_event_id"]:
            raise ValueError(f"event mismatch for paired sample {sample_id!r}")
        row = {
            "sample_id": sample_id,
            "source_event_id": left["source_event_id"],
            "baseline_pixels": left["pixels"],
            "candidate_pixels": right["pixels"],
        }
        for metric in METRICS:
            row[f"baseline_{metric}"] = left[metric]
            row[f"candidate_{metric}"] = right[metric]
            row[f"delta_{metric}_candidate_minus_baseline"] = right[metric] - left[metric]
        rows.append(row)
    return rows


def _group_rows(rows: Iterable[dict[str, Any]], unit: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row["sample_id"] if unit == "sample" else row["source_event_id"]
        grouped.setdefault(str(key), []).append(row)
    units: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        baseline_pixels = float(sum(int(item["baseline_pixels"]) for item in members))
        candidate_pixels = float(sum(int(item["candidate_pixels"]) for item in members))
        if baseline_pixels <= 0 or candidate_pixels <= 0:
            raise ValueError(f"nonpositive pixel total for bootstrap unit {key!r}")
        unit_row: dict[str, Any] = {
            "unit_id": key,
            "baseline_pixels": baseline_pixels,
            "candidate_pixels": candidate_pixels,
        }
        for side, pixels in (("baseline", baseline_pixels), ("candidate", candidate_pixels)):
            weights = np.asarray([float(item[f"{side}_pixels"]) for item in members])
            for metric in METRICS:
                values = np.asarray([float(item[f"{side}_{metric}"]) for item in members])
                if metric == "rmse":
                    value = float(np.sqrt(np.sum(weights * np.square(values)) / pixels))
                else:
                    # P90 is intentionally a pixel-weighted *sample/event-level*
                    # summary: the CSV does not retain individual residuals, so
                    # this never mislabels a reconstructed global pixel P90.
                    value = float(np.sum(weights * values) / pixels)
                unit_row[f"{side}_{metric}"] = value
        units.append(unit_row)
    return units


def _resampled_deltas(
    units: list[dict[str, Any]], draws: int, seed: int
) -> tuple[dict[str, float], dict[str, np.ndarray], float]:
    count = len(units)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(draws, count))
    baseline_pixels = np.asarray([item["baseline_pixels"] for item in units])
    candidate_pixels = np.asarray([item["candidate_pixels"] for item in units])
    observed: dict[str, float] = {}
    deltas: dict[str, np.ndarray] = {}
    for metric in METRICS:
        baseline = np.asarray([item[f"baseline_{metric}"] for item in units])
        candidate = np.asarray([item[f"candidate_{metric}"] for item in units])
        if metric == "rmse":
            observed_baseline = np.sqrt(
                np.sum(baseline_pixels * np.square(baseline)) / baseline_pixels.sum()
            )
            observed_candidate = np.sqrt(
                np.sum(candidate_pixels * np.square(candidate)) / candidate_pixels.sum()
            )
            bootstrap_baseline = np.sqrt(
                (baseline_pixels[indices] * np.square(baseline[indices])).sum(axis=1)
                / baseline_pixels[indices].sum(axis=1)
            )
            bootstrap_candidate = np.sqrt(
                (candidate_pixels[indices] * np.square(candidate[indices])).sum(axis=1)
                / candidate_pixels[indices].sum(axis=1)
            )
        else:
            observed_baseline = np.sum(baseline_pixels * baseline) / baseline_pixels.sum()
            observed_candidate = np.sum(candidate_pixels * candidate) / candidate_pixels.sum()
            bootstrap_baseline = (
                baseline_pixels[indices] * baseline[indices]
            ).sum(axis=1) / baseline_pixels[indices].sum(axis=1)
            bootstrap_candidate = (
                candidate_pixels[indices] * candidate[indices]
            ).sum(axis=1) / candidate_pixels[indices].sum(axis=1)
        observed[f"delta_{metric}"] = float(observed_candidate - observed_baseline)
        deltas[metric] = bootstrap_candidate - bootstrap_baseline
    candidate_win_rate = float(
        np.mean(
            np.asarray([item["candidate_mae"] for item in units])
            < np.asarray([item["baseline_mae"] for item in units])
        )
    )
    return observed, deltas, candidate_win_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit", choices=("sample", "event"), default="sample")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError("paired bootstrap requires at least 100 draws")

    rows = _aligned_rows(_read_rows(args.baseline), _read_rows(args.candidate))
    units = _group_rows(rows, args.unit)
    observed, deltas, win_rate = _resampled_deltas(units, args.draws, args.seed)
    metrics: dict[str, dict[str, float]] = {}
    for metric, samples in deltas.items():
        metrics[metric] = {
            "observed_delta_candidate_minus_baseline": observed[f"delta_{metric}"],
            "bootstrap_mean_delta_candidate_minus_baseline": float(samples.mean()),
            "ci95_lower": float(np.quantile(samples, 0.025)),
            "ci95_upper": float(np.quantile(samples, 0.975)),
        }
    payload = {
        "scope": "paired validation analysis; no test split",
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "bootstrap_unit": args.unit,
        "bootstrap_draws": args.draws,
        "seed": args.seed,
        "aligned_samples": len(rows),
        "bootstrap_units": len(units),
        "p90_definition": "pixel-weighted mean of per-unit P90 values; not reconstructed global pixel P90",
        "candidate_mae_win_rate": win_rate,
        "metrics": metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "paired_validation_units.csv", rows)
    atomic_write_json(args.output / "paired_bootstrap.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Describe train/val/test depth composition without using test for model selection."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.preprocessing import resolve_depth_stratification_bins
from utils.config import load_config
from utils.misc import atomic_write_json


QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 1.0)


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    result: dict[str, float | int] = {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }
    for quantile, value in zip(QUANTILES, np.quantile(finite, QUANTILES)):
        label = f"q{100 * quantile:g}".replace(".", "p")
        result[label] = float(value)
    return result


def _depth_bins(values: np.ndarray, edges: Sequence[float]) -> list[dict[str, Any]]:
    boundaries = sorted(set(float(value) for value in edges))[1:-1]
    indices = np.digitize(values, boundaries, right=True)
    total = max(1, int(values.size))
    rows: list[dict[str, Any]] = []
    for index in range(len(boundaries) + 1):
        count = int(np.count_nonzero(indices == index))
        rows.append(
            {
                "bin": index,
                "lower_m": float("-inf") if index == 0 else boundaries[index - 1],
                "upper_m": float("inf") if index == len(boundaries) else boundaries[index],
                "pixels": count,
                "fraction": float(count / total),
            }
        )
    return rows


def summarize_split(
    dataset: FloodDepthDataset, depth_edges: Sequence[float]
) -> dict[str, Any]:
    pooled: list[np.ndarray] = []
    sample_means: list[float] = []
    sample_p90: list[float] = []
    sample_maxima: list[float] = []
    positive_fractions: list[float] = []
    day_differences: list[float] = []
    origins: Counter[str] = Counter()
    years: Counter[str] = Counter()

    for sample in dataset:
        valid = sample["masks"]["valid_depth_mask"].numpy() > 0.5
        values = sample["label"].numpy()[valid].astype(np.float32, copy=False)
        if values.size:
            pooled.append(values)
            sample_means.append(float(values.mean()))
            sample_p90.append(float(np.quantile(values, 0.9)))
            sample_maxima.append(float(values.max()))
            day = sample["reliability"][9:10].numpy()
            day_differences.append(float(day[valid].mean()))
        positive_fractions.append(float(valid.mean()))
        metadata = sample["metadata"]
        origins[str(metadata["sample_origin"])] += 1
        years[str(metadata["event_start"])[:4]] += 1

    values = np.concatenate(pooled) if pooled else np.empty(0, dtype=np.float32)
    return {
        "split": dataset.split,
        "samples": len(dataset),
        "source_events": len(set(dataset.event_ids)),
        "sample_origins": dict(sorted(origins.items())),
        "event_start_years": dict(sorted(years.items())),
        "positive_pixel_fraction_sample_macro": _distribution(
            np.asarray(positive_fractions)
        ),
        "positive_depth_m_pixel": _distribution(values),
        "positive_depth_m_sample_mean": _distribution(np.asarray(sample_means)),
        "positive_depth_m_sample_p90": _distribution(np.asarray(sample_p90)),
        "positive_depth_m_sample_maximum": _distribution(np.asarray(sample_maxima)),
        "sensor_day_difference_positive_sample_macro": _distribution(
            np.asarray(day_differences)
        ),
        "frozen_train_depth_bins": _depth_bins(values, depth_edges),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/split_depth_shift_with_test.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    datasets = {
        split: FloodDepthDataset(
            config["dataset"]["contract"],
            config["dataset"]["train_stats"],
            split,
            transform=None,
        )
        for split in ("train", "val", "test")
    }
    depth_edges = resolve_depth_stratification_bins(
        config["loss"], datasets["train"].normalizer
    )
    payload = {
        "scope": (
            "descriptive benchmark audit; test labels are reported only after the "
            "Hydro-v6 checkpoint was frozen and are not used for candidate selection"
        ),
        "depth_edges_m": depth_edges,
        "splits": {
            split: summarize_split(dataset, depth_edges)
            for split, dataset in datasets.items()
        },
    }
    atomic_write_json(args.output, payload)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

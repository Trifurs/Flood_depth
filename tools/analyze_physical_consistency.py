#!/usr/bin/env python3
"""Audit train/val targets before choosing a weak physical depth constraint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from datasets.flooddepth_dataset import FloodDepthDataset
from models.terrain_features import masked_average
from utils.config import load_config
from utils.misc import atomic_write_json


CROSS = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)
LAPLACIAN = torch.tensor(
    [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


def _distribution(values: list[np.ndarray]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    merged = np.concatenate(values).astype(np.float64, copy=False)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return {"count": 0}
    quantiles = np.quantile(merged, [0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "count": int(merged.size),
        "mean": float(merged.mean()),
        "median": float(quantiles[0]),
        "p75": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "maximum": float(merged.max()),
    }


def _macro(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def _terrain_order_summary(
    terrain_steps: list[np.ndarray], depth_steps: list[np.ndarray]
) -> dict[str, Any]:
    if not terrain_steps:
        return {"count": 0}
    terrain = np.concatenate(terrain_steps).astype(np.float64, copy=False)
    depth = np.concatenate(depth_steps).astype(np.float64, copy=False)
    finite = np.isfinite(terrain) & np.isfinite(depth)
    terrain, depth = terrain[finite], depth[finite]
    result: dict[str, Any] = {}
    for minimum_step in (0.02, 0.05, 0.10, 0.20):
        selected = (np.abs(terrain) >= minimum_step) & (np.abs(terrain) <= 0.75)
        count = int(np.count_nonzero(selected))
        key = f"terrain_abs_{minimum_step:.2f}_to_0.75m"
        if count == 0:
            result[key] = {"count": 0}
            continue
        signed_depth = np.sign(terrain[selected]) * depth[selected]
        absolute_terrain = np.abs(terrain[selected])
        ratio = -signed_depth / np.maximum(absolute_terrain, 1e-6)
        result[key] = {
            "count": count,
            "opposite_or_flat_fraction": float(np.mean(signed_depth <= 0.0)),
            "at_least_quarter_compensation_fraction": float(
                np.mean(signed_depth <= -0.25 * absolute_terrain)
            ),
            "at_least_half_compensation_fraction": float(
                np.mean(signed_depth <= -0.50 * absolute_terrain)
            ),
            "compensation_ratio_median": float(np.median(ratio)),
            "compensation_ratio_p25": float(np.quantile(ratio, 0.25)),
            "compensation_ratio_p75": float(np.quantile(ratio, 0.75)),
        }
    return result


def analyze_split(dataset: FloodDepthDataset, pixel_size_m: float = 20.0) -> dict[str, Any]:
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    curvature_values: list[np.ndarray] = []
    depth_pair_values: list[np.ndarray] = []
    terrain_pair_values: list[np.ndarray] = []
    signed_depth_steps: list[np.ndarray] = []
    signed_terrain_steps: list[np.ndarray] = []
    wse_pair_values: list[np.ndarray] = []
    wse_slope_values: list[np.ndarray] = []
    wse_slope_by_relief: dict[str, list[np.ndarray]] = {"low": [], "medium": [], "high": []}
    time_weight_values: list[np.ndarray] = []
    curvature_macro: list[float] = []
    high_relief_depth_macro: list[float] = []
    depth_terrain_spearman: list[float] = []
    positive_pixels = 0
    interior_pixels = 0
    neighbour_pairs = 0

    for sample in dataset:
        target = sample["label"].unsqueeze(0).float()
        positive = (sample["masks"]["valid_depth_mask"].unsqueeze(0) > 0.5)
        dem_valid = (sample["validity"]["dem_valid"].unsqueeze(0) > 0.5)
        valid = positive & dem_valid
        raw = sample["terrain_raw"].unsqueeze(0).float()
        elevation = raw[:, 0:1]
        dem_float = dem_valid.to(torch.float32)
        z_hyd = masked_average(elevation, dem_float, kernel_size=9)
        local_second = masked_average(elevation.square(), dem_float, kernel_size=9)
        local_relief = (local_second - z_hyd.square()).clamp_min(0.0).sqrt()
        wse = z_hyd + target

        neighbourhood = F.conv2d(valid.to(torch.float32), CROSS, padding=1)
        interior = valid & (neighbourhood >= 4.999)
        curvature = F.conv2d(wse, LAPLACIAN, padding=1).abs()
        selected_curvature = curvature[interior].numpy()
        if selected_curvature.size:
            curvature_values.append(selected_curvature)
            curvature_macro.append(float(selected_curvature.mean()))

        time_difference = sample["reliability"][9:10].unsqueeze(0).float()
        selected_time_weight = torch.exp(
            -time_difference.clamp_min(0.0) / 0.25
        )[interior].numpy()
        if selected_time_weight.size:
            time_weight_values.append(selected_time_weight)

        relief_selected = valid & (local_relief >= 1.0)
        relief_differences: list[np.ndarray] = []
        for dim in (-1, -2):
            if dim == -1:
                pair = relief_selected[..., :, 1:] & relief_selected[..., :, :-1]
                difference = (target[..., :, 1:] - target[..., :, :-1]).abs()
            else:
                pair = relief_selected[..., 1:, :] & relief_selected[..., :-1, :]
                difference = (target[..., 1:, :] - target[..., :-1, :]).abs()
            selected = difference[pair].numpy()
            if selected.size:
                relief_differences.append(selected)
        if relief_differences:
            high_relief_depth_macro.append(
                float(np.concatenate(relief_differences).mean())
            )

        for dy, dx, distance_m in ((0, 1, pixel_size_m), (1, 0, pixel_size_m), (1, 1, pixel_size_m * np.sqrt(2.0)), (1, -1, pixel_size_m * np.sqrt(2.0))):
            if dy == 0 and dx == 1:
                left = (..., slice(None), slice(None, -1))
                right = (..., slice(None), slice(1, None))
            elif dy == 1 and dx == 0:
                left = (..., slice(None, -1), slice(None))
                right = (..., slice(1, None), slice(None))
            elif dx == 1:
                left = (..., slice(None, -1), slice(None, -1))
                right = (..., slice(1, None), slice(1, None))
            else:
                left = (..., slice(None, -1), slice(1, None))
                right = (..., slice(1, None), slice(None, -1))
            pair = valid[left] & valid[right]
            depth_step = target[right] - target[left]
            terrain_step = z_hyd[right] - z_hyd[left]
            depth_difference = depth_step.abs()
            terrain_difference = terrain_step.abs()
            wse_difference = (wse[right] - wse[left]).abs()
            count = int(pair.sum().item())
            neighbour_pairs += count
            if count:
                depth_pair_values.append(depth_difference[pair].numpy())
                terrain_pair_values.append(terrain_difference[pair].numpy())
                wse_pair_values.append(wse_difference[pair].numpy())
                signed_depth_steps.append(depth_step[pair].numpy())
                signed_terrain_steps.append(terrain_step[pair].numpy())
                slope = (wse_difference[pair] / float(distance_m)).numpy()
                wse_slope_values.append(slope)
                relief_pair = 0.5 * (local_relief[left] + local_relief[right])
                relief = relief_pair[pair].numpy()
                for name, selection in (
                    ("low", relief < 1.0),
                    ("medium", (relief >= 1.0) & (relief < 12.0)),
                    ("high", relief >= 12.0),
                ):
                    if np.any(selection):
                        wse_slope_by_relief[name].append(slope[selection])

        depth_np = target[valid].numpy()
        terrain_np = z_hyd[valid].numpy()
        if depth_np.size > 1 and np.ptp(depth_np) > 0 and np.ptp(terrain_np) > 0:
            coefficient = float(spearmanr(depth_np, terrain_np).statistic)
            if np.isfinite(coefficient):
                depth_terrain_spearman.append(coefficient)
        positive_pixels += int(valid.sum().item())
        interior_pixels += int(interior.sum().item())

    return {
        "split": dataset.split,
        "samples": len(dataset),
        "positive_dem_valid_pixels": positive_pixels,
        "five_pixel_interior_pixels": interior_pixels,
        "five_pixel_interior_fraction": (
            float(interior_pixels / positive_pixels) if positive_pixels else 0.0
        ),
        "valid_four_neighbour_pairs": neighbour_pairs,
        "target_wse_laplacian_pixel": _distribution(curvature_values),
        "target_wse_laplacian_sample_macro": _macro(curvature_macro),
        "target_depth_four_neighbour_difference_m": _distribution(depth_pair_values),
        "smoothed_dsm_four_neighbour_difference_m": _distribution(terrain_pair_values),
        "target_wse_four_neighbour_difference_m": _distribution(wse_pair_values),
        "reference_wse_slope_m_per_m": _distribution(wse_slope_values),
        "reference_wse_slope_by_relief": {
            name: _distribution(values) for name, values in wse_slope_by_relief.items()
        },
        "target_depth_terrain_order": _terrain_order_summary(
            signed_terrain_steps, signed_depth_steps
        ),
        "target_high_relief_depth_continuity_sample_macro": _macro(
            high_relief_depth_macro
        ),
        "target_depth_vs_smoothed_dsm_spearman_sample_macro": _macro(
            depth_terrain_spearman
        ),
        "current_wse_time_weight_on_interior": _distribution(time_weight_values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val"), default=("train", "val")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/physical_target_consistency.json"),
    )
    parser.add_argument("--pixel-size-m", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = {
        "scope": "target-only train/val diagnostic; test is intentionally unavailable",
        "z_hyd_definition": "9x9 mask-aware DSM mean",
        "splits": {},
    }
    for split in args.splits:
        dataset = FloodDepthDataset(
            config["dataset"]["contract"],
            config["dataset"]["train_stats"],
            split,
            transform=None,
        )
        report["splits"][split] = analyze_split(dataset, args.pixel_size_m)
    atomic_write_json(args.output, report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

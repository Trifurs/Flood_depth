#!/usr/bin/env python3
"""Build robust normalization and label diagnostics from train pixels only."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio

from datasets.contract import DatasetContract, MODEL_CONTINUOUS_GROUPS, sha256_file
from utils.config import load_config
from utils.misc import atomic_write_json


def _load_train_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    if not rows:
        raise RuntimeError("Manifest has no train rows")
    return rows


def _valid_band(path: Path, band_index: int) -> np.ndarray:
    with rasterio.open(path) as dataset:
        array = dataset.read(band_index + 1, masked=False)
        mask = dataset.read_masks(band_index + 1) > 0
        mask &= np.isfinite(array)
        if dataset.nodata is not None:
            if np.isnan(dataset.nodata):
                mask &= ~np.isnan(array)
            else:
                mask &= array != dataset.nodata
    return array[mask].astype(np.float64, copy=False)


def _summarize(values: np.ndarray, band: str, transform: str = "identity") -> dict[str, object]:
    if values.size == 0:
        raise RuntimeError(f"No valid train values for {band}")
    finite = values[np.isfinite(values)]
    if finite.size != values.size:
        raise RuntimeError(f"Non-finite train values reached statistics for {band}")
    quantiles = np.quantile(finite, [0.005, 0.01, 0.99, 0.995])
    std = float(np.std(finite, dtype=np.float64))
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError(f"Invalid train standard deviation for {band}: {std}")
    return {
        "band": band,
        "transform": transform,
        "count": int(finite.size),
        "mean": float(np.mean(finite, dtype=np.float64)),
        "std": std,
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "p0.5": float(quantiles[0]),
        "p1": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "p99.5": float(quantiles[3]),
    }


def _collect_group_band(
    rows: list[dict[str, str]],
    root: Path,
    path_column: str,
    band_index: int,
    transform: Callable[[np.ndarray, dict[str, str]], np.ndarray] | None = None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for row in rows:
        values = _valid_band(root / row[path_column], band_index)
        if transform is not None:
            values = transform(values, row)
        if values.size:
            chunks.append(values)
    if not chunks:
        return np.empty(0, dtype=np.float64)
    # Memory is bounded to one band (about 6.9M values here); rasters are never cached.
    return np.concatenate(chunks)


def _duration_days(row: dict[str, str]) -> int:
    start = date.fromisoformat(row["event_start"])
    end = date.fromisoformat(row["event_end"])
    return max(1, (end - start).days + 1)


def _log_count(values: np.ndarray, _: dict[str, str]) -> np.ndarray:
    return np.log1p(np.clip(values, 0.0, None))


def _normalized_day(values: np.ndarray, row: dict[str, str]) -> np.ndarray:
    present = values >= 0.0
    return np.clip(values[present] / float(_duration_days(row)), 0.0, 1.0)


def _positive_prior_and_depths(
    rows: list[dict[str, str]], contract: DatasetContract
) -> tuple[dict[str, object], np.ndarray]:
    root = contract.dataset_root
    mask_group = contract.group("masks")
    indices = {
        name: contract.band_index("masks", name)
        for name in (
            "valid_depth_mask",
            "permanent_water_mask",
            "extreme_high_mask",
            "DEM_valid_mask",
            "slope_valid_mask",
            "S1_event_composite_valid_mask",
            "S2_event_composite_valid_mask",
        )
    }
    positive_pixels = 0
    eligible_pixels = 0
    depth_chunks: list[np.ndarray] = []
    for row in rows:
        with rasterio.open(root / row[mask_group["path_column"]]) as dataset:
            masks = dataset.read(masked=False) > 0
        with rasterio.open(root / row[contract.group("terrain")["path_column"]]) as dataset:
            terrain_raster_valid = np.logical_and.reduce(dataset.read_masks() > 0, axis=0)
        positive = masks[indices["valid_depth_mask"]]
        eligible = (
            masks[indices["DEM_valid_mask"]]
            & masks[indices["slope_valid_mask"]]
            & terrain_raster_valid
            & (masks[indices["S1_event_composite_valid_mask"]] | masks[indices["S2_event_composite_valid_mask"]])
            & ~masks[indices["permanent_water_mask"]]
            & ~masks[indices["extreme_high_mask"]]
        )
        positive_pixels += int(np.count_nonzero(positive & eligible))
        eligible_pixels += int(np.count_nonzero(eligible))
        label_path = root / row[contract.group("label")["path_column"]]
        with rasterio.open(label_path) as dataset:
            label = dataset.read(1, masked=False)
            label_valid = dataset.read_masks(1) > 0
            if dataset.nodata is not None:
                label_valid &= label != dataset.nodata
            label_valid &= np.isfinite(label) & positive
        if np.any(label_valid):
            depth_chunks.append(label[label_valid].astype(np.float64, copy=False))
    if eligible_pixels == 0 or not depth_chunks:
        raise RuntimeError("Train split has no eligible output pixels or positive depth pixels")
    raw_fraction = positive_pixels / eligible_pixels
    conservative = float(np.clip(raw_fraction, 0.01, 0.5))
    return (
        {
            "mode": "auto",
            "method": "observed train valid-depth fraction among eligible output pixels; conservative proxy, not a true flood prior",
            "positive_pixels": positive_pixels,
            "eligible_pixels": eligible_pixels,
            "raw_fraction": raw_fraction,
            "clip_bounds": [0.01, 0.5],
            "value": conservative,
        },
        np.concatenate(depth_chunks),
    )


def build_stats(config_path: Path, output_override: Path | None = None) -> Path:
    config = load_config(config_path)
    dataset_config = config["dataset"]
    contract = DatasetContract.load(dataset_config["contract"])
    contract.verify_fingerprints(include_normalization=False)
    root = contract.dataset_root
    manifest = contract.manifest_path
    rows = _load_train_rows(manifest)
    expected_train = int(contract.payload["sample_counts"]["train"])
    if len(rows) != expected_train:
        raise RuntimeError(f"Train row count changed: contract={expected_train}, manifest={len(rows)}")

    groups: dict[str, list[dict[str, object]]] = {}
    for group in MODEL_CONTINUOUS_GROUPS:
        group_contract = contract.group(group)
        entries: list[dict[str, object]] = []
        for band_index, description in enumerate(group_contract["band_descriptions"]):
            values = _collect_group_band(
                rows, root, str(group_contract["path_column"]), band_index
            )
            entries.append(_summarize(values, str(description)))
        groups[group] = entries
        print(f"computed {group}: {len(entries)} bands", flush=True)

    qa_groups: dict[str, list[dict[str, object]]] = {}
    for group, specifications in {
        "s1_qa": [
            ("event_observation_count", _log_count, "log1p(max(x,0))"),
            ("selected_event_day_offset", _normalized_day, "clip(x/event_duration_days,0,1); x<0 is missing"),
        ],
        "s2_qa": [
            ("pre_clear_observation_count", _log_count, "log1p(max(x,0))"),
            ("event_clear_observation_count", _log_count, "log1p(max(x,0))"),
            ("selected_event_day_offset", _normalized_day, "clip(x/event_duration_days,0,1); x<0 is missing"),
        ],
    }.items():
        entries = []
        group_contract = contract.group(group)
        for description, transform, transform_name in specifications:
            band_index = contract.band_index(group, description)
            values = _collect_group_band(
                rows,
                root,
                str(group_contract["path_column"]),
                band_index,
                transform,
            )
            entries.append(_summarize(values, description, transform_name))
        qa_groups[group] = entries
        print(f"computed {group}: {len(entries)} safe QA features", flush=True)

    prior, depths = _positive_prior_and_depths(rows, contract)
    depth_quantiles = np.quantile(depths, [0.0, 0.25, 0.5, 0.75, 1.0])
    depth_bins = sorted({float(value) for value in depth_quantiles})
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "exact per-band train-only scan with valid raster masks; at most one band retained in memory",
        "scope": "train split only; valid pixels only; val/test excluded",
        "dataset_root": str(root),
        "train_samples": len(rows),
        "manifest_sha256": sha256_file(manifest),
        "source_contract_sha256_before_stats_binding": contract.hash,
        "groups": groups,
        "qa_groups": qa_groups,
        "positive_prior": prior,
        "train_depth": {
            **_summarize(depths, "depth_m"),
            "stratification_quantiles": [float(value) for value in depth_quantiles],
            "stratification_bin_edges": depth_bins,
        },
        "notes": [
            "The prior is only an observed-label fraction proxy and is not claimed to be the true flood prevalence.",
            "selected_relative_orbit, orbit pass, and selected_pre_observation_count are excluded.",
            "No validation or test pixel contributes to any statistic or bin boundary.",
        ],
    }
    output = output_override or Path(dataset_config["train_stats"])
    output = output.expanduser().resolve()
    atomic_write_json(output, payload)
    stats_hash = sha256_file(output)

    contract_payload = dict(contract.payload)
    normalization = dict(contract_payload["normalization"])
    normalization["selected"] = {
        "path": str(output),
        "sha256": stats_hash,
        "scope": payload["scope"],
        "manifest_sha256": payload["manifest_sha256"],
    }
    contract_payload["normalization"] = normalization
    atomic_write_json(contract.path, contract_payload)
    print(f"wrote {output} sha256={stats_hash}")
    print(f"bound stats fingerprint into {contract.path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pa_hydrokan/subset150_main.xml")
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_stats(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

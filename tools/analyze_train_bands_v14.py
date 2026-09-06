#!/usr/bin/env python3
"""Train-only, raster-stratified band statistics for Hydro-v14.

This diagnostic never opens validation or test rows and never writes to the source
dataset.  It produces statistics for discovery; candidate retraining remains the
decision rule for band selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from utils.config import load_config
from utils.misc import atomic_write_json


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    xr, yr = _rank(x[mask]), _rank(y[mask])
    sx, sy = xr.std(), yr.std()
    if sx == 0 or sy == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _read(path: Path, expected: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        descriptions = [str(value) for value in src.descriptions]
        if descriptions != expected:
            raise RuntimeError(f"band descriptions changed for {path}: {descriptions} != {expected}")
        array = src.read().astype(np.float32, copy=False)
        valid = src.read_masks() > 0
        valid &= np.isfinite(array)
        if src.nodata is not None:
            valid &= array != src.nodata
    return array, valid


def _summarize(values: np.ndarray, name: str, positive: np.ndarray, valid_fraction: float = 1.0) -> dict[str, object]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"band": name, "count": 0, "valid_fraction": 0.0}
    quantiles = np.quantile(finite, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "band": name,
        "count": int(finite.size),
        "valid_fraction": float(valid_fraction),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "near_constant_fraction": float(np.mean(np.abs(finite - np.median(finite)) <= max(float(finite.std()) * 1e-3, 1e-8))),
        "positive_depth_spearman": _spearman(values, positive),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pa_hydrokan/subset1000_v13_compact.xml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/optimization/hydrov14/bands"))
    parser.add_argument("--samples-per-raster", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.samples_per_raster < 1:
        raise ValueError("samples-per-raster must be positive")
    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)
    contract = DatasetContract.load(config["dataset"]["contract"])
    contract.verify_fingerprints(include_normalization=True)
    spec = resolve_band_spec(config, contract)
    with contract.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    arrays: dict[str, list[np.ndarray]] = {}
    validity: dict[str, list[np.ndarray]] = {}
    valid_counts: dict[str, int] = {}
    pixel_counts: dict[str, int] = {}
    depths: list[np.ndarray] = []
    root = contract.dataset_root
    mask_names = [str(value) for value in contract.group("masks")["band_descriptions"]]
    mask_index = {name: index for index, name in enumerate(mask_names)}
    selected_fields: list[tuple[str, str, int]] = []
    for group in ("s1_t1", "s1_t2", "s1_change", "s2_t1", "s2_t2", "s2_change", "terrain"):
        for name in spec.names(group):
            selected_fields.append((group, name, contract.band_index(group, name)))
    for name in spec.names("s1_conditioning"):
        for group in ("s1_t1", "s1_t2"):
            descriptions = [str(value) for value in contract.group(group)["band_descriptions"]]
            if name in descriptions:
                selected_fields.append((group, name, descriptions.index(name)))
                break
    for row in rows:
        loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for group in {field[0] for field in selected_fields}:
            loaded[group] = _read(root / row[contract.group(group)["path_column"]], [str(value) for value in contract.group(group)["band_descriptions"]])
        masks, _ = _read(root / row[contract.group("masks")["path_column"]], mask_names)
        label, label_valid = _read(root / row[contract.group("label")["path_column"]], ["depth_m"])
        common = label_valid[0] & (masks[mask_index["valid_depth_mask"]] > 0)
        for group, _, index in selected_fields:
            field_name = f"{group}/{_}"
            valid_counts[field_name] = valid_counts.get(field_name, 0) + int(loaded[group][1][index].sum())
            pixel_counts[field_name] = pixel_counts.get(field_name, 0) + int(loaded[group][1][index].size)
            common &= loaded[group][1][index]
        positions = np.flatnonzero(common)
        if positions.size == 0:
            continue
        chosen = rng.choice(positions, size=min(args.samples_per_raster, positions.size), replace=False)
        for group, name, index in selected_fields:
            arrays.setdefault(f"{group}/{name}", []).append(loaded[group][0][index].reshape(-1)[chosen].astype(np.float64))
            validity.setdefault(f"{group}/{name}", []).append(np.ones(chosen.size, dtype=np.float64))
        depths.append(label[0].reshape(-1)[chosen].astype(np.float64))
    if not depths:
        raise RuntimeError("No train pixels were available for band statistics")
    flat = {name: np.concatenate(chunks) for name, chunks in arrays.items()}
    positive = np.concatenate(depths)
    stats = [_summarize(values, name, positive, valid_counts.get(name, 0) / max(pixel_counts.get(name, 1), 1)) for name, values in flat.items()]
    correlations = []
    for i, (name_a, values_a) in enumerate(flat.items()):
        for name_b, values_b in list(flat.items())[i + 1:]:
            group_a = name_a.split("/", 1)[0]
            group_b = name_b.split("/", 1)[0]
            if group_a.split("_")[0] != group_b.split("_")[0]:
                continue
            correlations.append({"band_a": name_a, "band_b": name_b, "modality": group_a.split("_")[0], "spearman": _spearman(values_a, values_b)})
    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in stats for key in row})
    with (args.output / "train_band_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(stats)
    corr_fields = ["band_a", "band_b", "modality", "spearman"]
    with (args.output / "train_band_correlations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=corr_fields); writer.writeheader(); writer.writerows(correlations)
    report = {
        "config": str(args.config), "split": "train", "train_rows": len(rows),
        "samples_per_raster_cap": args.samples_per_raster, "sampled_pixels": int(positive.size),
        "sampling": "equal cap per raster from common valid selected-input and positive-label pixels",
        "band_spec": spec.as_dict(), "statistics": stats, "correlations": correlations,
        "train_depth": {
            "count": int(positive.size),
            "p90": float(np.quantile(positive, 0.90)),
            "p95": float(np.quantile(positive, 0.95)),
            "p99": float(np.quantile(positive, 0.99)),
            "sampling": "same raster-stratified pixel sample; threshold is diagnostic until full train-depth stats are supplied",
        },
        "interpretation": "Spearman and univariate association identify possible redundancy only; candidate retraining decides deletion. NDWI/MNDWI changes remain paired with their source spectral bands for review.",
        "normalized_mask_zero_semantics": "not used here; model masking at normalized zero means center/typical-value masking, not physical zero.",
    }
    atomic_write_json(args.output / "train_band_report.json", report)
    print(json.dumps({"sampled_pixels": int(positive.size), "bands": len(stats), "correlations": len(correlations)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

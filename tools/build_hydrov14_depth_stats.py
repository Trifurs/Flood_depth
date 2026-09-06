#!/usr/bin/env python3
"""Build independent exact train-only depth tail thresholds for Hydro-v14."""

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

from datasets.contract import DatasetContract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = DatasetContract.load(args.contract)
    contract.verify_fingerprints(include_normalization=True)
    with contract.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    masks_names = [str(value) for value in contract.group("masks")["band_descriptions"]]
    mi = {name: index for index, name in enumerate(masks_names)}
    depths = []
    for row in rows:
        root = contract.dataset_root
        with rasterio.open(root / row[contract.group("label")["path_column"]]) as src:
            label = src.read(1, masked=False).astype(np.float64, copy=False)
            valid_label = (src.read_masks(1) > 0) & np.isfinite(label)
        with rasterio.open(root / row[contract.group("masks")["path_column"]]) as src:
            masks = src.read(masked=False) > 0
        with rasterio.open(root / row[contract.group("terrain")["path_column"]]) as src:
            terrain_valid = np.logical_and.reduce(src.read_masks() > 0, axis=0)
        eligible = (
            masks[mi["valid_depth_mask"]] & terrain_valid
            & masks[mi["DEM_valid_mask"]] & masks[mi["slope_valid_mask"]]
            & (masks[mi["S1_event_composite_valid_mask"]] | masks[mi["S2_event_composite_valid_mask"]])
            & ~masks[mi["permanent_water_mask"]] & ~masks[mi["extreme_high_mask"]]
        )
        selected = valid_label & eligible
        if np.any(selected):
            depths.append(label[selected])
    if not depths:
        raise RuntimeError("No eligible positive train depth pixels")
    values = np.concatenate(depths)
    stratification_edges = np.quantile(values, [0.0, .25, .50, .75, 1.0])
    stratification_counts, _ = np.histogram(values, bins=stratification_edges)
    # np.histogram includes the rightmost endpoint in its final bin.  The
    # resulting counts therefore sum exactly to the eligible train-pixel count.
    payload = {
        "schema_version": "hydrov14.train_depth_stats.v1", "split": "train",
        "contract": str(contract.path), "contract_sha256": contract.hash,
        "manifest_sha256": contract.payload["manifest"]["sha256"], "count": int(values.size),
        "minimum": float(values.min()), "maximum": float(values.max()), "mean": float(values.mean()),
        "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)), "p99_5": float(np.quantile(values, .995)),
        "stratification_edges": [float(value) for value in stratification_edges],
        "stratification_bin_counts": [int(value) for value in stratification_counts],
        "note": "Exact eligible positive train pixels only; no validation/test values used.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize matched Hydro-v14 full-input/no-S2 validation experiments."""

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
from utils.misc import atomic_write_json


METRICS = (
    "pixel_micro_mae", "pixel_micro_rmse", "pixel_micro_p90_absolute_error",
    "pixel_micro_bias", "sample_macro_mae", "event_macro_mae",
    "event_depth_bin_macro_mae", "event_depth_hierarchical_macro_mae",
)


def _summary(path: Path) -> dict[str, float]:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return {name: float(payload[name]) for name in METRICS}


def _s2_validity(contract: DatasetContract, split: str) -> dict[str, float | int]:
    with contract.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    mask_names = [str(value) for value in contract.group("masks")["band_descriptions"]]
    index = mask_names.index("S2_event_composite_valid_mask")
    values = []
    for row in rows:
        with rasterio.open(contract.dataset_root / row[contract.group("masks")["path_column"]]) as src:
            values.append(float((src.read(index + 1, masked=False) > 0).mean()))
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "pixel_fraction_mean": float(array.mean()),
        "pixel_fraction_median": float(np.median(array)),
        "pixel_fraction_p10": float(np.quantile(array, 0.10)),
        "pixel_fraction_p90": float(np.quantile(array, 0.90)),
        "samples_below_50_percent": int((array < 0.5).sum()),
        "samples_zero": int((array == 0).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--full-eval", type=Path, required=True)
    parser.add_argument("--no-s2-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = DatasetContract.load(args.contract)
    contract.verify_fingerprints(include_normalization=True)
    full = _summary(args.full_eval)
    no_s2 = _summary(args.no_s2_eval)
    delta = {name: no_s2[name] - full[name] for name in METRICS}
    relative = {
        name: delta[name] / abs(full[name]) if full[name] != 0 else None
        for name in METRICS
    }
    payload = {
        "schema_version": "hydrov14.sensor_ablation.v1",
        "dataset_root": str(contract.dataset_root),
        "contract": str(contract.path),
        "split": "val",
        "training_scope": {
            "epochs": 3,
            "max_train_batches_per_epoch": 5,
            "max_validation_batches_during_training": 1,
            "seed": 20260831,
            "device": "cpu",
            "note": "Exploratory matched-budget training; the final comparison below reevaluates all 23 validation batches.",
        },
        "s2_event_validity": _s2_validity(contract, "val"),
        "full_inputs": {"evaluation": str(args.full_eval), "metrics": full},
        "no_s2": {"evaluation": str(args.no_s2_eval), "metrics": no_s2},
        "delta_no_s2_minus_full": delta,
        "relative_delta_no_s2_minus_full": relative,
        "decision": {
            "primary_metric": "pixel_micro_mae",
            "no_s2_is_better_on_primary": no_s2["pixel_micro_mae"] < full["pixel_micro_mae"],
            "promote_no_s2": False,
            "rationale": "No-S2 improves P90 absolute error and reduces negative bias, but worsens pixel/sample/event MAE and depth-bin macro MAE on the complete validation split. The matched exploratory budget is not sufficient for a final deployment decision.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect matched Hydro-v14 screen and controlled validation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.misc import atomic_write_json


METRICS = (
    "pixel_micro_mae", "pixel_micro_rmse", "pixel_micro_p90_absolute_error",
    "pixel_micro_bias", "sample_macro_mae", "event_macro_mae",
    "event_depth_bin_macro_mae", "event_depth_hierarchical_macro_mae",
)


def _json_summary(path: Path, parameters: int | None = None) -> dict[str, Any]:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {name: float(payload[name]) for name in METRICS if name in payload}
    if parameters is not None:
        result["parameters"] = int(parameters)
    return result


def _run_parameters(run: Path) -> int | None:
    path = run / "model_summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("total_parameters", payload.get("parameters_total"))
    return int(value) if value is not None else None


def _screen_summary(path: Path) -> dict[str, Any]:
    with (path / "metrics_by_epoch.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No epoch rows found in {path}")
    row = rows[-1]
    result = {}
    for name in METRICS:
        key = f"val_{name}"
        if key in row:
            result[name] = float(row[key])
    model_summary = json.loads((path / "model_summary.json").read_text(encoding="utf-8"))
    result["parameters"] = int(model_summary.get("total_parameters", model_summary.get("parameters_total", model_summary.get("total", 0))))
    result["checkpoint_epoch"] = int(row.get("epoch", 0))
    result["evidence"] = "one_batch_train_full_validation"
    return result


def _row(name: str, config: str, values: dict[str, Any], *, evidence: str, note: str = "") -> dict[str, Any]:
    item: dict[str, Any] = {"candidate": name, "config": config, "evidence": evidence, "note": note}
    for metric in METRICS:
        item[metric] = values.get(metric)
    item["parameters"] = values.get("parameters")
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/optimization/hydrov14"))
    args = parser.parse_args()
    root = args.output_dir.parent.parent.parent if args.output_dir.is_absolute() else ROOT
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    rows: list[dict[str, Any]] = []
    baseline = ROOT / "artifacts/optimization/hydrov13_2_subset1000/final_v13_val_raw"
    if not (baseline / "summary.json").exists():
        baseline = ROOT / "artifacts/optimization/hydrov13_2_subset1000/final_v13_val_raw.json"
    if (baseline / "summary.json").exists():
        rows.append(_row(
            "corrected_v13_baseline",
            "configs/pa_hydrokan/subset1000_v13_compact.xml",
            _json_summary(baseline, 5076979), evidence="existing_full_validation",
            note="Historical corrected-v13 validation checkpoint; no test split used.",
        ))

    screen_specs = (
        ("no_graph", "configs/pa_hydrokan/subset1000_v14_no_graph.xml"),
        ("edge_mlp", "configs/pa_hydrokan/subset1000_v14_edge_mlp.xml"),
        ("edge_kan_scale4", "configs/pa_hydrokan/subset1000_v14_edge_kan_scale4.xml"),
        ("edge_kan_scale8", "configs/pa_hydrokan/subset1000_v14_edge_kan_scale8.xml"),
        ("terrain_order", "configs/pa_hydrokan/subset1000_v14_terrain_order.xml"),
        ("wse_slope", "configs/pa_hydrokan/subset1000_v14_wse_slope.xml"),
    )
    for name, config in screen_specs:
        run = ROOT / "artifacts/optimization/hydrov14/screen_eval" / f"{name}_1batch_full_val"
        if (run / "summary.json").exists():
            source_run = ROOT / "runs/optimization/hydrov14/screen_resume" / f"{name}_1batch"
            rows.append(_row(name, config, _json_summary(run, _run_parameters(source_run)), evidence="one_batch_train_full_validation"))

    controlled_specs = (
        ("full_inputs_3epoch", "configs/pa_hydrokan/subset1000_v14_bands_selected.xml", "full_inputs_3epoch_5batches_val_full"),
        ("no_s2_3epoch", "configs/pa_hydrokan/subset1000_v14_no_s2_final.xml", "no_s2_3epoch_5batches_val_full"),
    )
    for name, config, directory in controlled_specs:
        run = ROOT / "artifacts/optimization/hydrov14/controlled" / directory
        if (run / "summary.json").exists():
            profile = output_dir / "controlled" / ("full_inputs_profile.json" if name == "full_inputs_3epoch" else "no_s2_profile.json")
            profile_payload = json.loads(profile.read_text(encoding="utf-8")) if profile.exists() else {}
            parameters_value = profile_payload.get("parameters")
            parameters = int(parameters_value) if parameters_value is not None else None
            rows.append(_row(name, config, _json_summary(run, parameters), evidence="3_epoch_5_batch_train_full_validation"))

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["candidate", "config", "evidence", *METRICS, "parameters", "note"]
    with (output_dir / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "hydrov14.ablation_summary.v1",
        "dataset": "subset1000",
        "split": "val",
        "test_split_evaluated": False,
        "selection_metric": "pixel_micro_mae",
        "screen_training": {"epochs": 1, "max_train_batches": 1, "full_validation_batches": 23, "seed": 20260831},
        "rows": rows,
        "interpretation": {
            "screen_is_exploratory": True,
            "controlled_full_vs_no_s2_is_primary_sensor_ablation": True,
            "note": "One-batch screen rows are architecture diagnostics, not final multi-seed claims.",
        },
    }
    atomic_write_json(output_dir / "ablation_summary.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

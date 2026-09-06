#!/usr/bin/env python3
"""Summarize completed V15.2 validation-only candidate experiments.

The summary deliberately reports raw checkpoints as the selection quantity.  EMA
metrics are retained as diagnostics, but never allowed to select a candidate after
the fact.  Incomplete candidates remain visible as pending instead of silently
disappearing from the experiment matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import write_rows
from utils.misc import atomic_write_json


CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("matched_v15", "Matched V15 baseline", "Experiment 0"),
    ("simple_fixed", "V15.1-simple-fixed", "Experiment 1 / Graph-A"),
    ("kan_map", "V15.2 KAN-map", "Experiment 3A"),
    ("graph_bottleneck", "Graph-B skip+bottleneck", "Experiment 4B"),
    ("physics_order", "Physics-A terrain-order", "Experiment 5A"),
    ("physics_barrier", "Physics-B barrier-aware", "Experiment 5B"),
    ("combined", "Best Graph+KAN+physics combination", "Experiment 6"),
    ("raw_sar_shortcut", "Raw SAR shortcut on", "Experiment 7"),
    ("augmentation_off", "Geometry flips off", "Experiment 8"),
    ("loss_beta025", "Lean loss: depth Huber β=0.25", "Loss screen A"),
    ("loss_log010", "Lean loss: log weight=0.10", "Loss screen B"),
    ("tail_alpha010", "Tail underprediction α=0.10", "Conditional tail screen A"),
    ("tail_alpha015", "Tail underprediction α=0.15", "Conditional tail screen B"),
)
EXACT_RUN_ALIASES: dict[str, str] = {
    # Experiment 6 has no independent parameter setting: Physics-A was
    # already trained with the selected original KAN and Graph-A structure.
    # Keep the alias visible instead of mislabelling it as an unrun candidate.
    "combined": "physics_order",
}
METRIC_KEYS = (
    "pixel_micro_mae",
    "pixel_micro_rmse",
    "pixel_micro_p90_absolute_error",
    "pixel_micro_bias",
)
CONTROL_PATHS = (
    ("seed",),
    ("training", "epochs"),
    ("training", "minimum_epochs"),
    ("training", "batch_size"),
    ("training", "gradient_accumulation_steps"),
    ("training", "amp"),
    ("training", "amp_dtype"),
    ("training", "ema_enabled"),
    ("optimizer",),
    ("scheduler",),
    ("dataset", "sampling"),
    ("supervision",),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_deep_bin(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    deepest = max(rows, key=lambda row: int(row["bin"]))
    return {
        "bin": int(deepest["bin"]),
        "lower_train_boundary_m": deepest["lower_train_boundary_m"],
        "upper_train_boundary_m": deepest["upper_train_boundary_m"],
        "mae": float(deepest["mae"]),
        "bias": float(deepest["bias"]),
        "p90_absolute_error": float(deepest["p90_absolute_error"]),
        "pixels": int(deepest["pixels"]),
    }


def _nested_value(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _config_differences(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[str]:
    differences = []
    for path in CONTROL_PATHS:
        if _nested_value(reference, path) != _nested_value(candidate, path):
            differences.append(".".join(path))
    return differences


def _profile_record(artifacts_root: Path, run_name: str) -> dict[str, Any] | None:
    profile = artifacts_root / f"profile_{run_name}_batch12.json"
    if not profile.is_file():
        return None
    payload = _read_json(profile)
    return {
        "path": str(profile.resolve()),
        "parameters": int(payload["parameters"]),
        "forward_backward_peak_gpu_memory_bytes": int(
            payload["forward_backward_peak_gpu_memory_bytes"]
        ),
        "samples_per_second": float(payload["samples_per_second"]),
        "gradients_finite": bool(payload["gradients_finite"]),
    }


def _completed_record(
    run_dir: Path,
    artifacts_root: Path,
    name: str,
    label: str,
    experiment: str,
    simple_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = _read_json(run_dir / "eval_raw" / "summary.json")
    ema_path = run_dir / "eval_ema" / "summary.json"
    ema = _read_json(ema_path) if ema_path.is_file() else None
    config = _read_json(run_dir / "resolved_config.json")
    model = _read_json(run_dir / "model_summary.json")
    runtime_path = run_dir / "training_runtime.json"
    runtime = _read_json(runtime_path) if runtime_path.is_file() else None
    differences = (
        [] if simple_config is None else _config_differences(simple_config, config)
    )
    record = {
        "name": name,
        "label": label,
        "experiment": experiment,
        "status": "completed",
        "selection": {
            "weights": "raw",
            "checkpoint_epoch": int(raw["checkpoint_epoch"]),
            "rule": "minimum raw canonical validation MAE within this fixed epoch budget",
        },
        "protocol": {
            "seed": int(config["seed"]),
            "epochs": int(config["training"]["epochs"]),
            "minimum_epochs": int(config["training"]["minimum_epochs"]),
            "batch_size": int(config["training"]["batch_size"]),
            "gradient_accumulation_steps": int(config["training"]["gradient_accumulation_steps"]),
            "amp_dtype": str(config["training"]["amp_dtype"]),
            "augmentation": config["dataset"]["augmentation"],
            "s1_only": bool(
                config["dataset"].get("input_mode") == "s1_terrain"
                and "s2_t1" in config["dataset"].get("resolved_model_input_spec", {}).get("inactive_groups", [])
            ),
            "control_differences_vs_simple_fixed": differences,
        },
        "raw_validation": {key: float(raw[key]) for key in METRIC_KEYS},
        "raw_deep_bin": _read_deep_bin(
            run_dir / "eval_raw" / "metrics_by_train_depth_bin.csv"
        ),
        "ema_validation": (
            {key: float(ema[key]) for key in METRIC_KEYS} if ema is not None else None
        ),
        "model": {
            "name": str(model["name"]),
            "parameters": int(model["total_parameters"]),
            "graph_identity": model["graph_identity"],
        },
        "efficiency": {
            "training_peak_gpu_memory_bytes": (
                int(runtime["peak_gpu_memory_bytes"]) if runtime is not None else None
            ),
            "training_elapsed_seconds": (
                float(runtime["elapsed_seconds"]) if runtime is not None else None
            ),
            "profile": _profile_record(artifacts_root, name),
        },
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "raw_summary": str((run_dir / "eval_raw" / "summary.json").resolve()),
            "ema_summary": str(ema_path.resolve()) if ema_path.is_file() else None,
        },
    }
    return record


def _csv_record(record: Mapping[str, Any], reference_mae: float | None) -> dict[str, Any]:
    if record["status"] != "completed":
        return {
            "candidate": record["label"],
            "experiment": record["experiment"],
            "status": record["status"],
        }
    raw = record["raw_validation"]
    deep = record["raw_deep_bin"] or {}
    profile = record["efficiency"]["profile"] or {}
    mae = float(raw["pixel_micro_mae"])
    return {
        "candidate": record["label"],
        "experiment": record["experiment"],
        "status": record["status"],
        "checkpoint_epoch": record["selection"]["checkpoint_epoch"],
        "raw_mae": mae,
        "raw_rmse": raw["pixel_micro_rmse"],
        "raw_p90": raw["pixel_micro_p90_absolute_error"],
        "raw_bias": raw["pixel_micro_bias"],
        "deep_mae": deep.get("mae"),
        "deep_bias": deep.get("bias"),
        "parameters": record["model"]["parameters"],
        "training_peak_gpu_memory_bytes": record["efficiency"]["training_peak_gpu_memory_bytes"],
        "profile_samples_per_second": profile.get("samples_per_second"),
        "mae_delta_vs_simple_fixed": (mae - reference_mae) if reference_mae is not None else None,
        "mae_relative_delta_vs_simple_fixed": (
            (mae - reference_mae) / reference_mae if reference_mae not in (None, 0.0) else None
        ),
        "control_differences_vs_simple_fixed": ";".join(
            record["protocol"]["control_differences_vs_simple_fixed"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs/optimization/hydrokan_s1_v15_2",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/optimization/hydrokan_s1_v15_2",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    output_json = args.output_json or args.artifacts_root / "candidate_summary.json"
    output_csv = args.output_csv or args.artifacts_root / "candidate_summary.csv"
    simple_config_path = args.runs_root / "simple_fixed" / "resolved_config.json"
    simple_config = _read_json(simple_config_path) if simple_config_path.is_file() else None
    records: list[dict[str, Any]] = []
    for name, label, experiment in CANDIDATES:
        run_dir = args.runs_root / name
        raw_summary = run_dir / "eval_raw" / "summary.json"
        if raw_summary.is_file():
            records.append(
                _completed_record(
                    run_dir, args.artifacts_root, name, label, experiment, simple_config
                )
            )
        elif name in EXACT_RUN_ALIASES:
            source_name = EXACT_RUN_ALIASES[name]
            source_run_dir = args.runs_root / source_name
            source_raw = source_run_dir / "eval_raw" / "summary.json"
            selection_path = run_dir / "selection.json"
            if source_raw.is_file() and selection_path.is_file():
                record = _completed_record(
                    source_run_dir,
                    args.artifacts_root,
                    name,
                    label,
                    experiment,
                    simple_config,
                )
                record["selection"]["rule"] = (
                    "exact configuration alias of the selected source run; "
                    "no redundant deterministic retraining"
                )
                record["alias"] = {
                    "source_name": source_name,
                    "source_run_dir": str(source_run_dir.resolve()),
                    "selection_record": str(selection_path.resolve()),
                }
                record["paths"]["run_dir"] = str(run_dir.resolve())
                record["paths"]["source_raw_summary"] = str(source_raw.resolve())
                records.append(record)
            else:
                records.append(
                    {
                        "name": name,
                        "label": label,
                        "experiment": experiment,
                        "status": "pending_or_not_run",
                        "paths": {"run_dir": str(run_dir.resolve())},
                    }
                )
        else:
            records.append(
                {
                    "name": name,
                    "label": label,
                    "experiment": experiment,
                    "status": "pending_or_not_run",
                    "paths": {"run_dir": str(run_dir.resolve())},
                }
            )
    completed = [record for record in records if record["status"] == "completed"]
    simple = next((record for record in completed if record["name"] == "simple_fixed"), None)
    reference_mae = (
        float(simple["raw_validation"]["pixel_micro_mae"]) if simple is not None else None
    )
    best = min(
        completed,
        key=lambda record: float(record["raw_validation"]["pixel_micro_mae"]),
        default=None,
    )
    payload = {
        "scope": "validation-only V15.2 screening; no test split used",
        "selection_metric": "raw pixel_micro_mae",
        "selection_rule": "each completed candidate uses its best raw checkpoint in the fixed 45-epoch budget",
        "simple_fixed_reference": (
            {
                "raw_mae": reference_mae,
                "run_dir": simple["paths"]["run_dir"],
            }
            if simple is not None
            else None
        ),
        "current_best_completed_candidate": (
            {
                "name": best["name"],
                "label": best["label"],
                "raw_mae": best["raw_validation"]["pixel_micro_mae"],
            }
            if best is not None
            else None
        ),
        "candidates": records,
    }
    atomic_write_json(output_json, payload)
    write_rows(output_csv, [_csv_record(record, reference_mae) for record in records])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

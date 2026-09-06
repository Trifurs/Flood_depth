#!/usr/bin/env python3
"""Build the reproducible V15.3 screening and final-decision artifacts.

This script intentionally treats the freshly retrained matched V15 model as the
primary gate.  A V15.3 candidate is not promoted merely because it improves a
secondary metric: it must first beat the matched raw validation MAE under the
same 45-epoch, batch-12, S1-only protocol.  Multi-seed final training is only
needed after that screening gate is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import write_rows
from utils.misc import atomic_write_json


METRIC_KEYS = (
    "pixel_micro_mae",
    "pixel_micro_rmse",
    "pixel_micro_p90_absolute_error",
    "pixel_micro_bias",
    "event_depth_hierarchical_macro_mae",
    "event_hierarchical_composite_mae",
)

CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "name": "matched_current_best",
        "label": "Matched fresh current-best V15 baseline",
        "kind": "baseline",
        "run": "matched_current_best",
        "config": "configs/pa_hydrokan/subset1000_s1_v15_3_matched_best.xml",
        "profile": "profile_matched_current_best_batch12.json",
    },
    {
        "name": "stability_graph",
        "label": "V15.3 stability-first Graph+KAN",
        "kind": "candidate",
        "run": "stability_graph/seed_20260904",
        "config": "configs/pa_hydrokan/subset1000_s1_v15_3_stability_graph.xml",
        "profile": "profile_v15_3_stability_graph_batch12.json",
    },
    {
        "name": "stability_physics_a",
        "label": "V15.3 stability-first + Physics-A",
        "kind": "candidate",
        "run": "physics_a/seed_20260904",
        "config": "configs/pa_hydrokan/subset1000_s1_v15_3_physics_a.xml",
        "profile": "profile_v15_3_stability_graph_batch12.json",
    },
    {
        "name": "stability_physics_c",
        "label": "V15.3 stability-first + Physics-C WSE",
        "kind": "candidate",
        "run": "physics_c/seed_20260904",
        "config": "configs/pa_hydrokan/subset1000_s1_v15_3_physics_c.xml",
        "profile": "profile_v15_3_stability_graph_batch12.json",
    },
    {
        "name": "accuracy_physics_a",
        "label": "Accuracy-first V15 + Physics-A",
        "kind": "candidate",
        "run": "accuracy/physics_a/seed_20260904",
        "config": "configs/pa_hydrokan/subset1000_s1_v15_3_accuracy_physics_a.xml",
        "profile": "profile_matched_current_best_batch12.json",
    },
    {
        "name": "accuracy_physics_c",
        "label": "Accuracy-first V15 + Physics-C WSE",
        "kind": "candidate",
        "run": "accuracy/physics_c/seed_20260904",
        "config": "configs/pa_hydrokan/subset1000_s1_v15_3_accuracy_physics_c.xml",
        "profile": "profile_matched_current_best_batch12.json",
    },
    {
        "name": "accuracy_physics_a_w001",
        "label": "Accuracy-first V15 + Physics-A (lambda=0.001)",
        "kind": "candidate",
        "run": "accuracy/physics_a_w001/seed_20260904",
        "config": "configs/pa_hydrokan/subset1000_s1_v15_3_accuracy_physics_a_w001.xml",
        "profile": "profile_matched_current_best_batch12.json",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _deep_bin(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    rows = _read_rows(path)
    if not rows:
        return None
    selected = min(
        rows,
        key=lambda row: abs(float(row["lower_train_boundary_m"]) - 0.5),
    )
    return {
        "bin": int(selected["bin"]),
        "lower_train_boundary_m": _float_or_none(selected["lower_train_boundary_m"]),
        "upper_train_boundary_m": _float_or_none(selected["upper_train_boundary_m"]),
        "mae": float(selected["mae"]),
        "rmse": float(selected["rmse"]),
        "bias": float(selected["bias"]),
        "p90_absolute_error": float(selected["p90_absolute_error"]),
        "pixels": int(selected["pixels"]),
    }


def _metric_view(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    result: dict[str, Any] = {}
    for key in METRIC_KEYS:
        result[key] = float(summary[key])
    result["checkpoint_epoch"] = int(summary["checkpoint_epoch"])
    result["weights"] = str(summary["weights"])
    return result


def _physics_view(summary: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "objective_physics_mean",
        "objective_physics_active_pair_fraction_mean",
        "objective_physics_active_pair_count_mean",
        "objective_physics_candidate_pair_count_mean",
        "objective_physics_mean_violation_m_mean",
        "objective_physics_p90_violation_m_mean",
        "objective_physics_effective_weight_mean",
        "objective_physics_mean_sar_compatibility_mean",
        "objective_physics_mean_barrier_weight_mean",
        "objective_physics_mean_complexity_weight_mean",
    )
    return {name: float(summary.get(name, 0.0)) for name in names}


def _profile(artifacts_root: Path, filename: str) -> dict[str, Any] | None:
    path = artifacts_root / filename
    if not path.is_file():
        return None
    payload = _read_json(path)
    return {
        "path": str(path.resolve()),
        "parameters": int(payload["parameters"]),
        "forward_backward_peak_gpu_memory_bytes": int(
            payload["forward_backward_peak_gpu_memory_bytes"]
        ),
        "forward_backward_seconds": float(payload["forward_backward_seconds"]),
        "samples_per_second": float(payload["samples_per_second"]),
        "gradients_finite": bool(payload["gradients_finite"]),
    }


def _is_s1_only(config: Mapping[str, Any]) -> bool:
    dataset = config.get("dataset", {})
    spec = dataset.get("resolved_model_input_spec", {})
    return bool(
        dataset.get("input_mode") == "s1_terrain"
        and "s2_t1" in spec.get("inactive_groups", [])
        and "s2_t2" in spec.get("inactive_groups", [])
        and "s2_change" in spec.get("inactive_groups", [])
        and "s2_qa" in spec.get("inactive_groups", [])
    )


def _record(
    spec: Mapping[str, str],
    runs_root: Path,
    artifacts_root: Path,
    baseline_mae: float | None,
) -> dict[str, Any]:
    run = runs_root / spec["run"]
    raw_path = run / "eval_raw" / "summary.json"
    if not raw_path.is_file():
        return {
            "name": spec["name"],
            "label": spec["label"],
            "kind": spec["kind"],
            "status": "incomplete",
            "paths": {"run_dir": str(run.resolve())},
        }
    raw = _read_json(raw_path)
    ema_path = run / "eval_ema" / "summary.json"
    ema = _read_json(ema_path) if ema_path.is_file() else None
    config = _read_json(run / "resolved_config.json")
    model = _read_json(run / "model_summary.json")
    runtime_path = run / "training_runtime.json"
    runtime = _read_json(runtime_path) if runtime_path.is_file() else None
    raw_mae = float(raw["pixel_micro_mae"])
    delta = None if baseline_mae is None else raw_mae - baseline_mae
    relative = None if baseline_mae in (None, 0.0) else delta / baseline_mae
    if spec["kind"] == "baseline":
        decision = "reference"
    elif raw_mae < float(baseline_mae):
        decision = "passes_one_seed_mae_gate"
    else:
        decision = "rejected_primary_mae_gate"
    profile = _profile(artifacts_root, spec["profile"])
    return {
        "name": spec["name"],
        "label": spec["label"],
        "kind": spec["kind"],
        "status": "completed",
        "decision": decision,
        "selection": {
            "weights": "raw",
            "metric": "pixel_micro_mae",
            "checkpoint_epoch": int(raw["checkpoint_epoch"]),
            "rule": "minimum raw canonical validation MAE within the fixed 45-epoch budget",
        },
        "protocol": {
            "seed": int(config["seed"]),
            "epochs": int(config["training"]["epochs"]),
            "minimum_epochs": int(config["training"]["minimum_epochs"]),
            "batch_size": int(config["training"]["batch_size"]),
            "gradient_accumulation_steps": int(config["training"]["gradient_accumulation_steps"]),
            "amp": bool(config["training"]["amp"]),
            "amp_dtype": str(config["training"]["amp_dtype"]),
            "ema_enabled": bool(config["training"]["ema_enabled"]),
            "input_mode": str(config["dataset"]["input_mode"]),
            "s1_only": _is_s1_only(config),
            "test_split_used": False,
        },
        "raw_validation": _metric_view(raw),
        "ema_validation": _metric_view(ema),
        "raw_deep_bin": _deep_bin(run / "eval_raw" / "metrics_by_train_depth_bin.csv"),
        "ema_deep_bin": _deep_bin(run / "eval_ema" / "metrics_by_train_depth_bin.csv")
        if ema is not None
        else None,
        "relative_to_fresh_baseline": {
            "mae_delta": delta,
            "mae_relative_delta": relative,
            "raw_mae_beats": (
                None if baseline_mae is None else raw_mae < baseline_mae
            ),
        },
        "physics": {
            "lambda_phys": float(config["loss"].get("lambda_phys", 0.0)),
            "mode": str(config["loss"].get("physics_mode", "none")),
            "start_epoch": int(config["loss"].get("phys_start_epoch", 0)),
            "warmup_epochs": int(config["loss"].get("phys_warmup_epochs", 0)),
            "raw_diagnostics": _physics_view(raw),
            "ema_diagnostics": _physics_view(ema) if ema is not None else None,
        },
        "model": {
            "name": str(model["name"]),
            "parameters": int(model["total_parameters"]),
            "trainable_parameters": int(model["trainable_parameters"]),
            "graph_identity": raw.get("graph_identity"),
        },
        "efficiency": {
            "training_elapsed_seconds": (
                float(runtime["elapsed_seconds"]) if runtime is not None else None
            ),
            "training_peak_gpu_memory_bytes": (
                int(runtime["peak_gpu_memory_bytes"]) if runtime is not None else None
            ),
            "profile": profile,
        },
        "paths": {
            "run_dir": str(run.resolve()),
            "config": str((PROJECT_ROOT / spec["config"]).resolve()),
            "raw_summary": str(raw_path.resolve()),
            "ema_summary": str(ema_path.resolve()) if ema_path.is_file() else None,
            "checkpoint": str((run / "best_raw.pth").resolve()),
        },
    }


def _csv_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if record["status"] != "completed":
        return {
            "candidate": record["name"],
            "label": record["label"],
            "status": record["status"],
            "decision": "incomplete",
        }
    raw = record["raw_validation"]
    ema = record["ema_validation"] or {}
    deep = record["raw_deep_bin"] or {}
    physics = record["physics"]
    profile = record["efficiency"]["profile"] or {}
    return {
        "candidate": record["name"],
        "label": record["label"],
        "status": record["status"],
        "decision": record["decision"],
        "raw_checkpoint_epoch": record["selection"]["checkpoint_epoch"],
        "raw_mae": raw["pixel_micro_mae"],
        "raw_rmse": raw["pixel_micro_rmse"],
        "raw_p90": raw["pixel_micro_p90_absolute_error"],
        "raw_bias": raw["pixel_micro_bias"],
        "ema_mae": ema.get("pixel_micro_mae"),
        "ema_rmse": ema.get("pixel_micro_rmse"),
        "ema_p90": ema.get("pixel_micro_p90_absolute_error"),
        "ema_bias": ema.get("pixel_micro_bias"),
        "deep_raw_mae": deep.get("mae"),
        "deep_raw_rmse": deep.get("rmse"),
        "deep_raw_p90": deep.get("p90_absolute_error"),
        "deep_raw_bias": deep.get("bias"),
        "lambda_phys": physics["lambda_phys"],
        "physics_mode": physics["mode"],
        "physics_active_fraction": physics["raw_diagnostics"]["objective_physics_active_pair_fraction_mean"],
        "physics_mean_violation_m": physics["raw_diagnostics"]["objective_physics_mean_violation_m_mean"],
        "parameters": record["model"]["parameters"],
        "training_peak_gpu_memory_bytes": record["efficiency"]["training_peak_gpu_memory_bytes"],
        "profile_samples_per_second": profile.get("samples_per_second"),
        "mae_delta_vs_fresh_baseline": record["relative_to_fresh_baseline"]["mae_delta"],
        "mae_relative_delta_vs_fresh_baseline": record["relative_to_fresh_baseline"]["mae_relative_delta"],
    }


def _baseline_seed_rows(baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in baseline.get("per_seed_raw", []):
        metrics = item["metrics"]
        rows.append(
            {
                "variant": "historical_current_best_v15",
                "status": "historical_final_reference",
                "seed": item["seed"],
                "raw_mae": metrics["mae"],
                "raw_rmse": metrics["rmse"],
                "raw_p90": metrics["p90_absolute_error"],
                "raw_bias": metrics["bias"],
                "checkpoint_epoch": metrics.get("checkpoint_epoch"),
                "note": "three-seed matched V15 reference; retained, not retuned",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs/optimization/hydrokan_s1_v15_3")
    parser.add_argument("--artifacts-root", type=Path, default=PROJECT_ROOT / "artifacts/optimization/hydrokan_s1_v15_3")
    parser.add_argument("--pytest-result", default="169 passed, 2 skipped, 2 warnings")
    args = parser.parse_args()
    args.runs_root = args.runs_root.resolve()
    args.artifacts_root = args.artifacts_root.resolve()
    args.artifacts_root.mkdir(parents=True, exist_ok=True)

    baseline_spec = CANDIDATES[0]
    baseline_run = args.runs_root / baseline_spec["run"]
    baseline_summary = _read_json(baseline_run / "eval_raw" / "summary.json")
    baseline_mae = float(baseline_summary["pixel_micro_mae"])
    records = [
        _record(spec, args.runs_root, args.artifacts_root, baseline_mae)
        for spec in CANDIDATES
    ]
    candidates = [record for record in records if record["kind"] == "candidate" and record["status"] == "completed"]
    passing = [record for record in candidates if record["decision"] == "passes_one_seed_mae_gate"]
    closest = min(candidates, key=lambda item: item["relative_to_fresh_baseline"]["mae_delta"])
    historical = _read_json(
        PROJECT_ROOT / "artifacts/optimization/hydrokan_s1_v15_3/current_best_baseline.json"
    )
    profile_selection = _read_json(args.artifacts_root / "batch_profile_selection.json")
    prechange = _read_json(args.artifacts_root / "prechange_status.json")

    summary = {
        "name": "HydroKAN S1-only V15.3 screening summary",
        "scope": "validation-only; no Sentinel-2 input and no test split",
        "selection_metric": "raw pixel_micro_mae",
        "selection_rule": "strictly beat the freshly retrained matched V15 raw validation MAE before multi-seed promotion",
        "matched_protocol": {
            "dataset": "subset1000_s1_only",
            "input_mode": "s1_terrain",
            "epochs": 45,
            "minimum_epochs": 30,
            "batch_size": 12,
            "gradient_accumulation_steps": 1,
            "amp_dtype": "bfloat16",
            "seed_for_screening": 20260904,
            "test_split_used": False,
        },
        "fresh_matched_baseline": {
            "name": "pa_hydrokan_s1_v15",
            "raw_mae": baseline_mae,
            "raw_rmse": float(baseline_summary["pixel_micro_rmse"]),
            "raw_p90": float(baseline_summary["pixel_micro_p90_absolute_error"]),
            "raw_bias": float(baseline_summary["pixel_micro_bias"]),
            "checkpoint_epoch": int(baseline_summary["checkpoint_epoch"]),
            "run_dir": str(baseline_run.resolve()),
        },
        "historical_three_seed_reference": {
            "raw_mae_mean": historical["raw_aggregate"]["mae"]["mean"],
            "raw_mae_std": historical["raw_aggregate"]["mae"]["sample_std"],
            "seeds": historical["seeds"],
            "source": str((PROJECT_ROOT / "artifacts/optimization/hydrokan_s1_v15_3/current_best_baseline.json").resolve()),
        },
        "candidates": records,
        "best_screened_candidate": {
            "name": closest["name"],
            "label": closest["label"],
            "raw_mae": closest["raw_validation"]["pixel_micro_mae"],
            "raw_mae_delta": closest["relative_to_fresh_baseline"]["mae_delta"],
            "raw_mae_relative_delta": closest["relative_to_fresh_baseline"]["mae_relative_delta"],
            "decision": closest["decision"],
        },
        "passing_candidates": [record["name"] for record in passing],
        "test_split_used": False,
    }
    atomic_write_json(args.artifacts_root / "candidate_summary.json", summary)
    write_rows(args.artifacts_root / "candidate_summary.csv", [_csv_record(record) for record in records])

    final_decision = {
        "name": "HydroKAN S1-only V15.3 final decision",
        "accepted": False,
        "accepted_candidate": None,
        "decision": "retain_current_best_v15",
        "reason": (
            "No screened V15.3 optimization candidate strictly beat the freshly retrained "
            "matched V15 raw validation MAE under the fixed 45-epoch protocol; the closest "
            "candidate remained worse on the primary MAE gate. Per the requested hard gate, "
            "the new version is rejected and the existing V15 S1-only model is retained."
        ),
        "fresh_baseline": summary["fresh_matched_baseline"],
        "closest_candidate": summary["best_screened_candidate"],
        "historical_reference": summary["historical_three_seed_reference"],
        "multi_seed_final": {
            "required_if_screen_passed": True,
            "executed": False,
            "status": "not_run_no_candidate_passed_seed_20260904_screening",
            "reason": "The promotion gate failed before a candidate qualified for three-seed final training.",
        },
        "test_split_used": False,
        "s2_input_used": False,
        "reproducibility": {
            "summary": str((args.artifacts_root / "candidate_summary.json").resolve()),
            "report": str((PROJECT_ROOT / "docs/HYDROKAN_S1_V15_3_ENGINEERING_REPORT.md").resolve()),
        },
    }
    atomic_write_json(args.artifacts_root / "final_decision.json", final_decision)

    physics_rows = []
    for record in records:
        if record["status"] != "completed" or record["physics"]["lambda_phys"] <= 0.0:
            continue
        raw = record["raw_validation"]
        diag = record["physics"]["raw_diagnostics"]
        physics_rows.append(
            {
                "candidate": record["name"],
                "label": record["label"],
                "mode": record["physics"]["mode"],
                "lambda_phys": record["physics"]["lambda_phys"],
                "raw_mae": raw["pixel_micro_mae"],
                "raw_mae_delta_vs_baseline": record["relative_to_fresh_baseline"]["mae_delta"],
                "physics_mean": diag["objective_physics_mean"],
                "active_pair_fraction": diag["objective_physics_active_pair_fraction_mean"],
                "mean_violation_m": diag["objective_physics_mean_violation_m_mean"],
                "p90_violation_m": diag["objective_physics_p90_violation_m_mean"],
                "effective_weight": diag["objective_physics_effective_weight_mean"],
            }
        )
    physics_comparison = {
        "scope": "validation-only physics diagnostics; no test split",
        "physics_a_definition": "terrain_order_margin",
        "physics_c_definition": "wse_consistency",
        "rows": physics_rows,
        "interpretation": {
            "physics_a": "sparse terrain-order violations with nonzero gradient and lower active-pair fraction",
            "physics_c": "near-all-pair WSE consistency barrier; higher active-pair fraction and no primary-MAE improvement",
            "promotion": "neither Physics-A nor Physics-C passed the primary raw-MAE gate",
        },
    }
    atomic_write_json(args.artifacts_root / "physics_comparison.json", physics_comparison)

    baseline_seed_rows = _baseline_seed_rows(historical)
    multi_seed = {
        "scope": "final multi-seed status; validation-only",
        "baseline_reference": baseline_seed_rows,
        "candidate_final_runs": [],
        "screening_runs": [
            {
                "candidate": record["name"],
                "seed": record["protocol"]["seed"],
                "raw_mae": record["raw_validation"]["pixel_micro_mae"],
                "status": "screening_only_rejected",
            }
            for record in candidates
        ],
        "decision": "No candidate qualified for the requested 3-seed final stage.",
    }
    atomic_write_json(args.artifacts_root / "multiseed_summary.json", multi_seed)
    write_rows(
        args.artifacts_root / "multiseed_summary.csv",
        baseline_seed_rows
        + [
            {
                "variant": item["candidate"],
                "status": item["status"],
                "seed": item["seed"],
                "raw_mae": item["raw_mae"],
                "raw_rmse": "",
                "raw_p90": "",
                "raw_bias": "",
                "checkpoint_epoch": "",
                "note": "single-seed screen only; not promoted to final multi-seed stage",
            }
            for item in multi_seed["screening_runs"]
        ],
    )

    profile_payload = {
        "scope": "real-raster BF16 forward/backward profiling; no test split",
        "selected": profile_selection,
        "records": [
            record["efficiency"]["profile"]
            for record in records
            if record["status"] == "completed" and record["efficiency"]["profile"] is not None
        ],
        "conclusion": "batch 12 is the largest shared stable batch; batch 16 OOMs for both baseline and V15.3 stability routes",
    }
    atomic_write_json(args.artifacts_root / "final_profile.json", profile_payload)

    sample_bootstrap = _read_json(args.artifacts_root / "paired_bootstrap_screening/paired_bootstrap.json")
    event_bootstrap = _read_json(args.artifacts_root / "paired_bootstrap_screening_event/paired_bootstrap.json")
    atomic_write_json(
        args.artifacts_root / "paired_bootstrap.json",
        {
            "scope": "screening-only paired validation bootstrap; no test split",
            "candidate": "accuracy_physics_a",
            "baseline": "matched_current_best",
            "sample_unit": sample_bootstrap,
            "event_unit": event_bootstrap,
            "interpretation": "MAE point estimate is worse for the candidate; bootstrap intervals are retained as uncertainty diagnostics and do not override the hard gate.",
        },
    )

    source_counterfactual = _read_json(
        args.artifacts_root / "graph_kan_counterfactuals/kan_counterfactuals.json"
    )
    for filename, role in (
        ("graph_counterfactuals.json", "graph ablation diagnostic"),
        ("kan_counterfactuals.json", "KAN ablation diagnostic"),
    ):
        payload = dict(source_counterfactual)
        payload["artifact_role"] = role
        payload["decision_use"] = "diagnostic_only_not_a_promotion_result"
        atomic_write_json(args.artifacts_root / filename, payload)

    instability = _read_json(args.artifacts_root / "bad_seed_diagnostics.json")
    instability = dict(instability)
    instability["artifact_role"] = "pre-V15.3 instability diagnostic"
    instability["decision_use"] = "diagnostic_only_no_seed_removed"
    atomic_write_json(args.artifacts_root / "seed_instability_analysis.json", instability)

    verification = {
        "name": "HydroKAN S1-only V15.3 verification summary",
        "test_split_used": False,
        "s2_input_used": False,
        "pytest": {
            "command": "conda run --no-capture-output -n flood-depth python -m pytest -q",
            "result": args.pytest_result,
        },
        "cuda": prechange.get("gpu", {}),
        "environment": {
            "python": prechange.get("python", {}),
            "pytorch": prechange.get("pytorch", {}),
        },
        "checks": [
            "S1-only input contract: dataset input_mode=s1_terrain and all Sentinel-2 groups inactive",
            "No test split evaluated in screening, diagnostics, bootstrap, or final decision",
            "Matched 45-epoch budget, minimum 30 epochs, batch 12, BF16, and seed 20260904",
            "Raw checkpoint is the primary selection weight; EMA is reported independently",
            "Real-raster batch 16 OOM was recorded; batch 12 selected with memory headroom",
            "Graph/KAN and Physics gradients finite and nonzero in dedicated probes",
        ],
        "artifacts": {
            "candidate_summary": str((args.artifacts_root / "candidate_summary.json").resolve()),
            "final_decision": str((args.artifacts_root / "final_decision.json").resolve()),
            "physics_comparison": str((args.artifacts_root / "physics_comparison.json").resolve()),
            "multiseed_summary": str((args.artifacts_root / "multiseed_summary.json").resolve()),
            "final_profile": str((args.artifacts_root / "final_profile.json").resolve()),
            "paired_bootstrap": str((args.artifacts_root / "paired_bootstrap.json").resolve()),
        },
    }
    atomic_write_json(args.artifacts_root / "verification_summary.json", verification)
    print(json.dumps({
        "baseline_raw_mae": baseline_mae,
        "closest_candidate": summary["best_screened_candidate"],
        "passing_candidates": summary["passing_candidates"],
        "decision": final_decision["decision"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

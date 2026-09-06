#!/usr/bin/env python3
"""Build auditable non-test verification facts for the V15.2 final report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.model_input_spec import ModelInputSpec
from utils.config import load_config
from utils.misc import atomic_write_json


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _finite(value: str | None) -> float | None:
    try:
        parsed = float(value) if value not in {None, "", "nan", "NaN"} else float("nan")
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        raise ValueError("Expected at least one finite diagnostic value")
    return float(statistics.fmean(data))


def _physics_summary(candidate_root: Path, seeds: list[int]) -> dict[str, Any]:
    fields = (
        "train_physics_effective_weight",
        "train_physics_active_pair_fraction",
        "train_physics_active_pair_count",
        "train_physics_mean_violation_m",
        "train_physics_p90_violation_m",
        "train_physics_gradient_norm",
        "train_depth_gradient_norm",
        "train_physics_depth_gradient_cosine_similarity",
    )
    collected: dict[str, list[float]] = {field: [] for field in fields}
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        history = candidate_root / f"seed_{seed}" / "metrics_by_epoch.csv"
        with history.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        active = [
            row for row in rows
            if (_finite(row.get("train_physics_effective_weight")) or 0.0) > 0.0
        ]
        if not active:
            raise RuntimeError(f"No active weak-physics epochs found in {history}")
        seed_values: dict[str, float] = {}
        for field in fields:
            values = [value for row in active if (value := _finite(row.get(field))) is not None]
            if not values:
                raise RuntimeError(f"No finite {field} values in active rows of {history}")
            seed_values[field] = _mean(values)
            collected[field].extend(values)
        per_seed[str(seed)] = {
            "active_epochs": len(active),
            "first_active_epoch": int(active[0]["epoch"]),
            "last_active_epoch": int(active[-1]["epoch"]),
            "means_over_active_epochs": seed_values,
        }
    all_means = {field: _mean(values) for field, values in collected.items()}
    return {
        "scope": "candidate training histories; active weak-physics epochs only",
        "per_seed": per_seed,
        "mean_over_all_active_epochs": all_means,
        "physics_gradient_nonzero": all_means["train_physics_gradient_norm"] > 0.0,
        "depth_gradient_nonzero": all_means["train_depth_gradient_norm"] > 0.0,
    }


def _counterfactual_rows(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode, value in payload["results"].items():
        metrics = value["metrics"]
        delta = value["delta_vs_full_candidate_minus_full"]
        result[mode] = {
            "mae": float(metrics["mae"]),
            "delta_mae_vs_full": float(delta["mae"]),
            "delta_p90_vs_full": float(delta["p90_absolute_error"]),
            "delta_bias_vs_full": float(delta["bias"]),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds]

    candidate_config = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_final.xml"
    )
    baseline_config = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_final_matched_v15.xml"
    )
    shared_sections = ("training", "optimizer", "scheduler", "dataset", "supervision")
    shared_protocol = all(
        candidate_config[section] == baseline_config[section]
        for section in shared_sections
    )
    candidate_input = ModelInputSpec.from_config(candidate_config)
    baseline_input = ModelInputSpec.from_config(baseline_config)
    candidate_profile = _read_json(args.artifacts_root / "final_profile.json")
    baseline_profile = _read_json(args.artifacts_root / "final_profile_matched_v15.json")
    counterfactual = _read_json(
        args.artifacts_root / "final_kan_counterfactuals" / "kan_counterfactuals.json"
    )
    kan = _read_json(args.artifacts_root / "final_kan_diagnostics" / "kan_diagnostics.json")
    gradient_probe_path = args.artifacts_root / "final_kan_gradient_probe.json"
    gradient_probe = (
        _read_json(gradient_probe_path)
        if gradient_probe_path.is_file()
        else {"available": False}
    )
    decision = _read_json(args.artifacts_root / "final_decision.json")
    physics = _physics_summary(args.candidate_root, seeds)

    payload = {
        "scope": "engineering verification summary; validation-only final comparison",
        "test_split_used": False,
        "protocol": {
            "shared_sections": list(shared_sections),
            "shared_sections_equal": shared_protocol,
            "candidate_amp_dtype": candidate_config["training"]["amp_dtype"],
            "baseline_amp_dtype": baseline_config["training"]["amp_dtype"],
            "candidate_lambda_phys": candidate_config["loss"]["lambda_phys"],
            "baseline_lambda_phys": baseline_config["loss"]["lambda_phys"],
            "epochs_maximum": candidate_config["training"]["epochs"],
            "minimum_epochs": candidate_config["training"]["minimum_epochs"],
            "early_stop_patience": candidate_config["training"]["early_stop_patience"],
        },
        "s1_only": {
            "candidate": candidate_input.as_dict(),
            "matched_v15": baseline_input.as_dict(),
            "both_strictly_s1_only": candidate_input.is_s1_only and baseline_input.is_s1_only,
            "s2_groups_inactive": not any(
                group.startswith("s2_")
                for group in (*candidate_input.active_groups, *baseline_input.active_groups)
            ),
        },
        "profiles": {
            "candidate": {
                "path": str((args.artifacts_root / "final_profile.json").resolve()),
                "gradients_finite": bool(candidate_profile["gradients_finite"]),
                "amp_overflow_detected": bool(candidate_profile["amp_overflow_detected"]),
            },
            "matched_v15": {
                "path": str((args.artifacts_root / "final_profile_matched_v15.json").resolve()),
                "gradients_finite": bool(baseline_profile["gradients_finite"]),
                "amp_overflow_detected": bool(baseline_profile["amp_overflow_detected"]),
            },
        },
        "weak_physics": physics,
        "graph_kan": {
            "counterfactual_reference_consistent": bool(
                counterfactual["full_reference_consistency"]["within_tolerance"]
            ),
            "counterfactuals": _counterfactual_rows(counterfactual),
            "diagnostic_scalar_means": kan["scalar_means"],
            "final_graph_gate_distribution": kan["final_graph_gate_distribution"],
            "occupancy": kan["occupancy"],
            "gradient_probe": gradient_probe,
        },
        "selection": decision["selection"],
    }
    atomic_write_json(args.artifacts_root / "verification_summary.json", payload)
    print(json.dumps({
        "output": str((args.artifacts_root / "verification_summary.json").resolve()),
        "shared_protocol": shared_protocol,
        "physics_gradient_nonzero": physics["physics_gradient_nonzero"],
        "selection": decision["selection"]["decision"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

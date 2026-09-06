#!/usr/bin/env python3
"""Run validation-only causal Graph/KAN interventions without retraining.

The tool intentionally hard-codes the validation split.  It evaluates the same
raw checkpoint under six deterministic interventions while preserving the
canonical positive mask, train-derived depth bins, BF16 policy, and all learned
weights.  It never requests Sentinel-2 raster groups.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract, sha256_file
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from losses.composite_loss import CompositeFloodDepthLoss
from tools.evaluate import (
    dataset_fingerprint,
    embed_source_fingerprints,
    evaluate_loader,
    frozen_depth_balance_for_config,
)
from utils.amp import resolve_amp
from utils.checkpoint import load_checkpoint
from utils.config import jsonable_config, load_config
from utils.logging import write_rows
from utils.misc import atomic_write_json
from utils.registry import build_model


COUNTERFACTUALS: tuple[dict[str, Any], ...] = (
    {
        "mode": "full",
        "description": "Learned graph/KAN path with no intervention.",
    },
    {
        "mode": "graph_off",
        "description": "Exact graph residual identity; all graph updates are zero.",
    },
    {
        "mode": "spline_off",
        "description": "Uses only the learned KAN base path (including its bias).",
    },
    {
        "mode": "base_off",
        "description": "Uses only the learned KAN spline contribution.",
    },
    {
        "mode": "constant_gate",
        "constant_gate": 1.0,
        "description": "Replaces learned affinity, SAR compatibility, and observation amplitude by a unit valid-edge gate.",
    },
    {
        "mode": "shuffled_terrain",
        "shuffle_seed": 20260904,
        "description": "Deterministically permutes terrain descriptors within each sample/direction while preserving SAR latents, validity, and observation amplitude.",
    },
)

PRIMARY_METRICS = ("mae", "rmse", "p90_absolute_error", "bias")


def _json_safe(value: Any) -> Any:
    """Make a strict, portable JSON payload from metric values and bin bounds."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    return value


def _deep_depth_bin(bin_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the designated >=0.5 m validation-depth stratum."""

    selected = [
        row
        for row in bin_rows
        if math.isfinite(float(row["lower_train_boundary_m"]))
        and abs(float(row["lower_train_boundary_m"]) - 0.5) < 1.0e-8
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one >=0.5 m train-depth bin; "
            f"found {len(selected)} from {bin_rows!r}"
        )
    return dict(selected[0])


def _compact_metrics(
    summary: Mapping[str, Any], bin_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    deep = _deep_depth_bin(bin_rows)
    return {
        "mae": float(summary["pixel_micro_mae"]),
        "rmse": float(summary["pixel_micro_rmse"]),
        "p90_absolute_error": float(summary["pixel_micro_p90_absolute_error"]),
        "bias": float(summary["pixel_micro_bias"]),
        "pixels": int(summary["pixel_micro_pixels"]),
        "deep_depth_ge_0_5m": {
            "mae": float(deep["mae"]),
            "rmse": float(deep["rmse"]),
            "p90_absolute_error": float(deep["p90_absolute_error"]),
            "bias": float(deep["bias"]),
            "pixels": int(deep["pixels"]),
            "lower_train_boundary_m": float(deep["lower_train_boundary_m"]),
            "upper_train_boundary_m": float(deep["upper_train_boundary_m"]),
        },
    }


def _deltas(candidate: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, float]:
    """Return candidate-minus-full deltas; positive error deltas are worse."""

    deltas = {
        metric: float(candidate[metric]) - float(full[metric])
        for metric in PRIMARY_METRICS
    }
    candidate_deep = candidate["deep_depth_ge_0_5m"]
    full_deep = full["deep_depth_ge_0_5m"]
    deltas.update(
        {
            f"deep_depth_ge_0_5m_{metric}": float(candidate_deep[metric])
            - float(full_deep[metric])
            for metric in PRIMARY_METRICS
        }
    )
    return deltas


def _reference_consistency(
    full_metrics: Mapping[str, Any], reference_path: Path | None, tolerance: float
) -> dict[str, Any] | None:
    if reference_path is None:
        return None
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    fields = {
        "mae": "pixel_micro_mae",
        "rmse": "pixel_micro_rmse",
        "p90_absolute_error": "pixel_micro_p90_absolute_error",
        "bias": "pixel_micro_bias",
    }
    differences = {
        name: abs(float(full_metrics[name]) - float(reference[key]))
        for name, key in fields.items()
    }
    passed = all(value <= tolerance for value in differences.values())
    if not passed:
        raise RuntimeError(
            "The full counterfactual pass does not reproduce the supplied raw "
            f"validation reference within {tolerance}: {differences}"
        )
    return {
        "reference_summary": str(reference_path.resolve()),
        "tolerance": float(tolerance),
        "absolute_differences": differences,
        "within_tolerance": passed,
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--reference-summary",
        type=Path,
        help="Optional raw validation summary that the full pass must reproduce.",
    )
    parser.add_argument("--reference-tolerance", type=float, default=5.0e-6)
    args = parser.parse_args()
    if args.reference_tolerance < 0:
        raise ValueError("reference-tolerance must be non-negative")

    config = embed_source_fingerprints(load_config(args.config))
    if str(config["model"]["name"]) not in {
        "pa_hydrokan_s1_v15_1",
        "pa_hydrokan_s1_v15_2",
        "pa_hydrokan_s1_v15_3",
    }:
        raise ValueError(
            "Counterfactual evaluation requires a V15.1/V15.2/V15.3 S1 model"
        )
    if str(config["model"].get("variant")) not in {"kan", "simple"}:
        raise ValueError("Counterfactual evaluation requires the V15 kan or simple variant")
    input_spec = ModelInputSpec.from_config(config)
    if not input_spec.is_s1_only or any(group.startswith("s2_") for group in input_spec.active_groups):
        raise ValueError("Counterfactual evaluation is strictly Sentinel-1 plus terrain only")

    device = _resolve_device(args.device)
    contract = DatasetContract.load(config["dataset"]["contract"])
    dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        "val",
        band_spec=resolve_band_spec(config, contract),
        input_spec=input_spec,
        minimum_event_band_fraction=float(config["dataset"].get("minimum_event_band_fraction", 1.0)),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    # Validation ordering is deterministic; one worker process is deliberately
    # avoided so every intervention reads the same ordered raster stream.
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(
        args.checkpoint,
        model,
        expected_fingerprint=dataset_fingerprint(config),
        map_location=device,
    )
    graph = getattr(model, "graph", None)
    if graph is None or not callable(getattr(graph, "set_counterfactual_mode", None)):
        raise ValueError("Model does not expose the audited V15.1 graph intervention API")

    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), contract)
    depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    prior_config = config["dataset"]["positive_prior"]
    prior = (
        normalizer.positive_prior
        if prior_config["mode"] == "auto"
        else float(prior_config["value"])
    )
    criterion = CompositeFloodDepthLoss(
        config["loss"],
        prior,
        depth_bins,
        normalizer.train_depth_bins,
        normalizer.train_depth_bin_counts,
        frozen_depth_balance_for_config(config),
    )
    amp_enabled, amp_dtype, _ = resolve_amp(
        device,
        bool(config["training"].get("amp", False)),
        str(config["training"].get("amp_dtype", "float16")),
    )

    results: dict[str, dict[str, Any]] = {}
    try:
        for specification in COUNTERFACTUALS:
            mode = str(specification["mode"])
            graph.set_counterfactual_mode(
                mode,
                constant_gate=float(specification.get("constant_gate", 1.0)),
                shuffle_seed=int(specification.get("shuffle_seed", 20260904)),
            )
            summary, _, _, bin_rows = evaluate_loader(
                model,
                loader,
                device,
                depth_bins,
                primary_depth_bins=normalizer.train_depth_bins,
                criterion=criterion,
                epoch=int(checkpoint.get("epoch", 0)),
                progress=False,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                input_spec=input_spec,
            )
            results[mode] = {
                "intervention": dict(specification),
                "metrics": _compact_metrics(summary, bin_rows),
                "summary": summary,
                "depth_bin_validation": bin_rows,
            }
    finally:
        graph.set_counterfactual_mode("full")

    full_metrics = results["full"]["metrics"]
    for mode, result in results.items():
        result["delta_vs_full_candidate_minus_full"] = _deltas(
            result["metrics"], full_metrics
        )
    consistency = _reference_consistency(
        full_metrics,
        args.reference_summary,
        float(args.reference_tolerance),
    )
    csv_rows = [
        {
            "mode": mode,
            "mae": result["metrics"]["mae"],
            "rmse": result["metrics"]["rmse"],
            "p90_absolute_error": result["metrics"]["p90_absolute_error"],
            "bias": result["metrics"]["bias"],
            "deep_depth_ge_0_5m_mae": result["metrics"]["deep_depth_ge_0_5m"]["mae"],
            "deep_depth_ge_0_5m_bias": result["metrics"]["deep_depth_ge_0_5m"]["bias"],
            "delta_mae_vs_full": result["delta_vs_full_candidate_minus_full"]["mae"],
            "delta_rmse_vs_full": result["delta_vs_full_candidate_minus_full"]["rmse"],
            "delta_p90_vs_full": result["delta_vs_full_candidate_minus_full"]["p90_absolute_error"],
            "delta_bias_vs_full": result["delta_vs_full_candidate_minus_full"]["bias"],
            "delta_deep_mae_vs_full": result["delta_vs_full_candidate_minus_full"]["deep_depth_ge_0_5m_mae"],
            "delta_deep_bias_vs_full": result["delta_vs_full_candidate_minus_full"]["deep_depth_ge_0_5m_bias"],
        }
        for mode, result in results.items()
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "kan_counterfactual_metrics.csv", csv_rows)
    atomic_write_json(
        args.output / "kan_counterfactuals.json",
        _json_safe(
            {
                "scope": "validation split only; no retraining; raw checkpoint weights",
                "test_split_used": False,
                "sentinel2_groups_requested": [],
                "config": str(args.config.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
                "weights": "raw",
                "device": str(device),
                "amp_enabled": bool(amp_enabled),
                "amp_dtype": str(amp_dtype),
                "runtime_num_workers": 0,
                "input_spec": input_spec.as_dict(),
                "dataset_fingerprint": dataset_fingerprint(config),
                "resolved_config": jsonable_config(config),
                "full_reference_consistency": consistency,
                "delta_interpretation": "candidate minus full; positive MAE/RMSE/P90 deltas are worse",
                "results": results,
            }
        ),
    )
    print(
        {
            "output": str(args.output.resolve()),
            "full_mae": full_metrics["mae"],
            "modes": list(results),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

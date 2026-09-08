#!/usr/bin/env python3
"""Evaluate any configured model through one XML-driven entry point.

Usage:
    python evaluate.py configs/pa_hydrokan.xml

The ``runtime.evaluation`` section selects the split, checkpoint, output, and
optional prediction export.  If no checkpoint is specified, the selected
``best_raw.pth`` from the same XML run tag is used.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools._neural_regression import run_evaluation as run_learned_evaluation
from tools._run_terrain_baseline import run_model as run_traditional_evaluation
from tools.evaluate_pa_hydrokan import run_evaluation as run_pa_hydrokan_evaluation
from utils.config import load_config
from utils.model_dispatch import configured_model_kind
from utils.run_paths import (
    RuntimeConfigError,
    allow_existing_output,
    ensure_output_is_available,
    evaluation_checkpoint_path,
    evaluation_output_path,
    evaluation_split,
    optional_nonnegative_int,
    runtime_section,
)


def _checkpoint(config: Mapping[str, Any]) -> Path:
    path = evaluation_checkpoint_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"configured evaluation checkpoint does not exist: {path}. "
            "Run python train.py <config.xml> first, or set runtime.evaluation.checkpoint."
        )
    return path


def run_from_config(config_path: Path) -> dict[str, Any]:
    """Dispatch evaluation based solely on the selected XML configuration."""

    path = config_path.expanduser().resolve(strict=True)
    config = load_config(path)
    kind, identifier = configured_model_kind(config)
    evaluation = runtime_section(config, "evaluation")
    output = evaluation_output_path(config)
    ensure_output_is_available(
        output,
        allow_existing=allow_existing_output(config, "evaluation"),
        operation="evaluation",
    )
    split = evaluation_split(config)
    max_batches = optional_nonnegative_int(
        evaluation.get("max_batches"), "runtime.evaluation.max_batches"
    )
    if kind == "traditional":
        return run_traditional_evaluation(
            config,
            split,
            identifier,
            output,
            max_batches,
            bool(evaluation.get("save_predictions", False)),
        )

    checkpoint = _checkpoint(config)
    if kind == "pa_hydrokan":
        weights = str(evaluation.get("weights", "raw"))
        if weights not in {"raw", "ema"}:
            raise RuntimeConfigError("runtime.evaluation.weights must be 'raw' or 'ema'")
        validity_mask = evaluation.get("validity_mask")
        return run_pa_hydrokan_evaluation(
            path,
            checkpoint,
            split,
            str(config.get("device", "auto")),
            output,
            bool(evaluation.get("save_predictions", False)),
            max_batches,
            weights,
            None if validity_mask is None else str(validity_mask),
            None,
        )

    if str(evaluation.get("weights", "raw")) != "raw":
        raise RuntimeConfigError("learned comparator evaluation supports only raw weights")
    return run_learned_evaluation(
        Namespace(
            config=path,
            checkpoint=checkpoint,
            split=split,
            device=str(config.get("device", "auto")),
            num_workers=None,
            max_batches=max_batches,
            output=output,
        ),
        identifier,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Model XML configuration.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(run_from_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train any configured model through one XML-driven entry point.

Usage:
    python train.py configs/pa_hydrokan.xml

The XML selects the model, data contract, hyperparameters, run tag, and output
directory.  Traditional methods have no fitting stage, so their invocation runs
the configured deterministic validation/test evaluation instead.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools._neural_regression import run_training as run_learned_training
from tools._run_terrain_baseline import run_model as run_traditional_evaluation
from tools.train_pa_hydrokan import run_training as run_pa_hydrokan_training
from utils.config import load_config
from utils.model_dispatch import configured_model_kind
from utils.run_paths import (
    RuntimeConfigError,
    allow_existing_output,
    ensure_output_is_available,
    evaluation_output_path,
    evaluation_split,
    optional_nonnegative_int,
    optional_path,
    runtime_section,
    train_output_path,
)


def _pa_training_args(config_path: Path, config: Mapping[str, Any], output: Path) -> Namespace:
    train = runtime_section(config, "train")
    init_weights = str(train.get("init_weights", "raw"))
    if init_weights not in {"raw", "ema"}:
        raise RuntimeConfigError("runtime.train.init_weights must be 'raw' or 'ema'")
    return Namespace(
        config=config_path,
        device=None,
        epochs=None,
        batch_size=None,
        num_workers=None,
        resume=optional_path(train.get("resume"), "runtime.train.resume"),
        init_checkpoint=optional_path(
            train.get("init_checkpoint"), "runtime.train.init_checkpoint"
        ),
        init_weights=init_weights,
        max_train_batches=optional_nonnegative_int(
            train.get("max_train_batches"), "runtime.train.max_train_batches"
        ),
        max_val_batches=optional_nonnegative_int(
            train.get("max_val_batches"), "runtime.train.max_val_batches"
        ),
        no_amp=False,
        seed=None,
        allow_fingerprint_mismatch=bool(
            train.get("allow_fingerprint_mismatch", False)
        ),
        output=output,
    )


def _learned_training_args(config_path: Path, config: Mapping[str, Any], output: Path) -> Namespace:
    train = runtime_section(config, "train")
    unsupported = [
        name
        for name in ("resume", "init_checkpoint")
        if optional_path(train.get(name), f"runtime.train.{name}") is not None
    ]
    if unsupported:
        raise RuntimeConfigError(
            "learned comparison training does not implement checkpoint continuation; "
            f"remove {unsupported} from the configuration"
        )
    return Namespace(
        config=config_path,
        device=None,
        epochs=None,
        batch_size=None,
        num_workers=None,
        max_train_batches=optional_nonnegative_int(
            train.get("max_train_batches"), "runtime.train.max_train_batches"
        ),
        max_val_batches=optional_nonnegative_int(
            train.get("max_val_batches"), "runtime.train.max_val_batches"
        ),
        no_amp=False,
        seed=None,
        output=output,
    )


def run_from_config(config_path: Path) -> Path | dict[str, Any]:
    """Dispatch one XML configuration without accepting model-specific CLI options."""

    path = config_path.expanduser().resolve(strict=True)
    config = load_config(path)
    kind, identifier = configured_model_kind(config)
    if kind == "traditional":
        output = evaluation_output_path(config)
        ensure_output_is_available(
            output,
            allow_existing=allow_existing_output(config, "evaluation"),
            operation="traditional evaluation",
        )
        return run_traditional_evaluation(
            config,
            evaluation_split(config),
            identifier,
            output,
            optional_nonnegative_int(
                runtime_section(config, "evaluation").get("max_batches"),
                "runtime.evaluation.max_batches",
            ),
            bool(runtime_section(config, "evaluation").get("save_predictions", False)),
        )

    output = train_output_path(config)
    train = runtime_section(config, "train")
    resume = optional_path(train.get("resume"), "runtime.train.resume")
    if resume is not None:
        if output != resume.resolve().parent:
            raise RuntimeConfigError(
                "runtime.train.output/default path must equal the parent directory of "
                "runtime.train.resume"
            )
    else:
        ensure_output_is_available(
            output,
            allow_existing=allow_existing_output(config, "train"),
            operation="training",
        )
    if kind == "pa_hydrokan":
        return run_pa_hydrokan_training(_pa_training_args(path, config, output))
    return run_learned_training(_learned_training_args(path, config, output), identifier)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Model XML configuration.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_from_config(args.config)
    if isinstance(result, Path):
        print(f"training output: {result}")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

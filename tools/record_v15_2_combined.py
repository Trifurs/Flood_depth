#!/usr/bin/env python3
"""Record the non-duplicated Experiment-6 combination selection.

When the selected KAN and Graph are already the controls used by a winning
physics candidate, Experiment 6 introduces no new training variable.  This
tool records that exact alias transparently instead of spending a second,
deterministically identical 45-epoch run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.model_input_spec import ModelInputSpec
from utils.config import load_config
from utils.misc import atomic_write_json


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--alias-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    spec = ModelInputSpec.from_config(config)
    model = config["model"]
    loss = config["loss"]
    if not spec.is_s1_only:
        raise RuntimeError("combined model must remain S1+terrain only")
    if not bool(model["graph_enabled"]) or int(model["kan_grid_size"]) <= 0:
        raise RuntimeError("combined model must retain Graph and KAN")
    if float(loss.get("lambda_phys", 0.0)) <= 0.0:
        raise RuntimeError("combined model must retain a nonzero weak physics loss")
    raw = _read_json(args.source_run / "eval_raw" / "summary.json")
    ema = _read_json(args.source_run / "eval_ema" / "summary.json")
    payload = {
        "scope": "validation-only Experiment 6 selection; no test split used",
        "decision": "alias_selected_physics_order_run",
        "reason": (
            "The original KAN and Graph-A skip-only path won their respective "
            "screening decisions, and Physics-A was trained on that exact pair. "
            "A new 45-epoch run would be an identical deterministic duplicate, "
            "not an additional controlled variable."
        ),
        "source_run": str(args.source_run.resolve()),
        "combined_config": str(args.config.resolve()),
        "combined_run_alias": str(args.alias_run.resolve()),
        "structure": {
            "s1_only": spec.as_dict(),
            "graph_enabled": bool(model["graph_enabled"]),
            "graph_feature_stride": int(model["graph_feature_stride"]),
            "kan_grid_size": int(model["kan_grid_size"]),
            "kan_spline_order": int(model["kan_spline_order"]),
            "physics_mode": str(loss["physics_mode"]),
            "lambda_phys": float(loss["lambda_phys"]),
        },
        "source_raw_validation": {
            key: raw[key]
            for key in (
                "checkpoint_epoch",
                "pixel_micro_mae",
                "pixel_micro_rmse",
                "pixel_micro_p90_absolute_error",
                "pixel_micro_bias",
            )
        },
        "source_ema_validation": {
            key: ema[key]
            for key in (
                "checkpoint_epoch",
                "pixel_micro_mae",
                "pixel_micro_rmse",
                "pixel_micro_p90_absolute_error",
                "pixel_micro_bias",
            )
        },
    }
    args.alias_run.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.alias_run / "selection.json", payload)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

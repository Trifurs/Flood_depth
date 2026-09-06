#!/usr/bin/env python3
"""Archive the non-comparable V15/V15.1 context before V15.2 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import jsonable_config, load_config
from utils.misc import atomic_write_json


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _candidate_by_name(summary: dict[str, object], name: str) -> dict[str, object]:
    candidates = summary.get("candidates", [])
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("name") == name:
            return candidate
    raise KeyError(f"candidate {name!r} not found")


def _training_controls(path: Path) -> dict[str, object]:
    config = load_config(path)
    training = config["training"]
    return {
        "config": str(path),
        "seed": int(config["seed"]),
        "amp_dtype": str(training.get("amp_dtype", "float16")),
        "batch_size": int(training["batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "epochs": int(training["epochs"]),
        "minimum_epochs": int(training.get("minimum_epochs", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    historical_path = PROJECT_ROOT / "artifacts/optimization/hydrokan_s1_v15_1/final_decision.json"
    candidates_path = PROJECT_ROOT / "artifacts/optimization/hydrokan_s1_v15_1/candidate_summary.json"
    historical = _read_json(historical_path)
    summary = _read_json(candidates_path)
    historical_baseline = historical["historical_baseline"]
    if not isinstance(historical_baseline, dict):
        raise TypeError("historical_baseline must be an object")
    simple = _candidate_by_name(summary, "V15.1-simple")
    v15_config = PROJECT_ROOT / str(historical_baseline["config"])
    simple_config = PROJECT_ROOT / str(simple["config"])
    output = {
        "scope": "pre-V15.2 context only; not a strict V15.2 comparison baseline",
        "selection_scope_for_v15_2": "validation_only",
        "test_split_used_by_v15_2": False,
        "source_artifacts": {
            "v15_1_final_decision": str(historical_path),
            "v15_1_candidate_summary": str(candidates_path),
            "v15_1_report": str(PROJECT_ROOT / "docs/HYDROKAN_S1_V15_1_ENGINEERING_REPORT.md"),
        },
        "historical_v15": {
            "name": historical_baseline["name"],
            "checkpoint": historical_baseline["checkpoint"],
            "checkpoint_epoch_zero_based": historical_baseline["checkpoint_epoch_zero_based"],
            "canonical_validation_metrics": historical_baseline["canonical_metrics"],
            "training_controls": _training_controls(v15_config),
        },
        "v15_1_simple": {
            "run_dir": simple["run_dir"],
            "raw": simple["raw"],
            "ema": simple["ema"],
            "matched_epoch": simple["matched_epoch"],
            "training_controls": _training_controls(simple_config),
        },
        "why_historical_v15_is_not_the_strict_baseline": [
            "Historical V15 used a different seed and FP16 precision.",
            "Historical V15 and V15.1 used batch size 8 with gradient accumulation 2, while V15.2 must select a common real batch.",
            "The V15.2 code contains correctness fixes that change the effective training protocol.",
            "V15.2 comparisons therefore begin with a newly trained matched V15 validation baseline.",
        ],
        "v15_2_protocol_target": jsonable_config(
            _training_controls(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_matched_v15.xml")
        ),
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

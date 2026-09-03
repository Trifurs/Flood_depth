#!/usr/bin/env python3
"""Build auditable final-decision and profile manifests from completed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.misc import atomic_write_json


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/optimization/hydrov14"))
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    ablation = read_json(output_dir / "ablation_summary.json")
    sensor = read_json(output_dir / "sensor_ablation_summary.json")
    physics = read_json(output_dir / "physics_diagnostics.json")
    profiles = {}
    for name in ("full_inputs_profile", "no_s2_profile"):
        path = output_dir / "controlled" / f"{name}.json"
        if path.exists():
            profiles[name] = read_json(path)

    rows = {row["candidate"]: row for row in ablation["rows"]}
    baseline = rows.get("corrected_v13_baseline", {})
    full = rows.get("full_inputs_3epoch", {})
    no_s2 = rows.get("no_s2_3epoch", {})
    decision = {
        "schema_version": "hydrov14.final_decision.v1",
        "dataset": "subset1000",
        "validation_only": True,
        "test_split_evaluated": False,
        "selected_band_spec": "configs/pa_hydrokan/subset1000_v14_bands_selected.xml",
        "default_sensor_choice": "full_inputs",
        "default_model_config": "configs/pa_hydrokan/subset1000_v14_bands_selected.xml",
        "experimental_full_config": "configs/pa_hydrokan/subset1000_v14_final.xml",
        "claims": {
            "exceeds_corrected_v13": False,
            "kan_independent_value_established": False,
            "physics_effectiveness_established": False,
            "no_s2_promoted": bool(sensor["decision"]["promote_no_s2"]),
        },
        "evidence": {
            "corrected_v13_pixel_micro_mae": baseline.get("pixel_micro_mae"),
            "full_inputs_short_budget_pixel_micro_mae": full.get("pixel_micro_mae"),
            "no_s2_short_budget_pixel_micro_mae": no_s2.get("pixel_micro_mae"),
            "screen_rows": [row for row in ablation["rows"] if row["candidate"] in {
                "no_graph", "edge_mlp", "edge_kan_scale4", "edge_kan_scale8", "terrain_order", "wse_slope"
            }],
        },
        "reasoning": [
            "The 3-epoch/5-batch controlled run is exploratory and is not a like-for-like full-budget superiority claim against the historical corrected-v13 checkpoint.",
            "The one-epoch screen activates neither delayed physics terms nor a meaningful learned KAN residual; it cannot establish component superiority.",
            "Full-input is retained because no-S2 improves P90/RMSE/bias slightly but worsens the primary pixel-micro MAE and sample/event macro metrics on complete validation.",
        ],
        "artifact_references": {
            "ablation_summary": "artifacts/optimization/hydrov14/ablation_summary.json",
            "sensor_ablation_summary": "artifacts/optimization/hydrov14/sensor_ablation_summary.json",
            "physics_diagnostics": "artifacts/optimization/hydrov14/physics_diagnostics.json",
            "kan_diagnostics": "artifacts/optimization/hydrov14/kan_diagnostics.json",
        },
    }
    atomic_write_json(output_dir / "final_decision.json", decision)
    atomic_write_json(output_dir / "final_profile.json", {
        "schema_version": "hydrov14.final_profile.v1",
        "variants": profiles,
        "profile_scope": "CPU, one profiling iteration, finite-gradient smoke profile",
        "screen_parameter_count": {name: rows[name].get("parameters") for name in rows if name in {"no_graph", "edge_mlp", "edge_kan_scale4", "edge_kan_scale8", "terrain_order", "wse_slope"}},
    })
    print(json.dumps({"decision": decision, "profile_variants": list(profiles)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

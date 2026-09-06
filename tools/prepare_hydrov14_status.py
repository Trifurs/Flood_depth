#!/usr/bin/env python3
"""Write a compact, evidence-backed Hydro-v14 pre-change status artifact."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from utils.config import load_config
from utils.misc import atomic_write_json


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"missing": str(path)}


def main() -> int:
    config_path = ROOT / "configs/pa_hydrokan/subset1000_v13_2_final.xml"
    config = load_config(config_path)
    contract = DatasetContract.load(config["dataset"]["contract"])
    spec = resolve_band_spec(config, contract)
    baseline_summary = ROOT / "artifacts/optimization/hydrov13_2_subset1000/final_v13_val_raw.json/summary.json"
    baseline_checkpoint = ROOT / "runs/optimization/hydrov13_2_subset1000/final_v13/best_raw.pth"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Hydro-v14 implementation baseline; no train/val/test split or raw data was changed",
        "commit_before_changes": "0dde046eab4b23604eb301d893c51c6a5ea6b30f",
        "git_status_before_changes": "clean",
        "git_status_at_capture": _git("status", "--short"),
        "pytest_initial": {"command": "conda run --no-capture-output -n flood-depth python -m pytest -q", "passed": 104, "failed": 0, "skipped": 2, "warnings": 3},
        "data": {
            "priority": ["/home/whu/桌面/myData/Flood_depth/subset1000", "/home/whu/桌面/myData/Flood_depth/subset150"],
            "selected": str(contract.dataset_root),
            "contract": str(contract.path),
            "contract_sha256": contract.hash,
            "manifest": str(contract.manifest_path),
            "manifest_sha256": contract.payload["manifest"]["sha256"],
            "sample_counts": contract.payload["sample_counts"],
            "manifest_counts": contract.payload["manifest"]["split_counts"],
            "unique_event_chains": contract.payload["manifest"].get("unique_event_chains"),
            "unique_source_events": contract.payload["manifest"].get("unique_source_events"),
            "normalization": contract.payload["normalization"].get("selected"),
        },
        "baseline": {
            "label": "corrected-v13 / v13.2 subset1000 final raw validation re-evaluation",
            "config": str(config_path),
            "checkpoint": str(baseline_checkpoint),
            "summary": str(baseline_summary),
            "metrics": _json(baseline_summary),
        },
        "band_spec": spec.as_dict(),
        "scientific_semantics": {
            "depth_output": "conditional_positive_v2",
            "dsm": "DSM / ground-like terrain proxy; not bare-earth DTM, riverbed elevation, or hydraulic terrain",
            "model_scope": "continuous positive flood-depth estimation; not flood extent segmentation",
            "hydrodynamics_claim": "not a PINN, shallow-water solver, flow-velocity model, or mass-conservation model",
        },
    }
    output = ROOT / "artifacts/optimization/hydrov14/prechange_status.json"
    atomic_write_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

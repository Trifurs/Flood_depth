#!/usr/bin/env python3
"""Capture the S1-only experiment baseline and implementation identity."""

from __future__ import annotations

from datetime import datetime, timezone
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec
from datasets.reliability_spec import ReliabilitySpec
from utils.config import load_config
from utils.misc import atomic_write_json


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    config_path = ROOT / "configs/pa_hydrokan/subset1000_s1_v14_final.xml"
    config = load_config(config_path)
    contract = DatasetContract.load(config["dataset"]["contract"])
    input_spec = ModelInputSpec.from_config(config)
    band_spec = resolve_band_spec(config, contract)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "PA-HydroKAN-S1-v14 optical-free implementation baseline",
        "base_commit_before_s1_changes": "3ca4e40cc428f6a5f6483f5407636d32aaa0d442",
        "git_head_at_capture": _git("rev-parse", "HEAD"),
        "git_status_at_capture": _git("status", "--short"),
        "pytest_before_s1_model": {
            "command": "conda run -n flood-depth python -m pytest -q",
            "passed": 116,
            "failed": 0,
            "skipped": 2,
        },
        "data": {
            "root": str(contract.dataset_root),
            "contract": str(contract.path),
            "contract_sha256": contract.hash,
            "manifest": str(contract.manifest_path),
            "manifest_sha256": contract.payload["manifest"]["sha256"],
            "sample_counts": contract.payload["sample_counts"],
            "normalization": contract.payload["normalization"]["selected"],
        },
        "input_spec": input_spec.as_dict(),
        "input_spec_sha256": input_spec.sha256,
        "active_groups_sha256": input_spec.active_groups_sha256,
        "reliability_spec": ReliabilitySpec.from_mode(input_spec.mode).as_dict(),
        "resolved_model_bands": band_spec.as_dict(),
        "scientific_scope": {
            "sensor": "Sentinel-1 SAR only",
            "terrain": "DSM / ground-like terrain proxy; not a bare-earth DTM",
            "graph": "static topographic affinity; not a hydraulic solver",
            "target": "continuous positive flood-depth estimation",
            "optical_input": False,
        },
    }
    output = ROOT / "artifacts/optimization/hydrokan_s1_v14/prechange_status.json"
    atomic_write_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

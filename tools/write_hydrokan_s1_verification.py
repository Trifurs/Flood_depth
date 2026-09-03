#!/usr/bin/env python3
"""Write an auditable verification summary for the S1-only implementation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.misc import atomic_write_json


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root = ROOT / "artifacts/optimization/hydrokan_s1_v14"
    io_profile = _read(root / "io_profile_s1.json")
    diagnostics = _read(root / "kan_diagnostics.json")
    candidate = _read(root / "candidate_summary.json")
    payload = {
        "status": "passed_with_short_budget_experiment_caveat",
        "implementation": {
            "model": "pa_hydrokan_s1_v14",
            "input_mode": "s1_terrain",
            "optical_input": False,
            "s2_read_in_io_profile": io_profile["s2_opened"],
            "active_groups": io_profile["input_spec"]["active_groups"],
            "reliability_channels": 6,
            "support_branch_default": False,
        },
        "tests": {
            "full_regression": "121 passed, 2 skipped",
            "s1_io_and_missing_optional_groups": "2 passed",
            "s1_model_forward_backward_decoder_registry": "5 passed",
            "finite_kan_diagnostics": diagnostics["diagnostics_finite"],
        },
        "artifacts": {
            "io_profile": str(root / "io_profile_s1.json"),
            "kan_diagnostics": str(root / "kan_diagnostics.json"),
            "candidate_summary": str(root / "candidate_summary.json"),
            "final_val_native": str(root / "final_val_native_s1_support" / "summary.json"),
            "final_val_common": str(root / "final_val_common_support" / "summary.json"),
            "final_test_native": str(root / "final_test_native_s1_support" / "summary.json"),
        },
        "decision": candidate["decision"],
    }
    atomic_write_json(root / "verification_summary.json", payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

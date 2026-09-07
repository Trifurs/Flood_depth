#!/usr/bin/env python3
"""Run the named FwDET v2.0 comparison model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools._run_terrain_baseline import main_for_model


if __name__ == "__main__":
    raise SystemExit(main_for_model("fwdet_v2", Path("configs/compare/fwdet_v2.xml")))

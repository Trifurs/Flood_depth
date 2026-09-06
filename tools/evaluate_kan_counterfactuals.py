#!/usr/bin/env python3
"""Stable public entry point for validation-only V15 KAN interventions."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_v15_2_kan_counterfactuals import main


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate the U-Net depth-regression comparison model."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools._neural_regression import evaluation_main

if __name__ == "__main__":
    raise SystemExit(evaluation_main("unet_depth_regression", "configs/unet_depth_regression.xml"))

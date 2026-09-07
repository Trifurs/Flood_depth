#!/usr/bin/env python3
"""Train the DLSIM Attention U-Net comparison model."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools._neural_regression import train_main

if __name__ == "__main__":
    raise SystemExit(
        train_main(
            "dlsim_attention_unet",
            "configs/compare/deep_learning/dlsim_attention_unet.xml",
        )
    )

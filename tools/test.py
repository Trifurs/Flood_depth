#!/usr/bin/env python3
"""Formal held-out test evaluation and georeferenced prediction export."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate import run_evaluation
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    output = args.output or Path("runs/test") / (
        f"test_{args.checkpoint.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    summary = run_evaluation(
        args.config,
        args.checkpoint,
        "test",
        args.device,
        output.resolve(),
        args.save_predictions,
        args.max_batches,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

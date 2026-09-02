#!/usr/bin/env python3
"""Summarize a few structured samples without modifying source data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from datasets.flooddepth_dataset import FloodDepthDataset
from utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-samples", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], args.split
    )
    for index in range(min(args.max_samples, len(dataset))):
        sample = dataset[index]
        print(sample["metadata"]["sample_id"])
        print(
            {
                key: (tuple(value.shape), str(value.dtype), bool(torch.isfinite(value).all()))
                for key, value in sample.items()
                if isinstance(value, torch.Tensor)
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

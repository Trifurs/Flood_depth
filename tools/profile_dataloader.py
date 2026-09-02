#!/usr/bin/env python3
"""Measure real-raster DataLoader throughput for a small worker candidate set."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from tools.evaluate import embed_source_fingerprints
from tools.train import create_dataloaders
from utils.config import load_config
from utils.misc import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args(); base = embed_source_fingerprints(load_config(args.config)); rows = []
    for workers in (0, 2, 4, 8):
        config = dict(base); config["training"] = dict(base["training"])
        config["training"]["num_workers"] = workers; config["training"]["persistent_workers"] = workers > 0
        loader, _, _, _ = create_dataloaders(config); start = time.perf_counter(); samples = batches = 0
        for batch in loader:
            samples += int(batch["label"].shape[0]); batches += 1
            if batches >= args.batches: break
        elapsed = time.perf_counter() - start
        rows.append({"num_workers": workers, "batches": batches, "samples": samples,
                     "elapsed_seconds": elapsed, "samples_per_second": samples / elapsed})
        print(rows[-1])
    atomic_write_json(args.output, {"config": str(args.config.resolve()), "results": rows,
                                    "selected_num_workers": max(rows, key=lambda x: x["samples_per_second"])["num_workers"]})
    return 0
if __name__ == "__main__": raise SystemExit(main())

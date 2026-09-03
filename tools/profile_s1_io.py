#!/usr/bin/env python3
"""Record group-level I/O for the optical-free data path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.model_input_spec import ModelInputSpec
from tools.train import create_dataloaders
from utils.config import load_config
from utils.misc import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()
    config = load_config(args.config)
    spec = ModelInputSpec.from_config(config)
    train_loader, val_loader, _, _ = create_dataloaders(config)
    rows = []
    for split, loader in (("train", train_loader), ("val", val_loader)):
        started = time.perf_counter()
        group_counts: dict[str, int] = {}
        opened_files = 0
        read_bands = 0
        batches = 0
        samples = 0
        for batch in loader:
            batches += 1
            samples += int(batch["label"].shape[0])
            metadata = batch["metadata"]
            profile = metadata["io_profile"]
            # Default collation stacks scalar profile values and transposes the
            # per-group mapping into tensors indexed by sample.
            opened_files += int(profile["opened_files"].sum().item())
            read_bands += int(profile["read_bands"].sum().item())
            for group, count in profile["read_band_counts"].items():
                group_counts[group] = group_counts.get(group, 0) + int(count.sum().item())
            if batches >= args.batches:
                break
        rows.append({
            "split": split,
            "batches": batches,
            "samples": samples,
            "elapsed_seconds": time.perf_counter() - started,
            "active_groups": list(spec.active_groups),
            "opened_files": opened_files,
            "read_bands": read_bands,
            "read_band_counts": group_counts,
            "unexpected_s2_groups": sorted(group for group in group_counts if group.startswith("s2_")),
        })
    payload = {
        "config": str(args.config.resolve()),
        "input_spec": spec.as_dict(),
        "results": rows,
        "s2_opened": any(row["unexpected_s2_groups"] for row in rows),
    }
    atomic_write_json(args.output, payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

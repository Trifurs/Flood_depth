#!/usr/bin/env python3
"""Validation zero-masking sensitivity for every selected band and semantic group."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from losses.composite_loss import CompositeFloodDepthLoss
from tools.evaluate import dataset_fingerprint, embed_source_fingerprints, evaluate_loader
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.logging import write_rows
from utils.misc import atomic_write_json
from utils.registry import build_model


class MaskedInputModel(nn.Module):
    def __init__(self, model: nn.Module, masks: list[tuple[str, int]]) -> None:
        super().__init__(); self.model = model; self.masks = masks
        self.heads = getattr(model, "heads", None)
    def forward(self, inputs):
        altered = dict(inputs)
        for group, index in self.masks:
            if group not in altered: continue
            if altered[group] is inputs[group]: altered[group] = inputs[group].clone()
            altered[group][:, index].zero_()
        return self.model(altered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val"); parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); config = embed_source_fingerprints(load_config(args.config))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    contract = DatasetContract.load(config["dataset"]["contract"]); spec = resolve_band_spec(config, contract)
    dataset = FloodDepthDataset(config["dataset"]["contract"], config["dataset"]["train_stats"], args.split, band_spec=spec)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    model = build_model(config).to(device); load_checkpoint(args.checkpoint, model, expected_fingerprint=dataset_fingerprint(config), map_location=device)
    normalizer = RobustNormalizer(config["dataset"]["train_stats"], contract); bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    prior_cfg = config["dataset"]["positive_prior"]; prior = normalizer.positive_prior if prior_cfg["mode"] == "auto" else float(prior_cfg["value"])
    criterion = CompositeFloodDepthLoss(config["loss"], prior, bins, normalizer.train_depth_bins)
    candidates: list[tuple[str, list[tuple[str, int]]]] = [("baseline", [])]
    for group in ("s1_t1", "s1_t2", "s1_change", "s2_t1", "s2_t2", "s2_change", "terrain"):
        for index, name in enumerate(spec.names(group)): candidates.append((f"band:{group}:{name}", [(group, index)]))
    candidates.extend((f"group:{group}", [(group, i) for i in range(spec.channels(group))]) for group in ("s1_t1", "s1_t2", "s1_change", "s2_t1", "s2_t2", "s2_change", "terrain"))
    angle_masks = []
    for group in ("s1_t1", "s1_t2"):
        for i, name in enumerate(spec.names(group)):
            if name in {"angle_pre_deg", "angle_event_deg"}: angle_masks.append((group, i))
    candidates.append(("group:incidence_conditioning", angle_masks))
    rows, details = [], {}
    for name, masks in candidates:
        summary, _, _, bin_rows = evaluate_loader(MaskedInputModel(model, masks), loader, device, bins,
            primary_depth_bins=normalizer.train_depth_bins, criterion=criterion, progress=False,
            amp_enabled=False)
        row = {"candidate": name, "masked_channels": len(masks), "pixel_micro_mae": summary["pixel_micro_mae"],
               "pixel_micro_rmse": summary["pixel_micro_rmse"], "pixel_micro_p90_absolute_error": summary["pixel_micro_p90_absolute_error"]}
        if rows:
            row["mae_delta"] = row["pixel_micro_mae"] - rows[0]["pixel_micro_mae"]
        else: row["mae_delta"] = 0.0
        rows.append(row); details[name] = {"summary": summary, "train_depth_bins": bin_rows, "masks": masks}
        print(name, row["pixel_micro_mae"], row["mae_delta"])
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "band_mask_importance.csv", rows)
    atomic_write_json(args.output / "band_mask_importance.json", {"rows": rows, "details": details})
    return 0


if __name__ == "__main__": raise SystemExit(main())

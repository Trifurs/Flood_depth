#!/usr/bin/env python3
"""Validation-only semantic band/group masking diagnostics.

The tool is a sensitivity diagnostic.  It never uses test data and never treats a
zero in normalized space as a physical zero: zero means the train-centered typical
value used for masking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from losses.composite_loss import CompositeFloodDepthLoss
from tools.evaluate import dataset_fingerprint, embed_source_fingerprints, evaluate_loader
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.logging import write_rows
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model


class MaskedInputModel(nn.Module):
    def __init__(self, model: nn.Module, masks: list[tuple[str, int]]) -> None:
        super().__init__()
        self.model, self.masks = model, masks
        self.heads = getattr(model, "heads", None)

    def forward(self, inputs):
        altered = dict(inputs)
        for group, index in self.masks:
            if group not in altered:
                continue
            altered[group] = altered[group].clone()
            altered[group][:, index].zero_()
        return self.model(altered)


def _q_thresholds() -> tuple[float, float, str]:
    path = ROOT / "artifacts/optimization/hydrov14/bands/train_depth_statistics.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload["p90"]), float(payload["p95"]), str(path)
    raise RuntimeError("Run build_hydrov14_depth_stats.py before deep-water diagnostics")


@torch.no_grad()
def _deep_metrics(model, loader, device, q90: float, q95: float) -> dict[str, float]:
    errors90, errors95 = [], []
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        output = model(prepare_model_inputs(batch))["conditional_depth"]
        error = (output - batch["label"]).detach().float()
        positive = batch["masks"]["valid_depth_mask"] > 0.5
        for threshold, target_list in ((q90, errors90), (q95, errors95)):
            selected = positive & (batch["label"] >= threshold)
            if torch.any(selected):
                target_list.extend(error[selected].cpu().tolist())
    def summarize(values: list[float], suffix: str) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {f"train_{suffix}_mae": float(np.abs(array).mean()) if array.size else float("nan"), f"train_{suffix}_bias": float(array.mean()) if array.size else float("nan"), f"train_{suffix}_pixels": float(array.size)}
    return summarize(errors90, "q90_deep") | summarize(errors95, "q95_deep")


def _default_groups(spec) -> dict[str, list[tuple[str, str]]]:
    def pair(prefix: str, pre: str, event: str, delta: str) -> list[tuple[str, str]]:
        return [("s1_t1", pre), ("s1_t2", event), ("s1_change", delta)]
    groups = {
        "S1_VV_PAIR": pair("", "VV_pre_db", "VV_event_db", "VV_delta_db"),
        "S1_VH_PAIR": pair("", "VH_pre_db", "VH_event_db", "VH_delta_db"),
        "S1_ANOMALY_RAW": [("s1_change", "anomaly_raw")],
        "S1_ANOMALY_SELECTION": [("s1_change", "anomaly_selection")],
        "S1_INCIDENCE_CONDITIONING": [("s1_conditioning", "angle_pre_deg"), ("s1_conditioning", "angle_event_deg")],
        "NDWI_DELTA": [("s2_change", "NDWI_delta")],
        "MNDWI_DELTA": [("s2_change", "MNDWI_delta")],
        "WATER_CHANGE_SELECTION": [("s2_change", "water_change_selection")],
        "DSM_ELEVATION": [("terrain", "elevation_m_DSM")],
        "INPUT_SLOPE": [("terrain", "slope_deg")],
    }
    for band in ("B2", "B3", "B4", "B8", "B11", "B12"):
        groups[f"S2_{band}_PAIR"] = [("s2_t1", f"{band}_pre_reflectance"), ("s2_t2", f"{band}_event_reflectance")]
    return groups


def _resolve_groups(path: Path | None, spec) -> dict[str, list[tuple[str, str]]]:
    if path is None:
        raw = _default_groups(spec)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("groups", payload)
    resolved = {}
    for name, values in raw.items():
        resolved[name] = [(str(item[0]), str(item[1])) for item in values]
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-file", type=Path)
    args = parser.parse_args()
    config = embed_source_fingerprints(load_config(args.config))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    contract = DatasetContract.load(config["dataset"]["contract"])
    spec = resolve_band_spec(config, contract)
    dataset = FloodDepthDataset(config["dataset"]["contract"], config["dataset"]["train_stats"], args.split, band_spec=spec)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, expected_fingerprint=dataset_fingerprint(config), map_location=device)
    normalizer = RobustNormalizer(config["dataset"]["train_stats"], contract)
    bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    prior_cfg = config["dataset"]["positive_prior"]
    prior = normalizer.positive_prior if prior_cfg["mode"] == "auto" else float(prior_cfg["value"])
    criterion = CompositeFloodDepthLoss(config["loss"], prior, bins, normalizer.train_depth_bins, normalizer.train_depth_bin_counts)
    q90, q95, threshold_source = _q_thresholds()
    candidates: list[tuple[str, list[tuple[str, int]]]] = [("baseline", [])]
    all_groups = _resolve_groups(args.groups_file, spec)
    available = {group: set(spec.names(group)) for group in spec.groups}
    available["s1_conditioning"] = set(spec.names("s1_conditioning"))
    for name, definitions in all_groups.items():
        masks = []
        missing = []
        for group, band in definitions:
            if band not in available.get(group, set()):
                missing.append(f"{group}/{band}"); continue
            masks.append((group, list(spec.names(group)).index(band)))
        if not missing:
            candidates.append((f"group:{name}", masks))
    rows, details = [], {}
    for candidate, masks in candidates:
        wrapped = MaskedInputModel(model, masks)
        summary, _, _, bin_rows = evaluate_loader(wrapped, loader, device, bins, primary_depth_bins=normalizer.train_depth_bins, criterion=criterion, progress=False, amp_enabled=False)
        deep = _deep_metrics(wrapped, loader, device, q90, q95)
        row = {"candidate": candidate, "masked_channels": len(masks), "masked_channels_names": ",".join(f"{group}:{spec.names(group)[index]}" for group, index in masks), "pixel_micro_mae": summary.get("pixel_micro_mae"), "pixel_micro_rmse": summary.get("pixel_micro_rmse"), "pixel_micro_p90_absolute_error": summary.get("pixel_micro_p90_absolute_error"), "pixel_micro_bias": summary.get("pixel_micro_bias"), "sample_macro_mae": summary.get("sample_macro_mae"), **deep}
        row["mae_delta"] = float(row["pixel_micro_mae"] - rows[0]["pixel_micro_mae"]) if rows else 0.0
        rows.append(row); details[candidate] = {"summary": summary, "train_depth_bins": bin_rows, "masks": masks, "deep_thresholds_m": {"q90": q90, "q95": q95}}
        print(candidate, row["pixel_micro_mae"], row["mae_delta"])
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "band_mask_importance.csv", rows)
    atomic_write_json(args.output / "band_mask_importance.json", {"rows": rows, "details": details, "split": "val", "normalized_mask_zero_semantics": "center/typical-value masking, not physical zero", "train_depth_threshold_source": threshold_source})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

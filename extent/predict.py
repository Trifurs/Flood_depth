#!/usr/bin/env python3
"""Freeze and export one reusable flood-extent product for depth methods."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from datasets.flooddepth_dataset import FloodDepthDataset
from extent.protocol import build_ai4g_change_features, postprocess_extent
from extent.train import build_model, dataset_fingerprint, feature_parameters
from utils.checkpoint import load_checkpoint
from utils.config import jsonable_config, load_config
from utils.logging import write_rows
from utils.misc import atomic_write_json, move_to_device
from utils.raster_io import write_geotiff
from datasets.contract import sha256_file


def _binary_metrics(tp: int, fp: int, fn: int, prefix: str) -> dict[str, float]:
    iou = tp / max(1, tp + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        f"{prefix}_iou": iou,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": 2.0 * precision * recall / max(1e-12, precision + recall),
    }


def predict_split(
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], split
    )
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        expected_fingerprint=dataset_fingerprint(config),
        map_location=device,
        adopt_checkpoint_output_semantics=False,
    )
    checkpoint_config = checkpoint.get("resolved_config", {})
    if checkpoint_config.get("extent_model") != jsonable_config(config["extent_model"]):
        raise RuntimeError("Extent checkpoint/config mismatch")
    model.eval()
    rows: list[dict[str, Any]] = []
    raw_tp = raw_fp = raw_fn = 0
    buffered_tp = buffered_fp = buffered_fn = 0
    total_positive = 0
    total_eligible = total_raw = total_buffered = 0
    split_dir = output_root / split
    with torch.no_grad():
        for sample in tqdm(dataset, desc=f"extent-predict-{split}"):
            batch = move_to_device(
                {
                    key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
                    for key, value in sample.items()
                },
                device,
            )
            # Nested tensors require an explicit batch dimension as well.
            for namespace in ("extent_inputs", "validity", "masks"):
                batch[namespace] = {
                    key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) and value.ndim == 3 else value
                    for key, value in sample[namespace].items()
                }
            change, _ = build_ai4g_change_features(batch, **feature_parameters(config))
            logits = model(change)
            probability = torch.sigmoid(logits)
            raw, buffered = postprocess_extent(
                probability,
                batch["validity"]["output_valid"],
                probability_threshold=float(config["extent_model"]["probability_threshold"]),
                buffer_pixels=int(config["extent_model"]["buffer_pixels"]),
            )
            probability_np = probability[0, 0].float().cpu().numpy()
            raw_np = raw[0, 0].cpu().numpy()
            extent_np = buffered[0, 0].cpu().numpy()
            target_np = sample["masks"]["valid_depth_mask"][0].numpy() > 0.5
            output_valid_np = sample["validity"]["output_valid"][0].numpy() > 0.5
            target_np &= output_valid_np
            positive_count = int(np.count_nonzero(target_np))
            sample_raw_tp = int(np.count_nonzero(raw_np & target_np))
            sample_raw_fp = int(np.count_nonzero(raw_np & ~target_np & output_valid_np))
            sample_raw_fn = int(np.count_nonzero(~raw_np & target_np))
            sample_buffered_tp = int(np.count_nonzero(extent_np & target_np))
            sample_buffered_fp = int(np.count_nonzero(extent_np & ~target_np & output_valid_np))
            sample_buffered_fn = int(np.count_nonzero(~extent_np & target_np))
            eligible = int(np.count_nonzero(output_valid_np))
            raw_count = int(np.count_nonzero(raw_np))
            buffered_count = int(np.count_nonzero(extent_np))
            row = {
                "sample_id": sample["metadata"]["sample_id"],
                "source_event_id": sample["metadata"]["source_event_id"],
                "positive_pixels": positive_count,
                **_binary_metrics(sample_raw_tp, sample_raw_fp, sample_raw_fn, "raw"),
                **_binary_metrics(
                    sample_buffered_tp, sample_buffered_fp, sample_buffered_fn, "buffered"
                ),
                "raw_predicted_pixels": raw_count,
                "buffered_predicted_pixels": buffered_count,
                "eligible_pixels": eligible,
                "raw_predicted_area_fraction": raw_count / eligible if eligible else float("nan"),
                "buffered_predicted_area_fraction": buffered_count / eligible if eligible else float("nan"),
            }
            rows.append(row)
            total_positive += positive_count
            raw_tp += sample_raw_tp
            raw_fp += sample_raw_fp
            raw_fn += sample_raw_fn
            buffered_tp += sample_buffered_tp
            buffered_fp += sample_buffered_fp
            buffered_fn += sample_buffered_fn
            total_eligible += eligible
            total_raw += raw_count
            total_buffered += buffered_count
            sample_dir = split_dir / str(sample["metadata"]["sample_id"])
            crs = str(sample["metadata"]["crs"])
            transform = sample["metadata"]["transform"]
            write_geotiff(
                sample_dir / "flood_probability.tif",
                probability_np,
                crs=crs,
                transform=transform,
                valid_mask=output_valid_np,
                descriptions=["flood_probability_ai4g_mobilenet_iou"],
            )
            write_geotiff(
                sample_dir / "flood_extent_raw.tif",
                raw_np.astype(np.float32),
                crs=crs,
                transform=transform,
                valid_mask=output_valid_np,
                descriptions=["flood_extent_raw_threshold_0p5"],
            )
            write_geotiff(
                sample_dir / "flood_extent.tif",
                extent_np.astype(np.float32),
                crs=crs,
                transform=transform,
                valid_mask=output_valid_np,
                descriptions=["flood_extent_buffered_80m"],
            )
            atomic_write_json(sample_dir / "diagnostics.json", row)
    write_rows(split_dir / "extent_diagnostics_by_sample.csv", rows)
    summary = {
        "split": split,
        "sample_count": len(dataset),
        **_binary_metrics(raw_tp, raw_fp, raw_fn, "raw"),
        **_binary_metrics(buffered_tp, buffered_fp, buffered_fn, "buffered"),
        "raw_predicted_area_fraction": total_raw / total_eligible,
        "buffered_predicted_area_fraction": total_buffered / total_eligible,
        "positive_pixels": total_positive,
        "eligible_pixels": total_eligible,
        "label_semantics": "valid_depth_mask is the binary flood label",
        "binary_extent_metrics_available": True,
    }
    atomic_write_json(split_dir / "summary.json", summary)
    return summary


def run_prediction(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = {
        split: predict_split(config, checkpoint, split, output, device)
        for split in args.splits
    }
    product = {
        "product_type": "frozen_predicted_flood_extent",
        "model": "ai4g_mobilenet_v2_unet_iou",
        "paper": "Misra et al., Nature Communications 2025",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(args.config.expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "model_implementation_sha256": sha256_file(
            PROJECT_ROOT / "extent/ai4g_mobilenet_unet.py"
        ),
        "protocol_implementation_sha256": sha256_file(PROJECT_ROOT / "extent/protocol.py"),
        "dataset_fingerprint": dataset_fingerprint(config),
        "probability_threshold": float(config["extent_model"]["probability_threshold"]),
        "buffer_pixels": int(config["extent_model"]["buffer_pixels"]),
        "buffer_metres": 80.0,
        "prediction_uses_valid_depth_mask": False,
        "training_uses_valid_depth_mask_as_binary_label": True,
        "splits": summaries,
    }
    atomic_write_json(output / "extent_product.json", product)
    atomic_write_json(output / "resolved_config.json", jsonable_config(config))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(f"extent product output: {run_prediction(parse_args())}")

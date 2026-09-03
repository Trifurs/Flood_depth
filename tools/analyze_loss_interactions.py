#!/usr/bin/env python3
"""Measure loss-component gradients on a fixed handful of train batches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from datasets.flooddepth_dataset import prepare_model_inputs
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from losses.composite_loss import CompositeFloodDepthLoss
from tools.evaluate import embed_source_fingerprints
from tools.train import create_dataloaders
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.misc import move_to_device
from utils.registry import build_model


ITEMS = ("depth", "depth_linear", "depth_log", "nnpu", "uncertainty", "gradient", "auxiliary", "wse", "kan_regularization", "depth_bias")


def _grad(value, parameter):
    if not torch.is_tensor(value) or not value.requires_grad:
        return None
    return torch.autograd.grad(value, parameter, retain_graph=True, allow_unused=True)[0]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=Path("configs/pa_hydrokan/subset150_v13_corrected_final.xml")); parser.add_argument("--checkpoint", type=Path, default=Path("runs/optimization/hydrov13_1/final/best_raw.pth")); parser.add_argument("--output-dir", type=Path, default=Path("artifacts/optimization/hydrov13_2")); parser.add_argument("--device", default="auto"); parser.add_argument("--max-batches", type=int, default=8)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    config = embed_source_fingerprints(load_config(args.config)); config["training"]["num_workers"] = 0; config["training"]["persistent_workers"] = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    loader, _, dataset, _ = create_dataloaders(config); model = build_model(config).to(device); load_checkpoint(args.checkpoint, model, map_location=device)
    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), dataset.contract); bins = resolve_depth_stratification_bins(config["loss"], normalizer); pcfg = config["dataset"]["positive_prior"]; prior = normalizer.positive_prior if pcfg["mode"] == "auto" else float(pcfg["value"]); criterion = CompositeFloodDepthLoss(config["loss"], prior, bins, normalizer.train_depth_bins)
    reference_name, reference = next(((n, p) for n, p in model.named_parameters() if n.startswith("fusion") and p.requires_grad), next(iter(model.named_parameters())))
    rows = []
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= args.max_batches: break
        batch = move_to_device(cpu_batch, device); model.zero_grad(set_to_none=True); outputs = model(prepare_model_inputs(batch)); _, components = criterion(outputs, batch, epoch=20)
        grads = {}; values = {}
        for item in ITEMS:
            value = components.get("kan_magnitude", outputs.get("graph_diagnostics", {}).get("kan_coefficient_magnitude")) + components.get("kan_smoothness", outputs.get("graph_diagnostics", {}).get("kan_coefficient_smoothness")) if item == "kan_regularization" else components.get(item)
            values[item] = float(value.detach().float().mean().cpu()) if torch.is_tensor(value) else float("nan")
            grad = _grad(value, reference); grads[item] = grad.detach().float() if grad is not None else None
        main_grad = grads["depth"]; main_norm = float(torch.linalg.vector_norm(main_grad).cpu()) if main_grad is not None else 0.0
        for item in ITEMS:
            grad = grads[item]; norm = float(torch.linalg.vector_norm(grad).cpu()) if grad is not None else 0.0
            cosine = float(torch.nn.functional.cosine_similarity(grad.flatten(), main_grad.flatten(), dim=0).cpu()) if grad is not None and main_grad is not None and norm > 0 and main_norm > 0 else float("nan")
            if item == "wse": effective = criterion.wse_weight(20)
            elif item in {"nnpu", "uncertainty", "gradient", "auxiliary", "kan_regularization"}: effective = criterion.scheduled_weight({"kan_regularization": "kan", "uncertainty": "unc"}.get(item, item), 20)
            else: effective = float(config["loss"].get("lambda_depth", 1.0)) if item == "depth" else float(config["loss"].get("lambda_log", 0.0)) if item == "depth_log" else float(config["loss"].get("lambda_depth_bias", 0.0)) if item == "depth_bias" else 1.0
            rows.append({"batch": batch_index, "item": item, "loss_value": values[item], "gradient_norm": norm, "relative_to_depth_gradient": norm / max(main_norm, 1e-12), "cosine_to_depth": cosine, "effective_weight": effective, "finite": bool(torch.isfinite(torch.tensor(values[item])) and (grad is None or torch.isfinite(grad).all()))})
    fields = list(rows[0]);
    with (args.output_dir / "loss_gradient_interactions.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for item in ITEMS:
        selected = [r for r in rows if r["item"] == item]
        aggregate.append({"item": item, "mean_loss_value": sum(r["loss_value"] for r in selected) / max(1, len(selected)), "mean_gradient_norm": sum(r["gradient_norm"] for r in selected) / max(1, len(selected)), "mean_relative_to_depth_gradient": sum(r["relative_to_depth_gradient"] for r in selected) / max(1, len(selected)), "mean_cosine_to_depth": sum(r["cosine_to_depth"] for r in selected if r["cosine_to_depth"] == r["cosine_to_depth"]) / max(1, sum(r["cosine_to_depth"] == r["cosine_to_depth"] for r in selected)), "effective_weight_epoch20": selected[0]["effective_weight"] if selected else 0.0, "all_finite": all(r["finite"] for r in selected)})
    payload = {"config": str(args.config), "checkpoint": str(args.checkpoint), "device": str(device), "batches": min(args.max_batches, len(loader)), "reference_parameter": reference_name, "items": aggregate, "warning_threshold": 0.2, "notes": ["Cosines are with respect to the primary depth component on a shared fusion parameter; no test batches were used."]}
    (args.output_dir / "loss_gradient_interactions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8"); print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

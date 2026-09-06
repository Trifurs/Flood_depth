#!/usr/bin/env python3
"""Audit the behaviour of the legacy Hydro-v13 Graph-KAN.

The tool intentionally does not change the model.  It records the descriptor
presented to each KANLinear, its one-dimensional base/spline decomposition,
gate statistics, and the residual graph scale for a few train/validation
batches.  This gives a reproducible pre-v13.2 diagnosis without touching test.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.flooddepth_dataset import prepare_model_inputs
from tools.evaluate import embed_source_fingerprints
from tools.train import create_dataloaders
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.misc import move_to_device
from utils.registry import build_model
from losses.composite_loss import CompositeFloodDepthLoss
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins


def _quantiles(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return {name: float("nan") for name in ("min", "max", "mean", "std", "p01", "p05", "p25", "p50", "p75", "p95", "p99")}
    q = torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], device=x.device)
    values = torch.quantile(x, q)
    return {
        "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        **{name: float(value) for name, value in zip(("p01", "p05", "p25", "p50", "p75", "p95", "p99"), values)},
    }


class _Capture:
    def __init__(self, modules):
        self.inputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(modules))}
        self.outputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(modules))}
        self.handles = []
        for i, module in enumerate(modules):
            self.handles.append(module.register_forward_pre_hook(self._pre(i)))
            self.handles.append(module.register_forward_hook(self._post(i)))

    def _pre(self, index):
        def hook(_module, args):
            self.inputs[index].append(args[0].detach().float().cpu())
        return hook

    def _post(self, index):
        def hook(_module, _args, output):
            self.outputs[index].append(output.detach().float().cpu())
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def _decompose(module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = module.base_weight.device
    x = inputs.to(device=device, dtype=torch.float32)
    normalized = module.normalization(x) if module.normalization_mode == "legacy_layernorm" else x
    bounded = torch.tanh(normalized)
    base = F.linear(F.silu(normalized), module.base_weight.float(), module.base_bias.float())
    with torch.autocast(device_type=device.type, enabled=False):
        basis = module.b_spline_basis(bounded.float())
        spline = torch.einsum("...ik,oik->...o", basis, module.spline_coefficients.float())
    return base.detach().cpu(), spline.detach().cpu(), bounded.detach().cpu()


def _collect_pass(model, loader, device, criterion=None, epoch=0, max_batches=2, backward=False):
    graph = model.graph
    capture = _Capture(graph.edge_kans)
    graph_inputs: list[torch.Tensor] = []
    graph_outputs: list[torch.Tensor] = []
    def graph_hook(_module, args, output):
        graph_inputs.append(args[0].detach().float().cpu())
        graph_outputs.append(output[0].detach().float().cpu() if isinstance(output, tuple) else output.detach().float().cpu())
    gh = graph.register_forward_hook(graph_hook)
    model.train(mode=backward)
    batches = 0
    last_loss = None
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        inputs = prepare_model_inputs(batch)
        if backward:
            model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        if criterion is not None:
            last_loss, _ = criterion(outputs, batch, epoch)
            if backward:
                last_loss.backward()
        batches += 1
        if batches >= max_batches:
            break
    gh.remove(); capture.close()
    return capture, graph_inputs, graph_outputs, last_loss


def _rows_for_pass(model, capture, graph_inputs, graph_outputs, split, stage):
    rows_feature, rows_head = [], []
    for head, module in enumerate(model.graph.edge_kans):
        chunks = capture.inputs[head]
        out_chunks = capture.outputs[head]
        if not chunks:
            continue
        x = torch.cat(chunks, 0)
        y = torch.cat(out_chunks, 0)
        base, spline, bounded = _decompose(module, x)
        for feature in range(x.shape[-1]):
            values = x[..., feature]
            bvalues = bounded[..., feature]
            q = _quantiles(values)
            occupancy = torch.bucketize(values.flatten(), torch.linspace(-1, 1, module.grid_size + 1)[1:-1]).bincount(minlength=module.grid_size)
            row = {"split": split, "stage": stage, "head": head, "feature": feature, **q,
                   "out_of_range_fraction": float((values.abs() > 1).float().mean()),
                   "tanh_saturation_fraction": float((bvalues.abs() > 0.95).float().mean())}
            row.update({f"knot_bin_{i}": int(value) for i, value in enumerate(occupancy.tolist())})
            rows_feature.append(row)
        gate = torch.sigmoid(y.squeeze(-1))
        all_gate = gate.flatten()
        entropy = -(all_gate.clamp(1e-6, 1 - 1e-6) * all_gate.clamp(1e-6, 1 - 1e-6).log() + (1 - all_gate).clamp(1e-6, 1 - 1e-6) * (1 - all_gate).clamp(1e-6, 1 - 1e-6).log())
        grad_norm = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                grad_norm += float(parameter.grad.detach().float().square().sum())
        gamma = model.graph.gamma[head]
        input_rms = torch.cat(graph_inputs, 0).float().square().mean().sqrt() if graph_inputs else torch.tensor(float("nan"))
        output_rms = torch.cat(graph_outputs, 0).float().square().mean().sqrt() if graph_outputs else torch.tensor(float("nan"))
        rows_head.append({"split": split, "stage": stage, "head": head,
                          "gate_mean": float(all_gate.mean()), "gate_std": float(all_gate.std(unbiased=False)),
                          "gate_p05": float(torch.quantile(all_gate, .05)), "gate_p50": float(torch.quantile(all_gate, .50)),
                          "gate_p95": float(torch.quantile(all_gate, .95)), "gate_entropy": float(entropy.mean()),
                          "gate_lt_0.05": float((all_gate < .05).float().mean()), "gate_gt_0.95": float((all_gate > .95).float().mean()),
                          "gamma": float(gamma.detach().cpu()), "gamma_grad": float(model.graph.gamma.grad[head].detach().cpu()) if model.graph.gamma.grad is not None else 0.0,
                          "kan_parameter_grad_norm": grad_norm ** .5,
                          "base_rms": float(base.square().mean().sqrt()), "spline_rms": float(spline.square().mean().sqrt()),
                          "spline_base_rms_ratio": float(spline.square().mean().sqrt() / base.square().mean().sqrt().clamp_min(1e-8)),
                          "graph_update_input_rms_ratio": float((output_rms - input_rms).abs() / input_rms.clamp_min(1e-8))})
    return rows_feature, rows_head


def _v132_first_backward(config, checkpoint, device):
    """Run the new edge-factorized graph once to verify gradient flow."""
    cfg = embed_source_fingerprints(load_config(config)); cfg["training"]["num_workers"] = 0
    cfg["training"]["persistent_workers"] = False
    train_loader, _, train_ds, _ = create_dataloaders(cfg)
    normalizer = RobustNormalizer(Path(cfg["dataset"]["train_stats"]), train_ds.contract)
    bins = resolve_depth_stratification_bins(cfg["loss"], normalizer)
    prior_cfg = cfg["dataset"]["positive_prior"]
    prior = normalizer.positive_prior if prior_cfg["mode"] == "auto" else float(prior_cfg["value"])
    criterion = CompositeFloodDepthLoss(cfg["loss"], prior, bins, normalizer.train_depth_bins)
    model = build_model(cfg).to(device)
    if checkpoint is not None:
        load_checkpoint(checkpoint, model, map_location=device)
    model.train(); batch = move_to_device(next(iter(train_loader)), device)
    model.zero_grad(set_to_none=True)
    outputs = model(prepare_model_inputs(batch))
    loss, _ = criterion(outputs, batch, epoch=0)
    loss.backward()
    graph = model.graph
    edge_grad = sum(float(p.grad.detach().float().square().sum()) for p in graph.edge_kan.parameters() if p.grad is not None) ** 0.5
    gamma_grad = graph.raw_gamma.grad.detach().float().cpu().tolist() if graph.raw_gamma.grad is not None else [0.0] * graph.heads
    diagnostics = outputs.get("graph_diagnostics", {})
    return {
        "config": str(config), "checkpoint": str(checkpoint) if checkpoint else None,
        "loss": float(loss.detach().cpu()), "edge_kan_grad_norm": edge_grad,
        "edge_kan_grad_nonzero": bool(edge_grad > 0.0),
        "raw_gamma_grad": gamma_grad,
        "gamma": graph.gamma.detach().float().cpu().tolist(),
        "gamma_max": graph.gamma_max,
        "gate_mean": float(diagnostics["gate_mean"].detach().cpu()),
        "graph_update_input_rms_ratio": float(diagnostics["graph_update_rms_ratio"].detach().cpu()),
        "descriptor_bounded": bool(diagnostics["last_descriptors"].abs().max().item() <= 1.0 + 1e-6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pa_hydrokan/subset150_v13_corrected_final.xml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/optimization/hydrov13_1/final/best_raw.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/optimization/hydrov13_2"))
    parser.add_argument("--device", default="auto"); parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--v132-config", type=Path)
    parser.add_argument("--v132-checkpoint", type=Path)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    config = embed_source_fingerprints(load_config(args.config)); config["training"]["num_workers"] = 0; config["training"]["persistent_workers"] = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    train_loader, val_loader, train_ds, _ = create_dataloaders(config)
    normalizer = RobustNormalizer(Path(config["dataset"]["train_stats"]), train_ds.contract)
    bins = resolve_depth_stratification_bins(config["loss"], normalizer)
    prior_cfg = config["dataset"]["positive_prior"]; prior = normalizer.positive_prior if prior_cfg["mode"] == "auto" else float(prior_cfg["value"])
    criterion = CompositeFloodDepthLoss(config["loss"], prior, bins, normalizer.train_depth_bins)
    task_config = deepcopy(config["loss"]); task_config["lambda_kan"] = 0.0
    task_criterion = CompositeFloodDepthLoss(task_config, prior, bins, normalizer.train_depth_bins)
    trained = build_model(config).to(device); load_checkpoint(args.checkpoint, trained, map_location=device)
    fresh = build_model(config).to(device)
    all_feature, all_head = [], []; stage_meta = {}
    for name, model, loader, backward in (("trained_checkpoint", trained, train_loader, False), ("trained_checkpoint", trained, val_loader, False), ("init_first_backward", fresh, train_loader, True)):
        split = "train" if loader is train_loader else "val"
        cap, gi, go, loss = _collect_pass(model, loader, device, task_criterion if backward else None, backward=backward, max_batches=1 if backward else args.max_batches)
        fr, hr = _rows_for_pass(model, cap, gi, go, split, name); all_feature.extend(fr); all_head.extend(hr)
        stage_meta[f"{name}_{split}"] = {"batches": min(args.max_batches, len(loader)), "loss": float(loss.detach().cpu()) if loss is not None else None}
    with (args.output_dir / "kan_feature_occupancy.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(all_feature[0])); writer.writeheader(); writer.writerows(all_feature)
    with (args.output_dir / "kan_head_summary.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(all_head[0])); writer.writeheader(); writer.writerows(all_head)
    summary = {"config": str(args.config), "checkpoint": str(args.checkpoint), "device": str(device), "stages": stage_meta,
               "edge_features": list(getattr(trained.graph, "edge_kans")[0].in_features for _ in [0]),
               "heads": len(trained.graph.edge_kans), "grid_size": trained.graph.edge_kans[0].grid_size,
               "spline_order": trained.graph.edge_kans[0].spline_order,
               "init_kan_grad_norm": [row["kan_parameter_grad_norm"] for row in all_head if row["stage"] == "init_first_backward"],
               "init_task_only_kan_grad_norm_nonzero": any(row["kan_parameter_grad_norm"] > 0.0 for row in all_head if row["stage"] == "init_first_backward"),
               "init_gamma": [row["gamma"] for row in all_head if row["stage"] == "init_first_backward"],
               "trained_head_summary": [row for row in all_head if row["stage"] == "trained_checkpoint"],
               "notes": ["Legacy v13 uses explicit descriptor scaling followed by KANLinear tanh; this is the pre-v13.2 diagnosis."]}
    if args.v132_config is not None:
        summary["v13_2_first_backward"] = _v132_first_backward(
            args.v132_config, args.v132_checkpoint, device
        )
    (args.output_dir / "kan_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

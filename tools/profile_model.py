#!/usr/bin/env python3
"""Profile model size, real-raster latency, backward stability, and memory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets.flooddepth_dataset import prepare_model_inputs
from tools.evaluate import embed_source_fingerprints
from tools.train import create_dataloaders
from utils.amp import resolve_amp
from utils.config import load_config
from utils.misc import atomic_write_json, move_to_device
from utils.registry import build_model
from utils.checkpoint import load_checkpoint


def _slice_inputs(inputs, size: int):
    return {key: value[:size] for key, value in inputs.items()}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward_profile(model, inputs, device, active, dtype, iterations: int):
    timings = []
    model.eval()
    with torch.no_grad():
        for index in range(iterations + 5):
            _synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=active, dtype=dtype):
                outputs = model(inputs)
            _synchronize(device)
            if index >= 5:
                timings.append(time.perf_counter() - start)
    mean = sum(timings) / len(timings)
    return outputs, mean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    args = parser.parse_args(); config = embed_source_fingerprints(load_config(args.config))
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    config["training"]["num_workers"] = 0; config["training"]["persistent_workers"] = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    _, loader, _, _ = create_dataloaders(config)
    wait_start = time.perf_counter()
    batch = move_to_device(next(iter(loader)), device)
    loader_wait = time.perf_counter() - wait_start
    inputs = prepare_model_inputs(batch); model = build_model(config).to(device)
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint, model, map_location=device)
        if args.weights == "ema":
            ema_state = checkpoint.get("ema_model")
            if ema_state is None:
                raise ValueError("Checkpoint does not contain EMA model weights")
            model.load_state_dict(ema_state, strict=True)
    active, dtype, _ = resolve_amp(device, bool(config["training"]["amp"]), str(config["training"].get("amp_dtype", "float16")))
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    outputs_one, latency_one = _forward_profile(
        model, _slice_inputs(inputs, 1), device, active, dtype, args.iterations
    )
    outputs, latency = _forward_profile(
        model, inputs, device, active, dtype, args.iterations
    )
    forward_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    model.train(); model.zero_grad(set_to_none=True); _synchronize(device)
    backward_start = time.perf_counter()
    with torch.autocast(device_type=device.type, enabled=active, dtype=dtype):
        backward_outputs = model(inputs)
        profile_loss = (
            backward_outputs["depth"].mean()
            + backward_outputs["support_logits"].square().mean() * 0.01
            + backward_outputs["uncertainty_scale"].mean() * 0.01
        )
    profile_loss.backward(); _synchronize(device)
    forward_backward = time.perf_counter() - backward_start
    gradients = [p.grad.detach().float() for p in model.parameters() if p.grad is not None]
    gradients_finite = all(torch.isfinite(value).all().item() for value in gradients)
    gradient_norm = float(torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(value) for value in gradients
    ])).cpu()) if gradients else 0.0
    backward_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    parameters = sum(p.numel() for p in model.parameters())
    report = {"config": str(args.config.resolve()), "device": str(device), "amp_enabled": active,
              "amp_dtype": str(dtype), "parameters": parameters,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "batch_size": int(batch["label"].shape[0]), "iterations": args.iterations,
              "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
              "weights": args.weights if args.checkpoint else None,
              "module_parameters": {name: sum(p.numel() for p in module.parameters())
                                    for name, module in model.named_children()},
              "data_loader_first_batch_seconds": loader_wait,
              "batch1_forward_seconds_mean": latency_one,
              "latency_seconds_mean": latency,
              "forward_backward_seconds": forward_backward,
              "samples_per_second": int(batch["label"].shape[0]) / latency,
              "peak_gpu_memory_bytes": forward_peak,
              "forward_backward_peak_gpu_memory_bytes": backward_peak,
              "gradient_norm": gradient_norm,
              "gradients_finite": gradients_finite,
              "amp_overflow_detected": not gradients_finite,
              "output_ranges": {
                  key: {"minimum": float(value.detach().float().min().cpu()),
                        "maximum": float(value.detach().float().max().cpu())}
                  for key, value in outputs.items() if torch.is_tensor(value)
              }}
    atomic_write_json(args.output, report); print(report); return 0


if __name__ == "__main__": raise SystemExit(main())

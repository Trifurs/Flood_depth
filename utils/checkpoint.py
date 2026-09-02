"""Atomic, fingerprint-strict checkpoints with RNG restoration."""

from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch


class CheckpointError(RuntimeError):
    """Raised for incompatible or incomplete checkpoint state."""


def checkpoint_depth_output_semantics(checkpoint: Mapping[str, Any]) -> str:
    """Resolve output semantics, treating pre-v2 checkpoints as legacy products."""

    resolved = checkpoint.get("resolved_config", {})
    if isinstance(resolved, Mapping):
        model_config = resolved.get("model", {})
        if isinstance(model_config, Mapping):
            return str(
                model_config.get("depth_output_semantics", "probability_weighted_v1")
            )
    return "probability_weighted_v1"


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``map_location=cuda`` also moves the serialized CPU RNG tensor to CUDA,
    # while PyTorch's RNG setters require CPU byte tensors.  Accept list-like
    # states as well so checkpoints remain robust across serialization formats.
    torch_cpu = state["torch_cpu"]
    if torch.is_tensor(torch_cpu):
        torch_cpu = torch_cpu.detach().to(device="cpu", dtype=torch.uint8)
    else:
        torch_cpu = torch.as_tensor(torch_cpu, dtype=torch.uint8, device="cpu")
    torch.set_rng_state(torch_cpu)
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        cuda_states = []
        for cuda_state in state["torch_cuda"]:
            if torch.is_tensor(cuda_state):
                cuda_state = cuda_state.detach().to(device="cpu", dtype=torch.uint8)
            else:
                cuda_state = torch.as_tensor(
                    cuda_state, dtype=torch.uint8, device="cpu"
                )
            cuda_states.append(cuda_state)
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_metric: float,
    resolved_config: Mapping[str, Any],
    dataset_fingerprint: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> None:
    unwrapped = model.module if hasattr(model, "module") else model
    payload = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "rng_states": capture_rng_state(),
        "resolved_config": dict(resolved_config),
        "dataset_fingerprint": dict(dataset_fingerprint),
        "extra": dict(extra or {}),
    }
    atomic_torch_save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    expected_fingerprint: Mapping[str, str] | None = None,
    allow_fingerprint_mismatch: bool = False,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
    adopt_checkpoint_output_semantics: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if expected_fingerprint is not None and dict(checkpoint.get("dataset_fingerprint", {})) != dict(
        expected_fingerprint
    ):
        if not allow_fingerprint_mismatch:
            raise CheckpointError(
                "Dataset contract/manifest/normalization fingerprint differs from checkpoint; "
                "refusing resume without the explicit dangerous override"
            )
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.load_state_dict(checkpoint["model"], strict=True)
    if adopt_checkpoint_output_semantics:
        heads = getattr(unwrapped, "heads", None)
        setter = getattr(heads, "set_depth_output_semantics", None)
        if callable(setter):
            setter(checkpoint_depth_output_semantics(checkpoint))
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("grad_scaler") is not None:
        scaler.load_state_dict(checkpoint["grad_scaler"])
    if restore_rng and checkpoint.get("rng_states") is not None:
        restore_rng_state(checkpoint["rng_states"])
    return checkpoint

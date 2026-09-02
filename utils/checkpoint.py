"""Atomic, fingerprint-strict checkpoints with RNG restoration."""

from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
import hashlib
import json


class CheckpointError(RuntimeError):
    """Raised for incompatible or incomplete checkpoint state."""


def training_identity_sha256(
    resolved_config: Mapping[str, Any],
    dataset_fingerprint: Mapping[str, str],
    *,
    version: int = 2,
) -> str:
    """Hash every semantic field that must remain fixed across resume.

    Version 1 reproduces checkpoints written during the initial Hydro-v13 run.
    Version 2 additionally names the reliability schema explicitly; older files
    remain resumable only when their original v1 identity matches exactly.
    """

    identity_fields = {
        key: resolved_config.get(key)
        for key in ("model", "loss", "optimizer", "scheduler")
    }
    dataset_config = resolved_config.get("dataset", {})
    if isinstance(dataset_config, Mapping):
        identity_fields["model_bands"] = dataset_config.get("resolved_model_bands")
        if version >= 2:
            identity_fields["reliability_schema"] = dataset_config.get(
                "resolved_reliability_schema"
            )
    identity_fields["dataset_fingerprint"] = dict(dataset_fingerprint)
    return hashlib.sha256(
        json.dumps(identity_fields, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


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
    ema: Any = None,
) -> None:
    unwrapped = model.module if hasattr(model, "module") else model
    identity_hash = training_identity_sha256(
        resolved_config, dataset_fingerprint, version=2
    )
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
        "training_identity_sha256": identity_hash,
        "training_identity_version": 2,
        "ema": ema.state_dict() if ema is not None else None,
        "ema_model": ema.model_state_dict() if ema is not None else None,
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
    expected_training_identity_sha256: str | None = None,
    expected_legacy_training_identity_sha256: str | None = None,
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
    if expected_training_identity_sha256 is not None:
        saved_identity = checkpoint.get("training_identity_sha256")
        identity_version = int(checkpoint.get("training_identity_version", 1))
        expected_identity = (
            expected_training_identity_sha256
            if identity_version >= 2
            else expected_legacy_training_identity_sha256
        )
        if saved_identity is None or expected_identity is None or saved_identity != expected_identity:
            raise CheckpointError(
                "Training identity differs from checkpoint; model structure, BandSpec, "
                "reliability schema, loss, optimizer, scheduler, and dataset semantics "
                "must remain unchanged when resuming"
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

"""Small torchrun/DDP helpers; single-process remains the default."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def initialize_distributed(device_preference: str = "auto") -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    if device_preference == "cpu":
        device = torch.device("cpu")
    elif device_preference.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else device_preference)
    else:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size, local_rank


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

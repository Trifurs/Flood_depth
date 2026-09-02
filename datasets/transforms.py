"""Synchronous, physically safe raster augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


GEOMETRIC_TOP_LEVEL = {
    "s1_t1",
    "s1_t2",
    "s1_change",
    "s1_conditioning",
    "s1_qa",
    "s2_t1",
    "s2_t2",
    "s2_change",
    "s2_qa",
    "terrain",
    "terrain_raw",
    "label",
    "reliability",
}


def _map_tensors(value: Any, function: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim >= 2:
        return function(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    return value


@dataclass
class SynchronousAugment:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate90_probability: float = 0.5
    modality_dropout_probability: float = 0.1

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if torch.rand(()) < self.horizontal_flip_probability:
            sample = _map_tensors(sample, lambda tensor: torch.flip(tensor, dims=(-1,)))
        if torch.rand(()) < self.vertical_flip_probability:
            sample = _map_tensors(sample, lambda tensor: torch.flip(tensor, dims=(-2,)))
        if torch.rand(()) < self.rotate90_probability:
            turns = int(torch.randint(1, 4, ()).item())
            sample = _map_tensors(sample, lambda tensor: torch.rot90(tensor, turns, dims=(-2, -1)))
        if torch.rand(()) < self.modality_dropout_probability:
            modality = "s1" if torch.rand(()) < 0.5 else "s2"
            keys = ("s1_t1", "s1_t2", "s1_change") if modality == "s1" else (
                "s2_t1",
                "s2_t2",
                "s2_change",
            )
            if modality == "s1" and "s1_conditioning" in sample:
                keys = (*keys, "s1_conditioning")
            for key in keys:
                sample[key] = torch.zeros_like(sample[key])
            validity_key = f"{modality}_valid"
            sample["validity"][validity_key] = torch.zeros_like(
                sample["validity"][validity_key]
            )
            from datasets.preprocessing import RELIABILITY_NAMES

            reliability_name = f"{modality}_valid"
            reliability_index = RELIABILITY_NAMES.index(reliability_name)
            sample["reliability"][reliability_index].zero_()
            sample["validity"]["output_valid"] = sample["validity"]["dem_valid"] * torch.maximum(
                sample["validity"]["s1_valid"], sample["validity"]["s2_valid"]
            )
            sample["metadata"]["modality_dropout"] = modality
        else:
            sample["metadata"]["modality_dropout"] = "none"
        return sample

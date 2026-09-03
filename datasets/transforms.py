"""Synchronous, physically safe raster augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

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
    modality_dropout_probability: float | None = 0.1
    feature_dropout_probability: float | None = None
    sensor_missing_simulation_probability: float | None = None
    input_mode: str = "s1_s2_terrain"

    def __post_init__(self) -> None:
        self._legacy_compat = False
        if self.feature_dropout_probability is None and self.sensor_missing_simulation_probability is None:
            self._legacy_compat = True
            warnings.warn(
                "modality_dropout_probability is deprecated; mapping it to "
                "sensor_missing_simulation_probability for compatibility",
                UserWarning,
                stacklevel=2,
            )
            self.feature_dropout_probability = 0.0
            self.sensor_missing_simulation_probability = float(
                self.modality_dropout_probability or 0.0
            )
        else:
            self.feature_dropout_probability = float(self.feature_dropout_probability or 0.0)
            self.sensor_missing_simulation_probability = float(
                self.sensor_missing_simulation_probability or 0.0
            )
        for name in (
            "feature_dropout_probability",
            "sensor_missing_simulation_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.input_mode not in {"s1_s2_terrain", "s1_terrain"}:
            raise ValueError(f"Unknown input_mode {self.input_mode!r}")
        if self.input_mode == "s1_terrain":
            # A single available SAR modality must never be randomly removed.
            self.feature_dropout_probability = 0.0
            self.sensor_missing_simulation_probability = 0.0

    @staticmethod
    def _modality_keys(modality: str) -> tuple[str, ...]:
        return (f"{modality}_t1", f"{modality}_t2", f"{modality}_change")

    def _zero_features(self, sample: dict[str, Any], modality: str) -> None:
        for key in self._modality_keys(modality):
            sample[key] = torch.zeros_like(sample[key])
        if modality == "s1" and "s1_conditioning" in sample:
            sample["s1_conditioning"] = torch.zeros_like(sample["s1_conditioning"])
        extent = sample.get("extent_inputs")
        if isinstance(extent, dict):
            for key in ("s1_t1_db", "s1_t2_db", "s1_pair_valid"):
                if modality == "s1" and key in extent:
                    extent[key] = torch.zeros_like(extent[key])

    def _simulate_missing(self, sample: dict[str, Any], modality: str) -> None:
        from datasets.preprocessing import RELIABILITY_NAMES

        self._zero_features(sample, modality)
        qa = sample.get(f"{modality}_qa")
        if isinstance(qa, torch.Tensor):
            qa.zero_()
        validity = sample["validity"]
        for key in (
            f"{modality}_valid",
            f"{modality}_available",
        ):
            if key in validity:
                validity[key].zero_()
        for key in (
            f"{modality}_t1_valid_fraction",
            f"{modality}_t2_valid_fraction",
            f"{modality}_change_valid_fraction",
        ):
            if key in validity:
                validity[key].zero_()
        names = {
            "s1": (
                "s1_event_observation_count_z",
                "s1_event_day_z",
                "s1_valid",
                "s1_day_missing",
            ),
            "s2": (
                "s2_pre_clear_observation_count_z",
                "s2_event_clear_observation_count_z",
                "s2_event_day_z",
                "s2_valid",
                "s2_day_missing",
            ),
        }[modality]
        for name in names:
            index = RELIABILITY_NAMES.index(name)
            sample["reliability"][index].zero_()
        day_index = RELIABILITY_NAMES.index("absolute_normalized_sensor_day_difference")
        sample["reliability"][day_index].fill_(1.0)
        validity["output_valid"] = validity["dem_valid"] * torch.maximum(
            validity["s1_valid"], validity["s2_valid"]
        )

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample.setdefault("metadata", {})
        if torch.rand(()) < self.horizontal_flip_probability:
            sample = _map_tensors(sample, lambda tensor: torch.flip(tensor, dims=(-1,)))
        if torch.rand(()) < self.vertical_flip_probability:
            sample = _map_tensors(sample, lambda tensor: torch.flip(tensor, dims=(-2,)))
        if torch.rand(()) < self.rotate90_probability:
            turns = int(torch.randint(1, 4, ()).item())
            sample = _map_tensors(sample, lambda tensor: torch.rot90(tensor, turns, dims=(-2, -1)))
        if self.input_mode == "s1_terrain":
            sample["metadata"]["modality_dropout"] = "disabled_s1_only"
            sample["metadata"]["dropout_type"] = "disabled_s1_only"
        elif self._legacy_compat:
            legacy_missing = torch.rand(()) < float(
                self.sensor_missing_simulation_probability
            )
            if legacy_missing:
                modality = "s1" if torch.rand(()) < 0.5 else "s2"
                self._simulate_missing(sample, modality)
                sample["metadata"]["modality_dropout"] = modality
                sample["metadata"]["dropout_type"] = "sensor_missing_simulation"
                return sample
            sample["metadata"]["modality_dropout"] = "none"
            sample["metadata"]["dropout_type"] = "none"
        elif torch.rand(()) < float(self.feature_dropout_probability):
            modality = "s1" if torch.rand(()) < 0.5 else "s2"
            self._zero_features(sample, modality)
            sample["metadata"]["modality_dropout"] = modality
            sample["metadata"]["dropout_type"] = "feature_dropout"
        elif torch.rand(()) < float(self.sensor_missing_simulation_probability):
            modality = "s1" if torch.rand(()) < 0.5 else "s2"
            self._simulate_missing(sample, modality)
            sample["metadata"]["modality_dropout"] = modality
            sample["metadata"]["dropout_type"] = "sensor_missing_simulation"
        else:
            sample["metadata"]["modality_dropout"] = "none"
            sample["metadata"]["dropout_type"] = "none"
        return sample

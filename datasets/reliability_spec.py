"""Named reliability schemas shared by dataset, model, loss, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


SCHEMAS = {
    "s1_s2_terrain": (
        "s1_event_observation_count_z", "s1_event_day_z",
        "s2_pre_clear_observation_count_z", "s2_event_clear_observation_count_z",
        "s2_event_day_z", "s1_available", "s2_available", "dem_available",
        "event_duration_log_scaled", "absolute_normalized_sensor_day_difference",
        "s1_day_missing", "s2_day_missing",
    ),
    "s1_terrain": (
        "s1_event_observation_count_z", "s1_event_day_z", "s1_available",
        "dem_available", "event_duration_log_scaled", "s1_day_missing",
    ),
}


@dataclass(frozen=True)
class ReliabilitySpec:
    mode: str
    names: tuple[str, ...]

    @classmethod
    def from_mode(cls, mode: str) -> "ReliabilitySpec":
        try:
            return cls(str(mode), tuple(SCHEMAS[str(mode)]))
        except KeyError as exc:
            raise ValueError(f"No reliability schema for input mode {mode!r}") from exc

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"Reliability channel {name!r} is not active in {self.mode}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "names": list(self.names), "channels": len(self.names)}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")).hexdigest()

"""Explicit raster-group activation for model-specific dataset I/O."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


CORE_GROUPS = (
    "label", "masks", "s1_t1", "s1_t2", "s1_change", "s1_qa", "terrain",
)
S2_GROUPS = ("s2_t1", "s2_t2", "s2_change", "s2_qa")
CONTINUOUS_GROUPS = ("s1_t1", "s1_t2", "s1_change", "terrain", *S2_GROUPS[:3])


@dataclass(frozen=True)
class ModelInputSpec:
    """The complete, serializable input contract for one model family."""

    mode: str
    active_groups: tuple[str, ...]

    @classmethod
    def from_mode(cls, mode: str | None) -> "ModelInputSpec":
        resolved = str(mode or "s1_s2_terrain")
        if resolved == "s1_s2_terrain":
            groups = CORE_GROUPS + S2_GROUPS
        elif resolved == "s1_terrain":
            groups = CORE_GROUPS
        else:
            raise ValueError(
                f"Unknown dataset.input_mode {resolved!r}; expected 's1_s2_terrain' or 's1_terrain'"
            )
        return cls(resolved, tuple(groups))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ModelInputSpec":
        dataset = config.get("dataset", config)
        return cls.from_mode(dataset.get("input_mode") if isinstance(dataset, Mapping) else None)

    @property
    def is_s1_only(self) -> bool:
        return self.mode == "s1_terrain"

    @property
    def continuous_groups(self) -> tuple[str, ...]:
        return tuple(group for group in CONTINUOUS_GROUPS if group in self.active_groups)

    @property
    def qa_groups(self) -> tuple[str, ...]:
        return tuple(group for group in ("s1_qa", "s2_qa") if group in self.active_groups)

    def requires(self, group: str) -> bool:
        return group in self.active_groups

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "active_groups": list(self.active_groups),
            "inactive_groups": [group for group in S2_GROUPS if group not in self.active_groups],
            "continuous_groups": list(self.continuous_groups),
            "qa_groups": list(self.qa_groups),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    @property
    def active_groups_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(list(self.active_groups), separators=(",", ":")).encode("utf-8")
        ).hexdigest()

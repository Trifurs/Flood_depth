"""Strict, hash-aware data contract utilities.

The JSON contract is generated from the real dataset by ``tools/inspect_dataset.py``.
Runtime code resolves channel indices from band descriptions in that contract; it does
not infer scientific meaning from array positions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ContractError(RuntimeError):
    """Raised when the audited dataset contract cannot be trusted."""


EXPECTED_GROUPS: dict[str, dict[str, Any]] = {
    "label": {"path_column": "label_path", "bands": ["depth_m"], "role": "target"},
    "terrain": {
        "path_column": "dem_path",
        "bands": ["elevation_m_DSM", "slope_deg"],
        "role": "model_input",
    },
    "s1_t1": {
        "path_column": "s1_t1_path",
        "bands": ["VV_pre_db", "VH_pre_db", "angle_pre_deg"],
        "role": "model_input",
    },
    "s1_t2": {
        "path_column": "s1_t2_path",
        "bands": ["VV_event_db", "VH_event_db", "angle_event_deg"],
        "role": "model_input",
    },
    "s1_change": {
        "path_column": "s1_change_path",
        "bands": ["VV_delta_db", "VH_delta_db", "anomaly_raw", "anomaly_selection"],
        "role": "model_input",
    },
    "s1_qa": {
        "path_column": "s1_qa_path",
        "bands": [
            "event_observation_count",
            "selected_pre_observation_count",
            "selected_event_day_offset",
            "selected_relative_orbit",
            "selected_orbit_pass_code",
        ],
        "role": "quality_only",
    },
    "s2_t1": {
        "path_column": "s2_t1_path",
        "bands": [
            "B2_pre_reflectance",
            "B3_pre_reflectance",
            "B4_pre_reflectance",
            "B8_pre_reflectance",
            "B11_pre_reflectance",
            "B12_pre_reflectance",
        ],
        "role": "model_input",
    },
    "s2_t2": {
        "path_column": "s2_t2_path",
        "bands": [
            "B2_event_reflectance",
            "B3_event_reflectance",
            "B4_event_reflectance",
            "B8_event_reflectance",
            "B11_event_reflectance",
            "B12_event_reflectance",
        ],
        "role": "model_input",
    },
    "s2_change": {
        "path_column": "s2_change_path",
        "bands": ["NDWI_delta", "MNDWI_delta", "water_change_selection"],
        "role": "model_input",
    },
    "s2_qa": {
        "path_column": "s2_qa_path",
        "bands": [
            "pre_clear_observation_count",
            "event_clear_observation_count",
            "selected_event_day_offset",
        ],
        "role": "quality_only",
    },
    "masks": {
        "path_column": "masks_path",
        "bands": [
            "valid_depth_mask",
            "flood_mask",
            "unknown_mask",
            "permanent_water_mask",
            "extreme_high_mask",
            "DEM_valid_mask",
            "slope_valid_mask",
            "persistent_water",
            "S1_event_composite_valid_mask",
            "S2_event_composite_valid_mask",
        ],
        "role": "mask_only",
    },
}

MODEL_CONTINUOUS_GROUPS = (
    "s1_t1",
    "s1_t2",
    "s1_change",
    "s2_t1",
    "s2_t2",
    "s2_change",
    "terrain",
)

SAFE_QA_BANDS = {
    "s1_qa": ["event_observation_count", "selected_event_day_offset"],
    "s2_qa": [
        "pre_clear_observation_count",
        "event_clear_observation_count",
        "selected_event_day_offset",
    ],
}

DISABLED_QA_BANDS = {
    "s1_qa": [
        "selected_pre_observation_count",
        "selected_relative_orbit",
        "selected_orbit_pass_code",
    ],
    "s2_qa": [],
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_within(path: Path, root: Path) -> Path:
    """Resolve *path* and reject traversal outside *root*."""

    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"Path escapes dataset root: {path}") from exc
    return resolved


@dataclass(frozen=True)
class DatasetContract:
    """Validated runtime view of an audited flood-depth subset contract."""

    path: Path
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "DatasetContract":
        contract_path = Path(path).expanduser().resolve(strict=True)
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"Cannot read dataset contract {contract_path}: {exc}") from exc
        if payload.get("schema_version") != "1.0":
            raise ContractError(f"Unsupported contract schema: {payload.get('schema_version')!r}")
        if payload.get("status") != "ready":
            raise ContractError(f"Dataset contract is not ready: {payload.get('status')!r}")
        groups = payload.get("raster_groups")
        if not isinstance(groups, dict):
            raise ContractError("Contract has no raster_groups mapping")
        missing = set(EXPECTED_GROUPS).difference(groups)
        if missing:
            raise ContractError(f"Contract is missing groups: {sorted(missing)}")
        return cls(contract_path, payload)

    @property
    def dataset_root(self) -> Path:
        root = Path(str(self.payload["dataset_root"])).expanduser().resolve(strict=True)
        return root

    @property
    def manifest_path(self) -> Path:
        rel = Path(str(self.payload["manifest"]["relative_path"]))
        return ensure_within(self.dataset_root / rel, self.dataset_root)

    @property
    def hash(self) -> str:
        return sha256_file(self.path)

    def group(self, name: str) -> Mapping[str, Any]:
        try:
            return self.payload["raster_groups"][name]
        except KeyError as exc:
            raise ContractError(f"Unknown raster group: {name}") from exc

    def band_index(self, group: str, description: str) -> int:
        """Return the zero-based index resolved from audited descriptions."""

        descriptions = list(self.group(group)["band_descriptions"])
        try:
            return descriptions.index(description)
        except ValueError as exc:
            raise ContractError(f"Band {description!r} is absent from {group}: {descriptions}") from exc

    @property
    def main_input_channels(self) -> int:
        return sum(int(self.group(name)["band_count"]) for name in MODEL_CONTINUOUS_GROUPS)

    def verify_fingerprints(self, include_normalization: bool = True) -> None:
        """Fail closed when a key source file changed after auditing."""

        manifest = self.manifest_path
        observed = sha256_file(manifest)
        expected = str(self.payload["manifest"]["sha256"])
        if observed != expected:
            raise ContractError(
                f"Manifest fingerprint changed: expected {expected}, observed {observed}"
            )
        for rel, expected_hash in self.payload.get("key_file_sha256", {}).items():
            source = ensure_within(self.dataset_root / rel, self.dataset_root)
            observed_hash = sha256_file(source)
            if observed_hash != expected_hash:
                raise ContractError(
                    f"Key data file changed ({rel}): expected {expected_hash}, observed {observed_hash}"
                )
        if include_normalization:
            selected = self.payload.get("normalization", {}).get("selected")
            if selected and selected.get("sha256"):
                selected_path = Path(str(selected["path"]))
                if not selected_path.is_absolute():
                    selected_path = (self.path.parent.parent.parent / selected_path).resolve()
                if not selected_path.is_file():
                    raise ContractError(f"Selected normalization file is missing: {selected_path}")
                if sha256_file(selected_path) != selected["sha256"]:
                    raise ContractError(f"Normalization fingerprint changed: {selected_path}")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

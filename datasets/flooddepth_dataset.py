"""Lazy, strict loader for the audited event-aggregated flood-depth subset."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from datasets.contract import DatasetContract, MODEL_CONTINUOUS_GROUPS, ensure_within
from datasets.band_selection import BandSpec
from datasets.preprocessing import RELIABILITY_NAMES, RobustNormalizer


class DatasetIntegrityError(RuntimeError):
    """Raised instead of silently training on an inconsistent raster sample."""


class FloodDepthDataset(Dataset[dict[str, Any]]):
    """Return structured modalities, targets, masks, validity, and provenance.

    T2 rasters are asynchronous event-period composites, not regular time steps.
    Label-derived masks are returned solely for loss/evaluation and are never part of
    ``model_inputs``. Invalid target values are filled with zero only for tensor safety.
    """

    def __init__(
        self,
        contract_path: str | Path,
        stats_path: str | Path,
        split: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        verify_fingerprints: bool = True,
        band_spec: BandSpec | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        self.contract = DatasetContract.load(contract_path)
        if verify_fingerprints:
            self.contract.verify_fingerprints(include_normalization=True)
        self.normalizer = RobustNormalizer(Path(stats_path), self.contract)
        self.split = split
        self.transform = transform
        self.band_spec = band_spec or BandSpec.resolve(self.contract, None)
        with self.contract.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        expected = int(self.contract.payload["sample_counts"][split])
        if len(self.rows) != expected:
            raise DatasetIntegrityError(
                f"Split {split} count changed: contract={expected}, manifest={len(self.rows)}"
            )
        self.event_ids = [row.get("source_event_id", "") for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, row: dict[str, str], group: str) -> Path:
        column = str(self.contract.group(group)["path_column"])
        return ensure_within(self.contract.dataset_root / row[column], self.contract.dataset_root)

    def _read(
        self, row: dict[str, str], group: str, indexes: tuple[int, ...] | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        path = self._path(row, group)
        expected = self.contract.group(group)
        with rasterio.open(path) as dataset:
            descriptions = list(dataset.descriptions)
            if descriptions != list(expected["band_descriptions"]):
                raise DatasetIntegrityError(
                    f"Band descriptions changed for {path}: {descriptions} != {expected['band_descriptions']}"
                )
            rasterio_indexes = None if indexes is None else [index + 1 for index in indexes]
            array = dataset.read(indexes=rasterio_indexes, masked=False)
            masks = dataset.read_masks(indexes=rasterio_indexes) > 0
            masks &= np.isfinite(array)
            if dataset.nodata is not None:
                if np.isnan(dataset.nodata):
                    masks &= ~np.isnan(array)
                else:
                    masks &= array != dataset.nodata
            metadata = {
                "path": str(path),
                "width": dataset.width,
                "height": dataset.height,
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "transform": tuple(float(value) for value in dataset.transform),
                "resolution": tuple(float(value) for value in dataset.res),
                "nodata": dataset.nodata,
                "grid": (
                    dataset.width,
                    dataset.height,
                    dataset.crs.to_wkt() if dataset.crs else None,
                    *tuple(float(value) for value in dataset.transform),
                ),
            }
        return array, masks, metadata

    @staticmethod
    def _duration(row: dict[str, str]) -> int:
        return max(
            1,
            (date.fromisoformat(row["event_end"]) - date.fromisoformat(row["event_start"])).days
            + 1,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        arrays: dict[str, np.ndarray] = {}
        validity: dict[str, np.ndarray] = {}
        metadata_by_group: dict[str, dict[str, Any]] = {}
        reference_grid: tuple[Any, ...] | None = None
        for group in self.contract.payload["raster_groups"]:
            selected_indexes = (
                self.band_spec.read_indexes(self.contract, group)
                if group in MODEL_CONTINUOUS_GROUPS and not (
                    group == "terrain"  # raw terrain/physics requires both available bands
                )
                else None
            )
            array, valid, metadata = self._read(row, group, selected_indexes)
            if reference_grid is None:
                reference_grid = metadata["grid"]
            elif metadata["grid"] != reference_grid:
                raise DatasetIntegrityError(
                    f"Sample {row['sample_id']} has a grid mismatch in {group}"
                )
            arrays[group] = array
            validity[group] = valid
            metadata_by_group[group] = metadata

        mask_names = list(self.contract.group("masks")["band_descriptions"])
        masks_np = {
            name: (arrays["masks"][band_index] > 0)[None]
            for band_index, name in enumerate(mask_names)
        }
        label_raster_valid = validity["label"][0]
        positive = masks_np["valid_depth_mask"][0]
        if not np.array_equal(label_raster_valid, positive):
            raise DatasetIntegrityError(
                f"Label validity differs from valid_depth_mask for {row['sample_id']}"
            )
        label = np.where(positive, arrays["label"][0], 0.0).astype(np.float32)[None]

        continuous: dict[str, np.ndarray] = {}
        branch_valid_fractions: dict[str, np.ndarray] = {}
        for group in MODEL_CONTINUOUS_GROUPS:
            descriptions = list(self.band_spec.names(group))
            read_indexes = self.band_spec.read_indexes(self.contract, group)
            positions = [read_indexes.index(index) for index in self.band_spec.indexes(group)]
            selected_array = arrays[group][positions]
            selected_validity = validity[group][positions]
            branch_valid_fractions[group] = selected_validity.astype(np.float32).mean(
                axis=0, keepdims=True
            )
            continuous[group] = self.normalizer.continuous(
                group, descriptions, selected_array, selected_validity
            )
        terrain_raw = np.where(validity["terrain"], arrays["terrain"], 0.0).astype(np.float32)
        # Raw Sentinel-1 dB values are exposed only inside the dedicated extent
        # namespace.  Learned depth models keep their strict top-level whitelist and
        # therefore cannot receive these extra tensors accidentally.  The published
        # AI4G flood-change thresholds are defined in dB, so reconstructing them from
        # normalized/clipped model channels would be scientifically ambiguous.
        # Extent consumers historically used this namespace.  For selected-band
        # depth runs it contains only actually-read S1 bands; the depth model never
        # receives it through ``prepare_model_inputs``.
        s1_t1_raw = np.where(validity["s1_t1"], arrays["s1_t1"], 0.0).astype(np.float32)
        s1_t2_raw = np.where(validity["s1_t2"], arrays["s1_t2"], 0.0).astype(np.float32)
        s1_pair_valid = np.logical_and.reduce(
            validity["s1_t1"] & validity["s1_t2"], axis=0
        ).astype(np.float32)

        duration = self._duration(row)
        s1_descriptions = list(self.contract.group("s1_qa")["band_descriptions"])
        s2_descriptions = list(self.contract.group("s2_qa")["band_descriptions"])
        s1_qa = np.zeros_like(arrays["s1_qa"], dtype=np.float32)
        s2_qa = np.zeros_like(arrays["s2_qa"], dtype=np.float32)

        def qa(group: str, descriptions: list[str], name: str, day: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            position = descriptions.index(name)
            raw = arrays[group][position].astype(np.float32)
            valid = validity[group][position].copy()
            missing = (~valid) | ((raw < 0) if day else False)
            if day:
                transformed = np.clip(raw / float(duration), 0.0, 1.0)
                valid &= raw >= 0
                raw_normalized = np.where(valid, transformed, 0.0).astype(np.float32)
            else:
                transformed = np.log1p(np.clip(raw, 0.0, None))
                raw_normalized = np.where(valid, transformed, 0.0).astype(np.float32)
            normalized = self.normalizer.qa_feature(group, name, transformed, valid)
            return normalized, raw_normalized, missing.astype(np.float32)

        s1_obs, _, _ = qa("s1_qa", s1_descriptions, "event_observation_count")
        s1_day, s1_day_raw, s1_missing = qa(
            "s1_qa", s1_descriptions, "selected_event_day_offset", day=True
        )
        s2_pre, _, _ = qa("s2_qa", s2_descriptions, "pre_clear_observation_count")
        s2_event, _, _ = qa("s2_qa", s2_descriptions, "event_clear_observation_count")
        s2_day, s2_day_raw, s2_missing = qa(
            "s2_qa", s2_descriptions, "selected_event_day_offset", day=True
        )
        s1_qa[s1_descriptions.index("event_observation_count")] = s1_obs
        s1_qa[s1_descriptions.index("selected_event_day_offset")] = s1_day
        s2_qa[s2_descriptions.index("pre_clear_observation_count")] = s2_pre
        s2_qa[s2_descriptions.index("event_clear_observation_count")] = s2_event
        s2_qa[s2_descriptions.index("selected_event_day_offset")] = s2_day

        # Semantic validity and GeoTIFF/per-band validity must both hold. This avoids
        # treating a tensor-safety zero inserted at raster nodata as a real observation.
        terrain_raster_valid = np.logical_and.reduce(validity["terrain"], axis=0)
        # Modality availability is a stable semantic mask from the contract.  It
        # deliberately does not change when a model band is removed; branch-level
        # fractions above carry the selected raster validity into each encoder.
        s1_valid = (
            masks_np["S1_event_composite_valid_mask"][0]
        ).astype(np.float32)
        s2_valid = (
            masks_np["S2_event_composite_valid_mask"][0]
        ).astype(np.float32)
        dem_valid = (
            masks_np["DEM_valid_mask"][0]
            & masks_np["slope_valid_mask"][0]
            & terrain_raster_valid
        ).astype(np.float32)
        sensor_days_present = (s1_missing == 0) & (s2_missing == 0)
        day_difference = np.where(
            sensor_days_present, np.abs(s1_day_raw - s2_day_raw), 1.0
        ).astype(np.float32)
        duration_feature = np.full_like(
            s1_valid, np.clip(np.log1p(duration) / np.log(32.0), 0.0, 2.0), dtype=np.float32
        )
        reliability = np.stack(
            [
                s1_obs,
                s1_day,
                s2_pre,
                s2_event,
                s2_day,
                s1_valid,
                s2_valid,
                dem_valid,
                duration_feature,
                day_difference,
                s1_missing,
                s2_missing,
            ],
            axis=0,
        ).astype(np.float32)

        conditioning_parts: list[np.ndarray] = []
        for source_group, source_index in self.band_spec.conditioning_sources(self.contract):
            read_indexes = self.band_spec.read_indexes(self.contract, source_group)
            position = read_indexes.index(source_index)
            name = str(self.contract.group(source_group)["band_descriptions"][source_index])
            conditioning_parts.append(
                self.normalizer.continuous(
                    source_group,
                    [name],
                    arrays[source_group][position : position + 1],
                    validity[source_group][position : position + 1],
                )
            )

        sample: dict[str, Any] = {
            "s1_t1": torch.from_numpy(continuous["s1_t1"]),
            "s1_t2": torch.from_numpy(continuous["s1_t2"]),
            "s1_change": torch.from_numpy(continuous["s1_change"]),
            "s1_qa": torch.from_numpy(s1_qa),
            "s2_t1": torch.from_numpy(continuous["s2_t1"]),
            "s2_t2": torch.from_numpy(continuous["s2_t2"]),
            "s2_change": torch.from_numpy(continuous["s2_change"]),
            "s2_qa": torch.from_numpy(s2_qa),
            "terrain": torch.from_numpy(continuous["terrain"]),
            "terrain_raw": torch.from_numpy(terrain_raw),
            "extent_inputs": {
                "s1_t1_db": torch.from_numpy(s1_t1_raw),
                "s1_t2_db": torch.from_numpy(s1_t2_raw),
                "s1_pair_valid": torch.from_numpy(s1_pair_valid[None]),
            },
            "label": torch.from_numpy(label),
            "masks": {
                name: torch.from_numpy(value.astype(np.float32)) for name, value in masks_np.items()
            },
            "validity": {
                "s1_valid": torch.from_numpy(s1_valid[None]),
                "s2_valid": torch.from_numpy(s2_valid[None]),
                "dem_valid": torch.from_numpy(dem_valid[None]),
                # Explicit availability aliases make the semantic/physical
                # validity contract unambiguous to augmentation and diagnostics.
                "s1_available": torch.from_numpy(s1_valid[None]),
                "s2_available": torch.from_numpy(s2_valid[None]),
                "dem_available": torch.from_numpy(dem_valid[None]),
                "output_valid": torch.from_numpy((dem_valid * np.maximum(s1_valid, s2_valid))[None]),
                "s1_t1_valid_fraction": torch.from_numpy(branch_valid_fractions["s1_t1"]),
                "s1_t2_valid_fraction": torch.from_numpy(branch_valid_fractions["s1_t2"]),
                "s1_change_valid_fraction": torch.from_numpy(branch_valid_fractions["s1_change"]),
                "s2_t1_valid_fraction": torch.from_numpy(branch_valid_fractions["s2_t1"]),
                "s2_t2_valid_fraction": torch.from_numpy(branch_valid_fractions["s2_t2"]),
                "s2_change_valid_fraction": torch.from_numpy(branch_valid_fractions["s2_change"]),
                "dem_valid_fraction": torch.from_numpy(
                    validity["terrain"].astype(np.float32).mean(axis=0, keepdims=True)
                ),
            },
            "reliability": torch.from_numpy(reliability),
            "metadata": {
                "sample_id": row["sample_id"],
                "source_event_id": row["source_event_id"],
                "event_chain_id": row["event_chain_id"],
                "event_start": row["event_start"],
                "event_end": row["event_end"],
                "event_duration_days": duration,
                "sample_origin": row.get("sample_origin", ""),
                "split": self.split,
                "crs": metadata_by_group["label"]["crs"],
                "transform": metadata_by_group["label"]["transform"],
                "resolution": metadata_by_group["label"]["resolution"],
                "width": metadata_by_group["label"]["width"],
                "height": metadata_by_group["label"]["height"],
                "label_path": metadata_by_group["label"]["path"],
                "reliability_names": RELIABILITY_NAMES,
                "resolved_model_bands": self.band_spec.as_dict(),
                "branch_valid_fractions": {
                    key: value.mean().item() for key, value in branch_valid_fractions.items()
                },
            },
        }
        if conditioning_parts:
            sample["s1_conditioning"] = torch.from_numpy(
                np.concatenate(conditioning_parts, axis=0).astype(np.float32)
            )
        return self.transform(sample) if self.transform is not None else sample


MODEL_INPUT_KEYS = (
    "s1_t1",
    "s1_t2",
    "s1_change",
    "s2_t1",
    "s2_t2",
    "s2_change",
    "terrain",
    "terrain_raw",
    "reliability",
)


def prepare_model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """Whitelist only label-independent inputs before calling ``model.forward``."""

    missing = [key for key in MODEL_INPUT_KEYS if key not in batch]
    if missing:
        raise KeyError(f"Batch is missing model inputs: {missing}")
    validity = batch.get("validity")
    if not isinstance(validity, dict):
        raise KeyError("Batch has no validity mapping")
    result = {
        **{key: batch[key] for key in MODEL_INPUT_KEYS},
        "s1_valid": validity["s1_valid"],
        "s2_valid": validity["s2_valid"],
        "dem_valid": validity["dem_valid"],
        "branch_validity": {
            "s1_t1": validity["s1_t1_valid_fraction"],
            "s1_t2": validity["s1_t2_valid_fraction"],
            "s1_change": validity["s1_change_valid_fraction"],
            "s2_t1": validity["s2_t1_valid_fraction"],
            "s2_t2": validity["s2_t2_valid_fraction"],
            "s2_change": validity["s2_change_valid_fraction"],
        },
    }
    if "s1_conditioning" in batch:
        result["s1_conditioning"] = batch["s1_conditioning"]
    return result

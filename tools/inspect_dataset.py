#!/usr/bin/env python3
"""Audit the immutable subset150 raster dataset and emit a runtime contract."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio

from datasets.contract import (
    DISABLED_QA_BANDS,
    EXPECTED_GROUPS,
    MODEL_CONTINUOUS_GROUPS,
    SAFE_QA_BANDS,
    ContractError,
    ensure_within,
    sha256_file,
)
from utils.misc import atomic_write_json, atomic_write_text


class AuditError(RuntimeError):
    """Fatal dataset-integrity or semantic-contract error."""


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _same_nodata(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    if math.isnan(left) and math.isnan(right):
        return True
    return left == right


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AuditError(f"Manifest is empty: {path}")
    required = {
        "sample_id",
        "filename",
        "split",
        "source_event_id",
        "event_chain_id",
        "event_start",
        "event_end",
        *(spec["path_column"] for spec in EXPECTED_GROUPS.values()),
    }
    missing = required.difference(rows[0])
    if missing:
        raise AuditError(f"Manifest is missing required columns: {sorted(missing)}")
    return rows


def _metadata_signature(dataset: rasterio.io.DatasetReader) -> dict[str, Any]:
    return {
        "width": dataset.width,
        "height": dataset.height,
        "band_count": dataset.count,
        "band_descriptions": [description for description in dataset.descriptions],
        "dtypes": list(dataset.dtypes),
        "nodata": _json_scalar(dataset.nodata),
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "resolution": [float(dataset.res[0]), float(dataset.res[1])],
    }


def _grid_signature(dataset: rasterio.io.DatasetReader) -> tuple[Any, ...]:
    return (
        dataset.width,
        dataset.height,
        dataset.crs.to_wkt() if dataset.crs else None,
        *tuple(float(value) for value in dataset.transform),
    )


def _inspect_values(dataset: rasterio.io.DatasetReader) -> dict[str, Any]:
    values = dataset.read(masked=False)
    band_masks = dataset.read_masks()
    finite = np.isfinite(values)
    valid = band_masks > 0
    if dataset.nodata is not None:
        if math.isnan(dataset.nodata):
            valid &= ~np.isnan(values)
        else:
            valid &= values != dataset.nodata
    valid &= finite
    bands: list[dict[str, Any]] = []
    for index in range(dataset.count):
        selected = values[index][valid[index]]
        bands.append(
            {
                "description": dataset.descriptions[index],
                "valid_pixels": int(selected.size),
                "invalid_pixels": int(selected.size - selected.size + values[index].size - selected.size),
                "nonfinite_pixels": int((~finite[index]).sum()),
                "minimum": float(selected.min()) if selected.size else None,
                "maximum": float(selected.max()) if selected.size else None,
            }
        )
    return {
        "dataset_mask_invalid_pixels": int((dataset.dataset_mask() == 0).sum()),
        "per_band_mask_invalid_pixels": [int((mask == 0).sum()) for mask in band_masks],
        "bands": bands,
        "values": values,
        "valid": valid,
    }


def _merge_band_stats(target: dict[str, Any], observed: dict[str, Any]) -> None:
    target["valid_pixels"] += observed["valid_pixels"]
    target["invalid_pixels"] += observed["invalid_pixels"]
    target["nonfinite_pixels"] += observed["nonfinite_pixels"]
    if observed["minimum"] is not None:
        target["minimum"] = (
            observed["minimum"]
            if target["minimum"] is None
            else min(target["minimum"], observed["minimum"])
        )
        target["maximum"] = (
            observed["maximum"]
            if target["maximum"] is None
            else max(target["maximum"], observed["maximum"])
        )


def audit_dataset(root: Path, samples_per_split: int = 3) -> tuple[dict[str, Any], dict[str, Any], str]:
    root = root.expanduser().resolve(strict=True)
    errors: list[str] = []
    warnings: list[str] = []
    differences: list[dict[str, Any]] = []

    readme = root / "README.md"
    metadata_dir = root / "metadata"
    manifest_path = metadata_dir / "training_manifest.csv"
    if not readme.is_file() or not manifest_path.is_file():
        raise AuditError("Dataset must contain README.md and metadata/training_manifest.csv")
    locks = sorted(
        path for path in metadata_dir.rglob("*") if path.is_file() and "write_lock" in path.name.lower()
    )
    if locks:
        raise AuditError(f"Dataset write lock(s) present: {[str(path) for path in locks]}")

    rows = _load_manifest(manifest_path)
    split_counts = Counter(row["split"] for row in rows)
    if set(split_counts) != {"train", "val", "test"}:
        errors.append(f"Unexpected split set: {sorted(split_counts)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample_id is not unique")

    selected_ids: set[str] = set()
    for split in ("train", "val", "test"):
        split_ids = [row["sample_id"] for row in rows if row["split"] == split]
        if not split_ids:
            continue
        indices = np.linspace(0, len(split_ids) - 1, min(samples_per_split, len(split_ids)), dtype=int)
        selected_ids.update(split_ids[int(index)] for index in indices)

    schemas: dict[str, dict[str, Any]] = {}
    aggregate: dict[str, list[dict[str, Any]]] = {}
    crs_counts: dict[str, Counter[str]] = defaultdict(Counter)
    detailed_samples: list[dict[str, Any]] = []
    seen_paths: dict[str, set[Path]] = defaultdict(set)
    label_mask_mismatch = 0
    flood_valid_mismatch = 0
    label_positive_outside_valid = 0
    mask_value_counts: dict[str, Counter[int]] = defaultdict(Counter)
    all_file_counts: dict[str, dict[str, int]] = defaultdict(dict)

    # Manifest counts and actual directory counts must agree before any semantic use.
    for split in ("train", "val", "test"):
        for group, spec in EXPECTED_GROUPS.items():
            rel_parents = {
                Path(row[spec["path_column"]]).parent
                for row in rows
                if row["split"] == split
            }
            actual = 0
            for parent in rel_parents:
                directory = ensure_within(root / parent, root)
                actual += len(list(directory.glob("*.tif")))
            expected = split_counts[split]
            all_file_counts[split][group] = actual
            if actual != expected:
                errors.append(
                    f"{split}/{group} file count mismatch: manifest={expected}, actual={actual}"
                )

    for row_number, row in enumerate(rows, start=2):
        split = row["split"]
        if split not in {"train", "val", "test"}:
            errors.append(f"Row {row_number}: invalid split {split!r}")
            continue
        try:
            datetime.fromisoformat(row["event_start"])
            datetime.fromisoformat(row["event_end"])
        except ValueError:
            errors.append(f"Row {row_number}: invalid event date(s)")
        if row["event_start"] > row["event_end"]:
            errors.append(f"Row {row_number}: event_start is after event_end")

        sample_detail: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "split": split,
            "source_event_id": row["source_event_id"],
            "event_chain_id": row["event_chain_id"],
            "event_start": row["event_start"],
            "event_end": row["event_end"],
            "rasters": {},
        }
        reference_grid: tuple[Any, ...] | None = None
        row_values: dict[str, dict[str, Any]] = {}

        for group, expected_spec in EXPECTED_GROUPS.items():
            relative = Path(row[expected_spec["path_column"]])
            try:
                path = ensure_within(root / relative, root)
            except (FileNotFoundError, ContractError) as exc:
                errors.append(f"Row {row_number}/{group}: {exc}")
                continue
            if path.name != row["filename"]:
                errors.append(
                    f"Row {row_number}/{group}: basename {path.name!r} != filename {row['filename']!r}"
                )
            if path in seen_paths[group]:
                errors.append(f"Row {row_number}/{group}: duplicate raster path {relative}")
            seen_paths[group].add(path)

            try:
                with rasterio.open(path) as dataset:
                    signature = _metadata_signature(dataset)
                    grid = _grid_signature(dataset)
                    if reference_grid is None:
                        reference_grid = grid
                    elif grid != reference_grid:
                        errors.append(
                            f"Row {row_number}/{group}: raster is not aligned with label grid ({relative})"
                        )
                    if group not in schemas:
                        schemas[group] = signature
                        aggregate[group] = [
                            {
                                "description": description,
                                "valid_pixels": 0,
                                "invalid_pixels": 0,
                                "nonfinite_pixels": 0,
                                "minimum": None,
                                "maximum": None,
                            }
                            for description in signature["band_descriptions"]
                        ]
                    else:
                        baseline = schemas[group]
                        for key in (
                            "width",
                            "height",
                            "band_count",
                            "band_descriptions",
                            "dtypes",
                            "resolution",
                        ):
                            if signature[key] != baseline[key]:
                                errors.append(
                                    f"Row {row_number}/{group}: inconsistent {key}: "
                                    f"expected {baseline[key]!r}, observed {signature[key]!r}"
                                )
                        if not _same_nodata(signature["nodata"], baseline["nodata"]):
                            errors.append(
                                f"Row {row_number}/{group}: inconsistent nodata: "
                                f"expected {baseline['nodata']!r}, observed {signature['nodata']!r}"
                            )
                    crs_counts[group][str(signature["crs"])] += 1
                    inspected = _inspect_values(dataset)
                    row_values[group] = inspected
                    for destination, source in zip(aggregate[group], inspected["bands"]):
                        _merge_band_stats(destination, source)
                    if row["sample_id"] in selected_ids:
                        sample_detail["rasters"][group] = {
                            **signature,
                            "relative_path": relative.as_posix(),
                            "transform": [float(value) for value in dataset.transform],
                            "dataset_mask_invalid_pixels": inspected[
                                "dataset_mask_invalid_pixels"
                            ],
                            "per_band_mask_invalid_pixels": inspected[
                                "per_band_mask_invalid_pixels"
                            ],
                            "bands": inspected["bands"],
                        }
            except (OSError, rasterio.errors.RasterioError) as exc:
                errors.append(f"Row {row_number}/{group}: cannot read {relative}: {exc}")

        if "masks" in row_values:
            mask_values = row_values["masks"]["values"]
            descriptions = schemas["masks"]["band_descriptions"]
            for band_index, description in enumerate(descriptions):
                unique, counts = np.unique(mask_values[band_index], return_counts=True)
                for value, count in zip(unique, counts):
                    mask_value_counts[str(description)][int(value)] += int(count)
            valid_index = descriptions.index("valid_depth_mask")
            flood_index = descriptions.index("flood_mask")
            valid_depth = mask_values[valid_index] > 0
            flood = mask_values[flood_index] > 0
            flood_valid_mismatch += int(np.count_nonzero(valid_depth != flood))
            if "label" in row_values:
                label_data = row_values["label"]["values"][0]
                label_valid = row_values["label"]["valid"][0]
                label_mask_mismatch += int(np.count_nonzero(label_valid != valid_depth))
                label_positive_outside_valid += int(
                    np.count_nonzero((label_data > 0) & (~valid_depth) & label_valid)
                )
        if row["sample_id"] in selected_ids:
            detailed_samples.append(sample_detail)

    for group, expected_spec in EXPECTED_GROUPS.items():
        if group not in schemas:
            errors.append(f"No readable rasters for required group {group}")
            continue
        observed = schemas[group]["band_descriptions"]
        expected = expected_spec["bands"]
        if observed != expected:
            differences.append(
                {
                    "field": f"{group}.band_descriptions",
                    "expected": expected,
                    "observed": observed,
                    "decision": "fatal: scientific semantics cannot be inferred from reordered/renamed bands",
                }
            )
            errors.append(f"{group} band descriptions differ from the expected semantic contract")

    observed_channels = sum(schemas[name]["band_count"] for name in MODEL_CONTINUOUS_GROUPS if name in schemas)
    if observed_channels != 27:
        errors.append(f"Expected 27 continuous model-input channels, observed {observed_channels}")

    allowed_mask_values = {0, 1, 255}
    for description, counts in mask_value_counts.items():
        unexpected = set(counts).difference(allowed_mask_values)
        if unexpected:
            errors.append(f"Mask {description} contains unexpected values: {sorted(unexpected)}")
    mask_encoding = "0 for false and 1 or 255 for true; runtime converts values > 0 to boolean"
    differences.append(
        {
            "field": "mask_encoding",
            "expected": "binary semantics",
            "observed": sorted({value for counts in mask_value_counts.values() for value in counts}),
            "decision": mask_encoding,
        }
    )
    if label_mask_mismatch:
        errors.append(f"Label raster validity differs from valid_depth_mask at {label_mask_mismatch} pixels")
    if flood_valid_mismatch:
        errors.append(f"flood_mask differs from valid_depth_mask at {flood_valid_mismatch} pixels")

    key_files = [readme, manifest_path]
    key_files.extend(sorted(metadata_dir.glob("*.json")))
    subset_manifest = metadata_dir / "subset_manifest.csv"
    if subset_manifest.is_file():
        key_files.append(subset_manifest)
    key_hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in key_files}

    normalization_path = metadata_dir / "normalization_stats_subset.json"
    provided_normalization: dict[str, Any] = {
        "path": normalization_path.relative_to(root).as_posix()
        if normalization_path.is_file()
        else None,
        "sha256": sha256_file(normalization_path) if normalization_path.is_file() else None,
        "accepted": False,
        "reason": "missing",
    }
    if normalization_path.is_file():
        try:
            normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
            scope = str(normalization.get("scope", "")).lower()
            descriptions_match = True
            all_std_positive = True
            quantiles_present = True
            json_group_names = {
                "terrain": "DEM",
                "s1_t1": "S1_T1",
                "s1_t2": "S1_T2",
                "s1_change": "S1_change",
                "s2_t1": "S2_T1",
                "s2_t2": "S2_T2",
                "s2_change": "S2_change",
            }
            for group, json_name in json_group_names.items():
                entries = normalization.get("groups", {}).get(json_name, [])
                descriptions_match &= [entry.get("band") for entry in entries] == schemas[group][
                    "band_descriptions"
                ]
                all_std_positive &= all(float(entry.get("std", 0.0)) > 0 for entry in entries)
                quantiles_present &= all(
                    "p0.5" in entry and "p99.5" in entry for entry in entries
                )
            conditions = {
                "explicit_train_only_valid_pixels": "train" in scope
                and "val/test excluded" in scope
                and "valid" in scope,
                "train_sample_count_matches": normalization.get("train_samples")
                == split_counts["train"],
                "band_descriptions_match": bool(descriptions_match),
                "all_std_positive": bool(all_std_positive),
                "robust_quantiles_present": bool(quantiles_present),
            }
            provided_normalization["conditions"] = conditions
            provided_normalization["accepted"] = all(conditions.values())
            provided_normalization["reason"] = (
                "accepted"
                if provided_normalization["accepted"]
                else "rebuild train-only stats because one or more required conditions failed"
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            provided_normalization["reason"] = f"invalid normalization metadata: {exc}"

    for group, band_stats in aggregate.items():
        schemas[group]["aggregate_values"] = band_stats
        schemas[group]["observed_crs_counts"] = dict(crs_counts[group])
        schemas[group]["path_column"] = EXPECTED_GROUPS[group]["path_column"]
        schemas[group]["role"] = EXPECTED_GROUPS[group]["role"]

    status = "ready" if not errors else "failed"
    timestamp = datetime.now(timezone.utc).isoformat()
    audit: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": timestamp,
        "status": status,
        "dataset_root": str(root),
        "source_readme": readme.read_text(encoding="utf-8"),
        "manifest": {
            "relative_path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
            "row_count": len(rows),
            "columns": list(rows[0]),
            "split_counts": dict(split_counts),
            "unique_source_events": len({row["source_event_id"] for row in rows}),
            "unique_event_chains": len({row["event_chain_id"] for row in rows}),
            "sample_origins": dict(Counter(row.get("sample_origin", "") for row in rows)),
            "date_range": [min(row["event_start"] for row in rows), max(row["event_end"] for row in rows)],
        },
        "actual_file_counts": dict(all_file_counts),
        "raster_groups": schemas,
        "mask_value_counts": {
            description: {str(value): count for value, count in sorted(counts.items())}
            for description, counts in mask_value_counts.items()
        },
        "semantic_checks": {
            "main_continuous_input_channels": observed_channels,
            "label_description": schemas.get("label", {}).get("band_descriptions", [None])[0],
            "label_unit_decision": "depth_m is already in metres; no division by 100",
            "label_raster_vs_valid_depth_mask_mismatch_pixels": label_mask_mismatch,
            "flood_mask_vs_valid_depth_mask_mismatch_pixels": flood_valid_mismatch,
            "positive_label_outside_valid_pixels": label_positive_outside_valid,
            "event_period_semantics": "S1/T2 and S2/T2 are per-pixel event-period composites and are not synchronous single acquisitions",
            "task_semantics": "event-aggregated flood-depth estimation with partial positive depth labels",
        },
        "provided_normalization": provided_normalization,
        "key_file_sha256": key_hashes,
        "sampled_raster_details": detailed_samples,
        "expected_observed_differences": differences,
        "warnings": warnings,
        "errors": errors,
    }

    contract: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": timestamp,
        "status": status,
        "dataset_root": str(root),
        "dataset_name": "subset150",
        "manifest": audit["manifest"],
        "sample_counts": dict(split_counts),
        "patch": {
            "width": schemas.get("label", {}).get("width"),
            "height": schemas.get("label", {}).get("height"),
            "resolution": schemas.get("label", {}).get("resolution"),
            "crs_values": sorted(crs_counts.get("label", {})),
        },
        "raster_groups": schemas,
        "main_continuous_groups": list(MODEL_CONTINUOUS_GROUPS),
        "main_continuous_input_channels": observed_channels,
        "safe_qa_bands": SAFE_QA_BANDS,
        "disabled_qa_bands": DISABLED_QA_BANDS,
        "mask_encoding": {"observed_values": sorted({value for counts in mask_value_counts.values() for value in counts}), "true_rule": "value > 0"},
        "label": {
            "band": "depth_m",
            "unit": "metre",
            "nodata": schemas.get("label", {}).get("nodata"),
            "supervision_mask": "valid_depth_mask",
            "invalid_tensor_fill": 0.0,
        },
        "normalization": {
            "provided": provided_normalization,
            "selected": None,
            "required_generated_path": "artifacts/dataset_audit/subset150_train_stats.json",
        },
        "key_file_sha256": key_hashes,
        "scientific_semantics": {
            "t2": "per-pixel event-period composites; pixels may come from different dates",
            "cross_sensor_time": "S1/T2 and S2/T2 are generally asynchronous",
            "temporal_derivative_forbidden": True,
            "label_scope": "only valid_depth_mask pixels carry reliable continuous positive-depth supervision",
            "unknown_is_zero": False,
            "terrain": "elevation is DSM, not bare-earth DTM",
            "model_claim": "physics-guided event-scale reconstruction, not a shallow-water-equation PINN",
        },
    }

    report_lines = [
        "# subset150 dataset audit",
        "",
        f"- Status: **{status}**",
        f"- Audited at (UTC): `{timestamp}`",
        f"- Dataset root: `{root}`",
        f"- Manifest rows: {len(rows)} (train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']})",
        f"- Source events / event chains: {audit['manifest']['unique_source_events']} / {audit['manifest']['unique_event_chains']}",
        f"- Raster shape/resolution/CRS: {contract['patch']['height']}×{contract['patch']['width']}, {contract['patch']['resolution']} m, {contract['patch']['crs_values']}",
        f"- Main continuous channels: {observed_channels}",
        "- Label: `depth_m`, metres, nodata=-9999; invalid values are never supervision.",
        f"- Masks: 10 uint8 bands, observed encoding {contract['mask_encoding']['observed_values']}; loader uses `value > 0`.",
        f"- Provided normalization accepted: {provided_normalization['accepted']} ({provided_normalization['reason']}).",
        "",
        "## Audited raster groups",
        "",
        "| Group | Bands | dtype | nodata | role |",
        "|---|---:|---|---:|---|",
    ]
    for group, schema in schemas.items():
        report_lines.append(
            f"| `{group}` | {schema['band_count']} | {', '.join(schema['dtypes'])} | {schema['nodata']} | {schema['role']} |"
        )
    report_lines.extend(
        [
            "",
            "## Expected/observed decisions",
            "",
        ]
    )
    for difference in differences:
        report_lines.append(
            f"- `{difference['field']}`: expected {difference['expected']}; observed {difference['observed']}; decision: {difference['decision']}."
        )
    report_lines.extend(
        [
            "",
            "## Scientific contract",
            "",
            "S1/T2 and S2/T2 are event-period, per-pixel composites. They are not single scenes, pixels inside one patch may use different dates, and S1/S2 are generally asynchronous. T1/T2 therefore must not be treated as a regular time series or a temporal derivative.",
            "",
            "Only `valid_depth_mask=1` pixels provide reliable positive continuous-depth supervision. `unknown`, permanent water and extreme-high areas are not 0 m negatives. `flood_mask` equals `valid_depth_mask` in this subset and is not a complete wet/dry extent label.",
            "",
            "DSM is not DTM. The planned model is physics-guided event-scale reconstruction and does not solve the shallow-water equations.",
            "",
            "## Integrity",
            "",
            f"- Label raster validity mismatches: {label_mask_mismatch}",
            f"- `flood_mask` / `valid_depth_mask` mismatches: {flood_valid_mismatch}",
            f"- Fatal errors: {len(errors)}",
        ]
    )
    if errors:
        report_lines.extend(["", "## Errors", ""] + [f"- {error}" for error in errors])
    return audit, contract, "\n".join(report_lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Immutable dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Project audit output directory")
    parser.add_argument("--samples-per-split", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit, contract, report = audit_dataset(args.root, max(2, args.samples_per_split))
    except (AuditError, ContractError, FileNotFoundError) as exc:
        print(f"DATASET AUDIT FAILED: {exc}", file=sys.stderr)
        return 2
    output = args.output.expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing_contract_path = output / "subset150_contract.json"
    if existing_contract_path.is_file():
        try:
            existing = json.loads(existing_contract_path.read_text(encoding="utf-8"))
            selected = existing.get("normalization", {}).get("selected")
            if selected and selected.get("path") and selected.get("sha256"):
                selected_path = Path(str(selected["path"])).expanduser().resolve(strict=True)
                if (
                    sha256_file(selected_path) == selected["sha256"]
                    and selected.get("manifest_sha256") == contract["manifest"]["sha256"]
                ):
                    contract["normalization"]["selected"] = selected
        except (OSError, ValueError, json.JSONDecodeError):
            # A stale/invalid prior project contract is never trusted or required.
            pass
    atomic_write_json(output / "subset150_audit.json", audit)
    atomic_write_json(output / "subset150_contract.json", contract)
    atomic_write_text(output / "subset150_report.md", report)
    print(
        f"Dataset audit status={audit['status']}; "
        f"rows={audit['manifest']['row_count']}; "
        f"outputs={output}"
    )
    if audit["errors"]:
        for error in audit["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

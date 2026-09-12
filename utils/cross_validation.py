"""Leakage-safe event grouping and fold-specific configuration generation."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

from utils.config import jsonable_config, load_config
from utils.experiment_catalog import experiment_id
from utils.misc import atomic_write_text
from utils.model_dispatch import configured_model_kind


SOURCE_SPLITS = ("train", "val", "test")
ALL_SPLITS = frozenset(SOURCE_SPLITS)
RUNTIME_SOURCE_DIRECTORIES = (
    "compare",
    "datasets",
    "losses",
    "metrics",
    "models",
    "tools",
    "utils",
)
RUNTIME_ROOT_FILES = (
    "train.py",
    "evaluate.py",
    "validate_k_fold.py",
)


def resolved_config_sha256(path: Path) -> str:
    """Hash the fully merged semantic configuration, including all includes."""

    payload = json.dumps(
        jsonable_config(load_config(path)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_source_sha256(project_root: Path) -> tuple[str, int]:
    """Fingerprint Python sources that can affect a cross-validation job."""

    root = project_root.expanduser().resolve(strict=True)
    paths = [
        root / name
        for name in RUNTIME_ROOT_FILES
        if (root / name).is_file()
    ]
    for directory_name in RUNTIME_SOURCE_DIRECTORIES:
        directory = root / directory_name
        if directory.is_dir():
            paths.extend(directory.rglob("*.py"))
    selected = sorted(set(path.resolve() for path in paths))
    if not selected:
        raise ValueError(f"No runtime Python sources found below {root}")
    digest = hashlib.sha256()
    for path in selected:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), len(selected)


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    source = path.expanduser().resolve(strict=True)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {source}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    required = {"sample_id", "source_event_id", "split"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    if not rows:
        raise ValueError("Manifest contains no samples")
    return rows, fieldnames


def validate_source_events(rows: Sequence[Mapping[str, str]]) -> None:
    """Validate event identifiers and original split labels before regrouping."""

    for row in rows:
        event = str(row.get("source_event_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if not event:
            raise ValueError(f"Sample {row.get('sample_id')} has no source_event_id")
        if split not in ALL_SPLITS:
            raise ValueError(f"Sample {row.get('sample_id')} has invalid split {split!r}")


def assign_event_folds(
    rows: Sequence[Mapping[str, str]],
    k: int,
    *,
    seed: int,
    source_splits: Iterable[str] = SOURCE_SPLITS,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Assign balanced outer-test folds with indivisible source events."""

    if k < 2:
        raise ValueError("k must be at least 2")
    validate_source_events(rows)
    included_splits = frozenset(str(split) for split in source_splits)
    if not included_splits or not included_splits.issubset(ALL_SPLITS):
        raise ValueError(f"Invalid cross-validation source splits: {sorted(included_splits)}")
    counts = Counter(
        str(row["source_event_id"])
        for row in rows
        if str(row["split"]) in included_splits
    )
    if k > len(counts):
        raise ValueError(f"k={k} exceeds the {len(counts)} source events")

    def tie_break(event: str) -> str:
        return hashlib.sha256(f"{seed}:{event}".encode("utf-8")).hexdigest()

    ordered = sorted(counts.items(), key=lambda item: (-item[1], tie_break(item[0])))
    loads = [0] * k
    fold_events: list[list[str]] = [[] for _ in range(k)]
    assignment: dict[str, int] = {}
    for event, sample_count in ordered:
        fold = min(range(k), key=lambda index: (loads[index], len(fold_events[index]), index))
        assignment[event] = fold
        fold_events[fold].append(event)
        loads[fold] += int(sample_count)
    balance = [
        {
            "fold": index + 1,
            "outer_test_samples": loads[index],
            "outer_test_events": len(fold_events[index]),
            "remaining_samples": int(sum(loads) - loads[index]),
            "remaining_events": int(len(counts) - len(fold_events[index])),
        }
        for index in range(k)
    ]
    return assignment, balance


def select_inner_validation_events(
    rows: Sequence[Mapping[str, str]],
    assignment: Mapping[str, int],
    *,
    outer_test_fold: int,
    fraction: float,
    seed: int,
    source_splits: Iterable[str] = SOURCE_SPLITS,
) -> tuple[frozenset[str], dict[str, Any]]:
    """Select an event-disjoint inner validation set nearest a sample target.

    A deterministic subset-sum solver is practical here because the full sample
    pool contains only a few thousand patches. It fixes the requested event
    count, minimizes the difference from the requested sample count, and never
    splits a source event.
    """

    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("inner validation fraction must lie strictly between 0 and 1")
    if outer_test_fold < 0:
        raise ValueError("outer_test_fold must be zero-based and non-negative")
    included_splits = frozenset(str(split) for split in source_splits)
    counts = Counter(
        str(row["source_event_id"])
        for row in rows
        if str(row["split"]) in included_splits
    )
    missing = sorted(set(counts).difference(assignment))
    if missing:
        raise ValueError(f"Outer-fold assignment is missing events: {missing[:5]}")
    candidates = {
        event: int(sample_count)
        for event, sample_count in counts.items()
        if int(assignment[event]) != outer_test_fold
    }
    if len(candidates) < 2:
        raise ValueError("At least two non-test events are required for train/validation")
    total_samples = sum(candidates.values())
    target_samples = min(
        total_samples - 1,
        max(1, int(round(total_samples * float(fraction)))),
    )

    def tie_break(event: str) -> str:
        return hashlib.sha256(f"{seed}:{event}".encode("utf-8")).hexdigest()

    ordered = sorted(candidates.items(), key=lambda item: (tie_break(item[0]), item[0]))
    target_events = min(
        len(ordered) - 1,
        max(1, int(round(len(ordered) * float(fraction)))),
    )
    # Each Python integer is a compact bitset whose bit s indicates that sample
    # sum s is reachable.  Prefix states allow deterministic reconstruction while
    # constraining both the event count and sample count of inner validation.
    prefix: list[list[int]] = [[0] * (target_events + 1)]
    prefix[0][0] = 1
    for item_index, (_, sample_count) in enumerate(ordered, start=1):
        previous = prefix[-1]
        current = previous.copy()
        for event_count in range(min(item_index, target_events), 0, -1):
            current[event_count] |= previous[event_count - 1] << sample_count
        prefix.append(current)

    reachable = prefix[-1][target_events]
    feasible = [
        value
        for value in range(1, total_samples)
        if (reachable >> value) & 1
    ]
    if not feasible:
        raise RuntimeError(
            "No event/sample-balanced inner validation subset exists"
        )
    selected_samples = min(
        feasible,
        key=lambda value: (
            abs(value - target_samples),
            value < target_samples,
            value,
        ),
    )
    selected_indexes: list[int] = []
    cursor_samples = selected_samples
    cursor_events = target_events
    for item_index in range(len(ordered), 0, -1):
        if cursor_events == 0:
            break
        sample_count = ordered[item_index - 1][1]
        previous_reachable = prefix[item_index - 1][cursor_events - 1]
        include = cursor_samples >= sample_count and bool(
            previous_reachable & (1 << (cursor_samples - sample_count))
        )
        if include:
            selected_indexes.append(item_index - 1)
            cursor_samples -= sample_count
            cursor_events -= 1
    if cursor_samples != 0 or cursor_events != 0:
        raise RuntimeError("Inner-validation subset reconstruction failed")
    selected = frozenset(ordered[index][0] for index in selected_indexes)
    if not selected or len(selected) == len(candidates):
        raise RuntimeError("Inner validation must leave non-empty train and val events")
    return selected, {
        "target_samples": target_samples,
        "target_events": target_events,
        "selected_samples": selected_samples,
        "selected_events": len(selected),
        "remaining_samples": total_samples,
        "remaining_events": len(candidates),
        "requested_fraction": float(fraction),
        "realized_fraction": selected_samples / total_samples,
    }


def plan_cross_validation_folds(
    rows: Sequence[Mapping[str, str]],
    assignment: Mapping[str, int],
    k: int,
    *,
    inner_validation_fraction: float,
    seed: int,
    seed_stride: int,
    source_splits: Iterable[str] = SOURCE_SPLITS,
) -> tuple[dict[int, frozenset[str]], list[dict[str, Any]]]:
    """Plan train/inner-validation/outer-test roles for every outer fold."""

    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")
    included_splits = frozenset(str(split) for split in source_splits)
    event_counts = Counter(
        str(row["source_event_id"])
        for row in rows
        if str(row["split"]) in included_splits
    )
    if not event_counts:
        raise ValueError("No source events are available for cross-validation")
    plans: dict[int, frozenset[str]] = {}
    balance: list[dict[str, Any]] = []
    for outer_test_fold in range(k):
        inner_events, selection = select_inner_validation_events(
            rows,
            assignment,
            outer_test_fold=outer_test_fold,
            fraction=inner_validation_fraction,
            seed=seed + (outer_test_fold + 1) * seed_stride,
            source_splits=included_splits,
        )
        outer_events = {
            event for event, fold in assignment.items() if int(fold) == outer_test_fold
        }
        train_events = set(event_counts).difference(outer_events, inner_events)
        if not train_events or not inner_events or not outer_events:
            raise RuntimeError(
                f"Fold {outer_test_fold + 1} has an empty train/val/test event role"
            )
        plans[outer_test_fold] = inner_events
        balance.append(
            {
                "fold": outer_test_fold + 1,
                "training_samples": sum(event_counts[event] for event in train_events),
                "training_events": len(train_events),
                "inner_validation_target_samples": int(selection["target_samples"]),
                "inner_validation_target_events": int(selection["target_events"]),
                "inner_validation_samples": sum(
                    event_counts[event] for event in inner_events
                ),
                "inner_validation_events": len(inner_events),
                "inner_validation_fraction_of_remaining": float(
                    selection["realized_fraction"]
                ),
                "outer_test_samples": sum(event_counts[event] for event in outer_events),
                "outer_test_events": len(outer_events),
                "source_samples": sum(event_counts.values()),
                "source_events": len(event_counts),
            }
        )
    return plans, balance


def assignment_rows(
    rows: Sequence[Mapping[str, str]],
    assignment: Mapping[str, int],
    *,
    source_splits: Iterable[str] = SOURCE_SPLITS,
) -> list[dict[str, Any]]:
    included_splits = frozenset(str(split) for split in source_splits)
    original_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sample_counts = Counter(
        str(row["source_event_id"])
        for row in rows
        if str(row["split"]) in included_splits
    )
    for row in rows:
        if str(row["split"]) in included_splits:
            original_split_counts[str(row["source_event_id"])][str(row["split"])] += 1
    return [
        {
            "source_event_id": event,
            "fold": fold + 1,
            "samples": sample_counts[event],
            "original_train_samples": original_split_counts[event]["train"],
            "original_val_samples": original_split_counts[event]["val"],
            "original_test_samples": original_split_counts[event]["test"],
        }
        for event, fold in sorted(assignment.items(), key=lambda item: (item[1], item[0]))
    ]


def expected_fold_roles(
    rows: Sequence[Mapping[str, str]],
    assignment: Mapping[str, int],
    outer_test_fold: int,
    inner_validation_events: Iterable[str],
    source_splits: Iterable[str] = SOURCE_SPLITS,
) -> dict[str, str]:
    """Return the exact sample-to-role mapping for one nested outer fold."""

    if outer_test_fold < 0:
        raise ValueError("outer_test_fold must be zero-based and non-negative")
    inner_events = frozenset(str(event) for event in inner_validation_events)
    included_splits = frozenset(str(split) for split in source_splits)
    outer_events = {
        str(event)
        for event, fold in assignment.items()
        if int(fold) == outer_test_fold
    }
    overlap = sorted(inner_events.intersection(outer_events))
    if overlap:
        raise ValueError(
            f"Events cannot be both inner validation and outer test: {overlap[:5]}"
        )
    unknown_inner = sorted(inner_events.difference(assignment))
    if unknown_inner:
        raise ValueError(f"Unknown inner-validation events: {unknown_inner[:5]}")
    roles: dict[str, str] = {}
    for row in rows:
        if str(row["split"]) not in included_splits:
            continue
        sample_id = str(row["sample_id"])
        if not sample_id or sample_id in roles:
            raise ValueError(f"Missing or duplicate sample_id: {sample_id!r}")
        event = str(row["source_event_id"])
        if event not in assignment:
            raise ValueError(f"Outer-fold assignment is missing event {event!r}")
        roles[sample_id] = (
            "test"
            if int(assignment[event]) == outer_test_fold
            else "val"
            if event in inner_events
            else "train"
        )
    if set(roles.values()) != {"train", "val", "test"}:
        raise ValueError("Every fold must have non-empty train, val, and test roles")
    return roles


def validate_fold_manifest_rows(
    fold_rows: Sequence[Mapping[str, str]],
    source_rows: Sequence[Mapping[str, str]],
    assignment: Mapping[str, int],
    outer_test_fold: int,
    inner_validation_events: Iterable[str],
    source_splits: Iterable[str] = SOURCE_SPLITS,
) -> dict[str, int]:
    """Fail if a persisted fold manifest differs from the deterministic plan."""

    included_splits = frozenset(str(split) for split in source_splits)
    expected = expected_fold_roles(
        source_rows,
        assignment,
        outer_test_fold,
        inner_validation_events,
        included_splits,
    )
    source_by_sample = {
        str(row["sample_id"]): row
        for row in source_rows
        if str(row["split"]) in included_splits
    }
    observed: dict[str, str] = {}
    event_roles: dict[str, set[str]] = defaultdict(set)
    changed_metadata: list[str] = []
    for row in fold_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in observed:
            raise ValueError(f"Fold manifest has a missing/duplicate sample: {sample_id!r}")
        split = str(row.get("split", ""))
        observed[sample_id] = split
        event_roles[str(row.get("source_event_id", ""))].add(split)
        source_row = source_by_sample.get(sample_id)
        if source_row is not None and any(
            str(row.get(name, "")) != str(value)
            for name, value in source_row.items()
            if name != "split"
        ):
            changed_metadata.append(sample_id)
    if observed != expected:
        missing = sorted(set(expected).difference(observed))[:5]
        extra = sorted(set(observed).difference(expected))[:5]
        changed = sorted(
            sample_id
            for sample_id in set(expected).intersection(observed)
            if expected[sample_id] != observed[sample_id]
        )[:5]
        raise ValueError(
            "Fold manifest differs from its deterministic sample-role plan: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    if changed_metadata:
        raise ValueError(
            "Fold manifest changed immutable source metadata for samples: "
            f"{changed_metadata[:5]}"
        )
    leaking = {
        event: sorted(roles)
        for event, roles in event_roles.items()
        if len(roles) != 1
    }
    if leaking:
        raise ValueError(f"Events cross fold roles: {list(leaking.items())[:5]}")
    return {
        split: sum(role == split for role in observed.values())
        for split in ("train", "val", "test")
    }


def write_fold_manifest(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
    assignment: Mapping[str, int],
    outer_test_fold: int,
    inner_validation_events: Iterable[str],
    source_splits: Iterable[str] = SOURCE_SPLITS,
) -> dict[str, int]:
    """Write one CV manifest after regrouping every configured source split."""

    included_splits = frozenset(str(split) for split in source_splits)
    roles = expected_fold_roles(
        rows,
        assignment,
        outer_test_fold,
        inner_validation_events,
        included_splits,
    )
    output_rows: list[dict[str, str]] = []
    for source_row in rows:
        if str(source_row["split"]) not in included_splits:
            continue
        row = dict(source_row)
        row["split"] = roles[str(row["sample_id"])]
        output_rows.append(row)
    counts = {
        split: sum(row["split"] == split for row in output_rows)
        for split in ("train", "val", "test")
    }
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    writer.writerows(output_rows)
    atomic_write_text(path, buffer.getvalue())
    return counts


def _typed_element(parent: ET.Element, name: str, type_name: str, value: Any) -> None:
    child = ET.SubElement(parent, name, {"type": type_name})
    child.text = "none" if value is None else str(value)


def _typed_list(parent: ET.Element, name: str, values: Iterable[str]) -> None:
    child = ET.SubElement(parent, name, {"type": "list", "item_type": "str"})
    for value in values:
        item = ET.SubElement(child, "item")
        item.text = str(value)


def write_fold_config(
    source_config: Path,
    destination: Path,
    *,
    contract: Path,
    manifest: Path,
    train_stats: Path,
    training_output: Path,
    session_root: Path,
    k: int,
    fold: int,
    seed: int,
    source_splits: Iterable[str] = SOURCE_SPLITS,
    inner_validation_fraction: float = 0.125,
    resume_checkpoint: Path | None = None,
) -> Path:
    """Write a small XML overlay that keeps all model hyperparameters inherited."""

    source = source_config.expanduser().resolve(strict=True)
    resolved = load_config(source)
    source_config_sha256 = resolved_config_sha256(source)
    kind, _ = configured_model_kind(resolved)
    if kind == "traditional":
        raise ValueError("Traditional methods are not trainable k-fold experiments")
    root = ET.Element("config")
    ET.SubElement(root, "include", {"path": str(source)})
    _typed_element(root, "runs_root", "path", session_root)
    _typed_element(root, "artifacts_root", "path", training_output.parent / "artifacts")
    _typed_element(root, "seed", "int", seed)

    dataset = ET.SubElement(root, "dataset")
    _typed_element(dataset, "contract", "path", contract)
    _typed_element(dataset, "manifest", "path", manifest)
    _typed_element(dataset, "train_stats", "path", train_stats)

    runtime = ET.SubElement(root, "runtime")
    train = ET.SubElement(runtime, "train")
    _typed_element(train, "output", "path", training_output)
    _typed_element(
        train,
        "resume",
        "path" if resume_checkpoint is not None else "none",
        resume_checkpoint,
    )
    _typed_element(train, "allow_existing_output", "bool", "false")
    evaluation = ET.SubElement(runtime, "evaluation")
    _typed_element(evaluation, "split", "str", "test")

    if kind == "pa_hydrokan":
        loss = ET.SubElement(root, "loss")
        _typed_element(
            loss,
            "frozen_depth_weights_artifact",
            "path",
            training_output / "frozen_depth_weights.json",
        )

    cross_validation = ET.SubElement(root, "cross_validation")
    _typed_element(cross_validation, "k", "int", k)
    _typed_element(cross_validation, "fold", "int", fold)
    _typed_element(cross_validation, "split_unit", "str", "source_event_id")
    _typed_element(
        cross_validation,
        "protocol",
        "str",
        "outer_k_fold_with_event_grouped_inner_holdout",
    )
    _typed_list(cross_validation, "source_splits", source_splits)
    _typed_element(
        cross_validation,
        "inner_validation_fraction",
        "float",
        inner_validation_fraction,
    )
    _typed_element(
        cross_validation,
        "source_config",
        "path",
        source,
    )
    _typed_element(
        cross_validation,
        "source_config_sha256",
        "str",
        source_config_sha256,
    )
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode", xml_declaration=False)
    atomic_write_text(destination, xml + "\n")
    return destination


def config_inventory(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        config = load_config(path)
        rows.append(
            {
                "model": experiment_id(config),
                "config": str(path.resolve()),
                "resolved_config_sha256": resolved_config_sha256(path),
            }
        )
    return rows

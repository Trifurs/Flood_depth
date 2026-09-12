#!/usr/bin/env python3
"""Run nested event-grouped k-fold testing for every deep-learning experiment.

Usage:
    python validate_k_fold.py          # k=5
    python validate_k_fold.py 10
    python validate_k_fold.py --resume
    python validate_k_fold.py --resume runs/.../cross_validation/k5/<session>

All samples from the original train, validation, and test splits are regrouped
by ``source_event_id`` into balanced outer folds.  In each run, one outer fold
is test-only; the other folds are split again by event into training and an
inner early-stopping validation set.  Every sample is outer-tested exactly once.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from datasets.contract import DatasetContract, sha256_file
from tools.prepare_flooddepthnet_s1_terrain_assets import build_assets
from utils.config import load_config
from utils.cross_validation import (
    assign_event_folds,
    assignment_rows,
    config_inventory,
    plan_cross_validation_folds,
    read_manifest,
    runtime_source_sha256,
    validate_fold_manifest_rows,
    write_fold_config,
    write_fold_manifest,
)
from utils.experiment_catalog import (
    deep_learning_config_paths,
    experiment_display_name,
    experiment_id,
)
from utils.logging import setup_logging
from utils.misc import atomic_write_json
from utils.model_dispatch import configured_model_kind
from utils.reporting import numerical_summary_rows, scalar_row, write_union_rows
from utils.run_paths import runtime_section, started_at_run_id
from utils.tensorboard import add_metadata, add_scalars, create_summary_writer, flush


LOGGER = logging.getLogger("cross_validation")
PROJECT_ROOT = Path(__file__).resolve().parent
JOB_WORKER = PROJECT_ROOT / "tools" / "_cross_validation_job.py"
TRAINING_COMPLETION_FILES = (
    "best_raw.pth",
    "resolved_config.json",
    "training_summary.json",
)
EVALUATION_COMPLETION_FILES = (
    "summary.json",
    "resolved_config.json",
    "metrics_by_sample.csv",
    "metrics_by_event.csv",
    "metrics_by_train_depth_bin.csv",
)
PROTOCOL_SCHEMA_VERSION = 3


class CrossValidationJobError(RuntimeError):
    """Raised when an isolated model/fold worker exits unsuccessfully."""


class CudaHealthError(RuntimeError):
    """Raised when a formal run cannot safely use the configured CUDA device."""


class CrossValidationProtocolError(RuntimeError):
    """Raised before results from different code/config protocols can be mixed."""


def _isolated_worker_environment(mkl_threading_layer: str) -> dict[str, str]:
    """Build a subprocess environment compatible with PyTorch's GNU OpenMP."""

    layer = str(mkl_threading_layer).strip().upper()
    if layer != "GNU":
        raise ValueError(
            "runtime.cross_validation.worker_mkl_threading_layer must be GNU "
            "because this PyTorch build loads libgomp"
        )
    environment = os.environ.copy()
    environment["MKL_THREADING_LAYER"] = layer
    return environment


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read cross-validation JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Cross-validation JSON is not a mapping: {path}")
    return value


def _validate_protocol_inventory(
    protocol: dict[str, Any],
    current_inventory: list[dict[str, str]],
    *,
    runtime_sha256: str,
) -> None:
    """Fail closed when a resumed session predates or differs from current code."""

    schema = int(protocol.get("schema_version", 0))
    if schema != PROTOCOL_SCHEMA_VERSION:
        raise CrossValidationProtocolError(
            "This cross-validation session uses a legacy split protocol and cannot "
            "be resumed safely. The current protocol outer-tests every original "
            "train/val/test sample exactly once. Start a new session with "
            "`python validate_k_fold.py 5`."
        )
    if str(protocol.get("protocol", "")) != (
        "outer_k_fold_with_event_grouped_inner_holdout"
    ) or list(protocol.get("source_splits", [])) != ["train", "val", "test"]:
        raise CrossValidationProtocolError(
            "The resumed session does not use the complete-sample nested outer-test "
            "protocol. Start a new session with `python validate_k_fold.py 5`."
        )
    recorded_inventory = protocol.get("models")
    if recorded_inventory != current_inventory:
        recorded_by_model = {
            str(item.get("model")): str(item.get("resolved_config_sha256"))
            for item in recorded_inventory or []
            if isinstance(item, dict)
        }
        current_by_model = {
            str(item["model"]): str(item["resolved_config_sha256"])
            for item in current_inventory
        }
        changed = sorted(
            model
            for model in set(recorded_by_model) | set(current_by_model)
            if recorded_by_model.get(model) != current_by_model.get(model)
        )
        changed_text = f" ({', '.join(changed)})" if changed else ""
        raise CrossValidationProtocolError(
            "Resolved model configurations differ from the resumed protocol"
            f"{changed_text}. Start a new session; "
            "mixing old and new folds is prohibited."
        )
    recorded_runtime = str(protocol.get("runtime_source_sha256", ""))
    if recorded_runtime != runtime_sha256:
        raise CrossValidationProtocolError(
            "Runtime Python sources differ from the resumed protocol. Start a new "
            "cross-validation session so every model/fold uses identical code."
        )


def _assert_protocol_inputs_unchanged(
    model_configs: tuple[Path, ...],
    expected_inventory: list[dict[str, str]],
    *,
    expected_runtime_sha256: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> None:
    """Guard a long campaign against edits made between isolated jobs."""

    current_inventory = config_inventory(model_configs)
    current_runtime_sha256, _ = runtime_source_sha256(PROJECT_ROOT)
    if current_inventory != expected_inventory:
        raise CrossValidationProtocolError(
            "A model configuration changed after this cross-validation session "
            "started. Completed jobs were preserved; start a new session instead "
            "of mixing protocols."
        )
    if current_runtime_sha256 != expected_runtime_sha256:
        raise CrossValidationProtocolError(
            "Runtime source code changed after this cross-validation session started. "
            "Completed jobs were preserved; start a new session instead of mixing code."
        )
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise CrossValidationProtocolError(
            "The source manifest changed after this cross-validation session started. "
            "Completed jobs were preserved; rebuild all folds in a new session."
        )


def _training_is_complete(path: Path) -> bool:
    return all((path / name).is_file() for name in TRAINING_COMPLETION_FILES)


def _evaluation_is_complete(path: Path) -> bool:
    return all((path / name).is_file() for name in EVALUATION_COMPLETION_FILES)


def _completed_job_results(
    model_root: Path,
    *,
    model_id: str,
    display_name: str,
    fold: int,
    source_config: Path,
) -> list[dict[str, Any]] | None:
    """Recover a complete job from durable summaries, including legacy sessions."""

    if not _training_is_complete(model_root / "train"):
        return None
    results: list[dict[str, Any]] = []
    for split in ("val", "test"):
        evaluation_root = model_root / split
        if not _evaluation_is_complete(evaluation_root):
            return None
        summary_path = evaluation_root / "summary.json"
        results.append(
            scalar_row(
                {
                    **_load_mapping(summary_path),
                    "model": model_id,
                    "display_name": display_name,
                    "fold": int(fold),
                    "split": split,
                    "cv_role": (
                        "inner_validation" if split == "val" else "outer_test"
                    ),
                    "source_config": str(source_config),
                }
            )
        )
    return results


def _archive_incomplete_attempt(model_root: Path) -> Path:
    """Move one incomplete, non-resumable attempt aside without deleting evidence."""

    archive = (
        model_root
        / "failed_attempts"
        / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    )
    archive.mkdir(parents=True, exist_ok=False)
    moved = False
    for name in (
        "train",
        "val",
        "test",
        "artifacts",
        "job_result.json",
        "job_status.json",
    ):
        source = model_root / name
        if source.exists():
            source.rename(archive / name)
            moved = True
    if not moved:
        archive.rmdir()
        raise RuntimeError(f"No incomplete attempt exists below {model_root}")
    return archive


def _resume_checkpoint_or_archive(
    model_root: Path, *, model_kind: str
) -> Path | None:
    """Return a valid last checkpoint or archive an unusable partial attempt."""

    training = model_root / "train"
    if not training.exists() or _training_is_complete(training):
        return None
    checkpoint_name = "last.pth" if model_kind == "pa_hydrokan" else "last_raw.pth"
    checkpoint = training / checkpoint_name
    if checkpoint.is_file():
        return checkpoint
    archive = _archive_incomplete_attempt(model_root)
    LOGGER.warning(
        "Archived incomplete attempt without a resumable checkpoint: %s", archive
    )
    return None


def _require_healthy_cuda(required: bool) -> None:
    """Fail before a formal run can silently fall back to CPU after a driver reset."""

    if not required:
        return
    try:
        probe = subprocess.run(
            ("nvidia-smi", "-L"),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CudaHealthError(
            "Formal cross-validation requires a healthy CUDA device, but the NVIDIA "
            f"driver probe failed: {exc}. Reboot or repair the driver first."
        ) from exc
    if probe.returncode != 0 or not probe.stdout.strip():
        detail = (probe.stderr or probe.stdout).strip()
        raise CudaHealthError(
            "Formal cross-validation requires a healthy CUDA device, but nvidia-smi "
            f"failed ({detail or f'exit code {probe.returncode}'}). Reboot or reset "
            "the NVIDIA driver before resuming; CPU fallback is intentionally blocked."
        )


def latest_resumable_session(runs_root: Path, k: int) -> Path:
    """Return the newest non-complete session for ``--resume`` without a path."""

    root = runs_root / "cross_validation" / f"k{k}"
    candidates: list[Path] = []
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or not (child / "protocol.json").is_file():
                continue
            status_path = child / "status.json"
            status = (
                _load_mapping(status_path).get("status")
                if status_path.is_file()
                else None
            )
            if status != "complete":
                candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No resumable k={k} session exists below {root}")

    def activity_key(path: Path) -> tuple[int, str]:
        status_path = path / "status.json"
        activity = (
            status_path.stat().st_mtime_ns
            if status_path.is_file()
            else path.stat().st_mtime_ns
        )
        return activity, path.name

    return max(candidates, key=activity_key)


def _run_isolated_job(
    *,
    overlay: Path,
    training_output: Path,
    model_root: Path,
    fold: int,
    source_config: Path,
    mkl_threading_layer: str,
) -> None:
    command = (
        sys.executable,
        str(JOB_WORKER),
        "--config",
        str(overlay),
        "--run-directory",
        str(training_output),
        "--model-root",
        str(model_root),
        "--fold",
        str(fold),
        "--source-config",
        str(source_config),
    )
    completed = subprocess.run(
        command,
        check=False,
        env=_isolated_worker_environment(mkl_threading_layer),
    )
    if completed.returncode != 0:
        signal_text = (
            f"signal {-completed.returncode}"
            if completed.returncode < 0
            else f"exit code {completed.returncode}"
        )
        raise CrossValidationJobError(
            f"Isolated worker failed with {signal_text}: model_root={model_root}. "
            "Completed jobs are preserved; after resolving the worker/GPU problem, "
            "continue with `python validate_k_fold.py --resume`."
        )


def _preflight_isolated_worker(
    *, mkl_threading_layer: str, require_cuda: bool
) -> None:
    """Verify the exact worker import/runtime environment before any model job."""

    command = [sys.executable, str(JOB_WORKER), "--preflight"]
    if require_cuda:
        command.append("--require-cuda")
    completed = subprocess.run(
        command,
        check=False,
        env=_isolated_worker_environment(mkl_threading_layer),
    )
    if completed.returncode != 0:
        message = (
            "Isolated worker preflight failed. No model job was started; inspect the "
            "NumPy/PyTorch/MKL/CUDA diagnostic above."
        )
        LOGGER.error(message)
        raise CrossValidationJobError(message)


def _metric(summary: list[dict[str, Any]], model: str, name: str) -> tuple[float, float]:
    row = next(
        item
        for item in summary
        if item.get("model") == model
        and item.get("split") == "test"
        and item.get("metric") == name
    )
    return float(row["mean"]), float(row["std"])


def _display_fold_results(rows: list[dict[str, Any]]) -> None:
    LOGGER.info("━" * 130)
    LOGGER.info(
        "%-32s | %4s | %8s | %8s | %8s | %9s | %10s | %12s",
        "Model",
        "Fold",
        "MAE",
        "RMSE",
        "R²",
        "Event MAE",
        "sample/s",
        "parameters",
    )
    LOGGER.info("─" * 130)
    for row in rows:
        if row.get("split") != "test":
            continue
        LOGGER.info(
            "%-32s | %4d | %8.4f | %8.4f | %8.4f | %9.4f | %10.2f | %12s",
            str(row.get("display_name", row["model"]))[:32],
            int(row["fold"]),
            float(row.get("pixel_micro_mae", float("nan"))),
            float(row.get("pixel_micro_rmse", float("nan"))),
            float(row.get("pixel_micro_r2", float("nan"))),
            float(row.get("event_macro_mae", float("nan"))),
            float(row.get("efficiency_samples_per_second", float("nan"))),
            f"{int(row.get('total_parameters', 0)):,}",
        )
    LOGGER.info("━" * 130)


def _display_aggregate(
    summary: list[dict[str, Any]], model_configs: tuple[Path, ...]
) -> None:
    LOGGER.info("Outer-test cross-fold mean ± sample standard deviation")
    LOGGER.info("━" * 126)
    LOGGER.info(
        "%-34s | %-20s | %-20s | %-20s | %-20s",
        "Model",
        "MAE (m)",
        "RMSE (m)",
        "R²",
        "Event MAE (m)",
    )
    LOGGER.info("─" * 126)
    for config_path in model_configs:
        config = load_config(config_path)
        model = experiment_id(config)
        mae = _metric(summary, model, "pixel_micro_mae")
        rmse = _metric(summary, model, "pixel_micro_rmse")
        r2 = _metric(summary, model, "pixel_micro_r2")
        event_mae = _metric(summary, model, "event_macro_mae")
        LOGGER.info(
            "%-34s | %8.4f ± %-8.4f | %8.4f ± %-8.4f | %8.4f ± %-8.4f | %8.4f ± %-8.4f",
            experiment_display_name(config)[:34],
            *mae,
            *rmse,
            *r2,
            *event_mae,
        )
    LOGGER.info("━" * 126)


def run_cross_validation(
    k: int | None,
    *,
    output: Path | None = None,
    prepare_only: bool = False,
    resume: Path | None = None,
) -> Path:
    if resume is not None and output is not None:
        raise ValueError("--resume and --output are mutually exclusive")
    if resume is not None and prepare_only:
        raise ValueError("--resume cannot be combined with --prepare-only")
    main_config = load_config(
        Path(__file__).resolve().parent / "configs" / "pa_hydrokan.xml"
    )
    cv_config = runtime_section(main_config, "cross_validation")
    default_k = int(cv_config.get("default_k", 5))
    selected_k = default_k if k is None else int(k)
    protocol: dict[str, Any] | None = None
    is_resuming = resume is not None
    if is_resuming:
        session = resume.expanduser().resolve(strict=True)
        if not session.is_dir():
            raise NotADirectoryError(
                f"Cross-validation session is not a directory: {session}"
            )
        protocol = _load_mapping(session / "protocol.json")
        protocol_k = int(protocol["k"])
        if k is not None and int(k) != protocol_k:
            raise ValueError(
                f"Requested k={k} does not match resumed session k={protocol_k}"
            )
        selected_k = protocol_k
    if selected_k < 2:
        raise ValueError("k must be at least 2")
    if str(cv_config.get("split_unit", "source_event_id")) != "source_event_id":
        raise ValueError("runtime.cross_validation.split_unit must be source_event_id")
    configured_source_splits = list(
        cv_config.get("source_splits", ["train", "val", "test"])
    )
    if configured_source_splits != ["train", "val", "test"]:
        raise ValueError(
            "runtime.cross_validation.source_splits must be [train, val, test] "
            "so every current sample participates in outer cross-validation"
        )
    source_splits = tuple(
        protocol.get("source_splits", configured_source_splits)
        if protocol is not None
        else configured_source_splits
    )
    if list(source_splits) != configured_source_splits:
        raise CrossValidationProtocolError(
            "The resumed session uses a different source-split pool. Start a new "
            "session instead of mixing fold definitions."
        )
    inner_validation_fraction = float(
        protocol.get(
            "inner_validation_fraction",
            cv_config.get("inner_validation_fraction", 0.125),
        )
        if protocol is not None
        else cv_config.get("inner_validation_fraction", 0.125)
    )
    if not 0.0 < inner_validation_fraction < 1.0:
        raise ValueError(
            "runtime.cross_validation.inner_validation_fraction must lie in (0, 1)"
        )
    k = selected_k
    seed_stride = int(
        protocol.get("fold_seed_stride", 1)
        if protocol is not None
        else cv_config.get("fold_seed_stride", 1)
    )
    if seed_stride <= 0:
        raise ValueError("runtime.cross_validation.fold_seed_stride must be positive")
    worker_mkl_threading_layer = str(
        protocol.get(
            "worker_mkl_threading_layer",
            cv_config.get("worker_mkl_threading_layer", "GNU"),
        )
        if protocol is not None
        else cv_config.get("worker_mkl_threading_layer", "GNU")
    ).strip().upper()
    _isolated_worker_environment(worker_mkl_threading_layer)
    require_cuda = bool(cv_config.get("require_cuda", True))
    model_configs = deep_learning_config_paths(include_ablations=True)
    inventory = config_inventory(model_configs)
    runtime_sha256, runtime_source_files = runtime_source_sha256(PROJECT_ROOT)
    if protocol is not None:
        _validate_protocol_inventory(
            protocol,
            inventory,
            runtime_sha256=runtime_sha256,
        )
    else:
        session = (
            output.expanduser().resolve()
            if output is not None
            else Path(main_config["runs_root"])
            / "cross_validation"
            / f"k{k}"
            / started_at_run_id(main_config)
        )
        if session.exists():
            raise FileExistsError(f"Cross-validation output already exists: {session}")
        session.mkdir(parents=True)
    setup_logging(
        session / "cross_validation.log",
        show_python_warnings=bool(
            main_config["logging"].get("show_python_warnings", True)
        ),
    )
    _require_healthy_cuda(require_cuda and not prepare_only)
    if not prepare_only:
        _preflight_isolated_worker(
            mkl_threading_layer=worker_mkl_threading_layer,
            require_cuda=require_cuda,
        )

    manifest_path = Path(
        protocol["source_manifest"]
        if protocol is not None
        else main_config["dataset"]["manifest"]
    )
    manifest_sha256 = sha256_file(manifest_path)
    if protocol is not None and str(protocol.get("source_manifest_sha256", "")) != manifest_sha256:
        raise CrossValidationProtocolError(
            "The source manifest differs from the resumed protocol. Start a new "
            "session so fold membership and train-only statistics remain reproducible."
        )
    rows, fieldnames = read_manifest(manifest_path)
    seed = int(protocol["seed"]) if protocol is not None else int(main_config["seed"])
    assignment, _ = assign_event_folds(
        rows,
        k,
        seed=seed,
        source_splits=source_splits,
    )
    inner_validation_events, balance = plan_cross_validation_folds(
        rows,
        assignment,
        k,
        inner_validation_fraction=inner_validation_fraction,
        seed=seed,
        seed_stride=seed_stride,
        source_splits=source_splits,
    )
    original_split_samples = {
        split: sum(str(row["split"]) == split for row in rows)
        for split in source_splits
    }
    original_split_events = {
        split: len(
            {
                str(row["source_event_id"])
                for row in rows
                if str(row["split"]) == split
            }
        )
        for split in source_splits
    }
    source_samples = sum(original_split_samples.values())
    source_events = len(assignment)
    if protocol is not None and protocol.get("fold_balance") != balance:
        raise CrossValidationProtocolError(
            "The recomputed outer/inner fold plan differs from the resumed protocol. "
            "Start a new session; existing fold results cannot be mixed."
        )
    if protocol is None:
        assignment_table = assignment_rows(
            rows,
            assignment,
            source_splits=source_splits,
        )
        write_union_rows(session / "event_fold_assignment.csv", assignment_table)
        event_sample_counts = Counter(
            str(row["source_event_id"])
            for row in rows
            if str(row["split"]) in source_splits
        )
        write_union_rows(
            session / "inner_validation_event_assignment.csv",
            (
                {
                    "fold": fold_index + 1,
                    "source_event_id": event,
                    "samples": event_sample_counts[event],
                    "role": "inner_validation",
                }
                for fold_index, events in sorted(inner_validation_events.items())
                for event in sorted(events)
            ),
        )
        write_union_rows(session / "fold_balance.csv", balance)
        atomic_write_json(
            session / "protocol.json",
            {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "k": k,
                "seed": seed,
                "fold_seed_stride": seed_stride,
                "split_unit": "source_event_id",
                "protocol": "outer_k_fold_with_event_grouped_inner_holdout",
                "source_splits": list(source_splits),
                "source_samples": source_samples,
                "source_events": source_events,
                "original_split_samples": original_split_samples,
                "original_split_events": original_split_events,
                "inner_validation_fraction": inner_validation_fraction,
                "outer_test_policy": "each_sample_exactly_once",
                "checkpoint_selection": "inner_validation_only",
                "source_manifest": str(manifest_path),
                "source_manifest_sha256": manifest_sha256,
                "runtime_source_sha256": runtime_sha256,
                "runtime_source_files": runtime_source_files,
                "normalization_reservoir_capacity": int(
                    cv_config.get("normalization_reservoir_capacity", 1_000_000)
                ),
                "job_isolation": "one_subprocess_per_model_fold",
                "worker_mkl_threading_layer": worker_mkl_threading_layer,
                "models": inventory,
                "fold_balance": balance,
            },
        )
    LOGGER.info("━" * 78)
    LOGGER.info(
        "%s nested event-grouped %d-fold testing | models=%d | all samples=%d",
        "Resuming" if is_resuming else "Starting",
        k,
        len(model_configs),
        source_samples,
    )
    LOGGER.info(
        "Outer-test fold loads: %s",
        [int(item["outer_test_samples"]) for item in balance],
    )
    LOGGER.info(
        "Inner-validation fold loads: %s",
        [int(item["inner_validation_samples"]) for item in balance],
    )
    LOGGER.info(
        "Original split labels regrouped: train=%d | val=%d | test=%d",
        original_split_samples["train"],
        original_split_samples["val"],
        original_split_samples["test"],
    )
    LOGGER.info("Every sample is assigned to exactly one outer-test fold.")
    LOGGER.info("Output directory: %s", session)
    LOGGER.info("━" * 78)

    reservoir_capacity = int(
        protocol.get("normalization_reservoir_capacity", 1_000_000)
        if protocol is not None
        else cv_config.get("normalization_reservoir_capacity", 1_000_000)
    )
    if reservoir_capacity <= 0:
        raise ValueError(
            "runtime.cross_validation.normalization_reservoir_capacity must be positive"
        )
    fold_assets: list[dict[str, Any]] = []
    for fold_index in range(k):
        fold_number = fold_index + 1
        fold_root = session / "folds" / f"fold_{fold_number:02d}"
        fold_manifest = fold_root / "manifest.csv"
        if is_resuming and fold_manifest.is_file():
            fold_rows, _ = read_manifest(fold_manifest)
            counts = validate_fold_manifest_rows(
                fold_rows,
                rows,
                assignment,
                fold_index,
                inner_validation_events[fold_index],
                source_splits,
            )
            contract = fold_root / "assets" / "dataset_contract.json"
            stats = fold_root / "assets" / "train_stats.json"
            expected_counts = {
                "train": int(balance[fold_index]["training_samples"]),
                "val": int(balance[fold_index]["inner_validation_samples"]),
                "test": int(balance[fold_index]["outer_test_samples"]),
            }
            if counts != expected_counts:
                raise RuntimeError(
                    f"Fold {fold_number} role counts changed: "
                    f"{counts} != {expected_counts}"
                )
            if contract.is_file() and stats.is_file():
                dataset_contract = DatasetContract.load(contract)
                dataset_contract.verify_fingerprints(include_normalization=True)
                selected_stats = dataset_contract.payload["normalization"]["selected"]
                if str(selected_stats["sha256"]) != sha256_file(stats):
                    raise RuntimeError(
                        f"Fold {fold_number} train statistics changed: {stats}"
                    )
                if dataset_contract.manifest_path != fold_manifest.resolve(strict=True):
                    raise RuntimeError(
                        f"Fold {fold_number} contract points to a different manifest"
                    )
                contract_counts = {
                    split: int(dataset_contract.payload["sample_counts"][split])
                    for split in ("train", "val", "test")
                }
                if contract_counts != counts:
                    raise RuntimeError(
                        f"Fold {fold_number} contract counts changed: "
                        f"{contract_counts} != {counts}"
                    )
                LOGGER.info(
                    "Reusing fold %02d/%02d assets | train=%d | inner-val=%d | outer-test=%d",
                    fold_number,
                    k,
                    counts["train"],
                    counts["val"],
                    counts["test"],
                )
            else:
                LOGGER.info(
                    "Completing fold %02d/%02d assets | train=%d | inner-val=%d | outer-test=%d",
                    fold_number,
                    k,
                    counts["train"],
                    counts["val"],
                    counts["test"],
                )
                contract, stats = build_assets(
                    Path(main_config["dataset"]["root"]),
                    fold_root / "assets",
                    reservoir_capacity,
                    seed + fold_number * seed_stride,
                    fold_manifest,
                )
        else:
            counts = write_fold_manifest(
                fold_manifest,
                rows,
                fieldnames,
                assignment,
                fold_index,
                inner_validation_events[fold_index],
                source_splits,
            )
            LOGGER.info(
                "Preparing fold %02d/%02d assets | train=%d | inner-val=%d | outer-test=%d",
                fold_number,
                k,
                counts["train"],
                counts["val"],
                counts["test"],
            )
            contract, stats = build_assets(
                Path(main_config["dataset"]["root"]),
                fold_root / "assets",
                reservoir_capacity,
                seed + fold_number * seed_stride,
                fold_manifest,
            )
        fold_assets.append(
            {
                "fold": fold_number,
                "root": fold_root,
                "manifest": fold_manifest,
                "contract": contract,
                "stats": stats,
                "counts": counts,
            }
        )

    if prepare_only:
        atomic_write_json(
            session / "status.json",
            {"status": "assets_ready", "folds": k, "models": len(model_configs)},
        )
        LOGGER.info("Fold assets prepared; --prepare-only skipped model training.")
        return session

    writer = create_summary_writer(
        session / "tensorboard",
        enabled=bool(main_config["logging"].get("tensorboard", False)),
        flush_seconds=int(main_config["logging"].get("tensorboard_flush_seconds", 30)),
        logger=LOGGER,
    )
    add_metadata(
        writer,
        {
            "operation": "event_grouped_cross_validation",
            "k": k,
            "models": len(model_configs),
            "source_splits": list(source_splits),
            "source_samples": source_samples,
            "outer_test_policy": "each_sample_exactly_once",
            "inner_validation_fraction": inner_validation_fraction,
        },
    )
    results: list[dict[str, Any]] = []
    total_trainings = k * len(model_configs)
    completed = 0
    finished = 0
    try:
        for fold_asset in fold_assets:
            fold_number = int(fold_asset["fold"])
            fold_root = Path(fold_asset["root"])
            for source_config in model_configs:
                _assert_protocol_inputs_unchanged(
                    model_configs,
                    inventory,
                    expected_runtime_sha256=runtime_sha256,
                    manifest_path=manifest_path,
                    expected_manifest_sha256=manifest_sha256,
                )
                source = load_config(source_config)
                model_id = experiment_id(source)
                display_name = experiment_display_name(source)
                model_root = fold_root / "models" / model_id
                training_output = model_root / "train"
                completed += 1
                existing_results = _completed_job_results(
                    model_root,
                    model_id=model_id,
                    display_name=display_name,
                    fold=fold_number,
                    source_config=source_config,
                )
                if existing_results is not None:
                    LOGGER.info(
                        "[%d/%d] Reusing completed %s | fold=%02d/%02d",
                        completed,
                        total_trainings,
                        display_name,
                        fold_number,
                        k,
                    )
                    results.extend(existing_results)
                    for result in existing_results:
                        add_scalars(
                            writer,
                            result,
                            step=fold_number,
                            prefix=f"cross_validation/{model_id}/{result['split']}",
                        )
                    finished += 1
                    write_union_rows(session / "metrics_by_fold.csv", results)
                    atomic_write_json(
                        session / "status.json",
                        {
                            "status": "running",
                            "folds": k,
                            "models": len(model_configs),
                            "trainings_completed": finished,
                            "total_trainings": total_trainings,
                            "resumed": is_resuming,
                        },
                    )
                    continue

                model_kind, _ = configured_model_kind(source)
                _require_healthy_cuda(require_cuda)
                resume_checkpoint = (
                    _resume_checkpoint_or_archive(
                        model_root,
                        model_kind=model_kind,
                    )
                    if is_resuming
                    else None
                )
                overlay = write_fold_config(
                    source_config,
                    fold_root / "configs" / f"{model_id}.xml",
                    contract=Path(fold_asset["contract"]),
                    manifest=Path(fold_asset["manifest"]),
                    train_stats=Path(fold_asset["stats"]),
                    training_output=training_output,
                    session_root=session,
                    k=k,
                    fold=fold_number,
                    seed=seed + fold_number * seed_stride,
                    source_splits=source_splits,
                    inner_validation_fraction=inner_validation_fraction,
                    resume_checkpoint=resume_checkpoint,
                )
                LOGGER.info(
                    "[%d/%d] Running isolated job %s | fold=%02d/%02d%s",
                    completed,
                    total_trainings,
                    display_name,
                    fold_number,
                    k,
                    f" | resume={resume_checkpoint.name}"
                    if resume_checkpoint is not None
                    else "",
                )
                _run_isolated_job(
                    overlay=overlay,
                    training_output=training_output,
                    model_root=model_root,
                    fold=fold_number,
                    source_config=source_config,
                    mkl_threading_layer=worker_mkl_threading_layer,
                )
                job_results = _completed_job_results(
                    model_root,
                    model_id=model_id,
                    display_name=display_name,
                    fold=fold_number,
                    source_config=source_config,
                )
                if job_results is None:
                    raise RuntimeError(
                        f"Worker exited successfully without complete results: {model_root}"
                    )
                results.extend(job_results)
                for result in job_results:
                    add_scalars(
                        writer,
                        result,
                        step=fold_number,
                        prefix=f"cross_validation/{model_id}/{result['split']}",
                    )
                flush(writer)
                write_union_rows(session / "metrics_by_fold.csv", results)
                finished += 1
                atomic_write_json(
                    session / "status.json",
                    {
                        "status": "running",
                        "folds": k,
                        "models": len(model_configs),
                        "trainings_completed": finished,
                        "total_trainings": total_trainings,
                        "resumed": is_resuming,
                    },
                )
    except BaseException as exc:
        setup_logging(session / "cross_validation.log")
        write_union_rows(session / "metrics_by_fold.csv", results)
        atomic_write_json(
            session / "status.json",
            {
                "status": "failed",
                "folds": k,
                "models": len(model_configs),
                "trainings_completed": finished,
                "active_training_index": completed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        if isinstance(exc, CrossValidationJobError):
            LOGGER.error("Cross-validation stopped | %s", exc)
        else:
            LOGGER.exception("Cross-validation stopped after an error")
        raise
    finally:
        if writer is not None:
            writer.close()

    summaries = numerical_summary_rows(
        results,
        group_keys=("model", "display_name", "split"),
    )
    write_union_rows(session / "metrics_by_fold.csv", results)
    write_union_rows(session / "metrics_summary.csv", summaries)
    atomic_write_json(
        session / "summary.json",
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "k": k,
            "protocol": "outer_k_fold_with_event_grouped_inner_holdout",
            "source_splits": list(source_splits),
            "source_samples": source_samples,
            "source_events": source_events,
            "fold_balance": balance,
            "fold_results": results,
            "metric_summaries": summaries,
        },
    )
    atomic_write_json(
        session / "status.json",
        {
            "status": "complete",
            "folds": k,
            "models": len(model_configs),
            "trainings": total_trainings,
            "resumed": is_resuming,
        },
    )
    setup_logging(session / "cross_validation.log")
    _display_fold_results(results)
    _display_aggregate(summaries, model_configs)
    LOGGER.info("Complete cross-validation report: %s", session)
    return session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "k",
        type=int,
        nargs="?",
        default=None,
        help="Number of folds (default: runtime.cross_validation.default_k, currently 5).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create and audit fold manifests/assets without starting training.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="SESSION",
        help=(
            "Resume a session directory. Omit SESSION to select the newest "
            "unfinished session for the requested/default k."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume is not None and args.output is not None:
        raise SystemExit("--resume and --output are mutually exclusive")
    if args.resume is not None and args.prepare_only:
        raise SystemExit("--resume cannot be combined with --prepare-only")
    resume_path: Path | None = None
    if args.resume is not None:
        if args.resume == "latest":
            config = load_config(PROJECT_ROOT / "configs" / "pa_hydrokan.xml")
            cv_config = runtime_section(config, "cross_validation")
            selected_k = (
                int(cv_config.get("default_k", 5)) if args.k is None else int(args.k)
            )
            resume_path = latest_resumable_session(
                Path(config["runs_root"]), selected_k
            )
        else:
            resume_path = Path(args.resume)
    try:
        run_cross_validation(
            args.k,
            output=args.output,
            prepare_only=args.prepare_only,
            resume=resume_path,
        )
    except CudaHealthError as exc:
        LOGGER.error("%s", exc)
        return 2
    except CrossValidationJobError:
        return 1
    except CrossValidationProtocolError as exc:
        LOGGER.error("%s", exc)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

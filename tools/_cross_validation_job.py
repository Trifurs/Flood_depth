#!/usr/bin/env python3
"""Run one isolated model/fold training-and-evaluation job.

This is an internal worker for ``validate_k_fold.py``.  One process owns one
CUDA context and exits after the model has been trained and evaluated, which
prevents allocator/context state from accumulating across a long factorial
cross-validation campaign.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# MKL must initialize before PyTorch loads libgomp.  The controller additionally
# pins MKL_THREADING_LAYER=GNU for this process.
import numpy as np
import torch

import train as unified_train
from utils.config import load_config
from utils.evaluation_suite import evaluate_deep_run
from utils.experiment_catalog import experiment_display_name, experiment_id
from utils.cross_validation import resolved_config_sha256
from utils.misc import atomic_write_json
from utils.reporting import scalar_row
from utils.run_paths import runtime_section


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


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON result {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON result is not a mapping: {path}")
    return value


def training_is_complete(run_directory: Path) -> bool:
    return all(
        (run_directory / name).is_file() for name in TRAINING_COMPLETION_FILES
    )


def evaluation_is_complete(output: Path) -> bool:
    return all((output / name).is_file() for name in EVALUATION_COMPLETION_FILES)


def require_configured_cuda(config: Mapping[str, Any]) -> None:
    """Prevent a formal fold worker from silently selecting CPU."""

    required = bool(
        runtime_section(config, "cross_validation").get("require_cuda", True)
    )
    if not required:
        return
    if str(config.get("device", "auto")).lower() == "cpu":
        raise RuntimeError(
            "Formal cross-validation requires CUDA, but device=cpu is configured."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Formal cross-validation requires CUDA, but PyTorch cannot access a CUDA "
            "device. Reboot or repair the NVIDIA driver before resuming."
        )
    try:
        probe = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del probe
    except Exception as exc:
        raise RuntimeError(
            "PyTorch could not initialize a healthy CUDA context. Reboot or repair "
            "the NVIDIA driver before resuming."
        ) from exc


def _release_cuda_context_objects() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def preflight_worker(*, require_cuda: bool) -> None:
    """Exercise the worker's NumPy, PyTorch, OpenMP, and optional CUDA stack."""

    if not np.isfinite(np.asarray([0.0], dtype=np.float32)).all():
        raise RuntimeError("NumPy worker preflight produced a non-finite value")
    device_name = "cpu"
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Worker preflight requires CUDA, but PyTorch cannot access it."
            )
        probe = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        device_name = torch.cuda.get_device_name(probe.device)
        del probe
    print(
        "Worker preflight passed | "
        f"MKL_THREADING_LAYER={os.environ.get('MKL_THREADING_LAYER', '<unset>')} | "
        f"NumPy={np.__version__} | PyTorch={torch.__version__} | device={device_name}",
        flush=True,
    )


def run_job(
    config_path: Path,
    run_directory: Path,
    model_root: Path,
    fold: int,
    source_config: Path,
) -> list[dict[str, Any]]:
    """Finish one job, reusing any complete training/evaluation products."""

    config_path = config_path.expanduser().resolve(strict=True)
    run_directory = run_directory.expanduser().resolve()
    model_root = model_root.expanduser().resolve()
    source_config = source_config.expanduser().resolve(strict=True)
    config = load_config(config_path)
    protocol_config = config.get("cross_validation", {})
    if str(protocol_config.get("protocol", "")) != (
        "outer_k_fold_with_event_grouped_inner_holdout"
    ):
        raise RuntimeError("Worker received an unsupported cross-validation protocol")
    if list(protocol_config.get("source_splits", [])) != ["train", "val", "test"]:
        raise RuntimeError(
            "Worker requires the complete train/val/test sample pool"
        )
    if int(protocol_config.get("fold", -1)) != int(fold):
        raise RuntimeError("Worker fold argument differs from the generated overlay")
    expected_source_sha256 = str(protocol_config.get("source_config_sha256", ""))
    actual_source_sha256 = resolved_config_sha256(source_config)
    if not expected_source_sha256 or actual_source_sha256 != expected_source_sha256:
        raise RuntimeError(
            "The source model configuration changed between controller validation "
            "and worker startup; refusing to mix cross-validation protocols."
        )
    require_configured_cuda(config)
    model_id = experiment_id(config)
    display_name = experiment_display_name(config)
    model_root.mkdir(parents=True, exist_ok=True)
    status_path = model_root / "job_status.json"
    atomic_write_json(
        status_path,
        {
            "status": "running",
            "model": model_id,
            "display_name": display_name,
            "fold": int(fold),
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        if training_is_complete(run_directory):
            completed_run = run_directory
            print(
                f"Reusing completed training | model={display_name} | fold={fold:02d}",
                flush=True,
            )
        else:
            completed_run = unified_train.run_from_config(config_path)
            if completed_run.expanduser().resolve() != run_directory:
                raise RuntimeError(
                    "Cross-validation worker received an unexpected training output: "
                    f"{completed_run} != {run_directory}"
                )
        if not training_is_complete(completed_run):
            raise RuntimeError(
                f"Training returned without complete checkpoint metadata: {completed_run}"
            )
        _release_cuda_context_objects()

        results: list[dict[str, Any]] = []
        for split in ("val", "test"):
            evaluation_root = model_root / split
            summary_path = evaluation_root / "summary.json"
            if evaluation_is_complete(evaluation_root):
                summary: Mapping[str, Any] = _load_mapping(summary_path)
                print(
                    f"Reusing completed evaluation | model={display_name} | "
                    f"fold={fold:02d} | split={split}",
                    flush=True,
                )
            else:
                print(
                    f"Evaluating {display_name} | fold={fold:02d} | split={split}",
                    flush=True,
                )
                summary = evaluate_deep_run(
                    completed_run,
                    evaluation_root,
                    split=split,
                )
            results.append(
                scalar_row(
                    {
                        **summary,
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
            _release_cuda_context_objects()

        payload = {
            "status": "complete",
            "model": model_id,
            "display_name": display_name,
            "fold": int(fold),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }
        atomic_write_json(model_root / "job_result.json", payload)
        atomic_write_json(
            status_path,
            {key: value for key, value in payload.items() if key != "results"},
        )
        return results
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "status": "failed",
                "model": model_id,
                "display_name": display_name,
                "fold": int(fold),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--source-config", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight:
        preflight_worker(require_cuda=bool(args.require_cuda))
        return 0
    if args.require_cuda:
        raise SystemExit("--require-cuda is only valid with --preflight")
    required = {
        "--config": args.config,
        "--run-directory": args.run_directory,
        "--model-root": args.model_root,
        "--fold": args.fold,
        "--source-config": args.source_config,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"Missing required worker arguments: {', '.join(missing)}")
    run_job(
        args.config,
        args.run_directory,
        args.model_root,
        args.fold,
        args.source_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

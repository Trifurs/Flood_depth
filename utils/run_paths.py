"""Configuration-driven paths and runtime options for model-agnostic entry points."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class RuntimeConfigError(ValueError):
    """Raised when an XML runtime section is incomplete or unsafe."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeConfigError(f"{name} must be a configuration mapping")
    return value


def runtime_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one optional ``runtime`` subsection with a stable empty default."""

    runtime = _mapping(config.get("runtime"), "runtime")
    return _mapping(runtime.get(name), f"runtime.{name}")


def optional_path(value: Any, name: str) -> Path | None:
    """Convert an XML optional path while rejecting ambiguous values."""

    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    if isinstance(value, str):
        return Path(value).expanduser().resolve()
    raise RuntimeConfigError(f"{name} must be a path or none, got {type(value).__name__}")


def optional_nonnegative_int(value: Any, name: str) -> int | None:
    """Validate a nullable batch-limit/count configuration value."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeConfigError(f"{name} must be an integer or none")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"{name} must be an integer or none") from exc
    if result < 0:
        raise RuntimeConfigError(f"{name} must be non-negative")
    return result


def model_identifier(config: Mapping[str, Any]) -> str:
    """Resolve the per-model directory name without relying on a CLI wrapper."""

    value = config.get("run_name")
    if value is None:
        model = _mapping(config.get("model"), "model")
        compare = _mapping(config.get("compare"), "compare")
        value = model.get("name", compare.get("method"))
    identifier = str(value or "").strip()
    if not identifier or Path(identifier).name != identifier:
        raise RuntimeConfigError(f"invalid model run identifier: {identifier!r}")
    return identifier


def run_tag(config: Mapping[str, Any]) -> str:
    """Return the configuration-owned run label used across every model."""

    runtime = _mapping(config.get("runtime"), "runtime")
    value = str(runtime.get("run_tag", f"seed_{config.get('seed', 'default')}")).strip()
    if not value or Path(value).name != value:
        raise RuntimeConfigError(f"runtime.run_tag must be one path component, got {value!r}")
    return value


def _runs_root(config: Mapping[str, Any]) -> Path:
    value = config.get("runs_root")
    if value is None:
        raise RuntimeConfigError("configuration has no runs_root")
    return Path(value)


def train_output_path(config: Mapping[str, Any]) -> Path:
    """Resolve the configured training directory or its common default."""

    train = runtime_section(config, "train")
    explicit = optional_path(train.get("output"), "runtime.train.output")
    if explicit is not None:
        return explicit
    ablation = _mapping(config.get("ablation"), "ablation")
    if ablation:
        variant = str(ablation.get("variant_id", "")).strip()
        if not variant or Path(variant).name != variant:
            raise RuntimeConfigError(f"invalid ablation.variant_id: {variant!r}")
        return _runs_root(config) / "ablation" / variant / run_tag(config)
    return _runs_root(config) / "train" / model_identifier(config) / run_tag(config)


def evaluation_split(config: Mapping[str, Any]) -> str:
    """Return the one configured validation/test split."""

    evaluation = runtime_section(config, "evaluation")
    split = str(evaluation.get("split", "val"))
    if split not in {"val", "test"}:
        raise RuntimeConfigError("runtime.evaluation.split must be 'val' or 'test'")
    return split


def evaluation_output_path(config: Mapping[str, Any]) -> Path:
    """Resolve the configured evaluation directory or its common default."""

    evaluation = runtime_section(config, "evaluation")
    explicit = optional_path(evaluation.get("output"), "runtime.evaluation.output")
    if explicit is not None:
        return explicit
    return (
        _runs_root(config)
        / "evaluate"
        / model_identifier(config)
        / evaluation_split(config)
        / run_tag(config)
    )


def evaluation_checkpoint_path(config: Mapping[str, Any]) -> Path:
    """Resolve an explicit checkpoint or the selected checkpoint for this run tag."""

    evaluation = runtime_section(config, "evaluation")
    explicit = optional_path(evaluation.get("checkpoint"), "runtime.evaluation.checkpoint")
    return explicit if explicit is not None else train_output_path(config) / "best_raw.pth"


def allow_existing_output(config: Mapping[str, Any], operation: str) -> bool:
    """Read the explicit opt-in that permits replacing an existing result folder."""

    section = runtime_section(config, operation)
    return bool(section.get("allow_existing_output", False))


def ensure_output_is_available(
    path: Path,
    *,
    allow_existing: bool,
    operation: str,
) -> None:
    """Fail closed before a formal run could overwrite prior evidence."""

    if path.exists() and not allow_existing:
        raise FileExistsError(
            f"{operation} output already exists: {path}. "
            "Change runtime.run_tag, choose an explicit output path, or set the "
            "operation's allow_existing_output=true deliberately."
        )

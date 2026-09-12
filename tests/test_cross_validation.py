from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from datasets.contract import ContractError, DatasetContract, sha256_file
from utils.config import load_config
from utils.cross_validation import (
    assign_event_folds,
    config_inventory,
    plan_cross_validation_folds,
    resolved_config_sha256,
    runtime_source_sha256,
    validate_fold_manifest_rows,
    write_fold_config,
    write_fold_manifest,
)
from utils.experiment_catalog import deep_learning_config_paths, experiment_id
from validate_k_fold import (
    EVALUATION_COMPLETION_FILES,
    TRAINING_COMPLETION_FILES,
    _completed_job_results,
    _isolated_worker_environment,
    _resume_checkpoint_or_archive,
    _validate_protocol_inventory,
    CrossValidationProtocolError,
    PROTOCOL_SCHEMA_VERSION,
    latest_resumable_session,
)


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for event, count, split in (
        ("event_a", 7, "train"),
        ("event_b", 5, "train"),
        ("event_c", 4, "val"),
        ("event_d", 3, "train"),
        ("event_e", 2, "val"),
        ("event_f", 1, "train"),
        ("original_test_event", 3, "test"),
    ):
        rows.extend(
            {
                "sample_id": f"{event}_{index}",
                "source_event_id": event,
                "split": split,
            }
            for index in range(count)
        )
    return rows


def test_event_grouped_folds_are_disjoint_balanced_and_deterministic() -> None:
    rows = _rows()
    first, balance = assign_event_folds(rows, 3, seed=42)
    second, _ = assign_event_folds(rows, 3, seed=42)
    assert first == second
    assert set(first) == {
        "event_a",
        "event_b",
        "event_c",
        "event_d",
        "event_e",
        "event_f",
        "original_test_event",
    }
    loads = [int(item["outer_test_samples"]) for item in balance]
    assert max(loads) - min(loads) <= 1
    assert sum(loads) == 25


def test_full_sample_nested_fold_manifests_have_no_event_leakage(
    tmp_path: Path,
) -> None:
    rows = _rows()
    assignment, _ = assign_event_folds(rows, 3, seed=7)
    inner_by_fold, balance = plan_cross_validation_folds(
        rows,
        assignment,
        3,
        inner_validation_fraction=0.25,
        seed=7,
        seed_stride=1,
    )
    test_appearances: dict[str, int] = {row["sample_id"]: 0 for row in rows}
    for fold_index in range(3):
        path = tmp_path / f"fold_{fold_index + 1}.csv"
        counts = write_fold_manifest(
            path,
            rows,
            ("sample_id", "source_event_id", "split"),
            assignment,
            outer_test_fold=fold_index,
            inner_validation_events=inner_by_fold[fold_index],
        )
        with path.open(encoding="utf-8", newline="") as handle:
            written = list(csv.DictReader(handle))
        assert len(written) == len(rows)
        by_event: dict[str, set[str]] = {}
        for row in written:
            by_event.setdefault(row["source_event_id"], set()).add(row["split"])
            if row["split"] == "test":
                test_appearances[row["sample_id"]] += 1
        assert all(len(splits) == 1 for splits in by_event.values())
        assert counts == {
            "train": int(balance[fold_index]["training_samples"]),
            "val": int(balance[fold_index]["inner_validation_samples"]),
            "test": int(balance[fold_index]["outer_test_samples"]),
        }
    assert set(test_appearances.values()) == {1}


def test_inner_validation_is_selected_only_from_non_test_events() -> None:
    rows = _rows()
    assignment, _ = assign_event_folds(rows, 3, seed=19)
    first, balance = plan_cross_validation_folds(
        rows,
        assignment,
        3,
        inner_validation_fraction=0.25,
        seed=19,
        seed_stride=2,
    )
    second, _ = plan_cross_validation_folds(
        rows,
        assignment,
        3,
        inner_validation_fraction=0.25,
        seed=19,
        seed_stride=2,
    )
    assert first == second
    for fold_index, inner_events in first.items():
        outer_events = {
            event for event, assigned_fold in assignment.items()
            if assigned_fold == fold_index
        }
        assert inner_events
        assert outer_events
        assert inner_events.isdisjoint(outer_events)
        row = balance[fold_index]
        remaining_events = len(assignment) - len(outer_events)
        assert len(inner_events) == round(remaining_events * 0.25)
        assert (
            int(row["training_samples"])
            + int(row["inner_validation_samples"])
            + int(row["outer_test_samples"])
            == 25
        )


def test_persisted_fold_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    assignment, _ = assign_event_folds(rows, 3, seed=29)
    inner_by_fold, _ = plan_cross_validation_folds(
        rows,
        assignment,
        3,
        inner_validation_fraction=0.25,
        seed=29,
        seed_stride=1,
    )
    path = tmp_path / "fold.csv"
    write_fold_manifest(
        path,
        rows,
        ("sample_id", "source_event_id", "split"),
        assignment,
        outer_test_fold=0,
        inner_validation_events=inner_by_fold[0],
    )
    with path.open(encoding="utf-8", newline="") as handle:
        fold_rows = list(csv.DictReader(handle))
    original = fold_rows[0]["split"]
    fold_rows[0]["split"] = "val" if original != "val" else "train"
    with pytest.raises(ValueError, match="differs from its deterministic"):
        validate_fold_manifest_rows(
            fold_rows,
            rows,
            assignment,
            0,
            inner_by_fold[0],
        )


def test_generated_fold_config_inherits_model_and_overrides_only_fold_assets(
    tmp_path: Path,
) -> None:
    source = Path("configs/pa_hydrokan.xml").resolve()
    destination = tmp_path / "fold.xml"
    write_fold_config(
        source,
        destination,
        contract=tmp_path / "contract.json",
        manifest=tmp_path / "manifest.csv",
        train_stats=tmp_path / "stats.json",
        training_output=tmp_path / "model" / "train",
        session_root=tmp_path / "session",
        k=5,
        fold=2,
        seed=99,
    )
    config = load_config(destination)
    assert config["model"]["name"] == "pa_hydrokan"
    assert config["cross_validation"] == {
        "k": 5,
        "fold": 2,
        "split_unit": "source_event_id",
        "protocol": "outer_k_fold_with_event_grouped_inner_holdout",
        "source_splits": ["train", "val", "test"],
        "inner_validation_fraction": 0.125,
        "source_config": source,
        "source_config_sha256": resolved_config_sha256(source),
    }
    assert config["dataset"]["contract"] == (tmp_path / "contract.json").resolve()
    assert config["runtime"]["train"]["output"] == (tmp_path / "model" / "train").resolve()
    assert config["runtime"]["train"]["resume"] is None


def test_generated_fold_config_can_resume_a_model_owned_last_checkpoint(
    tmp_path: Path,
) -> None:
    source = Path("configs/compare/deep_learning/dlsim_linknet.xml").resolve()
    checkpoint = tmp_path / "model" / "train" / "last_raw.pth"
    destination = tmp_path / "fold.xml"
    write_fold_config(
        source,
        destination,
        contract=tmp_path / "contract.json",
        manifest=tmp_path / "manifest.csv",
        train_stats=tmp_path / "stats.json",
        training_output=checkpoint.parent,
        session_root=tmp_path / "session",
        k=5,
        fold=1,
        seed=99,
        resume_checkpoint=checkpoint,
    )
    config = load_config(destination)
    assert config["runtime"]["train"]["resume"] == checkpoint.resolve()


def test_contract_can_bind_a_hashed_fold_manifest_outside_the_dataset_root(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest = tmp_path / "session" / "fold.csv"
    manifest.parent.mkdir()
    manifest.write_text("sample_id,source_event_id,split\na,e,train\n", encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    contract = DatasetContract(
        contract_path,
        {
            "dataset_root": str(dataset_root),
            "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "key_file_sha256": {},
        },
    )
    assert contract.manifest_path == manifest.resolve()
    contract.verify_fingerprints(include_normalization=False)
    manifest.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ContractError, match="Manifest fingerprint changed"):
        contract.verify_fingerprints(include_normalization=False)


def test_cross_validation_catalog_has_one_full_model_and_seven_factorial_ablations() -> None:
    configs = [load_config(path) for path in deep_learning_config_paths()]
    identifiers = [experiment_id(config) for config in configs]
    assert identifiers.count("pa_hydrokan") == 1
    assert identifiers[-7:] == [
        "wo_rcp",
        "wo_tcf",
        "wo_tae_kan",
        "wo_rcp_tcf",
        "wo_rcp_tae_kan",
        "wo_tcf_tae_kan",
        "wo_rcp_tcf_tae_kan",
    ]


def test_resume_recovers_only_jobs_with_complete_training_and_evaluations(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    for name in TRAINING_COMPLETION_FILES:
        path = model_root / "train" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    for split in ("val", "test"):
        for name in EVALUATION_COMPLETION_FILES:
            path = model_root / split / name
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"pixel_micro_mae": 0.25} if name == "summary.json" else {}
            path.write_text(json.dumps(payload), encoding="utf-8")

    results = _completed_job_results(
        model_root,
        model_id="example",
        display_name="Example",
        fold=2,
        source_config=Path("example.xml"),
    )
    assert results is not None
    assert [row["split"] for row in results] == ["val", "test"]
    (model_root / "test" / "metrics_by_event.csv").unlink()
    assert (
        _completed_job_results(
            model_root,
            model_id="example",
            display_name="Example",
            fold=2,
            source_config=Path("example.xml"),
        )
        is None
    )


@pytest.mark.parametrize(
    ("model_kind", "checkpoint_name"),
    [("pa_hydrokan", "last.pth"), ("learned_comparator", "last_raw.pth")],
)
def test_incomplete_training_resumes_from_family_specific_last_checkpoint(
    tmp_path: Path, model_kind: str, checkpoint_name: str
) -> None:
    model_root = tmp_path / model_kind
    checkpoint = model_root / "train" / checkpoint_name
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    assert (
        _resume_checkpoint_or_archive(model_root, model_kind=model_kind) == checkpoint
    )


def test_latest_resume_ignores_complete_sessions(tmp_path: Path) -> None:
    root = tmp_path / "cross_validation" / "k5"
    failed = root / "20260909-120000-000001"
    complete = root / "20260909-130000-000001"
    for path, status in ((failed, "failed"), (complete, "complete")):
        path.mkdir(parents=True)
        (path / "protocol.json").write_text("{}", encoding="utf-8")
        (path / "status.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )
    assert latest_resumable_session(tmp_path, 5) == failed


def test_isolated_worker_forces_mkl_to_pytorch_compatible_gnu_layer(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MKL_THREADING_LAYER", "INTEL")
    worker_environment = _isolated_worker_environment("gnu")
    assert worker_environment["MKL_THREADING_LAYER"] == "GNU"
    with pytest.raises(ValueError, match="must be GNU"):
        _isolated_worker_environment("INTEL")


def test_config_inventory_hashes_fully_resolved_includes(tmp_path: Path) -> None:
    source = Path("configs/pa_hydrokan.xml").resolve()
    override = tmp_path / "changed.xml"
    override.write_text(
        "<config>\n"
        f'  <include path="{source}" />\n'
        "  <model><dropout type=\"float\">0.123</dropout></model>\n"
        "</config>\n",
        encoding="utf-8",
    )
    original = config_inventory((source,))[0]
    changed = config_inventory((override,))[0]
    assert original["model"] == changed["model"] == "pa_hydrokan"
    assert original["resolved_config_sha256"] != changed["resolved_config_sha256"]


def test_runtime_source_hash_detects_python_changes(tmp_path: Path) -> None:
    (tmp_path / "utils").mkdir()
    source = tmp_path / "utils" / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first, first_count = runtime_source_sha256(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second, second_count = runtime_source_sha256(tmp_path)
    assert first != second
    assert first_count == second_count == 1


def test_legacy_or_changed_resume_protocol_is_rejected() -> None:
    inventory = config_inventory((Path("configs/pa_hydrokan.xml").resolve(),))
    with pytest.raises(CrossValidationProtocolError, match="legacy split protocol"):
        _validate_protocol_inventory({}, inventory, runtime_sha256="runtime")
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": "outer_k_fold_with_event_grouped_inner_holdout",
        "source_splits": ["train", "val", "test"],
        "models": inventory,
        "runtime_source_sha256": "old-runtime",
    }
    with pytest.raises(CrossValidationProtocolError, match="Python sources differ"):
        _validate_protocol_inventory(protocol, inventory, runtime_sha256="new-runtime")

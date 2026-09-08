from __future__ import annotations

from pathlib import Path

import evaluate
import pytest
import train

from utils.config import load_config
from utils.model_dispatch import configured_model_kind
from utils.run_paths import (
    evaluation_checkpoint_path,
    evaluation_output_path,
    train_output_path,
)


def _configured(path: str, tmp_path: Path) -> dict:
    config = load_config(path)
    config["runtime"]["train"]["output"] = tmp_path / "train_output"
    config["runtime"]["evaluation"]["output"] = tmp_path / "evaluation_output"
    return config


def test_common_runtime_resolves_model_and_ablation_paths() -> None:
    pa = load_config("configs/pa_hydrokan.xml")
    assert configured_model_kind(pa) == ("pa_hydrokan", "pa_hydrokan")
    assert train_output_path(pa).as_posix().endswith("train/pa_hydrokan/seed_20260908")
    assert evaluation_output_path(pa).as_posix().endswith(
        "evaluate/pa_hydrokan/val/seed_20260908"
    )
    assert evaluation_checkpoint_path(pa).as_posix().endswith(
        "train/pa_hydrokan/seed_20260908/best_raw.pth"
    )

    ablation = load_config(
        "configs/ablation/pa_hydrokan_no_topographic_affinity_edge_kan.xml"
    )
    assert train_output_path(ablation).as_posix().endswith(
        "ablation/no_tae_kan/seed_20260908"
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("configs/pa_hydrokan.xml", ("pa_hydrokan", "pa_hydrokan")),
        (
            "configs/compare/deep_learning/dlsim_attention_unet.xml",
            ("learned_comparator", "dlsim_attention_unet"),
        ),
        (
            "configs/compare/deep_learning/dlsim_linknet.xml",
            ("learned_comparator", "dlsim_linknet"),
        ),
        (
            "configs/compare/deep_learning/unet_depth_regression.xml",
            ("learned_comparator", "unet_depth_regression"),
        ),
        (
            "configs/compare/deep_learning/resnet18_depth_regression.xml",
            ("learned_comparator", "resnet18_depth_regression"),
        ),
        (
            "configs/compare/deep_learning/unetplusplus_depth_regression.xml",
            ("learned_comparator", "unetplusplus_depth_regression"),
        ),
        ("configs/compare/traditional/fwdet_v2.xml", ("traditional", "fwdet_v2")),
        ("configs/compare/traditional/tsa.xml", ("traditional", "tsa")),
        ("configs/compare/traditional/fldepth.xml", ("traditional", "fldepth")),
    ],
)
def test_every_model_configuration_has_one_unified_dispatch_kind(
    path: str, expected: tuple[str, str]
) -> None:
    config = load_config(path)
    assert configured_model_kind(config) == expected
    assert config["runtime"]["run_tag"] == "seed_20260908"
    assert config["training"]["epochs"] == 160
    assert config["optimizer"]["beta1"] == 0.9
    assert config["optimizer"]["beta2"] == 0.999


def test_unified_train_dispatches_pa_without_cli_overrides(monkeypatch, tmp_path: Path) -> None:
    config = _configured("configs/pa_hydrokan.xml", tmp_path)
    monkeypatch.setattr(train, "load_config", lambda _: config)
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return args.output

    monkeypatch.setattr(train, "run_pa_hydrokan_training", fake_run)
    result = train.run_from_config(Path("configs/pa_hydrokan.xml"))
    assert result == tmp_path / "train_output"
    assert captured["args"].device is None
    assert captured["args"].epochs is None
    assert captured["args"].batch_size is None
    assert captured["args"].output == tmp_path / "train_output"


def test_unified_train_dispatches_traditional_evaluation(monkeypatch, tmp_path: Path) -> None:
    config = _configured("configs/compare/traditional/fwdet_v2.xml", tmp_path)
    monkeypatch.setattr(train, "load_config", lambda _: config)
    captured = {}

    def fake_run(config_value, split, method, output, max_batches, save_predictions):
        captured.update(
            split=split,
            method=method,
            output=output,
            max_batches=max_batches,
            save_predictions=save_predictions,
        )
        return {"method": method}

    monkeypatch.setattr(train, "run_traditional_evaluation", fake_run)
    assert train.run_from_config(Path("configs/compare/traditional/fwdet_v2.xml")) == {
        "method": "fwdet_v2"
    }
    assert captured == {
        "split": "val",
        "method": "fwdet_v2",
        "output": tmp_path / "evaluation_output",
        "max_batches": None,
        "save_predictions": False,
    }


def test_unified_train_dispatches_learned_comparator_without_cli_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    config = _configured(
        "configs/compare/deep_learning/dlsim_attention_unet.xml", tmp_path
    )
    monkeypatch.setattr(train, "load_config", lambda _: config)
    captured = {}

    def fake_run(args, identifier):
        captured["args"] = args
        captured["identifier"] = identifier
        return args.output

    monkeypatch.setattr(train, "run_learned_training", fake_run)
    result = train.run_from_config(
        Path("configs/compare/deep_learning/dlsim_attention_unet.xml")
    )
    assert result == tmp_path / "train_output"
    assert captured["identifier"] == "dlsim_attention_unet"
    assert captured["args"].device is None
    assert captured["args"].epochs is None
    assert captured["args"].batch_size is None
    assert captured["args"].num_workers is None
    assert captured["args"].output == tmp_path / "train_output"


def test_unified_evaluate_resolves_configured_checkpoint(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "trained" / "best_raw.pth"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    config = _configured("configs/pa_hydrokan.xml", tmp_path)
    config["runtime"]["evaluation"]["checkpoint"] = checkpoint
    monkeypatch.setattr(evaluate, "load_config", lambda _: config)
    captured = {}

    def fake_run(*args):
        captured["args"] = args
        return {"pixel_micro_mae": 0.0}

    monkeypatch.setattr(evaluate, "run_pa_hydrokan_evaluation", fake_run)
    result = evaluate.run_from_config(Path("configs/pa_hydrokan.xml"))
    assert result == {"pixel_micro_mae": 0.0}
    assert captured["args"][1] == checkpoint
    assert captured["args"][2] == "val"
    assert captured["args"][4] == tmp_path / "evaluation_output"

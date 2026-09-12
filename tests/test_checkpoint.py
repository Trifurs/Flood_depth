from __future__ import annotations

from copy import deepcopy

import torch

from tools.evaluate_pa_hydrokan import dataset_fingerprint
from utils.checkpoint import initialize_from_checkpoint, load_checkpoint, save_checkpoint
from utils.config import jsonable_config
from utils.registry import build_model


def test_checkpoint_round_trip_is_fingerprint_strict(production_config, tmp_path) -> None:
    model = build_model(production_config)
    fingerprint = dataset_fingerprint(production_config)
    path = tmp_path / "model.pth"
    save_checkpoint(
        path,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=0,
        best_metric=1.0,
        resolved_config=jsonable_config(production_config),
        dataset_fingerprint=fingerprint,
        training_context={"epochs": 1},
    )
    restored = build_model(production_config)
    checkpoint = load_checkpoint(
        path,
        restored,
        expected_fingerprint=fingerprint,
        expected_training_identity_sha256=checkpoint_identity(
            production_config, fingerprint
        ),
    )
    assert checkpoint["output_semantics"] == "conditional_positive"
    source = next(model.parameters()).detach()
    target = next(restored.parameters()).detach()
    assert torch.equal(source, target)


def checkpoint_identity(config, fingerprint):
    from utils.checkpoint import training_identity_sha256

    return training_identity_sha256(
        jsonable_config(config), fingerprint, training_context={"epochs": 1}
    )


def test_compatible_initialization_preserves_old_path_and_zeros_new_adapters(
    production_config, tmp_path
) -> None:
    source = deepcopy(production_config)
    source["model"]["topographic_context_scales_m"] = []
    source["model"]["global_context_enabled"] = False
    source["model"]["global_depth_calibration_enabled"] = False
    source["model"]["depth_range_calibration_enabled"] = False
    source_model = build_model(source)
    fingerprint = dataset_fingerprint(source)
    path = tmp_path / "source.pth"
    save_checkpoint(
        path,
        source_model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=0,
        best_metric=1.0,
        resolved_config=jsonable_config(source),
        dataset_fingerprint=fingerprint,
    )
    enhanced = deepcopy(source)
    enhanced["model"]["topographic_context_scales_m"] = [320.0, 1280.0, 5120.0]
    enhanced["model"]["decoder_skip_fusion"] = "concatenative"
    enhanced["model"]["global_context_enabled"] = True
    enhanced["model"]["global_depth_calibration_enabled"] = True
    enhanced["model"]["depth_range_calibration_enabled"] = True
    target_model = build_model(enhanced)

    _, report = initialize_from_checkpoint(
        path,
        target_model,
        transfer="compatible",
        expected_fingerprint=fingerprint,
    )

    source_weight = source_model.state_dict()["terrain.stem.0.0.weight"]
    target_weight = target_model.state_dict()["terrain.stem.0.0.weight"]
    assert torch.equal(target_weight, source_weight)
    assert torch.count_nonzero(target_model.terrain.context_adapter.weight) == 0
    assert torch.count_nonzero(
        target_model.global_depth_calibration_adapter.network[-1].weight
    ) == 0
    assert torch.count_nonzero(
        target_model.global_depth_calibration_adapter.network[-1].bias
    ) == 0
    assert report["expanded_tensors"] == []
    assert "terrain.context_adapter.weight" in report["new_tensors"]
    assert any(
        name.startswith("global_depth_calibration_adapter.")
        for name in report["new_tensors"]
    )
    assert any(
        name.startswith("heads.depth_range_calibration.")
        for name in report["new_tensors"]
    )
    assert report["loaded_parameter_fraction"] > 0.9


def test_compatible_initialization_marks_shape_changes_as_new(
    production_config, tmp_path
) -> None:
    source_model = build_model(production_config)
    fingerprint = dataset_fingerprint(production_config)
    path = tmp_path / "source.pth"
    save_checkpoint(
        path,
        source_model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=0,
        best_metric=1.0,
        resolved_config=jsonable_config(production_config),
        dataset_fingerprint=fingerprint,
    )
    tensor_name = "heads.depth_head.trunk.3.bias"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model"][tensor_name] = torch.zeros(2)
    torch.save(payload, path)
    target_model = build_model(production_config)

    _, report = initialize_from_checkpoint(
        path,
        target_model,
        transfer="compatible",
        expected_fingerprint=fingerprint,
    )

    assert tensor_name in report["new_tensors"]
    assert tensor_name in report["unused_source_tensors"]

from __future__ import annotations

from copy import deepcopy

import torch

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from tools.train_pa_hydrokan import apply_train_only_calibration, prepare_train_only_calibration
from models.evidence_calibration import GlobalEvidenceCalibration
from models.task_head import ContextualDepthRangeCalibration
from models.pa_hydrokan import PAHydroKANHeads


def test_train_calibration_builds_a_frozen_weight_curve(production_config) -> None:
    config = deepcopy(production_config)
    config["loss"]["soft_depth_balance"] = True
    contract = DatasetContract.load(config["dataset"]["contract"])
    dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        "train",
        band_spec=resolve_band_spec(config, contract),
        input_spec=ModelInputSpec.from_config(config),
        minimum_event_band_fraction=float(
            config["dataset"]["minimum_event_band_fraction"]
        ),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    payload = prepare_train_only_calibration(config, dataset)
    balance = apply_train_only_calibration(config, payload)
    assert balance is not None
    assert balance.train_positive_pixels > 0
    assert config["model"]["depth_initialization_bias"] < 0


def test_grouped_global_evidence_matches_dense_concatenation() -> None:
    module = GlobalEvidenceCalibration(7, hidden_channels=8, dropout=0.0).eval()
    groups = (torch.randn(3, 2, 9, 11), torch.randn(3, 5, 9, 11))
    dense_scale, dense_bias = module(torch.cat(groups, dim=1))
    grouped_scale, grouped_bias = module(groups)
    assert torch.allclose(grouped_scale, dense_scale, atol=1.0e-7, rtol=1.0e-6)
    assert torch.allclose(grouped_bias, dense_bias, atol=1.0e-7, rtol=1.0e-6)


def test_contextual_depth_range_calibration_is_identity_initialized() -> None:
    module = ContextualDepthRangeCalibration(
        8,
        groups=4,
        width=8,
        maximum_scale_residual=1.0,
        maximum_bias_residual=2.0,
    )
    features = torch.randn(2, 8, 16, 16, requires_grad=True)
    depth_logit = torch.randn(2, 1, 16, 16, requires_grad=True)
    calibrated, scale, bias = module(features, depth_logit)
    assert torch.equal(calibrated, depth_logit)
    assert torch.count_nonzero(scale) == 0
    assert torch.count_nonzero(bias) == 0
    calibrated.mean().backward()
    assert module.projection.weight.grad is not None
    assert torch.isfinite(module.projection.weight.grad).all()


def test_contextual_depth_range_calibration_strength_zero_recovers_baseline() -> None:
    module = ContextualDepthRangeCalibration(
        8,
        groups=4,
        width=8,
        maximum_scale_residual=1.0,
        maximum_bias_residual=2.0,
        strength=0.0,
    )
    with torch.no_grad():
        module.projection.weight.fill_(0.1)
        module.projection.bias.copy_(torch.tensor((0.2, -0.3)))
    features = torch.randn(2, 8, 16, 16)
    depth_logit = torch.randn(2, 1, 16, 16)
    calibrated, scale, bias = module(features, depth_logit)
    assert torch.equal(calibrated, depth_logit)
    assert torch.count_nonzero(scale) == 0
    assert torch.count_nonzero(bias) == 0


def test_disabled_uncertainty_head_has_no_parameters_and_returns_fixed_scale() -> None:
    heads = PAHydroKANHeads(
        8,
        4,
        epsilon=0.001,
        maximum=5.0,
        depth_initialization_bias=-1.0,
        uncertainty_initial_scale_m=0.35,
        uncertainty_head_enabled=False,
    )
    assert heads.uncertainty_head is None
    outputs = heads(torch.randn(2, 8, 16, 16))
    assert outputs["uncertainty_scale"].shape == (2, 1, 16, 16)
    assert torch.all(outputs["uncertainty_scale"] == 0.35)

from __future__ import annotations

from pathlib import Path

import torch

from datasets.flooddepth_dataset import prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from models.s1_hydrology_backbone_v15 import SARHydrologyEncoderV15
from models.heads import GlobalEventDepthScale
from utils.config import load_config
from utils.registry import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v15_is_registered_and_uses_zero_inflated_s1_head() -> None:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15.xml")
    model = build_model(config)
    assert model.__class__.__name__ == "PAHydroKANS1V15"
    assert model.heads.support_enabled
    assert model.heads.output_semantics == "probability_weighted_v1"
    assert model.sar_encoder.reliability_projection


def test_v15_backbone_has_separate_detail_and_reliability_paths() -> None:
    encoder = SARHydrologyEncoderV15(
        state_channels=2,
        change_channels=3,
        qa_channels=5,
        reliability_channels=6,
        channels=[16, 32, 64, 96],
        dropout=0.0,
        groups=8,
        conditioning_channels=2,
    )
    inputs = {
        "pre": torch.randn(2, 2, 65, 67),
        "event": torch.randn(2, 2, 65, 67),
        "change": torch.randn(2, 3, 65, 67),
        "qa": torch.randn(2, 5, 65, 67),
        "reliability": torch.randn(2, 6, 65, 67),
        "valid": torch.ones(2, 1, 65, 67),
        "conditioning": torch.randn(2, 2, 65, 67),
    }
    features, diagnostics = encoder(**inputs)
    assert [value.shape[-2:] for value in features] == [(65, 67), (33, 34), (17, 17), (9, 9)]
    assert diagnostics["change_gate_mean"].isfinite()
    assert diagnostics["quality_mean"].isfinite()


def test_v15_detail_path_respects_branch_validity() -> None:
    torch.manual_seed(11)
    encoder = SARHydrologyEncoderV15(
        state_channels=2,
        change_channels=3,
        qa_channels=5,
        reliability_channels=6,
        channels=[16, 32, 64, 96],
        dropout=0.0,
        groups=8,
        conditioning_channels=0,
    ).eval()
    common = {
        "pre": torch.randn(1, 2, 33, 35),
        "event": torch.randn(1, 2, 33, 35),
        "change": torch.randn(1, 3, 33, 35),
        "qa": torch.randn(1, 5, 33, 35),
        "reliability": torch.randn(1, 6, 33, 35),
        "valid": torch.ones(1, 1, 33, 35),
    }
    masked = dict(common)
    masked["branch_validity"] = {
        "s1_t1": torch.zeros(1, 1, 33, 35),
        "s1_t2": torch.ones(1, 1, 33, 35),
        "s1_change": torch.ones(1, 1, 33, 35),
    }
    altered = dict(masked)
    altered["pre"] = torch.full_like(common["pre"], 1.0e6)
    first, _ = encoder(**masked)
    second, _ = encoder(**altered)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right)


def test_v15_event_scale_is_opt_in_and_identity_at_initialization() -> None:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_eventscale_control.xml")
    model = build_model(config)
    assert isinstance(model.event_depth_scale, GlobalEventDepthScale)
    assert model.heads.depth_output_semantics == "conditional_positive_v2"
    model.heads.set_depth_output_semantics("probability_weighted_v1")
    assert model.heads.depth_output_semantics == "probability_weighted_v1"


def test_v15_real_raster_forward_backward_excludes_s2() -> None:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15.xml")
    from tools.train import create_dataloaders

    _, loader, _, _ = create_dataloaders(config)
    batch = next(iter(loader))
    model = build_model(config)
    inputs = prepare_model_inputs(batch, ModelInputSpec.from_config(config))
    outputs = model(inputs)
    assert outputs["depth"].shape == batch["label"].shape
    assert outputs["depth"].isfinite().all()
    assert outputs["support_probability"].isfinite().all()
    assert outputs["expected_depth"].isfinite().all()
    assert "modality_weights" not in outputs
    assert not any(str(key).startswith("s2_") for key in inputs)
    outputs["depth"].mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

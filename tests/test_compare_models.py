from __future__ import annotations

from pathlib import Path

import pytest
import torch

from utils.config import load_config
from utils.registry import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    PROJECT_ROOT / "configs/compare/subset150_dlsim_linknet.xml",
    PROJECT_ROOT / "configs/compare/subset150_dlsim_attention_unet.xml",
)


def synthetic_inputs(size: int = 64) -> dict[str, torch.Tensor]:
    batch = 1
    return {
        "s1_t1": torch.randn(batch, 3, size, size),
        "s1_t2": torch.randn(batch, 3, size, size),
        "s1_change": torch.randn(batch, 4, size, size),
        "s2_t1": torch.randn(batch, 6, size, size),
        "s2_t2": torch.randn(batch, 6, size, size),
        "s2_change": torch.randn(batch, 3, size, size),
        "terrain": torch.randn(batch, 2, size, size),
        "terrain_raw": torch.randn(batch, 2, size, size),
        "reliability": torch.randn(batch, 12, size, size),
        "s1_valid": torch.ones(batch, 1, size, size),
        "s2_valid": torch.ones(batch, 1, size, size),
        "dem_valid": torch.ones(batch, 1, size, size),
    }


@pytest.mark.parametrize("config_path", CONFIGS)
def test_comparison_model_contract_and_backward(config_path: Path) -> None:
    config = load_config(config_path)
    model = build_model(config)
    inputs = synthetic_inputs()
    outputs = model(inputs)
    for name in (
        "depth",
        "support_logits",
        "support_probability",
        "conditional_depth",
        "positive_depth",
        "expected_depth",
        "uncertainty_scale",
    ):
        assert outputs[name].shape == (1, 1, 64, 64)
        assert torch.isfinite(outputs[name]).all()
    assert outputs["physical_features"]["z_hyd"].shape == (1, 1, 64, 64)
    assert outputs["physical_features"]["local_relief"].shape == (1, 1, 64, 64)
    torch.testing.assert_close(outputs["depth"], outputs["conditional_depth"])
    torch.testing.assert_close(
        outputs["expected_depth"],
        outputs["support_probability"] * outputs["conditional_depth"],
    )
    (outputs["depth"].mean() + outputs["uncertainty_scale"].mean()).backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("config_path", CONFIGS)
def test_comparison_model_rejects_label_derived_input(config_path: Path) -> None:
    model = build_model(load_config(config_path))
    inputs = synthetic_inputs()
    inputs["valid_depth_mask"] = torch.ones(1, 1, 64, 64)
    with pytest.raises(ValueError, match="Label-derived"):
        model(inputs)

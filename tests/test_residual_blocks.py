"""Shape and gradient contracts for configurable PA-HydroKAN residual blocks."""

from __future__ import annotations

import pytest
import torch

from models.efficient_blocks import (
    EfficientResidualBlock,
    SpatialResidualBlock,
    residual_block,
)
from models.s1_hydrology_backbone import JointSARHydrologyEncoder


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    (
        ("efficient", EfficientResidualBlock),
        ("spatial", SpatialResidualBlock),
    ),
)
def test_residual_block_preserves_shape_and_gradient(kind, expected_type):
    module = residual_block(kind, channels=16, dropout=0.0, groups=8)
    value = torch.randn(2, 16, 17, 19, requires_grad=True)
    output = module(value)

    assert isinstance(module, expected_type)
    assert output.shape == value.shape
    output.mean().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_residual_block_rejects_unknown_kind():
    with pytest.raises(ValueError, match="efficient.*spatial"):
        residual_block("unknown", channels=16, dropout=0.0, groups=8)


def test_joint_sar_encoder_preserves_multiscale_contract():
    encoder = JointSARHydrologyEncoder(
        state_channels=2,
        change_channels=3,
        channels=(8, 16, 24, 32),
        dropout=0.0,
        groups=8,
        block_kind="spatial",
        conditioning_channels=2,
    )
    shape = (2, 1, 32, 32)
    valid = torch.ones(shape)
    reliability = [
        torch.randn(2, width, 32 // (2**scale), 32 // (2**scale))
        for scale, width in enumerate((8, 16, 24, 32))
    ]
    outputs, diagnostics = encoder(
        torch.randn(2, 2, 32, 32),
        torch.randn(2, 2, 32, 32),
        torch.randn(2, 3, 32, 32),
        valid,
        torch.randn(2, 2, 32, 32),
        {
            "s1_t1": valid,
            "s1_t2": valid,
            "s1_change": valid,
        },
        reliability,
    )

    assert [tuple(value.shape) for value in outputs] == [
        (2, 8, 32, 32),
        (2, 16, 16, 16),
        (2, 24, 8, 8),
        (2, 32, 4, 4),
    ]
    assert len(diagnostics["quality_gates"]) == 4
    assert len(diagnostics["change_evidence"]) == 4
    sum(value.mean() for value in outputs).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )

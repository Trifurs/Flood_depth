from __future__ import annotations

import pytest
import torch

from models.terrain_features import TerrainFeaturePyramid


def test_multiscale_terrain_adds_only_label_free_dsm_context() -> None:
    pyramid = TerrainFeaturePyramid(
        [8, 16, 32, 64],
        dropout=0.0,
        groups=8,
        context_kernel_sizes=[9, 33, 65],
    )
    first_convolution = pyramid.stem[0][0]
    assert isinstance(first_convolution, torch.nn.Conv2d)
    assert first_convolution.in_channels == 14

    normalized = torch.randn(1, 2, 96, 96)
    raw = torch.randn(1, 2, 96, 96)
    valid = torch.ones(1, 1, 96, 96)
    features, physical = pyramid(normalized, raw, valid)
    assert [tuple(value.shape) for value in features] == [
        (1, 8, 96, 96),
        (1, 16, 48, 48),
        (1, 32, 24, 24),
        (1, 64, 12, 12),
    ]
    assert torch.isfinite(torch.cat([value.flatten() for value in features])).all()
    assert physical["z_hyd"].shape == (1, 1, 96, 96)


@pytest.mark.parametrize("kernels", ([33, 65], [9, 32], [9, -1]))
def test_multiscale_terrain_rejects_invalid_context_kernels(kernels: list[int]) -> None:
    with pytest.raises(ValueError):
        TerrainFeaturePyramid([8, 16, 32, 64], context_kernel_sizes=kernels)

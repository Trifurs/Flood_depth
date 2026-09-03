from __future__ import annotations

import torch

from models.terrain_features_v14 import (
    ground_like_proxy,
    path_barrier_proxy,
    valid_central_gradients_v14,
)


def test_v14_gradients_use_metric_units_and_invalid_neighbours_are_zero() -> None:
    elevation = torch.arange(7, dtype=torch.float32).view(1, 1, 1, 7).expand(1, 1, 7, 7) * 20.0
    valid = torch.ones_like(elevation)
    gx, gy, gx_valid, gy_valid = valid_central_gradients_v14(elevation, valid, 20.0, 20.0)
    torch.testing.assert_close(gx[..., 3, 3], torch.ones(1, 1))
    torch.testing.assert_close(gy[..., 3, 3], torch.zeros(1, 1))
    assert gx_valid[..., 3, 3].item() == 1.0 and gy_valid[..., 3, 3].item() == 1.0

    valid[..., 3, 2] = 0.0
    gx, _, gx_valid, _ = valid_central_gradients_v14(elevation, valid, 20.0, 20.0)
    assert gx[..., 3, 3].item() == 0.0 and gx_valid[..., 3, 3].item() == 0.0


def test_ground_proxy_removes_isolated_dsm_obstacle() -> None:
    elevation = torch.full((1, 1, 9, 9), 100.0)
    elevation[..., 4, 4] = 130.0
    valid = torch.ones_like(elevation)
    proxy = ground_like_proxy(elevation, valid, kernel_size=3)
    torch.testing.assert_close(proxy[..., 4, 4], torch.tensor([[100.0]]))
    torch.testing.assert_close(proxy[..., 2:7, 2:7], torch.full((1, 1, 5, 5), 100.0))


def test_path_barrier_includes_intermediate_pixels_and_invalidates_broken_paths() -> None:
    elevation = torch.zeros(1, 1, 9, 9)
    ground = torch.zeros_like(elevation)
    valid = torch.ones_like(elevation)
    elevation[..., 4, 4] = 10.0
    barrier, edge_valid = path_barrier_proxy(
        elevation, valid, pixel_step=4, ground=ground, directions=((0, 1),)
    )
    assert barrier.shape == (1, 1, 1, 9, 9)
    assert barrier[..., 4, 6].item() == 10.0  # path 6 -> 2 crosses the ridge at 4
    assert edge_valid[..., 4, 6].item() == 1.0

    valid[..., 4, 4] = 0.0
    _, broken_valid = path_barrier_proxy(
        elevation, valid, pixel_step=4, ground=ground, directions=((0, 1),)
    )
    assert broken_valid[..., 4, 6].item() == 0.0

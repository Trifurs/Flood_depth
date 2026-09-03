from __future__ import annotations

import torch

from losses.depth_losses import tail_underprediction_loss
from losses.physics_losses import gated_terrain_order_loss, tolerant_wse_slope_loss


def _valid(shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = torch.ones(shape)
    return value, value.clone(), value.clone()


def test_v14_terrain_order_has_tolerance_and_correct_downhill_sign() -> None:
    z = torch.tensor([[[[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]]])
    positive, dem, sensor = _valid(tuple(z.shape))
    downhill_depth = 1.0 - z
    loss = gated_terrain_order_loss(
        downhill_depth, z, positive, dem, sensor,
        depth_order_tolerance_m=0.02, terrain_step_min_m=0.02, terrain_step_max_m=0.75,
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))
    uphill_depth = z.clone().requires_grad_()
    loss = gated_terrain_order_loss(uphill_depth, z, positive, dem, sensor)
    assert loss.item() > 0
    loss.backward()
    assert uphill_depth.grad is not None and torch.isfinite(uphill_depth.grad).all()


def test_v14_wse_slope_uses_diagonal_metric_distance_and_tolerance() -> None:
    rows = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1)
    cols = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5)
    depth = (rows + cols) * 0.3  # 0.015 per axis pixel; diagonal is 0.0212 per metre
    z = torch.zeros_like(depth)
    positive, dem, sensor = _valid(tuple(depth.shape))
    loss, diagnostics = tolerant_wse_slope_loss(
        depth, z, positive, dem, sensor, pixel_size_m=20.0,
        wse_slope_tolerance=0.02, relief=torch.zeros_like(depth), return_diagnostics=True,
    )
    assert loss.item() > 0 and diagnostics["violation_fraction"].item() > 0
    flat_loss = tolerant_wse_slope_loss(
        torch.zeros_like(depth), z, positive, dem, sensor, pixel_size_m=20.0,
    )
    torch.testing.assert_close(flat_loss, torch.tensor(0.0))


def test_v14_physics_and_tail_losses_are_zero_without_active_pairs() -> None:
    depth = torch.ones(1, 1, 3, 3, requires_grad=True)
    z = torch.zeros_like(depth)
    invalid = torch.zeros_like(depth)
    terrain = gated_terrain_order_loss(depth, z, invalid, invalid, invalid)
    slope = tolerant_wse_slope_loss(depth, z, invalid, invalid, invalid)
    tail = tail_underprediction_loss(depth, depth, invalid, tail_threshold_m=1.0)
    torch.testing.assert_close(terrain, torch.tensor(0.0))
    torch.testing.assert_close(slope, torch.tensor(0.0))
    torch.testing.assert_close(tail, torch.tensor(0.0))


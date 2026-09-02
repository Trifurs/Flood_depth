from __future__ import annotations

import torch

from losses.physics_losses import (
    reference_gated_wse_gradient_loss,
    terrain_order_violation_loss,
)
from metrics.physical_metrics import (
    local_wse_laplacian_reference_error,
    reference_gated_wse_gradient_mae,
    terrain_order_violation_metrics,
)


def _loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    z_hyd = torch.tensor(
        [[[[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]]]
    )
    valid = torch.ones_like(target, dtype=torch.bool) if mask is None else mask
    return reference_gated_wse_gradient_loss(
        prediction,
        target,
        z_hyd,
        valid,
        torch.ones_like(target),
        torch.ones_like(target),
        torch.zeros_like(target),
        ["event_a"],
        sigma_time=0.25,
        sigma_reference=0.12,
        sigma_terrain=0.75,
        beta=0.05,
    )


def test_reference_gated_wse_gradient_is_zero_for_matching_structure() -> None:
    target = torch.tensor(
        [[[[1.0, 0.9, 0.8], [1.0, 0.9, 0.8], [1.0, 0.9, 0.8]]]]
    )
    prediction = target.clone().requires_grad_()
    loss = _loss(prediction, target)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_reference_gated_wse_gradient_detects_local_shape_error_not_offset() -> None:
    target = torch.tensor(
        [[[[1.0, 0.9, 0.8], [1.0, 0.9, 0.8], [1.0, 0.9, 0.8]]]]
    )
    constant_offset = _loss(target + 0.5, target)
    torch.testing.assert_close(constant_offset, torch.tensor(0.0), atol=1e-7, rtol=0)
    distorted = target.clone()
    distorted[..., 1, 1] += 0.5
    assert _loss(distorted, target) > 0


def test_reference_gated_wse_gradient_ignores_invalid_endpoint_values() -> None:
    target = torch.tensor(
        [[[[1.0, 0.9, 0.8], [1.0, 0.9, 0.8], [1.0, 0.9, 0.8]]]]
    )
    prediction = target.clone()
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 0, 0] = False
    baseline = _loss(prediction, target, mask)
    altered_target = target.clone()
    altered_prediction = prediction.clone()
    altered_target[..., 0, 0] = -1e6
    altered_prediction[..., 0, 0] = 1e6
    changed = _loss(altered_prediction, altered_target, mask)
    torch.testing.assert_close(changed, baseline, atol=0, rtol=0)


def test_reference_gradient_pixel_micro_is_event_invariant() -> None:
    target = torch.tensor(
        [
            [[[1.0, 0.9, 0.8], [1.0, 0.9, 0.8], [1.0, 0.9, 0.8]]],
            [[[1.0, 0.9, 0.8], [1.0, 0.9, 0.8], [1.0, 0.9, 0.8]]],
        ]
    )
    prediction = target.clone()
    prediction[0, :, 1, 1] += 0.5
    z_hyd = torch.tensor(
        [[[[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]]]
    ).expand_as(target)
    valid = torch.ones_like(target)
    arguments = (
        prediction,
        target,
        z_hyd,
        valid,
        valid,
        valid,
        torch.zeros_like(target),
    )
    first = reference_gated_wse_gradient_loss(
        *arguments,
        ["event_a", "event_b"],
        sigma_time=0.25,
        sigma_reference=0.12,
        sigma_terrain=0.75,
        beta=0.05,
        aggregation_mode="pixel_micro",
    )
    renamed = reference_gated_wse_gradient_loss(
        *arguments,
        ["same", "same"],
        sigma_time=0.25,
        sigma_reference=0.12,
        sigma_terrain=0.75,
        beta=0.05,
        aggregation_mode="pixel_micro",
    )
    torch.testing.assert_close(first, renamed, rtol=0, atol=0)


def test_reference_physical_metrics_are_zero_for_matching_prediction() -> None:
    target = torch.tensor(
        [[[[1.0, 0.9, 0.8], [1.0, 0.9, 0.8], [1.0, 0.9, 0.8]]]]
    ).numpy()
    z_hyd = torch.tensor(
        [[[[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]]]
    ).numpy()
    valid = torch.ones((1, 1, 3, 3), dtype=torch.bool).numpy()
    day = torch.zeros((1, 1, 3, 3)).numpy()
    assert local_wse_laplacian_reference_error(target, target, z_hyd, valid) == 0.0
    assert reference_gated_wse_gradient_mae(
        target, target, z_hyd, valid, day
    ) == 0.0


def test_terrain_order_penalizes_uphill_deepening_but_not_flat_depth() -> None:
    z_hyd = torch.tensor(
        [[[[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]]]
    )
    valid = torch.ones_like(z_hyd)
    day = torch.zeros_like(z_hyd)

    def terrain_loss(depth: torch.Tensor) -> torch.Tensor:
        return terrain_order_violation_loss(
            depth,
            z_hyd,
            valid,
            valid,
            valid,
            day,
            ["event_a"],
            sigma_time=0.25,
            minimum_terrain_step_m=0.02,
            maximum_terrain_step_m=0.75,
            beta=0.02,
        )

    flat = torch.ones_like(z_hyd, requires_grad=True)
    torch.testing.assert_close(terrain_loss(flat), torch.tensor(0.0))
    downhill_deepening = (1.0 - z_hyd).requires_grad_()
    torch.testing.assert_close(
        terrain_loss(downhill_deepening), torch.tensor(0.0), atol=1e-7, rtol=0
    )
    uphill = z_hyd.clone().requires_grad_()
    uphill_loss = terrain_loss(uphill)
    assert uphill_loss > 0
    uphill_loss.backward()
    assert uphill.grad is not None
    assert torch.isfinite(uphill.grad).all()

    metric = terrain_order_violation_metrics(
        uphill.detach().numpy(),
        z_hyd.numpy(),
        valid.numpy() > 0,
        day.numpy(),
    )
    assert metric["mae"] > 0
    assert metric["fraction"] == 1.0


def test_terrain_order_ignores_pairs_with_an_invalid_endpoint() -> None:
    z_hyd = torch.tensor([[[[0.0, 0.1, 0.2]]]])
    depth = torch.tensor([[[[0.0, 0.1, 10.0]]]])
    valid = torch.ones_like(z_hyd)
    valid[..., 2] = 0.0
    loss = terrain_order_violation_loss(
        depth,
        z_hyd,
        valid,
        valid,
        valid,
        torch.zeros_like(z_hyd),
        ["event_a"],
        sigma_time=0.25,
        minimum_terrain_step_m=0.02,
        maximum_terrain_step_m=0.75,
        beta=0.02,
    )
    expected = terrain_order_violation_loss(
        depth[..., :2],
        z_hyd[..., :2],
        valid[..., :2],
        valid[..., :2],
        valid[..., :2],
        torch.zeros_like(z_hyd[..., :2]),
        ["event_a"],
        sigma_time=0.25,
        minimum_terrain_step_m=0.02,
        maximum_terrain_step_m=0.75,
        beta=0.02,
    )
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)

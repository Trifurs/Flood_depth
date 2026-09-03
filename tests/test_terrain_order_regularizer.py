import torch

from losses.physics_losses import terrain_order_violation_loss


def _loss(depth, z):
    positive = torch.ones_like(depth)
    valid = torch.ones_like(depth)
    return terrain_order_violation_loss(depth, z, positive, valid, valid, torch.zeros_like(depth), None, .5, .05, 5.0, .05, "pixel_micro")


def test_only_uphill_deepening_is_penalized_and_tolerance_applies():
    z = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    assert float(_loss(torch.tensor([[[[1.0, 1.1, 1.2, 1.3]]]]), z)) > 0
    assert float(_loss(torch.tensor([[[[1.3, 1.2, 1.1, 1.0]]]]), z)) == 0
    assert float(_loss(torch.tensor([[[[1.0, 1.02, 1.04, 1.06]]]]), z)) == 0


def test_no_valid_pair_returns_differentiable_zero():
    depth = torch.ones(1, 1, 3, 3, requires_grad=True)
    zero = torch.zeros_like(depth)
    value = terrain_order_violation_loss(depth, zero, zero, zero, zero, zero, None, .5, .05, 5.0, .05, "pixel_micro")
    assert torch.isfinite(value)
    value.backward()


def test_high_relief_edges_are_downweighted():
    from losses.physics_losses import terrain_order_violation_loss
    depth = torch.tensor([[[[0.0, 1.0, 1.1, 1.1]]]])
    z = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    ones = torch.ones_like(depth)
    low = terrain_order_violation_loss(
        depth, z, ones, ones, ones, torch.zeros_like(depth), None,
        .5, .05, 5.0, .05, "pixel_micro", torch.zeros_like(depth), 12.0, 8.0,
    )
    high = terrain_order_violation_loss(
        depth, z, ones, ones, ones, torch.zeros_like(depth), None,
        .5, .05, 5.0, .05, "pixel_micro", torch.tensor([[[[28.0, 28.0, 0.0, 0.0]]]]), 12.0, 8.0,
    )
    assert 0.0 < float(high) < float(low)

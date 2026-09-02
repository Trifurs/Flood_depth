from __future__ import annotations

import math

import torch

from models.heads import GlobalEventDepthScale


def test_event_depth_scale_is_identity_at_initialization_and_bounded() -> None:
    branch = GlobalEventDepthScale(
        channels=4, hidden_channels=3, maximum_absolute_log_scale=math.log(2.0)
    )
    features = torch.randn(2, 4, 5, 7, requires_grad=True)
    valid = torch.ones(2, 1, 10, 14)
    scale, log_scale = branch(features, valid)
    torch.testing.assert_close(scale, torch.ones_like(scale))
    torch.testing.assert_close(log_scale, torch.zeros_like(log_scale))

    final = branch.predictor[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.bias.fill_(100.0)
    upper, _ = branch(features, valid)
    assert torch.all(upper <= 2.0)
    assert torch.all(upper > 1.99)


def test_event_depth_scale_ignores_invalid_values_and_handles_empty_mask() -> None:
    torch.manual_seed(5)
    branch = GlobalEventDepthScale(channels=2, hidden_channels=2)
    final = branch.predictor[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.fill_(0.2)
        final.bias.fill_(0.1)

    features = torch.randn(1, 2, 4, 4)
    valid = torch.zeros(1, 1, 4, 4)
    valid[:, :, :2, :] = 1.0
    first, _ = branch(features, valid)
    altered = features.clone()
    altered[:, :, 2:, :] = 1e6
    second, _ = branch(altered, valid)
    torch.testing.assert_close(first, second)

    empty_scale, empty_log_scale = branch(features, torch.zeros_like(valid))
    torch.testing.assert_close(empty_scale, torch.ones_like(empty_scale))
    torch.testing.assert_close(empty_log_scale, torch.zeros_like(empty_log_scale))

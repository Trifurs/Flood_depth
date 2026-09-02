from __future__ import annotations

import io

import torch

from models.kan_layers import KANLinear


def test_kan_forward_gradient_state_and_repeatability() -> None:
    torch.manual_seed(7)
    layer = KANLinear(8, 3, grid_size=8, spline_order=3)
    inputs = torch.randn(2, 5, 8, requires_grad=True)
    first = layer(inputs)
    second = layer(inputs)
    assert first.shape == (2, 5, 3)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert layer.spline_coefficients.grad is not None
    stream = io.BytesIO()
    torch.save(layer.state_dict(), stream)
    stream.seek(0)
    restored = KANLinear(8, 3, grid_size=8, spline_order=3)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(first.detach(), restored(inputs.detach()), rtol=0, atol=0)

from __future__ import annotations

import torch

from models.hydro_edge_kan_v14 import HydroEdgeKANV14
from models.terrain_graph_kan import DIRECTIONS, _roll_with_boundary_mask


def _physical(size: int = 32) -> dict[str, torch.Tensor]:
    elevation = torch.zeros(1, 1, size, size)
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    elevation[0, 0] = x.float() * 0.2 + y.float() * 0.1
    valid = torch.ones_like(elevation)
    return {
        "dsm_elevation": elevation,
        "physics_elevation": elevation,
        "z_ground_proxy": elevation,
        "local_relief": torch.ones_like(elevation),
        "dem_valid": valid,
    }


def test_v14_static_edge_descriptors_are_reverse_symmetric() -> None:
    graph = HydroEdgeKANV14(8, heads=2, graph_scale=4, diagnostic_mode=True)
    features = torch.randn(1, 8, 8, 8)
    descriptor, raw, edge_valid, *_ = graph._descriptors(
        _physical(), features.shape[-2:], torch.ones(1, 1, 32, 32)
    )
    reverse = {direction: index for index, direction in enumerate(DIRECTIONS)}
    for index, (dy, dx) in enumerate(DIRECTIONS):
        opposite = reverse[(-dy, -dx)]
        reverse_map, boundary = _roll_with_boundary_mask(raw[:, opposite], dy, dx)
        comparable = (edge_valid[:, index] * boundary).expand_as(raw[:, index])
        assert torch.allclose(
            raw[:, index][comparable > 0.5], reverse_map[comparable > 0.5], atol=1e-5, rtol=1e-5
        )
    assert descriptor.shape[2] == 3


def test_v14_prior_penalizes_larger_barrier_and_gamma_is_bounded() -> None:
    graph = HydroEdgeKANV14(8, heads=2, graph_scale=4, diagnostic_mode=True)
    low = _physical()
    high = _physical()
    high["dsm_elevation"][..., 16, 12:21] += 20.0
    _, low_raw, *_ = graph._descriptors(low, (8, 8), torch.ones(1, 1, 32, 32))
    _, high_raw, *_ = graph._descriptors(high, (8, 8), torch.ones(1, 1, 32, 32))
    prior_low = -(graph.prior_scales.view(1, 1, 3, 1, 1) * low_raw).sum(dim=2)
    prior_high = -(graph.prior_scales.view(1, 1, 3, 1, 1) * high_raw).sum(dim=2)
    increased = high_raw[..., 1, :, :] > low_raw[..., 1, :, :]
    assert torch.any(increased)
    assert torch.all(prior_high[increased] <= prior_low[increased] + 1e-6)
    assert torch.all(graph.gamma > 0) and torch.all(graph.gamma < graph.gamma_max)


def test_v14_zero_residual_has_small_identity_update_and_trainable_kan() -> None:
    torch.manual_seed(20260903)
    graph = HydroEdgeKANV14(8, heads=2, graph_scale=4, zero_residual_init=True)
    features = torch.randn(1, 8, 8, 8, requires_grad=True)
    output, diagnostics = graph(
        features, _physical(), torch.zeros(1, 1, 32, 32), sensor_valid=torch.ones(1, 1, 32, 32)
    )
    ratio = (output - features).square().mean().sqrt() / features.square().mean().sqrt()
    assert ratio.item() < 0.1
    (output.square().mean()).backward()
    assert graph.edge_kan.spline_coefficients.grad is not None
    assert torch.isfinite(graph.edge_kan.spline_coefficients.grad).all()
    assert diagnostics["graph_update_rms_ratio"].item() < 0.1


def test_v14_invalid_sensor_edges_are_identity() -> None:
    graph = HydroEdgeKANV14(8, heads=2, graph_scale=8)
    features = torch.randn(1, 8, 4, 4)
    output, diagnostics = graph(
        features, _physical(32), torch.zeros(1, 1, 32, 32), sensor_valid=torch.zeros(1, 1, 32, 32)
    )
    torch.testing.assert_close(output, features)
    assert all(torch.isfinite(value).all() for value in diagnostics.values() if torch.is_tensor(value))

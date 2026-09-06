"""Focused V15.2 Graph/KAN representation tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from models.hydro_edge_kan_v15_2 import HydroEdgeKANV15_2


def _physical(batch_size: int = 2, size: int = 32) -> dict[str, torch.Tensor]:
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    elevation = (0.05 * x + 0.02 * y).float().view(1, 1, size, size)
    elevation = elevation.repeat(batch_size, 1, 1, 1)
    valid = torch.ones_like(elevation)
    return {
        "dsm_elevation": elevation,
        "physics_elevation": elevation,
        "z_ground_proxy": elevation,
        "local_relief": torch.full_like(elevation, 0.1),
        "dem_valid": valid,
    }


def _graph() -> HydroEdgeKANV15_2:
    return HydroEdgeKANV15_2(
        8,
        heads=2,
        grid_size=4,
        graph_feature_stride=4,
        mapping_temperatures=(2.25, 1.5, 1.25, 1.0),
        regularization_enabled=True,
    )


def test_v15_2_featurewise_mapping_has_independent_temperatures_and_no_layernorm() -> None:
    graph = _graph()
    raw = torch.ones(1, 1, 4, 1, 1)
    bounded = graph.map_edge_features(raw)
    expected = torch.tanh(
        torch.tensor([1.0 / 2.25, 1.0 / 1.5, 1.0 / 1.25, 1.0])
    )
    torch.testing.assert_close(bounded.reshape(-1), expected)
    assert isinstance(graph.edge_kan.normalization, nn.Identity)
    identity = graph.graph_identity()
    assert identity["mapping_temperature_by_feature"]["signed_grade"] == pytest.approx(2.25)
    assert identity["kan_explicit_scaling_has_layernorm"] is False


def test_v15_2_mapping_runs_once_and_effective_spline_regularization_uses_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    calls = 0
    original = graph.map_edge_features

    def counted(raw: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(raw)

    monkeypatch.setattr(graph, "map_edge_features", counted)
    features = torch.randn(2, 8, 8, 8, requires_grad=True)
    output, _ = graph(
        features,
        _physical(),
        torch.ones(2, 1, 32, 32),
        feature_stride=4,
    )
    assert calls == 1
    output.square().mean().backward()
    gradient = graph.edge_kan.spline_coefficients.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0

    with torch.no_grad():
        graph.edge_kan.spline_coefficients.fill_(1.0)
        graph.edge_kan.spline_scale.fill_(2.0)
    magnitude, curvature = graph.spline_regularization()
    assert magnitude.item() == pytest.approx(2.0, abs=1.0e-6)
    assert curvature.item() == pytest.approx(0.0, abs=1.0e-6)


def test_v15_2_masked_barrier_pooling_and_edge_eligibility_are_explicit() -> None:
    values = torch.tensor([[[[[10.0, 0.0], [0.0, 0.0]]]]])
    valid = torch.tensor([[[[[1.0, 0.0], [0.0, 0.0]]]]])
    pooled, fraction = HydroEdgeKANV15_2._masked_directional_pool(values, valid, (1, 1))
    assert pooled.item() == pytest.approx(10.0)
    assert fraction.item() == pytest.approx(0.25)

    graph = _graph()
    physical = _physical(batch_size=1)
    _, _, eligible, confidence = graph._edge_descriptors(
        physical, torch.full((1, 1, 32, 32), 0.49), (8, 8)
    )
    assert eligible.sum().item() == 0
    assert torch.isfinite(confidence).all()
    _, _, eligible, _ = graph._edge_descriptors(
        physical, torch.ones(1, 1, 32, 32), (8, 8)
    )
    assert eligible.sum().item() > 0


def test_v15_2_no_eligible_edge_and_constant_latent_are_identity() -> None:
    graph = _graph()
    features = torch.randn(2, 8, 8, 8)
    physical = _physical()
    no_edges, diagnostics = graph(
        features, physical, torch.zeros(2, 1, 32, 32), feature_stride=4
    )
    torch.testing.assert_close(no_edges, features, rtol=0.0, atol=0.0)
    assert diagnostics["edge_eligible_fraction"].item() == 0.0

    constant = torch.ones_like(features)
    constant_output, _ = graph(
        constant, physical, torch.ones(2, 1, 32, 32), feature_stride=4
    )
    torch.testing.assert_close(constant_output, constant, rtol=0.0, atol=1.0e-6)

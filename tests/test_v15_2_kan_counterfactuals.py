"""Unit coverage for the validation-only KAN causal interventions."""

from __future__ import annotations

import torch

from models.hydro_edge_kan_v15_1 import HydroEdgeKANV15_1


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


def test_v15_2_counterfactual_modes_are_finite_and_graph_off_is_identity() -> None:
    graph = HydroEdgeKANV15_1(8, heads=2, graph_feature_stride=4)
    features = torch.randn(2, 8, 8, 8)
    physical = _physical()
    sensor = torch.ones(2, 1, 32, 32)

    graph.set_counterfactual_mode("graph_off")
    identity, diagnostics = graph(features, physical, sensor, feature_stride=4)
    torch.testing.assert_close(identity, features, rtol=0.0, atol=0.0)
    assert float(diagnostics["graph_update_rms"]) == 0.0

    for mode in ("full", "spline_off", "base_off", "constant_gate", "shuffled_terrain"):
        graph.set_counterfactual_mode(mode, constant_gate=1.0, shuffle_seed=17)
        output, diagnostics = graph(features, physical, sensor, feature_stride=4)
        assert output.shape == features.shape
        assert torch.isfinite(output).all()
        assert torch.isfinite(diagnostics["graph_update_rms_ratio"])

    graph.set_counterfactual_mode("shuffled_terrain", shuffle_seed=17)
    first, _ = graph(features, physical, sensor, feature_stride=4)
    second, _ = graph(features, physical, sensor, feature_stride=4)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

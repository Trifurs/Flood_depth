import torch

from models.terrain_graph_kan_v13 import VectorizedTerrainGraphKANV131


def _inputs():
    torch.manual_seed(20260831)
    features = torch.randn(1, 8, 4, 4, requires_grad=True)
    physical = {
        key: torch.rand(1, 1, 32, 32)
        for key in ("dem_valid", "z_hyd", "slope", "z_barrier")
    }
    physical["dem_valid"].fill_(1)
    reliability = torch.rand(1, 1, 32, 32)
    weights = torch.softmax(torch.randn(1, 2, 4, 4), dim=1)
    sensor_valid = torch.ones(1, 1, 32, 32)
    return features, physical, reliability, weights, sensor_valid


def test_vectorized_and_reference_graph_paths_match() -> None:
    graph = VectorizedTerrainGraphKANV131(8, heads=2, grid_size=4)
    with torch.no_grad():
        graph.gamma.fill_(0.4)
    args = _inputs()
    fast, fast_diag = graph(*args)
    reference, ref_diag = graph.forward_reference(*args)
    torch.testing.assert_close(fast, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fast_diag["gate_mean"], ref_diag["gate_mean"])
    fast.square().mean().backward()
    assert all(torch.isfinite(value).all() for value in fast_diag.values() if torch.is_tensor(value))


def test_graph_zero_validity_is_finite_and_identity() -> None:
    graph = VectorizedTerrainGraphKANV131(8, heads=2, grid_size=4)
    args = list(_inputs())
    args[-1] = torch.zeros_like(args[-1])
    output, diagnostics = graph(*args)
    torch.testing.assert_close(output, args[0])
    assert all(torch.isfinite(value).all() for value in diagnostics.values() if torch.is_tensor(value))

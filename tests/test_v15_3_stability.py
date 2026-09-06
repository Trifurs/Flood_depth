"""Focused V15.3 stability and S1-only contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.hydro_edge_kan_v15_2 import HydroEdgeKANV15_2
from utils.config import load_config
from utils.registry import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _physical(size: int = 32) -> dict[str, torch.Tensor]:
    elevation = torch.zeros(1, 1, size, size)
    valid = torch.ones_like(elevation)
    return {
        "dsm_elevation": elevation,
        "physics_elevation": elevation,
        "z_ground_proxy": elevation,
        "local_relief": torch.ones_like(elevation),
        "dem_valid": valid,
    }


def test_v15_3_bounded_graph_message_is_finite_and_identity_on_constant_features() -> None:
    graph = HydroEdgeKANV15_2(
        8,
        heads=2,
        graph_feature_stride=4,
        edge_stats_path=None,
        message_mode="rational",
        message_scale=0.75,
        extreme_preservation_enabled=True,
        extreme_preservation_scale=0.75,
    )
    features = torch.ones(1, 8, 8, 8, requires_grad=True)
    output, diagnostics = graph(
        features,
        _physical(),
        torch.ones(1, 1, 32, 32),
        feature_stride=4,
    )
    torch.testing.assert_close(output, features, rtol=0.0, atol=1.0e-6)
    assert diagnostics["message_rms"].isfinite()
    assert diagnostics["extreme_preservation_gate_mean"].isfinite()
    output.mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_v15_3_neutral_gate_initialization_is_registered_and_s1_only() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_3_stability_graph.xml"
    )
    model = build_model(config)
    assert model.__class__.__name__ == "PAHydroKANS1V15_3"
    assert model.neutral_gate_initialization
    assert model.graph.message_mode == "rational"
    assert model.graph.extreme_preservation_enabled
    for module in (
        model.sar_encoder.pre_gate[0],
        model.sar_encoder.change_amplitude[0],
        model.fusion.terrain_gate[0],
        model.decoder.gates[0],
    ):
        assert module.weight.abs().sum().item() == pytest.approx(0.0)
        assert module.bias is None or module.bias.abs().sum().item() == pytest.approx(0.0)
    assert config["dataset"]["input_mode"] == "s1_terrain"
    assert not any(key.startswith("s2_") for key in config["dataset"])

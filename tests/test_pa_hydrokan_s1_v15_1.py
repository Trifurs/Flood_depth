from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from models.hydro_edge_kan_v15_1 import HydroEdgeKANV15_1
from utils.config import load_config
from utils.registry import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_PATH = PROJECT_ROOT / "artifacts/optimization/hydrokan_s1_v15_1/graph_edge_train_stats.json"


def _physical(size: int = 64) -> dict[str, torch.Tensor]:
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    elevation = (0.05 * x + 0.02 * y).float().view(1, 1, size, size)
    valid = torch.ones_like(elevation)
    return {
        "dsm_elevation": elevation,
        "physics_elevation": elevation,
        "z_ground_proxy": elevation,
        "local_relief": torch.full_like(elevation, 0.1),
        "dem_valid": valid,
    }


def _model_inputs(qa_channels: int = 2, size: int = 64) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    physical = _physical(size)
    return {
        "s1_t1": torch.randn(1, 2, size, size),
        "s1_t2": torch.randn(1, 2, size, size),
        "s1_change": torch.randn(1, 3, size, size),
        "s1_qa": torch.randn(1, qa_channels, size, size),
        "terrain": torch.randn(1, 2, size, size),
        "terrain_raw": torch.cat((physical["dsm_elevation"], torch.zeros_like(physical["dsm_elevation"])), dim=1),
        "reliability": torch.randn(1, 6, size, size),
        "s1_valid": torch.ones(1, 1, size, size),
        "s1_event_support": torch.ones(1, 1, size, size),
        "dem_valid": physical["dem_valid"],
        "s1_conditioning": torch.zeros(1, 2, size, size),
        "branch_validity": {
            "s1_t1": torch.ones(1, 1, size, size),
            "s1_t2": torch.ones(1, 1, size, size),
            "s1_change": torch.ones(1, 1, size, size),
            "terrain": torch.ones(1, 1, size, size),
        },
    }


def test_v15_1_configs_registry_build_and_consume_real_graph_stride() -> None:
    configs = {
        "corrected": "configs/pa_hydrokan/subset1000_s1_v15_1_corrected.xml",
        "kan": "configs/pa_hydrokan/subset1000_s1_v15_1_kan.xml",
        "simple": "configs/pa_hydrokan/subset1000_s1_v15_1_simple.xml",
        "final": "configs/pa_hydrokan/subset1000_s1_v15_1_final.xml",
    }
    for expected_variant, path in configs.items():
        model = build_model(load_config(PROJECT_ROOT / path))
        assert model.variant == (
            "simple" if expected_variant == "final" else expected_variant
        )
    simple_config = load_config(PROJECT_ROOT / configs["simple"])
    simple = build_model(simple_config)
    assert simple.graph_identity()["graph_node_spacing_m"] == 80.0
    assert simple.graph.graph_feature_stride == 4
    assert simple.graph.edge_stats_sha256
    assert isinstance(simple.graph.raw_symmetric_prior, type(None))
    torch.testing.assert_close(
        simple.fusion.terrain_alpha,
        torch.full_like(simple.fusion.terrain_alpha, 0.05),
        atol=1.0e-6,
        rtol=0.0,
    )
    final_depth_bias = simple.heads.depth_head.trunk[-1].bias
    torch.testing.assert_close(F.softplus(final_depth_bias), torch.full_like(final_depth_bias, 0.1), atol=1.0e-6, rtol=0.0)

    disabled = deepcopy(simple_config)
    disabled["model"]["context_enabled"] = False
    assert isinstance(build_model(disabled).context, nn.Identity)
    invalid = deepcopy(simple_config)
    invalid["model"]["graph_feature_stride"] = 8
    with pytest.raises(ValueError, match="actual 1/4"):
        build_model(invalid)


def test_v15_1_simple_cpu_forward_backward_is_s1_only() -> None:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_1_simple.xml")
    model = build_model(config)
    inputs = _model_inputs(qa_channels=2)
    outputs = model(inputs)
    assert outputs["depth"].shape == (1, 1, 64, 64)
    assert torch.isfinite(outputs["depth"]).all()
    assert "support_probability" not in outputs
    assert not any(str(key).startswith("s2_") for key in inputs)
    outputs["depth"].mean().backward()
    assert model.graph.edge_kan.spline_coefficients.grad is not None
    assert torch.isfinite(model.graph.edge_kan.spline_coefficients.grad).all()


def test_v15_2_graph_bottleneck_reuses_single_graph_with_small_residual() -> None:
    """Graph-B must add one small propagated residual, not a second graph."""

    config = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_graph_bottleneck.xml"
    )
    model = build_model(config)
    assert model.variant == "simple"
    assert model.graph_bottleneck_enabled
    assert model.graph_bottleneck_projection is not None
    assert model.graph_bottleneck_scale.item() == pytest.approx(0.03, abs=1.0e-6)
    assert model.graph_bottleneck_scale_max == pytest.approx(0.10)
    assert sum(name == "graph" for name, _ in model.named_children()) == 1

    outputs = model(_model_inputs(qa_channels=0))
    assert outputs["depth"].shape == (1, 1, 64, 64)
    diagnostics = outputs["graph_diagnostics"]
    assert torch.isfinite(diagnostics["graph_bottleneck_scale"])
    ratio = diagnostics["graph_bottleneck_residual_input_rms_ratio"]
    assert torch.isfinite(ratio)
    # Initial propagation is deliberately much smaller than the bottleneck
    # feature so this ablation starts as a controlled residual perturbation.
    assert ratio.item() < 0.10
    outputs["depth"].mean().backward()
    assert model.graph_bottleneck_projection.weight.grad is not None
    assert torch.isfinite(model.graph_bottleneck_projection.weight.grad).all()


def test_absolute_event_sar_shortcut_is_small_trainable_and_s1_only() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_simple_fixed.xml"
    )
    config["model"].update(
        {
            "absolute_sar_shortcut_enabled": True,
            "absolute_sar_shortcut_init": 0.03,
            "absolute_sar_shortcut_max": 0.10,
        }
    )
    model = build_model(config)
    encoder = model.sar_encoder
    assert encoder.absolute_sar_stem is not None
    torch.testing.assert_close(
        encoder.absolute_sar_shortcut_scale,
        torch.full_like(encoder.absolute_sar_shortcut_scale, 0.03),
        atol=1.0e-6,
        rtol=0.0,
    )
    inputs = _model_inputs(qa_channels=0)
    outputs = model(inputs)
    diagnostics = outputs["sar_diagnostics"]
    assert torch.isfinite(diagnostics["absolute_sar_shortcut_residual_input_rms_ratio"])
    assert diagnostics["absolute_sar_shortcut_residual_input_rms_ratio"] < 0.20
    outputs["depth"].mean().backward()
    assert encoder.absolute_sar_stem.weight.grad is not None
    assert torch.isfinite(encoder.absolute_sar_stem.weight.grad).all()
    assert not any(str(key).startswith("s2_") for key in inputs)


def test_absolute_sar_shortcut_does_not_shift_shared_initialization() -> None:
    """The one-variable ablation must retain bitwise-identical common weights."""

    disabled_config = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_combined.xml"
    )
    enabled_config = deepcopy(disabled_config)
    enabled_config["model"]["absolute_sar_shortcut_enabled"] = True

    torch.manual_seed(314159)
    disabled = build_model(disabled_config)
    disabled_rng_after_build = torch.random.get_rng_state()
    torch.manual_seed(314159)
    enabled = build_model(enabled_config)
    enabled_rng_after_build = torch.random.get_rng_state()

    disabled_state = disabled.state_dict()
    enabled_state = enabled.state_dict()
    common = [
        name
        for name, value in disabled_state.items()
        if name in enabled_state and enabled_state[name].shape == value.shape
    ]
    assert common
    assert torch.equal(disabled_rng_after_build, enabled_rng_after_build)
    assert all(torch.equal(disabled_state[name], enabled_state[name]) for name in common)


def test_v15_1_graph_stats_multihead_gradient_and_identity_contract() -> None:
    graph = HydroEdgeKANV15_1(
        8,
        heads=2,
        graph_feature_stride=4,
        edge_stats_path=STATS_PATH,
        diagnostics_enabled=False,
    )
    features = torch.randn(1, 8, 16, 16, requires_grad=True)
    physical = _physical(64)
    sensor = torch.ones(1, 1, 64, 64)
    output, diagnostics = graph(features, physical, sensor, feature_stride=4)
    assert output.shape == features.shape
    assert graph.graph_identity()["orthogonal_neighbour_distance_m"] == 80.0
    assert graph.graph_identity()["diagonal_neighbour_distance_m"] == pytest.approx(80.0 * 2**0.5)
    assert "edge_descriptors" not in diagnostics
    assert torch.isfinite(diagnostics["graph_update_rms_ratio"])
    assert torch.isfinite(diagnostics["spline_base_rms_ratio"])
    (output.square().mean()).backward()
    gradient = graph.edge_kan.spline_coefficients.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0

    invalid, _ = graph(features.detach(), physical, torch.zeros_like(sensor), feature_stride=4)
    torch.testing.assert_close(invalid, features.detach())
    constant = torch.ones_like(features.detach())
    constant_output, _ = graph(constant, physical, sensor, feature_stride=4)
    torch.testing.assert_close(constant_output, constant, atol=1.0e-6, rtol=0.0)
    with pytest.raises(ValueError, match="feature-stride mismatch"):
        graph(features.detach(), physical, sensor, feature_stride=8)


def test_v15_1_dataset_returns_only_real_configured_qa_and_no_s2() -> None:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_1_simple.xml")
    contract = DatasetContract.load(config["dataset"]["contract"])
    input_spec = ModelInputSpec.from_config(config)
    sample = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        "val",
        input_spec=input_spec,
        band_spec=resolve_band_spec(config, contract),
        minimum_event_band_fraction=config["dataset"]["minimum_event_band_fraction"],
        s1_qa_names=config["dataset"]["model_s1_qa_names"],
    )[0]
    assert sample["s1_qa"].shape[0] == 2
    assert sample["metadata"]["io_profile"]["read_band_counts"]["s1_qa"] == 2
    assert not any(key.startswith("s2_") for key in sample)
    assert not any(key.startswith("s2_") for key in sample["validity"])

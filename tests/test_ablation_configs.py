from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest
import torch

from utils.config import load_config
from utils.experiment_catalog import deep_learning_config_paths
from utils.graph_metadata import resolved_graph_identity
from utils.registry import build_model


CONFIGS = {
    "wo_rcp": "pa_hydrokan_wo_rcp.xml",
    "wo_tcf": "pa_hydrokan_wo_tcf.xml",
    "wo_tae_kan": "pa_hydrokan_wo_tae_kan.xml",
    "wo_rcp_tcf": "pa_hydrokan_wo_rcp_tcf.xml",
    "wo_rcp_tae_kan": "pa_hydrokan_wo_rcp_tae_kan.xml",
    "wo_tcf_tae_kan": "pa_hydrokan_wo_tcf_tae_kan.xml",
    "wo_rcp_tcf_tae_kan": "pa_hydrokan_wo_rcp_tcf_tae_kan.xml",
}

FLAG_BY_MODULE = {
    "RCP": "reliability_conditioning_enabled",
    "TCF": "terrain_conditioned_fusion_enabled",
    "TAE-KAN": "topographic_affinity_enabled",
}


def _config(variant_id: str) -> dict:
    return load_config(Path("configs/ablation") / CONFIGS[variant_id])


@pytest.mark.parametrize("variant_id", tuple(CONFIGS))
def test_ablation_config_builds_the_declared_factorial_intervention(
    variant_id: str,
) -> None:
    config = _config(variant_id)
    removed = set(config["ablation"]["removed_modules"])
    assert config["ablation"]["variant_id"] == variant_id
    model = build_model(config)
    flags = model.component_flags()
    for module, flag in FLAG_BY_MODULE.items():
        assert flags[flag] is (module not in removed)
    # LCA is not an independent factor: its effective state follows TAE-KAN.
    assert flags["latent_compatibility_enabled"] is ("TAE-KAN" not in removed)
    assert (resolved_graph_identity(config) is None) is ("TAE-KAN" in removed)


def test_ablation_design_contains_all_nonempty_combinations_exactly_once() -> None:
    observed = {
        frozenset(_config(variant)["ablation"]["removed_modules"])
        for variant in CONFIGS
    }
    modules = ("RCP", "TCF", "TAE-KAN")
    expected = {
        frozenset(selected)
        for size in range(1, len(modules) + 1)
        for selected in combinations(modules, size)
    }
    assert observed == expected


def test_disabled_top_level_paths_are_frozen() -> None:
    no_rcp = build_model(_config("wo_rcp"))
    assert not any(
        parameter.requires_grad for parameter in no_rcp.reliability_conditioner.parameters()
    )
    no_tcf = build_model(_config("wo_tcf"))
    assert not no_tcf.fusion.raw_terrain_mix.requires_grad
    no_tae = build_model(_config("wo_tae_kan"))
    assert not any(parameter.requires_grad for parameter in no_tae.graph.parameters())


def test_full_model_is_catalogued_once_and_has_no_redundant_ablation_config() -> None:
    paths = deep_learning_config_paths(include_ablations=True)
    assert sum(path.name == "pa_hydrokan.xml" for path in paths) == 1
    assert len(tuple(Path("configs/ablation").glob("*.xml"))) == 7


def _global_calibration_evidence(model, reliability: torch.Tensor):
    height = width = 8
    batch = reliability.shape[0]
    dtype = reliability.dtype
    state_channels = model.band_spec.channels("s1_t1")
    change_channels = model.band_spec.channels("s1_change")
    terrain_channels = model.band_spec.channels("terrain")
    conditioning_channels = model.band_spec.channels("s1_conditioning")
    one = torch.ones(batch, 1, height, width, dtype=dtype)
    inputs = {
        "s1_t1": torch.randn(batch, state_channels, height, width, dtype=dtype),
        "s1_t2": torch.randn(batch, state_channels, height, width, dtype=dtype),
        "s1_change": torch.randn(batch, change_channels, height, width, dtype=dtype),
        "terrain": torch.randn(batch, terrain_channels, height, width, dtype=dtype),
        "reliability": reliability,
        "s1_valid": one,
        "s1_event_support": one,
        "dem_valid": one,
    }
    physical = {
        "local_relief": torch.rand_like(one),
        "z_relative": torch.randn_like(one),
        "obstacle_residual": torch.randn_like(one),
        "dz_dx": torch.randn_like(one),
        "dz_dy": torch.randn_like(one),
        "slope": torch.rand_like(one),
        "topographic_context_positions": [
            torch.randn_like(one)
            for _ in model.terrain.topographic_context_scales_m
        ],
    }
    conditioning = (
        torch.randn(batch, conditioning_channels, height, width, dtype=dtype)
        if conditioning_channels
        else None
    )
    decoded = torch.randn(batch, 32, height, width, dtype=dtype)
    return model._global_calibration_evidence_parts(
        inputs,
        decoded,
        physical,
        {"s1_t1": one, "s1_t2": one, "s1_change": one},
        conditioning,
    )


def test_rcp_ablation_blocks_reliability_from_global_depth_calibration() -> None:
    no_rcp = build_model(_config("wo_rcp")).eval()
    channels = len(no_rcp.reliability_spec.names)
    # Reset the global RNG so every non-reliability tensor is identical.
    torch.manual_seed(19)
    first = _global_calibration_evidence(
        no_rcp, torch.randn(1, channels, 8, 8)
    )
    torch.manual_seed(19)
    second = _global_calibration_evidence(
        no_rcp, torch.randn(1, channels, 8, 8) + 100.0
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second))

    full = build_model(load_config(Path("configs/pa_hydrokan.xml"))).eval()
    torch.manual_seed(23)
    full_first = _global_calibration_evidence(
        full, torch.randn(1, channels, 8, 8)
    )
    torch.manual_seed(23)
    full_second = _global_calibration_evidence(
        full, torch.randn(1, channels, 8, 8) + 100.0
    )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(full_first, full_second)
    )

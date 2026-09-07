from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import load_config
from utils.graph_metadata import resolved_graph_identity
from utils.registry import build_model


CONFIGS = {
    "full": "pa_hydrokan_full.xml",
    "no_rcp": "pa_hydrokan_no_reliability_conditioning.xml",
    "no_tcf": "pa_hydrokan_no_terrain_conditioned_fusion.xml",
    "no_tae_kan": "pa_hydrokan_no_topographic_affinity_edge_kan.xml",
    "no_lca": "pa_hydrokan_no_latent_compatibility.xml",
}


EXPECTED_FLAGS = {
    "full": {
        "reliability_conditioning_enabled": True,
        "terrain_conditioned_fusion_enabled": True,
        "topographic_affinity_enabled": True,
        "latent_compatibility_enabled": True,
    },
    "no_rcp": {
        "reliability_conditioning_enabled": False,
        "terrain_conditioned_fusion_enabled": True,
        "topographic_affinity_enabled": True,
        "latent_compatibility_enabled": True,
    },
    "no_tcf": {
        "reliability_conditioning_enabled": True,
        "terrain_conditioned_fusion_enabled": False,
        "topographic_affinity_enabled": True,
        "latent_compatibility_enabled": True,
    },
    "no_tae_kan": {
        "reliability_conditioning_enabled": True,
        "terrain_conditioned_fusion_enabled": True,
        "topographic_affinity_enabled": False,
        "latent_compatibility_enabled": True,
    },
    "no_lca": {
        "reliability_conditioning_enabled": True,
        "terrain_conditioned_fusion_enabled": True,
        "topographic_affinity_enabled": True,
        "latent_compatibility_enabled": False,
    },
}


def _config(variant_id: str) -> dict:
    return load_config(Path("configs/ablation") / CONFIGS[variant_id])


def _trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@pytest.mark.parametrize("variant_id", tuple(CONFIGS))
def test_ablation_config_builds_the_declared_component_intervention(
    variant_id: str,
) -> None:
    config = _config(variant_id)
    assert config["ablation"]["variant_id"] == variant_id
    model = build_model(config)
    assert model.component_flags() == EXPECTED_FLAGS[variant_id]
    if variant_id == "no_tae_kan":
        assert resolved_graph_identity(config) is None
    else:
        assert resolved_graph_identity(config) is not None


def test_ablation_removes_trainable_parameters_only_from_disabled_paths() -> None:
    full = build_model(_config("full"))
    full_trainable = _trainable_parameters(full)

    no_rcp = build_model(_config("no_rcp"))
    assert not any(parameter.requires_grad for parameter in no_rcp.reliability_conditioner.parameters())
    assert _trainable_parameters(no_rcp) < full_trainable

    no_tcf = build_model(_config("no_tcf"))
    assert not no_tcf.fusion.raw_terrain_mix.requires_grad
    assert _trainable_parameters(no_tcf) < full_trainable

    no_tae_kan = build_model(_config("no_tae_kan"))
    assert not any(parameter.requires_grad for parameter in no_tae_kan.graph.parameters())
    assert _trainable_parameters(no_tae_kan) < full_trainable

    no_lca = build_model(_config("no_lca"))
    assert not any(
        parameter.requires_grad for parameter in no_lca.graph.latent_compatibility.parameters()
    )
    assert _trainable_parameters(no_lca) < full_trainable

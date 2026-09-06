"""PA-HydroKAN-S1-V15.3 stability-first S1 model.

V15.3 keeps the audited S1-only input contract and decoder/output semantics of
V15.1, but makes the deliberately small changes needed for a controlled
stability experiment:

* spatial gates start as constant, neutral gates rather than random masks;
* the 1/4 Graph/KAN uses the V15.2 feature-wise terrain map;
* graph messages can use a bounded rational/tanh response and an
  extreme-preservation confidence gate.

This is still a weakly physics-guided event-scale reconstruction model, not a
shallow-water solver.  No Sentinel-2 tensor is accepted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from datasets.band_selection import BandSpec, resolve_band_spec
from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec
from models.hydro_edge_kan_v15_1 import DEFAULT_EDGE_FEATURE_NAMES
from models.hydro_edge_kan_v15_2 import HydroEdgeKANV15_2
from models.pa_hydrokan_s1_v15_1 import PAHydroKANS1V15_1


def _zero_conv_logits(modules: Any) -> None:
    """Make a collection of sigmoid-logit convolutions spatially neutral."""

    for module in modules:
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"neutral gate initialization expected Conv2d, got {type(module)!r}")
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _apply_neutral_gate_initialization(model: PAHydroKANS1V15_1) -> dict[str, Any]:
    """Initialize only gates whose output is multiplied by a residual branch.

    The gate remains learnable.  Zero logits give a constant sigmoid(0)=0.5,
    so initialization does not inject a random spatial preference while the
    existing alpha/gamma parameters still control residual strength.
    """

    changed: list[str] = []
    sar_encoder = getattr(model, "sar_encoder", None)
    if sar_encoder is not None and hasattr(sar_encoder, "pre_gate"):
        _zero_conv_logits(sar_encoder.pre_gate)
        _zero_conv_logits(sar_encoder.change_amplitude)
        changed.extend(("sar_pre_context", "sar_change_amplitude"))
    fusion = getattr(model, "fusion", None)
    if fusion is not None and hasattr(fusion, "terrain_gate"):
        _zero_conv_logits(fusion.terrain_gate)
        changed.append("terrain_residual")
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "gates"):
        _zero_conv_logits(decoder.gates)
        changed.append("decoder_skip")
    return {
        "enabled": True,
        "sigmoid_logit_initialization": 0.0,
        "effective_sigmoid_gate_initialization": 0.5,
        "modules": changed,
    }


class PAHydroKANS1V15_3(PAHydroKANS1V15_1):
    """S1-only V15.3 model with bounded Graph/KAN message propagation."""

    _VARIANTS = {"kan", "simple"}

    def __init__(
        self,
        model_config: Mapping[str, Any],
        band_spec: BandSpec,
        raw_terrain_names: tuple[str, ...],
        input_spec: ModelInputSpec,
    ) -> None:
        config = dict(model_config)
        variant = str(config.get("variant", "simple"))
        if variant not in self._VARIANTS:
            raise ValueError(
                "PA-HydroKAN-S1-V15.3 requires variant 'kan' or 'simple', "
                f"got {variant!r}"
            )
        # V15.3 is a 1/4 graph experiment.  The parent builds the tested
        # encoder/decoder and real 1/4 feature route; the graph map below is
        # then replaced with the V15.2 audited feature-wise map.
        config["variant"] = variant
        config["graph_feature_stride"] = 4
        super().__init__(config, band_spec, raw_terrain_names, input_spec)
        self.variant = variant
        self.v15_3_stability = {
            "neutral_gate_initialization": bool(
                config.get("neutral_gate_initialization", True)
            ),
            "event_main_path": str(config.get("event_main_path", "event_state_first")),
            "bounded_graph_message": str(
                config.get("graph_message_mode", "rational")
            ),
            "extreme_preservation_gate": bool(
                config.get("extreme_preservation_enabled", True)
            ),
        }
        self.graph = HydroEdgeKANV15_2(
            int(config.get("graph_channels", 64)),
            heads=int(config.get("graph_heads", 2)),
            grid_size=int(config.get("kan_grid_size", 4)),
            spline_order=int(config.get("kan_spline_order", 3)),
            graph_feature_stride=4,
            terrain_pixel_size_m=float(config.get("terrain_pixel_size_m", 20.0)),
            edge_feature_names=tuple(
                config.get("graph_edge_feature_names", DEFAULT_EDGE_FEATURE_NAMES)
            ),
            edge_stats_path=config.get("graph_edge_stats"),
            mapping_temperature=float(config.get("graph_mapping_temperature", 1.25)),
            mapping_temperatures=config.get("graph_mapping_temperatures"),
            edge_minimum_dem_fraction=float(
                config.get("edge_minimum_dem_fraction", 0.5)
            ),
            edge_minimum_sar_fraction=float(
                config.get("edge_minimum_sar_fraction", 0.5)
            ),
            edge_minimum_barrier_valid_fraction=float(
                config.get("edge_minimum_barrier_valid_fraction", 0.5)
            ),
            gamma_init=float(config.get("kan_gamma_init_effective", 0.03)),
            gamma_max=float(config.get("kan_gamma_max", 0.25)),
            base_scale_init=float(config.get("kan_base_scale_init", 0.35)),
            spline_scale_init=float(config.get("kan_spline_scale_init", 1.0)),
            latent_scale_init=float(config.get("kan_latent_scale_init", 0.25)),
            symmetric_static_prior_enabled=bool(
                config.get("symmetric_static_prior_enabled", False)
            ),
            diagnostics_enabled=bool(
                config.get("diagnostics_enabled", config.get("diagnostic_mode", False))
            ),
            regularization_enabled=bool(config.get("kan_regularization_enabled", True)),
            message_mode=str(config.get("graph_message_mode", "rational")),
            message_scale=float(config.get("graph_message_scale", 0.75)),
            extreme_preservation_enabled=bool(
                config.get("extreme_preservation_enabled", True)
            ),
            extreme_preservation_scale=float(
                config.get("extreme_preservation_scale", 0.75)
            ),
        )
        self.neutral_gate_initialization = bool(
            config.get("neutral_gate_initialization", True)
        )
        self.neutral_gate_initialization_audit = (
            _apply_neutral_gate_initialization(self)
            if self.neutral_gate_initialization
            else {"enabled": False, "modules": []}
        )

    def graph_identity(self) -> dict[str, Any]:
        identity = self.graph.graph_identity(self._last_graph_feature_shape)
        identity["model_version"] = "v15_3_stability_first"
        identity["neutral_gate_initialization"] = self.neutral_gate_initialization
        identity["event_main_path"] = self.v15_3_stability["event_main_path"]
        return identity


def build_pa_hydrokan_s1_v15_3(config: Mapping[str, Any]) -> PAHydroKANS1V15_3:
    """Build the strict Sentinel-1-plus-terrain V15.3 model."""

    if "model" not in config:
        raise ValueError("PA-HydroKAN-S1-V15.3 builder requires the full resolved config")
    input_spec = ModelInputSpec.from_config(config)
    if not input_spec.is_s1_only:
        raise ValueError("PA-HydroKAN-S1-V15.3 requires dataset.input_mode='s1_terrain'")
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    raw_names = tuple(str(value) for value in contract.group("terrain")["band_descriptions"])
    model_config = dict(config["model"])
    if "kan_regularization_enabled" not in model_config:
        model_config["kan_regularization_enabled"] = float(
            config.get("loss", {}).get("lambda_kan", 0.0)
        ) != 0.0
    return PAHydroKANS1V15_3(model_config, band_spec, raw_names, input_spec)

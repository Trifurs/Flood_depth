"""Configurable PA-HydroKAN-v13 while preserving the v12 state dictionary."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from datasets.band_selection import BandSpec, resolve_band_spec
from datasets.contract import DatasetContract
from datasets.preprocessing import RELIABILITY_NAMES
from models.decoder_v13 import GatedFPNDecoder, SeparateFloodDepthHeads
from models.efficient_blocks import DilatedContext, GatedCrossStateEncoder
from models.fusion_v13 import ContentAwareFusionPyramid
from models.pa_hydrokan import FORBIDDEN_INPUTS, REQUIRED_INPUTS
from models.terrain_features_v13 import TerrainFeaturePyramidV13
from models.terrain_graph_kan_v13 import MultiHeadTerrainGraphKAN


class PAHydroKANV13(nn.Module):
    def __init__(self, model_config: Mapping[str, Any], band_spec: BandSpec) -> None:
        super().__init__()
        channels = [int(value) for value in model_config["channels"]]
        if len(channels) != 4:
            raise ValueError("PA-HydroKAN-v13 requires four scales")
        graph_scale = int(model_config["graph_scale"])
        if graph_scale != 2 ** (len(channels) - 1):
            raise ValueError(f"graph_scale={graph_scale} does not match {len(channels)} scales")
        dropout, groups = float(model_config["dropout"]), int(model_config["group_norm_groups"])
        block_kind = str(model_config.get("residual_block", "efficient"))
        conditioning_channels = band_spec.channels("s1_conditioning")
        self.band_spec = band_spec
        self.s1_encoder = GatedCrossStateEncoder(
            band_spec.channels("s1_t1"), band_spec.channels("s1_change"), channels,
            dropout, groups, block_kind, conditioning_channels,
        )
        self.s2_encoder = GatedCrossStateEncoder(
            band_spec.channels("s2_t1"), band_spec.channels("s2_change"), channels,
            dropout, groups, block_kind, 0,
        )
        self.terrain = TerrainFeaturePyramidV13(
            band_spec.channels("terrain"), channels, dropout, groups,
            float(model_config["terrain_pixel_size_m"]), block_kind,
        )
        self.fusion = ContentAwareFusionPyramid(
            channels, len(RELIABILITY_NAMES), dropout, groups, block_kind
        )
        self.context = DilatedContext(channels[-1], groups) if bool(model_config.get("context_enabled", True)) else nn.Identity()
        self.graph = MultiHeadTerrainGraphKAN(
            channels[-1], int(model_config["graph_heads"]), int(model_config["kan_grid_size"]),
            int(model_config["kan_spline_order"]), int(model_config["kan_edge_features"]),
            str(model_config.get("kan_input_normalization", "explicit_fixed_scaling")),
            str(model_config.get("graph_message_normalization", "gate_sum")), groups,
        )
        self.graph_enabled = bool(model_config.get("graph_enabled", True))
        self.decoder = GatedFPNDecoder(
            channels, dropout, groups, block_kind,
            bool(model_config.get("deep_supervision_enabled", True)),
        )
        self.heads = SeparateFloodDepthHeads(
            channels[0], groups, float(model_config["uncertainty_epsilon"]),
            float(model_config["uncertainty_maximum"]),
            str(model_config.get("depth_output_semantics", "conditional_positive_v2")),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        forbidden = FORBIDDEN_INPUTS.intersection(inputs)
        if forbidden:
            raise ValueError(f"Forbidden inputs: {sorted(forbidden)}")
        missing = REQUIRED_INPUTS.difference(inputs)
        if missing:
            raise KeyError(f"Missing PA-HydroKAN-v13 inputs: {sorted(missing)}")
        conditioning = inputs.get("s1_conditioning")
        branch_validity = inputs.get("branch_validity", {})
        if self.band_spec.channels("s1_conditioning") and conditioning is None:
            raise KeyError("Missing configured s1_conditioning")
        s1 = self.s1_encoder(
            inputs["s1_t1"], inputs["s1_t2"], inputs["s1_change"],
            inputs["s1_valid"], conditioning,
            {key.removeprefix("s1_"): value for key, value in branch_validity.items() if key.startswith("s1_")},
        )
        s2 = self.s2_encoder(
            inputs["s2_t1"], inputs["s2_t2"], inputs["s2_change"],
            inputs["s2_valid"], None,
            {key.removeprefix("s2_"): value for key, value in branch_validity.items() if key.startswith("s2_")},
        )
        terrain, physical = self.terrain(
            inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"]
        )
        fused, weights, terrain_gates, fusion_entropy = self.fusion(
            s1, s2, terrain, inputs["reliability"], inputs["s1_valid"],
            inputs["s2_valid"], physical["dem_valid_fractions"],
        )
        bottleneck = self.context(fused[-1])
        day_index = RELIABILITY_NAMES.index("absolute_normalized_sensor_day_difference")
        day_difference = inputs["reliability"][:, day_index : day_index + 1]
        sensor_valid = torch.maximum(inputs["s1_valid"], inputs["s2_valid"])
        if self.graph_enabled:
            bottleneck, diagnostics = self.graph(
                bottleneck, physical, day_difference, weights[-1], sensor_valid
            )
        else:
            zero = bottleneck.sum() * 0.0
            diagnostics = {"gate_mean": zero, "gamma_mean": zero,
                           "kan_coefficient_magnitude": zero,
                           "kan_coefficient_smoothness": zero}
        decoded, auxiliaries = self.decoder(
            bottleneck, fused, terrain, physical["dem_valid_fractions"]
        )
        outputs: dict[str, Any] = self.heads(decoded)
        outputs.update({
            "auxiliary_depths": auxiliaries, "modality_weights": weights,
            "terrain_gates": terrain_gates, "graph_diagnostics": diagnostics,
            "fusion_entropy": fusion_entropy,
            "physical_features": physical,
        })
        return outputs


def build_pa_hydrokan_v13(config: Mapping[str, Any]) -> PAHydroKANV13:
    if "model" not in config:
        raise ValueError("v13 builder requires the full resolved config")
    contract = DatasetContract.load(config["dataset"]["contract"])
    return PAHydroKANV13(config["model"], resolve_band_spec(config, contract))

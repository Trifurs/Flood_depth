"""PA-HydroKAN-v13.1: lighter fusion, decoder, and vectorized terrain graph."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from datasets.band_selection import BandSpec, resolve_band_spec
from datasets.contract import DatasetContract
from datasets.preprocessing import RELIABILITY_NAMES
from models.decoder_v13 import GatedFPNDecoderV131, SeparateFloodDepthHeads
from models.efficient_blocks import DilatedContext, GatedCrossStateEncoder
from models.fusion_v13 import ContentAwareFusionPyramidV131
from models.pa_hydrokan import FORBIDDEN_INPUTS, REQUIRED_INPUTS
from models.terrain_features_v13 import TerrainFeaturePyramidV13
from models.terrain_graph_kan_v13 import VectorizedTerrainGraphKANV131


class PAHydroKANV131(nn.Module):
    def __init__(self, model_config: Mapping[str, Any], band_spec: BandSpec) -> None:
        super().__init__()
        channels = [int(value) for value in model_config.get("channels", [32, 64, 128, 192])]
        if len(channels) != 4:
            raise ValueError("PA-HydroKAN-v13.1 requires four encoder scales")
        graph_scale = int(model_config.get("graph_scale", 8))
        if graph_scale != 2 ** (len(channels) - 1):
            raise ValueError("graph_scale must match the encoder scale pyramid")
        dropout = float(model_config.get("dropout", 0.10))
        groups = int(model_config.get("group_norm_groups", 8))
        block_kind = str(model_config.get("residual_block", "efficient"))
        self.band_spec = band_spec
        conditioning_channels = band_spec.channels("s1_conditioning")
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
            float(model_config.get("terrain_pixel_size_m", 20.0)), block_kind,
        )
        self.fusion = ContentAwareFusionPyramidV131(
            channels, len(RELIABILITY_NAMES), dropout, groups, block_kind
        )
        self.context = (
            DilatedContext(channels[-1], groups)
            if bool(model_config.get("context_enabled", True)) else nn.Identity()
        )
        edge_names = tuple(model_config.get("graph_edge_features", ()))
        if not edge_names:
            from models.terrain_graph_kan_v13 import V131_EDGE_FEATURE_NAMES
            edge_names = V131_EDGE_FEATURE_NAMES
        self.graph = VectorizedTerrainGraphKANV131(
            channels[-1],
            heads=int(model_config.get("graph_heads", 2)),
            grid_size=int(model_config.get("kan_grid_size", 8)),
            spline_order=int(model_config.get("kan_spline_order", 3)),
            normalization=str(model_config.get("kan_input_normalization", "explicit_fixed_scaling")),
            message_normalization=str(model_config.get("graph_message_normalization", "gate_sum")),
            graph_scale=graph_scale,
            terrain_pixel_size_m=float(model_config.get("terrain_pixel_size_m", 20.0)),
            edge_feature_names=edge_names,
            feature_scales=model_config.get("graph_feature_scales"),
            groups=groups,
        )
        self.graph_enabled = bool(model_config.get("graph_enabled", True))
        decoder_widths = [int(value) for value in model_config.get("decoder_widths", [64, 48, 32])]
        self.decoder = GatedFPNDecoderV131(
            channels, dropout, groups, block_kind,
            bool(model_config.get("deep_supervision_enabled", True)),
            decoder_widths,
            band_spec.channels("s1_change"),
        )
        self.heads = SeparateFloodDepthHeads(
            decoder_widths[-1], groups,
            float(model_config.get("uncertainty_epsilon", 0.001)),
            float(model_config.get("uncertainty_maximum", 5.0)),
            str(model_config.get("depth_output_semantics", "conditional_positive_v2")),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        forbidden = FORBIDDEN_INPUTS.intersection(inputs)
        if forbidden:
            raise ValueError(f"Forbidden inputs: {sorted(forbidden)}")
        missing = REQUIRED_INPUTS.difference(inputs)
        if missing:
            raise KeyError(f"Missing PA-HydroKAN-v13.1 inputs: {sorted(missing)}")
        branch_validity = inputs.get("branch_validity", {})
        conditioning = inputs.get("s1_conditioning")
        if self.band_spec.channels("s1_conditioning") and conditioning is None:
            raise KeyError("Missing configured s1_conditioning")
        s1 = self.s1_encoder(
            inputs["s1_t1"], inputs["s1_t2"], inputs["s1_change"], inputs["s1_valid"],
            conditioning,
            {key.removeprefix("s1_"): value for key, value in branch_validity.items() if key.startswith("s1_")},
        )
        s2 = self.s2_encoder(
            inputs["s2_t1"], inputs["s2_t2"], inputs["s2_change"], inputs["s2_valid"],
            None,
            {key.removeprefix("s2_"): value for key, value in branch_validity.items() if key.startswith("s2_")},
        )
        terrain, physical = self.terrain(
            inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"]
        )
        fused, weights, terrain_gates, fusion_entropy = self.fusion(
            s1, s2, terrain, inputs["reliability"], inputs["s1_valid"],
            inputs["s2_valid"], physical["dem_valid_fractions"], branch_validity,
        )
        bottleneck = self.context(fused[-1])
        day_index = RELIABILITY_NAMES.index("absolute_normalized_sensor_day_difference")
        day_difference = inputs["reliability"][:, day_index : day_index + 1]
        sensor_valid = torch.maximum(inputs["s1_valid"], inputs["s2_valid"])
        if self.graph_enabled:
            bottleneck, graph_diagnostics = self.graph(
                bottleneck, physical, day_difference, weights[-1], sensor_valid
            )
        else:
            zero = bottleneck.sum() * 0.0
            graph_diagnostics = {
                "gate_mean": zero, "gamma_mean": zero,
                "kan_coefficient_magnitude": zero,
                "kan_coefficient_smoothness": zero,
            }
        decoded, auxiliaries, decoder_gates = self.decoder(
            bottleneck, fused, terrain, physical["dem_valid_fractions"],
            sensor_valid, inputs["s1_change"],
        )
        outputs: dict[str, Any] = self.heads(decoded)
        outputs.update({
            "auxiliary_depths": auxiliaries,
            "modality_weights": weights,
            "fusion_entropy": fusion_entropy,
            "terrain_gates": terrain_gates,
            "decoder_gates": decoder_gates,
            "graph_diagnostics": graph_diagnostics,
            "physical_features": physical,
        })
        return outputs


def build_pa_hydrokan_v13_1(config: Mapping[str, Any]) -> PAHydroKANV131:
    if "model" not in config:
        raise ValueError("v13.1 builder requires the full resolved config")
    contract = DatasetContract.load(config["dataset"]["contract"])
    return PAHydroKANV131(config["model"], resolve_band_spec(config, contract))

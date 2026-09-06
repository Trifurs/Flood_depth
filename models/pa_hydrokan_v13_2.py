"""PA-HydroKAN-v13.2: corrected v13 trunk plus HydroEdgeKAN graph."""

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
from models.hydro_edge_kan import HydroEdgeKAN
from models.pa_hydrokan import FORBIDDEN_INPUTS, REQUIRED_INPUTS
from models.terrain_features_v13 import TerrainFeaturePyramidV13


class PAHydroKANV132(nn.Module):
    """A compact v13 trunk with an explicitly factored terrain edge graph."""

    def __init__(self, model_config: Mapping[str, Any], band_spec: BandSpec) -> None:
        super().__init__()
        channels = [int(value) for value in model_config.get("channels", [32, 64, 128, 192])]
        if len(channels) != 4:
            raise ValueError("PA-HydroKAN-v13.2 requires four encoder scales")
        graph_scale = int(model_config.get("graph_scale", 8))
        if graph_scale != 8:
            raise ValueError("v13.2 currently supports the controlled 1/8 graph scale")
        dropout, groups = float(model_config.get("dropout", 0.10)), int(model_config.get("group_norm_groups", 8))
        block_kind = str(model_config.get("residual_block", "efficient"))
        self.band_spec = band_spec
        self.s1_encoder = GatedCrossStateEncoder(band_spec.channels("s1_t1"), band_spec.channels("s1_change"), channels, dropout, groups, block_kind, band_spec.channels("s1_conditioning"))
        self.s2_encoder = GatedCrossStateEncoder(band_spec.channels("s2_t1"), band_spec.channels("s2_change"), channels, dropout, groups, block_kind, 0)
        self.terrain = TerrainFeaturePyramidV13(band_spec.channels("terrain"), channels, dropout, groups, float(model_config.get("terrain_pixel_size_m", 20.0)), block_kind)
        self.fusion = ContentAwareFusionPyramid(channels, len(RELIABILITY_NAMES), dropout, groups, block_kind)
        self.context = DilatedContext(channels[-1], groups) if bool(model_config.get("context_enabled", True)) else nn.Identity()
        centers = model_config.get("graph_feature_centers")
        scales = model_config.get("graph_feature_scales")
        self.graph = HydroEdgeKAN(
            channels[-1], heads=int(model_config.get("graph_heads", 4)),
            grid_size=int(model_config.get("kan_grid_size", 4)),
            spline_order=int(model_config.get("kan_spline_order", 3)), graph_scale=graph_scale,
            terrain_pixel_size_m=float(model_config.get("terrain_pixel_size_m", 20.0)),
            feature_centers=centers, feature_scales=scales,
            base_path=str(model_config.get("kan_base_path", "silu")),
            base_scale_init=float(model_config.get("kan_base_scale_init", 0.5)),
            spline_scale_init=float(model_config.get("kan_spline_scale_init", 1.0)),
            learnable_base_scale=bool(model_config.get("kan_learnable_base_scale", True)),
            learnable_spline_scale=bool(model_config.get("kan_learnable_spline_scale", True)),
            gamma_init_effective=float(model_config.get("kan_gamma_init_effective", 0.02)),
            gamma_max=float(model_config.get("kan_gamma_max", 0.25)),
            latent_compatibility_enabled=bool(model_config.get("latent_compatibility_enabled", True)),
            edge_gate_type=str(model_config.get("edge_gate_type", "kan")),
            groups=groups,
        )
        self.graph_enabled = bool(model_config.get("graph_enabled", True))
        self.decoder = GatedFPNDecoder(channels, dropout, groups, block_kind, bool(model_config.get("deep_supervision_enabled", True)))
        self.heads = SeparateFloodDepthHeads(
            channels[0], groups, float(model_config.get("uncertainty_epsilon", 0.001)),
            float(model_config.get("uncertainty_maximum", 5.0)),
            str(model_config.get("depth_output_semantics", "conditional_positive_v2")),
            model_config.get("depth_initialization_bias"),
            bool(model_config.get("support_backbone_gradient", False)),
            bool(model_config.get("uncertainty_backbone_gradient", True)),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        forbidden = FORBIDDEN_INPUTS.intersection(inputs)
        if forbidden: raise ValueError(f"Forbidden inputs: {sorted(forbidden)}")
        missing = REQUIRED_INPUTS.difference(inputs)
        if missing: raise KeyError(f"Missing PA-HydroKAN-v13.2 inputs: {sorted(missing)}")
        branch_validity = inputs.get("branch_validity", {})
        conditioning = inputs.get("s1_conditioning")
        if self.band_spec.channels("s1_conditioning") and conditioning is None:
            raise KeyError("Missing configured s1_conditioning")
        s1 = self.s1_encoder(inputs["s1_t1"], inputs["s1_t2"], inputs["s1_change"], inputs["s1_valid"], conditioning, {k.removeprefix("s1_"): v for k, v in branch_validity.items() if k.startswith("s1_")})
        s2 = self.s2_encoder(inputs["s2_t1"], inputs["s2_t2"], inputs["s2_change"], inputs["s2_valid"], None, {k.removeprefix("s2_"): v for k, v in branch_validity.items() if k.startswith("s2_")})
        terrain, physical = self.terrain(inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"])
        fused, weights, terrain_gates, fusion_entropy = self.fusion(s1, s2, terrain, inputs["reliability"], inputs["s1_valid"], inputs["s2_valid"], physical["dem_valid_fractions"])
        bottleneck = self.context(fused[-1])
        day_index = RELIABILITY_NAMES.index("absolute_normalized_sensor_day_difference")
        day_difference = inputs["reliability"][:, day_index:day_index + 1]
        sensor_valid = torch.maximum(inputs["s1_valid"], inputs["s2_valid"])
        if self.graph_enabled:
            bottleneck, graph_diagnostics = self.graph(bottleneck, physical, day_difference, weights[-1], sensor_valid)
        else:
            zero = bottleneck.sum() * 0.0
            graph_diagnostics = {"gate_mean": zero, "gamma_mean": zero, "kan_coefficient_magnitude": zero, "kan_coefficient_smoothness": zero}
        decoded, auxiliaries = self.decoder(bottleneck, fused, terrain, physical["dem_valid_fractions"])
        outputs = self.heads(decoded)
        outputs.update({"auxiliary_depths": auxiliaries, "modality_weights": weights, "terrain_gates": terrain_gates, "fusion_entropy": fusion_entropy, "graph_diagnostics": graph_diagnostics, "physical_features": physical})
        return outputs


def build_pa_hydrokan_v13_2(config: Mapping[str, Any]) -> PAHydroKANV132:
    if "model" not in config: raise ValueError("v13.2 builder requires the full resolved config")
    contract = DatasetContract.load(config["dataset"]["contract"])
    return PAHydroKANV132(config["model"], resolve_band_spec(config, contract))

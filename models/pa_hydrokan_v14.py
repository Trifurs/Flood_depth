"""PA-HydroKAN-v14: corrected-v13 trunk with independently ablatable modules."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from datasets.band_selection import BandSpec, resolve_band_spec
from datasets.contract import DatasetContract
from datasets.preprocessing import RELIABILITY_NAMES
from models.decoder_v14 import IndependentGatedFPNDecoderV14
from models.decoder_v13 import GatedFPNDecoderV131, SeparateFloodDepthHeads
from models.efficient_blocks import DilatedContext, GatedCrossStateEncoder
from models.fusion_v13 import ContentAwareFusionPyramidV131
from models.hydro_edge_kan_v14 import HydroEdgeKANV14
from models.pa_hydrokan import FORBIDDEN_INPUTS, REQUIRED_INPUTS
from models.terrain_features_v14 import TerrainFeaturePyramidV14


class PAHydroKANV14(nn.Module):
    """Continuous conditional-positive depth estimator with weak, named priors.

    DSM is used as a DSM/ground-like terrain proxy.  The graph performs KAN-gated
    latent feature aggregation and is not a PINN, hydraulic solver, or conservation
    model.
    """

    def __init__(self, model_config: Mapping[str, Any], band_spec: BandSpec, raw_terrain_names: tuple[str, ...]) -> None:
        super().__init__()
        channels = [int(value) for value in model_config.get("channels", [32, 64, 128, 192])]
        if len(channels) != 4:
            raise ValueError("PA-HydroKAN-v14 requires four encoder scales")
        self.band_spec = band_spec
        dropout = float(model_config.get("dropout", 0.10))
        groups = int(model_config.get("group_norm_groups", 8))
        block_kind = str(model_config.get("residual_block", "efficient"))
        self.s1_encoder = GatedCrossStateEncoder(
            band_spec.channels("s1_t1"), band_spec.channels("s1_change"), channels,
            dropout, groups, block_kind, band_spec.channels("s1_conditioning"),
        )
        self.s2_encoder = GatedCrossStateEncoder(
            band_spec.channels("s2_t1"), band_spec.channels("s2_change"), channels,
            dropout, groups, block_kind, 0,
        )
        self.s2_enabled = bool(model_config.get("s2_enabled", True))
        self.terrain = TerrainFeaturePyramidV14(
            band_spec.channels("terrain"), channels, dropout, groups,
            float(model_config.get("terrain_pixel_size_m", 20.0)), raw_terrain_names,
            int(model_config.get("ground_proxy_kernel_size", 9)),
            str(model_config.get("physics_elevation", "z_ground_proxy")), block_kind,
        )
        self.fusion = ContentAwareFusionPyramidV131(channels, len(RELIABILITY_NAMES), dropout, groups, block_kind)
        self.context = DilatedContext(channels[-1], groups) if bool(model_config.get("context_enabled", True)) else nn.Identity()
        self.graph_enabled = bool(model_config.get("graph_enabled", True))
        self.graph = HydroEdgeKANV14(
            channels[-1],
            heads=int(model_config.get("graph_heads", 2)),
            grid_size=int(model_config.get("kan_grid_size", 4)),
            spline_order=int(model_config.get("kan_spline_order", 3)),
            graph_scale=int(model_config.get("graph_scale", 4)),
            terrain_pixel_size_m=float(model_config.get("terrain_pixel_size_m", 20.0)),
            feature_centers=model_config.get("graph_feature_centers"),
            feature_scales=model_config.get("graph_feature_scales"),
            base_path=str(model_config.get("kan_base_path", "none")),
            base_scale_init=float(model_config.get("kan_base_scale_init", 0.0)),
            spline_scale_init=float(model_config.get("kan_spline_scale_init", 1.0)),
            learnable_base_scale=bool(model_config.get("kan_learnable_base_scale", False)),
            learnable_spline_scale=bool(model_config.get("kan_learnable_spline_scale", True)),
            zero_residual_init=bool(model_config.get("kan_zero_residual_init", True)),
            edge_gate_type=str(model_config.get("edge_gate_type", "kan")),
            prior_slope_init=float(model_config.get("prior_slope_init", 1.0)),
            prior_barrier_init=float(model_config.get("prior_barrier_init", 0.5)),
            prior_relief_init=float(model_config.get("prior_relief_init", 0.25)),
            prior_bias_init=float(model_config.get("prior_bias_init", 0.0)),
            gamma_init_effective=float(model_config.get("kan_gamma_init_effective", 0.02)),
            gamma_max=float(model_config.get("kan_gamma_max", 0.25)),
            latent_compatibility_enabled=bool(model_config.get("latent_compatibility_enabled", True)),
            path_statistic=str(model_config.get("path_barrier_statistic", "max")),
            path_quantile=float(model_config.get("path_barrier_quantile", 0.9)),
            diagnostic_mode=bool(model_config.get("diagnostic_mode", False)),
        )
        decoder_widths = [int(value) for value in model_config.get("decoder_widths", [64, 48, 32])]
        if str(model_config.get("decoder_type", "independent_v14")) == "legacy_v131":
            self.decoder = GatedFPNDecoderV131(
                channels, dropout, groups, block_kind,
                bool(model_config.get("deep_supervision_enabled", True)), decoder_widths,
                band_spec.channels("s1_change"),
            )
            self.legacy_decoder = True
        else:
            self.decoder = IndependentGatedFPNDecoderV14(
                channels, dropout, groups, block_kind,
                bool(model_config.get("deep_supervision_enabled", True)), decoder_widths,
            )
            self.legacy_decoder = False
        self.heads = SeparateFloodDepthHeads(
            decoder_widths[-1], groups,
            float(model_config.get("uncertainty_epsilon", 0.001)),
            float(model_config.get("uncertainty_maximum", 5.0)),
            str(model_config.get("depth_output_semantics", "conditional_positive_v2")),
            model_config.get("depth_initialization_bias"),
            bool(model_config.get("support_backbone_gradient", False)),
            bool(model_config.get("uncertainty_backbone_gradient", False)),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        forbidden = FORBIDDEN_INPUTS.intersection(inputs)
        if forbidden:
            raise ValueError(f"Forbidden inputs: {sorted(forbidden)}")
        missing = REQUIRED_INPUTS.difference(inputs)
        if missing:
            raise KeyError(f"Missing PA-HydroKAN-v14 inputs: {sorted(missing)}")
        branch_validity = inputs.get("branch_validity", {})
        conditioning = inputs.get("s1_conditioning")
        if self.band_spec.channels("s1_conditioning") and conditioning is None:
            raise KeyError("Missing configured s1_conditioning")
        s1 = self.s1_encoder(
            inputs["s1_t1"], inputs["s1_t2"], inputs["s1_change"], inputs["s1_valid"], conditioning,
            {key.removeprefix("s1_"): value for key, value in branch_validity.items() if key.startswith("s1_")},
        )
        if self.s2_enabled:
            s2 = self.s2_encoder(
                inputs["s2_t1"], inputs["s2_t2"], inputs["s2_change"], inputs["s2_valid"], None,
                {key.removeprefix("s2_"): value for key, value in branch_validity.items() if key.startswith("s2_")},
            )
            s2_valid = inputs["s2_valid"]
            fusion_reliability = inputs["reliability"]
            fusion_branch_validity = branch_validity
        else:
            # Keep the input contract stable while making the ablation explicit:
            # no S2 image, S2 validity, S2 QA, or S1--S2 timing feature can affect
            # the prediction.  Zero tensors here are normalized typical-value
            # placeholders, not physical reflectance values.
            s2 = [torch.zeros_like(value) for value in s1]
            s2_valid = torch.zeros_like(inputs["s2_valid"])
            fusion_reliability = inputs["reliability"].clone()
            s2_reliability_indices = (2, 3, 4, 6, 9, 11)
            fusion_reliability[:, list(s2_reliability_indices)] = 0.0
            fusion_branch_validity = dict(branch_validity)
            for key in ("s2_t1", "s2_t2", "s2_change"):
                if key in fusion_branch_validity:
                    fusion_branch_validity[key] = torch.zeros_like(fusion_branch_validity[key])
        terrain, physical = self.terrain(inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"])
        fused, weights, terrain_gates, fusion_entropy = self.fusion(
            s1, s2, terrain, fusion_reliability, inputs["s1_valid"], s2_valid,
            physical["dem_valid_fractions"], fusion_branch_validity,
        )
        bottleneck = self.context(fused[-1])
        day_index = RELIABILITY_NAMES.index("absolute_normalized_sensor_day_difference")
        day_difference = inputs["reliability"][:, day_index:day_index + 1]
        sensor_valid = torch.maximum(inputs["s1_valid"], s2_valid)
        if self.graph_enabled:
            graph_day_difference = day_difference if self.s2_enabled else torch.zeros_like(day_difference)
            bottleneck, graph_diagnostics = self.graph(bottleneck, physical, graph_day_difference, weights[-1], sensor_valid)
        else:
            zero = bottleneck.sum() * 0.0
            graph_diagnostics = {"gate_mean": zero, "gamma_mean": zero, "kan_coefficient_magnitude": zero, "kan_coefficient_smoothness": zero, "kan_monotonicity": zero, "kan_curve_smoothness": zero, "graph_update_rms_ratio": zero}
        if self.legacy_decoder:
            decoded, auxiliaries, decoder_gates = self.decoder(
                bottleneck, fused, terrain, physical["dem_valid_fractions"], sensor_valid, inputs["s1_change"],
            )
        else:
            decoded, auxiliaries, decoder_gates = self.decoder(
                bottleneck, fused, terrain, physical["dem_valid_fractions"], sensor_valid,
            )
        outputs = self.heads(decoded)
        outputs.update({
            "auxiliary_depths": auxiliaries,
            "modality_weights": weights,
            "terrain_gates": terrain_gates,
            "decoder_gates": decoder_gates,
            "fusion_entropy": fusion_entropy,
            "graph_diagnostics": graph_diagnostics,
            "physical_features": physical,
        })
        return outputs


def build_pa_hydrokan_v14(config: Mapping[str, Any]) -> PAHydroKANV14:
    if "model" not in config:
        raise ValueError("v14 builder requires the full resolved config")
    contract = DatasetContract.load(config["dataset"]["contract"])
    spec = resolve_band_spec(config, contract)
    raw_names = tuple(str(value) for value in contract.group("terrain")["band_descriptions"])
    return PAHydroKANV14(config["model"], spec, raw_names)

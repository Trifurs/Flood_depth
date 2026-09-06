"""PA-HydroKAN-S1-v15: SAR-first hydrologic depth model.

This is a new model identity.  It is intentionally not a compatibility alias for
v14: the state/change encoder, fusion path, context block and output head are
reconstructed for the sparse, partially observed optical-free setting.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from datasets.band_selection import BandSpec, resolve_band_spec
from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec
from datasets.reliability_spec import ReliabilitySpec
from models.hydro_edge_kan_s1 import HydroEdgeKANS1
from models.heads import GlobalEventDepthScale
from models.s1_hydrology_backbone_v15 import (
    HydrologyContextV15,
    SARHydrologyEncoderV15,
    SARReliabilityConditioner,
    S1HydrologyFusionV15,
)
from models.sar_hydro_decoder import SARHydroDecoder
from models.task_head import TaskHead
from models.terrain_features_v14 import TerrainFeaturePyramidV14


S1_V15_REQUIRED_INPUTS = {
    "s1_t1", "s1_t2", "s1_change", "s1_qa", "terrain", "terrain_raw",
    "reliability", "s1_valid", "s1_event_support", "dem_valid",
}
S1_V15_FORBIDDEN_INPUTS = {
    "label", "masks", "flood_mask", "valid_depth_mask", "split", "sample_origin",
}


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-5), 1.0 - 1.0e-5)
    return float(torch.logit(torch.tensor(value)))


class S1ZeroInflatedDepthHeadsV15(nn.Module):
    """Positive depth with an optional PU-trained zero-inflation gate.

    ``conditional_depth`` remains the physical positive-depth estimate.  When the
    support branch is enabled, ``depth`` is the soft expected depth used for
    deployment, while the conditional branch remains available for diagnostics and
    the supervised positive-depth objective.
    """

    def __init__(
        self,
        channels: int,
        groups: int,
        epsilon: float = 0.001,
        maximum: float = 5.0,
        support_enabled: bool = True,
        output_semantics: str = "probability_weighted_v1",
        support_floor: float = 0.02,
        support_initial_probability: float = 0.10,
        depth_initialization_bias: float = -2.5,
        uncertainty_backbone_gradient: bool = False,
        uncertainty_initial_scale_m: float | None = None,
    ) -> None:
        super().__init__()
        if output_semantics not in {"conditional_positive_v2", "probability_weighted_v1"}:
            raise ValueError(f"Unsupported S1-v15 depth semantics: {output_semantics!r}")
        if not 0.0 <= support_floor < 1.0:
            raise ValueError("support_floor must lie in [0, 1)")
        self.depth_head = TaskHead(channels, groups)
        self.uncertainty_head = TaskHead(channels, groups)
        self.support_head = TaskHead(channels, groups) if support_enabled else None
        self.support_enabled = bool(support_enabled)
        self.output_semantics = output_semantics
        self.depth_output_semantics = output_semantics
        self.support_floor = float(support_floor)
        self.epsilon = float(epsilon)
        self.maximum = float(maximum)
        self.uncertainty_backbone_gradient = bool(uncertainty_backbone_gradient)
        if self.epsilon <= 0 or self.maximum <= 0:
            raise ValueError("uncertainty epsilon and maximum must be positive")
        depth_final = self.depth_head.trunk[-1]
        assert isinstance(depth_final, nn.Conv2d)
        nn.init.constant_(depth_final.bias, float(depth_initialization_bias))
        if uncertainty_initial_scale_m is not None:
            scale = float(uncertainty_initial_scale_m)
            if not self.epsilon < scale < self.maximum + self.epsilon:
                raise ValueError(
                    "uncertainty_initial_scale_m must lie between epsilon and maximum"
                )
            uncertainty_final = self.uncertainty_head.trunk[-1]
            assert isinstance(uncertainty_final, nn.Conv2d)
            ratio = (scale - self.epsilon) / self.maximum
            nn.init.constant_(uncertainty_final.bias, _logit(ratio))
        if self.support_head is not None:
            support_final = self.support_head.trunk[-1]
            assert isinstance(support_final, nn.Conv2d)
            nn.init.constant_(support_final.bias, _logit(support_initial_probability))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        conditional = F.softplus(self.depth_head(features)) + self.epsilon
        uncertainty_features = features if self.uncertainty_backbone_gradient else features.detach()
        scale = self.epsilon + self.maximum * torch.sigmoid(
            self.uncertainty_head(uncertainty_features)
        )
        outputs: dict[str, torch.Tensor] = {
            "conditional_depth": conditional,
            "positive_depth": conditional,
            "uncertainty_scale": scale,
        }
        if self.support_head is not None:
            support_logits = self.support_head(features.detach())
            support_probability = torch.sigmoid(support_logits)
            expected = conditional * (
                self.support_floor + (1.0 - self.support_floor) * support_probability
            )
            outputs.update({
                "support_logits": support_logits,
                "support_probability": support_probability,
                "expected_depth": expected,
            })
            outputs["depth"] = (
                expected if self.depth_output_semantics == "probability_weighted_v1" else conditional
            )
        else:
            outputs["expected_depth"] = conditional
            outputs["depth"] = conditional
        return outputs

    def set_depth_output_semantics(self, value: str) -> None:
        if value not in {"conditional_positive_v2", "probability_weighted_v1"}:
            raise ValueError(f"Unsupported S1-v15 depth semantics: {value!r}")
        self.depth_output_semantics = value


class PAHydroKANS1V15(nn.Module):
    """SAR-first S1-only model with reliability, terrain and Edge-KAN reasoning."""

    def __init__(
        self,
        model_config: Mapping[str, Any],
        band_spec: BandSpec,
        raw_terrain_names: tuple[str, ...],
        input_spec: ModelInputSpec,
    ) -> None:
        super().__init__()
        if not input_spec.is_s1_only:
            raise ValueError("PA-HydroKAN-S1-v15 requires dataset.input_mode='s1_terrain'")
        self.input_spec = input_spec
        self.reliability_spec = ReliabilitySpec.from_mode(input_spec.mode)
        self.band_spec = band_spec
        channels = [int(value) for value in model_config.get("channels", [32, 64, 128, 192])]
        if len(channels) != 4:
            raise ValueError("PA-HydroKAN-S1-v15 requires four encoder scales")
        dropout = float(model_config.get("dropout", 0.10))
        groups = int(model_config.get("group_norm_groups", 8))
        block_kind = str(model_config.get("residual_block", "efficient"))
        self.p0_corrections_enabled = bool(
            model_config.get("p0_corrections_enabled", False)
        )
        reliability_channels = len(self.reliability_spec.names)
        qa_channels = int(model_config.get("s1_qa_channels", 5))
        self.deduplicated_sar_reliability = bool(
            model_config.get("deduplicated_sar_reliability", False)
        )
        if self.deduplicated_sar_reliability and qa_channels != 0:
            raise ValueError(
                "deduplicated_sar_reliability requires model.s1_qa_channels=0; "
                "observation count/day must enter through SARReliabilityConditioner only"
            )
        self.reliability_conditioner = (
            SARReliabilityConditioner(reliability_channels, channels, groups=groups)
            if self.deduplicated_sar_reliability
            else None
        )
        self.sar_encoder = SARHydrologyEncoderV15(
            band_spec.channels("s1_t1"), band_spec.channels("s1_change"),
            qa_channels, reliability_channels,
            channels,
            dropout,
            groups,
            block_kind,
            band_spec.channels("s1_conditioning"),
            self.p0_corrections_enabled,
            self.deduplicated_sar_reliability,
        )
        self.terrain = TerrainFeaturePyramidV14(
            band_spec.channels("terrain"), channels, dropout, groups,
            float(model_config.get("terrain_pixel_size_m", 20.0)), raw_terrain_names,
            int(model_config.get("ground_proxy_kernel_size", 9)),
            str(model_config.get("physics_elevation", "z_ground_proxy")), block_kind,
        )
        self.fusion = S1HydrologyFusionV15(
            channels, reliability_channels, dropout, groups, block_kind,
            float(model_config.get("terrain_mix_init", 0.30)),
            float(model_config.get("terrain_alpha_max", 1.0)),
            self.p0_corrections_enabled,
            self.deduplicated_sar_reliability,
        )
        self.context = (
            HydrologyContextV15(channels[-1], groups, dropout=0.05)
            if bool(model_config.get("context_enabled", True))
            else nn.Identity()
        )
        self.event_depth_scale = (
            GlobalEventDepthScale(
                channels[-1],
                int(model_config.get("event_depth_scale_hidden_channels", 64)),
                float(model_config.get("event_depth_scale_max_log_abs", 0.6931471805599453)),
            )
            if bool(model_config.get("event_depth_scale_enabled", False))
            else None
        )
        self.graph_enabled = bool(model_config.get("graph_enabled", True))
        self.actual_graph_feature_stride = 8
        self.graph_feature_stride = int(
            model_config.get("graph_feature_stride", self.actual_graph_feature_stride)
        )
        if self.p0_corrections_enabled and self.graph_feature_stride != self.actual_graph_feature_stride:
            raise ValueError(
                "PA-HydroKAN-S1-v15 places HydroEdgeKAN on the 1/8 bottleneck; "
                f"graph_feature_stride must be 8, got {self.graph_feature_stride}"
            )
        self.graph_descriptor_stride = (
            self.graph_feature_stride
            if self.p0_corrections_enabled
            else int(model_config.get("graph_scale", 4))
        )
        self._last_graph_feature_shape: tuple[int, int] | None = None
        self.graph = HydroEdgeKANS1(
            channels[-1], heads=int(model_config.get("graph_heads", 2)),
            grid_size=int(model_config.get("kan_grid_size", 4)),
            spline_order=int(model_config.get("kan_spline_order", 3)),
            graph_feature_stride=self.graph_descriptor_stride,
            terrain_pixel_size_m=float(model_config.get("terrain_pixel_size_m", 20.0)),
            feature_centers=model_config.get("graph_feature_centers"),
            feature_scales=model_config.get("graph_feature_scales"),
            gamma_init_effective=float(model_config.get("kan_gamma_init_effective", 0.06)),
            gamma_max=float(model_config.get("kan_gamma_max", 0.35)),
            latent_compatibility_enabled=bool(model_config.get("latent_compatibility_enabled", True)),
            diagnostics_enabled=bool(
                model_config.get(
                    "diagnostics_enabled", model_config.get("diagnostic_mode", False)
                )
            ),
            p0_corrected=self.p0_corrections_enabled,
        )
        widths = model_config.get("decoder_widths", [96, 64, 48, 32])
        self.decoder = SARHydroDecoder(
            channels, dropout, groups, block_kind, widths,
            int(model_config.get("auxiliary_count", 1)),
            int(model_config.get("auxiliary_stage", 0)),
        )
        self.heads = S1ZeroInflatedDepthHeadsV15(
            int(widths[-1]), groups,
            float(model_config.get("uncertainty_epsilon", 0.001)),
            float(model_config.get("uncertainty_maximum", 5.0)),
            bool(model_config.get("support_enabled", True)),
            str(model_config.get("depth_output_semantics", "probability_weighted_v1")),
            float(model_config.get("support_floor", 0.02)),
            float(model_config.get("support_initial_probability", 0.10)),
            float(model_config.get("depth_initialization_bias", -2.5)),
            bool(model_config.get("uncertainty_backbone_gradient", False)),
            model_config.get("uncertainty_initial_scale_m"),
        )

    def graph_identity(self) -> dict[str, Any]:
        if self.p0_corrections_enabled:
            return self.graph.graph_identity(self._last_graph_feature_shape)
        return {
            "legacy_configured_graph_scale": self.graph_descriptor_stride,
            "actual_graph_feature_stride": self.actual_graph_feature_stride,
            "legacy_node_spacing_m_used": self.graph.graph_pixel_size_m,
            "graph_feature_shape": (
                list(self._last_graph_feature_shape)
                if self._last_graph_feature_shape is not None
                else None
            ),
        }

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        forbidden = S1_V15_FORBIDDEN_INPUTS.intersection(inputs)
        forbidden.update(key for key in inputs if str(key).startswith("s2_"))
        if forbidden:
            raise ValueError(f"Forbidden non-S1 inputs: {sorted(forbidden)}")
        missing = S1_V15_REQUIRED_INPUTS.difference(inputs)
        if missing:
            raise KeyError(f"Missing PA-HydroKAN-S1-v15 inputs: {sorted(missing)}")
        branch_validity = inputs.get("branch_validity", {})
        conditioning = inputs.get("s1_conditioning")
        if self.band_spec.channels("s1_conditioning") and conditioning is None:
            raise KeyError("Missing configured S1 angle conditioning")
        reliability_features = (
            self.reliability_conditioner(inputs["reliability"], dict(branch_validity))
            if self.reliability_conditioner is not None
            else None
        )
        sar, sar_diagnostics = self.sar_encoder(
            inputs["s1_t1"], inputs["s1_t2"], inputs["s1_change"], inputs["s1_qa"],
            inputs["reliability"], inputs["s1_valid"], conditioning,
            dict(branch_validity), reliability_features,
        )
        terrain, physical = self.terrain(
            inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"]
        )
        fused, fusion_diagnostics = self.fusion(
            sar,
            terrain,
            physical,
            reliability_features if reliability_features is not None else inputs["reliability"],
            inputs["s1_event_support"],
        )
        bottleneck = self.context(fused[-1])
        observation_confidence = sar_diagnostics["quality_gates"][-1]
        if self.graph_enabled:
            self._last_graph_feature_shape = tuple(int(value) for value in bottleneck.shape[-2:])
            graph_args: dict[str, Any] = {}
            if self.p0_corrections_enabled:
                graph_args["feature_stride"] = self.actual_graph_feature_stride
            bottleneck, graph_diagnostics = self.graph(
                bottleneck,
                physical,
                inputs["s1_event_support"],
                observation_confidence,
                **graph_args,
            )
        else:
            zero = bottleneck.sum() * 0.0
            graph_diagnostics = {
                "gate_mean": zero, "valid_edge_fraction": zero,
                "static_topographic_affinity_mean": zero,
                "observation_confidence_mean": zero,
                "latent_compatibility_mean": zero,
                "kan_coefficient_magnitude": zero,
                "kan_coefficient_smoothness": zero,
                "kan_monotonicity": zero, "kan_curve_smoothness": zero,
                "gamma_mean": zero, "graph_update_rms_ratio": zero,
                "topographic_kan_logit_mean": zero,
                "topographic_affinity_mean": zero,
                "observation_amplitude_mean": zero,
                "final_graph_gate_mean": zero,
                "graph_input_rms": zero,
                "graph_update_rms": zero,
                "spline_output_rms": zero,
                "base_output_rms": zero,
                "spline_base_rms_ratio": zero,
            }
        decoded, auxiliaries, decoder_gates = self.decoder(
            bottleneck, fused, terrain, physical["dem_valid_fractions"],
            inputs["s1_event_support"], sar_diagnostics["change_evidence"],
        )
        outputs = self.heads(decoded)
        if self.event_depth_scale is not None:
            scale, log_scale = self.event_depth_scale(
                bottleneck,
                inputs["dem_valid"] * inputs["s1_event_support"],
            )
            conditional = outputs["conditional_depth"] * scale
            outputs["conditional_depth"] = conditional
            outputs["positive_depth"] = conditional
            if "support_probability" in outputs:
                support = outputs["support_probability"]
                expected = conditional * (
                    self.heads.support_floor
                    + (1.0 - self.heads.support_floor) * support
                )
                outputs["expected_depth"] = expected
                outputs["depth"] = (
                    expected
                    if self.heads.depth_output_semantics == "probability_weighted_v1"
                    else conditional
                )
            else:
                outputs["expected_depth"] = conditional
                outputs["depth"] = conditional
            outputs["event_depth_scale"] = scale
            outputs["event_log_depth_scale"] = log_scale
        outputs.update({
            "auxiliary_depths": auxiliaries,
            "decoder_gates": decoder_gates,
            "fusion_diagnostics": fusion_diagnostics,
            "sar_diagnostics": sar_diagnostics,
            "graph_diagnostics": graph_diagnostics,
            "physical_features": physical,
        })
        return outputs


def build_pa_hydrokan_s1_v15(config: Mapping[str, Any]) -> PAHydroKANS1V15:
    if "model" not in config:
        raise ValueError("PA-HydroKAN-S1-v15 builder requires the full resolved config")
    input_spec = ModelInputSpec.from_config(config)
    if not input_spec.is_s1_only:
        raise ValueError("PA-HydroKAN-S1-v15 requires dataset.input_mode='s1_terrain'")
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    raw_names = tuple(str(value) for value in contract.group("terrain")["band_descriptions"])
    return PAHydroKANS1V15(config["model"], band_spec, raw_names, input_spec)

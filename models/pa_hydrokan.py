"""Production PA-HydroKAN SAR-and-terrain model.

Paper name: Prior-Aware Hydrologic Kolmogorov-Arnold Network.
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
from models.evidence_calibration import GlobalEvidenceCalibration
from models.hydro_edge_kan import HydroEdgeKAN
from models.s1_hydrology_backbone import (
    HydrologyContext,
    JointSARHydrologyEncoder,
    SARHydrologyEncoder,
    S1HydrologyFusion,
    SARReliabilityConditioner,
)
from models.sar_hydro_decoder import SARHydroDecoder
from models.task_head import ContextualDepthRangeCalibration, TaskHead
from models.terrain_features import TerrainFeaturePyramid


PAPER_MODEL_NAME = "PA-HydroKAN"
PAPER_MODEL_EXPANSION = "Prior-Aware Hydrologic Kolmogorov-Arnold Network"


REQUIRED_INPUTS = {
    "s1_t1",
    "s1_t2",
    "s1_change",
    "terrain",
    "terrain_raw",
    "reliability",
    "s1_valid",
    "s1_event_support",
    "dem_valid",
}
FORBIDDEN_INPUTS = {
    "label",
    "masks",
    "flood_mask",
    "valid_depth_mask",
    "split",
    "sample_origin",
}


def _logit(value: float) -> float:
    bounded = min(max(float(value), 1.0e-5), 1.0 - 1.0e-5)
    return float(torch.logit(torch.tensor(bounded)))


class PAHydroKANHeads(nn.Module):
    """Conditional-Depth Head (CDH) with a detached uncertainty branch."""

    def __init__(
        self,
        channels: int,
        groups: int,
        *,
        epsilon: float,
        maximum: float,
        depth_initialization_bias: float,
        uncertainty_initial_scale_m: float,
        uncertainty_head_enabled: bool = True,
        topographic_context_channels: int = 0,
        hydrostatic_adapter_enabled: bool = False,
        depth_range_calibration_enabled: bool = False,
        depth_range_calibration_width: int = 16,
        depth_range_calibration_max_scale_residual: float = 1.0,
        depth_range_calibration_max_bias_residual: float = 2.0,
        depth_range_calibration_tail_threshold_m: float = 0.46,
        depth_range_calibration_tail_temperature_m: float = 0.10,
        depth_range_calibration_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0 or maximum <= 0.0:
            raise ValueError("uncertainty epsilon and maximum must be positive")
        if not epsilon < uncertainty_initial_scale_m < maximum + epsilon:
            raise ValueError("uncertainty_initial_scale_m is outside the supported range")
        self.depth_head = TaskHead(channels, groups)
        self.uncertainty_head = (
            TaskHead(channels, groups) if uncertainty_head_enabled else None
        )
        if hydrostatic_adapter_enabled and topographic_context_channels <= 0:
            raise ValueError(
                "hydrostatic head adaptation requires topographic context channels"
            )
        self.hydrostatic_adapter = (
            nn.Conv2d(topographic_context_channels, 1, 1, bias=False)
            if hydrostatic_adapter_enabled
            else None
        )
        if self.hydrostatic_adapter is not None:
            # The established conditional-depth mapping is the exact initial
            # function; training learns only a signed logit correction from
            # physically normalized multi-scale elevation position.
            nn.init.zeros_(self.hydrostatic_adapter.weight)
        self.epsilon = float(epsilon)
        self.maximum = float(maximum)
        self.uncertainty_initial_scale_m = float(uncertainty_initial_scale_m)
        self.register_buffer(
            "_fixed_uncertainty_scale",
            torch.tensor(self.uncertainty_initial_scale_m, dtype=torch.float32),
            persistent=False,
        )
        self.depth_range_calibration = (
            ContextualDepthRangeCalibration(
                channels,
                groups,
                width=int(depth_range_calibration_width),
                maximum_scale_residual=float(
                    depth_range_calibration_max_scale_residual
                ),
                maximum_bias_residual=float(
                    depth_range_calibration_max_bias_residual
                ),
                tail_threshold_m=float(
                    depth_range_calibration_tail_threshold_m
                ),
                tail_temperature_m=float(
                    depth_range_calibration_tail_temperature_m
                ),
                depth_epsilon=self.epsilon,
                strength=float(depth_range_calibration_strength),
            )
            if depth_range_calibration_enabled
            else None
        )
        self.depth_output_semantics = "conditional_positive"
        depth_final = self.depth_head.trunk[-1]
        assert isinstance(depth_final, nn.Conv2d)
        nn.init.constant_(depth_final.bias, float(depth_initialization_bias))
        if self.uncertainty_head is not None:
            uncertainty_final = self.uncertainty_head.trunk[-1]
            assert isinstance(uncertainty_final, nn.Conv2d)
            ratio = (float(uncertainty_initial_scale_m) - self.epsilon) / self.maximum
            nn.init.constant_(uncertainty_final.bias, _logit(ratio))

    def forward(
        self,
        features: torch.Tensor,
        topographic_context: list[torch.Tensor] | None = None,
        global_calibration: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        depth_logit = self.depth_head(features)
        if self.hydrostatic_adapter is not None:
            if not topographic_context:
                raise ValueError("hydrostatic head adapter received no topographic context")
            depth_logit = depth_logit + self.hydrostatic_adapter(
                torch.cat(topographic_context, dim=1)
            )
        if global_calibration is not None:
            scale_residual, bias_residual = global_calibration
            expected_shape = (depth_logit.shape[0], 1, 1, 1)
            if scale_residual.shape != expected_shape or bias_residual.shape != expected_shape:
                raise ValueError(
                    "global calibration scale and bias must have shape (B, 1, 1, 1)"
                )
            depth_logit = depth_logit * (1.0 + scale_residual) + bias_residual
        depth_range_scale = None
        depth_range_bias = None
        if self.depth_range_calibration is not None:
            depth_logit, depth_range_scale, depth_range_bias = (
                self.depth_range_calibration(features, depth_logit)
            )
        depth = F.softplus(depth_logit) + self.epsilon
        scale = (
            self.epsilon
            + self.maximum * torch.sigmoid(self.uncertainty_head(features.detach()))
            if self.uncertainty_head is not None
            else self._fixed_uncertainty_scale.to(dtype=features.dtype)
            .view(1, 1, 1, 1)
            .expand(features.shape[0], 1, *features.shape[-2:])
        )
        return {
            "conditional_depth": depth,
            "positive_depth": depth,
            "expected_depth": depth,
            "depth": depth,
            "uncertainty_scale": scale,
            "depth_range_scale": depth_range_scale,
            "depth_range_bias": depth_range_bias,
        }

    def set_depth_output_semantics(self, value: str) -> None:
        if value != "conditional_positive":
            raise ValueError("production inference supports conditional_positive output only")
        self.depth_output_semantics = value


class PAHydroKAN(nn.Module):
    """PA-HydroKAN: a prior-aware SAR/topography depth estimator.

    The named paper modules are RCP (reliability conditioning), TCSE
    (temporal-change SAR encoder), TPP (topographic-prior pyramid), TCF
    (terrain-conditioned fusion), TAE-KAN (topographic-affinity Edge-KAN),
    DGD (dual-gated decoder), and CDH (conditional-depth head).
    """

    def __init__(
        self,
        model_config: Mapping[str, Any],
        band_spec: BandSpec,
        raw_terrain_names: tuple[str, ...],
        input_spec: ModelInputSpec,
    ) -> None:
        super().__init__()
        if not input_spec.is_s1_only:
            raise ValueError("PAHydroKAN requires the S1-only input contract")
        self.input_spec = input_spec
        self.reliability_spec = ReliabilitySpec.from_mode(input_spec.mode)
        self.band_spec = band_spec
        self.reliability_conditioning_enabled = bool(
            model_config.get("reliability_conditioning_enabled", True)
        )
        self.terrain_conditioned_fusion_enabled = bool(
            model_config.get("terrain_conditioned_fusion_enabled", True)
        )
        self.topographic_affinity_enabled = bool(
            model_config.get("topographic_affinity_enabled", True)
        )
        # LCA is an internal TAE-KAN mechanism, so its runtime state cannot be
        # enabled independently when the parent graph path is disabled.  This
        # keeps component metadata faithful to the executed computation graph
        # while preserving the state-dictionary schema for every ablation.
        self.latent_compatibility_enabled = self.topographic_affinity_enabled and bool(
            model_config.get("latent_compatibility_enabled", True)
        )
        channels = [int(value) for value in model_config["channels"]]
        if len(channels) != 4 or any(value <= 0 for value in channels):
            raise ValueError("PAHydroKAN requires four positive encoder scales")
        dropout = float(model_config["dropout"])
        groups = int(model_config["group_norm_groups"])
        block_kind = str(model_config["residual_block"])
        sar_block_kind = str(model_config.get("sar_residual_block", block_kind))
        terrain_block_kind = str(
            model_config.get("terrain_residual_block", block_kind)
        )
        fusion_block_kind = str(
            model_config.get("fusion_residual_block", block_kind)
        )
        decoder_block_kind = str(
            model_config.get("decoder_residual_block", block_kind)
        )
        reliability_channels = len(self.reliability_spec.names)
        self.reliability_conditioner = SARReliabilityConditioner(
            reliability_channels, channels, groups=groups
        )
        encoder_variant = str(
            model_config.get("temporal_change_encoder", "decomposed")
        )
        if encoder_variant not in {"decomposed", "joint"}:
            raise ValueError(
                "model.temporal_change_encoder must be 'decomposed' or 'joint'"
            )
        encoder_type = (
            JointSARHydrologyEncoder
            if encoder_variant == "joint"
            else SARHydrologyEncoder
        )
        self.temporal_change_encoder = encoder_variant
        self.sar_encoder = encoder_type(
            band_spec.channels("s1_t1"),
            band_spec.channels("s1_change"),
            channels,
            dropout=dropout,
            groups=groups,
            block_kind=sar_block_kind,
            conditioning_channels=band_spec.channels("s1_conditioning"),
        )
        self.terrain = TerrainFeaturePyramid(
            band_spec.channels("terrain"),
            channels,
            dropout,
            groups,
            float(model_config["terrain_pixel_size_m"]),
            raw_terrain_names,
            int(model_config["ground_proxy_kernel_size"]),
            str(model_config["physics_elevation"]),
            terrain_block_kind,
            model_config.get("topographic_context_scales_m", ()),
        )
        self.fusion = S1HydrologyFusion(
            channels,
            dropout=dropout,
            groups=groups,
            block_kind=fusion_block_kind,
            terrain_mix_init=float(model_config["terrain_mix_init"]),
            terrain_alpha_max=float(model_config["terrain_alpha_max"]),
        )
        self.context = HydrologyContext(
            channels[-1],
            groups,
            dropout=0.05,
            global_context_enabled=bool(
                model_config.get("global_context_enabled", False)
            ),
        )
        graph_stride = int(model_config["graph_feature_stride"])
        if graph_stride != 8:
            raise ValueError("PAHydroKAN applies graph reasoning at encoder stride 8")
        self.graph = HydroEdgeKAN(
            channels[-1],
            heads=int(model_config["graph_heads"]),
            grid_size=int(model_config["kan_grid_size"]),
            spline_order=int(model_config["kan_spline_order"]),
            graph_feature_stride=graph_stride,
            terrain_pixel_size_m=float(model_config["terrain_pixel_size_m"]),
            feature_centers=model_config["graph_feature_centers"],
            feature_scales=model_config["graph_feature_scales"],
            gamma_init_effective=float(model_config["kan_gamma_init_effective"]),
            gamma_max=float(model_config["kan_gamma_max"]),
            latent_compatibility_enabled=self.latent_compatibility_enabled,
            path_barrier_mode=str(
                model_config.get("graph_path_barrier_mode", "full_resolution")
            ),
            diagnostics_enabled=bool(model_config.get("diagnostics_enabled", False)),
        )
        widths = [int(value) for value in model_config["decoder_widths"]]
        self.decoder = SARHydroDecoder(
            channels,
            dropout,
            groups,
            decoder_block_kind,
            widths,
            int(model_config["auxiliary_count"]),
            int(model_config.get("auxiliary_stage", 0)),
            skip_fusion=str(model_config.get("decoder_skip_fusion", "additive")),
        )
        self.global_depth_calibration_enabled = bool(
            model_config.get("global_depth_calibration_enabled", False)
        )
        self.depth_range_calibration_enabled = bool(
            model_config.get("depth_range_calibration_enabled", False)
        )
        self.uncertainty_head_enabled = bool(
            model_config.get("uncertainty_head_enabled", True)
        )
        self.global_depth_calibration_strength = float(
            model_config.get("global_depth_calibration_strength", 1.0)
        )
        self.global_depth_calibration_scale_strength = float(
            model_config.get(
                "global_depth_calibration_scale_strength",
                self.global_depth_calibration_strength,
            )
        )
        self.global_depth_calibration_bias_strength = float(
            model_config.get(
                "global_depth_calibration_bias_strength",
                self.global_depth_calibration_strength,
            )
        )
        if min(
            self.global_depth_calibration_strength,
            self.global_depth_calibration_scale_strength,
            self.global_depth_calibration_bias_strength,
        ) < 0.0:
            raise ValueError("global calibration strengths must be nonnegative")
        context_channels = len(model_config.get("topographic_context_scales_m", ()))
        state_channels = band_spec.channels("s1_t1")
        calibration_evidence_channels = (
            widths[-1]
            + 4 * state_channels
            + band_spec.channels("s1_change")
            + band_spec.channels("terrain")
            + band_spec.channels("s1_conditioning")
            + reliability_channels
            + 7
            + 6
            + context_channels
        )
        self.global_depth_calibration_adapter = (
            GlobalEvidenceCalibration(
                calibration_evidence_channels,
                hidden_channels=int(
                    model_config.get("global_depth_calibration_hidden", 96)
                ),
                dropout=float(
                    model_config.get("global_depth_calibration_dropout", 0.05)
                ),
                maximum_scale_residual=float(
                    model_config.get(
                        "global_depth_calibration_max_scale_residual", 0.5
                    )
                ),
                maximum_bias_residual=float(
                    model_config.get(
                        "global_depth_calibration_max_bias_residual", 4.0
                    )
                ),
            )
            if self.global_depth_calibration_enabled
            else None
        )
        self.heads = PAHydroKANHeads(
            widths[-1],
            groups,
            epsilon=float(model_config["uncertainty_epsilon"]),
            maximum=float(model_config["uncertainty_maximum"]),
            depth_initialization_bias=float(model_config["depth_initialization_bias"]),
            uncertainty_initial_scale_m=float(model_config["uncertainty_initial_scale_m"]),
            uncertainty_head_enabled=self.uncertainty_head_enabled,
            topographic_context_channels=len(
                model_config.get("topographic_context_scales_m", ())
            ),
            hydrostatic_adapter_enabled=bool(
                model_config.get("hydrostatic_head_adapter_enabled", False)
            ),
            depth_range_calibration_enabled=self.depth_range_calibration_enabled,
            depth_range_calibration_width=int(
                model_config.get("depth_range_calibration_width", 16)
            ),
            depth_range_calibration_max_scale_residual=float(
                model_config.get(
                    "depth_range_calibration_max_scale_residual", 1.0
                )
            ),
            depth_range_calibration_max_bias_residual=float(
                model_config.get(
                    "depth_range_calibration_max_bias_residual", 2.0
                )
            ),
            depth_range_calibration_tail_threshold_m=float(
                model_config.get(
                    "depth_range_calibration_tail_threshold_m", 0.46
                )
            ),
            depth_range_calibration_tail_temperature_m=float(
                model_config.get(
                    "depth_range_calibration_tail_temperature_m", 0.10
                )
            ),
            depth_range_calibration_strength=float(
                model_config.get("depth_range_calibration_strength", 1.0)
            ),
        )
        # Disabled components retain their state-dictionary entries so an
        # ablation can be evaluated against the same initialized/full checkpoint,
        # but they are excluded from optimization and DDP gradient accounting.
        if not self.reliability_conditioning_enabled:
            self.reliability_conditioner.requires_grad_(False)
        if not self.terrain_conditioned_fusion_enabled:
            self.fusion.disable_terrain_conditioned_paths()
        if not self.topographic_affinity_enabled:
            self.graph.requires_grad_(False)
        self._last_graph_feature_shape: tuple[int, int] | None = None

    def component_flags(self) -> dict[str, bool]:
        """Return the named-paper component status for logs and ablation reports."""

        return {
            "reliability_conditioning_enabled": self.reliability_conditioning_enabled,
            "terrain_conditioned_fusion_enabled": self.terrain_conditioned_fusion_enabled,
            "topographic_affinity_enabled": self.topographic_affinity_enabled,
            "latent_compatibility_enabled": self.latent_compatibility_enabled,
            "global_depth_calibration_enabled": self.global_depth_calibration_enabled,
            "depth_range_calibration_enabled": self.depth_range_calibration_enabled,
            "uncertainty_head_enabled": self.uncertainty_head_enabled,
            "joint_temporal_change_encoder": self.temporal_change_encoder == "joint",
        }

    def _global_calibration_evidence_parts(
        self,
        inputs: Mapping[str, torch.Tensor],
        decoded: torch.Tensor,
        physical: Mapping[str, Any],
        branch_validity: Mapping[str, torch.Tensor],
        conditioning: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Return ordered, bounded mask-agnostic evidence groups."""

        common_valid = inputs["s1_valid"].to(decoded.dtype)
        pre_valid = branch_validity.get(
            "s1_t1", branch_validity.get("t1", common_valid)
        ).to(decoded.dtype)
        event_valid = branch_validity.get(
            "s1_t2", branch_validity.get("t2", common_valid)
        ).to(decoded.dtype)
        change_valid = branch_validity.get(
            "s1_change", branch_validity.get("change", common_valid)
        ).to(decoded.dtype)
        pair_valid = torch.minimum(pre_valid, event_valid)
        event_support = inputs["s1_event_support"].to(decoded.dtype)
        dem_valid = inputs["dem_valid"].to(decoded.dtype)

        pre = inputs["s1_t1"].to(decoded.dtype) * pre_valid
        event = inputs["s1_t2"].to(decoded.dtype) * event_valid
        difference = (event - pre) * pair_valid
        change = inputs["s1_change"].to(decoded.dtype) * change_valid
        terrain = inputs["terrain"].to(decoded.dtype) * dem_valid
        reliability_source = inputs["reliability"].to(decoded.dtype)
        # RCP is the sole route by which acquisition-reliability metadata may
        # influence the prediction.  Keeping shape-compatible zeros here makes
        # ``w/o RCP`` a genuine intervention even though CDH retains the same
        # state-dictionary schema as the full model.
        reliability = (
            reliability_source
            if self.reliability_conditioning_enabled
            else torch.zeros_like(reliability_source)
        )
        if conditioning is None:
            conditioning_features = decoded.new_zeros(
                decoded.shape[0], 0, *decoded.shape[-2:]
            )
        else:
            conditioning_features = conditioning.to(decoded.dtype) * event_support

        relief = physical["local_relief"].to(decoded.dtype)
        terrain_scale = relief + 1.0
        physical_features = [
            torch.tanh(physical["z_relative"].to(decoded.dtype) / terrain_scale),
            torch.tanh(physical["obstacle_residual"].to(decoded.dtype) / terrain_scale),
            torch.tanh(physical["dz_dx"].to(decoded.dtype)),
            torch.tanh(physical["dz_dy"].to(decoded.dtype)),
            torch.log1p(relief.clamp_min(0.0)) / 5.0,
            torch.tanh(physical["slope"].to(decoded.dtype)),
        ]
        context = [
            value.to(decoded.dtype)
            for value in physical.get("topographic_context_positions", ())
        ]
        # Four contiguous groups avoid materializing one very large evidence
        # tensor while keeping the reduction kernel count small.  Their channel
        # order matches the feature schema learned by the calibration MLP.
        evidence = (
            decoded,
            torch.cat(
                (
                    pre,
                    event,
                    difference,
                    difference.abs(),
                    change,
                    terrain,
                    conditioning_features,
                    reliability,
                ),
                dim=1,
            ),
            torch.cat(
                (
                    common_valid,
                    pre_valid,
                    event_valid,
                    change_valid,
                    pair_valid,
                    event_support,
                    dem_valid,
                ),
                dim=1,
            ),
            torch.cat((*physical_features, *context), dim=1),
        )
        return tuple(
            torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
            for value in evidence
        )

    def graph_identity(self) -> dict[str, Any] | None:
        if not self.topographic_affinity_enabled:
            return None
        return self.graph.graph_identity(self._last_graph_feature_shape)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        forbidden = FORBIDDEN_INPUTS.intersection(inputs)
        if forbidden:
            raise ValueError(f"Forbidden non-S1 inputs: {sorted(forbidden)}")
        missing = REQUIRED_INPUTS.difference(inputs)
        if missing:
            raise KeyError(f"Missing flood-depth inputs: {sorted(missing)}")
        branch_validity = dict(inputs.get("branch_validity", {}))
        conditioning = inputs.get("s1_conditioning")
        if self.band_spec.channels("s1_conditioning") and conditioning is None:
            raise KeyError("Missing configured S1 angle conditioning")
        reliability_features = (
            self.reliability_conditioner(inputs["reliability"], branch_validity)
            if self.reliability_conditioning_enabled
            else self.reliability_conditioner.zero_features(inputs["reliability"])
        )
        sar, sar_diagnostics = self.sar_encoder(
            inputs["s1_t1"],
            inputs["s1_t2"],
            inputs["s1_change"],
            inputs["s1_valid"],
            conditioning,
            branch_validity,
            reliability_features,
        )
        terrain, physical = self.terrain(
            inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"]
        )
        fused, fusion_diagnostics = self.fusion(
            sar,
            terrain,
            physical,
            reliability_features,
            inputs["s1_event_support"],
            terrain_conditioned_fusion_enabled=self.terrain_conditioned_fusion_enabled,
        )
        bottleneck = self.context(fused[-1])
        if self.topographic_affinity_enabled:
            self._last_graph_feature_shape = tuple(
                int(value) for value in bottleneck.shape[-2:]
            )
            bottleneck, graph_diagnostics = self.graph(
                bottleneck,
                physical,
                inputs["s1_event_support"],
                sar_diagnostics["quality_gates"][-1],
                feature_stride=8,
            )
        else:
            self._last_graph_feature_shape = None
            graph_diagnostics = {}
        decoded, auxiliaries, decoder_gates = self.decoder(
            bottleneck,
            fused,
            terrain,
            physical["dem_valid_fractions"],
            inputs["s1_event_support"],
            sar_diagnostics["change_evidence"],
        )
        calibration_parts = (
            self._global_calibration_evidence_parts(
                inputs, decoded, physical, branch_validity, conditioning
            )
            if self.global_depth_calibration_adapter is not None
            else None
        )
        global_calibration = None
        if self.global_depth_calibration_adapter is not None:
            assert calibration_parts is not None
            uniform_scale, uniform_bias = self.global_depth_calibration_adapter(
                calibration_parts
            )
            global_calibration = (
                self.global_depth_calibration_scale_strength * uniform_scale,
                self.global_depth_calibration_bias_strength * uniform_bias,
            )
        outputs = self.heads(
            decoded,
            physical.get("topographic_context_positions"),
            global_calibration=global_calibration,
        )
        outputs.update(
            {
                "auxiliary_depths": auxiliaries,
                "decoder_gates": decoder_gates,
                "fusion_diagnostics": fusion_diagnostics,
                "sar_diagnostics": sar_diagnostics,
                "graph_diagnostics": graph_diagnostics,
                "physical_features": physical,
                "component_flags": self.component_flags(),
                "global_depth_calibration": global_calibration,
            }
        )
        return outputs


def build_pa_hydrokan(config: Mapping[str, Any]) -> PAHydroKAN:
    """Build PA-HydroKAN from a resolved configuration."""

    if "model" not in config:
        raise ValueError("A resolved model configuration is required")
    input_spec = ModelInputSpec.from_config(config)
    if not input_spec.is_s1_only:
        raise ValueError("The production model requires S1-only inputs")
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    raw_names = tuple(str(value) for value in contract.group("terrain")["band_descriptions"])
    return PAHydroKAN(config["model"], band_spec, raw_names, input_spec)

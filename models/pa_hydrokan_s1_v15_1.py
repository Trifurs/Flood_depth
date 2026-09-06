"""PA-HydroKAN-S1-V15.1 model variants.

The corrected variant deliberately reuses V15 with only the P0 semantics
enabled.  The KAN and simple variants share that stable decoder/output contract,
but place the new terrain-conditioned spatial-compatibility graph at the real
1/4 feature scale.  No variant accepts an optical/Sentinel-2 input.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from datasets.band_selection import BandSpec, resolve_band_spec
from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec
from models.encoders import ConvNormAct
from models.hydro_edge_kan_v15_1 import (
    DEFAULT_EDGE_FEATURE_NAMES,
    HydroEdgeKANV15_1,
)
from models.pa_hydrokan_s1_v15 import (
    PAHydroKANS1V15,
    S1_V15_FORBIDDEN_INPUTS,
    S1_V15_REQUIRED_INPUTS,
)
from models.s1_hydrology_backbone_v15_1 import (
    SARHydrologyEncoderV15_1Simple,
    S1TerrainResidualFusionV15_1,
)
from models.sar_hydro_decoder import SARHydroDecoder


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


class PAHydroKANS1V15_1(PAHydroKANS1V15):
    """S1-only V15.1 corrected, KAN, and lean-SAR candidates.

    ``corrected`` is intentionally a P0-only control. ``kan`` changes only graph
    placement/edge compatibility. ``simple`` subsequently replaces the repeated
    SAR and terrain residual paths while retaining the KAN graph from ``kan``.
    """

    _VARIANTS = {"corrected", "kan", "simple"}

    def __init__(
        self,
        model_config: Mapping[str, Any],
        band_spec: BandSpec,
        raw_terrain_names: tuple[str, ...],
        input_spec: ModelInputSpec,
    ) -> None:
        variant = str(model_config.get("variant", "corrected"))
        if variant not in self._VARIANTS:
            raise ValueError(
                "PA-HydroKAN-S1-V15.1 variant must be one of "
                f"{sorted(self._VARIANTS)}, got {variant!r}"
            )
        if not bool(model_config.get("p0_corrections_enabled", True)):
            raise ValueError("PA-HydroKAN-S1-V15.1 requires p0_corrections_enabled=true")

        # V15 itself places its historical graph at 1/8. Build its stable
        # components there, then replace the graph route for the 1/4 candidates.
        parent_config = dict(model_config)
        if variant != "corrected":
            parent_config["graph_feature_stride"] = 8
        super().__init__(parent_config, band_spec, raw_terrain_names, input_spec)
        self.variant = variant
        self._v15_1_model_config = dict(model_config)
        self._channels = [int(value) for value in model_config.get("channels", [32, 64, 128, 192])]

        if variant == "corrected":
            if int(model_config.get("graph_feature_stride", 8)) != 8:
                raise ValueError("V15.1 corrected is the 1/8 V15 control and requires graph_feature_stride=8")
            return

        graph_stride = int(model_config.get("graph_feature_stride", 4))
        if graph_stride != 4:
            raise ValueError("V15.1 KAN/simple graph must operate on the actual 1/4 feature")
        graph_channels = int(model_config.get("graph_channels", 64))
        graph_heads = int(model_config.get("graph_heads", 2))
        if graph_channels not in {64, 96}:
            raise ValueError("V15.1 graph_channels must be 64 or 96")
        if graph_channels % graph_heads:
            raise ValueError("V15.1 graph_channels must be divisible by graph_heads")

        self.actual_graph_feature_stride = graph_stride
        self.graph_feature_stride = graph_stride
        self.graph_descriptor_stride = graph_stride
        self.graph_input_projection = ConvNormAct(
            self._channels[2], graph_channels, 1,
            groups=int(model_config.get("group_norm_groups", 8)),
        )
        self.graph_output_projection = nn.Conv2d(graph_channels, self._channels[2], 1, bias=False)
        nn.init.orthogonal_(self.graph_output_projection.weight)
        self.graph = HydroEdgeKANV15_1(
            graph_channels,
            heads=graph_heads,
            grid_size=int(model_config.get("kan_grid_size", 4)),
            spline_order=int(model_config.get("kan_spline_order", 3)),
            graph_feature_stride=graph_stride,
            terrain_pixel_size_m=float(model_config.get("terrain_pixel_size_m", 20.0)),
            edge_feature_names=tuple(
                model_config.get("graph_edge_feature_names", DEFAULT_EDGE_FEATURE_NAMES)
            ),
            edge_stats_path=model_config.get("graph_edge_stats"),
            mapping_temperature=float(model_config.get("graph_mapping_temperature", 1.0)),
            gamma_init=float(model_config.get("kan_gamma_init_effective", 0.03)),
            gamma_max=float(model_config.get("kan_gamma_max", 0.25)),
            base_scale_init=float(model_config.get("kan_base_scale_init", 0.35)),
            spline_scale_init=float(model_config.get("kan_spline_scale_init", 1.0)),
            latent_scale_init=float(model_config.get("kan_latent_scale_init", 0.25)),
            symmetric_static_prior_enabled=bool(
                model_config.get("symmetric_static_prior_enabled", False)
            ),
            diagnostics_enabled=bool(
                model_config.get(
                    "diagnostics_enabled", model_config.get("diagnostic_mode", False)
                )
            ),
            regularization_enabled=bool(model_config.get("kan_regularization_enabled", True)),
        )
        self.graph_bottleneck_enabled = bool(
            model_config.get("graph_bottleneck_enabled", False)
        )
        if self.graph_bottleneck_enabled:
            bottleneck_init = float(
                model_config.get("graph_bottleneck_scale_init", 0.03)
            )
            bottleneck_max = float(
                model_config.get("graph_bottleneck_scale_max", 0.10)
            )
            if not 0.0 < bottleneck_init < bottleneck_max:
                raise ValueError(
                    "graph_bottleneck_scale_init must lie strictly between zero "
                    "and graph_bottleneck_scale_max"
                )
            # The graph has already produced an auditable 1/4 residual.  An
            # adaptive mean downsample plus 1x1 projection passes only that
            # residual to the bottleneck, without instantiating a second graph.
            self.graph_bottleneck_projection = nn.Conv2d(
                self._channels[2], self._channels[3], 1, bias=False
            )
            nn.init.orthogonal_(self.graph_bottleneck_projection.weight)
            self.graph_bottleneck_scale_max = bottleneck_max
            self.raw_graph_bottleneck_scale = nn.Parameter(
                torch.tensor(
                    _inverse_sigmoid(bottleneck_init / bottleneck_max),
                    dtype=torch.float32,
                )
            )
        else:
            self.graph_bottleneck_projection = None
            self.graph_bottleneck_scale_max = 0.0
            self.register_parameter("raw_graph_bottleneck_scale", None)

        if variant == "simple":
            dropout = float(model_config.get("dropout", 0.10))
            groups = int(model_config.get("group_norm_groups", 8))
            block_kind = str(model_config.get("residual_block", "efficient"))
            reliability_channels = len(self.reliability_spec.names)
            self.sar_encoder = SARHydrologyEncoderV15_1Simple(
                band_spec.channels("s1_t1"),
                band_spec.channels("s1_change"),
                int(model_config.get("s1_qa_channels", 2)),
                reliability_channels,
                self._channels,
                dropout,
                groups,
                block_kind,
                band_spec.channels("s1_conditioning"),
                float(model_config.get("pre_context_alpha_init", 0.10)),
                self.deduplicated_sar_reliability,
                bool(model_config.get("absolute_sar_shortcut_enabled", False)),
                float(model_config.get("absolute_sar_shortcut_init", 0.03)),
                float(model_config.get("absolute_sar_shortcut_max", 0.10)),
            )
            self.fusion = S1TerrainResidualFusionV15_1(
                self._channels,
                reliability_channels,
                groups=groups,
                terrain_alpha_init=float(model_config.get("terrain_mix_init", 0.05)),
                terrain_alpha_max=float(model_config.get("terrain_alpha_max", 1.0)),
                deduplicated_reliability=self.deduplicated_sar_reliability,
            )
            widths = model_config.get("decoder_widths", [96, 64, 48, 32])
            self.decoder = SARHydroDecoder(
                self._channels,
                dropout,
                groups,
                block_kind,
                widths,
                int(model_config.get("auxiliary_count", 1)),
                int(model_config.get("auxiliary_stage", 0)),
                change_injection_scale=0.0,
            )

    def graph_identity(self) -> dict[str, Any]:
        if self.variant == "corrected":
            return super().graph_identity()
        identity = self.graph.graph_identity(self._last_graph_feature_shape)
        identity["bottleneck_propagation_enabled"] = self.graph_bottleneck_enabled
        if self.graph_bottleneck_enabled:
            identity["bottleneck_residual_scale"] = float(
                self.graph_bottleneck_scale.detach().cpu()
            )
            identity["bottleneck_residual_scale_max"] = self.graph_bottleneck_scale_max
        return identity

    @property
    def graph_bottleneck_scale(self) -> torch.Tensor:
        if self.raw_graph_bottleneck_scale is None:
            raise RuntimeError("graph bottleneck propagation is disabled")
        return self.graph_bottleneck_scale_max * torch.sigmoid(
            self.raw_graph_bottleneck_scale
        )

    @staticmethod
    def _disabled_graph_diagnostics(reference: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = reference.sum() * 0.0
        return {
            "gate_mean": zero,
            "valid_edge_fraction": zero,
            "static_topographic_affinity_mean": zero,
            "observation_confidence_mean": zero,
            "latent_compatibility_mean": zero,
            "kan_coefficient_magnitude": zero,
            "kan_coefficient_smoothness": zero,
            "kan_monotonicity": zero,
            "kan_curve_smoothness": zero,
            "gamma_mean": zero,
            "graph_gamma_mean": zero,
            "graph_update_rms_ratio": zero,
            "topographic_kan_logit_mean": zero,
            "topographic_affinity_mean": zero,
            "observation_amplitude_mean": zero,
            "final_graph_gate_mean": zero,
            "graph_input_rms": zero,
            "graph_update_rms": zero,
            "spline_output_rms": zero,
            "base_output_rms": zero,
            "spline_base_rms_ratio": zero,
            "knot_boundary_saturation_fraction": zero,
        }

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        if self.variant == "corrected":
            return super().forward(inputs)
        forbidden = S1_V15_FORBIDDEN_INPUTS.intersection(inputs)
        forbidden.update(key for key in inputs if str(key).startswith("s2_"))
        if forbidden:
            raise ValueError(f"Forbidden non-S1 inputs: {sorted(forbidden)}")
        missing = S1_V15_REQUIRED_INPUTS.difference(inputs)
        if missing:
            raise KeyError(f"Missing PA-HydroKAN-S1-V15.1 inputs: {sorted(missing)}")
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
            inputs["s1_t1"],
            inputs["s1_t2"],
            inputs["s1_change"],
            inputs["s1_qa"],
            inputs["reliability"],
            inputs["s1_valid"],
            conditioning,
            dict(branch_validity),
            reliability_features,
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
        fused = list(fused)
        graph_feature = self.graph_input_projection(fused[2])
        self._last_graph_feature_shape = tuple(int(value) for value in graph_feature.shape[-2:])
        if self.graph_enabled:
            graph_output, graph_diagnostics = self.graph(
                graph_feature,
                physical,
                inputs["s1_event_support"],
                feature_stride=self.actual_graph_feature_stride,
            )
            # Keep the surrounding SAR/terrain feature as the residual identity;
            # only the graph *update* traverses the return projection.
            graph_delta = self.graph_output_projection(graph_output - graph_feature)
            fused[2] = fused[2] + graph_delta
            if self.graph_bottleneck_enabled:
                if self.graph_bottleneck_projection is None:
                    raise RuntimeError("graph bottleneck projection is unexpectedly absent")
                bottleneck_input = fused[3]
                source_height, source_width = graph_delta.shape[-2:]
                target_height, target_width = bottleneck_input.shape[-2:]
                if (
                    source_height % target_height != 0
                    or source_width % target_width != 0
                ):
                    raise RuntimeError(
                        "Graph-B requires an integer 1/4-to-bottleneck downsample; "
                        f"got graph={tuple(graph_delta.shape[-2:])}, "
                        f"bottleneck={tuple(bottleneck_input.shape[-2:])}"
                    )
                # The architecture uses an exact 1/4-to-1/8 integer-scale path.
                # Explicit average pooling is deterministic on CUDA, unlike the
                # adaptive pooling backward used by the initial implementation.
                pooled_delta = F.avg_pool2d(
                    graph_delta,
                    kernel_size=(source_height // target_height, source_width // target_width),
                    stride=(source_height // target_height, source_width // target_width),
                )
                bottleneck_delta = self.graph_bottleneck_projection(pooled_delta)
                scaled_bottleneck_delta = self.graph_bottleneck_scale * bottleneck_delta
                fused[3] = bottleneck_input + scaled_bottleneck_delta
                graph_diagnostics.update(
                    {
                        "graph_bottleneck_scale": self.graph_bottleneck_scale,
                        "graph_bottleneck_residual_input_rms_ratio": (
                            scaled_bottleneck_delta.float().square().mean().sqrt()
                            / bottleneck_input.float().square().mean().sqrt().clamp_min(1.0e-6)
                        ),
                    }
                )
            else:
                zero = graph_feature.sum() * 0.0
                graph_diagnostics.update(
                    {
                        "graph_bottleneck_scale": zero,
                        "graph_bottleneck_residual_input_rms_ratio": zero,
                    }
                )
        else:
            graph_diagnostics = self._disabled_graph_diagnostics(graph_feature)
            zero = graph_feature.sum() * 0.0
            graph_diagnostics.update(
                {
                    "graph_bottleneck_scale": zero,
                    "graph_bottleneck_residual_input_rms_ratio": zero,
                }
            )
        bottleneck = self.context(fused[-1])
        decoded, auxiliaries, decoder_gates = self.decoder(
            bottleneck,
            fused,
            terrain,
            physical["dem_valid_fractions"],
            inputs["s1_event_support"],
            sar_diagnostics["change_evidence"],
        )
        outputs = self.heads(decoded)
        if self.event_depth_scale is not None:
            scale, log_scale = self.event_depth_scale(
                bottleneck, inputs["dem_valid"] * inputs["s1_event_support"]
            )
            conditional = outputs["conditional_depth"] * scale
            outputs["conditional_depth"] = conditional
            outputs["positive_depth"] = conditional
            if "support_probability" in outputs:
                support = outputs["support_probability"]
                expected = conditional * (
                    self.heads.support_floor + (1.0 - self.heads.support_floor) * support
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
        outputs.update(
            {
                "auxiliary_depths": auxiliaries,
                "decoder_gates": decoder_gates,
                "fusion_diagnostics": fusion_diagnostics,
                "sar_diagnostics": sar_diagnostics,
                "graph_diagnostics": graph_diagnostics,
                "physical_features": physical,
            }
        )
        return outputs


def build_pa_hydrokan_s1_v15_1(config: Mapping[str, Any]) -> PAHydroKANS1V15_1:
    """Build a V15.1 model from a fully resolved S1-only config."""

    if "model" not in config:
        raise ValueError("PA-HydroKAN-S1-V15.1 builder requires the full resolved config")
    input_spec = ModelInputSpec.from_config(config)
    if not input_spec.is_s1_only:
        raise ValueError("PA-HydroKAN-S1-V15.1 requires dataset.input_mode='s1_terrain'")
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    raw_names = tuple(str(value) for value in contract.group("terrain")["band_descriptions"])
    model_config = dict(config["model"])
    if "kan_regularization_enabled" not in model_config:
        model_config["kan_regularization_enabled"] = float(
            config.get("loss", {}).get("lambda_kan", 0.0)
        ) != 0.0
    return PAHydroKANS1V15_1(model_config, band_spec, raw_names, input_spec)

"""Production PA-HydroKAN SAR-and-terrain model."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from datasets.band_selection import BandSpec, resolve_band_spec
from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec
from datasets.reliability_spec import ReliabilitySpec
from models.hydro_edge_kan import HydroEdgeKAN
from models.s1_hydrology_backbone import (
    HydrologyContext,
    SARHydrologyEncoder,
    S1HydrologyFusion,
    SARReliabilityConditioner,
)
from models.sar_hydro_decoder import SARHydroDecoder
from models.task_head import TaskHead
from models.terrain_features import TerrainFeaturePyramid


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
    """Predict conditional positive depth and a detached uncertainty scale."""

    def __init__(
        self,
        channels: int,
        groups: int,
        *,
        epsilon: float,
        maximum: float,
        depth_initialization_bias: float,
        uncertainty_initial_scale_m: float,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0 or maximum <= 0.0:
            raise ValueError("uncertainty epsilon and maximum must be positive")
        if not epsilon < uncertainty_initial_scale_m < maximum + epsilon:
            raise ValueError("uncertainty_initial_scale_m is outside the supported range")
        self.depth_head = TaskHead(channels, groups)
        self.uncertainty_head = TaskHead(channels, groups)
        self.epsilon = float(epsilon)
        self.maximum = float(maximum)
        self.depth_output_semantics = "conditional_positive"
        depth_final = self.depth_head.trunk[-1]
        uncertainty_final = self.uncertainty_head.trunk[-1]
        assert isinstance(depth_final, nn.Conv2d)
        assert isinstance(uncertainty_final, nn.Conv2d)
        nn.init.constant_(depth_final.bias, float(depth_initialization_bias))
        ratio = (float(uncertainty_initial_scale_m) - self.epsilon) / self.maximum
        nn.init.constant_(uncertainty_final.bias, _logit(ratio))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        depth = F.softplus(self.depth_head(features)) + self.epsilon
        scale = self.epsilon + self.maximum * torch.sigmoid(
            self.uncertainty_head(features.detach())
        )
        return {
            "conditional_depth": depth,
            "positive_depth": depth,
            "expected_depth": depth,
            "depth": depth,
            "uncertainty_scale": scale,
        }

    def set_depth_output_semantics(self, value: str) -> None:
        if value != "conditional_positive":
            raise ValueError("production inference supports conditional_positive output only")
        self.depth_output_semantics = value


class PAHydroKAN(nn.Module):
    """PA-HydroKAN: SAR-first, terrain-aware conditional-depth estimator."""

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
        channels = [int(value) for value in model_config["channels"]]
        if len(channels) != 4 or any(value <= 0 for value in channels):
            raise ValueError("PAHydroKAN requires four positive encoder scales")
        dropout = float(model_config["dropout"])
        groups = int(model_config["group_norm_groups"])
        block_kind = str(model_config["residual_block"])
        reliability_channels = len(self.reliability_spec.names)
        self.reliability_conditioner = SARReliabilityConditioner(
            reliability_channels, channels, groups=groups
        )
        self.sar_encoder = SARHydrologyEncoder(
            band_spec.channels("s1_t1"),
            band_spec.channels("s1_change"),
            channels,
            dropout=dropout,
            groups=groups,
            block_kind=block_kind,
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
            block_kind,
        )
        self.fusion = S1HydrologyFusion(
            channels,
            dropout=dropout,
            groups=groups,
            block_kind=block_kind,
            terrain_mix_init=float(model_config["terrain_mix_init"]),
            terrain_alpha_max=float(model_config["terrain_alpha_max"]),
        )
        self.context = HydrologyContext(channels[-1], groups, dropout=0.05)
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
            latent_compatibility_enabled=bool(model_config["latent_compatibility_enabled"]),
            diagnostics_enabled=bool(model_config.get("diagnostics_enabled", False)),
        )
        widths = [int(value) for value in model_config["decoder_widths"]]
        self.decoder = SARHydroDecoder(
            channels,
            dropout,
            groups,
            block_kind,
            widths,
            int(model_config["auxiliary_count"]),
            int(model_config.get("auxiliary_stage", 0)),
        )
        self.heads = PAHydroKANHeads(
            widths[-1],
            groups,
            epsilon=float(model_config["uncertainty_epsilon"]),
            maximum=float(model_config["uncertainty_maximum"]),
            depth_initialization_bias=float(model_config["depth_initialization_bias"]),
            uncertainty_initial_scale_m=float(model_config["uncertainty_initial_scale_m"]),
        )
        self._last_graph_feature_shape: tuple[int, int] | None = None

    def graph_identity(self) -> dict[str, Any]:
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
        reliability_features = self.reliability_conditioner(
            inputs["reliability"], branch_validity
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
        )
        bottleneck = self.context(fused[-1])
        self._last_graph_feature_shape = tuple(int(value) for value in bottleneck.shape[-2:])
        bottleneck, graph_diagnostics = self.graph(
            bottleneck,
            physical,
            inputs["s1_event_support"],
            sar_diagnostics["quality_gates"][-1],
            feature_stride=8,
        )
        decoded, auxiliaries, decoder_gates = self.decoder(
            bottleneck,
            fused,
            terrain,
            physical["dem_valid_fractions"],
            inputs["s1_event_support"],
            sar_diagnostics["change_evidence"],
        )
        outputs = self.heads(decoded)
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


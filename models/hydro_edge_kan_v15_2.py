"""V15.2 HydroEdgeKAN with feature-wise terrain mapping and audited edges.

The graph remains a single 1/4-resolution, eight-neighbour compatibility
module.  This revision changes only the terrain-edge representation: each KAN
input has its own robust temperature, eligibility is separated from continuous
observation confidence, and path barriers are pooled without invalid zeros.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from models.hydro_edge_kan_v15_1 import (
    DEFAULT_EDGE_FEATURE_NAMES,
    HydroEdgeKANV15_1,
)
from models.kan_layers import KANLinear
from models.terrain_features_v14 import path_barrier_proxy
from models.terrain_graph_kan import DIRECTIONS, _masked_pool


def _feature_temperature_values(
    feature_names: Sequence[str],
    mapping_temperatures: Mapping[str, float] | Sequence[float] | None,
    fallback: float,
) -> list[float]:
    """Resolve strictly positive temperatures in the declared feature order."""

    names = tuple(str(name) for name in feature_names)
    if mapping_temperatures is None:
        values = [float(fallback)] * len(names)
    elif isinstance(mapping_temperatures, Mapping):
        missing = [name for name in names if name not in mapping_temperatures]
        unknown = set(mapping_temperatures).difference(names)
        if missing or unknown:
            raise ValueError(
                "graph_mapping_temperatures must match graph_edge_feature_names; "
                f"missing={missing}, unknown={sorted(unknown)}"
            )
        values = [float(mapping_temperatures[name]) for name in names]
    elif isinstance(mapping_temperatures, Sequence) and not isinstance(
        mapping_temperatures, (str, bytes)
    ):
        values = [float(value) for value in mapping_temperatures]
        if len(values) != len(names):
            raise ValueError(
                "graph_mapping_temperatures length must equal graph_edge_feature_names"
            )
    else:
        raise TypeError("graph_mapping_temperatures must be a mapping, sequence, or None")
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0.0 for value in values):
        raise ValueError("all graph mapping temperatures must be finite and positive")
    return values


class HydroEdgeKANV15_2(HydroEdgeKANV15_1):
    """Single Graph/KAN layer with feature-wise mapping and edge semantics.

    The parent preserves the tested message/gate formulation and all
    evaluation-only counterfactual switches.  V15.2 replaces only the input
    mapping, barrier pooling, edge eligibility, and effective-spline penalty.
    """

    def __init__(
        self,
        channels: int,
        *,
        heads: int = 2,
        grid_size: int = 4,
        spline_order: int = 3,
        graph_feature_stride: int = 4,
        terrain_pixel_size_m: float = 20.0,
        edge_feature_names: Sequence[str] = DEFAULT_EDGE_FEATURE_NAMES,
        edge_stats_path: str | None = None,
        mapping_temperature: float = 1.0,
        mapping_temperatures: Mapping[str, float] | Sequence[float] | None = None,
        edge_minimum_dem_fraction: float = 0.5,
        edge_minimum_sar_fraction: float = 0.5,
        edge_minimum_barrier_valid_fraction: float = 0.5,
        gamma_init: float = 0.03,
        gamma_max: float = 0.25,
        base_scale_init: float = 0.35,
        spline_scale_init: float = 1.0,
        latent_scale_init: float = 0.25,
        symmetric_static_prior_enabled: bool = False,
        diagnostics_enabled: bool = False,
        regularization_enabled: bool = False,
        message_mode: str = "linear",
        message_scale: float = 1.0,
        extreme_preservation_enabled: bool = False,
        extreme_preservation_scale: float = 1.0,
    ) -> None:
        if not 0.0 < edge_minimum_dem_fraction <= 1.0:
            raise ValueError("edge_minimum_dem_fraction must lie in (0, 1]")
        if not 0.0 < edge_minimum_sar_fraction <= 1.0:
            raise ValueError("edge_minimum_sar_fraction must lie in (0, 1]")
        if not 0.0 < edge_minimum_barrier_valid_fraction <= 1.0:
            raise ValueError(
                "edge_minimum_barrier_valid_fraction must lie in (0, 1]"
            )
        names = tuple(str(name) for name in edge_feature_names)
        temperatures = _feature_temperature_values(
            names, mapping_temperatures, float(mapping_temperature)
        )
        super().__init__(
            channels,
            heads=heads,
            grid_size=grid_size,
            spline_order=spline_order,
            graph_feature_stride=graph_feature_stride,
            terrain_pixel_size_m=terrain_pixel_size_m,
            edge_feature_names=names,
            edge_stats_path=edge_stats_path,
            mapping_temperature=float(mapping_temperature),
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            base_scale_init=base_scale_init,
            spline_scale_init=spline_scale_init,
            latent_scale_init=latent_scale_init,
            symmetric_static_prior_enabled=symmetric_static_prior_enabled,
            diagnostics_enabled=diagnostics_enabled,
            regularization_enabled=regularization_enabled,
            message_mode=message_mode,
            message_scale=message_scale,
            extreme_preservation_enabled=extreme_preservation_enabled,
            extreme_preservation_scale=extreme_preservation_scale,
        )
        # V15.2 explicitly removes the unused LayerNorm from its fixed-scaling
        # KAN.  This intentionally produces a new state-dict schema rather than
        # silently carrying dormant parameters from V15.1.
        self.edge_kan = KANLinear(
            len(names),
            self.heads,
            grid_size=grid_size,
            spline_order=spline_order,
            normalization="explicit_fixed_scaling",
            input_bounding="prebounded",
            base_path="linear",
            base_scale_init=base_scale_init,
            spline_scale_init=spline_scale_init,
            learnable_base_scale=True,
            learnable_spline_scale=True,
            retain_explicit_layernorm=False,
        )
        self.register_buffer(
            "feature_temperatures",
            torch.tensor(temperatures, dtype=torch.float32).view(1, 1, len(names), 1, 1),
        )
        self.edge_minimum_dem_fraction = float(edge_minimum_dem_fraction)
        self.edge_minimum_sar_fraction = float(edge_minimum_sar_fraction)
        self.edge_minimum_barrier_valid_fraction = float(
            edge_minimum_barrier_valid_fraction
        )
        self._last_barrier_valid_fraction: torch.Tensor | None = None

    def graph_identity(self, feature_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        identity = super().graph_identity(feature_shape)
        identity.update(
            {
                "version": "v15_2_featurewise_mapping",
                "mapping_temperature_by_feature": {
                    name: float(self.feature_temperatures[0, 0, index, 0, 0].detach().cpu())
                    for index, name in enumerate(self.edge_feature_names)
                },
                "edge_eligibility": {
                    "minimum_dem_fraction": self.edge_minimum_dem_fraction,
                    "minimum_sar_fraction": self.edge_minimum_sar_fraction,
                    "minimum_barrier_valid_fraction": self.edge_minimum_barrier_valid_fraction,
                },
                "barrier_pooling": "masked_adaptive_mean",
                "barrier_statistic": "max",
                "kan_explicit_scaling_has_layernorm": False,
            }
        )
        return identity

    def map_edge_features(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply the sole robust, feature-wise bounded mapping before KAN."""

        robust = (raw - self.feature_centers.to(raw.dtype)) / self.feature_scales.to(raw.dtype)
        return torch.tanh(robust / self.feature_temperatures.to(raw.dtype))

    @staticmethod
    def _masked_directional_pool(
        values: torch.Tensor,
        valid: torch.Tensor,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool B,D,C,H,W values without treating invalid zeros as observations."""

        if values.shape != valid.shape or values.ndim != 5:
            raise ValueError("directional values and valid masks must have matching B,D,C,H,W shapes")
        batch, directions, channels, height, width = values.shape
        flat_values = values.reshape(batch * directions, channels, height, width)
        flat_valid = valid.reshape(batch * directions, channels, height, width)
        denominator = F.adaptive_avg_pool2d(flat_valid, size)
        pooled = F.adaptive_avg_pool2d(flat_values * flat_valid, size)
        pooled = pooled / denominator.clamp_min(1.0e-6)
        shape = (batch, directions, channels, *size)
        return pooled.reshape(shape), denominator.reshape(shape)

    def _edge_descriptors(
        self,
        physical: Mapping[str, torch.Tensor],
        sensor_valid: torch.Tensor,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dem_full = physical["dem_valid"]
        dem_fraction = F.adaptive_avg_pool2d(dem_full, size)
        sar_fraction = F.adaptive_avg_pool2d(sensor_valid, size)
        node_eligible = (
            (dem_fraction >= self.edge_minimum_dem_fraction)
            & (sar_fraction >= self.edge_minimum_sar_fraction)
        ).to(sar_fraction.dtype)
        elevation = _masked_pool(physical["physics_elevation"], dem_full, size)
        complexity = _masked_pool(physical["local_relief"], dem_full, size)
        neighbour_elevation, boundary = self._stack_roll(elevation)
        neighbour_complexity, _ = self._stack_roll(complexity)
        distance = elevation.new_tensor(self.neighbour_distances_m).view(
            1, len(DIRECTIONS), 1, 1, 1
        )
        signed_grade = (
            neighbour_elevation - elevation.unsqueeze(1)
        ) / distance.clamp_min(1.0e-6)
        absolute_grade = signed_grade.abs()

        barrier_full, barrier_path_valid = path_barrier_proxy(
            physical["dsm_elevation"],
            dem_full,
            self.graph_feature_stride,
            physical["z_ground_proxy"],
            statistic="max",
        )
        barrier, barrier_valid_fraction = self._masked_directional_pool(
            barrier_full, barrier_path_valid, size
        )
        local_complexity = 0.5 * (complexity.unsqueeze(1) + neighbour_complexity)
        raw_by_name = {
            "signed_grade": signed_grade,
            "absolute_grade": absolute_grade,
            "barrier_magnitude": barrier,
            "local_surface_complexity": local_complexity,
        }
        raw = torch.cat([raw_by_name[name] for name in self.edge_feature_names], dim=2)
        neighbour_node_eligible, _ = self._stack_roll(node_eligible)
        edge_eligible = (
            node_eligible.unsqueeze(1)
            * neighbour_node_eligible
            * boundary
            * (barrier_valid_fraction >= self.edge_minimum_barrier_valid_fraction).to(
                node_eligible.dtype
            )
        )
        neighbour_sar, _ = self._stack_roll(sar_fraction)
        neighbour_dem, _ = self._stack_roll(dem_fraction)
        sar_pair_fraction = torch.sqrt(
            (sar_fraction.unsqueeze(1).clamp(0.0, 1.0)
             * neighbour_sar.clamp(0.0, 1.0)).clamp_min(0.0)
        )
        dem_pair_fraction = torch.sqrt(
            (dem_fraction.unsqueeze(1).clamp(0.0, 1.0)
             * neighbour_dem.clamp(0.0, 1.0)).clamp_min(0.0)
        )
        edge_confidence = sar_pair_fraction * dem_pair_fraction
        self._last_barrier_valid_fraction = barrier_valid_fraction
        return raw, self.map_edge_features(raw), edge_eligible, edge_confidence

    def spline_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Regularize the actual scaled spline coefficients used in forward."""

        effective = self.edge_kan.effective_spline_coefficients
        # Group amplitude is averaged over feature/head pairs so every univariate
        # spline contributes equally despite having the same knot count today.
        magnitude = (effective.square().mean(dim=-1) + 1.0e-12).sqrt().mean()
        if effective.shape[-1] < 3:
            return magnitude, effective.sum() * 0.0
        curvature = (
            effective[..., 2:]
            - 2.0 * effective[..., 1:-1]
            + effective[..., :-2]
        ).square().mean()
        return magnitude, curvature

    def forward(
        self,
        features: torch.Tensor,
        physical: Mapping[str, torch.Tensor],
        sensor_valid: torch.Tensor,
        *,
        feature_stride: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        output, diagnostics = super().forward(
            features, physical, sensor_valid, feature_stride=feature_stride
        )
        if self.counterfactual_mode == "graph_off":
            return output, diagnostics
        barrier_valid_fraction = self._last_barrier_valid_fraction
        if barrier_valid_fraction is not None:
            diagnostics["barrier_valid_fraction_mean"] = barrier_valid_fraction.mean()
        diagnostics["edge_eligible_fraction"] = diagnostics["valid_edge_fraction"]
        diagnostics["edge_confidence_mean"] = diagnostics["observation_amplitude_mean"]
        return output, diagnostics

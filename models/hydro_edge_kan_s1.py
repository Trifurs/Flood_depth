"""S1-only topographic Edge-KAN aggregation.

This graph is a static topographic affinity operator.  It is not a hydraulic
solver and does not use a second sensor or a cross-sensor time difference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from models.kan_layers import KANLinear
from models.terrain_features_v14 import path_barrier_proxy
from models.terrain_graph_kan import DIRECTIONS, _masked_pool, _roll_with_boundary_mask


S1_EDGE_FEATURE_NAMES = (
    "signed_dz",
    "edge_slope",
    "relative_height",
    "path_barrier",
    "local_relief",
    "distance",
)


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(float(value), 1.0e-6)))


class HydroEdgeKANS1(nn.Module):
    """Eight-neighbour S1-only graph with static terrain affinity."""

    def __init__(
        self,
        channels: int,
        heads: int = 2,
        grid_size: int = 4,
        spline_order: int = 3,
        graph_scale: int = 4,
        terrain_pixel_size_m: float = 20.0,
        feature_centers: Sequence[float] | None = None,
        feature_scales: Sequence[float] | None = None,
        gamma_init_effective: float = 0.02,
        gamma_max: float = 0.25,
        latent_compatibility_enabled: bool = True,
        diagnostic_mode: bool = False,
    ) -> None:
        super().__init__()
        if heads not in {2, 4} or channels % heads:
            raise ValueError("HydroEdgeKANS1 heads must be 2 or 4 and divide channels")
        if graph_scale not in {4, 8}:
            raise ValueError("HydroEdgeKANS1 graph_scale must be 4 or 8")
        if terrain_pixel_size_m <= 0 or gamma_max <= 0 or not 0 <= gamma_init_effective < gamma_max:
            raise ValueError("invalid graph scale, pixel size, or gamma bounds")
        centers = list(feature_centers or [0.0] * 6)
        scales = list(feature_scales or [0.10, 0.10, 1.0, 1.0, 1.0, 1.0])
        if len(centers) != 6 or len(scales) != 6 or any(float(value) <= 0 for value in scales):
            raise ValueError("S1 graph feature centers/scales must contain six valid values")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_channels = channels // heads
        self.graph_scale = int(graph_scale)
        self.graph_pixel_size_m = float(graph_scale) * float(terrain_pixel_size_m)
        self.gamma_max = float(gamma_max)
        self.latent_compatibility_enabled = bool(latent_compatibility_enabled)
        self.diagnostic_mode = bool(diagnostic_mode)
        self.edge_feature_names = S1_EDGE_FEATURE_NAMES
        self.register_buffer(
            "feature_centers",
            torch.tensor(centers, dtype=torch.float32).view(1, 1, 6, 1, 1),
        )
        self.register_buffer(
            "feature_scales",
            torch.tensor(scales, dtype=torch.float32).view(1, 1, 6, 1, 1),
        )
        self.edge_kan = nn.ModuleList(
            [
                KANLinear(
                    6, 1, grid_size, spline_order,
                    normalization="explicit_fixed_scaling",
                    input_bounding="prebounded",
                    base_path="none",
                    base_scale_init=0.0,
                    spline_scale_init=1.0,
                    learnable_base_scale=False,
                    learnable_spline_scale=True,
                    zero_output_init=True,
                )
                for _ in range(heads)
            ]
        )
        self.latent_projection = nn.ModuleList(
            [nn.Conv2d(channels, self.head_channels, 1) for _ in range(heads)]
        )
        self.message = nn.ModuleList(
            [nn.Conv2d(channels, self.head_channels, 1, bias=False) for _ in range(heads)]
        )
        self.latent_compatibility = nn.ModuleList(
            [nn.Conv2d(2, 1, 1) for _ in range(heads)]
        )
        for projection in self.latent_compatibility:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.prior_raw_scales = nn.Parameter(
            torch.tensor([_inverse_softplus(value) for value in (0.5, 0.25, 0.10, 0.15, 0.10, 0.02)])
        )
        self.prior_bias = nn.Parameter(torch.zeros(heads))
        self.raw_gamma = nn.Parameter(
            torch.full(
                (heads,),
                _inverse_sigmoid(max(gamma_init_effective, 1.0e-5) / gamma_max),
            )
        )
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    @property
    def prior_scales(self) -> torch.Tensor:
        return F.softplus(self.prior_raw_scales)

    def spline_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = torch.cat(
            [layer.spline_coefficients.reshape(-1) for layer in self.edge_kan]
        )
        # Zero-initialized residual splines must have a finite derivative at the
        # identity point; sqrt(mean(x^2)) without epsilon yields 0/0 there.
        magnitude = (coefficients.square().mean() + 1.0e-12).sqrt()
        smoothness = torch.cat(
            [
                (layer.spline_coefficients[..., 2:] - 2.0 * layer.spline_coefficients[..., 1:-1] + layer.spline_coefficients[..., :-2]).reshape(-1)
                for layer in self.edge_kan
            ]
        ).square().mean()
        return magnitude, smoothness

    def function_regularization(self, points: int = 33) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        grid = torch.linspace(-1.0, 1.0, points, device=self.feature_centers.device)
        monotonicity, smoothness, diagnostics = [], [], {}
        for feature_index, name in enumerate(("signed_dz", "edge_slope")):
            probe = torch.zeros(points, 6, device=grid.device)
            probe[:, feature_index] = grid
            values = torch.stack([layer(probe).squeeze(-1) for layer in self.edge_kan])
            first = values[:, 1:] - values[:, :-1]
            second = values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
            monotonicity.append(F.relu(first).mean())
            smoothness.append(second.square().mean())
            diagnostics[f"{name}_violation_fraction"] = (first > 0).to(values.dtype).mean()
        return torch.stack(monotonicity).mean(), torch.stack(smoothness).mean(), diagnostics

    @staticmethod
    def _stack_roll(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rolled, boundaries = zip(
            *[_roll_with_boundary_mask(value, dy, dx) for dy, dx in DIRECTIONS]
        )
        return torch.stack(rolled, dim=1), torch.stack(boundaries, dim=1)

    def _descriptors(
        self,
        physical: Mapping[str, torch.Tensor],
        size: tuple[int, int],
        sensor_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dem_full = physical["dem_valid"]
        dem_fraction = F.adaptive_avg_pool2d(dem_full, size)
        sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, size)
        node_valid = (dem_fraction > 0.5).to(sensor_fraction.dtype) * (sensor_fraction > 0).to(sensor_fraction.dtype)
        z = _masked_pool(physical["physics_elevation"], dem_full, size)
        relative = _masked_pool(physical["z_relative"], dem_full, size)
        relief = _masked_pool(physical["local_relief"], dem_full, size)
        neighbour_z, boundary = self._stack_roll(z)
        neighbour_relative, _ = self._stack_roll(relative)
        neighbour_relief, _ = self._stack_roll(relief)
        distance_values = [self.graph_pixel_size_m * math.sqrt(dx * dx + dy * dy) for dy, dx in DIRECTIONS]
        distance = z.new_tensor(distance_values).view(1, len(DIRECTIONS), 1, 1, 1)
        signed_dz = (neighbour_z - z.unsqueeze(1)) / distance.clamp_min(1.0e-6)
        edge_slope = signed_dz.abs()
        relative_height = 0.5 * (relative.unsqueeze(1) + neighbour_relative)
        path_barrier, path_valid = path_barrier_proxy(
            physical["dsm_elevation"], dem_full, self.graph_scale,
            physical["z_ground_proxy"], statistic="max", quantile=0.9,
        )
        path_barrier = torch.stack(
            [F.adaptive_avg_pool2d(path_barrier[:, index], size) for index in range(len(DIRECTIONS))],
            dim=1,
        )
        path_valid = torch.stack(
            [F.adaptive_avg_pool2d(path_valid[:, index], size) for index in range(len(DIRECTIONS))],
            dim=1,
        )
        relief_pair = 0.5 * (relief.unsqueeze(1) + neighbour_relief)
        distance_normalized = (
            distance / max(self.graph_pixel_size_m, 1.0)
        ).expand(z.shape[0], -1, -1, size[0], size[1])
        raw = torch.cat(
            (signed_dz, edge_slope, relative_height, path_barrier, relief_pair, distance_normalized),
            dim=2,
        )
        descriptor = (
            (raw - self.feature_centers.to(raw.dtype)) / self.feature_scales.to(raw.dtype)
        ).clamp(-4.0, 4.0)
        valid_edge = node_valid.unsqueeze(1) * self._stack_roll(node_valid)[0] * boundary * path_valid
        return descriptor, raw, valid_edge, dem_fraction, sensor_fraction

    def forward(
        self,
        features: torch.Tensor,
        physical: Mapping[str, torch.Tensor],
        sensor_valid: torch.Tensor,
        observation_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        size = features.shape[-2:]
        descriptor, raw, valid_edge, dem_fraction, sensor_fraction = self._descriptors(
            physical, size, sensor_valid
        )
        if observation_confidence is None:
            observation_confidence = sensor_valid
        confidence = F.adaptive_avg_pool2d(observation_confidence, size)
        neighbour_confidence, _ = self._stack_roll(confidence)
        neighbour_sensor, _ = self._stack_roll(sensor_fraction)
        observation = confidence.unsqueeze(1) * neighbour_confidence * sensor_fraction.unsqueeze(1) * neighbour_sensor
        latent = torch.stack([projection(features) for projection in self.latent_projection], dim=1)
        neighbour_latent = torch.stack(
            [self._stack_roll(latent[:, head])[0] for head in range(self.heads)], dim=2
        )
        neighbour_features, _ = self._stack_roll(features)
        messages, affinities, gates = [], [], []
        prior = -(raw * self.prior_scales.view(1, 1, 6, 1, 1)).sum(dim=2, keepdim=True)
        flat_descriptor = descriptor.permute(0, 1, 3, 4, 2).reshape(-1, 6)
        for head, layer in enumerate(self.edge_kan):
            residual = layer(flat_descriptor).reshape(features.shape[0], len(DIRECTIONS), 1, *size)
            total = prior + self.prior_bias[head] + residual
            affinity = torch.sigmoid(total) * valid_edge
            latent_difference = neighbour_latent[:, :, head] - latent[:, head].unsqueeze(1)
            pair = torch.stack(
                (latent_difference.abs().mean(2), (latent[:, head].unsqueeze(1) * neighbour_latent[:, :, head]).mean(2)),
                dim=2,
            )
            compatibility = (
                torch.sigmoid(self.latent_compatibility[head](pair.reshape(-1, 2, *size))).reshape(features.shape[0], len(DIRECTIONS), 1, *size)
                if self.latent_compatibility_enabled
                else torch.ones_like(affinity)
            )
            gate = affinity * observation * compatibility
            message_input = (neighbour_features - features.unsqueeze(1)).reshape(-1, features.shape[1], *size)
            message = self.message[head](message_input).reshape(features.shape[0], len(DIRECTIONS), self.head_channels, *size)
            denominator = gate.sum(1, keepdim=True).clamp_min(1.0e-6)
            weighted = (gate * message).sum(1, keepdim=True) / denominator
            confidence_weight = gate.sum(1, keepdim=True) / valid_edge.sum(1, keepdim=True).clamp_min(1.0)
            messages.append(weighted.squeeze(1) * confidence_weight.squeeze(1) * self.gamma[head])
            affinities.append(affinity)
            gates.append(gate)
        update = self.output(torch.cat(messages, dim=1))
        output = features + update
        coefficient_magnitude, coefficient_smoothness = self.spline_regularization()
        monotonicity, curve_smoothness, monotonicity_diagnostics = self.function_regularization()
        gate_stack = torch.cat(gates, dim=2)
        affinity_stack = torch.cat(affinities, dim=2)
        diagnostics: dict[str, Any] = {
            "gate_mean": gate_stack.mean(),
            "gate_std": gate_stack.float().std(unbiased=False),
            "valid_edge_fraction": valid_edge.mean(),
            "static_topographic_affinity_mean": affinity_stack.sum() / valid_edge.sum().clamp_min(1.0),
            "observation_confidence_mean": observation.mean(),
            "latent_compatibility_mean": torch.stack([value.mean() for value in gates]).mean(),
            "kan_coefficient_magnitude": coefficient_magnitude,
            "kan_coefficient_smoothness": coefficient_smoothness,
            "kan_monotonicity": monotonicity,
            "kan_curve_smoothness": curve_smoothness,
            "gamma_mean": self.gamma.mean(),
            "gamma_values": self.gamma,
            "graph_update_rms_ratio": update.square().mean().sqrt() / features.square().mean().sqrt().clamp_min(1.0e-6),
        }
        diagnostics.update(monotonicity_diagnostics)
        if self.diagnostic_mode:
            diagnostics.update({
                "edge_descriptors": descriptor.detach(),
                "raw_edge_descriptors": raw.detach(),
                "static_topographic_affinity": affinity_stack.detach(),
                "valid_edges": valid_edge.detach(),
            })
        return output, diagnostics

"""Hydro-v14 symmetric physical-prior Edge-KAN.

The graph is a KAN-gated latent feature aggregation operator.  Its static
topographic affinity is symmetric by construction and is not a water-flow or
mass-conservation solver.
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


HYDRO_V14_EDGE_FEATURE_NAMES = (
    "absolute_edge_slope",
    "path_barrier_proxy",
    "local_relief_pair",
)


def _inverse_sigmoid(value: float) -> float:
    value = min(max(value, 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(float(value), 1e-6)))


class HydroEdgeKANV14(nn.Module):
    """Eight-neighbour graph using three symmetric topographic descriptors."""

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
        base_path: str = "none",
        base_scale_init: float = 0.0,
        spline_scale_init: float = 1.0,
        learnable_base_scale: bool = False,
        learnable_spline_scale: bool = True,
        zero_residual_init: bool = True,
        edge_gate_type: str = "kan",
        prior_slope_init: float = 1.0,
        prior_barrier_init: float = 0.5,
        prior_relief_init: float = 0.25,
        prior_bias_init: float = 0.0,
        gamma_init_effective: float = 0.02,
        gamma_max: float = 0.25,
        latent_compatibility_enabled: bool = True,
        path_statistic: str = "max",
        path_quantile: float = 0.9,
        diagnostic_mode: bool = False,
    ) -> None:
        super().__init__()
        if heads not in {2, 4} or channels % heads:
            raise ValueError("heads must be 2 or 4 and divide channels")
        if graph_scale not in {4, 8}:
            raise ValueError("Hydro-v14 graph_scale must be 4 or 8")
        if terrain_pixel_size_m <= 0 or gamma_max <= 0 or not 0 <= gamma_init_effective < gamma_max:
            raise ValueError("invalid graph scale, pixel size, or gamma bounds")
        if edge_gate_type not in {"kan", "mlp"}:
            raise ValueError("edge_gate_type must be kan or mlp")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_channels = channels // heads
        self.graph_scale = int(graph_scale)
        self.graph_pixel_size_m = float(graph_scale) * float(terrain_pixel_size_m)
        self.gamma_max = float(gamma_max)
        self.edge_gate_type = edge_gate_type
        self.edge_feature_names = HYDRO_V14_EDGE_FEATURE_NAMES
        self.path_statistic = path_statistic
        self.path_quantile = float(path_quantile)
        self.diagnostic_mode = bool(diagnostic_mode)
        self.latent_compatibility_enabled = bool(latent_compatibility_enabled)

        centers = list(feature_centers or [0.0, 0.0, 0.0])
        scales = list(feature_scales or [0.10, 1.0, 1.0])
        if len(centers) != 3 or len(scales) != 3 or any(float(value) <= 0 for value in scales):
            raise ValueError("Hydro-v14 edge centers/scales must contain three valid values")
        self.register_buffer("feature_centers", torch.tensor(centers, dtype=torch.float32).view(1, 1, 3, 1, 1))
        self.register_buffer("feature_scales", torch.tensor(scales, dtype=torch.float32).view(1, 1, 3, 1, 1))

        self.latent_projection = nn.ModuleList([
            nn.Conv2d(channels, self.head_channels, 1) for _ in range(self.heads)
        ])
        if edge_gate_type == "kan":
            self.edge_kan = KANLinear(
                3, 1, grid_size, spline_order,
                normalization="explicit_fixed_scaling", input_bounding="prebounded",
                base_path=base_path, base_scale_init=base_scale_init,
                spline_scale_init=spline_scale_init,
                learnable_base_scale=learnable_base_scale,
                learnable_spline_scale=learnable_spline_scale,
                zero_output_init=zero_residual_init,
            )
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))
            if zero_residual_init:
                nn.init.zeros_(self.edge_mlp[-1].weight)
                nn.init.zeros_(self.edge_mlp[-1].bias)

        self.prior_raw_scales = nn.Parameter(torch.tensor([
            _inverse_softplus(prior_slope_init),
            _inverse_softplus(prior_barrier_init),
            _inverse_softplus(prior_relief_init),
        ]))
        self.prior_bias = nn.Parameter(torch.full((self.heads,), float(prior_bias_init)))
        self.latent_compatibility = nn.ModuleList([nn.Conv2d(2, 1, 1) for _ in range(self.heads)])
        for projection in self.latent_compatibility:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.message = nn.ModuleList([
            nn.Conv2d(channels, self.head_channels, 1, bias=False) for _ in range(self.heads)
        ])
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        self.raw_gamma = nn.Parameter(torch.full((self.heads,), _inverse_sigmoid(max(gamma_init_effective, 1e-5) / gamma_max)))

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    @property
    def prior_scales(self) -> torch.Tensor:
        return F.softplus(self.prior_raw_scales)

    def spline_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.edge_gate_type != "kan":
            zero = self.output.weight.sum() * 0.0
            return zero, zero
        coefficients = self.edge_kan.spline_coefficients
        magnitude = coefficients.square().mean().sqrt()
        smoothness = (coefficients[..., 2:] - 2.0 * coefficients[..., 1:-1] + coefficients[..., :-2]).square().mean()
        return magnitude, smoothness

    def function_regularization(self, points: int = 33) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Regularize actual residual/prior functions on a fixed probe grid."""

        if self.edge_gate_type != "kan":
            zero = self.output.weight.sum() * 0.0
            return zero, zero, {}
        grid = torch.linspace(-1.0, 1.0, points, device=self.feature_centers.device, dtype=self.feature_centers.dtype)
        mono_terms, smooth_terms = [], []
        violation: dict[str, torch.Tensor] = {}
        for feature_index, feature_name in enumerate(self.edge_feature_names[:2]):
            probe = torch.zeros(points, 3, device=grid.device, dtype=grid.dtype)
            probe[:, feature_index] = grid
            residual = self.edge_kan(probe).squeeze(-1)
            prior = -self.prior_scales[feature_index] * grid
            total = prior + residual
            first = total[1:] - total[:-1]
            second = total[2:] - 2.0 * total[1:-1] + total[:-2]
            term = F.relu(first).mean()
            mono_terms.append(term)
            smooth_terms.append(second.square().mean())
            violation[f"{feature_name}_violation_fraction"] = (first > 0).to(total.dtype).mean()
        if not mono_terms:
            zero = self.output.weight.sum() * 0.0
            return zero, zero, violation
        return torch.stack(mono_terms).mean(), torch.stack(smooth_terms).mean(), violation

    @staticmethod
    def _stack_roll(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rolled, boundaries = zip(*[_roll_with_boundary_mask(value, dy, dx) for dy, dx in DIRECTIONS])
        return torch.stack(rolled, dim=1), torch.stack(boundaries, dim=1)

    def _descriptors(self, physical: Mapping[str, torch.Tensor], size: tuple[int, int], sensor_valid: torch.Tensor):
        dem_full = physical["dem_valid"]
        dem_fraction = F.adaptive_avg_pool2d(dem_full, size)
        sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, size)
        node_valid = (dem_fraction > 0.5).to(sensor_fraction.dtype) * (sensor_fraction > 0).to(sensor_fraction.dtype)
        z = _masked_pool(physical["physics_elevation"], dem_full, size)
        relief = _masked_pool(physical["local_relief"], dem_full, size)
        neighbour_z, boundary = self._stack_roll(z)
        neighbour_relief, _ = self._stack_roll(relief)
        neighbour_sensor, _ = self._stack_roll(sensor_fraction)
        neighbour_dem, _ = self._stack_roll(dem_fraction)
        distance_values = [self.graph_pixel_size_m * math.sqrt(dx * dx + dy * dy) for dy, dx in DIRECTIONS]
        distance = z.new_tensor(distance_values).view(1, len(DIRECTIONS), 1, 1, 1)
        abs_slope = (neighbour_z - z.unsqueeze(1)).abs() / distance.clamp_min(1e-6)

        dsm = physical["dsm_elevation"]
        path_barrier, path_valid = path_barrier_proxy(
            dsm, dem_full, self.graph_scale, physical["z_ground_proxy"],
            statistic=self.path_statistic, quantile=self.path_quantile,
        )
        path_barrier = torch.stack([F.adaptive_avg_pool2d(path_barrier[:, index], size) for index in range(len(DIRECTIONS))], dim=1)
        path_valid = torch.stack([F.adaptive_avg_pool2d(path_valid[:, index], size) for index in range(len(DIRECTIONS))], dim=1)
        relief_pair = 0.5 * (relief.unsqueeze(1) + neighbour_relief)
        raw_descriptor = torch.cat((abs_slope, path_barrier, relief_pair), dim=2)
        descriptor = (raw_descriptor - self.feature_centers.to(raw_descriptor.dtype)) / self.feature_scales.to(raw_descriptor.dtype)
        descriptor = descriptor.clamp(-4.0, 4.0)
        valid_edge = node_valid.unsqueeze(1) * self._stack_roll(node_valid)[0] * boundary * path_valid
        return descriptor, raw_descriptor, valid_edge, dem_fraction, sensor_fraction, node_valid

    def forward(
        self,
        features: torch.Tensor,
        physical: Mapping[str, torch.Tensor],
        reliability: torch.Tensor | None = None,
        modality_weights: torch.Tensor | None = None,
        sensor_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if sensor_valid is None:
            sensor_valid = torch.ones_like(physical["dem_valid"])
        size = features.shape[-2:]
        descriptor, raw_descriptor, valid_edge, dem_fraction, sensor_fraction, node_valid = self._descriptors(physical, size, sensor_valid)
        prior_base = (self.prior_scales.view(1, 1, 3, 1, 1) * raw_descriptor).sum(dim=2, keepdim=True)
        prior = self.prior_bias.view(1, 1, self.heads, 1, 1) - prior_base
        prior = prior.expand(-1, -1, self.heads, -1, -1)

        latent = torch.stack([projection(features) for projection in self.latent_projection], dim=1)
        neighbour_latent = torch.stack([self._stack_roll(latent[:, head])[0] for head in range(self.heads)], dim=2)
        latent_diff = neighbour_latent - latent.unsqueeze(1)
        sensor_neighbour = self._stack_roll(sensor_fraction)[0]
        dem_neighbour = self._stack_roll(dem_fraction)[0]
        observation = sensor_fraction.unsqueeze(1) * sensor_neighbour
        observation = observation * dem_fraction.unsqueeze(1) * dem_neighbour
        if reliability is not None:
            day = F.adaptive_avg_pool2d(reliability, size).clamp_min(0.0)
            day_neighbour = self._stack_roll(day)[0]
            observation = observation * torch.exp(-0.5 * (day.unsqueeze(1) + day_neighbour).clamp_min(0.0))

        messages, topo_affinity, gates = [], [], []
        prior_stats, residual_stats, total_stats = [], [], []
        for head in range(self.heads):
            flat = descriptor.permute(0, 1, 3, 4, 2).reshape(-1, 3)
            if self.edge_gate_type == "kan":
                residual = self.edge_kan(flat).reshape(features.shape[0], len(DIRECTIONS), 1, *size)
            else:
                residual = self.edge_mlp(flat).reshape(features.shape[0], len(DIRECTIONS), 1, *size)
            total = prior[:, :, head:head + 1] + residual
            topo = torch.sigmoid(total) * valid_edge
            latent_pair = torch.stack((latent_diff[:, :, head].abs().mean(2), (latent[:, head].unsqueeze(1) * neighbour_latent[:, :, head]).mean(2)), dim=2)
            latent_factor = torch.sigmoid(self.latent_compatibility[head](latent_pair.reshape(-1, 2, *size))).reshape(features.shape[0], len(DIRECTIONS), 1, *size) if self.latent_compatibility_enabled else torch.ones_like(topo)
            gate = topo * observation * latent_factor
            neighbour = self._stack_roll(features)[0]
            message_input = (neighbour - features.unsqueeze(1)).reshape(-1, features.shape[1], *size)
            message = self.message[head](message_input).reshape(features.shape[0], len(DIRECTIONS), self.head_channels, *size)
            denominator = gate.sum(1, keepdim=True).clamp_min(1e-6)
            weighted = (gate * message).sum(1, keepdim=True) / denominator
            confidence = gate.sum(1, keepdim=True) / valid_edge.sum(1, keepdim=True).clamp_min(1.0)
            messages.append(weighted.squeeze(1) * confidence.squeeze(1) * self.gamma[head])
            topo_affinity.append(topo)
            gates.append(gate)
            prior_stats.append(prior[:, :, head:head + 1].detach() if not self.training else prior[:, :, head:head + 1])
            residual_stats.append(residual)
            total_stats.append(total)
        update = self.output(torch.cat(messages, dim=1))
        output = features + update
        magnitude, coefficient_smoothness = self.spline_regularization()
        mono, curve_smoothness, mono_diag = self.function_regularization()
        topo = torch.cat(topo_affinity, dim=2)
        gate_stack = torch.cat(gates, dim=2)
        valid_count = valid_edge.sum().clamp_min(1.0)
        diagnostics: dict[str, Any] = {
            "gate_mean": gate_stack.sum() / valid_count.clamp_min(1.0),
            "gate_std": gate_stack.float().std(unbiased=False),
            "valid_edge_fraction": valid_edge.mean(),
            "topographic_affinity_mean": topo.sum() / valid_edge.sum().clamp_min(1.0),
            "observation_confidence_mean": observation.mean(),
            "prior_logit_mean": torch.cat(prior_stats, dim=2).mean(),
            "kan_residual_mean": torch.cat(residual_stats, dim=2).mean(),
            "total_logit_mean": torch.cat(total_stats, dim=2).mean(),
            "affinity_mean": topo.mean(),
            "gamma_mean": self.gamma.mean(),
            "gamma_values": self.gamma,
            "kan_coefficient_magnitude": magnitude,
            "kan_coefficient_smoothness": coefficient_smoothness,
            "kan_monotonicity": mono,
            "kan_curve_smoothness": curve_smoothness,
            "graph_update_rms_ratio": update.square().mean().sqrt() / features.square().mean().sqrt().clamp_min(1e-6),
            "path_barrier_invalid_fraction": (1.0 - (valid_edge > 0.5).to(features.dtype)).mean(),
        }
        diagnostics.update(mono_diag)
        if self.diagnostic_mode:
            diagnostics.update({
                "last_descriptors": descriptor.detach(),
                "prior_logit": torch.cat(prior_stats, dim=2).detach(),
                "kan_residual": torch.cat(residual_stats, dim=2).detach(),
                "total_logit": torch.cat(total_stats, dim=2).detach(),
                "topographic_affinity": topo.detach(),
            })
        return output, diagnostics

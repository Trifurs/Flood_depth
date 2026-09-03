"""HydroEdgeKAN: terrain-only feature-wise KAN with explicit edge factors.

The module is deliberately a local spatial compatibility operator.  Its KAN
sees only five topographic edge descriptors; observation confidence and latent
similarity are separate bounded factors and are not interpreted as hydraulic
quantities.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.kan_layers import KANLinear
from models.terrain_graph_kan import DIRECTIONS, _masked_pool, _roll_with_boundary_mask


HYDRO_EDGE_FEATURE_NAMES = (
    "signed_dz", "edge_slope", "relative_height_or_barrier", "local_relief", "neighbour_distance",
)


def _inverse_sigmoid(value: float) -> float:
    value = min(max(value, 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


class HydroEdgeKAN(nn.Module):
    """Vectorized eight-neighbour terrain-conditioned compatibility graph."""

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        grid_size: int = 4,
        spline_order: int = 3,
        graph_scale: int = 8,
        terrain_pixel_size_m: float = 20.0,
        feature_centers: Sequence[float] | None = None,
        feature_scales: Sequence[float] | None = None,
        base_path: str = "silu",
        base_scale_init: float = 0.5,
        spline_scale_init: float = 1.0,
        learnable_base_scale: bool = True,
        learnable_spline_scale: bool = True,
        gamma_init_effective: float = 0.02,
        gamma_max: float = 0.25,
        latent_compatibility_enabled: bool = True,
        edge_gate_type: str = "kan",
        groups: int = 8,
    ) -> None:
        super().__init__()
        if heads not in {2, 4} or channels % heads:
            raise ValueError("heads must be 2 or 4 and divide channels")
        if graph_scale <= 0 or terrain_pixel_size_m <= 0:
            raise ValueError("graph_scale and terrain_pixel_size_m must be positive")
        if not 0.0 < gamma_init_effective < gamma_max:
            raise ValueError("gamma_init_effective must lie in (0, gamma_max)")
        if gamma_max <= 0:
            raise ValueError("gamma_max must be positive")
        self.channels, self.heads = int(channels), int(heads)
        self.head_channels = channels // heads
        self.graph_scale = int(graph_scale)
        self.graph_pixel_size_m = float(graph_scale) * float(terrain_pixel_size_m)
        self.gamma_max = float(gamma_max)
        self.latent_compatibility_enabled = bool(latent_compatibility_enabled)
        if edge_gate_type not in {"kan", "mlp"}:
            raise ValueError("edge_gate_type must be kan or mlp")
        self.edge_gate_type = edge_gate_type
        self.edge_feature_names = HYDRO_EDGE_FEATURE_NAMES
        centers = list(feature_centers or [0.0, 0.0, 0.0, 0.0, self.graph_pixel_size_m])
        scales = list(feature_scales or [12.0, 0.06, 1.0, 5.0, self.graph_pixel_size_m / 2.0])
        if len(centers) != len(self.edge_feature_names) or len(scales) != len(self.edge_feature_names):
            raise ValueError("HydroEdgeKAN centers/scales must contain five values")
        if any(float(value) <= 0 for value in scales):
            raise ValueError("HydroEdgeKAN feature scales must be positive")
        self.register_buffer("feature_centers", torch.tensor(centers, dtype=torch.float32).view(1, 1, 1, 1, -1))
        self.register_buffer("feature_scales", torch.tensor(scales, dtype=torch.float32).view(1, 1, 1, 1, -1))
        self.latent_projection = nn.ModuleList([
            nn.Conv2d(channels, self.head_channels, 1) for _ in range(self.heads)
        ])
        if edge_gate_type == "kan":
            self.edge_kan = KANLinear(
                len(self.edge_feature_names), self.heads, grid_size, spline_order,
                normalization="explicit_fixed_scaling", input_bounding="prebounded",
                base_path=base_path, base_scale_init=base_scale_init,
                spline_scale_init=spline_scale_init,
                learnable_base_scale=learnable_base_scale,
                learnable_spline_scale=learnable_spline_scale,
            )
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(len(self.edge_feature_names), 16), nn.SiLU(), nn.Linear(16, self.heads))
        self.latent_compatibility = nn.ModuleList([
            nn.Conv2d(2, 1, 1, bias=True) for _ in range(self.heads)
        ])
        for projection in self.latent_compatibility:
            nn.init.zeros_(projection.weight); nn.init.zeros_(projection.bias)
        self.message = nn.ModuleList([
            nn.Conv2d(channels, self.head_channels, 1, bias=False)
            for _ in range(self.heads)
        ])
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        self.raw_gamma = nn.Parameter(torch.full((self.heads,), _inverse_sigmoid(gamma_init_effective / gamma_max)))

    @property
    def gamma(self) -> torch.Tensor:
        """Bounded, non-negative residual strengths used by all heads."""
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    def gamma_values(self) -> torch.Tensor:
        return self.gamma

    def spline_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.edge_gate_type != "kan":
            zero = self.output.weight.sum() * 0.0
            return zero, zero
        coefficients = self.edge_kan.spline_coefficients
        smoothness = (coefficients[..., 2:] - 2.0 * coefficients[..., 1:-1] + coefficients[..., :-2]).square().mean()
        # Group amplitude is intentionally weak; it does not shrink every
        # coefficient independently and therefore does not suppress curvature.
        amplitude = coefficients.square().mean(dim=-1).sqrt().mean()
        return amplitude, smoothness

    def _descriptor(self, z, barrier, relief, node_valid, direction):
        dy, dx = direction
        neighbour_z, boundary = _roll_with_boundary_mask(z, dy, dx)
        neighbour_barrier, _ = _roll_with_boundary_mask(barrier, dy, dx)
        neighbour_relief, _ = _roll_with_boundary_mask(relief, dy, dx)
        neighbour_valid, _ = _roll_with_boundary_mask(node_valid, dy, dx)
        distance = z.new_tensor(float((dx * dx + dy * dy) ** 0.5) * self.graph_pixel_size_m)
        dz = neighbour_z - z
        descriptor = torch.stack((
            dz,
            dz / distance.clamp_min(1e-6),
            0.5 * (barrier + neighbour_barrier),
            0.5 * (relief + neighbour_relief),
            torch.full_like(z, float(distance)),
        ), dim=-1)
        # Train-only robust statistics map every physical descriptor into the
        # fixed KAN domain.  The edge layer therefore does not apply a second
        # nonlinear transform (KANLinear is configured as ``prebounded``).
        normalized = ((descriptor - self.feature_centers) / self.feature_scales).clamp(-1.0, 1.0)
        valid_edge = node_valid * neighbour_valid * boundary
        return normalized[:, 0], valid_edge, boundary

    def forward(
        self,
        features: torch.Tensor,
        physical: Mapping[str, torch.Tensor],
        reliability: torch.Tensor,
        modality_weights: torch.Tensor,
        sensor_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        size = features.shape[-2:]
        dem_full = physical["dem_valid"]
        dem_fraction = F.adaptive_avg_pool2d(dem_full, size)
        sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, size)
        day = F.adaptive_avg_pool2d(reliability, size).clamp_min(0.0)
        node_valid = ((dem_fraction > 0.5) & (sensor_fraction > 0.0)).to(features.dtype)
        z = _masked_pool(physical["z_hyd"], dem_full, size)
        barrier = _masked_pool(physical["z_barrier"], dem_full, size)
        relief = _masked_pool(physical["local_relief"], dem_full, size)
        latent = torch.stack([projection(features) for projection in self.latent_projection], dim=1)
        message_by_head, gates_by_head, topo_by_head, confidence_by_head, latent_by_head = [], [], [], [], []
        descriptors = []
        valid_edges = []
        for direction in DIRECTIONS:
            descriptor, valid_edge, _ = self._descriptor(z, barrier, relief, node_valid, direction)
            descriptors.append(descriptor)
            valid_edges.append(valid_edge)
        descriptor = torch.stack(descriptors, dim=1)  # B,D,H,W,F
        valid_edge = torch.stack(valid_edges, dim=1)   # B,D,1,H,W
        if self.edge_gate_type == "kan":
            kan_logits = self.edge_kan(descriptor).permute(0, 1, 4, 2, 3)  # B,D,heads,H,W
        else:
            kan_logits = self.edge_mlp(descriptor).permute(0, 1, 4, 2, 3)
        for head in range(self.heads):
            neighbour_latent = torch.stack([
                _roll_with_boundary_mask(latent[:, head], dy, dx)[0] for dy, dx in DIRECTIONS
            ], dim=1)
            latent_diff = neighbour_latent - latent[:, head].unsqueeze(1)
            latent_pair = torch.stack((latent_diff.mean(2), latent_diff.abs().mean(2)), dim=2)
            if self.latent_compatibility_enabled:
                latent_factor = torch.sigmoid(self.latent_compatibility[head](latent_pair.reshape(-1, 2, *size))).reshape(features.shape[0], len(DIRECTIONS), 1, *size)
            else:
                latent_factor = torch.ones_like(valid_edge)
            sensor_neighbours = torch.stack([_roll_with_boundary_mask(sensor_fraction, dy, dx)[0] for dy, dx in DIRECTIONS], dim=1)
            dem_neighbours = torch.stack([_roll_with_boundary_mask(dem_fraction, dy, dx)[0] for dy, dx in DIRECTIONS], dim=1)
            day_neighbours = torch.stack([_roll_with_boundary_mask(day, dy, dx)[0] for dy, dx in DIRECTIONS], dim=1)
            observation = sensor_fraction.unsqueeze(1) * sensor_neighbours * dem_fraction.unsqueeze(1) * dem_neighbours
            observation = observation * torch.exp(-0.5 * (day.unsqueeze(1) + day_neighbours).clamp_min(0.0))
            gate = torch.sigmoid(kan_logits[:, :, head:head + 1]) * observation * latent_factor * valid_edge
            neighbour = torch.stack([_roll_with_boundary_mask(features, dy, dx)[0] for dy, dx in DIRECTIONS], dim=1)
            message_input = (neighbour - features.unsqueeze(1)).reshape(-1, features.shape[1], *size)
            message = self.message[head](message_input).reshape(features.shape[0], len(DIRECTIONS), self.head_channels, *size)
            denominator = gate.sum(1, keepdim=True).clamp_min(1e-6)
            weighted_mean = (gate * message).sum(1, keepdim=True) / denominator
            confidence = gate.sum(1, keepdim=True) / valid_edge.sum(1, keepdim=True).clamp_min(1.0)
            message_by_head.append((weighted_mean * confidence).squeeze(1) * self.gamma[head])
            gates_by_head.append(gate)
            topo_by_head.append(torch.sigmoid(kan_logits[:, :, head:head + 1]))
            confidence_by_head.append(observation)
            latent_by_head.append(latent_factor)
        messages = torch.cat(message_by_head, dim=1)
        output = features + self.output(messages)
        gate_stack = torch.cat(gates_by_head, dim=2)
        amplitude, smoothness = self.spline_regularization()
        valid_count = valid_edge.sum().clamp_min(1.0)
        return output, {
            "gate_mean": gate_stack.sum() / (valid_count * self.heads),
            "gate_std": gate_stack.float().std(unbiased=False),
            "valid_edge_fraction": valid_edge.mean(),
            "topographic_affinity_mean": torch.cat(topo_by_head, dim=2).mean(),
            "observation_confidence_mean": torch.cat(confidence_by_head, dim=2).mean(),
            "latent_compatibility_mean": torch.cat(latent_by_head, dim=2).mean(),
            "gamma_mean": self.gamma.mean(),
            "gamma_values": self.gamma,
            "kan_coefficient_magnitude": amplitude,
            "kan_coefficient_smoothness": smoothness,
            "kan_spline_group_amplitude": amplitude,
            "kan_curvature": smoothness,
            "graph_update_rms_ratio": (self.output(messages).square().mean().sqrt() / features.square().mean().sqrt().clamp_min(1e-6)),
            "last_descriptors": descriptor.detach(),
        }

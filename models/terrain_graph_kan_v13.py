"""Multi-head, identity-initialized terrain Graph-KAN for Hydro-v13."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from models.kan_layers import KANLinear
from models.terrain_graph_kan import DIRECTIONS, _masked_pool, _roll_with_boundary_mask


EDGE_FEATURE_NAMES = (
    "signed_dz", "absolute_dz", "slope", "barrier", "sensor_valid_fraction",
    "modality_concentration", "sensor_day_difference", "signed_latent_difference",
    "absolute_latent_difference", "dx", "dy", "distance",
)


class MultiHeadTerrainGraphKAN(nn.Module):
    def __init__(self, channels: int, heads: int, grid_size: int, spline_order: int,
                 edge_features: int, normalization: str, message_normalization: str,
                 groups: int = 8) -> None:
        super().__init__()
        if heads not in {2, 4} or channels % heads:
            raise ValueError("graph_heads must be 2 or 4 and divide channels")
        if not 1 <= edge_features <= len(EDGE_FEATURE_NAMES):
            raise ValueError(f"kan_edge_features must be 1..{len(EDGE_FEATURE_NAMES)}")
        if message_normalization not in {"edge_count", "gate_sum"}:
            raise ValueError("graph_message_normalization must be edge_count or gate_sum")
        self.heads, self.head_channels = heads, channels // heads
        self.edge_features = edge_features
        self.message_normalization = message_normalization
        self.feature_projection = nn.Conv2d(channels, heads, 1)
        self.edge_kans = nn.ModuleList([
            KANLinear(edge_features, 1, grid_size, spline_order, normalization=normalization)
            for _ in range(heads)
        ])
        self.message = nn.Conv2d(channels, channels, 1, groups=heads, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(heads))

    def spline_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = torch.stack([module.spline_coefficients for module in self.edge_kans])
        magnitude = coefficients.square().mean()
        smoothness = (coefficients[..., 2:] - 2 * coefficients[..., 1:-1] + coefficients[..., :-2]).square().mean()
        return magnitude, smoothness

    def forward(self, features, physical, reliability, modality_weights, sensor_valid):
        size = features.shape[-2:]
        dem_full = physical["dem_valid"]
        dem_fraction = F.adaptive_avg_pool2d(dem_full, size)
        sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, size)
        node_valid = (dem_fraction > 0.5).to(features.dtype) * (sensor_fraction > 0).to(features.dtype)
        z = _masked_pool(physical["z_hyd"], dem_full, size)
        slope = _masked_pool(physical["slope"], dem_full, size)
        barrier = _masked_pool(physical["z_barrier"], dem_full, size)
        time = F.adaptive_avg_pool2d(reliability, size)
        concentration = modality_weights.max(1, keepdim=True).values
        latent = self.feature_projection(features)
        messages = torch.zeros_like(features)
        gate_denominator = torch.zeros_like(node_valid).repeat(1, self.heads, 1, 1)
        edge_count = torch.zeros_like(node_valid)
        gate_total = features.new_zeros(())
        valid_total = features.new_zeros(())
        for dy, dx in DIRECTIONS:
            neighbour, boundary = _roll_with_boundary_mask(features, dy, dx)
            neighbour_valid, _ = _roll_with_boundary_mask(node_valid, dy, dx)
            neighbour_z, _ = _roll_with_boundary_mask(z, dy, dx)
            neighbour_slope, _ = _roll_with_boundary_mask(slope, dy, dx)
            neighbour_barrier, _ = _roll_with_boundary_mask(barrier, dy, dx)
            neighbour_latent, _ = _roll_with_boundary_mask(latent, dy, dx)
            valid_edge = node_valid * neighbour_valid * boundary
            dz = (neighbour_z - z) / 5.0
            latent_difference = neighbour_latent - latent
            descriptor = torch.cat((
                torch.tanh(dz), torch.tanh(dz.abs()),
                torch.tanh((slope + neighbour_slope) / 60.0),
                torch.tanh((barrier + neighbour_barrier) / 10.0),
                sensor_fraction, concentration, time,
                torch.tanh(latent_difference.mean(1, keepdim=True)),
                torch.tanh(latent_difference.abs().mean(1, keepdim=True)),
                torch.full_like(z, float(dx)), torch.full_like(z, float(dy)),
                torch.full_like(z, math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)),
            ), 1)[:, : self.edge_features]
            gates = []
            edge_last = descriptor.permute(0, 2, 3, 1)
            for kan in self.edge_kans:
                gates.append(torch.sigmoid(kan(edge_last).permute(0, 3, 1, 2)))
            gate = torch.cat(gates, 1) * valid_edge
            projected_message = self.message(neighbour - features).view(
                features.shape[0], self.heads, self.head_channels, *size
            )
            messages = messages + (gate.unsqueeze(2) * projected_message).flatten(1, 2)
            gate_denominator += gate
            edge_count += valid_edge
            gate_total += gate.sum()
            valid_total += valid_edge.sum() * self.heads
        if self.message_normalization == "gate_sum":
            denominator = gate_denominator.clamp_min(1e-6).unsqueeze(2)
        else:
            denominator = edge_count.clamp_min(1.0).unsqueeze(1).unsqueeze(2)
        normalized = (messages.view(features.shape[0], self.heads, self.head_channels, *size) / denominator)
        scaled = normalized * self.gamma.view(1, self.heads, 1, 1, 1)
        output = features + self.output(scaled.flatten(1, 2))
        magnitude, smoothness = self.spline_regularization()
        return output, {
            "gate_mean": gate_total / valid_total.clamp_min(1.0),
            "valid_edge_fraction": valid_total / (node_valid.sum().clamp_min(1.0) * len(DIRECTIONS) * self.heads),
            "mean_gate_map": gate_denominator.mean(1, keepdim=True) / edge_count.clamp_min(1.0),
            "gamma_mean": self.gamma.mean(), "kan_coefficient_magnitude": magnitude,
            "kan_coefficient_smoothness": smoothness,
        }

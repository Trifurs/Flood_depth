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

V131_EDGE_FEATURE_NAMES = (
    "signed_dz", "absolute_dz", "edge_slope", "barrier",
    "sensor_valid_fraction", "dem_valid_fraction",
    "modality_weight_concentration", "normalized_sensor_day_difference",
    "latent_signed_difference", "latent_absolute_difference", "neighbour_distance",
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
        reliability = F.adaptive_avg_pool2d(reliability, size)
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


class VectorizedTerrainGraphKANV131(nn.Module):
    """Two-head terrain graph with one KAN call per head over all directions."""

    def __init__(self, channels: int, heads: int = 2, grid_size: int = 8,
                 spline_order: int = 3, normalization: str = "explicit_fixed_scaling",
                 message_normalization: str = "gate_sum", graph_scale: int = 8,
                 terrain_pixel_size_m: float = 20.0,
                 edge_feature_names: tuple[str, ...] = V131_EDGE_FEATURE_NAMES,
                 feature_scales: dict[str, float] | None = None, groups: int = 8) -> None:
        super().__init__()
        if heads not in {2, 4} or channels % heads:
            raise ValueError("graph_heads must be 2 or 4 and divide channels")
        unknown = set(edge_feature_names).difference(V131_EDGE_FEATURE_NAMES)
        if unknown or len(edge_feature_names) != len(set(edge_feature_names)):
            raise ValueError(f"Invalid Graph-KAN edge feature names: {sorted(unknown)}")
        if message_normalization not in {"edge_count", "gate_sum"}:
            raise ValueError("graph_message_normalization must be edge_count or gate_sum")
        if graph_scale <= 0 or terrain_pixel_size_m <= 0:
            raise ValueError("graph_scale and terrain_pixel_size_m must be positive")
        self.heads = int(heads)
        self.head_channels = channels // heads
        self.message_normalization = message_normalization
        self.graph_scale = int(graph_scale)
        self.graph_pixel_size_m = float(graph_scale) * float(terrain_pixel_size_m)
        self.edge_feature_names = tuple(edge_feature_names)
        defaults = {
            "signed_dz": 5.0, "absolute_dz": 5.0, "edge_slope": 0.20,
            "barrier": 10.0, "sensor_valid_fraction": 1.0,
            "dem_valid_fraction": 1.0, "modality_weight_concentration": 1.0,
            "normalized_sensor_day_difference": 1.0,
            "latent_signed_difference": 1.0, "latent_absolute_difference": 1.0,
            "neighbour_distance": 1.0,
        }
        if feature_scales:
            defaults.update({key: float(value) for key, value in feature_scales.items()})
        scales = [defaults[name] for name in self.edge_feature_names]
        if any(value <= 0 for value in scales):
            raise ValueError("Graph-KAN feature scales must be positive")
        # Descriptors are laid out as (batch, direction, feature, height, width).
        # Keep feature scales on the same axis so broadcasting is independent of
        # the raster width (the previous trailing-axis layout was shape-unsafe).
        self.register_buffer("feature_scales", torch.tensor(scales).view(1, 1, -1, 1, 1))
        self.latent_projection = nn.ModuleList([
            nn.Conv2d(channels, self.head_channels, 1) for _ in range(self.heads)
        ])
        self.edge_kans = nn.ModuleList([
            KANLinear(len(self.edge_feature_names), 1, grid_size, spline_order, normalization=normalization)
            for _ in range(self.heads)
        ])
        self.message = nn.ModuleList([
            nn.Conv2d(channels, self.head_channels, 1, bias=False)
            for _ in range(self.heads)
        ])
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(self.heads))

    def spline_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = torch.stack([module.spline_coefficients for module in self.edge_kans])
        magnitude = coefficients.square().mean()
        smoothness = (
            coefficients[..., 2:] - 2 * coefficients[..., 1:-1] + coefficients[..., :-2]
        ).square().mean()
        return magnitude, smoothness

    @staticmethod
    def _roll_stack(value: torch.Tensor):
        rolled, boundaries = zip(*[_roll_with_boundary_mask(value, dy, dx) for dy, dx in DIRECTIONS])
        return torch.stack(rolled, dim=1), torch.stack(boundaries, dim=1)

    def _forward_impl(self, features, physical, reliability, modality_weights,
                      sensor_valid, vectorized: bool):
        size = features.shape[-2:]
        dem_full = physical["dem_valid"]
        dem_fraction = F.adaptive_avg_pool2d(dem_full, size)
        sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, size)
        reliability = F.adaptive_avg_pool2d(reliability, size)
        node_valid = (dem_fraction > 0.5).to(features.dtype) * (sensor_fraction > 0).to(features.dtype)
        z = _masked_pool(physical["z_hyd"], dem_full, size)
        barrier = _masked_pool(physical["z_barrier"], dem_full, size)
        slope = _masked_pool(physical["slope"], dem_full, size)
        neighbour, boundary = self._roll_stack(features)
        neighbour_valid, _ = self._roll_stack(node_valid)
        neighbour_z, _ = self._roll_stack(z)
        neighbour_barrier, _ = self._roll_stack(barrier)
        neighbour_slope, _ = self._roll_stack(slope)
        valid_edge = node_valid.unsqueeze(1) * neighbour_valid * boundary
        concentration = modality_weights.max(1, keepdim=True).values
        latent = torch.stack([projection(features) for projection in self.latent_projection], dim=1)
        neighbour_latent = torch.stack(
            [self._roll_stack(latent[:, head])[0] for head in range(self.heads)], dim=2
        )
        # B, directions, heads, channels, H, W
        latent_diff = neighbour_latent - latent.unsqueeze(1)
        dz = neighbour_z - z.unsqueeze(1)
        distance_values = [
            self.graph_pixel_size_m * (float(dx * dx + dy * dy) ** 0.5)
            for dy, dx in DIRECTIONS
        ]
        distance = features.new_tensor(distance_values).view(1, len(DIRECTIONS), 1, 1, 1)
        base_features = {
            "signed_dz": dz / 5.0,
            "absolute_dz": dz.abs() / 5.0,
            "edge_slope": dz / distance,
            "barrier": (barrier.unsqueeze(1) + neighbour_barrier) / 10.0,
            "sensor_valid_fraction": sensor_fraction.unsqueeze(1).expand_as(dz),
            "dem_valid_fraction": dem_fraction.unsqueeze(1).expand_as(dz),
            "modality_weight_concentration": concentration.unsqueeze(1).expand_as(dz),
            "normalized_sensor_day_difference": reliability.unsqueeze(1).expand_as(dz),
            "neighbour_distance": distance.expand_as(dz) / self.graph_pixel_size_m,
        }
        head_messages = []
        gates_for_diag = []
        for head in range(self.heads):
            head_values = dict(base_features)
            head_values["latent_signed_difference"] = latent_diff[:, :, head].mean(2, keepdim=True)
            head_values["latent_absolute_difference"] = latent_diff[:, :, head].abs().mean(2, keepdim=True)
            descriptor = torch.cat([head_values[name] for name in self.edge_feature_names], dim=2)
            descriptor = (descriptor / self.feature_scales).clamp(-4.0, 4.0)
            if vectorized:
                coefficients = self.edge_kans[head](descriptor.permute(0, 1, 3, 4, 2).reshape(-1, descriptor.shape[2]))
                coefficients = coefficients.reshape(features.shape[0], len(DIRECTIONS), *size).unsqueeze(2)
            else:
                coefficients = torch.stack([
                    self.edge_kans[head](descriptor[:, direction].permute(0, 2, 3, 1).reshape(-1, descriptor.shape[2]))
                    .reshape(features.shape[0], 1, *size)
                    for direction in range(len(DIRECTIONS))
                ], dim=1)
            gate = torch.sigmoid(coefficients) * valid_edge
            message_inputs = (neighbour[:, :, :, :, :] - features.unsqueeze(1))
            message = self.message[head](message_inputs.reshape(-1, features.shape[1], *size))
            message = message.reshape(features.shape[0], len(DIRECTIONS), self.head_channels, *size)
            denominator = gate.sum(1, keepdim=True).clamp_min(1e-6)
            weighted_mean = (gate * message).sum(1, keepdim=True) / denominator
            confidence = gate.sum(1, keepdim=True) / valid_edge.sum(1, keepdim=True).clamp_min(1.0)
            head_messages.append((weighted_mean * confidence).squeeze(1) * self.gamma[head])
            gates_for_diag.append(gate)
        messages = torch.cat(head_messages, dim=1)
        output = features + self.output(messages)
        gates = torch.cat(gates_for_diag, dim=2)
        magnitude, smoothness = self.spline_regularization()
        valid_count = valid_edge.sum().clamp_min(1.0)
        return output, {
            "gate_mean": gates.sum() / (valid_count * self.heads),
            "valid_edge_fraction": valid_edge.sum() / (node_valid.sum().clamp_min(1.0) * len(DIRECTIONS)),
            "mean_gate_map": gates.mean(2).mean(1, keepdim=True),
            "gamma_mean": self.gamma.mean(),
            "kan_coefficient_magnitude": magnitude,
            "kan_coefficient_smoothness": smoothness,
        }

    def forward(self, features, physical, reliability, modality_weights, sensor_valid):
        return self._forward_impl(features, physical, reliability, modality_weights, sensor_valid, True)

    def forward_reference(self, features, physical, reliability, modality_weights, sensor_valid):
        return self._forward_impl(features, physical, reliability, modality_weights, sensor_valid, False)

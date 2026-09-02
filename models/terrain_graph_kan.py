"""Vectorized eight-neighbour terrain-conditioned latent connectivity."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.encoders import ConvNormAct, group_count
from models.kan_layers import KANLinear
from datasets.preprocessing import RELIABILITY_NAMES


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _roll_with_boundary_mask(
    tensor: torch.Tensor, dy: int, dx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    neighbour = torch.roll(tensor, shifts=(dy, dx), dims=(-2, -1))
    boundary = torch.ones(
        (tensor.shape[0], 1, tensor.shape[-2], tensor.shape[-1]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if dy > 0:
        boundary[:, :, :dy, :] = 0
    elif dy < 0:
        boundary[:, :, dy:, :] = 0
    if dx > 0:
        boundary[:, :, :, :dx] = 0
    elif dx < 0:
        boundary[:, :, :, dx:] = 0
    return neighbour, boundary


def _masked_pool(
    values: torch.Tensor, valid: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    numerator = F.adaptive_avg_pool2d(values * valid, size)
    denominator = F.adaptive_avg_pool2d(valid, size)
    return numerator / denominator.clamp_min(1e-6)


class TerrainGraphKAN(nn.Module):
    """Eight-neighbour Graph-KAN at 1/8 scale; gates are latent, not fluxes."""

    def __init__(
        self,
        channels: int,
        grid_size: int = 8,
        spline_order: int = 3,
        dropout: float = 0.1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Conv2d(channels, 1, 1)
        self.edge_kan = KANLinear(8, 1, grid_size, spline_order)
        self.message = nn.Conv2d(channels, channels, 1, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(
        self,
        features: torch.Tensor,
        physical: dict[str, torch.Tensor],
        reliability: torch.Tensor,
        modality_weights: torch.Tensor,
        sensor_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        size = features.shape[-2:]
        dem_valid_full = physical["dem_valid"]
        dem_valid = (F.adaptive_avg_pool2d(dem_valid_full, size) > 0.5).to(features.dtype)
        sensor = (F.adaptive_max_pool2d(sensor_valid, size) > 0.5).to(features.dtype)
        node_valid = dem_valid * sensor
        z_hyd = _masked_pool(physical["z_hyd"], dem_valid_full, size)
        slope = _masked_pool(physical["slope"], dem_valid_full, size)
        barrier = _masked_pool(physical["z_barrier"], dem_valid_full, size)
        day_index = RELIABILITY_NAMES.index("absolute_normalized_sensor_day_difference")
        time_difference = F.adaptive_avg_pool2d(reliability[:, day_index : day_index + 1], size)
        sensor_reliability = modality_weights.max(dim=1, keepdim=True).values
        projected = self.feature_projection(features)

        message_sum = torch.zeros_like(features)
        edge_count = torch.zeros_like(node_valid)
        gate_sum = torch.zeros_like(node_valid)
        valid_edge_total = torch.zeros((), device=features.device, dtype=features.dtype)
        gate_total = torch.zeros((), device=features.device, dtype=features.dtype)
        for dy, dx in DIRECTIONS:
            neighbour_features, boundary = _roll_with_boundary_mask(features, dy, dx)
            neighbour_valid, _ = _roll_with_boundary_mask(node_valid, dy, dx)
            neighbour_z, _ = _roll_with_boundary_mask(z_hyd, dy, dx)
            neighbour_slope, _ = _roll_with_boundary_mask(slope, dy, dx)
            neighbour_barrier, _ = _roll_with_boundary_mask(barrier, dy, dx)
            neighbour_reliability, _ = _roll_with_boundary_mask(sensor_reliability, dy, dx)
            neighbour_projected, _ = _roll_with_boundary_mask(projected, dy, dx)
            edge_valid = node_valid * neighbour_valid * boundary
            dz = torch.tanh((neighbour_z - z_hyd) / 5.0)
            feature_difference = neighbour_projected - projected
            edge_features = torch.cat(
                (
                    dz,
                    dz.abs(),
                    torch.tanh((slope + neighbour_slope) / 60.0),
                    torch.tanh((barrier + neighbour_barrier) / 10.0),
                    0.5 * (sensor_reliability + neighbour_reliability),
                    time_difference,
                    torch.tanh(feature_difference),
                    torch.tanh(feature_difference.abs()),
                ),
                dim=1,
            )
            logits = self.edge_kan(edge_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            gate = torch.sigmoid(logits) * edge_valid
            message_sum = message_sum + gate * self.message(neighbour_features - features)
            edge_count = edge_count + edge_valid
            gate_sum = gate_sum + gate
            valid_edge_total = valid_edge_total + edge_valid.sum()
            gate_total = gate_total + gate.sum()
        aggregated = message_sum / edge_count.clamp_min(1.0)
        output = features + self.output(aggregated)
        valid_nodes = node_valid.sum().clamp_min(1.0)
        diagnostics = {
            "gate_mean": gate_total / valid_edge_total.clamp_min(1.0),
            "valid_edge_fraction": valid_edge_total
            / (valid_nodes * float(len(DIRECTIONS))).clamp_min(1.0),
            "mean_gate_map": gate_sum / edge_count.clamp_min(1.0),
        }
        return output, diagnostics

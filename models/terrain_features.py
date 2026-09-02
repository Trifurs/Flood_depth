"""Mask-aware online DSM derivatives and terrain pyramid."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.encoders import ConvNormAct, ResidualBlock


def masked_average(
    values: torch.Tensor, valid: torch.Tensor, kernel_size: int
) -> torch.Tensor:
    padding = kernel_size // 2
    numerator = F.avg_pool2d(values * valid, kernel_size, stride=1, padding=padding)
    denominator = F.avg_pool2d(valid, kernel_size, stride=1, padding=padding)
    return numerator / denominator.clamp_min(1e-6)


class TerrainFeaturePyramid(nn.Module):
    """Construct DSM-based proxies without writing derived data back to the dataset."""

    def __init__(
        self,
        channels: Sequence[int],
        dropout: float = 0.1,
        groups: int = 8,
        context_kernel_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        kernels = tuple(
            dict.fromkeys(int(value) for value in (context_kernel_sizes or (9,)))
        )
        if 9 not in kernels:
            raise ValueError("terrain context kernels must include the 9-pixel base scale")
        if any(value <= 0 or value % 2 == 0 for value in kernels):
            raise ValueError("terrain context kernel sizes must be positive odd integers")
        self.context_kernel_sizes = kernels
        # The original eight channels remain byte-for-byte compatible at (9,).
        # Every additional scale contributes signed relative height, a depression
        # proxy, and local relief.
        terrain_input_channels = 8 + 3 * (len(kernels) - 1)
        self.stem = nn.Sequential(
            ConvNormAct(terrain_input_channels, channels[0], 3, groups=groups),
            ResidualBlock(channels[0], dropout, groups),
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(channels[index - 1], channels[index], 3, 2, groups),
                    ResidualBlock(channels[index], dropout, groups),
                )
                for index in range(1, len(channels))
            ]
        )

    def forward(
        self,
        normalized_terrain: torch.Tensor,
        raw_terrain: torch.Tensor,
        dem_valid: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        valid = (dem_valid > 0.5).to(raw_terrain.dtype)
        elevation = raw_terrain[:, 0:1]
        slope = raw_terrain[:, 1:2]
        z_hyd = masked_average(elevation, valid, kernel_size=9)
        z_relative = torch.where(valid > 0, elevation - z_hyd, torch.zeros_like(elevation))
        z_barrier = F.relu(z_relative)
        local_second = masked_average(elevation.square(), valid, kernel_size=9)
        local_relief = (local_second - z_hyd.square()).clamp_min(0).sqrt()

        filled = torch.where(valid > 0, z_hyd, torch.zeros_like(z_hyd))
        dz_dx = F.pad(filled, (1, 1, 0, 0), mode="replicate")[:, :, :, 2:] - F.pad(
            filled, (1, 1, 0, 0), mode="replicate"
        )[:, :, :, :-2]
        dz_dy = F.pad(filled, (0, 0, 1, 1), mode="replicate")[:, :, 2:, :] - F.pad(
            filled, (0, 0, 1, 1), mode="replicate"
        )[:, :, :-2, :]
        dz_dx = 0.5 * dz_dx
        dz_dy = 0.5 * dz_dy

        count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        patch_mean = (elevation * valid).sum(dim=(-2, -1), keepdim=True) / count
        patch_variance = ((elevation - patch_mean).square() * valid).sum(
            dim=(-2, -1), keepdim=True
        ) / count
        patch_relative = (elevation - patch_mean) / patch_variance.sqrt().clamp_min(1.0)
        relief_scale = local_relief + 1.0
        base_parts = [
            normalized_terrain,
            torch.tanh(z_relative / relief_scale),
            torch.tanh(z_barrier / relief_scale),
            torch.tanh(dz_dx / 10.0),
            torch.tanh(dz_dy / 10.0),
            torch.log1p(local_relief.clamp_min(0)) / 5.0,
            torch.tanh(patch_relative),
        ]
        for kernel_size in self.context_kernel_sizes:
            if kernel_size == 9:
                continue
            context_mean = masked_average(elevation, valid, kernel_size)
            context_second = masked_average(elevation.square(), valid, kernel_size)
            context_relief = (
                context_second - context_mean.square()
            ).clamp_min(0).sqrt()
            context_relative = elevation - context_mean
            context_depression = F.relu(-context_relative)
            context_scale = context_relief + 1.0
            base_parts.extend(
                (
                    torch.tanh(context_relative / context_scale),
                    torch.tanh(context_depression / context_scale),
                    torch.log1p(context_relief) / 5.0,
                )
            )
        base = torch.cat(base_parts, dim=1)
        base = base * valid
        features = [self.stem(base)]
        for downsample in self.down:
            features.append(downsample(features[-1]))
        physical = {
            "z_hyd": z_hyd,
            "z_relative": z_relative,
            "z_barrier": z_barrier,
            "dz_dx": dz_dx,
            "dz_dy": dz_dy,
            "local_relief": local_relief,
            "slope": slope,
            "dem_valid": valid,
        }
        return features, physical

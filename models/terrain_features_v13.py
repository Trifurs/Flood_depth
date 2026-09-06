"""Resolution-aware, invalid-boundary-safe terrain features for Hydro-v13."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import residual_block
from models.encoders import ConvNormAct
from models.terrain_features import masked_average


def valid_central_gradients(
    elevation: torch.Tensor, valid: torch.Tensor, pixel_size_x: float, pixel_size_y: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left, right = elevation[..., :, :-2], elevation[..., :, 2:]
    left_valid, right_valid = valid[..., :, :-2], valid[..., :, 2:]
    gx_valid = left_valid * right_valid * valid[..., :, 1:-1]
    gx_core = (right - left) / (2.0 * pixel_size_x)
    gx = F.pad(torch.where(gx_valid > 0.5, gx_core, torch.zeros_like(gx_core)), (1, 1, 0, 0))
    gx_mask = F.pad(gx_valid, (1, 1, 0, 0))
    top, bottom = elevation[..., :-2, :], elevation[..., 2:, :]
    top_valid, bottom_valid = valid[..., :-2, :], valid[..., 2:, :]
    gy_valid = top_valid * bottom_valid * valid[..., 1:-1, :]
    gy_core = (bottom - top) / (2.0 * pixel_size_y)
    gy = F.pad(torch.where(gy_valid > 0.5, gy_core, torch.zeros_like(gy_core)), (0, 0, 1, 1))
    gy_mask = F.pad(gy_valid, (0, 0, 1, 1))
    return gx, gy, gx_mask, gy_mask


class TerrainFeaturePyramidV13(nn.Module):
    def __init__(self, input_channels: int, channels: list[int], dropout: float, groups: int,
                 pixel_size_m: float, block_kind: str = "efficient") -> None:
        super().__init__()
        if pixel_size_m <= 0:
            raise ValueError("terrain_pixel_size_m must be positive")
        self.pixel_size_m = float(pixel_size_m)
        # selected normalized terrain + six derived features + valid fraction
        self.stem = nn.Sequential(
            ConvNormAct(input_channels + 7, channels[0], 3, groups=groups),
            residual_block(block_kind, channels[0], dropout, groups),
        )
        self.down = nn.ModuleList([
            nn.Sequential(
                ConvNormAct(channels[i - 1], channels[i], 3, 2, groups),
                residual_block(block_kind, channels[i], dropout, groups),
            ) for i in range(1, len(channels))
        ])

    def forward(self, normalized: torch.Tensor, raw: torch.Tensor, dem_valid: torch.Tensor):
        valid = (dem_valid > 0.5).to(raw.dtype)
        elevation = raw[:, 0:1]
        slope = raw[:, 1:2]
        z_hyd = masked_average(elevation, valid, 9)
        relative = torch.where(valid > 0, elevation - z_hyd, torch.zeros_like(elevation))
        barrier = F.relu(relative)
        second = masked_average(elevation.square(), valid, 9)
        relief = (second - z_hyd.square()).clamp_min(0).sqrt()
        gx, gy, gx_valid, gy_valid = valid_central_gradients(
            elevation, valid, self.pixel_size_m, self.pixel_size_m
        )
        scale = relief + 1.0
        base = torch.cat((
            normalized, torch.tanh(relative / scale), torch.tanh(barrier / scale),
            torch.tanh(gx), torch.tanh(gy), torch.log1p(relief) / 5.0,
            gx_valid * gy_valid, valid,
        ), 1) * valid
        features = [self.stem(base) * valid]
        fractions = [valid]
        for layer in self.down:
            features.append(layer(features[-1]))
            fraction = F.adaptive_avg_pool2d(valid, features[-1].shape[-2:])
            fractions.append(fraction)
            features[-1] = features[-1] * fraction
        physical = {
            "z_hyd": z_hyd, "z_relative": relative, "z_barrier": barrier,
            "dz_dx": gx, "dz_dy": gy, "gradient_x_valid": gx_valid,
            "gradient_y_valid": gy_valid, "local_relief": relief, "slope": slope,
            "dem_valid": valid, "dem_valid_fractions": fractions,
        }
        return features, physical

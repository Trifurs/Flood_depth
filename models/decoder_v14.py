"""Hydro-v14 decoder with independent sensor and terrain gates."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import residual_block
from models.encoders import ConvNormAct, group_count


class IndependentGatedFPNDecoderV14(nn.Module):
    """FPN decoder whose sensor and DSM skip gates are not a softmax pair."""

    def __init__(
        self, channels: list[int], dropout: float, groups: int,
        block_kind: str, deep_supervision: bool,
        widths: list[int] | tuple[int, int, int] = (64, 48, 32),
    ) -> None:
        super().__init__()
        if len(widths) != 3 or any(int(width) <= 0 for width in widths):
            raise ValueError("decoder_widths must contain three positive widths")
        widths = [int(width) for width in widths]
        self.widths = widths
        self.bottleneck = ConvNormAct(channels[-1], widths[0], 1, groups=groups)
        skip_indices = (2, 1, 0)
        target_widths = (widths[1], widths[2], widths[2])
        previous_widths = (widths[0], widths[1], widths[2])
        self.up_projection = nn.ModuleList([nn.Conv2d(previous, target, 1) for previous, target in zip(previous_widths, target_widths)])
        self.sensor_lateral = nn.ModuleList([nn.Conv2d(channels[index], target, 1) for index, target in zip(skip_indices, target_widths)])
        self.terrain_lateral = nn.ModuleList([nn.Conv2d(channels[index], target, 1) for index, target in zip(skip_indices, target_widths)])
        self.gates = nn.ModuleList([nn.Conv2d(3 * target + 2, 2, 1) for target in target_widths])
        self.refine = nn.ModuleList([residual_block(block_kind, target, dropout, groups) for target in target_widths])
        self.auxiliary = nn.ModuleList([nn.Conv2d(target, 1, 1) for target in target_widths]) if deep_supervision else None
        self.final = residual_block(block_kind, widths[-1], dropout, groups)

    def forward(self, bottleneck, skips, terrain, dem_fractions, sensor_valid):
        decoded = self.bottleneck(bottleneck)
        auxiliaries = []
        gate_maps = []
        sensor_valid = sensor_valid.to(decoded.dtype)
        for stage, index in enumerate((2, 1, 0)):
            decoded = F.interpolate(decoded, skips[index].shape[-2:], mode="bilinear", align_corners=False)
            decoded = self.up_projection[stage](decoded)
            sensor_skip = self.sensor_lateral[stage](skips[index])
            terrain_skip = self.terrain_lateral[stage](terrain[index])
            sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, decoded.shape[-2:])
            dem_fraction = dem_fractions[index].to(decoded.dtype)
            logits = self.gates[stage](torch.cat((decoded, sensor_skip, terrain_skip, sensor_fraction, dem_fraction), dim=1))
            g_sensor = torch.sigmoid(logits[:, 0:1]) * sensor_fraction
            g_terrain = torch.sigmoid(logits[:, 1:2]) * dem_fraction
            decoded = self.refine[stage](decoded + g_sensor * sensor_skip + g_terrain * terrain_skip)
            gate_maps.append({
                "sensor": g_sensor,
                "terrain": g_terrain,
                "sensor_mean": g_sensor.mean(),
                "sensor_std": g_sensor.float().std(unbiased=False),
                "sensor_p05": torch.quantile(g_sensor.float().flatten(), 0.05),
                "sensor_p50": torch.quantile(g_sensor.float().flatten(), 0.50),
                "sensor_p95": torch.quantile(g_sensor.float().flatten(), 0.95),
                "terrain_mean": g_terrain.mean(),
                "terrain_std": g_terrain.float().std(unbiased=False),
                "terrain_p05": torch.quantile(g_terrain.float().flatten(), 0.05),
                "terrain_p50": torch.quantile(g_terrain.float().flatten(), 0.50),
                "terrain_p95": torch.quantile(g_terrain.float().flatten(), 0.95),
            })
            if self.auxiliary is not None:
                auxiliaries.append(F.softplus(self.auxiliary[stage](decoded)))
        return self.final(decoded), auxiliaries, gate_maps

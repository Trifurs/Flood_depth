"""Gated FPN decoder and separate Hydro-v13 output heads."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import residual_block
from models.encoders import ConvNormAct, group_count


class GatedFPNDecoder(nn.Module):
    def __init__(self, channels: list[int], dropout: float, groups: int, block_kind: str,
                 deep_supervision: bool) -> None:
        super().__init__()
        width = channels[0]
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels])
        self.gates = nn.ModuleList([nn.Conv2d(2 * width + 1, width, 1) for _ in channels[:-1]])
        self.refine = nn.ModuleList([residual_block(block_kind, width, dropout, groups) for _ in channels[:-1]])
        self.final = residual_block(block_kind, width, dropout, groups)
        self.auxiliary = nn.ModuleList([nn.Conv2d(width, 1, 1) for _ in channels[:-1]]) if deep_supervision else None

    def forward(self, bottleneck, skips, terrain, dem_fractions):
        decoded = self.lateral[-1](bottleneck)
        auxiliaries = []
        for stage, index in enumerate(range(len(skips) - 2, -1, -1)):
            skip = self.lateral[index](skips[index])
            decoded = F.interpolate(decoded, skip.shape[-2:], mode="bilinear", align_corners=False)
            terrain_hint = self.lateral[index](terrain[index])
            fraction = dem_fractions[index]
            gate = torch.sigmoid(self.gates[stage](torch.cat((decoded, skip + terrain_hint, fraction), 1)))
            decoded = self.refine[stage](decoded + gate * skip + (1 - gate) * terrain_hint)
            if self.auxiliary is not None:
                auxiliaries.append(F.softplus(self.auxiliary[stage](decoded)))
        return self.final(decoded), auxiliaries


class GatedFPNDecoderV131(nn.Module):
    """Three-width FPN with independent sensor/terrain gates."""

    def __init__(self, channels: list[int], dropout: float, groups: int,
                 block_kind: str, deep_supervision: bool,
                 widths: list[int] | tuple[int, int, int] = (64, 48, 32),
                 change_channels: int = 3) -> None:
        super().__init__()
        if len(widths) != 3 or any(int(width) <= 0 for width in widths):
            raise ValueError("decoder_widths must contain three positive widths")
        widths = [int(width) for width in widths]
        self.widths = widths
        self.bottleneck = ConvNormAct(channels[-1], widths[0], 1, groups=groups)
        skip_indices = (2, 1, 0)
        target_widths = (widths[1], widths[2], widths[2])
        previous_widths = (widths[0], widths[1], widths[2])
        self.up_projection = nn.ModuleList([
            nn.Conv2d(previous, target, 1)
            for previous, target in zip(previous_widths, target_widths)
        ])
        self.sensor_lateral = nn.ModuleList([
            nn.Conv2d(channels[index], target, 1)
            for index, target in zip(skip_indices, target_widths)
        ])
        self.terrain_lateral = nn.ModuleList([
            nn.Conv2d(channels[index], target, 1)
            for index, target in zip(skip_indices, target_widths)
        ])
        self.gates = nn.ModuleList([
            nn.Conv2d(3 * target + 2, 2, 1) for target in target_widths
        ])
        self.refine = nn.ModuleList([
            residual_block(block_kind, target, dropout, groups)
            for target in target_widths
        ])
        self.auxiliary = (
            nn.ModuleList([nn.Conv2d(target, 1, 1) for target in target_widths])
            if deep_supervision else None
        )
        self.final = residual_block(block_kind, widths[-1], dropout, groups)
        self.change_projection = (
            nn.Sequential(
                nn.Conv2d(change_channels, widths[-1], 3, padding=1, bias=False),
                nn.GroupNorm(group_count(widths[-1], groups), widths[-1]),
                nn.SiLU(),
            ) if change_channels > 0 else None
        )

    def forward(self, bottleneck, skips, terrain, dem_fractions,
                sensor_valid, change_evidence=None):
        decoded = self.bottleneck(bottleneck)
        auxiliaries = []
        gate_maps = []
        for stage, index in enumerate((2, 1, 0)):
            decoded = F.interpolate(decoded, skips[index].shape[-2:], mode="bilinear", align_corners=False)
            decoded = self.up_projection[stage](decoded)
            sensor = self.sensor_lateral[stage](skips[index])
            topo = self.terrain_lateral[stage](terrain[index])
            sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, decoded.shape[-2:])
            dem_fraction = dem_fractions[index]
            gate = torch.softmax(
                self.gates[stage](torch.cat((decoded, sensor, topo, sensor_fraction, dem_fraction), 1)),
                dim=1,
            )
            decoded = self.refine[stage](decoded + gate[:, 0:1] * sensor + gate[:, 1:2] * topo)
            gate_maps.append(gate)
            if self.auxiliary is not None:
                auxiliaries.append(F.softplus(self.auxiliary[stage](decoded)))
        if self.change_projection is not None and change_evidence is not None:
            decoded = decoded + self.change_projection(change_evidence)
        decoded = self.final(decoded)
        return decoded, auxiliaries, gate_maps


class TaskHead(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels), nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x):
        return self.trunk(x)


class SeparateFloodDepthHeads(nn.Module):
    def __init__(self, channels: int, groups: int, epsilon: float, maximum: float,
                 semantics: str = "conditional_positive_v2") -> None:
        super().__init__()
        if semantics != "conditional_positive_v2":
            raise ValueError("Hydro-v13 supports conditional_positive_v2")
        self.depth_output_semantics = semantics
        self.depth_head, self.support_head, self.uncertainty_head = (
            TaskHead(channels, groups), TaskHead(channels, groups), TaskHead(channels, groups)
        )
        self.epsilon, self.maximum = float(epsilon), float(maximum)

    def set_depth_output_semantics(self, value: str) -> None:
        if value != "conditional_positive_v2":
            raise ValueError("Hydro-v13 checkpoint must use conditional_positive_v2")

    def forward(self, features):
        support_logits = self.support_head(features)
        conditional = F.softplus(self.depth_head(features))
        support = torch.sigmoid(support_logits)
        scale = self.epsilon + self.maximum * torch.sigmoid(self.uncertainty_head(features))
        expected = support * conditional
        return {"depth": conditional, "conditional_depth": conditional,
                "positive_depth": conditional, "expected_depth": expected,
                "support_logits": support_logits, "support_probability": support,
                "uncertainty_scale": scale}

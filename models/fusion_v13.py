"""Content- and reliability-aware asynchronous fusion."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.asynchronous_fusion import masked_modality_softmax
from models.efficient_blocks import residual_block
from models.encoders import ConvNormAct


class ContentAwareFusionPyramid(nn.Module):
    def __init__(self, channels: list[int], reliability_channels: int, dropout: float, groups: int,
                 block_kind: str = "efficient") -> None:
        super().__init__()
        descriptor_channels = reliability_channels + 3 * 8 + 4
        self.summary = nn.ModuleList([nn.ModuleList([nn.Conv2d(w, 8, 1) for _ in range(3)]) for w in channels])
        self.sensor = nn.ModuleList([nn.Conv2d(w, w, 1) for w in channels])
        self.terrain = nn.ModuleList([nn.Conv2d(w, w, 1) for w in channels])
        self.logits = nn.ModuleList([
            nn.Sequential(ConvNormAct(descriptor_channels, 24, 3, groups=4), nn.Conv2d(24, 3, 1))
            for _ in channels
        ])
        self.cross = nn.ModuleList([nn.Conv2d(2 * w, w, 1) for w in channels])
        self.refine = nn.ModuleList([residual_block(block_kind, w, dropout, groups) for w in channels])
        self.cross_gamma = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in channels])

    def forward(self, s1, s2, terrain, reliability, s1_valid, s2_valid, dem_fractions,
                branch_validity=None):
        outputs, sensor_weights, terrain_gates, entropies = [], [], [], []
        for i, (a, b, topo) in enumerate(zip(s1, s2, terrain)):
            size = a.shape[-2:]
            rel = F.interpolate(reliability, size=size, mode="bilinear", align_corners=False)
            a_nearest = F.interpolate(s1_valid, size=size, mode="nearest")
            b_nearest = F.interpolate(s2_valid, size=size, mode="nearest")
            a_fraction = F.adaptive_avg_pool2d(s1_valid, size)
            b_fraction = F.adaptive_avg_pool2d(s2_valid, size)
            dem_fraction = dem_fractions[i]
            sa, sb, st = [layer(value) for layer, value in zip(self.summary[i], (a, b, topo))]
            descriptor = torch.cat((rel, sa, sb, st, a_fraction, b_fraction, dem_fraction,
                                    (a_fraction * b_fraction)), 1)
            logits = self.logits[i](descriptor)
            weights = masked_modality_softmax(logits[:, :2], torch.cat((a_nearest, b_nearest), 1))
            terrain_gate = torch.sigmoid(logits[:, 2:3]) * dem_fraction
            sensors = weights[:, 0:1] * self.sensor[i](a) + weights[:, 1:2] * self.sensor[i](b)
            cross = self.cross[i](torch.cat(((a - b).abs(), a * b), 1))
            both = a_nearest * b_nearest
            fused = sensors + self.cross_gamma[i] * both * cross + terrain_gate * self.terrain[i](topo)
            outputs.append(self.refine[i](fused))
            sensor_weights.append(weights)
            terrain_gates.append(terrain_gate)
            entropies.append(-(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1, keepdim=True))
        return outputs, sensor_weights, terrain_gates, entropies


class ContentAwareFusionPyramidV131(nn.Module):
    """Independent projections and branch-validity-aware fusion for v13.1."""

    def __init__(self, channels: list[int], reliability_channels: int, dropout: float,
                 groups: int, block_kind: str = "efficient") -> None:
        super().__init__()
        self.summary = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(width, 8, 1) for _ in range(3)])
            for width in channels
        ])
        self.s1_projection = nn.ModuleList([nn.Conv2d(width, width, 1) for width in channels])
        self.s2_projection = nn.ModuleList([nn.Conv2d(width, width, 1) for width in channels])
        self.terrain_projection = nn.ModuleList([nn.Conv2d(width, width, 1) for width in channels])
        descriptor_channels = reliability_channels + 24 + 10
        self.logits = nn.ModuleList([
            nn.Sequential(
                ConvNormAct(descriptor_channels, 24, 3, groups=4),
                nn.Conv2d(24, 3, 1),
            ) for _ in channels
        ])
        self.cross = nn.ModuleList([nn.Conv2d(2 * width, width, 1) for width in channels])
        self.refine = nn.ModuleList([
            residual_block(block_kind, width, dropout, groups) for width in channels
        ])
        self.cross_gamma = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in channels])

    def forward(self, s1, s2, terrain, reliability, s1_valid, s2_valid,
                dem_fractions, branch_validity=None):
        outputs, sensor_weights, terrain_gates, entropies = [], [], [], []
        if branch_validity is None:
            branch_validity = {
                key: value for key, value in (
                    ("s1_t1", s1_valid), ("s1_t2", s1_valid),
                    ("s1_change", s1_valid), ("s2_t1", s2_valid),
                    ("s2_t2", s2_valid), ("s2_change", s2_valid),
                )
            }
        for i, (a, b, topo) in enumerate(zip(s1, s2, terrain)):
            size = a.shape[-2:]
            rel = F.interpolate(reliability, size=size, mode="bilinear", align_corners=False)
            s1_available = F.interpolate(s1_valid, size=size, mode="nearest")
            s2_available = F.interpolate(s2_valid, size=size, mode="nearest")
            dem_fraction = dem_fractions[i]
            fractions = [
                F.adaptive_avg_pool2d(branch_validity[key], size)
                for key in ("s1_t1", "s1_t2", "s1_change", "s2_t1", "s2_t2", "s2_change")
            ]
            s1_fraction = torch.maximum(fractions[0], torch.maximum(fractions[1], fractions[2]))
            s2_fraction = torch.maximum(fractions[3], torch.maximum(fractions[4], fractions[5]))
            sa, sb, st = [layer(value) for layer, value in zip(self.summary[i], (a, b, topo))]
            descriptor = torch.cat(
                (rel, sa, sb, st, *fractions, s1_available, s2_available, dem_fraction,
                 s1_available * s2_available), 1
            )
            logits = self.logits[i](descriptor)
            weights = masked_modality_softmax(
                logits[:, :2], torch.cat((s1_available, s2_available), 1)
            )
            terrain_gate = torch.sigmoid(logits[:, 2:3]) * dem_fraction
            projected_s1 = self.s1_projection[i](a)
            projected_s2 = self.s2_projection[i](b)
            projected_terrain = self.terrain_projection[i](topo)
            sensors = weights[:, 0:1] * projected_s1 + weights[:, 1:2] * projected_s2
            cross = self.cross[i](torch.cat(((a - b).abs(), a * b), 1))
            both = s1_available * s2_available
            combined = sensors + self.cross_gamma[i] * both * cross + terrain_gate * projected_terrain
            effective_fraction = torch.maximum(
                torch.maximum(s1_fraction, s2_fraction), dem_fraction
            )
            outputs.append(self.refine[i](combined * effective_fraction))
            sensor_weights.append(weights)
            terrain_gates.append(terrain_gate)
            entropies.append(-(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1, keepdim=True))
        return outputs, sensor_weights, terrain_gates, entropies

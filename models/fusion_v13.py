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

    def forward(self, s1, s2, terrain, reliability, s1_valid, s2_valid, dem_fractions):
        outputs, sensor_weights, terrain_gates = [], [], []
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
        return outputs, sensor_weights, terrain_gates

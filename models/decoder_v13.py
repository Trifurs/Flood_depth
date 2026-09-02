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

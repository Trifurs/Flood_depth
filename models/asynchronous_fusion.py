"""QA- and validity-driven masked asynchronous S1/S2 fusion."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.encoders import ConvNormAct, ResidualBlock


def masked_modality_softmax(
    logits: torch.Tensor, availability: torch.Tensor
) -> torch.Tensor:
    """Normalize over available modalities; return zero if neither is available."""

    available = availability > 0.5
    safe_logits = logits.masked_fill(~available, -1.0e4)
    weights = torch.softmax(safe_logits, dim=1) * available.to(logits.dtype)
    denominator = weights.sum(dim=1, keepdim=True)
    return torch.where(
        denominator > 0, weights / denominator.clamp_min(1e-8), torch.zeros_like(weights)
    )


class AsynchronousFusionPyramid(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        reliability_channels: int = 12,
        dropout: float = 0.1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.s1_projection = nn.ModuleList([nn.Conv2d(width, width, 1) for width in channels])
        self.s2_projection = nn.ModuleList([nn.Conv2d(width, width, 1) for width in channels])
        self.terrain_projection = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for width in channels]
        )
        self.reliability_logits = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(reliability_channels, 16, 3, groups=4),
                    nn.Conv2d(16, 2, 1),
                )
                for _ in channels
            ]
        )
        self.refine = nn.ModuleList(
            [ResidualBlock(width, dropout, groups) for width in channels]
        )

    def forward(
        self,
        s1_features: list[torch.Tensor],
        s2_features: list[torch.Tensor],
        terrain_features: list[torch.Tensor],
        reliability: torch.Tensor,
        s1_valid: torch.Tensor,
        s2_valid: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        fused: list[torch.Tensor] = []
        weights_by_scale: list[torch.Tensor] = []
        for index, (s1, s2, terrain) in enumerate(
            zip(s1_features, s2_features, terrain_features)
        ):
            size = s1.shape[-2:]
            reliability_scale = F.interpolate(
                reliability, size=size, mode="bilinear", align_corners=False
            )
            availability = torch.cat(
                (
                    F.interpolate(s1_valid, size=size, mode="nearest"),
                    F.interpolate(s2_valid, size=size, mode="nearest"),
                ),
                dim=1,
            )
            logits = self.reliability_logits[index](reliability_scale)
            weights = masked_modality_softmax(logits, availability)
            combined = (
                weights[:, 0:1] * self.s1_projection[index](s1)
                + weights[:, 1:2] * self.s2_projection[index](s2)
                + self.terrain_projection[index](terrain)
            )
            fused.append(self.refine[index](combined))
            weights_by_scale.append(weights)
        return fused, weights_by_scale

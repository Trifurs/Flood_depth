"""Small task head shared by the optical-free S1 model family."""

from __future__ import annotations

import torch
from torch import nn

from models.encoders import group_count


class TaskHead(nn.Module):
    """A mask-agnostic convolutional prediction head for dense outputs."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

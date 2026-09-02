"""Independent S1/S2 shared-temporal encoders."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def group_count(channels: int, requested: int = 8) -> int:
    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 8,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(group_count(output_channels, groups), output_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3, groups=groups),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class PyramidBranch(nn.Module):
    def __init__(
        self, input_channels: int, channels: Sequence[int], dropout: float, groups: int
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(input_channels, channels[0], 3, groups=groups),
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

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(inputs)]
        for downsample in self.down:
            features.append(downsample(features[-1]))
        return features


class ModalityTemporalEncoder(nn.Module):
    """Encode T1/T2 with shared weights and change with an independent stem."""

    def __init__(
        self,
        temporal_input_channels: int,
        change_input_channels: int,
        channels: Sequence[int],
        dropout: float = 0.1,
        groups: int = 8,
        incidence_film: bool = False,
    ) -> None:
        super().__init__()
        self.temporal = PyramidBranch(
            temporal_input_channels, channels, dropout, groups
        )
        self.change = PyramidBranch(change_input_channels, channels, dropout, groups)
        self.fusions = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(5 * width, width, 1, groups=groups),
                    ResidualBlock(width, dropout, groups),
                )
                for width in channels
            ]
        )
        self.incidence_film = incidence_film
        self.film = (
            nn.ModuleList([nn.Conv2d(2, 4 * width, 1) for width in channels])
            if incidence_film
            else None
        )

    def forward(
        self, t1: torch.Tensor, t2: torch.Tensor, change: torch.Tensor
    ) -> list[torch.Tensor]:
        pre_features = self.temporal(t1)
        event_features = self.temporal(t2)
        change_features = self.change(change)
        angle_pair = torch.cat((t1[:, 2:3], t2[:, 2:3]), dim=1) if self.incidence_film else None
        outputs: list[torch.Tensor] = []
        for index, (pre, event, changed, fusion) in enumerate(
            zip(pre_features, event_features, change_features, self.fusions)
        ):
            if angle_pair is not None and self.film is not None:
                angle = F.interpolate(angle_pair, size=pre.shape[-2:], mode="bilinear", align_corners=False)
                gamma_pre, beta_pre, gamma_event, beta_event = self.film[index](angle).chunk(4, dim=1)
                pre = pre * (1.0 + 0.2 * torch.tanh(gamma_pre)) + 0.1 * torch.tanh(beta_pre)
                event = event * (1.0 + 0.2 * torch.tanh(gamma_event)) + 0.1 * torch.tanh(beta_event)
            difference = event - pre
            outputs.append(
                fusion(torch.cat((pre, event, difference, difference.abs(), changed), dim=1))
            )
        return outputs

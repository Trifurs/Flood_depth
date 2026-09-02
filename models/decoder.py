"""Shape-explicit lightweight U-Net/FPN decoder."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.encoders import ConvNormAct, ResidualBlock


class DecoderStage(nn.Module):
    def __init__(
        self, input_channels: int, skip_channels: int, output_channels: int, dropout: float, groups: int
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(input_channels + skip_channels, output_channels, 3, groups=groups),
            ResidualBlock(output_channels, dropout, groups),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        height_ratio = skip.shape[-2] / inputs.shape[-2]
        width_ratio = skip.shape[-1] / inputs.shape[-1]
        if not (1.0 <= height_ratio <= 2.1 and 1.0 <= width_ratio <= 2.1):
            raise ValueError(
                f"Unexpected decoder alignment: input={inputs.shape[-2:]}, skip={skip.shape[-2:]}"
            )
        upsampled = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((upsampled, skip), dim=1))


class HydroDecoder(nn.Module):
    def __init__(self, channels: Sequence[int], dropout: float = 0.1, groups: int = 8) -> None:
        super().__init__()
        reversed_channels = list(reversed(channels))
        self.stages = nn.ModuleList(
            [
                DecoderStage(
                    reversed_channels[index],
                    reversed_channels[index + 1],
                    reversed_channels[index + 1],
                    dropout,
                    groups,
                )
                for index in range(len(reversed_channels) - 1)
            ]
        )

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        output = bottleneck
        for stage, skip in zip(self.stages, reversed(skips[:-1])):
            output = stage(output, skip)
        return output

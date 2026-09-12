"""Small-batch efficient convolutional blocks used by the production model."""

from __future__ import annotations

import torch
from torch import nn

from models.encoders import ConvNormAct, group_count


class EfficientResidualBlock(nn.Module):
    """Depthwise residual block with channel attention."""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        groups: int = 8,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(group_count(hidden, groups), hidden),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        squeeze = max(4, channels // 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeeze, 1),
            nn.SiLU(),
            nn.Conv2d(squeeze, channels, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.GroupNorm(group_count(channels, groups), channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.block(inputs)
        update = update * self.channel_gate(update)
        return self.activation(inputs + self.norm(update))


class SpatialResidualBlock(nn.Module):
    """Full-convolution residual block for detail-sensitive depth regression.

    The efficient block above is deliberately inexpensive, but its only spatial
    operator is depthwise and therefore cannot learn cross-channel spatial
    patterns in one step.  Flood-depth regression depends on joint SAR and
    terrain neighbourhoods, so this optional block uses two ordinary 3x3
    convolutions while retaining the batch-size-stable GroupNorm contract.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


def residual_block(kind: str, channels: int, dropout: float, groups: int) -> nn.Module:
    """Construct a configured residual block."""

    if kind == "efficient":
        return EfficientResidualBlock(channels, dropout, groups)
    if kind == "spatial":
        return SpatialResidualBlock(channels, dropout, groups)
    raise ValueError(
        f"Unsupported residual block {kind!r}; expected 'efficient' or 'spatial'"
    )


class EfficientPyramidBranch(nn.Module):
    """Four-scale convolutional feature pyramid."""

    def __init__(
        self,
        inputs: int,
        channels: list[int],
        dropout: float,
        groups: int,
        block_kind: str,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(inputs, channels[0], 3, groups=groups),
            residual_block(block_kind, channels[0], dropout, groups),
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(channels[index - 1], channels[index], 3, 2, groups),
                    residual_block(block_kind, channels[index], dropout, groups),
                )
                for index in range(1, len(channels))
            ]
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        result = [self.stem(inputs)]
        for layer in self.down:
            result.append(layer(result[-1]))
        return result

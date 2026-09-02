"""Small-batch efficient convolutional blocks used by Hydro-v13."""

from __future__ import annotations

import torch
from torch import nn

from models.encoders import ConvNormAct, ResidualBlock, group_count


class EfficientResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0, groups: int = 8, expansion: int = 2) -> None:
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
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, squeeze, 1), nn.SiLU(),
            nn.Conv2d(squeeze, channels, 1), nn.Sigmoid()
        )
        self.norm = nn.GroupNorm(group_count(channels, groups), channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.block(inputs)
        update = update * self.channel_gate(update)
        return self.activation(inputs + self.norm(update))


def residual_block(kind: str, channels: int, dropout: float, groups: int) -> nn.Module:
    if kind == "legacy":
        return ResidualBlock(channels, dropout, groups)
    if kind == "efficient":
        return EfficientResidualBlock(channels, dropout, groups)
    raise ValueError(f"Unknown residual block kind {kind!r}")


class EfficientPyramidBranch(nn.Module):
    def __init__(self, inputs: int, channels: list[int], dropout: float, groups: int, block_kind: str) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(inputs, channels[0], 3, groups=groups),
            residual_block(block_kind, channels[0], dropout, groups),
        )
        self.down = nn.ModuleList([
            nn.Sequential(
                ConvNormAct(channels[i - 1], channels[i], 3, 2, groups),
                residual_block(block_kind, channels[i], dropout, groups),
            ) for i in range(1, len(channels))
        ])

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        result = [self.stem(inputs)]
        for layer in self.down:
            result.append(layer(result[-1]))
        return result


class GatedCrossStateEncoder(nn.Module):
    """Shared pre/event branch plus independent change evidence and validity gate."""

    def __init__(
        self, temporal_channels: int, change_channels: int, channels: list[int],
        dropout: float, groups: int, block_kind: str = "efficient",
        conditioning_channels: int = 0,
    ) -> None:
        super().__init__()
        self.temporal = EfficientPyramidBranch(temporal_channels, channels, dropout, groups, block_kind)
        self.change = EfficientPyramidBranch(change_channels, channels, dropout, groups, block_kind)
        self.state = nn.ModuleList([ConvNormAct(2 * width, width, 1, groups=groups) for width in channels])
        self.delta = nn.ModuleList([ConvNormAct(4 * width, width, 1, groups=groups) for width in channels])
        self.gate = nn.ModuleList([nn.Conv2d(2 * width + 2, width, 1) for width in channels])
        self.refine = nn.ModuleList([residual_block(block_kind, width, dropout, groups) for width in channels])
        self.conditioning = (
            nn.ModuleList([nn.Conv2d(conditioning_channels, 4 * width, 1) for width in channels])
            if conditioning_channels else None
        )

    def forward(
        self, pre: torch.Tensor, event: torch.Tensor, change: torch.Tensor,
        valid: torch.Tensor, conditioning: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        import torch.nn.functional as F
        pre_features = self.temporal(pre)
        event_features = self.temporal(event)
        change_features = self.change(change)
        outputs: list[torch.Tensor] = []
        for i, (before, after, changed) in enumerate(zip(pre_features, event_features, change_features)):
            if self.conditioning is not None:
                if conditioning is None:
                    raise KeyError("S1 conditioning was configured but not supplied")
                cond = F.interpolate(conditioning, before.shape[-2:], mode="bilinear", align_corners=False)
                gp, bp, ge, be = self.conditioning[i](cond).chunk(4, dim=1)
                before = before * (1 + 0.1 * torch.tanh(gp)) + 0.05 * torch.tanh(bp)
                after = after * (1 + 0.1 * torch.tanh(ge)) + 0.05 * torch.tanh(be)
            base = self.state[i](torch.cat((before, after), 1))
            difference = after - before
            evidence = self.delta[i](torch.cat((difference, difference.abs(), before * after, changed), 1))
            fraction = F.adaptive_avg_pool2d(valid, base.shape[-2:])
            availability = F.interpolate(valid, base.shape[-2:], mode="nearest")
            gate = torch.sigmoid(self.gate[i](torch.cat((base, evidence, fraction, availability), 1)))
            outputs.append(self.refine[i](base + gate * evidence * availability))
        return outputs


class DilatedContext(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels, bias=False)
            for d in (1, 2, 4)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels), nn.SiLU()
        )
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.gamma * self.project(torch.cat([path(inputs) for path in self.paths], 1))

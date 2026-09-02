"""MobileNetV2-U-Net flood segmentation adapted from Misra et al. (2025).

The published AI4G workflow uses early-fused Sentinel-1 change indicators with a
MobileNetV2 U-Net.  This is a clean-room PyTorch implementation built from
torchvision's MobileNetV2 blocks; no upstream model source is vendored.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class AI4GFloodExtentNet(nn.Module):
    """Two-channel early-fusion change detector with a MobileNetV2 encoder."""

    input_semantics = (
        "vv_thresholded_flood_change",
        "vh_thresholded_flood_change",
    )

    def __init__(self, decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16)) -> None:
        super().__init__()
        if len(decoder_channels) != 5 or any(channel <= 0 for channel in decoder_channels):
            raise ValueError("decoder_channels must contain five positive integers")
        backbone = mobilenet_v2(weights=None)
        original_stem = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            2,
            original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=False,
        )
        self.encoder = backbone.features
        c0, c1, c2, c3, c4 = decoder_channels
        self.decoder16 = DecoderBlock(1280, 96, c0)
        self.decoder8 = DecoderBlock(c0, 32, c1)
        self.decoder4 = DecoderBlock(c1, 24, c2)
        self.decoder2 = DecoderBlock(c2, 16, c3)
        self.decoder1 = DecoderBlock(c3, 2, c4)
        self.output = nn.Conv2d(c4, 1, 1)

    def forward(self, change_features: torch.Tensor) -> torch.Tensor:
        if change_features.ndim != 4 or change_features.shape[1] != 2:
            raise ValueError(
                "AI4GFloodExtentNet expects [B,2,H,W] VV/VH change indicators, "
                f"got {tuple(change_features.shape)}"
            )
        x = change_features
        skips: dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.encoder):
            x = block(x)
            if index in {1, 3, 6, 13}:
                skips[index] = x
        x = self.decoder16(x, skips[13])
        x = self.decoder8(x, skips[6])
        x = self.decoder4(x, skips[3])
        x = self.decoder2(x, skips[1])
        x = self.decoder1(x, change_features)
        return self.output(x)

"""Independent-gate decoder for the optical-free Hydro-v14 model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import residual_block
from models.encoders import ConvNormAct


class SARHydroDecoder(nn.Module):
    """Decode four SAR/terrain scales with separate residual skip gates."""

    def __init__(
        self,
        channels: Sequence[int],
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "efficient",
        widths: Sequence[int] = (96, 64, 48, 32),
        auxiliary_count: int = 1,
        auxiliary_stage: int = 0,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in widths]
        if widths != [96, 64, 48, 32]:
            raise ValueError("SARHydroDecoder requires widths [96, 64, 48, 32]")
        if auxiliary_count not in {0, 1, 2}:
            raise ValueError("auxiliary_count must be 0, 1, or 2")
        if auxiliary_stage not in {0, 1}:
            raise ValueError("auxiliary_stage must be 0 (1/4) or 1 (1/2)")
        self.widths = widths
        self.auxiliary_stage = int(auxiliary_stage)
        self.bottleneck = ConvNormAct(int(channels[-1]), widths[0], 1, groups=groups)
        skip_indices = (2, 1, 0)
        target_widths = (widths[1], widths[2], widths[3])
        previous_widths = (widths[0], widths[1], widths[2])
        self.up_projection = nn.ModuleList(
            [nn.Conv2d(previous, target, 1) for previous, target in zip(previous_widths, target_widths)]
        )
        self.sar_lateral = nn.ModuleList(
            [nn.Conv2d(int(channels[index]), target, 1) for index, target in zip(skip_indices, target_widths)]
        )
        self.terrain_lateral = nn.ModuleList(
            [nn.Conv2d(int(channels[index]), target, 1) for index, target in zip(skip_indices, target_widths)]
        )
        self.gates = nn.ModuleList(
            [nn.Conv2d(3 * target + 2, 2, 1) for target in target_widths]
        )
        self.refine = nn.ModuleList(
            [residual_block(block_kind, target, dropout, groups) for target in target_widths]
        )
        self.final = residual_block(block_kind, widths[-1], dropout, groups)
        self.auxiliary = (
            nn.ModuleList(
                [nn.Conv2d(target_widths[auxiliary_stage], 1, 1)]
                if auxiliary_count == 1
                else [nn.Conv2d(target_widths[0], 1, 1), nn.Conv2d(target_widths[1], 1, 1)]
            )
            if auxiliary_count
            else None
        )
        self.change_projection = nn.Sequential(
            nn.Conv2d(int(channels[0]), widths[-1], 1),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        sar_skips: Sequence[torch.Tensor],
        terrain_skips: Sequence[torch.Tensor],
        dem_fractions: Sequence[torch.Tensor],
        sensor_valid: torch.Tensor,
        change_evidence: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, Any]]]:
        decoded = self.bottleneck(bottleneck)
        auxiliaries: list[torch.Tensor] = []
        gate_maps: list[dict[str, Any]] = []
        sensor_valid = sensor_valid.to(decoded.dtype)
        for stage, index in enumerate((2, 1, 0)):
            decoded = F.interpolate(
                decoded, sar_skips[index].shape[-2:], mode="bilinear", align_corners=False
            )
            decoded = self.up_projection[stage](decoded)
            sar_skip = self.sar_lateral[stage](sar_skips[index])
            terrain_skip = self.terrain_lateral[stage](terrain_skips[index])
            sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, decoded.shape[-2:])
            dem_fraction = dem_fractions[index].to(decoded.dtype)
            logits = self.gates[stage](
                torch.cat((decoded, sar_skip, terrain_skip, sensor_fraction, dem_fraction), dim=1)
            )
            sar_gate = torch.sigmoid(logits[:, 0:1]) * sensor_fraction
            terrain_gate = torch.sigmoid(logits[:, 1:2]) * dem_fraction
            decoded = self.refine[stage](
                decoded + sar_gate * sar_skip + terrain_gate * terrain_skip
            )
            gate_maps.append({
                "sar": sar_gate,
                "terrain": terrain_gate,
                "sar_mean": sar_gate.mean(),
                "terrain_mean": terrain_gate.mean(),
            })
            if self.auxiliary is not None:
                if len(self.auxiliary) == 1 and stage == self.auxiliary_stage:
                    auxiliaries.append(F.softplus(self.auxiliary[0](decoded)))
                elif len(self.auxiliary) == 2 and stage in {0, 1}:
                    auxiliaries.append(F.softplus(self.auxiliary[stage](decoded)))
        if change_evidence:
            change = F.interpolate(
                self.change_projection(change_evidence[0]),
                decoded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded = decoded + 0.05 * change
        return self.final(decoded), auxiliaries, gate_maps

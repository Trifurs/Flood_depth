"""SAR-main / terrain-residual fusion for optical-free depth estimation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from models.efficient_blocks import residual_block
from models.encoders import ConvNormAct


def _inverse_softplus(value: float) -> float:
    import math

    return math.log(math.expm1(max(float(value), 1.0e-6)))


class SARTerrainResidualFusion(nn.Module):
    """Keep SAR as the main stream and add a small gated terrain residual."""

    def __init__(
        self,
        channels: Sequence[int],
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "efficient",
        terrain_residual_init: float = 0.05,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        self.sar_projection = nn.ModuleList(
            [ConvNormAct(width, width, 1, groups=groups) for width in widths]
        )
        self.terrain_projection = nn.ModuleList(
            [ConvNormAct(width, width, 1, groups=groups) for width in widths]
        )
        self.gate = nn.ModuleList(
            [nn.Conv2d(2 * width + 1, 1, 1) for width in widths]
        )
        self.refine = nn.ModuleList(
            [residual_block(block_kind, width, dropout, groups) for width in widths]
        )
        self.raw_residual_scale = nn.Parameter(
            torch.full((len(widths),), _inverse_softplus(terrain_residual_init))
        )

    @property
    def terrain_residual_scale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_residual_scale)

    def forward(
        self,
        sar: Sequence[torch.Tensor],
        terrain: Sequence[torch.Tensor],
        dem_fractions: Sequence[torch.Tensor],
        sensor_valid: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        outputs: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for index, (sar_feature, terrain_feature) in enumerate(zip(sar, terrain)):
            sar_main = self.sar_projection[index](sar_feature)
            terrain_residual = self.terrain_projection[index](terrain_feature)
            dem_fraction = dem_fractions[index].to(sar_main.dtype)
            sensor_fraction = torch.nn.functional.adaptive_avg_pool2d(
                sensor_valid.to(sar_main.dtype), sar_main.shape[-2:]
            )
            gate = torch.sigmoid(
                self.gate[index](
                    torch.cat((sar_main, terrain_residual, dem_fraction), dim=1)
                )
            ) * dem_fraction
            fused = sar_main * sensor_fraction + self.terrain_residual_scale[index] * gate * terrain_residual
            outputs.append(self.refine[index](fused))
            gates.append(gate)
        diagnostics: dict[str, object] = {
            "terrain_gates": gates,
            "terrain_gate_mean": torch.stack([gate.mean() for gate in gates]).mean(),
            "terrain_residual_scales": self.terrain_residual_scale,
        }
        return outputs, diagnostics

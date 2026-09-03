"""SAR-first hydrology backbone for optical-free flood-depth estimation.

The v14 S1 path used a compact state/change encoder and added terrain as a small
residual.  This module makes the information flow explicit: temporal states,
change evidence, SAR detail, acquisition reliability, QA and hydrologic terrain
proxies are kept as separate streams until each scale is gated.  It never uses
optical data or a multiplicative pre/event interaction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import EfficientPyramidBranch, residual_block
from models.encoders import ConvNormAct, group_count


def _pool_fraction(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(value, size)


class SARHydrologyEncoderV15(nn.Module):
    """Multi-path S1 encoder with reliability- and QA-aware change gating."""

    def __init__(
        self,
        state_channels: int,
        change_channels: int,
        qa_channels: int,
        reliability_channels: int,
        channels: Sequence[int],
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "efficient",
        conditioning_channels: int = 0,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        if len(widths) != 4 or any(value <= 0 for value in widths):
            raise ValueError("SARHydrologyEncoderV15 requires four positive scales")
        if min(state_channels, change_channels, qa_channels, reliability_channels) <= 0:
            raise ValueError("S1 state, change, QA, and reliability channels must be positive")
        self.widths = widths
        detail_widths = [max(16, min(96, width // 2)) for width in widths]
        self.temporal = EfficientPyramidBranch(
            int(state_channels), widths, dropout, groups, block_kind
        )
        self.change = EfficientPyramidBranch(
            int(change_channels), widths, dropout, groups, block_kind
        )
        # Detail sees signed and absolute change at the input resolution.  This
        # retains local SAR scattering edges that a deep state branch can smooth.
        self.detail = EfficientPyramidBranch(
            2 * int(state_channels) + int(change_channels) + 2 * int(state_channels),
            detail_widths,
            dropout,
            groups,
            block_kind,
        )
        self.detail_projection = nn.ModuleList(
            [ConvNormAct(detail_width, width, 1, groups=groups)
             for detail_width, width in zip(detail_widths, widths)]
        )
        self.state_mix = nn.ModuleList(
            [ConvNormAct(2 * width + 2, width, 1, groups=groups) for width in widths]
        )
        self.change_mix = nn.ModuleList(
            [ConvNormAct(3 * width, width, 1, groups=groups) for width in widths]
        )
        self.reliability_projection = nn.ModuleList(
            [ConvNormAct(int(reliability_channels), width, 1, groups=groups)
             for width in widths]
        )
        self.qa_projection = nn.ModuleList(
            [ConvNormAct(int(qa_channels), width, 3, groups=groups) for width in widths]
        )
        # State, change, detail, reliability, QA and three availability scalars.
        self.change_gate = nn.ModuleList(
            [nn.Conv2d(5 * width + 3, width, 1) for width in widths]
        )
        self.qa_gate = nn.ModuleList(
            [nn.Conv2d(width + 2, 1, 1) for width in widths]
        )
        self.refine = nn.ModuleList(
            [residual_block(block_kind, width, dropout, groups) for width in widths]
        )
        self.conditioning = (
            nn.ModuleList(
                [nn.Conv2d(int(conditioning_channels), 4 * width, 1) for width in widths]
            )
            if conditioning_channels
            else None
        )
        if self.conditioning is not None:
            for projection in self.conditioning:
                # Incidence correction starts as an identity path.  The network
                # learns only a small angular residual rather than re-learning SAR.
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    def forward(
        self,
        pre: torch.Tensor,
        event: torch.Tensor,
        change: torch.Tensor,
        qa: torch.Tensor,
        reliability: torch.Tensor,
        valid: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        branch_validity: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        branch_validity = branch_validity or {}
        pre_valid = branch_validity.get("s1_t1", branch_validity.get("t1", valid))
        event_valid = branch_validity.get("s1_t2", branch_validity.get("t2", valid))
        change_valid = branch_validity.get("s1_change", branch_validity.get("change", valid))

        pre_features = self.temporal(pre * pre_valid)
        event_features = self.temporal(event * event_valid)
        change_features = self.change(change * change_valid)
        # The detail path must obey the same branch-validity contract as the
        # pyramid paths.  Otherwise a missing pre/event observation can leak
        # arbitrary fill values into the highest-resolution SAR evidence.
        masked_pre = pre * pre_valid
        masked_event = event * event_valid
        masked_change = change * change_valid
        input_difference = masked_event - masked_pre
        detail_features = self.detail(
            torch.cat(
                (masked_pre, masked_event, masked_change,
                 input_difference, input_difference.abs()),
                dim=1,
            )
        )

        outputs: list[torch.Tensor] = []
        diagnostics: dict[str, Any] = {
            "change_gates": [],
            "quality_gates": [],
            "detail_gates": [],
            "reliability_gates": [],
            "angle_film_amplitude": [],
            "change_evidence": [],
        }
        for index, (before, after, changed, detail) in enumerate(
            zip(pre_features, event_features, change_features, detail_features)
        ):
            size = before.shape[-2:]
            pre_fraction = _pool_fraction(pre_valid, size)
            event_fraction = _pool_fraction(event_valid, size)
            change_fraction = _pool_fraction(change_valid, size)
            observation_fraction = _pool_fraction(valid, size)
            rel = self.reliability_projection[index](
                F.interpolate(reliability, size, mode="bilinear", align_corners=False)
            )
            qa_feature = self.qa_projection[index](
                F.interpolate(qa, size, mode="bilinear", align_corners=False)
            )
            if self.conditioning is not None:
                if conditioning is None:
                    raise KeyError("S1 angle conditioning was configured but absent")
                angle = F.interpolate(conditioning, size, mode="bilinear", align_corners=False)
                gamma_before, beta_before, gamma_after, beta_after = self.conditioning[index](angle).chunk(4, dim=1)
                before = before * (1.0 + 0.10 * torch.tanh(gamma_before)) + 0.05 * torch.tanh(beta_before)
                after = after * (1.0 + 0.10 * torch.tanh(gamma_after)) + 0.05 * torch.tanh(beta_after)
                diagnostics["angle_film_amplitude"].append(
                    torch.cat((gamma_before, beta_before, gamma_after, beta_after), dim=1).abs().mean()
                )
            else:
                diagnostics["angle_film_amplitude"].append(before.sum() * 0.0)

            signed_change = after - before
            state = self.state_mix[index](
                torch.cat((before, after, pre_fraction, event_fraction), dim=1)
            )
            change_evidence = self.change_mix[index](
                torch.cat((signed_change, signed_change.abs(), changed), dim=1)
            )
            detail = self.detail_projection[index](detail)
            quality = torch.sigmoid(
                self.qa_gate[index](torch.cat((qa_feature, event_fraction, change_fraction), dim=1))
            ) * observation_fraction
            gate = torch.sigmoid(
                self.change_gate[index](
                    torch.cat((state, change_evidence, detail, rel, qa_feature,
                               event_fraction, change_fraction, quality), dim=1)
                )
            ) * observation_fraction
            # The additive state path remains available when change is weak.  The
            # reliability stream is a calibration residual, not a sensor modality.
            output = state + gate * change_evidence + 0.20 * detail + 0.10 * rel
            output = output + 0.10 * quality * qa_feature
            output = self.refine[index](output) * observation_fraction
            outputs.append(output)
            diagnostics["change_gates"].append(gate)
            diagnostics["detail_gates"].append(gate * detail.abs().mean(dim=1, keepdim=True))
            diagnostics["reliability_gates"].append(rel.abs().mean(dim=1, keepdim=True))
            diagnostics["quality_gates"].append(quality)
            diagnostics["change_evidence"].append(change_evidence)

        diagnostics["change_gate_mean"] = torch.stack(
            [value.mean() for value in diagnostics["change_gates"]]
        ).mean()
        diagnostics["quality_mean"] = torch.stack(
            [value.mean() for value in diagnostics["quality_gates"]]
        ).mean()
        diagnostics["detail_gate_mean"] = torch.stack(
            [value.mean() for value in diagnostics["detail_gates"]]
        ).mean()
        diagnostics["reliability_residual_mean"] = torch.stack(
            [value.mean() for value in diagnostics["reliability_gates"]]
        ).mean()
        return outputs, diagnostics


class HydrologyContextV15(nn.Module):
    """Stable multi-dilation context with a nonzero but bounded residual."""

    def __init__(self, channels: int, groups: int = 8, dropout: float = 0.05) -> None:
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation,
                          groups=channels, bias=False),
                nn.GroupNorm(group_count(channels, groups), channels),
                nn.SiLU(inplace=True),
            )
            for dilation in (1, 2, 4, 8)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.gamma = nn.Parameter(torch.tensor(0.15))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.gamma.clamp(0.0, 0.50) * self.project(
            torch.cat([path(inputs) for path in self.paths], dim=1)
        )


class S1HydrologyFusionV15(nn.Module):
    """Fuse SAR, terrain and reliability with a hydrologic prior gate."""

    def __init__(
        self,
        channels: Sequence[int],
        reliability_channels: int,
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "efficient",
        terrain_mix_init: float = 0.30,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        self.sar_projection = nn.ModuleList(
            [ConvNormAct(width, width, 1, groups=groups) for width in widths]
        )
        self.terrain_projection = nn.ModuleList(
            [ConvNormAct(width, width, 1, groups=groups) for width in widths]
        )
        self.reliability_projection = nn.ModuleList(
            [ConvNormAct(reliability_channels, width, 1, groups=groups) for width in widths]
        )
        self.hydrology_projection = nn.ModuleList(
            [ConvNormAct(3, width, 1, groups=groups) for width in widths]
        )
        self.terrain_gate = nn.ModuleList(
            [nn.Conv2d(3 * width + 4, 1, 1) for width in widths]
        )
        self.sar_gate = nn.ModuleList(
            [nn.Conv2d(2 * width + 2, 1, 1) for width in widths]
        )
        self.refine = nn.ModuleList(
            [residual_block(block_kind, width, dropout, groups) for width in widths]
        )
        terrain_mix_init = min(max(float(terrain_mix_init), 0.02), 0.95)
        self.raw_terrain_mix = nn.Parameter(
            torch.full((len(widths),), torch.logit(torch.tensor(terrain_mix_init)))
        )

    @property
    def terrain_mix(self) -> torch.Tensor:
        return 0.05 + 0.95 * torch.sigmoid(self.raw_terrain_mix)

    def forward(
        self,
        sar: Sequence[torch.Tensor],
        terrain: Sequence[torch.Tensor],
        physical: Mapping[str, torch.Tensor],
        reliability: torch.Tensor,
        sensor_valid: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        outputs: list[torch.Tensor] = []
        terrain_gates: list[torch.Tensor] = []
        sar_gates: list[torch.Tensor] = []
        for index, (sar_value, terrain_value) in enumerate(zip(sar, terrain)):
            size = sar_value.shape[-2:]
            sar_main = self.sar_projection[index](sar_value)
            terrain_main = self.terrain_projection[index](terrain_value)
            reliability_main = self.reliability_projection[index](
                F.interpolate(reliability, size, mode="bilinear", align_corners=False)
            )
            dem = F.adaptive_avg_pool2d(physical["dem_valid"], size).to(sar_main.dtype)
            relief = F.adaptive_avg_pool2d(physical["local_relief"], size)
            obstacle = F.adaptive_avg_pool2d(physical["obstacle_residual"], size)
            relative = F.adaptive_avg_pool2d(physical["z_relative"], size)
            hydro_raw = torch.cat(
                (torch.tanh(relative / (relief + 1.0)),
                 torch.tanh(relief / 12.0),
                 torch.tanh(obstacle / (relief + 1.0))), dim=1
            ) * dem
            hydro = self.hydrology_projection[index](hydro_raw)
            sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, size).to(sar_main.dtype)
            terrain_gate = torch.sigmoid(
                self.terrain_gate[index](
                    torch.cat((sar_main, terrain_main, reliability_main, dem, hydro_raw), dim=1)
                )
            ) * dem
            sar_gate = torch.sigmoid(
                self.sar_gate[index](torch.cat((sar_main, reliability_main, sensor_fraction, dem), dim=1))
            ) * sensor_fraction
            fused = (
                # SAR is the identity/main stream.  The learned gate controls
                # auxiliary reliability modulation; random initialization must
                # never erase half of the only observation source.
                sar_main
                + 0.10 * sar_gate * reliability_main
                + self.terrain_mix[index] * terrain_gate * terrain_main
                + 0.15 * reliability_main
                + 0.15 * hydro
            )
            outputs.append(self.refine[index](fused))
            terrain_gates.append(terrain_gate)
            sar_gates.append(sar_gate)
        diagnostics = {
            "terrain_gates": terrain_gates,
            "sar_gates": sar_gates,
            "terrain_gate_mean": torch.stack([value.mean() for value in terrain_gates]).mean(),
            "sar_gate_mean": torch.stack([value.mean() for value in sar_gates]).mean(),
            "terrain_mix": self.terrain_mix,
        }
        return outputs, diagnostics

"""Sentinel-1 state/change encoder used by the optical-free Hydro-v14 model.

The encoder has one shared temporal branch for pre/event SAR, a separate change
branch, and an explicit quality gate.  It deliberately keeps the two temporal
states additive: no pre-event product is used as a feature.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import EfficientPyramidBranch, residual_block
from models.encoders import ConvNormAct


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softmax over evidence sources while preserving all-invalid pixels as zero."""

    valid = valid.to(dtype=logits.dtype)
    masked = logits.masked_fill(valid <= 0, -1.0e4)
    weights = torch.softmax(masked, dim=1)
    return weights * valid


class SARStateChangeEncoder(nn.Module):
    """Encode two SAR states and an independent SAR change observation."""

    def __init__(
        self,
        temporal_channels: int,
        change_channels: int,
        qa_channels: int,
        channels: Sequence[int],
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "efficient",
        conditioning_channels: int = 0,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        if len(widths) != 4 or any(value <= 0 for value in widths):
            raise ValueError("SARStateChangeEncoder requires four positive scales")
        if temporal_channels <= 0 or change_channels <= 0 or qa_channels <= 0:
            raise ValueError("SAR state, change, and QA channels must be positive")
        self.widths = widths
        self.temporal = EfficientPyramidBranch(
            temporal_channels, widths, dropout, groups, block_kind
        )
        self.change = EfficientPyramidBranch(
            change_channels, widths, dropout, groups, block_kind
        )
        self.state = nn.ModuleList(
            [ConvNormAct(2 * width, width, 1, groups=groups) for width in widths]
        )
        self.internal = nn.ModuleList(
            [ConvNormAct(2 * width, width, 1, groups=groups) for width in widths]
        )
        self.external = nn.ModuleList(
            [ConvNormAct(width, width, 1, groups=groups) for width in widths]
        )
        self.evidence_logits = nn.ModuleList(
            [nn.Conv2d(2 * width + 3, 2, 1) for width in widths]
        )
        self.refine = nn.ModuleList(
            [residual_block(block_kind, width, dropout, groups) for width in widths]
        )
        self.qa_projection = nn.ModuleList(
            [ConvNormAct(qa_channels, width, 3, groups=groups) for width in widths]
        )
        self.quality_gate = nn.ModuleList(
            [nn.Conv2d(width + 3, 1, 1) for width in widths]
        )
        self.conditioning = (
            nn.ModuleList(
                [nn.Conv2d(conditioning_channels, 4 * width, 1) for width in widths]
            )
            if conditioning_channels
            else None
        )
        if self.conditioning is not None:
            for projection in self.conditioning:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    def forward(
        self,
        pre: torch.Tensor,
        event: torch.Tensor,
        change: torch.Tensor,
        qa: torch.Tensor,
        valid: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        branch_validity: dict[str, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        branch_validity = branch_validity or {}
        pre_valid = branch_validity.get("s1_t1", branch_validity.get("t1", valid))
        event_valid = branch_validity.get("s1_t2", branch_validity.get("t2", valid))
        change_valid = branch_validity.get(
            "s1_change", branch_validity.get("change", valid)
        )
        pre_features = self.temporal(pre * pre_valid)
        event_features = self.temporal(event * event_valid)
        change_features = self.change(change * change_valid)

        outputs: list[torch.Tensor] = []
        diagnostics: dict[str, Any] = {
            "internal_weights": [],
            "external_weights": [],
            "quality_gates": [],
            "angle_film_amplitude": [],
            "change_evidence": [],
        }
        for index, (before, after, changed) in enumerate(
            zip(pre_features, event_features, change_features)
        ):
            if self.conditioning is not None:
                if conditioning is None:
                    raise KeyError("SAR angle conditioning was configured but absent")
                angle = F.interpolate(
                    conditioning,
                    before.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                gamma_before, beta_before, gamma_after, beta_after = self.conditioning[
                    index
                ](angle).chunk(4, dim=1)
                before = before * (1.0 + 0.10 * torch.tanh(gamma_before)) + 0.05 * torch.tanh(beta_before)
                after = after * (1.0 + 0.10 * torch.tanh(gamma_after)) + 0.05 * torch.tanh(beta_after)
                diagnostics["angle_film_amplitude"].append(
                    torch.cat(
                        (gamma_before, beta_before, gamma_after, beta_after), dim=1
                    ).abs().mean()
                )
            else:
                diagnostics["angle_film_amplitude"].append(before.sum() * 0.0)

            # Internal evidence uses signed and absolute temporal change.  There
            # is intentionally no multiplicative pre*event interaction.
            difference = after - before
            state = self.state[index](torch.cat((before, after), dim=1))
            internal = self.internal[index](torch.cat((difference, difference.abs()), dim=1))
            external = self.external[index](changed)
            pre_fraction = F.adaptive_avg_pool2d(pre_valid, state.shape[-2:])
            event_fraction = F.adaptive_avg_pool2d(event_valid, state.shape[-2:])
            change_fraction = F.adaptive_avg_pool2d(change_valid, state.shape[-2:])
            observation_fraction = F.adaptive_avg_pool2d(valid, state.shape[-2:])
            qa_feature = self.qa_projection[index](
                F.interpolate(qa, state.shape[-2:], mode="bilinear", align_corners=False)
            )
            quality = torch.sigmoid(
                self.quality_gate[index](
                    torch.cat(
                        (qa_feature, event_fraction, change_fraction, observation_fraction),
                        dim=1,
                    )
                )
            ) * observation_fraction

            internal_valid = torch.minimum(pre_fraction, event_fraction)
            evidence_valid = torch.cat(
                (internal_valid, change_fraction), dim=1
            )
            logits = self.evidence_logits[index](
                torch.cat((internal, external, event_fraction, change_fraction, quality), dim=1)
            )
            weights = _masked_softmax(logits, evidence_valid)
            evidence = weights[:, 0:1] * internal + weights[:, 1:2] * external
            output = self.refine[index](state + quality * evidence) * observation_fraction
            outputs.append(output)
            diagnostics["internal_weights"].append(weights[:, 0:1])
            diagnostics["external_weights"].append(weights[:, 1:2])
            diagnostics["quality_gates"].append(quality)
            diagnostics["change_evidence"].append(evidence)

        diagnostics["quality_mean"] = torch.stack(
            [value.mean() for value in diagnostics["quality_gates"]]
        ).mean()
        diagnostics["internal_weight_mean"] = torch.stack(
            [value.mean() for value in diagnostics["internal_weights"]]
        ).mean()
        diagnostics["external_weight_mean"] = torch.stack(
            [value.mean() for value in diagnostics["external_weights"]]
        ).mean()
        return outputs, diagnostics

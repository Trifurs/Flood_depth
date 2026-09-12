"""SAR-first hydrology backbone for flood-depth estimation.

Temporal states, change evidence, SAR detail, acquisition reliability, and
hydrologic terrain proxies remain separate until each scale is validity-gated.
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


class SARReliabilityConditioner(nn.Module):
    """Reliability-Conditioning Pyramid (RCP) for acquisition metadata.

    The input schema contains observation count, event day, availability,
    duration, and missingness.  Per-branch raster-valid fractions are appended
    once before each scale projection.  Downstream SAR and terrain paths consume
    the same projected feature instead of re-projecting QA/reliability channels.
    """

    _BRANCH_KEYS = ("s1_t1", "s1_t2", "s1_change")

    def __init__(
        self,
        reliability_channels: int,
        channels: Sequence[int],
        *,
        groups: int = 8,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        if reliability_channels <= 0 or not widths or any(value <= 0 for value in widths):
            raise ValueError("SARReliabilityConditioner requires positive input and scale widths")
        self.reliability_channels = int(reliability_channels)
        self.widths = tuple(widths)
        self.source_channels = self.reliability_channels + len(self._BRANCH_KEYS)
        self.projections = nn.ModuleList(
            [ConvNormAct(self.source_channels, width, 1, groups=groups) for width in widths]
        )

    @staticmethod
    def _size_at_scale(value: torch.Tensor, scale: int) -> tuple[int, int]:
        divisor = 2 ** int(scale)
        height, width = value.shape[-2:]
        return ((height + divisor - 1) // divisor, (width + divisor - 1) // divisor)

    @staticmethod
    def _branch_fraction(
        branch_validity: Mapping[str, torch.Tensor],
        key: str,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        aliases = {
            "s1_t1": "t1",
            "s1_t2": "t2",
            "s1_change": "change",
        }
        value = branch_validity.get(key, branch_validity.get(aliases[key], fallback))
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(f"branch validity {key!r} must have shape (B, 1, H, W)")
        return value

    def forward(
        self,
        reliability: torch.Tensor,
        branch_validity: Mapping[str, torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        if reliability.ndim != 4 or reliability.shape[1] != self.reliability_channels:
            raise ValueError(
                "reliability must have shape (B, "
                f"{self.reliability_channels}, H, W)"
            )
        branch_validity = branch_validity or {}
        fallback = torch.ones_like(reliability[:, :1])
        fractions = [
            self._branch_fraction(branch_validity, key, fallback)
            for key in self._BRANCH_KEYS
        ]
        result: list[torch.Tensor] = []
        for index, projection in enumerate(self.projections):
            size = self._size_at_scale(reliability, index)
            reliability_at_scale = F.interpolate(
                reliability, size, mode="bilinear", align_corners=False
            )
            fractions_at_scale = [
                F.adaptive_avg_pool2d(value, size).to(reliability.dtype)
                for value in fractions
            ]
            result.append(projection(torch.cat((reliability_at_scale, *fractions_at_scale), dim=1)))
        return result

    def zero_features(self, reliability: torch.Tensor) -> list[torch.Tensor]:
        """Return shape-compatible zero features for the RCP ablation.

        The ablation removes all acquisition-reliability and branch-availability
        conditioning while retaining the encoder's audited tensor contracts.
        """

        if reliability.ndim != 4 or reliability.shape[1] != self.reliability_channels:
            raise ValueError(
                "reliability must have shape (B, "
                f"{self.reliability_channels}, H, W)"
            )
        return [
            reliability.new_zeros(
                (reliability.shape[0], width, *self._size_at_scale(reliability, index))
            )
            for index, width in enumerate(self.widths)
        ]


def masked_two_way_change_mixer(
    internal_change: torch.Tensor,
    external_change: torch.Tensor,
    internal_valid_fraction: torch.Tensor,
    external_valid_fraction: torch.Tensor,
    mixer_logit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix only available change evidence without an ``event - 0`` fallback.

    The learned logit is anti-symmetrically paired to keep the production checkpoint
    parameterization intact.  Availability is applied before softmax, so a sole
    valid branch receives an exact weight of one and an all-invalid pixel stays
    finite and exactly zero.
    """

    if internal_change.shape != external_change.shape or internal_change.shape != mixer_logit.shape:
        raise ValueError("internal, external, and mixer-logit tensors must share BCHW shape")
    internal_active = internal_valid_fraction > 0.0
    external_active = external_valid_fraction > 0.0
    internal_active, external_active = torch.broadcast_tensors(
        internal_active, external_active
    )
    availability = torch.stack((internal_active, external_active), dim=1)
    if availability.shape[2] != 1:
        raise ValueError("change-valid fractions must have one channel")
    availability = availability.expand(-1, -1, internal_change.shape[1], -1, -1)
    logits = torch.stack((mixer_logit, -mixer_logit), dim=1)
    masked_logits = logits.masked_fill(~availability, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=1) * availability.to(logits.dtype)
    internal_weight = weights[:, 0]
    external_weight = weights[:, 1]
    mixed = internal_weight * internal_change + external_weight * external_change
    return mixed, internal_weight, external_weight


class JointSARHydrologyEncoder(nn.Module):
    """Joint Temporal-Change SAR Encoder (TCSE).

    The original decomposed TCSE evaluates three full feature pyramids for the
    two temporal states, the supplied change product, and a fourth lightweight
    detail representation.  That separation is useful when acquisitions are
    frequently absent, but it is unnecessarily expensive for the audited
    FloodDepthNet release where complete event observations are required.  This
    encoder exposes the same downstream contract while learning joint spatial
    filters from the states, signed/absolute temporal change, acquisition-angle
    conditioning, and branch-validity indicators in one pyramid.

    Reliability remains a distinct RCP feature pyramid and is injected through
    learned observation gates at every scale.  Consequently, selecting this
    encoder does not merge or invalidate the RCP ablation.
    """

    def __init__(
        self,
        state_channels: int,
        change_channels: int,
        channels: Sequence[int],
        *,
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "spatial",
        conditioning_channels: int = 0,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        if len(widths) != 4 or any(value <= 0 for value in widths):
            raise ValueError("JointSARHydrologyEncoder requires four positive scales")
        if min(state_channels, change_channels) <= 0:
            raise ValueError("S1 state and change channel counts must be positive")
        self.widths = widths
        self.state_channels = int(state_channels)
        self.change_channels = int(change_channels)
        self.conditioning_channels = int(conditioning_channels)
        # pre + event + external change + signed/absolute internal change,
        # optional acquisition angles, and four explicit availability maps.
        input_channels = (
            4 * self.state_channels
            + self.change_channels
            + self.conditioning_channels
            + 4
        )
        self.pyramid = EfficientPyramidBranch(
            input_channels, widths, dropout, groups, block_kind
        )
        self.change_projection = nn.ModuleList(
            [
                ConvNormAct(self.change_channels, width, 1, groups=groups)
                for width in widths
            ]
        )
        # Two scalar gates retain an interpretable distinction between
        # observation quality and the amplitude of the RCP residual.
        self.reliability_gate = nn.ModuleList(
            [nn.Conv2d(2 * width + 3, 2, 1) for width in widths]
        )

    def forward(
        self,
        pre: torch.Tensor,
        event: torch.Tensor,
        change: torch.Tensor,
        valid: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        branch_validity: Mapping[str, torch.Tensor] | None = None,
        reliability_features: Sequence[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        branch_validity = branch_validity or {}
        if reliability_features is None or len(reliability_features) != len(self.widths):
            raise ValueError("joint S1 encoder requires one reliability feature per scale")
        pre_valid = branch_validity.get("s1_t1", branch_validity.get("t1", valid))
        event_valid = branch_validity.get("s1_t2", branch_validity.get("t2", valid))
        change_valid = branch_validity.get(
            "s1_change", branch_validity.get("change", valid)
        )
        pair_valid = torch.minimum(pre_valid, event_valid)
        internal_change = (event - pre) * pair_valid
        if self.conditioning_channels:
            if conditioning is None:
                raise KeyError("S1 angle conditioning was configured but absent")
            conditioned = conditioning * event_valid
        else:
            conditioned = pre.new_zeros(pre.shape[0], 0, *pre.shape[-2:])
        joint = torch.cat(
            (
                pre * pre_valid,
                event * event_valid,
                change * change_valid,
                internal_change,
                internal_change.abs(),
                conditioned,
                pre_valid,
                event_valid,
                change_valid,
                pair_valid,
            ),
            dim=1,
        )
        features = self.pyramid(joint)

        outputs: list[torch.Tensor] = []
        quality_gates: list[torch.Tensor] = []
        reliability_gates: list[torch.Tensor] = []
        change_evidence: list[torch.Tensor] = []
        pair_fractions: list[torch.Tensor] = []
        pre_fractions: list[torch.Tensor] = []
        for index, feature in enumerate(features):
            size = feature.shape[-2:]
            pre_fraction = _pool_fraction(pre_valid, size).to(feature.dtype)
            event_fraction = _pool_fraction(event_valid, size).to(feature.dtype)
            change_fraction = _pool_fraction(change_valid, size).to(feature.dtype)
            pair_fraction = _pool_fraction(pair_valid, size).to(feature.dtype)
            reliability = reliability_features[index]
            if reliability.shape[-2:] != size:
                raise ValueError(
                    "reliability conditioner scale shape does not match joint features"
                )
            projected_change = self.change_projection[index](
                F.interpolate(
                    change * change_valid,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
            ) * change_fraction
            quality_logit, reliability_logit = self.reliability_gate[index](
                torch.cat(
                    (
                        feature,
                        reliability,
                        event_fraction,
                        change_fraction,
                        pair_fraction,
                    ),
                    dim=1,
                )
            ).chunk(2, dim=1)
            quality = torch.sigmoid(quality_logit) * event_fraction
            reliability_gate = torch.sigmoid(reliability_logit) * event_fraction
            output = (
                feature
                + 0.15 * reliability_gate * reliability
                + 0.10 * quality * projected_change
            ) * event_fraction
            outputs.append(output)
            quality_gates.append(quality)
            reliability_gates.append(reliability_gate)
            change_evidence.append(projected_change)
            pair_fractions.append(pair_fraction)
            pre_fractions.append(pre_fraction)

        zero = outputs[0].sum() * 0.0
        diagnostics: dict[str, Any] = {
            "change_gates": quality_gates,
            "quality_gates": quality_gates,
            "detail_gates": quality_gates,
            "reliability_gates": reliability_gates,
            "angle_film_amplitude": zero,
            "change_evidence": change_evidence,
            "internal_change_weights": [],
            "external_change_weights": [],
            "internal_change": [internal_change],
            "pair_valid_fractions": pair_fractions,
            "pre_context_gates": pre_fractions,
            "internal_weight_mean": zero,
            "external_weight_mean": zero,
            "pair_valid_fraction_mean": torch.stack(
                [value.mean() for value in pair_fractions]
            ).mean(),
            "pre_context_gate_mean": torch.stack(
                [value.mean() for value in pre_fractions]
            ).mean(),
            "change_gate_mean": torch.stack(
                [value.mean() for value in quality_gates]
            ).mean(),
            "quality_mean": torch.stack(
                [value.mean() for value in quality_gates]
            ).mean(),
            "detail_gate_mean": torch.stack(
                [value.mean() for value in quality_gates]
            ).mean(),
            "reliability_residual_mean": torch.stack(
                [value.mean() for value in reliability_gates]
            ).mean(),
        }
        return outputs, diagnostics


class SARHydrologyEncoder(nn.Module):
    """Temporal-Change SAR Encoder (TCSE) with reliability-aware gating."""

    def __init__(
        self,
        state_channels: int,
        change_channels: int,
        channels: Sequence[int],
        *,
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "efficient",
        conditioning_channels: int = 0,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        if len(widths) != 4 or any(value <= 0 for value in widths):
            raise ValueError("SARHydrologyEncoder requires four positive scales")
        if min(state_channels, change_channels) <= 0:
            raise ValueError("S1 state and change channel counts must be positive")
        self.widths = widths
        detail_widths = [max(16, min(96, width // 2)) for width in widths]
        # ``temporal`` is called once for pre and once for event. It must be
        # deterministic in train mode so equal acquisitions cannot manufacture a
        # random internal temporal difference. Regularization remains in the
        # post-fusion/refinement paths below.
        self.temporal = EfficientPyramidBranch(
            int(state_channels), widths, 0.0, groups, block_kind
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
        self.reliability_projection = None
        self.qa_projection = None
        # State, internal/external change, detail, conditioned reliability, and
        # three availability scalars. QA only enters the quality sub-gate.
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
        valid: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        branch_validity: Mapping[str, torch.Tensor] | None = None,
        reliability_features: Sequence[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        branch_validity = branch_validity or {}
        if reliability_features is None or len(reliability_features) != len(self.widths):
            raise ValueError("S1 encoder requires one reliability feature per scale")
        pre_valid = branch_validity.get("s1_t1", branch_validity.get("t1", valid))
        event_valid = branch_validity.get("s1_t2", branch_validity.get("t2", valid))
        change_valid = branch_validity.get("s1_change", branch_validity.get("change", valid))

        pair_valid = torch.minimum(pre_valid, event_valid)
        pre_features = self.temporal(pre * pre_valid)
        event_features = self.temporal(event * event_valid)
        change_features = self.change(change * change_valid)
        # The detail path must obey the same branch-validity contract as the
        # pyramid paths.  Otherwise a missing pre/event observation can leak
        # arbitrary fill values into the highest-resolution SAR evidence.
        masked_pre = pre * pair_valid
        masked_event = event * pair_valid
        masked_change = change * change_valid
        input_difference = (event - pre) * pair_valid
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
            "internal_change_weights": [],
            "external_change_weights": [],
            "internal_change": [],
            "pair_valid_fractions": [],
            "pre_context_gates": [],
        }
        for index, (before, after, changed, detail) in enumerate(
            zip(pre_features, event_features, change_features, detail_features)
        ):
            size = before.shape[-2:]
            pre_fraction = _pool_fraction(pre_valid, size)
            event_fraction = _pool_fraction(event_valid, size)
            change_fraction = _pool_fraction(change_valid, size)
            pair_fraction = _pool_fraction(pair_valid, size)
            rel = reliability_features[index]
            if rel.shape[-2:] != size:
                raise ValueError("reliability conditioner scale shape does not match temporal features")
            if self.conditioning is not None:
                if conditioning is None:
                    raise KeyError("S1 angle conditioning was configured but absent")
                angle = F.interpolate(conditioning, size, mode="bilinear", align_corners=False)
                gamma_before, beta_before, gamma_after, beta_after = self.conditioning[index](angle).chunk(4, dim=1)
                # A shared FiLM residual avoids turning identical pre/event
                # states into a synthetic temporal difference.
                gamma = 0.5 * (gamma_before + gamma_after)
                beta = 0.5 * (beta_before + beta_after)
                before = before * (1.0 + 0.10 * torch.tanh(gamma)) + 0.05 * torch.tanh(beta)
                after = after * (1.0 + 0.10 * torch.tanh(gamma)) + 0.05 * torch.tanh(beta)
                diagnostics["angle_film_amplitude"].append(
                    torch.cat((gamma_before, beta_before, gamma_after, beta_after), dim=1).abs().mean()
                )
            else:
                diagnostics["angle_film_amplitude"].append(before.sum() * 0.0)

            # The event state is the always-present main path.  Pre-event
            # information can only enter through a fraction-gated context term.
            event_state = after * event_fraction
            pre_context_state = self.state_mix[index](
                torch.cat((before * pre_fraction, event_state, pre_fraction, event_fraction), dim=1)
            )
            pre_context = pre_context_state * pre_fraction
            # Internal differences are legal only where both temporal branches are
            # present; external change stays independently validity-gated.
            internal_change = (after - before) * pair_fraction
            external_change = changed * change_fraction
            detail = self.detail_projection[index](detail) * torch.maximum(
                pair_fraction, change_fraction
            )
            change_descriptor = self.change_mix[index](
                torch.cat(
                    (internal_change, internal_change.abs(), external_change), dim=1
                )
            ) + detail
            quality = torch.sigmoid(
                self.qa_gate[index](torch.cat((rel, event_fraction, change_fraction), dim=1))
            ) * event_fraction
            gate_parts = [
                event_state, internal_change, external_change, change_descriptor, rel,
            ]
            gate_parts.extend((event_fraction, change_fraction, quality))
            mixer_logit = self.change_gate[index](torch.cat(gate_parts, dim=1))
            mixed_change, internal_weight, external_weight = masked_two_way_change_mixer(
                internal_change,
                external_change,
                pair_fraction,
                change_fraction,
                mixer_logit,
            )
            # No valid internal/external evidence produces an exact zero residual;
            # event availability remains the only requirement for an output state.
            output = self.refine[index](
                event_state + pre_context + quality * mixed_change
            ) * event_fraction
            outputs.append(output)
            diagnostics["change_gates"].append(quality)
            diagnostics["detail_gates"].append(
                torch.maximum(pair_fraction, change_fraction) * detail.abs().mean(dim=1, keepdim=True)
            )
            diagnostics["reliability_gates"].append(rel.abs().mean(dim=1, keepdim=True))
            diagnostics["quality_gates"].append(quality)
            diagnostics["change_evidence"].append(mixed_change)
            diagnostics["internal_change_weights"].append(internal_weight)
            diagnostics["external_change_weights"].append(external_weight)
            diagnostics["internal_change"].append(internal_change)
            diagnostics["pair_valid_fractions"].append(pair_fraction)
            diagnostics["pre_context_gates"].append(pre_fraction)

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
        diagnostics["internal_weight_mean"] = torch.stack(
            [value.mean() for value in diagnostics["internal_change_weights"]]
        ).mean()
        diagnostics["external_weight_mean"] = torch.stack(
            [value.mean() for value in diagnostics["external_change_weights"]]
        ).mean()
        diagnostics["pair_valid_fraction_mean"] = torch.stack(
            [value.mean() for value in diagnostics["pair_valid_fractions"]]
        ).mean()
        diagnostics["pre_context_gate_mean"] = torch.stack(
            [value.mean() for value in diagnostics["pre_context_gates"]]
        ).mean()
        return outputs, diagnostics


class HydrologyContext(nn.Module):
    """Multi-dilation context aggregator with a bounded residual update."""

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        dropout: float = 0.05,
        global_context_enabled: bool = False,
    ) -> None:
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
        self.global_context = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(16, channels // 4), 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(16, channels // 4), channels, 1),
            )
            if global_context_enabled
            else None
        )
        if self.global_context is not None:
            final = self.global_context[-1]
            assert isinstance(final, nn.Conv2d)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs + self.gamma.clamp(0.0, 0.50) * self.project(
            torch.cat([path(inputs) for path in self.paths], dim=1)
        )
        if self.global_context is not None:
            output = output + self.global_context(inputs)
        return output


class S1HydrologyFusion(nn.Module):
    """Terrain-Conditioned Fusion (TCF) of SAR, topography, and reliability."""

    def __init__(
        self,
        channels: Sequence[int],
        *,
        dropout: float = 0.10,
        groups: int = 8,
        block_kind: str = "efficient",
        terrain_mix_init: float = 0.30,
        terrain_alpha_max: float = 1.0,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        self.sar_projection = nn.ModuleList(
            [ConvNormAct(width, width, 1, groups=groups) for width in widths]
        )
        self.terrain_projection = nn.ModuleList(
            [ConvNormAct(width, width, 1, groups=groups) for width in widths]
        )
        # Reliability is projected once by ``SARReliabilityConditioner`` and
        # reused by both the SAR and terrain paths.
        self.reliability_projection = None
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
        self.terrain_alpha_max = float(terrain_alpha_max)
        terrain_mix_init = float(terrain_mix_init)
        if self.terrain_alpha_max <= 0.0:
            raise ValueError("terrain_alpha_max must be positive")
        if not 0.0 < terrain_mix_init < self.terrain_alpha_max:
            raise ValueError(
                "terrain_mix_init must lie strictly between 0 and terrain_alpha_max "
                "for the sigmoid parameterization"
            )
        self.raw_terrain_mix = nn.Parameter(
            torch.full(
                (len(widths),),
                torch.logit(
                    torch.tensor(terrain_mix_init / self.terrain_alpha_max)
                ),
            )
        )

    @property
    def terrain_mix(self) -> torch.Tensor:
        return self.terrain_alpha_max * torch.sigmoid(self.raw_terrain_mix)

    def disable_terrain_conditioned_paths(self) -> None:
        """Freeze paths that are bypassed by the ``w/o TCF`` ablation."""

        for module in (
            self.terrain_projection,
            self.hydrology_projection,
            self.terrain_gate,
            self.sar_gate,
        ):
            module.requires_grad_(False)
        self.raw_terrain_mix.requires_grad_(False)

    def forward(
        self,
        sar: Sequence[torch.Tensor],
        terrain: Sequence[torch.Tensor],
        physical: Mapping[str, torch.Tensor],
        reliability: Sequence[torch.Tensor],
        sensor_valid: torch.Tensor,
        *,
        terrain_conditioned_fusion_enabled: bool = True,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        outputs: list[torch.Tensor] = []
        terrain_gates: list[torch.Tensor] = []
        sar_gates: list[torch.Tensor] = []
        if not isinstance(reliability, (list, tuple)) or len(reliability) != len(self.sar_projection):
            raise ValueError("fusion requires one reliability feature per scale")
        for index, (sar_value, terrain_value) in enumerate(zip(sar, terrain)):
            size = sar_value.shape[-2:]
            sar_main = self.sar_projection[index](sar_value)
            reliability_main = reliability[index]
            if reliability_main.shape[-2:] != size:
                raise ValueError("reliability conditioner scale shape does not match fusion features")
            sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, size).to(sar_main.dtype)
            if terrain_conditioned_fusion_enabled:
                terrain_main = self.terrain_projection[index](terrain_value)
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
                terrain_gate = torch.sigmoid(
                    self.terrain_gate[index](
                        torch.cat((sar_main, terrain_main, reliability_main, dem, hydro_raw), dim=1)
                    )
                ) * dem
                sar_gate = torch.sigmoid(
                    self.sar_gate[index](
                        torch.cat((sar_main, reliability_main, sensor_fraction, dem), dim=1)
                    )
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
            else:
                # ``w/o TCF`` retains the SAR stream and RCP but removes every
                # terrain/proxy-derived additive path from this fusion block.
                terrain_gate = sensor_fraction.new_zeros(sensor_fraction.shape)
                sar_gate = sensor_fraction
                fused = sar_main + 0.10 * sar_gate * reliability_main + 0.15 * reliability_main
            outputs.append(self.refine[index](fused))
            terrain_gates.append(terrain_gate)
            sar_gates.append(sar_gate)
        diagnostics = {
            "terrain_gates": terrain_gates,
            "sar_gates": sar_gates,
            "terrain_gate_mean": torch.stack([value.mean() for value in terrain_gates]).mean(),
            "sar_gate_mean": torch.stack([value.mean() for value in sar_gates]).mean(),
            "terrain_mix": (
                self.terrain_mix
                if terrain_conditioned_fusion_enabled
                else torch.zeros_like(self.terrain_mix)
            ),
            "terrain_conditioned_fusion_enabled": bool(terrain_conditioned_fusion_enabled),
        }
        return outputs, diagnostics

"""Lean S1 state/change and terrain residual paths for PA-HydroKAN-S1-V15.1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import EfficientPyramidBranch, residual_block
from models.encoders import ConvNormAct
from models.s1_hydrology_backbone_v15 import _pool_fraction, masked_two_way_change_mixer


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


class SARHydrologyEncoderV15_1Simple(nn.Module):
    """Event-state-first S1 encoder with one masked internal/external mixer."""

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
        pre_alpha_init: float = 0.10,
        deduplicated_reliability: bool = False,
        absolute_sar_shortcut_enabled: bool = False,
        absolute_sar_shortcut_init: float = 0.03,
        absolute_sar_shortcut_max: float = 0.10,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        if len(widths) != 4 or any(value <= 0 for value in widths):
            raise ValueError("SARHydrologyEncoderV15_1Simple requires four positive scales")
        if not 0.0 < pre_alpha_init < 0.5:
            raise ValueError("pre_alpha_init must lie in (0, 0.5)")
        if qa_channels < 0:
            raise ValueError("qa_channels must be nonnegative")
        if qa_channels == 0 and not deduplicated_reliability:
            raise ValueError("zero QA channels require deduplicated_reliability=true")
        if absolute_sar_shortcut_enabled and not (
            0.0 < absolute_sar_shortcut_init < absolute_sar_shortcut_max
        ):
            raise ValueError(
                "absolute_sar_shortcut_init must lie in (0, absolute_sar_shortcut_max)"
            )
        self.widths = widths
        self.deduplicated_reliability = bool(deduplicated_reliability)
        # The shared temporal encoder must not draw independent dropout masks for
        # pre/event passes. Dropout remains in ``refine`` after change fusion.
        self.event_pre = EfficientPyramidBranch(state_channels, widths, 0.0, groups, block_kind)
        self.external_change = EfficientPyramidBranch(change_channels, widths, dropout, groups, block_kind)
        self.conditioner = (
            None if self.deduplicated_reliability else nn.ModuleList(
                [ConvNormAct(qa_channels + reliability_channels, width, 1, groups=groups) for width in widths]
            )
        )
        self.pre_gate = nn.ModuleList(
            [nn.Conv2d(2 * width + 2, 1, 1) for width in widths]
        )
        self.change_mixer = nn.ModuleList(
            [nn.Conv2d(4 * width + 3, width, 1) for width in widths]
        )
        self.change_amplitude = nn.ModuleList(
            [nn.Conv2d(width + 2, 1, 1) for width in widths]
        )
        self.refine = nn.ModuleList(
            [residual_block(block_kind, width, dropout, groups) for width in widths]
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
        self.raw_pre_alpha = nn.Parameter(
            torch.full((len(widths),), _logit(pre_alpha_init / 0.5))
        )
        self.absolute_sar_shortcut_enabled = bool(absolute_sar_shortcut_enabled)
        if self.absolute_sar_shortcut_enabled:
            # This path intentionally has no GroupNorm: it carries the already
            # standardized absolute event VV/VH level that normalization layers
            # in the main encoder can attenuate.  It remains a small residual.
            # Fork the CPU RNG while constructing this optional branch.  Without
            # this isolation, enabling it would consume random draws and shift
            # the initial weights of all later common modules, invalidating the
            # one-variable shortcut ablation despite a shared seed.
            with torch.random.fork_rng(devices=[]):
                self.absolute_sar_stem = nn.Conv2d(
                    state_channels, widths[0], 1, bias=False
                )
                self.absolute_sar_level_projection = nn.ModuleList(
                    [nn.Conv2d(widths[0], width, 1, bias=False) for width in widths]
                )
                nn.init.orthogonal_(self.absolute_sar_stem.weight)
                for projection in self.absolute_sar_level_projection:
                    nn.init.orthogonal_(projection.weight)
            self.absolute_sar_shortcut_max = float(absolute_sar_shortcut_max)
            self.raw_absolute_sar_shortcut_scale = nn.Parameter(
                torch.full(
                    (len(widths),),
                    _logit(absolute_sar_shortcut_init / absolute_sar_shortcut_max),
                )
            )
        else:
            self.absolute_sar_stem = None
            self.absolute_sar_level_projection = None
            self.absolute_sar_shortcut_max = 0.0
            self.register_parameter("raw_absolute_sar_shortcut_scale", None)

    @property
    def pre_alpha(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.raw_pre_alpha)

    @property
    def absolute_sar_shortcut_scale(self) -> torch.Tensor:
        if self.raw_absolute_sar_shortcut_scale is None:
            raise RuntimeError("absolute SAR shortcut is disabled")
        return self.absolute_sar_shortcut_max * torch.sigmoid(
            self.raw_absolute_sar_shortcut_scale
        )

    @staticmethod
    def _integer_average_downsample(
        value: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        """Deterministically downsample the fixed pyramid by integer averaging."""

        source_height, source_width = value.shape[-2:]
        target_height, target_width = size
        if (
            source_height % target_height != 0
            or source_width % target_width != 0
        ):
            raise ValueError(
                "absolute SAR shortcut requires integer pyramid downsampling; "
                f"got source={tuple(value.shape[-2:])}, target={size}"
            )
        return F.avg_pool2d(
            value,
            kernel_size=(source_height // target_height, source_width // target_width),
            stride=(source_height // target_height, source_width // target_width),
        )

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
        reliability_features: Sequence[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        branch_validity = branch_validity or {}
        if self.deduplicated_reliability:
            if reliability_features is None or len(reliability_features) != len(self.widths):
                raise ValueError("deduplicated S1 encoder requires one reliability feature per scale")
        elif reliability_features is not None:
            raise ValueError("legacy S1 encoder does not accept preconditioned reliability features")
        pre_valid = branch_validity.get("s1_t1", branch_validity.get("t1", valid))
        event_valid = branch_validity.get("s1_t2", branch_validity.get("t2", valid))
        change_valid = branch_validity.get("s1_change", branch_validity.get("change", valid))
        pair_valid = torch.minimum(pre_valid, event_valid)
        pre_features = self.event_pre(pre * pre_valid)
        event_features = self.event_pre(event * event_valid)
        external_features = self.external_change(change * change_valid)
        absolute_sar = (
            self.absolute_sar_stem(event * event_valid)
            if self.absolute_sar_stem is not None
            else None
        )
        outputs: list[torch.Tensor] = []
        diagnostics: dict[str, Any] = {
            "change_evidence": [],
            "internal_change_weights": [],
            "external_change_weights": [],
            "pair_valid_fractions": [],
            "pre_context_gates": [],
            "change_gates": [],
            "quality_gates": [],
            "angle_film_amplitude": [],
            "angle_gamma_rms_by_scale": [],
            "angle_beta_rms_by_scale": [],
            "angle_conditioner_output_input_rms_ratio_by_scale": [],
            "internal_change": [],
            "absolute_sar_shortcut_residual_ratios": [],
        }
        for index, (before, after, external) in enumerate(
            zip(pre_features, event_features, external_features)
        ):
            size = before.shape[-2:]
            pre_fraction = _pool_fraction(pre_valid, size)
            event_fraction = _pool_fraction(event_valid, size)
            change_fraction = _pool_fraction(change_valid, size)
            pair_fraction = _pool_fraction(pair_valid, size)
            if self.conditioning is not None:
                if conditioning is None:
                    raise KeyError("S1 angle conditioning was configured but absent")
                angle = F.interpolate(conditioning, size, mode="bilinear", align_corners=False)
                gamma_pre, beta_pre, gamma_event, beta_event = self.conditioning[index](angle).chunk(4, dim=1)
                input_rms = torch.cat((before, after), dim=1).float().square().mean().sqrt()
                gamma = 0.5 * (gamma_pre + gamma_event)
                beta = 0.5 * (beta_pre + beta_event)
                before = before * (1.0 + 0.10 * torch.tanh(gamma)) + 0.05 * torch.tanh(beta)
                after = after * (1.0 + 0.10 * torch.tanh(gamma)) + 0.05 * torch.tanh(beta)
                diagnostics["angle_film_amplitude"].append(
                    torch.cat((gamma_pre, beta_pre, gamma_event, beta_event), dim=1).abs().mean()
                )
                gamma_rms = torch.cat((gamma_pre, gamma_event), dim=1).float().square().mean().sqrt()
                beta_rms = torch.cat((beta_pre, beta_event), dim=1).float().square().mean().sqrt()
                diagnostics["angle_gamma_rms_by_scale"].append(gamma_rms)
                diagnostics["angle_beta_rms_by_scale"].append(beta_rms)
                diagnostics["angle_conditioner_output_input_rms_ratio_by_scale"].append(
                    (gamma_rms + beta_rms) / input_rms.clamp_min(1.0e-6)
                )
            else:
                diagnostics["angle_film_amplitude"].append(after.sum() * 0.0)
                zero = after.sum() * 0.0
                diagnostics["angle_gamma_rms_by_scale"].append(zero)
                diagnostics["angle_beta_rms_by_scale"].append(zero)
                diagnostics["angle_conditioner_output_input_rms_ratio_by_scale"].append(zero)
            condition = (
                reliability_features[index]
                if self.deduplicated_reliability
                else self.conditioner[index](
                    torch.cat(
                        (
                            F.interpolate(qa, size, mode="bilinear", align_corners=False),
                            F.interpolate(reliability, size, mode="bilinear", align_corners=False),
                        ),
                        dim=1,
                    )
                )
            )
            if condition.shape[-2:] != size:
                raise ValueError("reliability conditioner scale shape does not match temporal features")
            event_state = after * event_fraction
            pre_gate = torch.sigmoid(
                self.pre_gate[index](
                    torch.cat((event_state, before * pre_fraction, pre_fraction, event_fraction), dim=1)
                )
            ) * pre_fraction
            internal = (after - before) * pair_fraction
            external = external * change_fraction
            mixer_logit = self.change_mixer[index](
                torch.cat(
                    (event_state, internal, external, condition, event_fraction, change_fraction, pair_fraction),
                    dim=1,
                )
            )
            mixed, internal_weight, external_weight = masked_two_way_change_mixer(
                internal, external, pair_fraction, change_fraction, mixer_logit
            )
            change_gate = torch.sigmoid(
                self.change_amplitude[index](torch.cat((condition, event_fraction, change_fraction), dim=1))
            ) * event_fraction
            output = self.refine[index](
                event_state
                + self.pre_alpha[index] * pre_gate * before
                + change_gate * mixed
            ) * event_fraction
            if absolute_sar is not None:
                if self.absolute_sar_level_projection is None:
                    raise RuntimeError("absolute SAR level projections are unexpectedly absent")
                shortcut = self.absolute_sar_level_projection[index](
                    self._integer_average_downsample(absolute_sar, size)
                ) * event_fraction
                residual = self.absolute_sar_shortcut_scale[index] * shortcut
                main_output = output
                output = main_output + residual
                diagnostics["absolute_sar_shortcut_residual_ratios"].append(
                    residual.float().square().mean().sqrt()
                    / main_output.detach().float().square().mean().sqrt().clamp_min(1.0e-6)
                )
            else:
                diagnostics["absolute_sar_shortcut_residual_ratios"].append(
                    output.sum() * 0.0
                )
            outputs.append(output)
            diagnostics["change_evidence"].append(mixed)
            diagnostics["internal_change_weights"].append(internal_weight)
            diagnostics["external_change_weights"].append(external_weight)
            diagnostics["internal_change"].append(internal)
            diagnostics["pair_valid_fractions"].append(pair_fraction)
            diagnostics["pre_context_gates"].append(pre_gate)
            diagnostics["change_gates"].append(change_gate)
            diagnostics["quality_gates"].append(change_gate)
        for name, values in (
            ("internal_weight_mean", diagnostics["internal_change_weights"]),
            ("external_weight_mean", diagnostics["external_change_weights"]),
            ("pair_valid_fraction_mean", diagnostics["pair_valid_fractions"]),
            ("pre_context_gate_mean", diagnostics["pre_context_gates"]),
            ("change_gate_mean", diagnostics["change_gates"]),
            ("quality_mean", diagnostics["quality_gates"]),
        ):
            diagnostics[name] = torch.stack([value.mean() for value in values]).mean()
        diagnostics["angle_gamma_rms"] = torch.stack(
            diagnostics["angle_gamma_rms_by_scale"]
        ).mean()
        diagnostics["angle_beta_rms"] = torch.stack(
            diagnostics["angle_beta_rms_by_scale"]
        ).mean()
        diagnostics["angle_conditioner_output_input_rms_ratio"] = torch.stack(
            diagnostics["angle_conditioner_output_input_rms_ratio_by_scale"]
        ).mean()
        diagnostics["absolute_sar_shortcut_scale"] = (
            self.absolute_sar_shortcut_scale
            if self.absolute_sar_shortcut_enabled
            else outputs[0].new_zeros(len(self.widths))
        )
        diagnostics["absolute_sar_shortcut_residual_input_rms_ratio"] = torch.stack(
            diagnostics["absolute_sar_shortcut_residual_ratios"]
        ).mean()
        return outputs, diagnostics


class S1TerrainResidualFusionV15_1(nn.Module):
    """One optional terrain residual; missing DEM leaves the SAR path unchanged."""

    def __init__(
        self,
        channels: Sequence[int],
        reliability_channels: int,
        *,
        groups: int = 8,
        terrain_alpha_init: float = 0.05,
        terrain_alpha_max: float = 1.0,
        deduplicated_reliability: bool = False,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in channels]
        if not 0.0 < terrain_alpha_init < terrain_alpha_max:
            raise ValueError("terrain_alpha_init must lie in (0, terrain_alpha_max)")
        self.deduplicated_reliability = bool(deduplicated_reliability)
        self.terrain_alpha_max = float(terrain_alpha_max)
        self.terrain_projection = nn.ModuleList(
            [ConvNormAct(width, width, 1, groups=groups) for width in widths]
        )
        self.reliability_projection = (
            None if self.deduplicated_reliability else nn.ModuleList(
                [ConvNormAct(reliability_channels, width, 1, groups=groups) for width in widths]
            )
        )
        self.terrain_gate = nn.ModuleList(
            [nn.Conv2d(3 * width + 2, 1, 1) for width in widths]
        )
        self.raw_terrain_alpha = nn.Parameter(
            torch.full(
                (len(widths),),
                _logit(float(terrain_alpha_init) / float(terrain_alpha_max)),
            )
        )

    @property
    def terrain_alpha(self) -> torch.Tensor:
        return self.terrain_alpha_max * torch.sigmoid(self.raw_terrain_alpha)

    def forward(
        self,
        sar: Sequence[torch.Tensor],
        terrain: Sequence[torch.Tensor],
        physical: Mapping[str, torch.Tensor],
        reliability: torch.Tensor | Sequence[torch.Tensor],
        sensor_valid: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        outputs, gates, residual_ratios = [], [], []
        if self.deduplicated_reliability:
            if not isinstance(reliability, (list, tuple)) or len(reliability) != len(self.terrain_projection):
                raise ValueError("deduplicated terrain fusion requires one reliability feature per scale")
        elif not isinstance(reliability, torch.Tensor):
            raise ValueError("legacy terrain fusion requires a raw reliability tensor")
        for index, (sar_value, terrain_value) in enumerate(zip(sar, terrain)):
            size = sar_value.shape[-2:]
            terrain_main = self.terrain_projection[index](terrain_value)
            reliability_main = (
                reliability[index]
                if self.deduplicated_reliability
                else self.reliability_projection[index](
                    F.interpolate(reliability, size, mode="bilinear", align_corners=False)
                )
            )
            if reliability_main.shape[-2:] != size:
                raise ValueError("reliability conditioner scale shape does not match terrain features")
            dem_fraction = F.adaptive_avg_pool2d(physical["dem_valid"], size).to(sar_value.dtype)
            sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, size).to(sar_value.dtype)
            gate = torch.sigmoid(
                self.terrain_gate[index](
                    torch.cat((sar_value, terrain_main, reliability_main, dem_fraction, sensor_fraction), dim=1)
                )
            ) * dem_fraction
            residual = self.terrain_alpha[index] * gate * terrain_main
            outputs.append(sar_value + residual)
            gates.append(gate)
            residual_ratios.append(
                residual.float().square().mean().sqrt()
                / sar_value.float().square().mean().sqrt().clamp_min(1.0e-6)
            )
        return outputs, {
            "terrain_gates": gates,
            "terrain_alpha": self.terrain_alpha,
            "terrain_gate_mean": torch.stack([value.mean() for value in gates]).mean(),
            "terrain_residual_input_rms_ratio": torch.stack(residual_ratios).mean(),
        }

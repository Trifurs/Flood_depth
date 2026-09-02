"""Clean-room DLSIM-style comparison networks for the local raster contract.

The public DLSIM experiment uses a two-channel binary-change/slope input and
offers LinkNet and Attention U-Net backbones.  This module implements those two
architectural ideas from scratch.  The local dataset has no independent binary
flood-extent raster, so a learned 1x1 projection converts the seven audited S1/S2
change bands into one label-free change-evidence channel.  The second channel is
the normalized slope, matching DLSIM's terrain-conditioning role.

No source code from the upstream repository is copied.  Project-level output
heads, masks, losses, metrics, checkpoints, and train/val/test splits remain
shared with PA-HydroKAN so the resulting comparison is controlled and runnable.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.encoders import group_count
from models.heads import FloodDepthHeads
from models.terrain_features import masked_average


REQUIRED_INPUTS = {
    "s1_change",
    "s2_change",
    "terrain",
    "terrain_raw",
    "s1_valid",
    "s2_valid",
    "dem_valid",
}
FORBIDDEN_INPUTS = {
    "label",
    "masks",
    "flood_mask",
    "valid_depth_mask",
    "split",
    "sample_origin",
}


class ConvBlock(nn.Module):
    """Two-convolution block with small-batch-safe normalization."""

    def __init__(
        self, input_channels: int, output_channels: int, dropout: float, groups: int
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(output_channels, groups), output_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend(
            [
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(group_count(output_channels, groups), output_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpsampleProjection(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, groups: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(output_channels, groups), output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        resized = F.interpolate(inputs, size=size, mode="bilinear", align_corners=False)
        return self.projection(resized)


class LinkNetCore(nn.Module):
    """Additive-skip encoder-decoder corresponding to the DLSIM LinkNet option."""

    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        dropout: float,
        groups: int,
    ) -> None:
        super().__init__()
        if len(channels) < 3:
            raise ValueError("DLSIM LinkNet requires at least three resolution levels")
        self.encoders = nn.ModuleList(
            [
                ConvBlock(
                    input_channels if index == 0 else channels[index - 1],
                    width,
                    dropout,
                    groups,
                )
                for index, width in enumerate(channels)
            ]
        )
        reversed_pairs = list(zip(reversed(channels[1:]), reversed(channels[:-1])))
        self.upsamples = nn.ModuleList(
            [UpsampleProjection(source, target, groups) for source, target in reversed_pairs]
        )
        self.refinements = nn.ModuleList(
            [ConvBlock(target, target, dropout, groups) for _, target in reversed_pairs]
        )
        self.output_channels = int(channels[0])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded: list[torch.Tensor] = []
        features = inputs
        for index, encoder in enumerate(self.encoders):
            if index > 0:
                features = F.max_pool2d(features, 2)
            features = encoder(features)
            encoded.append(features)
        decoded = encoded[-1]
        for upsample, refinement, skip in zip(
            self.upsamples, self.refinements, reversed(encoded[:-1])
        ):
            decoded = upsample(decoded, skip.shape[-2:])
            decoded = refinement(decoded + skip)
        return decoded


class AttentionGate(nn.Module):
    def __init__(
        self, gate_channels: int, skip_channels: int, groups: int
    ) -> None:
        super().__init__()
        intermediate = max(1, min(gate_channels, skip_channels) // 2)
        norm_groups = group_count(intermediate, groups)
        self.gate = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate, 1, bias=False),
            nn.GroupNorm(norm_groups, intermediate),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate, 1, bias=False),
            nn.GroupNorm(norm_groups, intermediate),
        )
        self.score = nn.Conv2d(intermediate, 1, 1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = torch.sigmoid(self.score(F.relu(self.gate(gate) + self.skip(skip))))
        return skip * attention


class AttentionUNetCore(nn.Module):
    """Concatenative decoder with learned skip gates, as in DLSIM's AttU-Net option."""

    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        dropout: float,
        groups: int,
    ) -> None:
        super().__init__()
        if len(channels) < 3:
            raise ValueError("DLSIM Attention U-Net requires at least three resolution levels")
        self.encoders = nn.ModuleList(
            [
                ConvBlock(
                    input_channels if index == 0 else channels[index - 1],
                    width,
                    dropout,
                    groups,
                )
                for index, width in enumerate(channels)
            ]
        )
        reversed_pairs = list(zip(reversed(channels[1:]), reversed(channels[:-1])))
        self.upsamples = nn.ModuleList(
            [UpsampleProjection(source, target, groups) for source, target in reversed_pairs]
        )
        self.attention = nn.ModuleList(
            [AttentionGate(target, target, groups) for _, target in reversed_pairs]
        )
        self.decoders = nn.ModuleList(
            [ConvBlock(2 * target, target, dropout, groups) for _, target in reversed_pairs]
        )
        self.output_channels = int(channels[0])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded: list[torch.Tensor] = []
        features = inputs
        for index, encoder in enumerate(self.encoders):
            if index > 0:
                features = F.max_pool2d(features, 2)
            features = encoder(features)
            encoded.append(features)
        decoded = encoded[-1]
        for upsample, attention, decoder, skip in zip(
            self.upsamples,
            self.attention,
            self.decoders,
            reversed(encoded[:-1]),
        ):
            decoded = upsample(decoded, skip.shape[-2:])
            gated_skip = attention(decoded, skip)
            decoded = decoder(torch.cat((gated_skip, decoded), dim=1))
        return decoded


class DLSIMInputAdapter(nn.Module):
    """Map audited S1/S2 change bands and slope to DLSIM's two input roles."""

    def __init__(self) -> None:
        super().__init__()
        self.change_projection = nn.Conv2d(7, 1, 1)
        nn.init.constant_(self.change_projection.weight, 1.0 / 7.0)
        nn.init.zeros_(self.change_projection.bias)

    def forward(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s1_valid = (inputs["s1_valid"] > 0.5).to(inputs["s1_change"].dtype)
        s2_valid = (inputs["s2_valid"] > 0.5).to(inputs["s2_change"].dtype)
        dem_valid = (inputs["dem_valid"] > 0.5).to(inputs["terrain"].dtype)
        changes = torch.cat(
            (
                inputs["s1_change"] * s1_valid,
                inputs["s2_change"] * s2_valid,
            ),
            dim=1,
        )
        sensor_valid = torch.maximum(s1_valid, s2_valid)
        change_evidence = torch.sigmoid(self.change_projection(changes)) * sensor_valid
        normalized_slope = inputs["terrain"][:, 1:2] * dem_valid
        return torch.cat((change_evidence, normalized_slope), dim=1), change_evidence


def physical_terrain_features(
    inputs: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Create evaluator-required terrain diagnostics without learned main-model features."""

    raw = inputs["terrain_raw"]
    valid = (inputs["dem_valid"] > 0.5).to(raw.dtype)
    elevation = raw[:, 0:1]
    z_hyd = masked_average(elevation, valid, kernel_size=9)
    local_second = masked_average(elevation.square(), valid, kernel_size=9)
    local_relief = (local_second - z_hyd.square()).clamp_min(0.0).sqrt()
    return {
        "z_hyd": z_hyd,
        "local_relief": local_relief,
        "slope": raw[:, 1:2],
        "dem_valid": valid,
    }


class DLSIMAdapted(nn.Module):
    """Project adapter around a clean-room DLSIM LinkNet or Attention U-Net core."""

    def __init__(self, model_config: Mapping[str, Any], architecture: str) -> None:
        super().__init__()
        channels = [int(value) for value in model_config["channels"]]
        dropout = float(model_config["dropout"])
        groups = int(model_config["group_norm_groups"])
        self.input_adapter = DLSIMInputAdapter()
        if architecture == "linknet":
            self.core: nn.Module = LinkNetCore(2, channels, dropout, groups)
        elif architecture == "attention_unet":
            self.core = AttentionUNetCore(2, channels, dropout, groups)
        else:
            raise ValueError(f"Unknown DLSIM comparison architecture: {architecture!r}")
        self.architecture = architecture
        self.heads = FloodDepthHeads(
            channels[0],
            float(model_config["uncertainty_epsilon"]),
            float(model_config["uncertainty_maximum"]),
            str(model_config.get("depth_output_semantics", "probability_weighted_v1")),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        forbidden = FORBIDDEN_INPUTS.intersection(inputs)
        if forbidden:
            raise ValueError(
                "Label-derived/provenance fields were passed to comparison model: "
                f"{sorted(forbidden)}"
            )
        missing = REQUIRED_INPUTS.difference(inputs)
        if missing:
            raise KeyError(f"Missing DLSIM comparison inputs: {sorted(missing)}")
        model_input, change_evidence = self.input_adapter(inputs)
        decoded = self.core(model_input)
        outputs: dict[str, Any] = self.heads(decoded)
        outputs["physical_features"] = physical_terrain_features(inputs)
        outputs["comparison_diagnostics"] = {
            "change_evidence": change_evidence,
        }
        return outputs


def _build(config: Mapping[str, Any], architecture: str) -> DLSIMAdapted:
    model_config = config["model"] if "model" in config else config
    return DLSIMAdapted(model_config, architecture)


def build_dlsim_linknet(config: Mapping[str, Any]) -> DLSIMAdapted:
    return _build(config, "linknet")


def build_dlsim_attention_unet(config: Mapping[str, Any]) -> DLSIMAdapted:
    return _build(config, "attention_unet")

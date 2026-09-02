"""Continuous support/depth/uncertainty output heads."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GlobalEventDepthScale(nn.Module):
    """Infer one bounded depth multiplier from label-free event context.

    Local convolutional predictions can regress every pixel of a severe event toward
    the training-set mean.  This branch pools the valid bottleneck over the whole
    patch, then predicts a single positive multiplier shared by that sample.  The
    final layer is zero-initialized, so enabling the branch starts exactly at the
    identity transformation rather than perturbing depth outputs at initialization.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        maximum_absolute_log_scale: float = 0.6931471805599453,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if maximum_absolute_log_scale <= 0.0:
            raise ValueError("maximum_absolute_log_scale must be positive")
        self.maximum_absolute_log_scale = float(maximum_absolute_log_scale)
        self.predictor = nn.Sequential(
            nn.LayerNorm(2 * channels),
            nn.Linear(2 * channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        final = self.predictor[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self, features: torch.Tensor, output_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or output_valid.ndim != 4:
            raise ValueError("features and output_valid must be BCHW tensors")
        if features.shape[0] != output_valid.shape[0] or output_valid.shape[1] != 1:
            raise ValueError("output_valid must have shape (B, 1, H, W)")
        valid = (
            F.adaptive_avg_pool2d(output_valid.to(features.dtype), features.shape[-2:])
            > 0.5
        ).to(features.dtype)
        count = valid.sum(dim=(-2, -1), keepdim=True)
        mean = (features * valid).sum(dim=(-2, -1), keepdim=True) / count.clamp_min(1.0)
        variance = (
            (features - mean).square() * valid
        ).sum(dim=(-2, -1), keepdim=True) / count.clamp_min(1.0)
        pooled = torch.cat(
            (mean.flatten(1), torch.sqrt(variance + 1e-6).flatten(1)), dim=1
        )
        raw_log_scale = self.predictor(pooled).view(-1, 1, 1, 1)
        has_valid = (count > 0).to(features.dtype)
        log_scale = (
            self.maximum_absolute_log_scale * torch.tanh(raw_log_scale) * has_valid
        )
        return torch.exp(log_scale), log_scale


class FloodDepthHeads(nn.Module):
    def __init__(
        self,
        channels: int,
        uncertainty_epsilon: float = 1e-3,
        uncertainty_maximum: float = 20.0,
        depth_output_semantics: str = "probability_weighted_v1",
    ) -> None:
        super().__init__()
        self.output = nn.Conv2d(channels, 3, 1)
        self.uncertainty_epsilon = uncertainty_epsilon
        self.uncertainty_maximum = uncertainty_maximum
        self.set_depth_output_semantics(depth_output_semantics)

    def set_depth_output_semantics(self, value: str) -> None:
        allowed = {"probability_weighted_v1", "conditional_positive_v2"}
        if value not in allowed:
            raise ValueError(
                f"Unknown depth output semantics {value!r}; expected one of {sorted(allowed)}"
            )
        self.depth_output_semantics = value

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        support_logits, raw_depth, raw_scale = self.output(features).chunk(3, dim=1)
        support_probability = torch.sigmoid(support_logits)
        conditional_depth = F.softplus(raw_depth)
        scale = F.softplus(raw_scale).clamp(max=self.uncertainty_maximum)
        scale = scale + self.uncertainty_epsilon
        expected_depth = support_probability * conditional_depth
        depth = (
            conditional_depth
            if self.depth_output_semantics == "conditional_positive_v2"
            else expected_depth
        )
        return {
            "depth": depth,
            "support_logits": support_logits,
            "support_probability": support_probability,
            "conditional_depth": conditional_depth,
            # Backward-compatible alias used by v1 checkpoints and external callers.
            "positive_depth": conditional_depth,
            "expected_depth": expected_depth,
            "uncertainty_scale": scale,
        }

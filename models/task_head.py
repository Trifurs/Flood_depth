"""Small task head for the production flood-depth model."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.encoders import group_count


class TaskHead(nn.Module):
    """A mask-agnostic convolutional prediction head for dense outputs."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels, groups), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class ContextualDepthRangeCalibration(nn.Module):
    """Small, identity-initialized calibration internal to the depth head.

    Robust pixel losses can compress the upper tail even when decoded spatial
    structure remains informative. This branch predicts bounded, spatially
    varying logit scale and bias residuals from decoded features, the baseline
    depth logit, and mask-agnostic patch context. A fixed smooth tail support
    protects the already accurate shallow regime. It never consumes a flood or
    label-validity mask.
    """

    def __init__(
        self,
        channels: int,
        groups: int,
        *,
        width: int = 16,
        maximum_scale_residual: float = 1.0,
        maximum_bias_residual: float = 2.0,
        tail_threshold_m: float = 0.46,
        tail_temperature_m: float = 0.10,
        depth_epsilon: float = 0.001,
        strength: float = 1.0,
    ) -> None:
        super().__init__()
        if min(channels, width) <= 0:
            raise ValueError("depth-range calibration channels must be positive")
        if maximum_scale_residual <= 0.0 or maximum_bias_residual <= 0.0:
            raise ValueError("depth-range calibration bounds must be positive")
        if tail_threshold_m < 0.0 or tail_temperature_m <= 0.0:
            raise ValueError("depth-range calibration tail parameters are invalid")
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("depth-range calibration strength must be finite and nonnegative")
        self.local = nn.Sequential(
            nn.Conv2d(channels + 1, width, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(width, groups), width),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, bias=False),
            nn.GroupNorm(group_count(width, groups), width),
            nn.SiLU(inplace=True),
        )
        self.projection = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.maximum_scale_residual = float(maximum_scale_residual)
        self.maximum_bias_residual = float(maximum_bias_residual)
        self.tail_threshold_m = float(tail_threshold_m)
        self.tail_temperature_m = float(tail_temperature_m)
        self.depth_epsilon = float(depth_epsilon)
        # This is deliberately a non-persistent scalar: sweeping it changes no
        # learned tensor and therefore keeps checkpoints fully compatible.  A
        # value of zero recovers the pre-calibration depth head exactly.
        self.strength = float(strength)

    def forward(
        self, features: torch.Tensor, depth_logit: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if depth_logit.shape != features[:, :1].shape:
            raise ValueError("depth calibration logit must match the feature grid")
        local = self.local(
            torch.cat((features, torch.tanh(depth_logit / 4.0)), dim=1)
        )
        context = self.context(
            torch.cat(
                (
                    F.adaptive_avg_pool2d(local, 1),
                    F.adaptive_max_pool2d(local, 1),
                ),
                dim=1,
            )
        )
        raw_scale, raw_bias = self.projection(local + context).chunk(2, dim=1)
        scale = self.maximum_scale_residual * torch.tanh(
            raw_scale / self.maximum_scale_residual
        )
        bias = self.maximum_bias_residual * torch.tanh(
            raw_bias / self.maximum_bias_residual
        )
        baseline_depth = F.softplus(depth_logit) + self.depth_epsilon
        tail_support = torch.sigmoid(
            (baseline_depth - self.tail_threshold_m) / self.tail_temperature_m
        )
        effective_scale = self.strength * tail_support * scale
        effective_bias = self.strength * tail_support * bias
        calibrated = depth_logit * (1.0 + effective_scale) + effective_bias
        return calibrated, effective_scale, effective_bias

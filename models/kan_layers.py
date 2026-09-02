"""Internal fixed-grid cubic B-spline KAN layer."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """Learn additive univariate B-spline functions plus a SiLU base path.

    Inputs have shape ``[..., in_features]``. A fixed open-uniform knot grid is used;
    there is intentionally no dynamic knot rearrangement during training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or grid_size < 2 or spline_order < 1:
            raise ValueError("Invalid KAN dimensions/grid/order")
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        internal = torch.linspace(-1.0, 1.0, grid_size + 1)[1:-1]
        knots = torch.cat(
            (
                torch.full((spline_order + 1,), -1.0),
                internal,
                torch.full((spline_order + 1,), 1.0),
            )
        )
        self.register_buffer("knots", knots)
        self.n_basis = knots.numel() - spline_order - 1
        self.normalization = nn.LayerNorm(in_features)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        self.spline_coefficients = nn.Parameter(
            torch.empty(out_features, in_features, self.n_basis)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_coefficients, mean=0.0, std=0.02)
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.base_bias, -bound, bound)

    def b_spline_basis(self, bounded_inputs: torch.Tensor) -> torch.Tensor:
        x = bounded_inputs.clamp(-1.0, 1.0 - torch.finfo(bounded_inputs.dtype).eps)
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        basis = (
            (x.unsqueeze(-1) >= knots[:-1]) & (x.unsqueeze(-1) < knots[1:])
        ).to(x.dtype)
        for degree in range(1, self.spline_order + 1):
            count = knots.numel() - degree - 1
            left_denominator = knots[degree : degree + count] - knots[:count]
            right_denominator = knots[degree + 1 : degree + 1 + count] - knots[1 : 1 + count]
            left = torch.where(
                left_denominator > 0,
                (x.unsqueeze(-1) - knots[:count]) / left_denominator.clamp_min(1e-12),
                torch.zeros_like(x.unsqueeze(-1)),
            )
            right = torch.where(
                right_denominator > 0,
                (knots[degree + 1 : degree + 1 + count] - x.unsqueeze(-1))
                / right_denominator.clamp_min(1e-12),
                torch.zeros_like(x.unsqueeze(-1)),
            )
            basis = left * basis[..., :count] + right * basis[..., 1 : count + 1]
        return basis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"KANLinear expected last dimension {self.in_features}, got {inputs.shape[-1]}"
            )
        normalized = self.normalization(inputs)
        bounded = torch.tanh(normalized)
        base = F.linear(F.silu(normalized), self.base_weight, self.base_bias)
        basis = self.b_spline_basis(bounded)
        spline = torch.einsum("...ik,oik->...o", basis, self.spline_coefficients)
        return base + spline

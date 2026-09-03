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
        normalization: str = "legacy_layernorm",
        input_bounding: str = "internal_tanh",
        base_path: str = "silu",
        base_scale_init: float = 1.0,
        spline_scale_init: float = 1.0,
        learnable_base_scale: bool = False,
        learnable_spline_scale: bool = False,
        zero_output_init: bool = False,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or grid_size < 2 or spline_order < 1:
            raise ValueError("Invalid KAN dimensions/grid/order")
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        if normalization not in {"legacy_layernorm", "explicit_fixed_scaling"}:
            raise ValueError(f"Unknown KAN normalization {normalization!r}")
        self.normalization_mode = normalization
        if input_bounding not in {"internal_tanh", "prebounded", "none"}:
            raise ValueError("input_bounding must be internal_tanh, prebounded, or none")
        if base_path not in {"silu", "linear", "none"}:
            raise ValueError("base_path must be silu, linear, or none")
        self.input_bounding = input_bounding
        self.base_path = base_path
        self.zero_output_init = bool(zero_output_init)
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
        # Keep the legacy state dictionary byte-for-byte compatible when the
        # defaults are used.  New HydroEdgeKAN instances opt into explicit
        # feature-wise scales and therefore receive these parameters.
        self._base_scale_is_default = not learnable_base_scale and float(base_scale_init) == 1.0
        self._spline_scale_is_default = not learnable_spline_scale and float(spline_scale_init) == 1.0
        if not self._base_scale_is_default:
            value = torch.full((out_features, in_features), float(base_scale_init))
            if learnable_base_scale:
                self.base_scale = nn.Parameter(value)
            else:
                self.register_buffer("base_scale", value)
        if not self._spline_scale_is_default:
            value = torch.full((out_features, in_features), float(spline_scale_init))
            if learnable_spline_scale:
                self.spline_scale = nn.Parameter(value)
            else:
                self.register_buffer("spline_scale", value)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.zero_output_init:
            nn.init.zeros_(self.base_weight)
            nn.init.zeros_(self.spline_coefficients)
            nn.init.zeros_(self.base_bias)
        else:
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
        return self.forward_with_contributions(inputs)[0]

    def forward_with_contributions(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"KANLinear expected last dimension {self.in_features}, got {inputs.shape[-1]}"
            )
        normalized = (
            self.normalization(inputs)
            if self.normalization_mode == "legacy_layernorm"
            else inputs
        )
        bounded = (
            torch.tanh(normalized)
            if self.input_bounding == "internal_tanh"
            else normalized.clamp(-1.0, 1.0)
            if self.input_bounding == "prebounded"
            else normalized
        )
        if self.base_path == "silu":
            base_input = F.silu(normalized)
        elif self.base_path == "linear":
            base_input = normalized
        else:
            base_input = torch.zeros_like(normalized)
        # The leading dimensions are arbitrary (the graph uses B,D,H,W,F).
        base_terms = torch.einsum("...i,oi->...io", base_input, self.base_weight.to(base_input.dtype))
        if hasattr(self, "base_scale"):
            base_terms = base_terms * self.base_scale.to(base_terms.dtype).transpose(0, 1)
        base = base_terms.sum(dim=-2) + self.base_bias.to(base_terms.dtype)
        # B-spline recurrence is sensitive to half-precision knot arithmetic.  It
        # remains FP32 under autocast and is converted only after contraction.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            basis = self.b_spline_basis(bounded.float())
            spline_terms = torch.einsum("...ik,oik->...io", basis, self.spline_coefficients.float())
            spline = spline_terms.sum(dim=-2)
            if hasattr(self, "spline_scale"):
                spline = (spline_terms * self.spline_scale.float().transpose(0, 1)).sum(dim=-2)
        spline = spline.to(base.dtype)
        return base + spline, base, spline

    def featurewise_contributions(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(..., in_features, out_features)`` base/spline terms."""
        if inputs.shape[-1] != self.in_features:
            raise ValueError("KANLinear feature dimension mismatch")
        normalized = self.normalization(inputs) if self.normalization_mode == "legacy_layernorm" else inputs
        bounded = (
            torch.tanh(normalized)
            if self.input_bounding == "internal_tanh"
            else normalized.clamp(-1.0, 1.0)
            if self.input_bounding == "prebounded"
            else normalized
        )
        base_input = F.silu(normalized) if self.base_path == "silu" else normalized if self.base_path == "linear" else torch.zeros_like(normalized)
        base_terms = torch.einsum("...i,oi->...io", base_input, self.base_weight.to(base_input.dtype))
        if hasattr(self, "base_scale"):
            base_terms = base_terms * self.base_scale.to(base_terms.dtype).transpose(0, 1)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            basis = self.b_spline_basis(bounded.float())
            spline_terms = torch.einsum("...ik,oik->...io", basis, self.spline_coefficients.float())
            if hasattr(self, "spline_scale"):
                spline_terms = spline_terms * self.spline_scale.float().transpose(0, 1)
        return base_terms, spline_terms.to(base_terms.dtype)

"""Weak, mask-aware physical regularizers for event-composite depth estimates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _pair_slices(
    tensor: torch.Tensor, axis: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if axis == "x":
        return tensor[..., :, :-1], tensor[..., :, 1:]
    if axis == "y":
        return tensor[..., :-1, :], tensor[..., 1:, :]
    raise ValueError(f"Unknown pair axis: {axis}")


def reference_gated_wse_gradient_loss(
    depth: torch.Tensor,
    target: torch.Tensor,
    z_hyd: torch.Tensor,
    positive_mask: torch.Tensor,
    dem_valid: torch.Tensor,
    sensor_valid: torch.Tensor,
    normalized_day_difference: torch.Tensor,
    event_ids: Sequence[str] | None,
    sigma_time: float,
    sigma_reference: float,
    sigma_terrain: float,
    beta: float,
    aggregation_mode: str = "event_macro",
) -> torch.Tensor:
    """Match local reference WSE gradients only on plausible hydraulic links.

    Both endpoints must have reliable positive depth, DEM, and sensor support. A soft
    gate favors pairs whose reconstructed reference WSE is locally coherent and whose
    smoothed DSM step is small, while asynchronous sensor pairs receive less weight.
    Unlike an unconditional zero-curvature penalty, this term preserves non-zero
    reference water-surface gradients and avoids flattening every DSM-positive region.
    Aggregation can be pair-micro for deployment-aligned training or event-macro for
    frozen legacy experiments.
    """

    for name, value in (
        ("sigma_time", sigma_time),
        ("sigma_reference", sigma_reference),
        ("sigma_terrain", sigma_terrain),
        ("beta", beta),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    valid = (
        (positive_mask > 0.5) & (dem_valid > 0.5) & (sensor_valid > 0.5)
    )
    reference_wse = z_hyd + target
    predicted_wse = z_hyd + depth
    batch_size = depth.shape[0]
    sample_numerators = depth.new_zeros(batch_size)
    sample_denominators = depth.new_zeros(batch_size)

    for axis in ("x", "y"):
        valid_a, valid_b = _pair_slices(valid, axis)
        pair_valid = valid_a & valid_b
        pred_a, pred_b = _pair_slices(predicted_wse, axis)
        ref_a, ref_b = _pair_slices(reference_wse, axis)
        terrain_a, terrain_b = _pair_slices(z_hyd, axis)
        time_a, time_b = _pair_slices(normalized_day_difference, axis)
        predicted_gradient = pred_b - pred_a
        reference_gradient = (ref_b - ref_a).detach()
        terrain_step = (terrain_b - terrain_a).detach().abs()
        pair_time = 0.5 * (time_a + time_b)
        weight = (
            pair_valid.to(depth.dtype)
            * torch.exp(-reference_gradient.abs() / sigma_reference)
            * torch.exp(-terrain_step / sigma_terrain)
            * torch.exp(-pair_time.clamp_min(0.0) / sigma_time)
        )
        penalty = F.smooth_l1_loss(
            predicted_gradient,
            reference_gradient,
            reduction="none",
            beta=beta,
        )
        sample_numerators = sample_numerators + (penalty * weight).sum(
            dim=(1, 2, 3)
        )
        sample_denominators = sample_denominators + weight.sum(dim=(1, 2, 3))

    if aggregation_mode == "pixel_micro":
        denominator = sample_denominators.sum()
        if denominator.detach().item() == 0:
            return depth.sum() * 0.0
        return sample_numerators.sum() / denominator
    if aggregation_mode not in {"auto", "event_macro"}:
        raise ValueError(
            "physical aggregation_mode must be 'pixel_micro' or 'event_macro', "
            f"received {aggregation_mode!r}"
        )
    if event_ids is None or len(event_ids) != batch_size:
        event_ids = [str(index) for index in range(batch_size)]
    event_losses: list[torch.Tensor] = []
    for event in dict.fromkeys(str(item) for item in event_ids):
        indices = [index for index, item in enumerate(event_ids) if str(item) == event]
        numerator = sample_numerators[indices].sum()
        denominator = sample_denominators[indices].sum()
        if denominator.detach().item() > 0:
            event_losses.append(numerator / denominator)
    if not event_losses:
        return depth.sum() * 0.0
    return torch.stack(event_losses).mean()


def terrain_order_violation_loss(
    depth: torch.Tensor,
    z_hyd: torch.Tensor,
    positive_mask: torch.Tensor,
    dem_valid: torch.Tensor,
    sensor_valid: torch.Tensor,
    normalized_day_difference: torch.Tensor,
    event_ids: Sequence[str] | None,
    sigma_time: float,
    minimum_terrain_step_m: float,
    maximum_terrain_step_m: float,
    beta: float,
    aggregation_mode: str = "event_macro",
) -> torch.Tensor:
    """Penalize depth changes that follow reliable local terrain changes uphill.

    The constraint acts only on non-trivial, non-barrier steps of the low-frequency
    DSM proxy. Constant depth and downhill deepening remain unpenalized, so this does
    not impose globally flat water or treat the DSM as a riverbed DTM.
    """

    if sigma_time <= 0 or beta <= 0:
        raise ValueError("sigma_time and beta must be positive")
    if minimum_terrain_step_m < 0:
        raise ValueError("minimum_terrain_step_m must be nonnegative")
    if maximum_terrain_step_m <= minimum_terrain_step_m:
        raise ValueError(
            "maximum_terrain_step_m must exceed minimum_terrain_step_m"
        )
    valid = (
        (positive_mask > 0.5) & (dem_valid > 0.5) & (sensor_valid > 0.5)
    )
    batch_size = depth.shape[0]
    sample_numerators = depth.new_zeros(batch_size)
    sample_denominators = depth.new_zeros(batch_size)
    for axis in ("x", "y"):
        valid_a, valid_b = _pair_slices(valid, axis)
        depth_a, depth_b = _pair_slices(depth, axis)
        terrain_a, terrain_b = _pair_slices(z_hyd, axis)
        time_a, time_b = _pair_slices(normalized_day_difference, axis)
        terrain_step = (terrain_b - terrain_a).detach()
        absolute_step = terrain_step.abs()
        confident_step = (
            (absolute_step >= minimum_terrain_step_m)
            & (absolute_step <= maximum_terrain_step_m)
        )
        pair_valid = valid_a & valid_b & confident_step
        signed_depth_step = terrain_step.sign() * (depth_b - depth_a)
        violation = F.relu(signed_depth_step)
        penalty = F.smooth_l1_loss(
            violation, torch.zeros_like(violation), reduction="none", beta=beta
        )
        pair_time = 0.5 * (time_a + time_b)
        weight = pair_valid.to(depth.dtype) * torch.exp(
            -pair_time.clamp_min(0.0) / sigma_time
        )
        sample_numerators = sample_numerators + (penalty * weight).sum(
            dim=(1, 2, 3)
        )
        sample_denominators = sample_denominators + weight.sum(dim=(1, 2, 3))

    if aggregation_mode == "pixel_micro":
        denominator = sample_denominators.sum()
        if denominator.detach().item() == 0:
            return depth.sum() * 0.0
        return sample_numerators.sum() / denominator
    if aggregation_mode not in {"auto", "event_macro"}:
        raise ValueError(
            "physical aggregation_mode must be 'pixel_micro' or 'event_macro', "
            f"received {aggregation_mode!r}"
        )
    if event_ids is None or len(event_ids) != batch_size:
        event_ids = [str(index) for index in range(batch_size)]
    event_losses: list[torch.Tensor] = []
    for event in dict.fromkeys(str(item) for item in event_ids):
        indices = [index for index, item in enumerate(event_ids) if str(item) == event]
        numerator = sample_numerators[indices].sum()
        denominator = sample_denominators[indices].sum()
        if denominator.detach().item() > 0:
            event_losses.append(numerator / denominator)
    if not event_losses:
        return depth.sum() * 0.0
    return torch.stack(event_losses).mean()


def weak_wse_laplacian_loss(
    depth: torch.Tensor,
    z_hyd: torch.Tensor,
    positive_mask: torch.Tensor,
    dem_valid: torch.Tensor,
    sensor_valid: torch.Tensor,
    normalized_day_difference: torch.Tensor,
    sigma_time: float,
) -> torch.Tensor:
    """Penalize local second differences only inside reliable positive regions.

    This is deliberately not a global flat-water constraint, mass conservation term,
    or shallow-water-equation residual. DSM-derived ``z_hyd`` is only a low-frequency
    topographic proxy.
    """

    if sigma_time <= 0:
        raise ValueError("sigma_time must be positive")
    dtype, device = depth.dtype, depth.device
    cross = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    laplacian = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    valid = (
        (positive_mask > 0.5) & (dem_valid > 0.5) & (sensor_valid > 0.5)
    ).to(dtype)
    neighbourhood_count = F.conv2d(valid, cross, padding=1)
    interior = neighbourhood_count >= 4.999
    eta = z_hyd + depth
    curvature = F.conv2d(eta, laplacian, padding=1).abs()
    time_weight = torch.exp(-normalized_day_difference.clamp_min(0.0) / sigma_time)
    weight = interior.to(dtype) * time_weight
    denominator = weight.sum()
    if denominator.item() == 0:
        return depth.sum() * 0.0
    return (curvature * weight).sum() / denominator

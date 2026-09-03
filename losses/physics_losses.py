"""Weak, mask-aware physical regularizers for event-composite depth estimates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from models.terrain_graph_kan import DIRECTIONS, _roll_with_boundary_mask


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
    local_relief: torch.Tensor | None = None,
    high_relief_threshold_m: float = 12.0,
    high_relief_decay_m: float = 8.0,
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
    if high_relief_threshold_m < 0 or high_relief_decay_m <= 0:
        raise ValueError("invalid high-relief downweighting parameters")
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
        relief_weight = 1.0
        if local_relief is not None:
            relief_a, relief_b = _pair_slices(local_relief, axis)
            pair_relief = 0.5 * (relief_a + relief_b).detach().clamp_min(0.0)
            relief_weight = 1.0 / (
                1.0
                + F.relu(pair_relief - high_relief_threshold_m)
                / high_relief_decay_m
            )
        terrain_step = (terrain_b - terrain_a).detach()
        absolute_step = terrain_step.abs()
        confident_step = (
            (absolute_step >= minimum_terrain_step_m)
            & (absolute_step <= maximum_terrain_step_m)
        )
        pair_valid = valid_a & valid_b & confident_step
        signed_depth_step = terrain_step.sign() * (depth_b - depth_a)
        # A small depth tolerance prevents the weak ordering prior from
        # reacting to sub-pixel label noise or quantisation.  ``beta`` is the
        # configured tolerance (and also the SmoothL1 transition scale).
        violation = F.relu(signed_depth_step - beta)
        penalty = F.smooth_l1_loss(
            violation, torch.zeros_like(violation), reduction="none", beta=beta
        )
        pair_time = 0.5 * (time_a + time_b)
        weight = pair_valid.to(depth.dtype) * torch.exp(
            -pair_time.clamp_min(0.0) / sigma_time
        ) * relief_weight
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


def _weighted_robust_mean(values: torch.Tensor, weights: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("robust beta must be positive")
    numerator = F.smooth_l1_loss(values, torch.zeros_like(values), reduction="none", beta=beta) * weights
    denominator = weights.sum()
    return numerator.sum() / denominator.clamp_min(1e-6) if denominator.detach().item() > 0 else values.sum() * 0.0


def gated_terrain_order_loss(
    depth: torch.Tensor,
    physics_elevation: torch.Tensor,
    positive_mask: torch.Tensor,
    dem_valid: torch.Tensor,
    sensor_valid: torch.Tensor,
    *,
    depth_order_tolerance_m: float = 0.02,
    huber_beta_m: float = 0.05,
    terrain_step_min_m: float = 0.02,
    terrain_step_max_m: float = 0.75,
    terrain_sigma_m: float = 0.75,
    relief: torch.Tensor | None = None,
    relief_sigma_m: float = 12.0,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply a detached fixed topographic gate to tolerant terrain ordering.

    This is a weak structural prior on reliable positive neighbours. It is not a
    shallow-water equation residual and never uses a learned KAN gate to switch the
    constraint off.
    """

    if depth_order_tolerance_m < 0 or terrain_step_min_m < 0 or terrain_step_max_m <= terrain_step_min_m:
        raise ValueError("invalid terrain-order step/tolerance bounds")
    if terrain_sigma_m <= 0 or huber_beta_m <= 0 or relief_sigma_m <= 0:
        raise ValueError("terrain-order scales must be positive")
    valid = (positive_mask > 0.5) & (dem_valid > 0.5) & (sensor_valid > 0.5)
    numerators, denominators, violations, violating_weights = [], [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        z_b, boundary = _roll_with_boundary_mask(physics_elevation, dy, dx)
        valid_b, _ = _roll_with_boundary_mask(valid.to(depth.dtype), dy, dx)
        depth_b, _ = _roll_with_boundary_mask(depth, dy, dx)
        pair_valid = valid.to(depth.dtype) * valid_b * boundary
        step = (z_b - physics_elevation).detach()
        abs_step = step.abs()
        step_gate = ((abs_step >= terrain_step_min_m) & (abs_step <= terrain_step_max_m)).to(depth.dtype)
        fixed_prior = torch.exp(-abs_step / terrain_sigma_m)
        if relief is not None:
            relief_b, _ = _roll_with_boundary_mask(relief, dy, dx)
            pair_relief = 0.5 * (relief + relief_b).detach().clamp_min(0.0)
            fixed_prior = fixed_prior * torch.exp(-pair_relief / relief_sigma_m)
        weight = (pair_valid * step_gate * fixed_prior).detach()
        violation = F.relu(step.sign() * (depth_b - depth) - depth_order_tolerance_m)
        numerators.append(F.smooth_l1_loss(violation, torch.zeros_like(violation), reduction="none", beta=huber_beta_m) * weight)
        denominators.append(weight)
        violations.append((violation.detach() * weight, weight))
        violating_weights.append((violation > 0).to(depth.dtype) * weight)
    numerator = torch.stack([value.sum() for value in numerators]).sum()
    denominator = torch.stack([value.sum() for value in denominators]).sum()
    loss = numerator / denominator.clamp_min(1e-6) if denominator.detach().item() > 0 else depth.sum() * 0.0
    if not return_diagnostics:
        return loss
    weighted_violation = torch.stack([value.sum() for value, _ in violations]).sum()
    violation_count = torch.stack([value.sum() for value in violating_weights]).sum()
    active = denominator.clamp_min(1e-6)
    return loss, {
        "violation_fraction": violation_count / active,
        "violation_magnitude": weighted_violation / active,
        "gate_weight_sum": denominator.detach(),
    }


def tolerant_wse_slope_loss(
    depth: torch.Tensor,
    physics_elevation: torch.Tensor,
    positive_mask: torch.Tensor,
    dem_valid: torch.Tensor,
    sensor_valid: torch.Tensor,
    *,
    pixel_size_m: float = 20.0,
    wse_slope_tolerance: float = 0.02,
    huber_beta: float = 0.01,
    relief: torch.Tensor | None = None,
    relief_threshold_m: float = 12.0,
    affinity_threshold: float = 0.5,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize only WSE slopes above a finite tolerance on reliable low-relief pairs."""

    if pixel_size_m <= 0 or wse_slope_tolerance < 0 or huber_beta <= 0:
        raise ValueError("invalid WSE-slope parameters")
    valid = (positive_mask > 0.5) & (dem_valid > 0.5) & (sensor_valid > 0.5)
    eta = physics_elevation + depth
    losses, weights, excesses, violating_weights = [], [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        eta_b, boundary = _roll_with_boundary_mask(eta, dy, dx)
        z_b, _ = _roll_with_boundary_mask(physics_elevation, dy, dx)
        valid_b, _ = _roll_with_boundary_mask(valid.to(depth.dtype), dy, dx)
        pair_valid = valid.to(depth.dtype) * valid_b * boundary
        distance = pixel_size_m * ((dx * dx + dy * dy) ** 0.5)
        slope = (eta_b - eta).abs() / distance
        excess = F.relu(slope - wse_slope_tolerance)
        fixed_prior = torch.exp(-(z_b - physics_elevation).detach().abs() / max(pixel_size_m * 0.1, 1e-6))
        if relief is not None:
            relief_b, _ = _roll_with_boundary_mask(relief, dy, dx)
            pair_relief = 0.5 * (relief + relief_b).detach().clamp_min(0.0)
            fixed_prior = fixed_prior * (pair_relief <= relief_threshold_m).to(depth.dtype)
        weight = (pair_valid * fixed_prior).detach()
        losses.append(F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none", beta=huber_beta) * weight)
        weights.append(weight)
        excesses.append(excess.detach() * weight)
        violating_weights.append((excess > 0).to(depth.dtype) * weight)
    numerator = torch.stack([value.sum() for value in losses]).sum()
    denominator = torch.stack([value.sum() for value in weights]).sum()
    loss = numerator / denominator.clamp_min(1e-6) if denominator.detach().item() > 0 else depth.sum() * 0.0
    if not return_diagnostics:
        return loss
    excess_total = torch.stack([value.sum() for value in excesses]).sum()
    violating_total = torch.stack([value.sum() for value in violating_weights]).sum()
    active = denominator.clamp_min(1e-6)
    return loss, {
        "violation_fraction": violating_total / active,
        "violation_magnitude": excess_total / active,
        "gate_weight_sum": denominator.detach(),
    }

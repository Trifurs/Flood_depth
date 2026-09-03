"""Terrain proxies for Hydro-v14.

The module intentionally exposes a DSM-derived *ground-like terrain proxy* rather
than claiming that DSM is a bare-earth DTM.  All operations are mask-aware and keep
metric units explicit.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.efficient_blocks import residual_block
from models.encoders import ConvNormAct


def _masked_mean(values: torch.Tensor, valid: torch.Tensor, kernel_size: int) -> torch.Tensor:
    padding = kernel_size // 2
    numerator = F.avg_pool2d(values * valid, kernel_size, stride=1, padding=padding)
    denominator = F.avg_pool2d(valid, kernel_size, stride=1, padding=padding)
    return numerator / denominator.clamp_min(1e-6)


def _masked_min(values: torch.Tensor, valid: torch.Tensor, kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a finite local minimum and whether a valid value was present."""

    padding = kernel_size // 2
    invalid_fill = torch.finfo(values.dtype).max
    filled = torch.where(valid > 0.5, values, torch.full_like(values, invalid_fill))
    minimum = -F.max_pool2d(-filled, kernel_size, stride=1, padding=padding)
    count = F.avg_pool2d(valid, kernel_size, stride=1, padding=padding)
    has_value = count > 0
    return torch.where(has_value, minimum, torch.zeros_like(minimum)), has_value.to(values.dtype)


def _masked_max(values: torch.Tensor, valid: torch.Tensor, kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    invalid_fill = torch.finfo(values.dtype).min
    filled = torch.where(valid > 0.5, values, torch.full_like(values, invalid_fill))
    maximum = F.max_pool2d(filled, kernel_size, stride=1, padding=kernel_size // 2)
    count = F.avg_pool2d(valid, kernel_size, stride=1, padding=kernel_size // 2)
    has_value = count > 0
    return torch.where(has_value, maximum, torch.zeros_like(maximum)), has_value.to(values.dtype)


def ground_like_proxy(
    elevation: torch.Tensor,
    valid: torch.Tensor,
    kernel_size: int = 9,
) -> torch.Tensor:
    """Estimate a conservative DSM low envelope with a mask-aware opening."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("ground_proxy_kernel_size must be a positive odd integer")
    eroded, eroded_valid = _masked_min(elevation, valid, kernel_size)
    opened, opened_valid = _masked_max(eroded, eroded_valid, kernel_size)
    local_fallback = _masked_mean(elevation, valid, kernel_size)
    return torch.where(opened_valid > 0.5, opened, local_fallback)


def valid_central_gradients_v14(
    elevation: torch.Tensor,
    valid: torch.Tensor,
    pixel_size_x: float,
    pixel_size_y: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if pixel_size_x <= 0 or pixel_size_y <= 0:
        raise ValueError("pixel sizes must be positive")
    left, right = elevation[..., :, :-2], elevation[..., :, 2:]
    gx_valid = valid[..., :, :-2] * valid[..., :, 1:-1] * valid[..., :, 2:]
    gx_core = (right - left) / (2.0 * pixel_size_x)
    gx = F.pad(torch.where(gx_valid > 0.5, gx_core, torch.zeros_like(gx_core)), (1, 1, 0, 0))
    gx_mask = F.pad(gx_valid, (1, 1, 0, 0))
    top, bottom = elevation[..., :-2, :], elevation[..., 2:, :]
    gy_valid = valid[..., :-2, :] * valid[..., 1:-1, :] * valid[..., 2:, :]
    gy_core = (bottom - top) / (2.0 * pixel_size_y)
    gy = F.pad(torch.where(gy_valid > 0.5, gy_core, torch.zeros_like(gy_core)), (0, 0, 1, 1))
    gy_mask = F.pad(gy_valid, (0, 0, 1, 1))
    return gx, gy, gx_mask, gy_mask


def path_barrier_proxy(
    elevation: torch.Tensor,
    valid: torch.Tensor,
    pixel_step: int,
    ground: torch.Tensor | None = None,
    directions: Sequence[tuple[int, int]] | None = None,
    statistic: str = "max",
    quantile: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute high-resolution DSM crest barriers for vectorized graph edges.

    Returns ``(barrier, edge_valid)`` shaped ``(B, D, 1, H, W)``.  The path includes
    both endpoints and intermediate original pixels, so diagonal paths and metric
    scale are represented without per-pixel Python loops.
    """

    if pixel_step < 1:
        raise ValueError("pixel_step must be >= 1")
    if statistic not in {"max", "quantile"}:
        raise ValueError("statistic must be max or quantile")
    if statistic == "quantile" and not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    dirs = tuple(directions or ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)))
    valid = (valid > 0.5).to(elevation.dtype)
    ground = elevation if ground is None else ground
    outputs, valids = [], []
    fractions = torch.linspace(0.0, 1.0, pixel_step + 1, device=elevation.device, dtype=elevation.dtype)
    for dy, dx in dirs:
        samples, sample_valid = [], []
        for fraction in fractions:
            offset_y = int(round(float(dy) * float(pixel_step) * float(fraction)))
            offset_x = int(round(float(dx) * float(pixel_step) * float(fraction)))
            shifted = torch.roll(elevation, shifts=(offset_y, offset_x), dims=(-2, -1))
            boundary = torch.ones_like(valid)
            if offset_y > 0: boundary[..., :offset_y, :] = 0
            elif offset_y < 0: boundary[..., offset_y:, :] = 0
            if offset_x > 0: boundary[..., :, :offset_x] = 0
            elif offset_x < 0: boundary[..., :, offset_x:] = 0
            samples.append(shifted)
            # ``torch.roll`` samples the source at ``(p - offset)``.  The
            # validity mask must follow the same sampling path; checking the
            # unshifted mask would validate only the starting pixel repeatedly.
            sample_valid.append(torch.roll(valid, shifts=(offset_y, offset_x), dims=(-2, -1)) * boundary)
        values = torch.stack(samples, dim=1)
        path_valid = torch.stack(sample_valid, dim=1).prod(dim=1)
        values = torch.where(torch.stack(sample_valid, dim=1) > 0.5, values, torch.full_like(values, torch.finfo(values.dtype).min))
        crest = values.max(dim=1).values if statistic == "max" else torch.quantile(values.masked_fill(path_valid.unsqueeze(1) < 0.5, 0.0), quantile, dim=1)
        neighbour_ground = torch.roll(ground, shifts=(dy * pixel_step, dx * pixel_step), dims=(-2, -1))
        barrier = F.relu(crest - torch.maximum(ground, neighbour_ground))
        endpoint_valid = valid * path_valid * torch.roll(valid, shifts=(dy * pixel_step, dx * pixel_step), dims=(-2, -1))
        outputs.append(barrier * endpoint_valid)
        valids.append(endpoint_valid)
    return torch.stack(outputs, dim=1), torch.stack(valids, dim=1)


class TerrainFeaturePyramidV14(nn.Module):
    """Resolution-aware terrain pyramid with name-resolved DSM/slope channels."""

    def __init__(
        self, input_channels: int, channels: list[int], dropout: float, groups: int,
        pixel_size_m: float, terrain_band_names: Sequence[str],
        ground_proxy_kernel_size: int = 9, physics_elevation: str = "z_ground_proxy",
        block_kind: str = "efficient",
    ) -> None:
        super().__init__()
        if pixel_size_m <= 0:
            raise ValueError("terrain_pixel_size_m must be positive")
        names = tuple(str(name) for name in terrain_band_names)
        if len(names) != len(set(names)):
            raise ValueError("terrain band names must be unique")
        if "elevation_m_DSM" not in names:
            raise ValueError("Hydro-v14 terrain requires elevation_m_DSM")
        if physics_elevation not in {"z_local_mean", "z_ground_proxy"}:
            raise ValueError("physics_elevation must be z_local_mean or z_ground_proxy")
        self.pixel_size_m = float(pixel_size_m)
        self.terrain_band_names = names
        self.elevation_index = names.index("elevation_m_DSM")
        self.slope_index = names.index("slope_deg") if "slope_deg" in names else None
        self.ground_proxy_kernel_size = int(ground_proxy_kernel_size)
        self.physics_elevation = physics_elevation
        self.stem = nn.Sequential(
            ConvNormAct(input_channels + 8, channels[0], 3, groups=groups),
            residual_block(block_kind, channels[0], dropout, groups),
        )
        self.down = nn.ModuleList([
            nn.Sequential(ConvNormAct(channels[i - 1], channels[i], 3, 2, groups), residual_block(block_kind, channels[i], dropout, groups))
            for i in range(1, len(channels))
        ])

    def forward(self, normalized: torch.Tensor, raw: torch.Tensor, dem_valid: torch.Tensor):
        valid = (dem_valid > 0.5).to(raw.dtype)
        elevation = raw[:, self.elevation_index:self.elevation_index + 1]
        if self.slope_index is not None:
            slope = raw[:, self.slope_index:self.slope_index + 1]
        else:
            slope = None
        z_local_mean = _masked_mean(elevation, valid, 9)
        z_ground_proxy = ground_like_proxy(elevation, valid, self.ground_proxy_kernel_size)
        physics_elevation = z_ground_proxy if self.physics_elevation == "z_ground_proxy" else z_local_mean
        obstacle = F.relu(elevation - z_ground_proxy) * valid
        relative = (elevation - physics_elevation) * valid
        second = _masked_mean(elevation.square(), valid, 9)
        relief = (second - z_local_mean.square()).clamp_min(0).sqrt() * valid
        gx, gy, gx_valid, gy_valid = valid_central_gradients_v14(elevation, valid, self.pixel_size_m, self.pixel_size_m)
        derived_slope = torch.sqrt(gx.square() + gy.square())
        if slope is None:
            slope_for_features = derived_slope
        else:
            slope_for_features = slope * valid
        scale = relief + 1.0
        base = torch.cat((normalized, torch.tanh(relative / scale), torch.tanh(obstacle / scale), torch.tanh(gx), torch.tanh(gy), torch.log1p(relief) / 5.0, torch.tanh(derived_slope), gx_valid * gy_valid, valid), dim=1) * valid
        features = [self.stem(base) * valid]
        fractions = [valid]
        for layer in self.down:
            features.append(layer(features[-1]))
            fraction = F.adaptive_avg_pool2d(valid, features[-1].shape[-2:])
            fractions.append(fraction)
            features[-1] = features[-1] * fraction
        physical = {
            "dsm_elevation": elevation,
            "z_local_mean": z_local_mean,
            "z_ground_proxy": z_ground_proxy,
            "z_hyd": physics_elevation,
            "physics_elevation": physics_elevation,
            "obstacle_residual": obstacle,
            "z_barrier": obstacle,
            "z_relative": relative,
            "dz_dx": gx,
            "dz_dy": gy,
            "derived_slope_m_per_m": derived_slope,
            "slope": slope_for_features,
            "local_relief": relief,
            "gradient_x_valid": gx_valid,
            "gradient_y_valid": gy_valid,
            "dem_valid": valid,
            "dem_valid_fractions": fractions,
            "terrain_valid": valid,
        }
        return features, physical

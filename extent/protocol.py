"""Published Sentinel-1 change features and frozen extent post-processing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def build_ai4g_change_features(
    batch: Mapping[str, Any],
    *,
    vv_water_threshold_db: float = -17.5,
    vh_water_threshold_db: float = -22.5,
    minimum_drop_db: float = 5.0,
    vv_valid_floor_db: float = -30.0,
    vh_valid_floor_db: float = -32.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the two public AI4G VV/VH flood-change indicator channels."""

    extent_inputs = batch.get("extent_inputs")
    if not isinstance(extent_inputs, Mapping):
        raise KeyError("Batch has no dedicated extent_inputs mapping")
    pre = extent_inputs["s1_t1_db"]
    event = extent_inputs["s1_t2_db"]
    pair_valid = extent_inputs["s1_pair_valid"] > 0.5
    if pre.ndim != 4 or event.shape != pre.shape or pre.shape[1] < 2:
        raise ValueError("Raw S1 pre/event inputs must share shape [B,C>=2,H,W]")
    semantic_valid = batch["validity"]["s1_valid"] > 0.5
    valid = pair_valid & semantic_valid
    vv_change = (
        (event[:, 0:1] < vv_water_threshold_db)
        & (pre[:, 0:1] > vv_water_threshold_db)
        & ((pre[:, 0:1] - event[:, 0:1]) > minimum_drop_db)
        & (event[:, 0:1] >= vv_valid_floor_db)
        & (pre[:, 0:1] >= vv_valid_floor_db)
        & valid
    )
    vh_change = (
        (event[:, 1:2] < vh_water_threshold_db)
        & (pre[:, 1:2] > vh_water_threshold_db)
        & ((pre[:, 1:2] - event[:, 1:2]) > minimum_drop_db)
        & (event[:, 1:2] >= vh_valid_floor_db)
        & (pre[:, 1:2] >= vh_valid_floor_db)
        & valid
    )
    return torch.cat((vv_change, vh_change), dim=1).to(pre.dtype), valid


def postprocess_extent(
    probabilities: torch.Tensor,
    output_valid: torch.Tensor,
    *,
    probability_threshold: float = 0.5,
    buffer_pixels: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold once, then apply the paper's 80 m buffer at local 20 m pixels."""

    if probabilities.ndim != 4 or probabilities.shape[1] != 1:
        raise ValueError("Extent probabilities must have shape [B,1,H,W]")
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability_threshold must be in (0,1)")
    if buffer_pixels < 0:
        raise ValueError("buffer_pixels cannot be negative")
    valid = output_valid > 0.5
    raw = (probabilities >= probability_threshold) & valid
    if buffer_pixels == 0:
        return raw, raw
    kernel = 2 * int(buffer_pixels) + 1
    buffered = F.max_pool2d(raw.to(probabilities.dtype), kernel, stride=1, padding=buffer_pixels)
    return raw, (buffered > 0.5) & valid

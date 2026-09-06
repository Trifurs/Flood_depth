"""Verified graph placement metadata shared by training and evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


S1_GRAPH_MODELS = {
    "pa_hydrokan_s1_v14",
    "pa_hydrokan_s1_v15",
    "pa_hydrokan_s1_v15_1",
}


def resolved_graph_identity(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return static graph metadata implied by a resolved S1 model config."""

    model = config.get("model", {})
    dataset = config.get("dataset", {})
    if not isinstance(model, Mapping) or str(model.get("name")) not in S1_GRAPH_MODELS:
        return None
    stride = int(model.get("graph_feature_stride", 8))
    if stride not in {4, 8}:
        raise ValueError("model.graph_feature_stride must be 4 or 8")
    pixel_size = float(model.get("terrain_pixel_size_m", 20.0))
    if pixel_size <= 0:
        raise ValueError("model.terrain_pixel_size_m must be positive")
    patch_size = int(dataset.get("patch_size", 0)) if isinstance(dataset, Mapping) else 0
    identity = {
        "graph_feature_stride": stride,
        "graph_node_spacing_m": pixel_size * stride,
        "graph_feature_shape": (
            [math.ceil(patch_size / stride), math.ceil(patch_size / stride)]
            if patch_size > 0
            else None
        ),
        "orthogonal_neighbour_distance_m": pixel_size * stride,
        "diagonal_neighbour_distance_m": pixel_size * stride * math.sqrt(2.0),
    }
    stats_sha256 = model.get("graph_edge_stats_sha256")
    if stats_sha256 is not None:
        identity["edge_stats_sha256"] = str(stats_sha256)
    return identity


def runtime_graph_identity(model: Any) -> dict[str, Any] | None:
    """Read actual forward-time graph metadata without depending on wrappers."""

    unwrapped = getattr(model, "module", model)
    getter = getattr(unwrapped, "graph_identity", None)
    return getter() if callable(getter) else None

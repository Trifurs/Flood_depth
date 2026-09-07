"""Registry for the single supported flood-depth model."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from models.flood_depth_model import build_flood_depth_model


MODEL_NAME = "flood_depth_s1"


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build the production SAR-and-terrain model."""

    name = str(config["model"]["name"])
    if name != MODEL_NAME:
        raise KeyError(f"Unsupported model {name!r}; expected {MODEL_NAME!r}")
    return build_flood_depth_model(config)

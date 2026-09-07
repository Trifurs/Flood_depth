"""Registry for the production PA-HydroKAN model."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from models.pa_hydrokan import build_pa_hydrokan


MODEL_NAME = "pa_hydrokan"


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build the production SAR-and-terrain PA-HydroKAN model."""

    name = str(config["model"]["name"])
    if name != MODEL_NAME:
        raise KeyError(f"Unsupported model {name!r}; expected {MODEL_NAME!r}")
    return build_pa_hydrokan(config)

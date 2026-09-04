"""Explicit registry for the main model and audited comparison adapters."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from models.pa_hydrokan_s1_v14 import build_pa_hydrokan_s1_v14
from models.pa_hydrokan_s1_v15 import build_pa_hydrokan_s1_v15


def _builders():
    return {
        "pa_hydrokan_s1_v14": build_pa_hydrokan_s1_v14,
        "pa_hydrokan_s1_v15": build_pa_hydrokan_s1_v15,
    }


def build_model(config: Mapping[str, Any]) -> nn.Module:
    name = str(config["model"]["name"])
    builders = _builders()
    if name not in builders:
        raise KeyError(
            f"Unknown model {name!r}; registered models are {sorted(builders)}"
        )
    return builders[name](config)

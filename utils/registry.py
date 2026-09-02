"""Explicit registry for the main model and audited comparison adapters."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from models.pa_hydrokan import build_pa_hydrokan


def _builders():
    # Lazy import keeps the main-model path independent from optional comparisons.
    from compare.dlsim_adapted import (
        build_dlsim_attention_unet,
        build_dlsim_linknet,
    )

    return {
        "pa_hydrokan": build_pa_hydrokan,
        "dlsim_linknet_adapted": build_dlsim_linknet,
        "dlsim_attention_unet_adapted": build_dlsim_attention_unet,
    }


def build_model(config: Mapping[str, Any]) -> nn.Module:
    name = str(config["model"]["name"])
    builders = _builders()
    if name not in builders:
        raise KeyError(
            f"Unknown model {name!r}; registered models are {sorted(builders)}"
        )
    return builders[name](config)

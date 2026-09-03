"""Explicit registry for the main model and audited comparison adapters."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from models.pa_hydrokan import build_pa_hydrokan
from models.pa_hydrokan_v13 import build_pa_hydrokan_v13
from models.pa_hydrokan_v13_1 import build_pa_hydrokan_v13_1
from models.pa_hydrokan_v13_2 import build_pa_hydrokan_v13_2
from models.pa_hydrokan_v14 import build_pa_hydrokan_v14
from models.pa_hydrokan_s1_v14 import build_pa_hydrokan_s1_v14
from models.pa_hydrokan_s1_v15 import build_pa_hydrokan_s1_v15


def _builders():
    # Lazy import keeps the main-model path independent from optional comparisons.
    from compare.dlsim_adapted import (
        build_dlsim_attention_unet,
        build_dlsim_linknet,
    )

    return {
        "pa_hydrokan": build_pa_hydrokan,
        "pa_hydrokan_v13": build_pa_hydrokan_v13,
        "pa_hydrokan_v13_1": build_pa_hydrokan_v13_1,
        "pa_hydrokan_v13_2": build_pa_hydrokan_v13_2,
        "pa_hydrokan_v14": build_pa_hydrokan_v14,
        "pa_hydrokan_s1_v14": build_pa_hydrokan_s1_v14,
        "pa_hydrokan_s1_v15": build_pa_hydrokan_s1_v15,
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

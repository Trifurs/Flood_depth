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
from models.pa_hydrokan_s1_v15_1 import build_pa_hydrokan_s1_v15_1
from models.pa_hydrokan_s1_v15_2 import build_pa_hydrokan_s1_v15_2
from models.pa_hydrokan_s1_v15_3 import build_pa_hydrokan_s1_v15_3


def _builders():
    return {
        "pa_hydrokan": build_pa_hydrokan,
        "pa_hydrokan_v13": build_pa_hydrokan_v13,
        "pa_hydrokan_v13_1": build_pa_hydrokan_v13_1,
        "pa_hydrokan_v13_2": build_pa_hydrokan_v13_2,
        "pa_hydrokan_v14": build_pa_hydrokan_v14,
        "pa_hydrokan_s1_v14": build_pa_hydrokan_s1_v14,
        "pa_hydrokan_s1_v15": build_pa_hydrokan_s1_v15,
        "pa_hydrokan_s1_v15_1": build_pa_hydrokan_s1_v15_1,
        "pa_hydrokan_s1_v15_2": build_pa_hydrokan_s1_v15_2,
        "pa_hydrokan_s1_v15_3": build_pa_hydrokan_s1_v15_3,
    }


def build_model(config: Mapping[str, Any]) -> nn.Module:
    name = str(config["model"]["name"])
    builders = _builders()
    if name not in builders:
        raise KeyError(
            f"Unknown model {name!r}; registered models are {sorted(builders)}"
        )
    return builders[name](config)

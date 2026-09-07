"""Named non-learned comparison models for flood-depth reconstruction."""

from collections.abc import Callable

import numpy as np

from .fldepth import estimate_depth as estimate_fldepth
from .fwdet_v2 import estimate_depth as estimate_fwdet_v2
from .tsa import estimate_depth as estimate_tsa


_ESTIMATORS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "fwdet_v2": estimate_fwdet_v2,
    "tsa": estimate_tsa,
    "fldepth": estimate_fldepth,
}


def available_methods() -> tuple[str, ...]:
    """Return stable identifiers for the named comparison models."""

    return tuple(_ESTIMATORS)


def estimator_for(method: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Resolve one named comparison model."""

    try:
        return _ESTIMATORS[method]
    except KeyError as exc:
        raise KeyError(f"Unknown comparison model {method!r}; choices={available_methods()}") from exc


__all__ = [
    "available_methods",
    "estimator_for",
    "estimate_fldepth",
    "estimate_fwdet_v2",
    "estimate_tsa",
]

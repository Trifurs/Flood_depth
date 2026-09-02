from __future__ import annotations

import pytest

from datasets.preprocessing import (
    PreprocessingError,
    resolve_depth_stratification_bins,
)


class _Normalizer:
    train_depth_bins = [0.1, 0.23, 0.48, 24.82]
    stats = {"train_depth": {"minimum": 0.1, "maximum": 24.82}}


def test_tail_strata_accept_frozen_train_extrema() -> None:
    edges = [0.1, 0.23, 0.48, 0.83, 1.22, 2.14, 24.82]
    assert resolve_depth_stratification_bins(
        {"depth_stratification_edges_m": edges}, _Normalizer()  # type: ignore[arg-type]
    ) == edges


def test_tail_strata_reject_non_train_endpoint() -> None:
    with pytest.raises(PreprocessingError, match="train maximum"):
        resolve_depth_stratification_bins(
            {"depth_stratification_edges_m": [0.1, 0.48, 2.14]},
            _Normalizer(),  # type: ignore[arg-type]
        )


def test_tail_strata_must_retain_primary_edges() -> None:
    with pytest.raises(PreprocessingError, match="original train-quantile edge"):
        resolve_depth_stratification_bins(
            {"depth_stratification_edges_m": [0.1, 0.23, 0.83, 24.82]},
            _Normalizer(),  # type: ignore[arg-type]
        )

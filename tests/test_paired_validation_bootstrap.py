"""Unit tests for paired validation bootstrap aggregation."""

from __future__ import annotations

import numpy as np

from tools.paired_validation_bootstrap import _group_rows, _resampled_deltas


def _row(
    sample_id: str,
    event_id: str,
    pixels: int,
    baseline_mae: float,
    candidate_mae: float,
    baseline_rmse: float,
    candidate_rmse: float,
) -> dict[str, float | int | str]:
    return {
        "sample_id": sample_id,
        "source_event_id": event_id,
        "baseline_pixels": pixels,
        "candidate_pixels": pixels,
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "baseline_rmse": baseline_rmse,
        "candidate_rmse": candidate_rmse,
        "baseline_p90_absolute_error": baseline_mae + 0.1,
        "candidate_p90_absolute_error": candidate_mae + 0.1,
        "baseline_bias": -baseline_mae,
        "candidate_bias": -candidate_mae,
    }


def test_event_grouping_reconstructs_rmse_from_squared_sample_values() -> None:
    units = _group_rows(
        [
            _row("a", "event-a", 1, 1.0, 0.5, 1.0, 0.5),
            _row("b", "event-a", 3, 3.0, 1.5, 3.0, 1.5),
        ],
        unit="event",
    )

    assert len(units) == 1
    expected_baseline_rmse = np.sqrt((1.0 * 1.0**2 + 3.0 * 3.0**2) / 4.0)
    expected_candidate_rmse = np.sqrt((1.0 * 0.5**2 + 3.0 * 1.5**2) / 4.0)
    np.testing.assert_allclose(units[0]["baseline_rmse"], expected_baseline_rmse)
    np.testing.assert_allclose(units[0]["candidate_rmse"], expected_candidate_rmse)


def test_paired_bootstrap_returns_finite_candidate_deltas() -> None:
    units = _group_rows(
        [
            _row("a", "event-a", 2, 0.4, 0.3, 0.5, 0.4),
            _row("b", "event-b", 2, 0.6, 0.5, 0.7, 0.6),
        ],
        unit="sample",
    )

    observed, deltas, win_rate = _resampled_deltas(units, draws=100, seed=7)

    assert observed["delta_mae"] < 0.0
    assert observed["delta_rmse"] < 0.0
    assert 0.0 <= win_rate <= 1.0
    assert all(np.isfinite(values).all() for values in deltas.values())

from __future__ import annotations

import numpy as np
import pytest

from metrics.aggregator import EvaluationAggregator


def test_event_depth_bin_macro_metric_matches_training_reduction() -> None:
    aggregator = EvaluationAggregator([0.10, 0.23, 0.48, 24.82])
    target_a = np.array([0.10, 0.23, 0.30, 0.60], dtype=np.float32)
    error_a = np.array([1.0, 1.0, 3.0, 5.0], dtype=np.float32)
    target_b = np.full(4, 0.10, dtype=np.float32)
    error_b = np.full(4, 2.0, dtype=np.float32)
    for sample, event, target, error in (
        ("sample_a", "event_a", target_a, error_a),
        ("sample_b", "event_b", target_b, error_b),
    ):
        aggregator.add(
            sample,
            event,
            target + error,
            target,
            np.ones_like(target),
            np.ones_like(target, dtype=bool),
        )
    summary, _, events, bins = aggregator.summarize()
    assert summary["event_macro_mae"] == pytest.approx(2.25)
    assert summary["event_depth_bin_macro_mae"] == pytest.approx(2.5)
    assert summary["event_depth_bin_nonempty_cells"] == 4
    assert len(events) == 2
    assert [row["pixels"] for row in bins] == [6, 1, 1]


def test_event_hierarchical_metric_matches_hierarchical_loss_reduction() -> None:
    aggregator = EvaluationAggregator(
        [0.10, 0.23, 0.48, 0.83, 1.22, 2.14, 24.82],
        [0.10, 0.23, 0.48, 24.82],
    )
    target_a = np.array([0.10, 0.30, 0.60, 0.90, 1.50, 3.00], dtype=np.float32)
    error_a = np.array([1.0, 3.0, 5.0, 9.0, 13.0, 17.0], dtype=np.float32)
    target_b = np.full(6, 0.10, dtype=np.float32)
    error_b = np.full(6, 2.0, dtype=np.float32)
    for sample, event, target, error in (
        ("sample_a", "event_a", target_a, error_a),
        ("sample_b", "event_b", target_b, error_b),
    ):
        aggregator.add(
            sample,
            event,
            target + error,
            target,
            np.ones_like(target),
            np.ones_like(target, dtype=bool),
        )
    summary, _, events, _ = aggregator.summarize()
    assert summary["event_depth_bin_macro_mae"] == pytest.approx(5.0)
    assert summary["event_depth_hierarchical_macro_mae"] == pytest.approx(3.5)
    assert summary["event_hierarchical_composite_mae"] == pytest.approx(4.25)
    assert summary["event_depth_hierarchical_nonempty_primary_groups"] == 4
    assert summary["event_depth_hierarchical_nonempty_cells"] == 7
    assert len(events) == 2

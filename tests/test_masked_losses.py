from __future__ import annotations

import torch

from losses.depth_losses import (
    depth_bin_macro_mean,
    event_depth_bin_macro_mean,
    event_depth_exceedance_loss,
    event_depth_hierarchical_bias_loss,
    event_depth_hierarchical_macro_mean,
    event_macro_masked_mean,
    masked_micro_mean,
    positive_depth_losses,
    sample_depth_bin_macro_mean,
)


def test_invalid_and_unknown_target_values_do_not_change_supervised_loss() -> None:
    torch.manual_seed(2)
    positive_depth = torch.rand(2, 1, 8, 8)
    final_depth = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8) + 0.1
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, 2:5, 2:6] = True
    first = positive_depth_losses(
        positive_depth, final_depth, target, mask, ["event_a", "event_b"], 0.5, 0.5
    )["depth"]
    altered = target.clone()
    altered[~mask] = torch.linspace(-1e6, 1e6, int((~mask).sum()))
    second = positive_depth_losses(
        positive_depth, final_depth, altered, mask, ["event_a", "event_b"], 0.5, 0.5
    )["depth"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_pixel_micro_mean_weights_pixels_not_events() -> None:
    values = torch.tensor(
        [
            [[[10.0, 99.0, 99.0]]],
            [[[0.0, 0.0, 0.0]]],
        ]
    )
    mask = torch.tensor(
        [
            [[[True, False, False]]],
            [[[True, True, True]]],
        ]
    )
    torch.testing.assert_close(masked_micro_mean(values, mask), torch.tensor(2.5))
    torch.testing.assert_close(
        event_macro_masked_mean(values, mask, ["small", "large"]),
        torch.tensor(5.0),
    )


def test_pixel_micro_l1_is_exact_mae_and_event_invariant() -> None:
    target = torch.zeros((2, 1, 1, 3))
    prediction = torch.tensor(
        [
            [[[4.0, 8.0, 8.0]]],
            [[[0.0, 0.0, 0.0]]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor(
        [
            [[[True, False, False]]],
            [[[True, True, True]]],
        ]
    )
    first = positive_depth_losses(
        prediction,
        None,
        target,
        mask,
        ["event_a", "event_b"],
        0.0,
        0.0,
        aggregation_mode="pixel_micro",
        linear_loss="l1",
    )["depth"]
    renamed = positive_depth_losses(
        prediction,
        None,
        target,
        mask,
        ["same", "same"],
        0.0,
        0.0,
        aggregation_mode="pixel_micro",
        linear_loss="l1",
    )["depth"]
    torch.testing.assert_close(first, torch.tensor(1.0))
    torch.testing.assert_close(first, renamed, rtol=0, atol=0)
    first.backward()
    assert prediction.grad is not None


def test_global_depth_bin_macro_balances_strata_without_events() -> None:
    values = torch.tensor([[[[1.0, 1.0, 1.0, 9.0]]]])
    target = torch.tensor([[[[0.10, 0.10, 0.10, 1.00]]]])
    result = depth_bin_macro_mean(
        values,
        target,
        torch.ones_like(target, dtype=torch.bool),
        [0.10, 0.50, 2.00],
    )
    # Three shallow pixels and one deep pixel get equal stratum influence.
    torch.testing.assert_close(result, torch.tensor(5.0))


def test_sample_depth_bin_macro_balances_rasters_and_strata() -> None:
    values = torch.tensor(
        [
            [[[1.0, 1.0, 9.0]]],
            [[[2.0, 2.0, 2.0]]],
        ]
    )
    target = torch.tensor(
        [
            [[[0.10, 0.20, 1.00]]],
            [[[0.10, 0.10, 0.10]]],
        ]
    )
    result = sample_depth_bin_macro_mean(
        values,
        target,
        torch.ones_like(target, dtype=torch.bool),
        [0.10, 0.50, 2.00],
    )
    # sample 0: mean(shallow=1, deep=9)=5; sample 1: shallow=2; macro=3.5.
    torch.testing.assert_close(result, torch.tensor(3.5))


def test_sample_depth_bin_supervised_loss_ignores_event_ids() -> None:
    target = torch.tensor(
        [
            [[[0.10, 0.20, 1.00]]],
            [[[0.10, 0.60, 1.50]]],
        ]
    )
    prediction = (target + torch.tensor([[[[0.1, 0.2, 0.3]]]])).requires_grad_()
    mask = torch.ones_like(target, dtype=torch.bool)
    arguments = (prediction, None, target, mask)
    separate = positive_depth_losses(
        *arguments,
        ["event_a", "event_b"],
        0.5,
        0.0,
        train_depth_bins=[0.10, 0.50, 2.00],
        aggregation_mode="sample_depth_bin",
    )
    merged = positive_depth_losses(
        *arguments,
        ["same", "same"],
        0.5,
        0.0,
        train_depth_bins=[0.10, 0.50, 2.00],
        aggregation_mode="sample_depth_bin",
    )
    for key in ("depth", "depth_linear", "depth_log", "depth_bias"):
        torch.testing.assert_close(separate[key], merged[key], rtol=0, atol=0)
    separate["depth"].backward()
    assert prediction.grad is not None


def test_depth_bin_macro_supervised_loss_is_event_invariant() -> None:
    target = torch.tensor(
        [
            [[[0.10, 0.20, 1.00]]],
            [[[0.10, 0.60, 1.50]]],
        ]
    )
    prediction = (target + 0.25).requires_grad_()
    mask = torch.ones_like(target, dtype=torch.bool)
    arguments = (prediction, None, target, mask)
    first = positive_depth_losses(
        *arguments,
        ["event_a", "event_b"],
        0.5,
        0.0,
        train_depth_bins=[0.10, 0.50, 2.00],
        aggregation_mode="depth_bin_macro",
    )["depth"]
    renamed = positive_depth_losses(
        *arguments,
        ["same", "same"],
        0.5,
        0.0,
        train_depth_bins=[0.10, 0.50, 2.00],
        aggregation_mode="depth_bin_macro",
    )["depth"]
    torch.testing.assert_close(first, renamed, rtol=0, atol=0)
    first.backward()
    assert prediction.grad is not None


def test_event_depth_bin_macro_preserves_event_and_depth_stratum_balance() -> None:
    values = torch.tensor(
        [
            [[[1.0, 1.0, 3.0, 5.0]]],
            [[[2.0, 2.0, 2.0, 2.0]]],
        ]
    )
    target = torch.tensor(
        [
            [[[0.10, 0.23, 0.30, 0.60]]],
            [[[0.10, 0.10, 0.10, 0.10]]],
        ]
    )
    result = event_depth_bin_macro_mean(
        values,
        target,
        torch.ones_like(target, dtype=torch.bool),
        ["event_a", "event_b"],
        [0.10, 0.23, 0.48, 24.82],
    )
    # event_a: mean(mean(1,1), 3, 5) = 3; event_b: 2; macro = 2.5.
    torch.testing.assert_close(result, torch.tensor(2.5))


def test_hierarchical_macro_refines_deep_tail_without_overweighting_it() -> None:
    values = torch.tensor(
        [
            [[[1.0, 3.0, 5.0, 9.0, 13.0, 17.0]]],
            [[[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]],
        ]
    )
    target = torch.tensor(
        [
            [[[0.10, 0.30, 0.60, 0.90, 1.50, 3.00]]],
            [[[0.10, 0.10, 0.10, 0.10, 0.10, 0.10]]],
        ]
    )
    result = event_depth_hierarchical_macro_mean(
        values,
        target,
        torch.ones_like(target, dtype=torch.bool),
        ["event_a", "event_b"],
        [0.10, 0.23, 0.48, 24.82],
        [0.10, 0.23, 0.48, 0.83, 1.22, 2.14, 24.82],
    )
    # event_a: mean(shallow=1, mid=3, deep=mean(5,9,13,17)=11) = 5;
    # event_b has only shallow=2; event macro = 3.5.
    torch.testing.assert_close(result, torch.tensor(3.5))


def test_hierarchical_bias_penalizes_signed_cell_means_before_averaging() -> None:
    target = torch.tensor(
        [
            [[[0.10, 0.30, 0.60, 0.90, 1.50, 3.00]]],
            [[[0.10, 0.10, 0.10, 0.10, 0.10, 0.10]]],
        ]
    )
    error = torch.tensor(
        [
            [[[1.0, -1.0, 2.0, 2.0, -3.0, -3.0]]],
            [[[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]],
        ]
    )
    prediction = (target + error).requires_grad_()
    result = event_depth_hierarchical_bias_loss(
        prediction,
        target,
        torch.ones_like(target, dtype=torch.bool),
        ["event_a", "event_b"],
        [0.10, 0.23, 0.48, 24.82],
        [0.10, 0.23, 0.48, 0.83, 1.22, 2.14, 24.82],
        beta=1.0,
    )
    # event_a: mean(shallow=0.5, mid=0.5, deep=2.0) = 1.0.
    # event_b: its alternating shallow errors cancel before the penalty = 0.0.
    # Equal event macro average = 0.5.
    torch.testing.assert_close(result, torch.tensor(0.5))
    result.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_deep_underprediction_factor_is_one_sided_and_depth_gated() -> None:
    target = torch.tensor([[[[1.0, 1.0]]]])
    prediction = torch.tensor([[[[0.0, 2.0]]]], requires_grad=True)
    mask = torch.ones_like(target, dtype=torch.bool)
    baseline = positive_depth_losses(
        prediction, None, target, mask, ["event_a"], 0.0, 0.0
    )["depth_linear"]
    asymmetric = positive_depth_losses(
        prediction,
        None,
        target,
        mask,
        ["event_a"],
        0.0,
        0.0,
        underprediction_factor=2.0,
        underprediction_min_depth_m=0.5,
    )["depth_linear"]
    shallow_gated = positive_depth_losses(
        prediction,
        None,
        target,
        mask,
        ["event_a"],
        0.0,
        0.0,
        underprediction_factor=2.0,
        underprediction_min_depth_m=1.5,
    )["depth_linear"]
    torch.testing.assert_close(baseline, torch.tensor(0.5))
    torch.testing.assert_close(asymmetric, torch.tensor(0.75))
    torch.testing.assert_close(shallow_gated, baseline)
    asymmetric.backward()
    assert prediction.grad is not None


def test_depth_exceedance_loss_rewards_correct_frozen_threshold_order() -> None:
    target = torch.tensor([[[[0.2, 1.5]]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    prediction = target.clone().requires_grad_()
    correct = event_depth_exceedance_loss(
        prediction,
        target,
        valid,
        ["event_a"],
        [0.1, 0.5, 2.0],
        temperature_m=0.1,
    )
    reversed_prediction = event_depth_exceedance_loss(
        target.flip(-1),
        target,
        valid,
        ["event_a"],
        [0.1, 0.5, 2.0],
        temperature_m=0.1,
    )
    assert correct < reversed_prediction
    correct.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()

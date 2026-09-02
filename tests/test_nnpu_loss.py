from __future__ import annotations

import pytest
import torch

from losses.pu_loss import nnpu_logistic_loss


@pytest.mark.parametrize(
    ("positive", "unlabeled"),
    [
        ([1, 0, 0, 0], [0, 1, 1, 1]),
        ([0, 0, 0, 0], [1, 1, 1, 1]),
        ([1, 1, 1, 1], [0, 0, 0, 0]),
        ([0, 0, 0, 0], [0, 0, 0, 0]),
    ],
)
def test_nnpu_empty_masks_are_finite(positive: list[int], unlabeled: list[int]) -> None:
    logits = torch.tensor(
        [[[[-1000.0, -10.0, 10.0, 1000.0]]]], requires_grad=True
    )
    p = torch.tensor(positive, dtype=torch.bool).view_as(logits)
    u = torch.tensor(unlabeled, dtype=torch.bool).view_as(logits)
    result = nnpu_logistic_loss(logits, p, u, 0.13, ["event"])
    assert all(torch.isfinite(value) for value in result.values())
    result["nnpu"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_nnpu_pixel_micro_is_event_invariant() -> None:
    logits = torch.tensor(
        [
            [[[-2.0, -1.0, 0.0, 1.0]]],
            [[[2.0, 1.0, 0.0, -1.0]]],
        ],
        requires_grad=True,
    )
    positive = torch.tensor(
        [
            [[[True, False, False, False]]],
            [[[True, True, False, False]]],
        ]
    )
    unlabeled = ~positive
    first = nnpu_logistic_loss(
        logits,
        positive,
        unlabeled,
        0.13,
        ["event_a", "event_b"],
        aggregation_mode="pixel_micro",
    )
    renamed = nnpu_logistic_loss(
        logits,
        positive,
        unlabeled,
        0.13,
        ["same", "same"],
        aggregation_mode="pixel_micro",
    )
    for key in first:
        torch.testing.assert_close(first[key], renamed[key], rtol=0, atol=0)
    first["nnpu"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

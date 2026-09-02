from __future__ import annotations

import torch

from models.heads import FloodDepthHeads


def test_conditional_v2_does_not_attenuate_depth_by_support() -> None:
    head = FloodDepthHeads(4, depth_output_semantics="conditional_positive_v2")
    outputs = head(torch.randn(2, 4, 8, 8))
    torch.testing.assert_close(outputs["depth"], outputs["conditional_depth"])
    torch.testing.assert_close(
        outputs["expected_depth"],
        outputs["support_probability"] * outputs["conditional_depth"],
    )


def test_legacy_v1_retains_probability_weighted_depth() -> None:
    head = FloodDepthHeads(4, depth_output_semantics="probability_weighted_v1")
    outputs = head(torch.randn(2, 4, 8, 8))
    torch.testing.assert_close(outputs["depth"], outputs["expected_depth"])

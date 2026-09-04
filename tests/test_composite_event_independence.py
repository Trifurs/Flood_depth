from __future__ import annotations

import copy

import torch

from datasets.contract import DatasetContract
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from losses.composite_loss import CompositeFloodDepthLoss


def test_sample_depth_bin_composite_loss_ignores_event_ids(config: dict) -> None:
    loss_config = dict(config["loss"])
    loss_config["supervised_reduction"] = "sample_depth_bin"
    contract = DatasetContract.load(config["dataset"]["contract"])
    normalizer = RobustNormalizer(config["dataset"]["train_stats"], contract)
    objective = CompositeFloodDepthLoss(
        loss_config,
        positive_prior=0.13,
        train_depth_bins=resolve_depth_stratification_bins(loss_config, normalizer),
    )
    target = torch.linspace(0.05, 2.0, 50).reshape(2, 1, 5, 5)
    depth = (target + 0.15).requires_grad_()
    zeros = torch.zeros_like(target)
    ones = torch.ones_like(target)
    outputs = {
        "depth": depth,
        "conditional_depth": depth,
        "positive_depth": depth,
        "expected_depth": depth,
        "support_logits": zeros,
        "uncertainty_scale": ones,
        "physical_features": {"z_hyd": zeros},
    }
    batch = {
        "label": target,
        "masks": {
            "valid_depth_mask": ones,
            "permanent_water_mask": zeros,
            "extreme_high_mask": zeros,
        },
        "validity": {
            "output_valid": ones,
            "s1_valid": ones,
            "s2_valid": ones,
            "dem_valid": ones,
        },
        "reliability": torch.zeros((2, 10, 5, 5)),
        "metadata": {"source_event_id": ["event_a", "event_b"]},
    }
    renamed = copy.deepcopy(batch)
    renamed["metadata"]["source_event_id"] = ["same", "same"]

    first_total, first_components = objective(outputs, batch, epoch=20)
    second_total, second_components = objective(outputs, renamed, epoch=20)

    torch.testing.assert_close(first_total, second_total, rtol=0, atol=0)
    assert first_components.keys() == second_components.keys()
    for key in first_components:
        torch.testing.assert_close(
            first_components[key], second_components[key], rtol=0, atol=0
        )
    first_total.backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()

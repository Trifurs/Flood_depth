import torch

import losses.composite_loss as composite
from losses.composite_loss import CompositeFloodDepthLoss


def test_zero_scheduled_weights_skip_optional_objectives(monkeypatch) -> None:
    config = {
        "lambda_depth": 1.0, "lambda_log": 0.0, "lambda_final": 0.0,
        "lambda_depth_bias": 0.0, "lambda_depth_exceedance": 0.0,
        "lambda_pu": 0.0, "lambda_unc": 0.0, "lambda_gradient": 0.0,
        "lambda_auxiliary": 0.0, "lambda_kan": 0.0, "lambda_wse": 0.0,
        "wse_start_epoch": 0, "wse_warmup_epochs": 1,
    }
    def fail(*args, **kwargs):
        raise AssertionError("disabled objective was evaluated")
    for name in ("event_depth_exceedance_loss", "nnpu_logistic_loss", "laplace_nll_loss",
                 "masked_gradient_consistency_loss", "auxiliary_depth_loss",
                 "reference_gated_wse_gradient_loss", "terrain_order_violation_loss",
                 "weak_wse_laplacian_loss"):
        monkeypatch.setattr(composite, name, fail)
    label = torch.ones(1, 1, 4, 4)
    batch = {
        "label": label,
        "masks": {"valid_depth_mask": torch.ones_like(label),
                  "permanent_water_mask": torch.zeros_like(label),
                  "extreme_high_mask": torch.zeros_like(label)},
        "validity": {"output_valid": torch.ones_like(label),
                     "s1_valid": torch.ones_like(label),
                     "s2_valid": torch.ones_like(label),
                     "dem_valid": torch.ones_like(label)},
        "reliability": torch.zeros(1, 12, 4, 4),
    }
    outputs = {"depth": label, "positive_depth": label,
               "conditional_depth": label, "support_logits": torch.zeros_like(label),
               "uncertainty_scale": torch.ones_like(label),
               "physical_features": {"z_hyd": torch.zeros_like(label)}}
    total, terms = CompositeFloodDepthLoss(config, 0.1)(outputs, batch, 0)
    assert torch.isfinite(total) and terms["pu_effective_weight"] == 0

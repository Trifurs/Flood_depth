"""Configured PA-HydroKAN objective with strict partial-label masking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from losses.depth_losses import (
    event_depth_exceedance_loss,
    laplace_nll_loss,
    positive_depth_losses,
)
from losses.physics_losses import (
    reference_gated_wse_gradient_loss,
    terrain_order_violation_loss,
    weak_wse_laplacian_loss,
)
from losses.pu_loss import nnpu_logistic_loss


def _event_ids(batch: Mapping[str, Any]) -> Sequence[str] | None:
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    values = metadata.get("source_event_id")
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    if isinstance(values, str):
        return [values]
    return None


class CompositeFloodDepthLoss(nn.Module):
    def __init__(
        self,
        loss_config: Mapping[str, Any],
        positive_prior: float,
        train_depth_bins: Sequence[float] | None = None,
        primary_depth_bins: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.config = dict(loss_config)
        self.positive_prior = float(positive_prior)
        self.train_depth_bins = tuple(float(value) for value in (train_depth_bins or ()))
        self.primary_depth_bins = tuple(
            float(value) for value in (primary_depth_bins or ())
        )

    def wse_weight(self, epoch: int) -> float:
        target = float(self.config["lambda_wse"])
        start = int(self.config["wse_start_epoch"])
        warmup = max(1, int(self.config["wse_warmup_epochs"]))
        if epoch < start:
            return 0.0
        return target * min(1.0, (epoch - start + 1) / warmup)

    def forward(
        self, outputs: Mapping[str, Any], batch: Mapping[str, Any], epoch: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        label = batch["label"]
        masks = batch["masks"]
        validity = batch["validity"]
        positive = masks["valid_depth_mask"] > 0.5
        unlabeled = (
            (validity["output_valid"] > 0.5)
            & ~positive
            & ~(masks["permanent_water_mask"] > 0.5)
            & ~(masks["extreme_high_mask"] > 0.5)
        )
        aggregation_mode = str(self.config.get("supervised_reduction", "auto"))
        event_independent = aggregation_mode in {
            "pixel_micro",
            "depth_bin_macro",
            "sample_depth_bin",
        }
        # Pixel-first deployment must be invariant to event labels. Event metadata
        # remains available only for frozen legacy objectives and diagnostics.
        events = None if event_independent else _event_ids(batch)
        auxiliary_aggregation = (
            "pixel_micro"
            if aggregation_mode in {"pixel_micro", "depth_bin_macro"}
            else "event_macro"
        )
        lambda_final = float(self.config["lambda_final"])
        components = positive_depth_losses(
            outputs.get("conditional_depth", outputs["positive_depth"]),
            outputs.get("expected_depth", outputs["depth"])
            if lambda_final != 0.0
            else None,
            label,
            positive,
            events,
            float(self.config["lambda_log"]),
            lambda_final,
            self.train_depth_bins,
            self.primary_depth_bins,
            float(self.config.get("depth_bias_beta_m", 0.1)),
            float(self.config.get("depth_underprediction_factor", 1.0)),
            float(self.config.get("depth_underprediction_min_m", 0.0)),
            aggregation_mode,
            str(self.config.get("depth_linear_loss", "smooth_l1")),
        )
        exceedance = event_depth_exceedance_loss(
            outputs["depth"],
            label,
            positive,
            events,
            self.train_depth_bins,
            float(self.config.get("depth_exceedance_temperature_m", 0.1)),
            auxiliary_aggregation,
        )
        components["depth_exceedance"] = exceedance
        pu = nnpu_logistic_loss(
            outputs["support_logits"],
            positive,
            unlabeled,
            self.positive_prior,
            events,
            auxiliary_aggregation,
        )
        components.update(pu)
        uncertainty = laplace_nll_loss(
            outputs["depth"],
            label,
            outputs["uncertainty_scale"],
            positive,
            events,
            self.train_depth_bins,
            self.primary_depth_bins,
            aggregation_mode,
        )
        components["uncertainty"] = uncertainty
        sensor_valid = torch.maximum(validity["s1_valid"], validity["s2_valid"])
        day_difference = batch["reliability"][:, 9:10]
        wse_mode = str(self.config.get("wse_mode", "absolute_laplacian"))
        if wse_mode == "reference_gated_gradient":
            wse = reference_gated_wse_gradient_loss(
                outputs["depth"],
                label,
                outputs["physical_features"]["z_hyd"],
                positive,
                validity["dem_valid"],
                sensor_valid,
                day_difference,
                events,
                float(self.config["wse_time_sigma"]),
                float(self.config["wse_reference_sigma_m"]),
                float(self.config["wse_terrain_sigma_m"]),
                float(self.config["wse_gradient_beta_m"]),
                auxiliary_aggregation,
            )
        elif wse_mode == "terrain_order":
            wse = terrain_order_violation_loss(
                outputs["depth"],
                outputs["physical_features"]["z_hyd"],
                positive,
                validity["dem_valid"],
                sensor_valid,
                day_difference,
                events,
                float(self.config["wse_time_sigma"]),
                float(self.config["terrain_order_min_step_m"]),
                float(self.config["terrain_order_max_step_m"]),
                float(self.config["terrain_order_beta_m"]),
                auxiliary_aggregation,
            )
        elif wse_mode == "absolute_laplacian":
            wse = weak_wse_laplacian_loss(
                outputs["depth"],
                outputs["physical_features"]["z_hyd"],
                positive,
                validity["dem_valid"],
                sensor_valid,
                day_difference,
                float(self.config["wse_time_sigma"]),
            )
        else:
            raise ValueError(
                "loss.wse_mode must be 'absolute_laplacian', "
                f"'reference_gated_gradient', or 'terrain_order', got {wse_mode!r}"
            )
        components["wse"] = wse
        effective_wse = self.wse_weight(epoch)
        total = (
            float(self.config["lambda_depth"]) * components["depth"]
            + float(self.config.get("lambda_depth_bias", 0.0))
            * components["depth_bias"]
            + float(self.config.get("lambda_depth_exceedance", 0.0)) * exceedance
            + float(self.config["lambda_pu"]) * components["nnpu"]
            + float(self.config["lambda_unc"]) * uncertainty
            + effective_wse * wse
        )
        components["total"] = total
        components["wse_effective_weight"] = total.new_tensor(effective_wse)
        components["positive_pixels"] = total.new_tensor(float(positive.sum().item()))
        components["unlabeled_pixels"] = total.new_tensor(float(unlabeled.sum().item()))
        return total, components

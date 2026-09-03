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
    tail_underprediction_loss,
)
from losses.physics_losses import (
    gated_terrain_order_loss,
    reference_gated_wse_gradient_loss,
    terrain_order_violation_loss,
    tolerant_wse_slope_loss,
    weak_wse_laplacian_loss,
)
from losses.pu_loss import nnpu_logistic_loss
from losses.multiscale_losses import auxiliary_depth_loss, masked_gradient_consistency_loss
from datasets.preprocessing import RELIABILITY_NAMES
from losses.task_adaptive_depth_loss import task_adaptive_positive_depth_loss


def _reliability_names(batch: Mapping[str, Any]) -> tuple[str, ...]:
    values = batch.get("reliability_names")
    if isinstance(values, (list, tuple)) and values and all(isinstance(value, str) for value in values):
        return tuple(values)
    if isinstance(values, (list, tuple)) and values and all(
        isinstance(value, (list, tuple)) and value and isinstance(value[0], str)
        for value in values
    ):
        return tuple(str(value[0]) for value in values)
    metadata = batch.get("metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get("reliability_names")
        if isinstance(nested, (list, tuple)) and nested and all(
            isinstance(value, (list, tuple)) and value and isinstance(value[0], str)
            for value in nested
        ):
            return tuple(str(value[0]) for value in nested)
    return RELIABILITY_NAMES


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
        train_depth_bin_counts: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.config = dict(loss_config)
        self.lambda_absolute_laplacian = float(
            self.config.get("lambda_absolute_laplacian", self.config.get("lambda_wse", 0.0))
        )
        self.positive_prior = float(positive_prior)
        self.train_depth_bins = tuple(float(value) for value in (train_depth_bins or ()))
        self.primary_depth_bins = tuple(
            float(value) for value in (primary_depth_bins or ())
        )
        self.train_depth_bin_counts = tuple(float(value) for value in (train_depth_bin_counts or ()))

    def wse_weight(self, epoch: int) -> float:
        target = float(self.config["lambda_wse"])
        if str(self.config.get("wse_mode", "absolute_laplacian")) == "absolute_laplacian" and "lambda_absolute_laplacian" in self.config:
            target = self.lambda_absolute_laplacian
        start = int(self.config["wse_start_epoch"])
        warmup = max(1, int(self.config["wse_warmup_epochs"]))
        if epoch < start:
            return 0.0
        return target * min(1.0, (epoch - start + 1) / warmup)

    def scheduled_weight(self, name: str, epoch: int) -> float:
        target = float(self.config.get(f"lambda_{name}", 0.0))
        start = int(self.config.get(f"{name}_start_epoch", 0))
        warmup = int(self.config.get(f"{name}_warmup_epochs", 0))
        if epoch < start:
            return 0.0
        if warmup <= 0:
            weight = target
        else:
            weight = target * min(1.0, (epoch - start + 1) / warmup)
        if name == "auxiliary":
            decay = str(self.config.get("auxiliary_decay", "constant"))
            decay_start = float(self.config.get("auxiliary_decay_start_fraction", 0.2))
            decay_end = float(self.config.get("auxiliary_decay_end_fraction", 0.5))
            total_epochs = max(1.0, float(self.config.get("schedule_total_epochs", self.config.get("epochs", 1))))
            fraction = float(epoch) / total_epochs
            if decay in {"linear", "cosine"} and fraction > decay_start:
                progress = min(1.0, (fraction - decay_start) / max(decay_end - decay_start, 1e-6))
                multiplier = 1.0 - progress if decay == "linear" else 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
                weight *= multiplier
        return weight

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
        effective_pu = self.scheduled_weight("pu", epoch)
        effective_unc = self.scheduled_weight("unc", epoch)
        effective_gradient = self.scheduled_weight("gradient", epoch)
        effective_auxiliary = self.scheduled_weight("auxiliary", epoch)
        effective_kan = self.scheduled_weight("kan", epoch)
        effective_tail = self.scheduled_weight("tail", epoch)
        effective_wse = self.wse_weight(epoch)
        zero = label.sum() * 0.0
        lambda_final = float(self.config["lambda_final"])
        prediction = outputs.get("conditional_depth", outputs["positive_depth"])
        if str(self.config.get("objective_mode", "legacy")) == "task_adaptive":
            components = task_adaptive_positive_depth_loss(
                prediction, label, positive, self.train_depth_bins,
                beta_m=float(self.config.get("depth_huber_beta_m", 0.5)),
                log_weight=float(self.config.get("lambda_log", 0.15)),
                balance=bool(self.config.get("soft_depth_balance", True)),
                under_alpha=float(self.config.get("tail_underprediction_alpha", 0.0)),
                under_min_m=float(self.config.get("depth_underprediction_min_m", 0.48)),
                balance_alpha=float(self.config.get("soft_depth_balance_alpha", 0.5)),
                balance_tau=float(self.config.get("soft_depth_balance_tau", 10.0)),
                train_bin_counts=self.train_depth_bin_counts or None,
            )
        else:
            components = positive_depth_losses(
                prediction,
                outputs.get("expected_depth", outputs["depth"])
                if lambda_final != 0.0 and bool(self.config.get("support_weighted_supervision", True)) else None,
                label, positive, events, float(self.config["lambda_log"]),
                lambda_final, self.train_depth_bins, self.primary_depth_bins,
                float(self.config.get("depth_bias_beta_m", 0.1)),
                float(self.config.get("depth_underprediction_factor", 1.0)),
                float(self.config.get("depth_underprediction_min_m", 0.0)),
                aggregation_mode, str(self.config.get("depth_linear_loss", "smooth_l1")),
                float(self.config.get("depth_huber_beta_m", 1.0)),
                float(self.config.get("log_depth_huber_beta", 1.0)),
            )
        exceedance = (
            event_depth_exceedance_loss(
                outputs["depth"], label, positive, events, self.train_depth_bins,
                float(self.config.get("depth_exceedance_temperature_m", 0.1)),
                auxiliary_aggregation,
            ) if float(self.config.get("lambda_depth_exceedance", 0.0)) != 0.0 else zero
        )
        components["depth_exceedance"] = exceedance
        tail_threshold = self.config.get("tail_threshold_m")
        if tail_threshold is None:
            tail_threshold = self.train_depth_bins[-2] if len(self.train_depth_bins) >= 3 else 0.0
        tail = (
            tail_underprediction_loss(
                prediction, label, positive, float(tail_threshold),
                float(self.config.get("tail_margin_m", 0.15)),
                float(self.config.get("tail_huber_beta_m", 0.25)),
                "pixel_micro" if aggregation_mode in {"auto", "pixel_micro", "depth_bin_macro"} else "sample_macro",
            ) if effective_tail != 0.0 else zero
        )
        components["tail"] = tail
        if effective_pu != 0.0 and "support_logits" not in outputs:
            raise ValueError(
                "PU support loss is enabled, but this model exposes no support branch"
            )
        pu = (
            nnpu_logistic_loss(
                outputs["support_logits"], positive, unlabeled, self.positive_prior,
                events, auxiliary_aggregation,
            ) if effective_pu != 0.0 else {
                "nnpu": zero, "pu_positive_risk": zero,
                "pu_negative_risk_raw": zero, "pu_negative_risk_nonnegative": zero,
            }
        )
        components.update(pu)
        uncertainty = (
            laplace_nll_loss(
                outputs["depth"] if bool(self.config.get("uncertainty_backprop_to_depth", True)) else outputs["depth"].detach(),
                label, outputs["uncertainty_scale"], positive, events,
                self.train_depth_bins, self.primary_depth_bins, aggregation_mode,
            ) if effective_unc != 0.0 else zero
        )
        components["uncertainty"] = uncertainty
        gradient = (
            masked_gradient_consistency_loss(
                outputs.get("conditional_depth", outputs["depth"]), label, positive,
                float(self.config.get("gradient_huber_beta_m", 0.1)),
            ) if effective_gradient != 0.0 else zero
        )
        components["gradient"] = gradient
        auxiliary, auxiliary_terms = (
            auxiliary_depth_loss(
                outputs.get("auxiliary_depths", ()), label, positive,
                self.config.get("auxiliary_depth_weights", ()),
                float(self.config.get("depth_huber_beta_m", 1.0)),
            ) if effective_auxiliary != 0.0 else (zero, [])
        )
        components["auxiliary"] = auxiliary
        for index, value in enumerate(auxiliary_terms):
            components[f"auxiliary_{index}"] = value
        sensor_valid = validity["s1_valid"]
        if "s2_valid" in validity:
            sensor_valid = torch.maximum(sensor_valid, validity["s2_valid"])
        reliability_names = _reliability_names(batch)
        day_difference = batch["reliability"].new_zeros(
            batch["reliability"].shape[0], 1, *batch["reliability"].shape[-2:]
        )
        if "absolute_normalized_sensor_day_difference" in reliability_names:
            day_index = reliability_names.index("absolute_normalized_sensor_day_difference")
            day_difference = batch["reliability"][:, day_index : day_index + 1]
        wse_mode = str(self.config.get("wse_mode", "absolute_laplacian"))
        if effective_wse == 0.0:
            wse = zero
        elif wse_mode == "reference_gated_gradient":
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
                outputs["physical_features"].get("local_relief"),
                float(self.config.get("terrain_order_high_relief_threshold_m", 12.0)),
                float(self.config.get("terrain_order_high_relief_decay_m", 8.0)),
            )
        elif wse_mode == "v14_terrain_order":
            wse, order_diag = gated_terrain_order_loss(
                outputs["conditional_depth"], outputs["physical_features"]["physics_elevation"],
                positive, validity["dem_valid"], sensor_valid,
                depth_order_tolerance_m=float(self.config.get("depth_order_tolerance_m", 0.02)),
                huber_beta_m=float(self.config.get("terrain_order_huber_beta_m", 0.05)),
                terrain_step_min_m=float(self.config.get("terrain_step_min_m", 0.02)),
                terrain_step_max_m=float(self.config.get("terrain_step_max_m", 0.75)),
                terrain_sigma_m=float(self.config.get("terrain_sigma_m", 0.75)),
                relief=outputs["physical_features"].get("local_relief"),
                relief_sigma_m=float(self.config.get("terrain_relief_sigma_m", 12.0)),
                return_diagnostics=True,
            )
            components["terrain_order_violation_fraction"] = order_diag["violation_fraction"]
            components["terrain_order_violation_magnitude"] = order_diag["violation_magnitude"]
        elif wse_mode == "v14_wse_slope":
            wse, slope_diag = tolerant_wse_slope_loss(
                outputs["conditional_depth"], outputs["physical_features"]["physics_elevation"],
                positive, validity["dem_valid"], sensor_valid,
                pixel_size_m=float(self.config.get("terrain_pixel_size_m", 20.0)),
                wse_slope_tolerance=float(self.config.get("wse_slope_tolerance", 0.02)),
                huber_beta=float(self.config.get("wse_slope_huber_beta", 0.01)),
                relief=outputs["physical_features"].get("local_relief"),
                relief_threshold_m=float(self.config.get("wse_relief_threshold_m", 12.0)),
                return_diagnostics=True,
            )
            components["wse_slope_violation_fraction"] = slope_diag["violation_fraction"]
            components["wse_slope_violation_magnitude"] = slope_diag["violation_magnitude"]
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
                f"'reference_gated_gradient', 'terrain_order', 'v14_terrain_order', or 'v14_wse_slope', got {wse_mode!r}"
            )
        components["wse"] = wse
        if effective_kan != 0.0:
            kan_magnitude = outputs.get("graph_diagnostics", {}).get(
                "kan_coefficient_magnitude", zero
            )
            kan_smoothness = outputs.get("graph_diagnostics", {}).get(
                "kan_coefficient_smoothness", zero
            )
        else:
            kan_magnitude = zero
            kan_smoothness = zero
        components["kan_magnitude"] = kan_magnitude
        components["kan_smoothness"] = kan_smoothness
        components["kan_monotonicity"] = outputs.get("graph_diagnostics", {}).get("kan_monotonicity", zero)
        components["kan_curve_smoothness"] = outputs.get("graph_diagnostics", {}).get("kan_curve_smoothness", zero)
        total = (
            float(self.config["lambda_depth"]) * components["depth"]
            + float(self.config.get("lambda_depth_bias", 0.0))
            * components["depth_bias"]
            + float(self.config.get("lambda_depth_exceedance", 0.0)) * exceedance
            + effective_tail * tail
            + effective_pu * components["nnpu"]
            + effective_unc * uncertainty
            + effective_gradient * gradient
            + effective_auxiliary * auxiliary
            + effective_wse * wse
            + effective_kan * (kan_magnitude + kan_smoothness)
            + float(self.config.get("lambda_kan_mono", 0.0)) * components["kan_monotonicity"]
            + float(self.config.get("lambda_kan_smooth", 0.0)) * components["kan_curve_smoothness"]
        )
        components["total"] = total
        components["wse_effective_weight"] = total.new_tensor(effective_wse)
        components["pu_effective_weight"] = total.new_tensor(effective_pu)
        components["unc_effective_weight"] = total.new_tensor(effective_unc)
        components["gradient_effective_weight"] = total.new_tensor(effective_gradient)
        components["auxiliary_effective_weight"] = total.new_tensor(effective_auxiliary)
        components["kan_effective_weight"] = total.new_tensor(effective_kan)
        components["tail_effective_weight"] = total.new_tensor(effective_tail)
        components["positive_pixels"] = total.new_tensor(float(positive.sum().item()))
        components["unlabeled_pixels"] = total.new_tensor(float(unlabeled.sum().item()))
        return total, components

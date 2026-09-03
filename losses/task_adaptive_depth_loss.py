"""Small, train-only task-adaptive depth objective for v13.2."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from losses.depth_losses import masked_micro_mean, sample_depth_bin_macro_mean, depth_bin_macro_mean


def soft_depth_balance_weights(target: torch.Tensor, positive: torch.Tensor,
                               train_bins: Sequence[float], minimum: float = 0.5,
                               maximum: float = 3.0, alpha: float = 0.5,
                               tau: float = 10.0,
                               train_bin_counts: Sequence[float] | None = None) -> torch.Tensor:
    """Capped inverse-frequency soft-bin weights with a mean-one normalization.

    ``train_bin_counts`` is expected to be frozen from train-only statistics.  When
    absent, counts are derived from the current positive tensor for backwards
    compatibility with the v13 diagnostic helper; Hydro-v14 configs should provide
    frozen counts when available.
    """
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid soft depth weight bounds")
    if alpha < 0 or tau < 0:
        raise ValueError("alpha and tau must be nonnegative")
    edges = sorted(float(x) for x in train_bins)
    selected = positive > 0.5
    if len(edges) < 2:
        raw = torch.ones_like(target)
    else:
        internal = target.new_tensor(edges[1:-1])
        bin_index = torch.bucketize(target, internal, right=False)
        n_bins = len(edges) - 1
        if train_bin_counts is None:
            counts = torch.stack([(selected & (bin_index == i)).sum() for i in range(n_bins)]).to(target.dtype)
        else:
            if len(train_bin_counts) != n_bins:
                raise ValueError("train_bin_counts must have one entry per depth bin")
            counts = target.new_tensor([float(value) for value in train_bin_counts])
        raw_by_bin = (counts + float(tau)).clamp_min(1e-6).pow(-float(alpha))
        raw = raw_by_bin[bin_index]
    selected = positive > 0.5
    mean = raw[selected].mean() if torch.any(selected) else raw.new_tensor(1.0)
    weights = (raw / mean.clamp_min(1e-6)).clamp(minimum, maximum)
    # Re-normalize after clipping so the supervised term remains on the same
    # scale as the legacy Huber objective.
    mean2 = weights[selected].mean() if torch.any(selected) else weights.new_tensor(1.0)
    return (weights / mean2.clamp_min(1e-6)).clamp(minimum, maximum)


def tail_underprediction_factor(prediction: torch.Tensor, target: torch.Tensor,
                                positive: torch.Tensor, train_bins: Sequence[float],
                                alpha: float = 0.0) -> torch.Tensor:
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if alpha == 0 or len(train_bins) < 3:
        return torch.ones_like(target)
    threshold = float(sorted(float(x) for x in train_bins)[2])
    scale = max(float(sorted(float(x) for x in train_bins)[-2]) - threshold, 0.1)
    tail = torch.sigmoid((target - threshold) / (0.25 * scale)) * (target >= threshold).to(target.dtype)
    under = (prediction.detach() < target).to(target.dtype)
    return 1.0 + float(alpha) * tail * under * (positive > 0.5).to(target.dtype)


def task_adaptive_positive_depth_loss(prediction: torch.Tensor, target: torch.Tensor,
                                      positive: torch.Tensor, train_bins: Sequence[float],
                                      beta_m: float = 0.5, log_weight: float = 0.15,
                                      balance: bool = True, under_alpha: float = 0.0,
                                      under_min_m: float = 0.48,
                                      balance_alpha: float = 0.5,
                                      balance_tau: float = 10.0,
                                      train_bin_counts: Sequence[float] | None = None) -> dict[str, torch.Tensor]:
    if beta_m <= 0 or log_weight < 0:
        raise ValueError("beta_m must be positive and log_weight nonnegative")
    metric_pixels = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta_m)
    if balance:
        metric_pixels = metric_pixels * soft_depth_balance_weights(
            target, positive, train_bins, alpha=balance_alpha, tau=balance_tau,
            train_bin_counts=train_bin_counts,
        )
    if under_alpha:
        factor = tail_underprediction_factor(prediction, target, positive, train_bins, under_alpha)
        factor = torch.where(target >= float(under_min_m), factor, torch.ones_like(factor))
        metric_pixels = metric_pixels * factor
    metric = masked_micro_mean(metric_pixels, positive)
    log_pixels = F.smooth_l1_loss(torch.log1p(prediction), torch.log1p(target.clamp_min(0.0)), reduction="none", beta=1.0)
    logarithmic = masked_micro_mean(log_pixels, positive)
    total = metric + float(log_weight) * logarithmic
    return {"depth": total, "depth_linear": metric, "depth_log": logarithmic,
            "depth_final": total.detach() * 0.0, "depth_bias": total.detach() * 0.0}

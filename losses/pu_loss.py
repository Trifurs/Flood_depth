"""Numerically stable non-negative positive-unlabeled logistic risk."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from losses.depth_losses import event_macro_masked_mean, masked_micro_mean


def nnpu_logistic_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    positive_prior: float,
    event_ids: Sequence[str] | None = None,
    aggregation_mode: str = "event_macro",
) -> dict[str, torch.Tensor]:
    if not 0.0 < positive_prior < 1.0:
        raise ValueError(f"positive_prior must be in (0,1), received {positive_prior}")
    if aggregation_mode == "pixel_micro":
        reducer = lambda values, mask: masked_micro_mean(values, mask)
    elif aggregation_mode in {"auto", "event_macro"}:
        reducer = lambda values, mask: event_macro_masked_mean(
            values, mask, event_ids
        )
    else:
        raise ValueError(
            "nnPU aggregation_mode must be 'pixel_micro' or 'event_macro', "
            f"received {aggregation_mode!r}"
        )
    positive_loss = F.softplus(-logits)
    negative_loss = F.softplus(logits)
    positive_risk = float(positive_prior) * reducer(positive_loss, positive_mask)
    unlabeled_negative = reducer(negative_loss, unlabeled_mask)
    positive_as_negative = reducer(negative_loss, positive_mask)
    raw_negative_risk = unlabeled_negative - float(positive_prior) * positive_as_negative
    nonnegative_risk = torch.clamp_min(raw_negative_risk, 0.0)
    return {
        "nnpu": positive_risk + nonnegative_risk,
        "pu_positive_risk": positive_risk,
        "pu_negative_risk_raw": raw_negative_risk,
        "pu_negative_risk_nonnegative": nonnegative_risk,
    }

"""Binary flood-extent losses with explicit validity masking."""

from __future__ import annotations

import torch


def masked_soft_iou_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pure Soft-IoU loss and score, averaged across input rasters."""

    if logits.shape != target.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, target, and valid_mask must share [B,1,H,W]")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("binary extent tensors must have shape [B,1,H,W]")
    valid = (valid_mask > 0.5).to(logits.dtype)
    truth = (target > 0.5).to(logits.dtype)
    probability = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probability * truth * valid).sum(dim=dimensions)
    union = ((probability + truth - probability * truth) * valid).sum(dim=dimensions)
    score = (intersection + epsilon) / (union + epsilon)
    return 1.0 - score.mean(), score.mean()

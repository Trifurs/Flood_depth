"""Compatibility facade for frozen, train-only soft depth balancing.

New training must construct :class:`FrozenSoftDepthBalance` once from the
canonical train split. This module retains the historic function name for
external callers, but it no longer rescales weights using the current batch.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from losses.frozen_soft_depth_balance import FrozenSoftDepthBalance


def _validate_edges(train_bins: Sequence[float]) -> tuple[float, ...]:
    edges = tuple(sorted(float(value) for value in train_bins))
    if len(edges) < 2:
        return edges
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("train_bins must be strictly increasing")
    return edges


def soft_depth_balance_weights(
    target: torch.Tensor,
    positive: torch.Tensor,
    train_bins: Sequence[float],
    minimum: float = 0.5,
    maximum: float = 3.0,
    alpha: float = 0.5,
    tau: float = 10.0,
    train_bin_counts: Sequence[float] | None = None,
) -> torch.Tensor:
    """Return a static train-derived curve without minibatch normalization.

    ``train_bin_counts`` is a legacy compatibility input. It creates a frozen
    count-based approximation when exact train targets are unavailable. Calls
    with positive pixels and no train statistics are rejected rather than silently
    deriving a target weight from the current minibatch.
    """

    if alpha < 0.0 or tau < 0.0:
        raise ValueError("soft_depth_balance alpha and tau must be nonnegative")
    selected = positive > 0.5
    if not bool(torch.any(selected)):
        return torch.ones_like(target)
    edges = _validate_edges(train_bins)
    if len(edges) < 2:
        return torch.ones_like(target)
    if train_bin_counts is None:
        raise ValueError(
            "soft-depth balance requires frozen train-only statistics; pass a "
            "FrozenSoftDepthBalance to task_adaptive_positive_depth_loss"
        )
    balance = FrozenSoftDepthBalance.from_bin_counts(
        edges,
        train_bin_counts,
        minimum=minimum,
        maximum=maximum,
        alpha=alpha,
        tau=tau,
    )
    return balance.weights(target, positive)

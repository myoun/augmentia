"""Training losses for conditional V4 prefix distributions."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from v3.factorized import FactorizedPolicyError

from .batch import PrefixBatchV4, assert_visit_conservation


def conditional_prefix_cross_entropy(
    logits: torch.Tensor,
    prefixes: PrefixBatchV4,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross entropy over opened legal children at each visited prefix.

    Each prefix term is weighted by its descendant visit mass divided by its
    decision's root visits. Therefore the sum is exactly the negative log
    likelihood of the complete-action visit distribution under products of
    conditional probabilities. Zero-mass prefixes contribute no loss.
    """
    if logits.shape != prefixes.visit_counts.shape:
        raise ValueError("prefix logits and visit counts must have the same shape")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    assert_visit_conservation(prefixes)

    training_mask = prefixes.legal_mask & prefixes.opened_mask
    if torch.any((prefixes.visit_counts > 0) & ~training_mask):
        raise FactorizedPolicyError("a positive target must be an opened legal child")
    masked_logits = logits.masked_fill(~training_mask, torch.finfo(logits.dtype).min)
    row_visits = prefixes.visit_counts.sum(dim=1)
    target = prefixes.visit_counts / row_visits.clamp_min(1).unsqueeze(1)
    log_probabilities = F.log_softmax(masked_logits, dim=1)
    row_loss = -(target * log_probabilities).sum(dim=1)
    row_loss = torch.where(row_visits > 0, row_loss * prefixes.prefix_mass, torch.zeros_like(row_loss))

    if reduction == "none":
        return row_loss
    total = row_loss.sum()
    if reduction == "sum":
        return total
    decision_count = prefixes.complete_action_visits.shape[0]
    return total / max(1, decision_count)

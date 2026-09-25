"""Training losses for conditional V4 prefix distributions."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from v3.factorized import FactorizedPolicyError

from .batch import PrefixBatchV4


def _require_validated_prefixes(prefixes: PrefixBatchV4) -> None:
    """Reject batches that bypassed the conservation-checking collator."""

    if not prefixes.validated:
        raise FactorizedPolicyError(
            "prefix batch must be validated by collate_factorized_batch"
        )


def conditional_prefix_policy_metrics(
    logits: torch.Tensor,
    prefixes: PrefixBatchV4,
    *,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cross entropy, target entropy, and KL for V4 prefix targets.

    Each prefix term is weighted by its descendant visit mass divided by its
    decision's root visits. The weighted conditional target entropy therefore
    obeys the same chain rule as the policy cross entropy, so CE = H + KL at
    decision level. Zero-mass prefixes contribute no metric mass.
    """
    if logits.shape != prefixes.visit_counts.shape:
        raise ValueError("prefix logits and visit counts must have the same shape")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    _require_validated_prefixes(prefixes)

    training_mask = prefixes.legal_mask & prefixes.opened_mask
    masked_logits = logits.masked_fill(~training_mask, torch.finfo(logits.dtype).min)
    row_visits = prefixes.visit_counts.sum(dim=1)
    target = prefixes.visit_counts / row_visits.clamp_min(1).unsqueeze(1)
    log_probabilities = F.log_softmax(masked_logits, dim=1)
    log_target = torch.where(target > 0, target.log(), torch.zeros_like(target))

    row_cross_entropy = -(target * log_probabilities).sum(dim=1)
    row_target_entropy = -(target * log_target).sum(dim=1)
    row_kl = (target * (log_target - log_probabilities)).sum(dim=1)

    has_visits = row_visits > 0
    row_cross_entropy = torch.where(
        has_visits,
        row_cross_entropy * prefixes.prefix_mass,
        torch.zeros_like(row_cross_entropy),
    )
    row_target_entropy = torch.where(
        has_visits,
        row_target_entropy * prefixes.prefix_mass,
        torch.zeros_like(row_target_entropy),
    )
    row_kl = torch.where(
        has_visits,
        row_kl * prefixes.prefix_mass,
        torch.zeros_like(row_kl),
    )

    if reduction == "none":
        return row_cross_entropy, row_target_entropy, row_kl

    cross_entropy = row_cross_entropy.sum()
    target_entropy = row_target_entropy.sum()
    kl = row_kl.sum()
    if reduction == "sum":
        return cross_entropy, target_entropy, kl

    decision_count = prefixes.complete_action_visits.shape[0]
    divisor = max(1, decision_count)
    return cross_entropy / divisor, target_entropy / divisor, kl / divisor


def mixed_row_shadow_mask(prefixes: PrefixBatchV4) -> torch.Tensor:
    """Return visited rows that contain both opened and closed legal children."""

    legal = prefixes.legal_mask
    opened = legal & prefixes.opened_mask
    closed = legal & ~prefixes.opened_mask
    return (
        (prefixes.visit_counts.sum(dim=1) > 0)
        & opened.any(dim=1)
        & closed.any(dim=1)
    )


def mixed_row_all_legal_shadow_cross_entropy(
    logits: torch.Tensor,
    prefixes: PrefixBatchV4,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """All-legal CE over visited prefix rows that contain opened and closed children.

    This is an auxiliary training signal for the opened-only objective. The
    primary production loss remains :func:`conditional_prefix_cross_entropy`.
    Rows with no opened child (including zero-target, all-closed rows) are
    excluded, so the shadow objective cannot assign them a gradient.
    """

    if logits.shape != prefixes.visit_counts.shape:
        raise ValueError("prefix logits and visit counts must have the same shape")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    _require_validated_prefixes(prefixes)
    mixed_visited = mixed_row_shadow_mask(prefixes)

    legal = prefixes.legal_mask
    row_visits = prefixes.visit_counts.sum(dim=1)
    masked_logits = logits.masked_fill(
        ~legal, torch.finfo(logits.dtype).min
    )
    target = prefixes.visit_counts / row_visits.clamp_min(1).unsqueeze(1)
    log_probabilities = F.log_softmax(masked_logits, dim=1)
    row_cross_entropy = -(target * log_probabilities).sum(dim=1)
    row_cross_entropy = torch.where(
        mixed_visited,
        row_cross_entropy * prefixes.prefix_mass,
        torch.zeros_like(row_cross_entropy),
    )
    if reduction == "none":
        return row_cross_entropy

    total = row_cross_entropy.sum()
    if reduction == "sum":
        return total
    decision_count = prefixes.complete_action_visits.shape[0]
    return total / max(1, decision_count)


def conditional_prefix_cross_entropy(
    logits: torch.Tensor,
    prefixes: PrefixBatchV4,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross entropy over opened legal children at each visited prefix."""
    cross_entropy, _target_entropy, _kl = conditional_prefix_policy_metrics(
        logits, prefixes, reduction=reduction
    )
    return cross_entropy

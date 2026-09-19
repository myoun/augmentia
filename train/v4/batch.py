"""Tensor collation for conditional factorized V4 policy decisions.

The engine continues to serialize complete ``ActionDescriptorV3`` mappings.  This
module deterministically expands those mappings into one training row per trie
prefix.  Complete-action visits remain in the batch so conservation can be checked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch

from v3.factorized import FactorizedPolicyError, Path, Token, factorized_action_path
from v3.tensorize import (
    ABILITY_IDS,
    ACTION_KIND_IDS,
    TARGET_KIND_IDS,
    V3Batch,
    V3Vocabulary,
    tensorize_cases,
)

# Zero is padding for every categorical tensor.
FACTOR_FIELD_IDS = {
    "family": 1,
    "piece": 2,
    "destination": 3,
    "promotion": 4,
    "ability": 5,
    "target_kind": 6,
    "target_square": 7,
    "target_sequence": 8,
    "card": 9,
    "offer": 10,
    "choice": 11,
    "option": 12,
    "complete": 13,
}
FACTOR_FIELD_COUNT = len(FACTOR_FIELD_IDS)
FACTOR_CATEGORY_COUNT = max(
    max(ACTION_KIND_IDS.values()), max(ABILITY_IDS.values()), max(TARGET_KIND_IDS.values())
)


@dataclass(frozen=True)
class FactorEdgeBatch:
    """Structured typed trie edges, shaped ``[rows, width]``.

    ``index`` keeps raw stable V3 indexes (including zero) and uses ``-1`` for
    non-index values and padding. ``category`` contains the appropriate stable V3
    enum for family, ability, and target kind. ``ordinal`` is one-based only for
    ordered ``target_N`` fields. Keeping these channels separate prevents unrelated
    types with the same wire value from aliasing.
    """

    field: torch.Tensor
    category: torch.Tensor
    index: torch.Tensor
    ordinal: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device | str) -> "FactorEdgeBatch":
        return FactorEdgeBatch(
            **{item.name: getattr(self, item.name).to(device) for item in fields(self)}
        )


@dataclass(frozen=True)
class PrefixBatchV4:
    """Ragged prefix rows packed as ``R x L`` and ``R x C`` tensors."""

    state_index: torch.Tensor
    prefix: FactorEdgeBatch
    children: FactorEdgeBatch
    legal_mask: torch.Tensor
    opened_mask: torch.Tensor
    visit_counts: torch.Tensor
    prefix_mass: torch.Tensor
    complete_action_visits: torch.Tensor
    complete_action_mask: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.state_index.numel())

    def to(self, device: torch.device | str) -> "PrefixBatchV4":
        return PrefixBatchV4(
            state_index=self.state_index.to(device),
            prefix=self.prefix.to(device),
            children=self.children.to(device),
            legal_mask=self.legal_mask.to(device),
            opened_mask=self.opened_mask.to(device),
            visit_counts=self.visit_counts.to(device),
            prefix_mass=self.prefix_mass.to(device),
            complete_action_visits=self.complete_action_visits.to(device),
            complete_action_mask=self.complete_action_mask.to(device),
        )


@dataclass(frozen=True)
class V4Batch:
    states: V3Batch
    prefixes: PrefixBatchV4

    @property
    def batch_size(self) -> int:
        return self.states.batch_size

    def to(self, device: torch.device | str) -> "V4Batch":
        return V4Batch(self.states.to(device), self.prefixes.to(device))


@dataclass(frozen=True)
class _PrefixRow:
    state_index: int
    prefix: Path
    children: tuple[Token, ...]
    counts: tuple[int, ...]
    opened: tuple[bool, ...]
    mass: float


class _Trie:
    __slots__ = ("children", "action_index")

    def __init__(self) -> None:
        self.children: dict[Token, _Trie] = {}
        self.action_index: int | None = None


def collate_factorized_batch(
    states: V3Batch,
    action_sets: Sequence[Sequence[Mapping[str, Any]]],
    complete_action_visits: Sequence[Sequence[int]],
    *,
    opened_action_masks: Sequence[Sequence[bool]] | None = None,
    device: torch.device | str | None = None,
) -> V4Batch:
    """Collate complete V3 actions and visits into conditional prefix rows.

    By default every legal action is treated as opened. A caller with explicit
    search opening information can provide ``opened_action_masks``; an opened
    zero-visit child remains in the conditional softmax, while a closed child does
    not become a false zero target.
    """
    batch_size = states.batch_size
    if batch_size == 0 or len(action_sets) != batch_size or len(complete_action_visits) != batch_size:
        raise FactorizedPolicyError("state, action, and visit batches must be non-empty and aligned")
    if opened_action_masks is not None and len(opened_action_masks) != batch_size:
        raise FactorizedPolicyError("opened action masks must align with the state batch")

    rows: list[_PrefixRow] = []
    max_actions = max(len(actions) for actions in action_sets)
    if max_actions == 0:
        raise FactorizedPolicyError("each factorized decision requires a legal action")
    complete_visits = torch.zeros((batch_size, max_actions), dtype=torch.long)
    complete_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)

    for state_index, (actions, visits) in enumerate(zip(action_sets, complete_action_visits)):
        if not actions or len(actions) != len(visits):
            raise FactorizedPolicyError("actions and visits must be non-empty and aligned")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in visits):
            raise FactorizedPolicyError("visits must be non-negative integers")
        opened = (
            tuple(True for _ in actions)
            if opened_action_masks is None
            else tuple(opened_action_masks[state_index])
        )
        if len(opened) != len(actions) or any(not isinstance(value, bool) for value in opened):
            raise FactorizedPolicyError("opened action masks must be boolean and action-aligned")
        if any(count > 0 and not is_open for count, is_open in zip(visits, opened)):
            raise FactorizedPolicyError("a positive-visit action cannot be closed")
        root_total = sum(visits)
        if root_total <= 0:
            raise FactorizedPolicyError("each decision must contain at least one visit")

        complete_visits[state_index, : len(visits)] = torch.tensor(visits, dtype=torch.long)
        complete_mask[state_index, : len(visits)] = True
        trie = _build_trie(actions)

        def descend(node: _Trie, prefix: Path) -> tuple[int, bool]:
            if not node.children:
                if node.action_index is None:
                    raise FactorizedPolicyError("trie node has no child or complete action")
                index = node.action_index
                return visits[index], opened[index]
            child_data: list[tuple[Token, int, bool]] = []
            for token, child in sorted(node.children.items(), key=lambda pair: _wire_token(pair[0])):
                count, is_open = descend(child, (*prefix, token))
                child_data.append((token, count, is_open))
            subtotal = sum(item[1] for item in child_data)
            rows.append(
                _PrefixRow(
                    state_index,
                    prefix,
                    tuple(item[0] for item in child_data),
                    tuple(item[1] for item in child_data),
                    tuple(item[2] for item in child_data),
                    subtotal / root_total,
                )
            )
            return subtotal, any(item[2] for item in child_data)

        conserved, _ = descend(trie, ())
        if conserved != root_total:
            raise FactorizedPolicyError("root visits do not equal complete-action visits")

    # The traversal above is post-order, matching V3 aggregate_prefix_visits and its
    # serialized target order. Row identities do not affect the loss.
    max_prefix = max(1, max(len(row.prefix) for row in rows))
    max_children = max(len(row.children) for row in rows)
    prefix_edges = _allocate_edges(len(rows), max_prefix)
    child_edges = _allocate_edges(len(rows), max_children)
    state_index_tensor = torch.empty(len(rows), dtype=torch.long)
    legal_mask = torch.zeros((len(rows), max_children), dtype=torch.bool)
    opened_mask = torch.zeros_like(legal_mask)
    visit_counts = torch.zeros((len(rows), max_children), dtype=torch.long)
    prefix_mass = torch.empty(len(rows), dtype=torch.float32)

    for row_index, row in enumerate(rows):
        state_index_tensor[row_index] = row.state_index
        prefix_mass[row_index] = row.mass
        _write_edges(prefix_edges, row_index, row.prefix)
        _write_edges(child_edges, row_index, row.children)
        legal_mask[row_index, : len(row.children)] = True
        opened_mask[row_index, : len(row.children)] = torch.tensor(row.opened)
        visit_counts[row_index, : len(row.children)] = torch.tensor(row.counts, dtype=torch.long)

    prefix_batch = PrefixBatchV4(
        state_index_tensor,
        FactorEdgeBatch(**prefix_edges),
        FactorEdgeBatch(**child_edges),
        legal_mask,
        opened_mask,
        visit_counts,
        prefix_mass,
        complete_visits,
        complete_mask,
    )
    assert_visit_conservation(prefix_batch)
    result = V4Batch(states, prefix_batch)
    return result if device is None else result.to(device)


def tensorize_factorized_cases(
    cases: Sequence[Mapping[str, Any]],
    vocabulary: V3Vocabulary,
    complete_action_visits: Sequence[Sequence[int]],
    *,
    opened_action_masks: Sequence[Sequence[bool]] | None = None,
    device: torch.device | str | None = None,
) -> V4Batch:
    """Tensorize serialized V3 observation/action cases and their V4 visits."""
    states = tensorize_cases(cases, vocabulary)
    actions = [case["legal_actions"]["actions"] for case in cases]
    return collate_factorized_batch(
        states,
        actions,
        complete_action_visits,
        opened_action_masks=opened_action_masks,
        device=device,
    )


def assert_visit_conservation(batch: PrefixBatchV4) -> None:
    """Raise when any prefix or root violates descendant-visit conservation."""
    if batch.visit_counts.shape != batch.legal_mask.shape or batch.opened_mask.shape != batch.legal_mask.shape:
        raise FactorizedPolicyError("prefix child tensors must have the same shape")
    if torch.any(batch.visit_counts < 0) or torch.any((batch.visit_counts > 0) & ~batch.opened_mask):
        raise FactorizedPolicyError("positive visits require an opened legal child")
    if torch.any(batch.opened_mask & ~batch.legal_mask):
        raise FactorizedPolicyError("opened children must be legal")
    root_visits = (batch.complete_action_visits * batch.complete_action_mask).sum(dim=1)
    for decision in range(batch.complete_action_visits.shape[0]):
        indexes = torch.nonzero(batch.state_index == decision, as_tuple=False).flatten()
        roots = [int(index) for index in indexes if not bool(batch.prefix.mask[index].any())]
        if len(roots) != 1:
            raise FactorizedPolicyError("each decision must have exactly one root prefix")
        root_count = batch.visit_counts[roots[0]][batch.legal_mask[roots[0]]].sum()
        if not torch.equal(root_count, root_visits[decision]):
            raise FactorizedPolicyError("root visits do not equal complete-action visits")
    row_totals = (batch.visit_counts * batch.legal_mask).sum(dim=1)
    expected_mass = row_totals / root_visits[batch.state_index]
    if not torch.allclose(expected_mass, batch.prefix_mass, atol=1e-7, rtol=1e-7):
        raise FactorizedPolicyError("prefix mass does not equal descendant/root visits")

    # Every internal non-root prefix must conserve the visit count on the edge
    # from its parent. This catches corruption away from the root as well.
    row_by_prefix: dict[tuple[int, tuple[tuple[int, int, int, int], ...]], int] = {}
    for row in range(batch.row_count):
        key = (int(batch.state_index[row]), _edge_row_key(batch.prefix, row))
        if key in row_by_prefix:
            raise FactorizedPolicyError("duplicate prefix row")
        row_by_prefix[key] = row
    for (decision, prefix), row in row_by_prefix.items():
        if not prefix:
            continue
        parent = row_by_prefix.get((decision, prefix[:-1]))
        if parent is None:
            raise FactorizedPolicyError("prefix row is missing its parent")
        child_key = prefix[-1]
        matches = [
            column
            for column, key in enumerate(_edge_row_key(batch.children, parent))
            if key == child_key
        ]
        if len(matches) != 1:
            raise FactorizedPolicyError("parent does not contain the prefix edge exactly once")
        column = matches[0]
        if not torch.equal(batch.visit_counts[parent, column], row_totals[row]):
            raise FactorizedPolicyError("parent/child prefix visits are not conserved")
        if bool(batch.opened_mask[parent, column]) != bool(batch.opened_mask[row].any()):
            raise FactorizedPolicyError("parent/child opened masks are inconsistent")


def _edge_row_key(edges: FactorEdgeBatch, row: int) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (
            int(edges.field[row, column]),
            int(edges.category[row, column]),
            int(edges.index[row, column]),
            int(edges.ordinal[row, column]),
        )
        for column in range(edges.field.shape[1])
        if bool(edges.mask[row, column])
    )


def _build_trie(actions: Sequence[Mapping[str, Any]]) -> _Trie:
    root = _Trie()
    for action_index, action in enumerate(actions):
        node = root
        for token in factorized_action_path(action):
            if node.action_index is not None:
                raise FactorizedPolicyError("a complete action is a prefix of another action")
            node = node.children.setdefault(token, _Trie())
        if node.children or node.action_index is not None:
            raise FactorizedPolicyError("complete actions must have unique canonical paths")
        node.action_index = action_index
    return root


def _wire_token(token: Token) -> str:
    return f"{token[0]}:{token[1]}"


def _allocate_edges(rows: int, width: int) -> dict[str, torch.Tensor]:
    return {
        "field": torch.zeros((rows, width), dtype=torch.long),
        "category": torch.zeros((rows, width), dtype=torch.long),
        "index": torch.full((rows, width), -1, dtype=torch.long),
        "ordinal": torch.zeros((rows, width), dtype=torch.long),
        "mask": torch.zeros((rows, width), dtype=torch.bool),
    }


def _write_edges(tensors: dict[str, torch.Tensor], row: int, tokens: Sequence[Token]) -> None:
    for column, token in enumerate(tokens):
        field, category, index, ordinal = encode_factor_edge(token)
        tensors["field"][row, column] = field
        tensors["category"][row, column] = category
        tensors["index"][row, column] = index
        tensors["ordinal"][row, column] = ordinal
        tensors["mask"][row, column] = True


def encode_factor_edge(token: Token) -> tuple[int, int, int, int]:
    """Encode one typed grammar edge as field/category/index/ordinal integers."""
    field_name, value = token
    ordinal = 0
    if field_name.startswith("target_") and field_name[7:].isdigit():
        ordinal = int(field_name[7:]) + 1
        field_name = "target_sequence"
    try:
        field = FACTOR_FIELD_IDS[field_name]
    except KeyError as error:
        raise FactorizedPolicyError(f"unknown factor field {field_name!r}") from error

    category = 0
    index = -1
    if field_name == "family":
        category = _enum(ACTION_KIND_IDS, value, field_name)
    elif field_name == "ability":
        category = _enum(ABILITY_IDS, value, field_name)
    elif field_name == "target_kind":
        category = _enum(TARGET_KIND_IDS, value, field_name)
    elif field_name == "complete":
        if value != "action":
            raise FactorizedPolicyError("complete edge must be complete:action")
        category = 1
    elif field_name == "promotion" and value == "none":
        category = 1
    elif isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        index = value
    else:
        raise FactorizedPolicyError(f"invalid {field_name} factor value {value!r}")
    return field, category, index, ordinal


def _enum(values: Mapping[str, int], value: str | int, field: str) -> int:
    try:
        return values[value]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise FactorizedPolicyError(f"unknown {field} factor value {value!r}") from error


# Clear aliases for callers that prefer the version in the function name.
collate_v4_batch = collate_factorized_batch

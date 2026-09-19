"""Canonical V4 factorized-policy grammar derived from complete V3 actions.

This module deliberately operates on serialized ActionDescriptorV3 mappings. Rust must
implement the same grammar before V4 records are accepted by the engine runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

FACTOR_POLICY_SCHEMA_ID = "augmentia.factorized-policy.v4"
Token = tuple[str, str | int]
Path = tuple[Token, ...]


class FactorizedPolicyError(ValueError):
    """A legal-action set cannot be represented by the canonical V4 grammar."""


def factorized_action_path(action: Mapping[str, Any]) -> Path:
    """Return the one typed V4 path for a serialized ActionDescriptorV3."""
    kind = action.get("kind")
    if kind == "move":
        return (
            ("family", "move"),
            ("piece", _integer(action, "piece")),
            ("destination", _integer(action, "to")),
            ("promotion", _optional_integer(action, "promotion")),
            ("complete", "action"),
        )
    if kind == "piece_ability":
        return (
            ("family", "piece_ability"),
            ("piece", _integer(action, "piece")),
            ("ability", _string(action, "ability")),
            *_target_path(action.get("target")),
            ("complete", "action"),
        )
    if kind == "card":
        return (
            ("family", "card"),
            ("card", _integer(action, "card")),
            *_target_path(action.get("target")),
            ("complete", "action"),
        )
    if kind == "draft_pick":
        return (
            ("family", "draft_pick"),
            ("offer", _integer(action, "offer_index")),
            ("complete", "action"),
        )
    if kind == "resolve_choice":
        return (
            ("family", "resolve_choice"),
            ("choice", _integer(action, "choice_slot")),
            ("option", _integer(action, "option_index")),
            ("complete", "action"),
        )
    raise FactorizedPolicyError(f"unknown ActionDescriptorV3 kind {kind!r}")


def factorized_action_wire_path(action: Mapping[str, Any]) -> tuple[str, ...]:
    """Stable serialized form shared with the Rust V4 grammar."""
    return tuple(f"{field}:{value}" for field, value in factorized_action_path(action))


def _target_path(target: Any) -> Path:
    if not isinstance(target, Mapping):
        raise FactorizedPolicyError("action target must be an object")
    kind = target.get("kind")
    if kind == "none":
        return (("target_kind", "none"),)
    if kind in {"square", "piece_at"}:
        return (("target_kind", kind), ("target_square", _integer(target, "square")))
    if kind == "squares":
        squares = target.get("squares")
        if not isinstance(squares, list) or not squares:
            raise FactorizedPolicyError("squares target must contain at least one square")
        path: list[Token] = [("target_kind", "squares")]
        for position, square in enumerate(squares):
            if not isinstance(square, int) or isinstance(square, bool) or not 0 <= square < 64:
                raise FactorizedPolicyError("target square must be in 0..63")
            path.append((f"target_{position}", square))
        return tuple(path)
    raise FactorizedPolicyError(f"unknown action target kind {kind!r}")


def _integer(value: Mapping[str, Any], field_name: str) -> int:
    result = value.get(field_name)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise FactorizedPolicyError(f"{field_name} must be a non-negative integer")
    return result


def _optional_integer(value: Mapping[str, Any], field_name: str) -> str | int:
    result = value.get(field_name)
    if result is None:
        return "none"
    return _integer(value, field_name)


def _string(value: Mapping[str, Any], field_name: str) -> str:
    result = value.get(field_name)
    if not isinstance(result, str) or not result:
        raise FactorizedPolicyError(f"{field_name} must be a non-empty string")
    return result


@dataclass
class FactorizedTrieNode:
    children: dict[Token, "FactorizedTrieNode"] = field(default_factory=dict)
    action_index: int | None = None


@dataclass(frozen=True)
class FactorizedPrefixTarget:
    prefix: Path
    children: tuple[Token, ...]
    visit_counts: tuple[int, ...]


def build_factorized_trie(actions: Sequence[Mapping[str, Any]]) -> FactorizedTrieNode:
    """Build a trie and reject duplicate complete paths or prefix collisions."""
    root = FactorizedTrieNode()
    for action_index, action in enumerate(actions):
        node = root
        for token in factorized_action_path(action):
            if node.action_index is not None:
                raise FactorizedPolicyError("a complete action is a prefix of another action")
            node = node.children.setdefault(token, FactorizedTrieNode())
        if node.children or node.action_index is not None:
            raise FactorizedPolicyError("complete actions must have unique canonical paths")
        node.action_index = action_index
    return root


def aggregate_prefix_visits(
    actions: Sequence[Mapping[str, Any]], visits: Sequence[int]
) -> tuple[FactorizedPrefixTarget, ...]:
    """Aggregate complete-action MCTS visits into full legal distributions per prefix."""
    if len(actions) != len(visits) or not actions:
        raise FactorizedPolicyError("actions and visits must be non-empty and aligned")
    if any(not isinstance(visit, int) or isinstance(visit, bool) or visit < 0 for visit in visits):
        raise FactorizedPolicyError("visits must be non-negative integers")
    root = build_factorized_trie(actions)
    result: list[FactorizedPrefixTarget] = []

    def descend(node: FactorizedTrieNode, prefix: Path) -> int:
        child_counts = []
        for token, child in sorted(
            node.children.items(), key=lambda item: f"{item[0][0]}:{item[0][1]}"
        ):
            child_counts.append((token, descend(child, (*prefix, token))))
        if child_counts:
            result.append(
                FactorizedPrefixTarget(
                    prefix,
                    tuple(token for token, _ in child_counts),
                    tuple(count for _, count in child_counts),
                )
            )
            return sum(count for _, count in child_counts)
        if node.action_index is None:
            raise FactorizedPolicyError("trie node has neither children nor an action")
        return visits[node.action_index]

    descend(root, ())
    return tuple(result)

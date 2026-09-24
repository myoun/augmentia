"""Strict reader and validator for factorized V4 self-play records.

The V4 replay contract is intentionally versioned independently from the older
V2 target, whose missing opening metadata was previously treated as all-open.
A production record must carry the complete-action opening
mask as well as the opening mask for every factorized child.  The masks are
propagated to :class:`FactorizedTrainingExampleV4`; the trainer must never infer
an all-open mask from a missing field.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from v3.factorized import (
    FACTOR_POLICY_SCHEMA_ID,
    FactorizedPrefixTarget,
    Path as FactorizedPath,
    Token,
    aggregate_prefix_visits,
    factorized_action_path,
)
from v3.schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID, SchemaError, validate_golden_suite

# Current production lineage.  Keep the old names available as explicit,
# inspectable legacy constants; they are never accepted by the default parser.
GAME_RECORD_V5_SCHEMA_ID = "augmentia.replay.game.v5"
FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID = "augmentia.factorized-mcts-visits.v3"
# Backwards-compatible aliases used by callers that imported the original names.
GAME_RECORD_V4_SCHEMA_ID = GAME_RECORD_V5_SCHEMA_ID
FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID = FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID

LEGACY_GAME_RECORD_V4_SCHEMA_ID = "augmentia.replay.game.v4"
LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID = "augmentia.factorized-mcts-visits.v2"
RULESET_ID_V4 = "augment_chess_20260915"


@dataclass(frozen=True)
class FactorizedTrainingExampleV4:
    """One validated decision and the exact masks used by its loss.

    ``opened_action_mask`` is aligned with ``legal_actions['actions']``.
    ``opened_masks`` is aligned with ``prefix_targets`` and each item is aligned
    with that prefix's ``children`` tuple.  Keeping the two masks separate is
    intentional: the root mask is complete-action metadata, while the prefix
    masks are the conditional rows consumed by the factorized scorer.
    """

    decision_id: str
    observation: Mapping[str, Any]
    legal_actions: Mapping[str, Any]
    complete_visit_counts: tuple[int, ...]
    prefix_targets: tuple[FactorizedPrefixTarget, ...]
    value: float | None
    # These fields have defaults solely to keep the dataclass pleasant to use in
    # small synthetic tests.  Production parsing and V4Trainer reject empty masks.
    opened_action_mask: tuple[bool, ...] = ()
    opened_masks: tuple[tuple[bool, ...], ...] = ()
    target_schema_id: str = FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID
    replay_schema_id: str = GAME_RECORD_V5_SCHEMA_ID
    legacy: bool = False
    synthetic: bool = False



class _TrieNode:
    def __init__(self) -> None:
        self.children: dict[Token, _TrieNode] = {}
        self.action_index: int | None = None


def load_game_v4(
    path: str | Path, *, allow_legacy: bool = False
) -> tuple[FactorizedTrainingExampleV4, ...]:
    source = Path(path)
    record = json.loads(source.read_text(encoding="utf-8"))
    return parse_game_v4(record, str(source), allow_legacy=allow_legacy)


def parse_game_v4(
    record: Any,
    path: str = "record",
    *,
    allow_legacy: bool = False,
) -> tuple[FactorizedTrainingExampleV4, ...]:
    """Parse one V5 replay record.

    Legacy V4/V2 records are accepted only when ``allow_legacy`` is explicitly
    true. They are tagged on every returned example so a trainer can reject a
    mixed batch instead of silently changing its semantics.
    """

    if not isinstance(record, dict):
        raise SchemaError(f"{path}: V4 game must be an object")
    record_schema = record.get("record_schema_id")
    if record_schema == GAME_RECORD_V5_SCHEMA_ID:
        if allow_legacy:
            # A v5 record is always strict; an explicitly enabled legacy mode
            # must not weaken production validation.
            allow_legacy = False
        target_lineage = record.get("factorized_target_schema_id")
        if target_lineage != FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID:
            raise SchemaError(
                f"{path}: expected top-level factorized target schema "
                f"{FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID}"
            )
        replay_lineage = GAME_RECORD_V5_SCHEMA_ID
        is_legacy = False
    elif record_schema == LEGACY_GAME_RECORD_V4_SCHEMA_ID:
        if not allow_legacy:
            raise SchemaError(
                f"{path}: legacy replay schema {LEGACY_GAME_RECORD_V4_SCHEMA_ID} is "
                "not accepted by the production V4 loader; pass allow_legacy=True "
                "for an explicitly quarantined legacy analysis"
            )
        target_lineage = record.get(
            "factorized_target_schema_id", LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID
        )
        if target_lineage != LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID:
            raise SchemaError(f"{path}: legacy target lineage mismatch")
        replay_lineage = LEGACY_GAME_RECORD_V4_SCHEMA_ID
        is_legacy = True
    else:
        expected = GAME_RECORD_V5_SCHEMA_ID
        raise SchemaError(f"{path}: expected {expected}")

    if record.get("observation_schema_id") != OBSERVATION_SCHEMA_ID:
        raise SchemaError(f"{path}: observation schema mismatch")
    if record.get("action_schema_id") != ACTION_SCHEMA_ID:
        raise SchemaError(f"{path}: action schema mismatch")
    if record.get("factorized_policy_schema_id") != FACTOR_POLICY_SCHEMA_ID:
        raise SchemaError(f"{path}: factorized policy schema mismatch")
    game_id = record.get("game_id")
    steps = record.get("steps")
    if not isinstance(game_id, str) or not game_id or not isinstance(steps, list) or not steps:
        raise SchemaError(f"{path}: V4 game must have an ID and steps")
    ruleset = record.get("ruleset", {})
    if not isinstance(ruleset, dict) or ruleset.get("id") != RULESET_ID_V4:
        raise SchemaError(f"{path}: V4 replay must use {RULESET_ID_V4}")
    result = _terminal_result(record.get("termination"))
    examples: list[FactorizedTrainingExampleV4] = []
    for index, step in enumerate(steps):
        step_path = f"{path}.steps[{index}]"
        if not isinstance(step, dict) or step.get("decision_index") != index:
            raise SchemaError(f"{step_path}: decision index must be contiguous")
        case = {
            "observation": step.get("observation"),
            "legal_actions": step.get("legal_actions"),
        }
        validate_golden_suite(
            {
                "fixture_schema_id": "augmentia.golden.observation-action.v3",
                "observation_schema_id": OBSERVATION_SCHEMA_ID,
                "action_schema_id": ACTION_SCHEMA_ID,
                "ruleset_id": RULESET_ID_V4,
                "cases": [{"name": f"{game_id}:{index}", **case}],
            },
            RULESET_ID_V4,
        )
        actions = case["legal_actions"]["actions"]
        visits, prefix_targets = validate_factorized_target_v4(
            step.get("policy_target"),
            actions,
            step_path,
            allow_legacy=is_legacy,
        )
        target_record = step["policy_target"]
        prefix_opened_masks = (
            tuple(tuple(item["opened_masks"]) for item in target_record["prefix_targets"])
            if not is_legacy
            else ()
        )
        opened_action_mask = (
            tuple(bool(item) for item in target_record.get("opened_action_mask", ()))
            if not is_legacy
            else ()
        )
        played = step.get("played_action_index")
        if (
            not isinstance(played, int)
            or isinstance(played, bool)
            or not 0 <= played < len(actions)
            or visits[played] <= 0
            or (not is_legacy and not opened_action_mask[played])
        ):
            raise SchemaError(f"{step_path}: played action must have positive opened visits")
        actor = step.get("actor")
        value = None
        if result == "draw":
            value = 0.0
        elif result in {"white_win", "black_win"} and actor in {"white", "black"}:
            value = 1.0 if result.removesuffix("_win") == actor else -1.0
        examples.append(
            FactorizedTrainingExampleV4(
                decision_id=f"{game_id}:{index}",
                observation=case["observation"],
                legal_actions=case["legal_actions"],
                complete_visit_counts=visits,
                prefix_targets=prefix_targets,
                value=value,
                opened_action_mask=opened_action_mask,
                opened_masks=prefix_opened_masks,
                target_schema_id=target_lineage,
                replay_schema_id=replay_lineage,
                legacy=is_legacy,
            )
        )
    return tuple(examples)


def validate_factorized_target_v4(
    target: Any,
    actions: Sequence[Mapping[str, Any]],
    path: str = "target",
    *,
    allow_legacy: bool = False,
) -> tuple[tuple[int, ...], tuple[FactorizedPrefixTarget, ...]]:
    """Validate a V3 target and return visits plus prefix targets.

    The return shape is retained for existing callers. Prefix masks are strictly
    validated against the action-aligned root mask and preserved on parsed
    :class:`FactorizedTrainingExampleV4` instances.
    """

    if not isinstance(target, dict):
        raise SchemaError(f"{path}: factorized target must be an object")
    schema_id = target.get("schema_id")
    if schema_id == LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID:
        if not allow_legacy:
            raise SchemaError(
                f"{path}: legacy target schema is not accepted; pass allow_legacy=True "
                "only for a quarantined legacy analysis"
            )
        return _validate_legacy_target(target, actions, path)
    if schema_id != FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID:
        raise SchemaError(
            f"{path}: factorized target schema mismatch; expected "
            f"{FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID}"
        )
    if target.get("factorized_policy_schema_id") != FACTOR_POLICY_SCHEMA_ID:
        raise SchemaError(f"{path}: factorized policy schema mismatch")
    _require_exact_keys(
        target,
        {
            "schema_id",
            "factorized_policy_schema_id",
            "opened_action_mask",
            "complete_visit_counts",
            "completed_simulations",
            "root_value",
            "prefix_targets",
            "trie_sha256",
            "search_semantics_id",
            "root_selection_id",
            "chance_sampling_id",
            "chance_outcome_samples",
        },
        path,
    )
    opened_action_mask = _validate_bool_vector(
        target.get("opened_action_mask"), len(actions), f"{path}.opened_action_mask"
    )
    visits = _validate_visits(target.get("complete_visit_counts"), len(actions), path)
    simulations = target.get("completed_simulations")
    if (
        not isinstance(simulations, int)
        or isinstance(simulations, bool)
        or simulations <= 0
        or sum(visits) != simulations
    ):
        raise SchemaError(f"{path}: complete visits must sum to completed simulations")
    if not any(opened_action_mask):
        raise SchemaError(f"{path}: at least one complete action must be opened")
    if any(count > 0 and not is_open for count, is_open in zip(visits, opened_action_mask)):
        raise SchemaError(f"{path}: a positive-visit action must be opened")
    root_value = target.get("root_value")
    if (
        not isinstance(root_value, (int, float))
        or isinstance(root_value, bool)
        or not math.isfinite(root_value)
        or not -1.0001 <= root_value <= 1.0001
    ):
        raise SchemaError(f"{path}: invalid root value")
    for field in ("search_semantics_id", "root_selection_id", "chance_sampling_id"):
        if not isinstance(target.get(field), str) or not target[field]:
            raise SchemaError(f"{path}: {field} must be non-empty")
    chance_samples = target.get("chance_outcome_samples")
    if (
        not isinstance(chance_samples, int)
        or isinstance(chance_samples, bool)
        or not 1 <= chance_samples <= 64
    ):
        raise SchemaError(f"{path}: chance_outcome_samples must be in 1..=64")

    prefix_wire, _ = _aggregate_prefix_wire(
        actions, visits, opened_action_mask, path
    )
    supplied_prefixes = target.get("prefix_targets")
    if not isinstance(supplied_prefixes, list):
        raise SchemaError(f"{path}: prefix_targets must be a list")
    for prefix_index, supplied_prefix in enumerate(supplied_prefixes):
        prefix_item_path = f"{path}.prefix_targets[{prefix_index}]"
        if not isinstance(supplied_prefix, dict):
            raise SchemaError(f"{prefix_item_path}: prefix target must be an object")
        _require_exact_keys(
            supplied_prefix,
            {"prefix", "children", "visit_counts", "opened_masks"},
            prefix_item_path,
        )
        children = supplied_prefix.get("children")
        prefix_values = supplied_prefix.get("prefix")
        counts = supplied_prefix.get("visit_counts")
        child_masks = supplied_prefix.get("opened_masks")
        if (
            not isinstance(prefix_values, list)
            or any(not isinstance(value, str) for value in prefix_values)
            or not isinstance(children, list)
            or any(not isinstance(value, str) for value in children)
            or not isinstance(counts, list)
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts)
            or len(counts) != len(children)
            or not isinstance(child_masks, list)
            or len(child_masks) != len(children)
            or any(type(value) is not bool for value in child_masks)
        ):
            raise SchemaError(f"{prefix_item_path}: malformed prefix arrays")
    if supplied_prefixes != prefix_wire:
        raise SchemaError(
            f"{path}: prefix targets/opened masks do not match complete visits and actions"
        )
    digest = _target_digest(opened_action_mask, prefix_wire)
    if target.get("trie_sha256") != digest:
        raise SchemaError(f"{path}: factorized trie checksum mismatch")
    # ``aggregate_prefix_visits`` remains the canonical source for the returned
    # V3 objects; the stricter wire check above adds the mask invariant.
    expected = aggregate_prefix_visits(actions, visits)
    return tuple(visits), expected


def make_factorized_target_v4(
    actions: Sequence[Mapping[str, Any]],
    visits: Sequence[int],
    root_value: float,
    opened_action_mask: Sequence[bool] | None = None,
    *,
    synthetic_all_open: bool = False,
    search_semantics_id: str = "augmentia.mcts-v4.factorized-gumbel-widening-public-chance.v2",
    root_selection_id: str = "augmentia.gumbel-sequential-halving.v1",
    chance_sampling_id: str = "augmentia.public-information-multisample-chance.v1",
    chance_outcome_samples: int = 8,
) -> dict[str, Any]:
    """Build a V3 target.

    ``opened_action_mask`` is required for production use.  Omitting it is
    rejected unless the caller explicitly sets ``synthetic_all_open=True`` for
    a narrow all-open fixture; production replay files and trainer inputs must
    provide the field.
    """

    if not isinstance(visits, (list, tuple)):
        raise SchemaError("target.complete_visit_counts must be a sequence")
    normalized_visits = _validate_visits(list(visits), len(actions), "target")
    if opened_action_mask is None:
        if not synthetic_all_open:
            raise SchemaError(
                "opened_action_mask is required; use synthetic_all_open=True only "
                "for an explicit all-open fixture"
            )
        normalized_opened = tuple(True for _ in actions)
    else:
        if not isinstance(opened_action_mask, (list, tuple)):
            raise SchemaError("target.opened_action_mask must be a boolean sequence")
        normalized_opened = _validate_bool_vector(
            list(opened_action_mask), len(actions), "target.opened_action_mask"
        )
    if not any(normalized_opened):
        raise SchemaError("target.opened_action_mask must open at least one action")
    if any(count > 0 and not is_open for count, is_open in zip(normalized_visits, normalized_opened)):
        raise SchemaError("a positive-visit action must be opened")
    prefix_wire, _ = _aggregate_prefix_wire(
        actions, normalized_visits, normalized_opened, "target"
    )
    return {
        "schema_id": FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
        "factorized_policy_schema_id": FACTOR_POLICY_SCHEMA_ID,
        "complete_visit_counts": list(normalized_visits),
        "opened_action_mask": list(normalized_opened),
        "completed_simulations": sum(normalized_visits),
        "root_value": root_value,
        "prefix_targets": prefix_wire,
        "trie_sha256": _target_digest(normalized_opened, prefix_wire),
        "search_semantics_id": search_semantics_id,
        "root_selection_id": root_selection_id,
        "chance_sampling_id": chance_sampling_id,
        "chance_outcome_samples": chance_outcome_samples,
    }


def validate_legacy_factorized_target_v4(
    target: Any,
    actions: Sequence[Mapping[str, Any]],
    path: str = "target",
) -> tuple[tuple[int, ...], tuple[FactorizedPrefixTarget, ...]]:
    """Explicitly validate a quarantined v2 target."""

    if not isinstance(target, dict) or target.get("schema_id") != LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID:
        raise SchemaError(f"{path}: expected legacy factorized target schema")
    return _validate_legacy_target(target, actions, path)


def make_legacy_factorized_target_v2(
    actions: Sequence[Mapping[str, Any]],
    visits: Sequence[int],
    root_value: float,
    *,
    search_semantics_id: str = "augmentia.mcts-v4.factorized.v1",
    root_selection_id: str = "augmentia.gumbel-sequential-halving.v1",
    chance_sampling_id: str = "augmentia.public-information-multisample-chance.v1",
    chance_outcome_samples: int = 8,
) -> dict[str, Any]:
    """Build an explicitly quarantined v2 target for migration experiments.

    This helper never emits a production lineage and is intentionally separate
    from :func:`make_factorized_target_v4`.
    """

    if not isinstance(visits, (list, tuple)):
        raise SchemaError("target.complete_visit_counts must be a sequence")
    normalized_visits = _validate_visits(list(visits), len(actions), "target")
    expected = aggregate_prefix_visits(actions, normalized_visits)
    wire = [_legacy_prefix_to_wire(item) for item in expected]
    return {
        "schema_id": LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
        "factorized_policy_schema_id": FACTOR_POLICY_SCHEMA_ID,
        "complete_visit_counts": list(normalized_visits),
        "completed_simulations": sum(normalized_visits),
        "root_value": root_value,
        "prefix_targets": wire,
        "trie_sha256": _legacy_prefix_digest(wire),
        "search_semantics_id": search_semantics_id,
        "root_selection_id": root_selection_id,
        "chance_sampling_id": chance_sampling_id,
        "chance_outcome_samples": chance_outcome_samples,
    }


# Explicit name for callers that use the replay-version naming convention.
make_legacy_factorized_target_v4 = make_legacy_factorized_target_v2


def _validate_legacy_target(
    target: Mapping[str, Any], actions: Sequence[Mapping[str, Any]], path: str
) -> tuple[tuple[int, ...], tuple[FactorizedPrefixTarget, ...]]:
    """Validate a quarantined v2 target without inventing opening metadata."""

    if target.get("factorized_policy_schema_id") != FACTOR_POLICY_SCHEMA_ID:
        raise SchemaError(f"{path}: factorized policy schema mismatch")
    visits = _validate_visits(target.get("complete_visit_counts"), len(actions), path)
    simulations = target.get("completed_simulations")
    if (
        not isinstance(simulations, int)
        or isinstance(simulations, bool)
        or simulations <= 0
        or sum(visits) != simulations
    ):
        raise SchemaError(f"{path}: complete visits must sum to completed simulations")
    root_value = target.get("root_value")
    if (
        not isinstance(root_value, (int, float))
        or isinstance(root_value, bool)
        or not math.isfinite(root_value)
        or not -1.0001 <= root_value <= 1.0001
    ):
        raise SchemaError(f"{path}: invalid root value")
    for field in ("search_semantics_id", "root_selection_id", "chance_sampling_id"):
        if not isinstance(target.get(field), str) or not target[field]:
            raise SchemaError(f"{path}: {field} must be non-empty")
    chance_samples = target.get("chance_outcome_samples")
    if (
        not isinstance(chance_samples, int)
        or isinstance(chance_samples, bool)
        or not 1 <= chance_samples <= 64
    ):
        raise SchemaError(f"{path}: chance_outcome_samples must be in 1..=64")
    expected = aggregate_prefix_visits(actions, visits)
    wire = [_legacy_prefix_to_wire(item) for item in expected]
    if target.get("prefix_targets") != wire:
        raise SchemaError(f"{path}: prefix targets do not match complete visits")
    if target.get("trie_sha256") != _legacy_prefix_digest(wire):
        raise SchemaError(f"{path}: factorized trie checksum mismatch")
    return tuple(visits), expected


def _aggregate_prefix_wire(
    actions: Sequence[Mapping[str, Any]],
    visits: Sequence[int],
    opened: Sequence[bool],
    path: str,
) -> tuple[list[dict[str, Any]], tuple[tuple[bool, ...], ...]]:
    """Aggregate visits and derive each child mask as descendant OR."""

    if not actions:
        raise SchemaError(f"{path}: at least one legal action is required")
    if len(actions) != len(visits) or len(actions) != len(opened):
        raise SchemaError(f"{path}: action, visit, and opened-mask lengths must align")
    root = _TrieNode()
    for action_index, action in enumerate(actions):
        node = root
        for token in factorized_action_path(action):
            if node.action_index is not None:
                raise SchemaError(f"{path}: a complete action is a prefix of another action")
            node = node.children.setdefault(token, _TrieNode())
        if node.children or node.action_index is not None:
            raise SchemaError(f"{path}: complete actions must have unique canonical paths")
        node.action_index = action_index

    wire: list[dict[str, Any]] = []
    masks: list[tuple[bool, ...]] = []

    def descend(node: _TrieNode, prefix: FactorizedPath) -> tuple[int, bool]:
        if not node.children:
            if node.action_index is None:
                raise SchemaError(f"{path}: trie node has no child or complete action")
            return visits[node.action_index], opened[node.action_index]
        children: list[tuple[Token, int, bool]] = []
        for token, child in sorted(node.children.items(), key=lambda pair: _wire_token(pair[0])):
            count, is_open = descend(child, (*prefix, token))
            children.append((token, count, is_open))
        child_masks = tuple(item[2] for item in children)
        wire.append(
            {
                "prefix": [f"{field}:{value}" for field, value in prefix],
                "children": [f"{token[0]}:{token[1]}" for token, _, _ in children],
                "visit_counts": [item[1] for item in children],
                "opened_masks": list(child_masks),
            }
        )
        masks.append(child_masks)
        return sum(item[1] for item in children), any(child_masks)

    descend(root, ())
    return wire, tuple(masks)


def _validate_visits(value: Any, expected_len: int, path: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != expected_len
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value)
    ):
        raise SchemaError(f"{path}.complete_visit_counts must be non-negative integers")
    return tuple(value)


def _validate_bool_vector(value: Any, expected_len: int, path: str) -> tuple[bool, ...]:
    if (
        not isinstance(value, list)
        or len(value) != expected_len
        or any(type(item) is not bool for item in value)
    ):
        raise SchemaError(f"{path} must be a boolean list of length {expected_len}")
    return tuple(value)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing {sorted(missing)}")
        if extra:
            detail.append(f"unexpected {sorted(extra)}")
        raise SchemaError(f"{path}: target keys {'; '.join(detail)}")


def _prefix_digest(wire: Sequence[Mapping[str, Any]]) -> str:
    """Legacy v2 digest helper (prefix targets only)."""

    return hashlib.sha256(
        json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _target_digest(
    opened_action_mask: Sequence[bool], prefix_wire: Sequence[Mapping[str, Any]]
) -> str:
    """Hash the Rust-compatible compact mask/prefix envelope."""

    envelope = {
        "opened_action_mask": list(opened_action_mask),
        "prefix_targets": list(prefix_wire),
    }
    return hashlib.sha256(
        json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _legacy_prefix_to_wire(target: FactorizedPrefixTarget) -> dict[str, Any]:
    return {
        "prefix": [f"{field}:{value}" for field, value in target.prefix],
        "children": [f"{field}:{value}" for field, value in target.children],
        "visit_counts": list(target.visit_counts),
    }


def _legacy_prefix_digest(wire: Sequence[Mapping[str, Any]]) -> str:
    return _prefix_digest(wire)


def _wire_token(token: Token) -> str:
    return f"{token[0]}:{token[1]}"


def _terminal_result(termination: Any) -> str | None:
    if isinstance(termination, dict) and termination.get("kind") == "engine_terminal":
        result = termination.get("result")
        if result in {"white_win", "black_win", "draw"}:
            return result
    return None

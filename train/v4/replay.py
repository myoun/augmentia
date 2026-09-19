"""Strict reader and validator for factorized V4 self-play records."""

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
    aggregate_prefix_visits,
)
from v3.schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID, SchemaError, validate_golden_suite

GAME_RECORD_V4_SCHEMA_ID = "augmentia.replay.game.v4"
FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID = "augmentia.factorized-mcts-visits.v2"
RULESET_ID_V4 = "augment_chess_20260915"


@dataclass(frozen=True)
class FactorizedTrainingExampleV4:
    decision_id: str
    observation: Mapping[str, Any]
    legal_actions: Mapping[str, Any]
    complete_visit_counts: tuple[int, ...]
    prefix_targets: tuple[FactorizedPrefixTarget, ...]
    value: float | None


def load_game_v4(path: str | Path) -> tuple[FactorizedTrainingExampleV4, ...]:
    source = Path(path)
    record = json.loads(source.read_text(encoding="utf-8"))
    return parse_game_v4(record, str(source))


def parse_game_v4(
    record: Any, path: str = "record"
) -> tuple[FactorizedTrainingExampleV4, ...]:
    if not isinstance(record, dict) or record.get("record_schema_id") != GAME_RECORD_V4_SCHEMA_ID:
        raise SchemaError(f"{path}: expected {GAME_RECORD_V4_SCHEMA_ID}")
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
    examples = []
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
        target = validate_factorized_target_v4(step.get("policy_target"), actions, step_path)
        played = step.get("played_action_index")
        if (
            not isinstance(played, int)
            or isinstance(played, bool)
            or not 0 <= played < len(actions)
            or target[0][played] <= 0
        ):
            raise SchemaError(f"{step_path}: played action must have positive visits")
        actor = step.get("actor")
        value = None
        if result == "draw":
            value = 0.0
        elif result in {"white_win", "black_win"} and actor in {"white", "black"}:
            value = 1.0 if result.removesuffix("_win") == actor else -1.0
        examples.append(
            FactorizedTrainingExampleV4(
                f"{game_id}:{index}",
                case["observation"],
                case["legal_actions"],
                target[0],
                target[1],
                value,
            )
        )
    return tuple(examples)


def validate_factorized_target_v4(
    target: Any, actions: Sequence[Mapping[str, Any]], path: str = "target"
) -> tuple[tuple[int, ...], tuple[FactorizedPrefixTarget, ...]]:
    if not isinstance(target, dict):
        raise SchemaError(f"{path}: factorized target must be an object")
    if target.get("schema_id") != FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID:
        raise SchemaError(f"{path}: factorized target schema mismatch")
    if target.get("factorized_policy_schema_id") != FACTOR_POLICY_SCHEMA_ID:
        raise SchemaError(f"{path}: factorized policy schema mismatch")
    visits = target.get("complete_visit_counts")
    simulations = target.get("completed_simulations")
    if (
        not isinstance(visits, list)
        or len(visits) != len(actions)
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in visits)
        or not isinstance(simulations, int)
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
    wire = [_prefix_to_wire(item) for item in expected]
    if target.get("prefix_targets") != wire:
        raise SchemaError(f"{path}: prefix targets do not match complete visits")
    digest = hashlib.sha256(
        json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if target.get("trie_sha256") != digest:
        raise SchemaError(f"{path}: factorized trie checksum mismatch")
    return tuple(visits), expected


def make_factorized_target_v4(
    actions: Sequence[Mapping[str, Any]],
    visits: Sequence[int],
    root_value: float,
    *,
    search_semantics_id: str = "augmentia.mcts-v4.factorized.v1",
    root_selection_id: str = "augmentia.gumbel-sequential-halving.v1",
    chance_sampling_id: str = "augmentia.public-information-multisample-chance.v1",
    chance_outcome_samples: int = 8,
) -> dict[str, Any]:
    prefix = aggregate_prefix_visits(actions, visits)
    wire = [_prefix_to_wire(item) for item in prefix]
    return {
        "schema_id": FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID,
        "factorized_policy_schema_id": FACTOR_POLICY_SCHEMA_ID,
        "complete_visit_counts": list(visits),
        "completed_simulations": sum(visits),
        "root_value": root_value,
        "prefix_targets": wire,
        "trie_sha256": hashlib.sha256(
            json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "search_semantics_id": search_semantics_id,
        "root_selection_id": root_selection_id,
        "chance_sampling_id": chance_sampling_id,
        "chance_outcome_samples": chance_outcome_samples,
    }


def _prefix_to_wire(target: FactorizedPrefixTarget) -> dict[str, Any]:
    return {
        "prefix": [f"{field}:{value}" for field, value in target.prefix],
        "children": [f"{field}:{value}" for field, value in target.children],
        "visit_counts": list(target.visit_counts),
    }


def _terminal_result(termination: Any) -> str | None:
    if isinstance(termination, dict) and termination.get("kind") == "engine_terminal":
        result = termination.get("result")
        if result in {"white_win", "black_win", "draw"}:
            return result
    return None

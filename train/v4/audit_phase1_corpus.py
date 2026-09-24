"""Audit legacy-to-v3 V4 corpus regeneration without mutating either corpus.

The audit pairs games by ``game_index``, verifies seed/config/trajectory identity,
strictly validates the current v5/v3 replay lineage, and quantifies how much
conditional softmax support the historical all-open view would incorrectly add.
Its JSON report is deterministic and self-hash sealed, so rerunning the same audit
is byte-idempotent while any changed input or existing report fails closed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .dataset import RUN_MANIFEST_SCHEMA_ID, VALIDATION_SCHEMA_ID
from .replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    FACTOR_POLICY_SCHEMA_ID,
    GAME_RECORD_V5_SCHEMA_ID,
    LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
    LEGACY_GAME_RECORD_V4_SCHEMA_ID,
    FactorizedTrainingExampleV4,
    parse_game_v4,
)

AUDIT_REPORT_SCHEMA_ID = "augmentia.v4-phase1-corpus-regeneration-audit.v1"
RULESET_ID = "augment_chess_20260915"
OBSERVATION_SCHEMA_ID = "augmentia.observation.v3.20260917.10"
ACTION_SCHEMA_ID = "augmentia.action.v3"

FALSE_ZERO_DEFINITION = (
    "closed_child_mass=sum(prefix_mass*closed_child_count) over prefix rows; "
    "weighted_closed_support_fraction=sum(prefix_mass*closed_child_count/"
    "legal_child_count); these quantify the conditional all-open normalizer "
    "support added by closed legal children, not model probability mass"
)
TRAJECTORY_HASH_DEFINITION = (
    "sha256(compact canonical JSON of an ordered list of "
    "{state_fingerprint,action,next_state_fingerprint} digest steps)"
)

PRODUCER_CONFIG_FIELDS = (
    "mode_id",
    "game_index_start",
    "games",
    "base_seed",
    "seed_derivation",
    "simulations",
    "chance_outcome_samples",
    "temperature",
    "temperature_decisions",
    "max_decisions",
    "workers",
    "execution_provider",
    "intra_threads",
    "search_semantics_id",
    "root_selection_id",
    "chance_sampling_id",
)


class CorpusAuditError(RuntimeError):
    """Raised when regeneration identity or corpus invariants are not sealed."""


@dataclass(frozen=True)
class GameArtifact:
    game_index: int
    game_id: str
    seed_hex: str
    relative_path: str
    file_sha256: str
    trajectory_sha256: str
    decisions: int
    termination: Mapping[str, Any]
    path: Path


@dataclass(frozen=True)
class CorpusManifest:
    label: str
    path: Path
    file_sha256: str
    payload: Mapping[str, Any]
    lineage: tuple[str, str]
    validation_path: Path
    validation_file_sha256: str
    validation: Mapping[str, Any]
    games: tuple[GameArtifact, ...]

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])


def canonical_json_bytes(value: Any, *, sort_keys: bool = True) -> bytes:
    """Serialize one audit value to the report's canonical JSON byte form."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusAuditError(f"cannot read {label} JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CorpusAuditError(f"{label} JSON must be an object: {path}")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CorpusAuditError(f"{label} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise CorpusAuditError(f"{label} must be hexadecimal") from error
    return value.lower()


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CorpusAuditError(f"{label} must be an integer >= {minimum}")
    return value


def _safe_relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CorpusAuditError(f"{label} must be a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise CorpusAuditError(f"unsafe {label}: {value}")
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise CorpusAuditError(f"{label} escapes its corpus root: {value}") from error
    return path


def _expected_seed_hex(base_seed: Any, game_index: int) -> str | None:
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        return None
    return f"{(base_seed + game_index) & ((1 << 64) - 1):016x}"


def _load_manifest(
    path: str | Path,
    *,
    label: str,
    expected_lineage: tuple[str, str],
) -> CorpusManifest:
    source = Path(path).resolve()
    payload = _load_json_object(source, f"{label} manifest")
    if payload.get("manifest_schema_id") != RUN_MANIFEST_SCHEMA_ID:
        raise CorpusAuditError(f"{label} manifest schema mismatch: {source}")
    lineage = (
        payload.get("replay_schema_id"),
        payload.get("factorized_target_schema_id"),
    )
    if lineage != expected_lineage:
        raise CorpusAuditError(
            f"{label} replay/target lineage mismatch: expected "
            f"{expected_lineage[0]}/{expected_lineage[1]}, got {lineage[0]}/{lineage[1]}"
        )
    if payload.get("factorized_policy_schema_id") != FACTOR_POLICY_SCHEMA_ID:
        raise CorpusAuditError(f"{label} factorized policy schema mismatch")
    if payload.get("observation_schema_id") != OBSERVATION_SCHEMA_ID:
        raise CorpusAuditError(f"{label} observation schema mismatch")
    if payload.get("action_schema_id") != ACTION_SCHEMA_ID:
        raise CorpusAuditError(f"{label} action schema mismatch")
    ruleset = payload.get("ruleset")
    if not isinstance(ruleset, dict) or ruleset.get("id") != RULESET_ID:
        raise CorpusAuditError(f"{label} must use {RULESET_ID}")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CorpusAuditError(f"{label} run_id must be non-empty")
    game_count = _require_int(payload.get("games"), f"{label}.games", minimum=1)
    start = _require_int(
        payload.get("game_index_start"),
        f"{label}.game_index_start",
        minimum=0,
    )
    raw_games = payload.get("games_artifacts")
    if not isinstance(raw_games, list) or len(raw_games) != game_count:
        raise CorpusAuditError(f"{label} games_artifacts count mismatch")
    expected_seed = _expected_seed_hex(payload.get("base_seed"), 0)

    games: list[GameArtifact] = []
    seen_indexes: set[int] = set()
    seen_seeds: set[str] = set()
    for position, raw in enumerate(raw_games):
        if not isinstance(raw, dict):
            raise CorpusAuditError(f"{label} game entry {position} must be an object")
        game_index = _require_int(
            raw.get("game_index"), f"{label}.games[{position}].game_index"
        )
        if game_index in seen_indexes:
            raise CorpusAuditError(f"{label} duplicate game_index {game_index}")
        seen_indexes.add(game_index)
        game_id = raw.get("game_id")
        seed_hex = raw.get("seed_hex")
        if not isinstance(game_id, str) or not game_id:
            raise CorpusAuditError(f"{label} game {game_index} has no game_id")
        if not isinstance(seed_hex, str) or len(seed_hex) != 16:
            raise CorpusAuditError(f"{label} game {game_index} seed_hex is invalid")
        seed_hex = seed_hex.lower()
        if seed_hex in seen_seeds:
            raise CorpusAuditError(f"{label} duplicate seed_hex {seed_hex}")
        seen_seeds.add(seed_hex)
        expected_game_seed = _expected_seed_hex(payload.get("base_seed"), game_index)
        if expected_game_seed is not None and seed_hex != expected_game_seed:
            raise CorpusAuditError(
                f"{label} game {game_index} seed derivation mismatch: "
                f"expected {expected_game_seed}, got {seed_hex}"
            )
        file_sha256 = _require_sha256(
            raw.get("file_sha256"), f"{label} game {game_index} file_sha256"
        )
        trajectory_sha256 = _require_sha256(
            raw.get("trajectory_sha256"),
            f"{label} game {game_index} trajectory_sha256",
        )
        decisions = _require_int(
            raw.get("decisions"), f"{label} game {game_index} decisions", minimum=1
        )
        termination = raw.get("termination")
        if not isinstance(termination, dict):
            raise CorpusAuditError(f"{label} game {game_index} termination is invalid")
        relative_path = raw.get("relative_path")
        game_path = _safe_relative_path(
            source.parent, relative_path, f"{label} game {game_index} path"
        )
        if not game_path.is_file():
            raise CorpusAuditError(f"{label} game file is missing: {game_path}")
        actual_file_sha256 = sha256_file(game_path)
        if actual_file_sha256 != file_sha256:
            raise CorpusAuditError(
                f"{label} game {game_index} file SHA-256 mismatch: "
                f"expected {file_sha256}, got {actual_file_sha256}"
            )
        games.append(
            GameArtifact(
                game_index=game_index,
                game_id=game_id,
                seed_hex=seed_hex,
                relative_path=str(relative_path),
                file_sha256=file_sha256,
                trajectory_sha256=trajectory_sha256,
                decisions=decisions,
                termination=dict(termination),
                path=game_path,
            )
        )
    expected_indexes = set(range(start, start + game_count))
    if seen_indexes != expected_indexes:
        raise CorpusAuditError(
            f"{label} game indexes must be contiguous {start}..{start + game_count - 1}"
        )
    if expected_seed is None or len(seen_seeds) != game_count:
        raise CorpusAuditError(f"{label} seed provenance is incomplete")
    games.sort(key=lambda game: game.game_index)

    validation_path = source.parent / "validation.json"
    if not validation_path.is_file():
        raise CorpusAuditError(f"{label} validation report is missing: {validation_path}")
    validation = _load_json_object(validation_path, f"{label} validation")
    if validation.get("validation_schema_id") != VALIDATION_SCHEMA_ID:
        raise CorpusAuditError(f"{label} validation schema mismatch")
    if validation.get("run_id") != run_id:
        raise CorpusAuditError(f"{label} validation run_id mismatch")
    if validation.get("games") != game_count:
        raise CorpusAuditError(f"{label} validation game count mismatch")
    decision_total = sum(game.decisions for game in games)
    if validation.get("decisions") != decision_total:
        raise CorpusAuditError(
            f"{label} validation decision count mismatch: "
            f"expected {decision_total}, got {validation.get('decisions')}"
        )
    if validation.get("unique_seeds") != game_count:
        raise CorpusAuditError(f"{label} validation unique seed count mismatch")
    terminal = _require_int(
        validation.get("terminal_games"), f"{label}.terminal_games"
    )
    truncated = _require_int(
        validation.get("truncated_games"), f"{label}.truncated_games"
    )
    if terminal + truncated != game_count:
        raise CorpusAuditError(f"{label} terminal/truncated accounting mismatch")

    return CorpusManifest(
        label=label,
        path=source,
        file_sha256=sha256_file(source),
        payload=payload,
        lineage=lineage,
        validation_path=validation_path,
        validation_file_sha256=sha256_file(validation_path),
        validation=validation,
        games=tuple(games),
    )


def _trajectory_digest(record: Mapping[str, Any], label: str) -> str:
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise CorpusAuditError(f"{label} trajectory has no steps")
    digest_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise CorpusAuditError(f"{label} step {index} must be an object")
        played = step.get("played_action_index")
        legal = step.get("legal_actions")
        if (
            not isinstance(played, int)
            or isinstance(played, bool)
            or not isinstance(legal, dict)
            or not isinstance(legal.get("actions"), list)
            or not 0 <= played < len(legal["actions"])
        ):
            raise CorpusAuditError(f"{label} step {index} played action is invalid")
        digest_steps.append(
            {
                "state_fingerprint": step.get("state_fingerprint"),
                "action": legal["actions"][played],
                "next_state_fingerprint": step.get("next_state_fingerprint"),
            }
        )
    return sha256_bytes(canonical_json_bytes(digest_steps, sort_keys=False))


def _validate_record_identity(
    corpus: CorpusManifest,
    game: GameArtifact,
    record: Mapping[str, Any],
) -> str:
    label = f"{corpus.label} game {game.game_index}"
    if record.get("game_id") != game.game_id:
        raise CorpusAuditError(f"{label} record game_id mismatch")
    if record.get("seed_hex") != game.seed_hex:
        raise CorpusAuditError(f"{label} record seed_hex mismatch")
    if record.get("mode_id") != corpus.payload.get("mode_id"):
        raise CorpusAuditError(f"{label} record mode_id mismatch")
    if record.get("ruleset") != corpus.payload.get("ruleset"):
        raise CorpusAuditError(f"{label} record ruleset mismatch")
    generator = record.get("generator")
    if not isinstance(generator, dict) or generator.get("run_id") != corpus.run_id:
        raise CorpusAuditError(f"{label} record generator run_id mismatch")
    if generator.get("engine_commit") != corpus.payload.get("engine_commit"):
        raise CorpusAuditError(f"{label} record engine_commit mismatch")
    steps = record.get("steps")
    if not isinstance(steps, list) or len(steps) != game.decisions:
        raise CorpusAuditError(
            f"{label} record decision count mismatch: "
            f"expected {game.decisions}, got {len(steps) if isinstance(steps, list) else None}"
        )
    if record.get("termination") != game.termination:
        raise CorpusAuditError(f"{label} record termination mismatch")
    record_trajectory = _require_sha256(
        record.get("trajectory_sha256"), f"{label} record trajectory_sha256"
    )
    if record_trajectory != game.trajectory_sha256:
        raise CorpusAuditError(
            f"{label} manifest/record trajectory hash mismatch: "
            f"{game.trajectory_sha256} != {record_trajectory}"
        )
    recomputed = _trajectory_digest(record, label)
    if recomputed != game.trajectory_sha256:
        raise CorpusAuditError(
            f"{label} trajectory content hash mismatch: "
            f"expected {game.trajectory_sha256}, got {recomputed}"
        )
    return recomputed


def _player_config_id(corpus: CorpusManifest) -> str:
    payload = corpus.payload
    return (
        "mcts-v4:"
        f"sims={payload.get('simulations')}:"
        f"chance={payload.get('chance_outcome_samples')}:"
        f"root={payload.get('root_selection_id')}:"
        f"search={payload.get('search_semantics_id')}"
    )


def _validate_players(corpus: CorpusManifest, record: Mapping[str, Any], game: GameArtifact) -> None:
    players = record.get("players")
    evaluator = corpus.payload.get("evaluator")
    if not isinstance(players, list) or len(players) != 2 or not isinstance(evaluator, dict):
        raise CorpusAuditError(
            f"{corpus.label} game {game.game_index} player provenance is invalid"
        )
    expected_agent = evaluator.get("agent_id")
    expected_config = _player_config_id(corpus)
    for player in players:
        if not isinstance(player, dict):
            raise CorpusAuditError(
                f"{corpus.label} game {game.game_index} player entry is invalid"
            )
        if player.get("agent_id") != expected_agent or player.get("config_id") != expected_config:
            raise CorpusAuditError(
                f"{corpus.label} game {game.game_index} player config mismatch"
            )


def _empty_metric_accumulator() -> dict[str, Any]:
    return {
        "games": 0,
        "decisions": 0,
        "legal_actions": 0,
        "opened_actions": 0,
        "closed_actions": 0,
        "positive_visit_actions": 0,
        "opened_positive_visit_actions": 0,
        "opened_zero_visit_actions": 0,
        "closed_zero_visit_actions": 0,
        "closed_positive_visit_actions": 0,
        "prefix_rows": 0,
        "visited_prefix_rows": 0,
        "mixed_prefix_rows": 0,
        "visited_mixed_prefix_rows": 0,
        "all_open_legal_child_slots": 0,
        "all_open_closed_child_slots": 0,
        "row_visits_total": 0,
        "visited_row_visits_total": 0,
        "prefix_mass_total": 0.0,
        "prefix_mass_min": None,
        "prefix_mass_max": None,
        "false_zero_closed_child_mass_total": 0.0,
        "false_zero_closed_child_mass_max_decision": 0.0,
        "false_zero_mass_total": 0.0,
        "false_zero_mass_max_decision": 0.0,
        "false_zero_affected_rows": 0,
        "false_zero_affected_decisions": 0,
        "decisions_with_mixed_rows": 0,
    }


def audit_opened_examples(
    examples: Sequence[FactorizedTrainingExampleV4],
    *,
    mixed_row_limit: int | None = 20,
) -> dict[str, Any]:
    """Aggregate opened-mask and all-open false-zero metrics for v3 examples."""

    metrics = _empty_metric_accumulator()
    mixed_rows: list[dict[str, Any]] = []
    for example in examples:
        if example.target_schema_id != FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID:
            raise CorpusAuditError("opened metrics require factorized target v3")
        if example.legacy or example.synthetic:
            raise CorpusAuditError("opened metrics cannot inspect legacy or synthetic examples")
        actions = example.legal_actions.get("actions")
        visits = example.complete_visit_counts
        opened = example.opened_action_mask
        if not isinstance(actions, list) or not actions:
            raise CorpusAuditError(f"decision {example.decision_id} has no legal actions")
        if len(visits) != len(actions) or len(opened) != len(actions):
            raise CorpusAuditError(f"decision {example.decision_id} action alignment mismatch")
        if len(example.opened_masks) != len(example.prefix_targets):
            raise CorpusAuditError(f"decision {example.decision_id} prefix mask alignment mismatch")
        root_visits = sum(visits)
        if root_visits <= 0:
            raise CorpusAuditError(f"decision {example.decision_id} has no visits")
        opened_count = sum(opened)
        closed_count = len(actions) - opened_count
        opened_positive = sum(
            count > 0 and is_open for count, is_open in zip(visits, opened)
        )
        opened_zero = sum(
            count == 0 and is_open for count, is_open in zip(visits, opened)
        )
        closed_zero = sum(
            count == 0 and not is_open for count, is_open in zip(visits, opened)
        )
        closed_positive = sum(
            count > 0 and not is_open for count, is_open in zip(visits, opened)
        )
        positive = sum(count > 0 for count in visits)
        if opened_positive != positive or closed_count < 0:
            raise CorpusAuditError(
                f"decision {example.decision_id} positive/open conservation mismatch"
            )

        decision_closed_child_mass = 0.0
        decision_false_zero = 0.0
        decision_has_mixed = False
        for prefix, child_masks in zip(
            example.prefix_targets, example.opened_masks, strict=True
        ):
            counts = prefix.visit_counts
            children = prefix.children
            if len(counts) != len(children) or len(child_masks) != len(children):
                raise CorpusAuditError(
                    f"decision {example.decision_id} prefix child alignment mismatch"
                )
            if any(count > 0 and not is_open for count, is_open in zip(counts, child_masks)):
                raise CorpusAuditError(
                    f"decision {example.decision_id} has a positive closed prefix child"
                )
            row_visits = sum(counts)
            prefix_mass = row_visits / root_visits
            row_opened = sum(child_masks)
            row_closed = len(child_masks) - row_opened
            mixed = row_opened > 0 and row_closed > 0
            closed_share = row_closed / len(child_masks)
            weighted_closed_child_mass = prefix_mass * row_closed
            weighted_false_zero = prefix_mass * closed_share
            decision_closed_child_mass += weighted_closed_child_mass
            decision_false_zero += weighted_false_zero

            metrics["prefix_rows"] += 1
            metrics["all_open_legal_child_slots"] += len(child_masks)
            metrics["all_open_closed_child_slots"] += row_closed
            metrics["row_visits_total"] += row_visits
            metrics["prefix_mass_total"] += prefix_mass
            if row_visits > 0:
                metrics["visited_prefix_rows"] += 1
                metrics["visited_row_visits_total"] += row_visits
            if mixed:
                metrics["mixed_prefix_rows"] += 1
                decision_has_mixed = True
                if row_visits > 0:
                    metrics["visited_mixed_prefix_rows"] += 1
                mixed_rows.append(
                    {
                        "decision_id": example.decision_id,
                        "prefix": [
                            f"{field}:{value}" for field, value in prefix.prefix
                        ],
                        "row_visits": row_visits,
                        "prefix_mass": prefix_mass,
                        "legal_child_count": len(child_masks),
                        "opened_child_count": row_opened,
                        "closed_child_count": row_closed,
                        "closed_support_fraction": closed_share,
                        "weighted_closed_child_count": weighted_closed_child_mass,
                        "weighted_closed_support_fraction": weighted_false_zero,
                    }
                )
            if row_closed > 0:
                metrics["false_zero_affected_rows"] += 1
                metrics["false_zero_closed_child_mass_total"] += (
                    weighted_closed_child_mass
                )
                if weighted_false_zero > 0.0:
                    metrics["false_zero_mass_total"] += weighted_false_zero
            if metrics["prefix_mass_min"] is None or prefix_mass < metrics[
                "prefix_mass_min"
            ]:
                metrics["prefix_mass_min"] = prefix_mass
            if metrics["prefix_mass_max"] is None or prefix_mass > metrics[
                "prefix_mass_max"
            ]:
                metrics["prefix_mass_max"] = prefix_mass

        metrics["decisions"] += 1
        metrics["legal_actions"] += len(actions)
        metrics["opened_actions"] += opened_count
        metrics["closed_actions"] += closed_count
        metrics["positive_visit_actions"] += positive
        metrics["opened_positive_visit_actions"] += opened_positive
        metrics["opened_zero_visit_actions"] += opened_zero
        metrics["closed_zero_visit_actions"] += closed_zero
        metrics["closed_positive_visit_actions"] += closed_positive
        metrics["false_zero_closed_child_mass_max_decision"] = max(
            metrics["false_zero_closed_child_mass_max_decision"],
            decision_closed_child_mass,
        )
        metrics["false_zero_mass_max_decision"] = max(
            metrics["false_zero_mass_max_decision"], decision_false_zero
        )
        if decision_false_zero > 0.0:
            metrics["false_zero_affected_decisions"] += 1
        if decision_has_mixed:
            metrics["decisions_with_mixed_rows"] += 1

    decisions = metrics["decisions"]
    if decisions <= 0:
        raise CorpusAuditError("opened metrics require at least one decision")
    metrics["false_zero_mass_mean_per_decision"] = (
        metrics["false_zero_mass_total"] / decisions
    )
    metrics["weighted_closed_support_fraction_total"] = metrics[
        "false_zero_mass_total"
    ]
    metrics["weighted_closed_support_fraction_mean_per_decision"] = metrics[
        "false_zero_mass_mean_per_decision"
    ]
    metrics["false_zero_closed_child_mass_mean_per_decision"] = (
        metrics["false_zero_closed_child_mass_total"] / decisions
    )
    metrics["false_zero_definition"] = FALSE_ZERO_DEFINITION
    total_mixed_rows = len(mixed_rows)
    if mixed_row_limit is not None:
        if not isinstance(mixed_row_limit, int) or isinstance(mixed_row_limit, bool) or mixed_row_limit < 0:
            raise CorpusAuditError("mixed_row_limit must be a non-negative integer or None")
        selected_mixed_rows = sorted(
            mixed_rows,
            key=lambda row: (
                -row["weighted_closed_child_count"],
                row["decision_id"],
                row["prefix"],
            ),
        )[:mixed_row_limit]
    else:
        selected_mixed_rows = mixed_rows
    metrics["mixed_rows_total"] = total_mixed_rows
    metrics["mixed_rows_reported"] = len(selected_mixed_rows)
    metrics["mixed_rows_omitted"] = total_mixed_rows - len(selected_mixed_rows)
    metrics["mixed_rows"] = selected_mixed_rows
    return metrics


def audit_legacy_examples(
    examples: Sequence[FactorizedTrainingExampleV4],
) -> dict[str, Any]:
    """Summarize the historical corpus without inventing opened metadata."""

    decisions = 0
    legal_actions = 0
    positive_visits = 0
    prefix_rows = 0
    visited_prefix_rows = 0
    row_visits_total = 0
    prefix_mass_total = 0.0
    for example in examples:
        if example.target_schema_id != LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID:
            raise CorpusAuditError("legacy summary requires factorized target v2")
        actions = example.legal_actions.get("actions", [])
        visits = example.complete_visit_counts
        if len(actions) != len(visits) or not actions:
            raise CorpusAuditError(f"legacy decision {example.decision_id} is misaligned")
        root_visits = sum(visits)
        if root_visits <= 0:
            raise CorpusAuditError(f"legacy decision {example.decision_id} has no visits")
        decisions += 1
        legal_actions += len(actions)
        positive_visits += sum(count > 0 for count in visits)
        for prefix in example.prefix_targets:
            row_visits = sum(prefix.visit_counts)
            prefix_rows += 1
            row_visits_total += row_visits
            prefix_mass_total += row_visits / root_visits
            if row_visits > 0:
                visited_prefix_rows += 1
    return {
        "decisions": decisions,
        "legal_actions": legal_actions,
        "positive_visit_actions": positive_visits,
        "assumed_all_open_actions": legal_actions,
        "prefix_rows": prefix_rows,
        "visited_prefix_rows": visited_prefix_rows,
        "row_visits_total": row_visits_total,
        "prefix_mass_total": prefix_mass_total,
        "opened_metadata_available": False,
        "interpretation": "historical all-open reference only; never promotion evidence",
    }


def _producer_config(corpus: CorpusManifest) -> dict[str, Any]:
    return {
        "ruleset": copy.deepcopy(corpus.payload.get("ruleset")),
        "evaluator": copy.deepcopy(corpus.payload.get("evaluator")),
        **{field: copy.deepcopy(corpus.payload.get(field)) for field in PRODUCER_CONFIG_FIELDS},
    }


def _corpus_identity(corpus: CorpusManifest) -> dict[str, Any]:
    return {
        "label": corpus.label,
        "run_id": corpus.run_id,
        "run_kind": corpus.payload.get("run_kind"),
        "manifest_schema_id": corpus.payload.get("manifest_schema_id"),
        "manifest_sha256": corpus.file_sha256,
        "validation_sha256": corpus.validation_file_sha256,
        "replay_schema_id": corpus.lineage[0],
        "factorized_target_schema_id": corpus.lineage[1],
        "source_seal_tag": corpus.payload.get("source_seal_tag"),
        "source_seal_commit": corpus.payload.get("source_seal_commit"),
        "engine_commit": corpus.payload.get("engine_commit"),
        "games": len(corpus.games),
        "game_index_start": corpus.games[0].game_index,
        "game_index_end": corpus.games[-1].game_index,
    }


def _merge_metrics(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key in {
            "false_zero_definition",
            "false_zero_mass_mean_per_decision",
            "false_zero_closed_child_mass_mean_per_decision",
            "weighted_closed_support_fraction_total",
            "weighted_closed_support_fraction_mean_per_decision",
            "mixed_rows",
        }:
            continue
        if key in {
            "prefix_mass_min",
            "prefix_mass_max",
            "false_zero_closed_child_mass_max_decision",
            "false_zero_mass_max_decision",
        }:
            if value is not None and (
                target.get(key) is None
                or (key in {"prefix_mass_min"} and value < target[key])
                or (key not in {"prefix_mass_min"} and value > target[key])
            ):
                target[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value
        else:
            target[key] = value


def audit_manifest_pair(
    legacy_manifest_path: str | Path,
    clean_manifest_path: str | Path,
    *,
    mixed_row_limit: int | None = 20,
) -> dict[str, Any]:
    """Audit one legacy/current manifest pair and return an unsealed report."""

    if mixed_row_limit is not None and (
        not isinstance(mixed_row_limit, int)
        or isinstance(mixed_row_limit, bool)
        or mixed_row_limit < 0
    ):
        raise CorpusAuditError("mixed_row_limit must be a non-negative integer or None")
    legacy = _load_manifest(
        legacy_manifest_path,
        label="legacy",
        expected_lineage=(
            LEGACY_GAME_RECORD_V4_SCHEMA_ID,
            LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
        ),
    )
    clean = _load_manifest(
        clean_manifest_path,
        label="clean-v3",
        expected_lineage=(
            GAME_RECORD_V5_SCHEMA_ID,
            FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
        ),
    )
    if len(legacy.games) != len(clean.games):
        raise CorpusAuditError(
            f"game count mismatch: legacy {len(legacy.games)}, clean {len(clean.games)}"
        )
    legacy_config = _producer_config(legacy)
    clean_config = _producer_config(clean)
    if legacy_config != clean_config:
        differing = sorted(
            key
            for key in set(legacy_config) | set(clean_config)
            if legacy_config.get(key) != clean_config.get(key)
        )
        raise CorpusAuditError(
            f"producer config mismatch for fields: {', '.join(differing)}"
        )
    for field in ("source_seal_tag", "source_seal_commit"):
        if legacy.payload.get(field) != clean.payload.get(field):
            raise CorpusAuditError(f"source seal mismatch for {field}")
    if legacy.payload.get("ruleset") != clean.payload.get("ruleset"):
        raise CorpusAuditError("ruleset provenance mismatch")

    clean_by_index = {game.game_index: game for game in clean.games}
    legacy_by_index = {game.game_index: game for game in legacy.games}
    if set(legacy_by_index) != set(clean_by_index):
        raise CorpusAuditError("legacy/current game_index sets differ")

    pair_rows: list[dict[str, Any]] = []
    all_mixed_rows: list[dict[str, Any]] = []
    aggregate_clean = _empty_metric_accumulator()
    aggregate_legacy = {
        "decisions": 0,
        "legal_actions": 0,
        "positive_visit_actions": 0,
        "assumed_all_open_actions": 0,
        "prefix_rows": 0,
        "visited_prefix_rows": 0,
        "row_visits_total": 0,
        "prefix_mass_total": 0.0,
        "opened_metadata_available": False,
    }
    for legacy_game in legacy.games:
        clean_game = clean_by_index[legacy_game.game_index]
        if legacy_game.seed_hex != clean_game.seed_hex:
            raise CorpusAuditError(
                f"seed mismatch for game_index {legacy_game.game_index}: "
                f"{legacy_game.seed_hex} != {clean_game.seed_hex}"
            )
        if legacy_game.decisions != clean_game.decisions:
            raise CorpusAuditError(
                f"decision count mismatch for game_index {legacy_game.game_index}: "
                f"{legacy_game.decisions} != {clean_game.decisions}"
            )
        if legacy_game.termination != clean_game.termination:
            raise CorpusAuditError(
                f"termination mismatch for game_index {legacy_game.game_index}"
            )
        if legacy_game.trajectory_sha256 != clean_game.trajectory_sha256:
            raise CorpusAuditError(
                f"trajectory hash mismatch for game_index {legacy_game.game_index}: "
                f"{legacy_game.trajectory_sha256} != {clean_game.trajectory_sha256}"
            )

        legacy_record = _load_json_object(legacy_game.path, "legacy game")
        clean_record = _load_json_object(clean_game.path, "clean-v3 game")
        legacy_trajectory = _validate_record_identity(legacy, legacy_game, legacy_record)
        clean_trajectory = _validate_record_identity(clean, clean_game, clean_record)
        _validate_players(legacy, legacy_record, legacy_game)
        _validate_players(clean, clean_record, clean_game)
        if legacy_trajectory != clean_trajectory:
            raise CorpusAuditError(
                f"recomputed trajectory mismatch for game_index {legacy_game.game_index}"
            )

        try:
            legacy_examples = parse_game_v4(
                legacy_record, str(legacy_game.path), allow_legacy=True
            )
        except Exception as error:
            raise CorpusAuditError(
                f"legacy game {legacy_game.game_index} failed quarantined validation: {error}"
            ) from error
        try:
            clean_examples = parse_game_v4(clean_record, str(clean_game.path))
        except Exception as error:
            raise CorpusAuditError(
                f"clean-v3 game {clean_game.game_index} failed production validation: {error}"
            ) from error
        if len(legacy_examples) != len(clean_examples):
            raise CorpusAuditError(
                f"parsed decision count mismatch for game_index {legacy_game.game_index}"
            )
        clean_metrics = audit_opened_examples(
            clean_examples, mixed_row_limit=None
        )
        clean_metrics["games"] = 1
        for mixed_row in clean_metrics["mixed_rows"]:
            all_mixed_rows.append(
                {
                    "game_index": clean_game.game_index,
                    "clean_game_id": clean_game.game_id,
                    **mixed_row,
                }
            )
        legacy_metrics = audit_legacy_examples(legacy_examples)
        legacy_metrics["games"] = 1
        _merge_metrics(aggregate_clean, clean_metrics)
        _merge_metrics(aggregate_legacy, legacy_metrics)
        pair_rows.append(
            {
                "game_index": legacy_game.game_index,
                "seed_hex": legacy_game.seed_hex,
                "decisions": legacy_game.decisions,
                "legacy_game_id": legacy_game.game_id,
                "clean_game_id": clean_game.game_id,
                "legacy_file_sha256": legacy_game.file_sha256,
                "clean_file_sha256": clean_game.file_sha256,
                "trajectory_sha256": legacy_game.trajectory_sha256,
                "trajectory_match": True,
                "clean_metrics": clean_metrics,
            }
        )

    if mixed_row_limit is None:
        selected_mixed_rows: list[dict[str, Any]] = []
        reported_mixed_rows = len(all_mixed_rows)
    else:
        selected_mixed_rows = sorted(
            all_mixed_rows,
            key=lambda row: (
                -row["weighted_closed_child_count"],
                row["game_index"],
                row["decision_id"],
                row["prefix"],
            ),
        )[:mixed_row_limit]
        for pair in pair_rows:
            metrics_for_pair = pair["clean_metrics"]
            metrics_for_pair["mixed_rows"] = []
            metrics_for_pair["mixed_rows_reported"] = 0
            metrics_for_pair["mixed_rows_omitted"] = metrics_for_pair[
                "mixed_rows_total"
            ]
        reported_mixed_rows = len(selected_mixed_rows)
    aggregate_clean["mixed_rows"] = selected_mixed_rows
    aggregate_clean["mixed_rows_total"] = len(all_mixed_rows)
    aggregate_clean["mixed_rows_reported"] = reported_mixed_rows
    aggregate_clean["mixed_rows_omitted"] = len(all_mixed_rows) - reported_mixed_rows
    aggregate_clean["false_zero_mass_mean_per_decision"] = (
        aggregate_clean["false_zero_mass_total"]
        / aggregate_clean["decisions"]
    )
    aggregate_clean["weighted_closed_support_fraction_total"] = aggregate_clean[
        "false_zero_mass_total"
    ]
    aggregate_clean["weighted_closed_support_fraction_mean_per_decision"] = (
        aggregate_clean["false_zero_mass_mean_per_decision"]
    )
    aggregate_clean["false_zero_closed_child_mass_mean_per_decision"] = (
        aggregate_clean["false_zero_closed_child_mass_total"]
        / aggregate_clean["decisions"]
    )
    aggregate_clean["false_zero_definition"] = FALSE_ZERO_DEFINITION
    aggregate_legacy["interpretation"] = (
        "historical all-open reference only; never promotion evidence"
    )
    pairing_merkle = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "game_index": row["game_index"],
                    "seed_hex": row["seed_hex"],
                    "trajectory_sha256": row["trajectory_sha256"],
                }
                for row in pair_rows
            ]
        )
    )
    return {
        "schema_id": AUDIT_REPORT_SCHEMA_ID,
        "status": "pass",
        "audit_contract": {
            "legacy_lineage": list(legacy.lineage),
            "clean_lineage": list(clean.lineage),
            "trajectory_hash_definition": TRAJECTORY_HASH_DEFINITION,
            "false_zero_definition": FALSE_ZERO_DEFINITION,
            "mixed_row_limit": mixed_row_limit,
            "mixed_row_location": (
                "trajectory_pairing.games[].clean_metrics.mixed_rows"
                if mixed_row_limit is None
                else "clean_opened_metrics.mixed_rows"
            ),
            "non_mutating": True,
        },
        "source_corpora": {
            "legacy": _corpus_identity(legacy),
            "clean_v3": _corpus_identity(clean),
        },
        "producer_regeneration": {
            "status": "match",
            "config": legacy_config,
            "source_seal_tag": legacy.payload.get("source_seal_tag"),
            "source_seal_commit": legacy.payload.get("source_seal_commit"),
            "legacy_engine_commit": legacy.payload.get("engine_commit"),
            "clean_engine_commit": clean.payload.get("engine_commit"),
            "engine_commit_changed_for_contract_recovery": (
                legacy.payload.get("engine_commit") != clean.payload.get("engine_commit")
            ),
        },
        "trajectory_pairing": {
            "status": "one_to_one",
            "games_compared": len(pair_rows),
            "trajectory_matches": len(pair_rows),
            "paired_trajectory_merkle_sha256": pairing_merkle,
            "games": pair_rows,
        },
        "clean_opened_metrics": aggregate_clean,
        "legacy_reference_metrics": aggregate_legacy,
    }


def seal_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic report sealed with its canonical payload hash."""

    payload = copy.deepcopy(dict(report))
    payload.pop("report_sha256", None)
    payload["report_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def verify_report_seal(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise CorpusAuditError("audit report must be an object")
    declared = report.get("report_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise CorpusAuditError("report_sha256 is missing or invalid")
    payload = copy.deepcopy(dict(report))
    payload.pop("report_sha256", None)
    actual = sha256_bytes(canonical_json_bytes(payload))
    if actual != declared:
        raise CorpusAuditError(
            f"report hash mismatch: expected {declared}, got {actual}"
        )


def write_audit_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write a sealed report atomically; identical reruns are no-ops."""

    verify_report_seal(report)
    output = Path(path).resolve()
    serialized = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if output.exists():
        try:
            existing = output.read_bytes()
        except OSError as error:
            raise CorpusAuditError(f"cannot read existing report {output}: {error}") from error
        if existing == serialized:
            return output
        raise CorpusAuditError(
            f"existing audit report differs; refusing overwrite: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(serialized)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def audit_phase1_corpus(
    legacy_manifest_path: str | Path,
    clean_manifest_path: str | Path,
    *,
    output: str | Path | None = None,
    mixed_row_limit: int | None = 20,
) -> dict[str, Any]:
    """Audit, seal, and optionally idempotently write one corpus pair."""

    report = seal_report(
        audit_manifest_pair(
            legacy_manifest_path,
            clean_manifest_path,
            mixed_row_limit=mixed_row_limit,
        )
    )
    if output is not None:
        write_audit_report(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_manifest", type=Path)
    parser.add_argument("clean_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mixed-row-limit",
        type=int,
        default=20,
        help="number of highest-impact mixed rows to include (0=aggregate only)",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="include every mixed row (intended for the external artifact report)",
    )
    parser.add_argument("--print-report", action="store_true")
    args = parser.parse_args()
    if args.mixed_row_limit < 0:
        parser.error("--mixed-row-limit must be non-negative")
    mixed_row_limit = None if args.full_report else args.mixed_row_limit
    try:
        report = audit_phase1_corpus(
            args.legacy_manifest,
            args.clean_manifest,
            output=args.output,
            mixed_row_limit=mixed_row_limit,
        )
    except CorpusAuditError as error:
        parser.exit(2, f"corpus audit failed: {error}\n")
    summary = {
        "schema_id": report["schema_id"],
        "status": report["status"],
        "games_compared": report["trajectory_pairing"]["games_compared"],
        "trajectory_matches": report["trajectory_pairing"]["trajectory_matches"],
        "report_sha256": report["report_sha256"],
        "output": str(args.output.resolve()) if args.output is not None else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.print_report:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

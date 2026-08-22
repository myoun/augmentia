"""Strict Python validation for Rust-authored ObservationV3 and ActionV3 JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

FIXTURE_SCHEMA_ID = "augmentia.golden.observation-action.v3"
OBSERVATION_SCHEMA_ID = "augmentia.observation.v3"
ACTION_SCHEMA_ID = "augmentia.action.v3"
RULESET_ID = "augment_chess_20260807"
RUNTIME_VOCABULARY_SCHEMA_ID = "augmentia.runtime-vocabulary.v3"
SPATIAL_SHAPE = [102, 8, 8]
SPATIAL_SIZE = 102 * 8 * 8
GLOBAL_SHAPE = [32]
RUNTIME_FIELD_COUNT = 147

_PLAYERS = {"self_player", "opponent"}
_ABILITIES = {
    "wizard_lightning",
    "wizard_shield",
    "wizard_meteor",
    "wizard_time_stop",
    "colossus_sector",
    "merchant_buy",
}
_TARGET_KINDS = {"none", "square", "piece_at", "squares"}
_ACTION_KINDS = {"move", "piece_ability", "card", "draft_pick", "resolve_choice"}
_RUNTIME_TYPES = {
    "none",
    "bool",
    "unsigned",
    "signed",
    "player",
    "piece",
    "card",
    "piece_kind",
    "square",
    "text",
    "list",
    "record",
}


class SchemaError(ValueError):
    """Raised when a record violates the V3 Rust/Python contract."""


def default_fixture_path() -> Path:
    return Path(__file__).with_name("fixtures") / "observation_action_v3.json"


def default_runtime_vocabulary_path() -> Path:
    return Path(__file__).with_name("fixtures") / "runtime_vocabulary_v3.json"


def load_runtime_vocabulary(path: str | Path | None = None) -> dict[str, Any]:
    vocabulary_path = (
        Path(path) if path is not None else default_runtime_vocabulary_path()
    )
    with vocabulary_path.open("r", encoding="utf-8") as vocabulary_file:
        payload = json.load(vocabulary_file)
    vocabulary = _mapping(payload, "runtime_vocabulary")
    _exact_keys(vocabulary, {"schema_id", "ruleset_id", "tokens"}, "runtime_vocabulary")
    _equal(
        vocabulary.get("schema_id"),
        RUNTIME_VOCABULARY_SCHEMA_ID,
        "runtime_vocabulary.schema_id",
    )
    _equal(vocabulary.get("ruleset_id"), RULESET_ID, "runtime_vocabulary.ruleset_id")
    tokens = _sequence(vocabulary.get("tokens"), "runtime_vocabulary.tokens")
    normalized = [
        _string(token, f"runtime_vocabulary.tokens[{index}]")
        for index, token in enumerate(tokens)
    ]
    if not normalized:
        raise SchemaError("runtime_vocabulary.tokens must not be empty")
    if normalized != sorted(set(normalized)):
        raise SchemaError("runtime_vocabulary.tokens must be sorted and unique")
    return vocabulary


def load_golden_suite(path: str | Path | None = None) -> dict[str, Any]:
    fixture_path = Path(path) if path is not None else default_fixture_path()
    with fixture_path.open("r", encoding="utf-8") as fixture_file:
        payload = json.load(fixture_file)
    validate_golden_suite(payload)
    return payload


def validate_golden_suite(payload: Any) -> None:
    suite = _mapping(payload, "suite")
    _exact_keys(
        suite,
        {
            "fixture_schema_id",
            "observation_schema_id",
            "action_schema_id",
            "ruleset_id",
            "cases",
        },
        "suite",
    )
    _equal(suite.get("fixture_schema_id"), FIXTURE_SCHEMA_ID, "suite.fixture_schema_id")
    _equal(
        suite.get("observation_schema_id"),
        OBSERVATION_SCHEMA_ID,
        "suite.observation_schema_id",
    )
    _equal(suite.get("action_schema_id"), ACTION_SCHEMA_ID, "suite.action_schema_id")
    _equal(suite.get("ruleset_id"), RULESET_ID, "suite.ruleset_id")
    cases = _sequence(suite.get("cases"), "suite.cases")
    if not cases:
        raise SchemaError("suite.cases must not be empty")

    names: set[str] = set()
    for case_index, raw_case in enumerate(cases):
        path = f"suite.cases[{case_index}]"
        case = _mapping(raw_case, path)
        _exact_keys(case, {"name", "observation", "legal_actions"}, path)
        name = _string(case.get("name"), f"{path}.name")
        if name in names:
            raise SchemaError(f"{path}.name duplicates {name!r}")
        names.add(name)
        observation = _validate_observation(case.get("observation"), f"{path}.observation")
        _validate_legal_actions(
            case.get("legal_actions"),
            f"{path}.legal_actions",
            observation,
        )


def _validate_observation(payload: Any, path: str) -> dict[str, int]:
    observation = _mapping(payload, path)
    _exact_keys(
        observation,
        {
            "schema_id",
            "ruleset_id",
            "spatial_shape",
            "spatial",
            "global_shape",
            "global",
            "state",
            "pieces",
            "cards",
            "phase",
            "runtime",
            "piece_slot_count",
            "card_slot_count",
        },
        path,
    )
    _equal(observation.get("schema_id"), OBSERVATION_SCHEMA_ID, f"{path}.schema_id")
    _equal(observation.get("ruleset_id"), RULESET_ID, f"{path}.ruleset_id")
    _equal(observation.get("spatial_shape"), SPATIAL_SHAPE, f"{path}.spatial_shape")
    _equal(observation.get("global_shape"), GLOBAL_SHAPE, f"{path}.global_shape")
    _bounded_vector(observation.get("spatial"), SPATIAL_SIZE, f"{path}.spatial")
    _bounded_vector(observation.get("global"), GLOBAL_SHAPE[0], f"{path}.global")

    piece_slot_count = _unsigned(observation.get("piece_slot_count"), f"{path}.piece_slot_count")
    card_slot_count = _unsigned(observation.get("card_slot_count"), f"{path}.card_slot_count")
    pieces = _sequence(observation.get("pieces"), f"{path}.pieces")
    piece_slots: set[int] = set()
    for index, raw_piece in enumerate(pieces):
        piece_path = f"{path}.pieces[{index}]"
        piece = _mapping(raw_piece, piece_path)
        _exact_keys(
            piece,
            {"slot", "square", "owner", "kind_code", "capture_count"},
            piece_path,
        )
        slot = _slot(piece.get("slot"), piece_slot_count, f"{piece_path}.slot")
        if slot in piece_slots:
            raise SchemaError(f"{piece_path}.slot duplicates local piece slot {slot}")
        piece_slots.add(slot)
        _square(piece.get("square"), f"{piece_path}.square")
        _player(piece.get("owner"), f"{piece_path}.owner")
        _piece_kind(piece.get("kind_code"), f"{piece_path}.kind_code")
        _unsigned(piece.get("capture_count"), f"{piece_path}.capture_count")

    cards = _sequence(observation.get("cards"), f"{path}.cards")
    card_slots: set[int] = set()
    for index, raw_card in enumerate(cards):
        card_path = f"{path}.cards[{index}]"
        card = _mapping(raw_card, card_path)
        _exact_keys(
            card,
            {
                "slot",
                "definition_id",
                "owner",
                "half_stars",
                "charges",
                "available_in_owner_turns",
            },
            card_path,
        )
        slot = _slot(card.get("slot"), card_slot_count, f"{card_path}.slot")
        if slot in card_slots:
            raise SchemaError(f"{card_path}.slot duplicates local card slot {slot}")
        card_slots.add(slot)
        _string(card.get("definition_id"), f"{card_path}.definition_id")
        _player(card.get("owner"), f"{card_path}.owner")
        _unsigned(card.get("half_stars"), f"{card_path}.half_stars")
        _unsigned(card.get("charges"), f"{card_path}.charges")
        available = card.get("available_in_owner_turns")
        if available is not None:
            _unsigned(available, f"{card_path}.available_in_owner_turns")

    _validate_global_state(observation.get("state"), f"{path}.state")
    phase = _mapping(observation.get("phase"), f"{path}.phase")
    phase_kind = _string(phase.get("kind"), f"{path}.phase.kind")
    _validate_phase(
        phase,
        phase_kind,
        f"{path}.phase",
        piece_slot_count,
        card_slot_count,
    )

    runtime = _sequence(observation.get("runtime"), f"{path}.runtime")
    if len(runtime) != RUNTIME_FIELD_COUNT:
        raise SchemaError(
            f"{path}.runtime has {len(runtime)} fields; expected {RUNTIME_FIELD_COUNT}"
        )
    for index, raw_feature in enumerate(runtime):
        feature_path = f"{path}.runtime[{index}]"
        feature = _mapping(raw_feature, feature_path)
        _exact_keys(feature, {"field_id", "value"}, feature_path)
        _equal(feature.get("field_id"), index + 1, f"{feature_path}.field_id")
        _validate_runtime_value(
            feature.get("value"),
            f"{feature_path}.value",
            piece_slot_count,
            card_slot_count,
        )
    return {"piece_slots": piece_slot_count, "card_slots": card_slot_count}


def _validate_runtime_value(
    payload: Any,
    path: str,
    piece_slot_count: int,
    card_slot_count: int,
) -> None:
    value = _mapping(payload, path)
    value_type = _string(value.get("type"), f"{path}.type")
    if value_type not in _RUNTIME_TYPES:
        raise SchemaError(f"{path}.type has unknown value {value_type!r}")
    payload_value = value.get("value")
    if value_type == "none":
        _exact_keys(value, {"type"}, path)
    elif value_type == "bool":
        _exact_keys(value, {"type", "value"}, path)
        if type(payload_value) is not bool:
            raise SchemaError(f"{path}.value must be bool")
    elif value_type == "unsigned":
        _exact_keys(value, {"type", "value"}, path)
        _unsigned(payload_value, f"{path}.value")
    elif value_type == "signed":
        _exact_keys(value, {"type", "value"}, path)
        _integer(payload_value, f"{path}.value")
    elif value_type == "player":
        _exact_keys(value, {"type", "value"}, path)
        _player(payload_value, f"{path}.value")
    elif value_type == "piece":
        _exact_keys(value, {"type", "value"}, path)
        _slot(payload_value, piece_slot_count, f"{path}.value")
    elif value_type == "card":
        _exact_keys(value, {"type", "value"}, path)
        _slot(payload_value, card_slot_count, f"{path}.value")
    elif value_type == "piece_kind":
        _exact_keys(value, {"type", "value"}, path)
        _piece_kind(payload_value, f"{path}.value")
    elif value_type == "square":
        _exact_keys(value, {"type", "value"}, path)
        _square(payload_value, f"{path}.value")
    elif value_type == "text":
        _exact_keys(value, {"type", "value"}, path)
        _string(payload_value, f"{path}.value")
    elif value_type == "list":
        _exact_keys(value, {"type", "value"}, path)
        for index, item in enumerate(_sequence(payload_value, f"{path}.value")):
            _validate_runtime_value(
                item,
                f"{path}.value[{index}]",
                piece_slot_count,
                card_slot_count,
            )
    elif value_type == "record":
        _exact_keys(value, {"type", "value"}, path)
        names: set[str] = set()
        for index, raw_field in enumerate(_sequence(payload_value, f"{path}.value")):
            field_path = f"{path}.value[{index}]"
            field = _mapping(raw_field, field_path)
            _exact_keys(field, {"name", "value"}, field_path)
            name = _string(field.get("name"), f"{field_path}.name")
            if name in names:
                raise SchemaError(f"{field_path}.name duplicates {name!r}")
            names.add(name)
            _validate_runtime_value(
                field.get("value"),
                f"{field_path}.value",
                piece_slot_count,
                card_slot_count,
            )


def _validate_legal_actions(
    payload: Any,
    path: str,
    observation: dict[str, int],
) -> None:
    legal = _mapping(payload, path)
    _exact_keys(legal, {"schema_id", "actions"}, path)
    _equal(legal.get("schema_id"), ACTION_SCHEMA_ID, f"{path}.schema_id")
    actions = _sequence(legal.get("actions"), f"{path}.actions")
    serialized: set[str] = set()
    for index, raw_descriptor in enumerate(actions):
        descriptor_path = f"{path}.actions[{index}]"
        descriptor = _mapping(raw_descriptor, descriptor_path)
        canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        if canonical in serialized:
            raise SchemaError(f"{descriptor_path} duplicates an earlier descriptor")
        serialized.add(canonical)
        _validate_descriptor(descriptor, descriptor_path, observation)


def _validate_descriptor(
    descriptor: dict[str, Any],
    path: str,
    observation: dict[str, int],
) -> None:
    kind = _string(descriptor.get("kind"), f"{path}.kind")
    if kind not in _ACTION_KINDS:
        raise SchemaError(f"{path}.kind has unknown value {kind!r}")
    if kind == "move":
        _exact_keys(
            descriptor,
            {"kind", "piece", "piece_kind", "from", "to", "promotion"},
            path,
        )
        _slot(descriptor.get("piece"), observation["piece_slots"], f"{path}.piece")
        _piece_kind(descriptor.get("piece_kind"), f"{path}.piece_kind")
        _square(descriptor.get("from"), f"{path}.from")
        _square(descriptor.get("to"), f"{path}.to")
        promotion = descriptor.get("promotion")
        if promotion is not None:
            _piece_kind(promotion, f"{path}.promotion")
    elif kind == "piece_ability":
        _exact_keys(
            descriptor,
            {"kind", "piece", "piece_kind", "from", "ability", "target"},
            path,
        )
        _slot(descriptor.get("piece"), observation["piece_slots"], f"{path}.piece")
        _piece_kind(descriptor.get("piece_kind"), f"{path}.piece_kind")
        _square(descriptor.get("from"), f"{path}.from")
        ability = _string(descriptor.get("ability"), f"{path}.ability")
        if ability not in _ABILITIES:
            raise SchemaError(f"{path}.ability has unknown value {ability!r}")
        _validate_target(descriptor.get("target"), f"{path}.target")
    elif kind == "card":
        _exact_keys(descriptor, {"kind", "card", "definition_id", "target"}, path)
        _slot(descriptor.get("card"), observation["card_slots"], f"{path}.card")
        _string(descriptor.get("definition_id"), f"{path}.definition_id")
        _validate_target(descriptor.get("target"), f"{path}.target")
    elif kind == "draft_pick":
        _exact_keys(descriptor, {"kind", "offer_index", "definition_id"}, path)
        _unsigned(descriptor.get("offer_index"), f"{path}.offer_index")
        _string(descriptor.get("definition_id"), f"{path}.definition_id")
    elif kind == "resolve_choice":
        _exact_keys(
            descriptor,
            {"kind", "choice_slot", "handler_id", "option_index", "option"},
            path,
        )
        _equal(descriptor.get("choice_slot"), 0, f"{path}.choice_slot")
        handler_id = _string(descriptor.get("handler_id"), f"{path}.handler_id")
        option_index = _unsigned(descriptor.get("option_index"), f"{path}.option_index")
        option = _string(descriptor.get("option"), f"{path}.option")
        expected_option = _model_choice_option_token(handler_id, option_index)
        if option != expected_option:
            raise SchemaError(
                f"{path}.option must be stable token {expected_option!r}, got {option!r}"
            )


def _validate_target(payload: Any, path: str) -> None:
    target = _mapping(payload, path)
    kind = _string(target.get("kind"), f"{path}.kind")
    if kind not in _TARGET_KINDS:
        raise SchemaError(f"{path}.kind has unknown value {kind!r}")
    if kind == "none":
        _exact_keys(target, {"kind"}, path)
    elif kind in {"square", "piece_at"}:
        _exact_keys(target, {"kind", "square"}, path)
        _square(target.get("square"), f"{path}.square")
    elif kind == "squares":
        _exact_keys(target, {"kind", "squares"}, path)
        for index, square in enumerate(_sequence(target.get("squares"), f"{path}.squares")):
            _square(square, f"{path}.squares[{index}]")


def _bounded_vector(payload: Any, expected_len: int, path: str) -> None:
    values = _sequence(payload, path)
    if len(values) != expected_len:
        raise SchemaError(f"{path} has length {len(values)}; expected {expected_len}")
    for index, value in enumerate(values):
        if type(value) not in {int, float}:
            raise SchemaError(f"{path}[{index}] must be numeric")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SchemaError(f"{path}[{index}] must be finite and within [0, 1]")


def _validate_global_state(payload: Any, path: str) -> None:
    state = _mapping(payload, path)
    _exact_keys(
        state,
        {
            "mode_id",
            "ply",
            "quiet_ply",
            "actions_remaining",
            "cards_used_this_turn",
            "half_star_totals",
            "turns_taken",
            "completed_drafts",
            "shared_turn_count",
            "current_repetition_count",
            "status",
            "deathmatch",
            "en_passant",
            "applied_rules",
        },
        path,
    )
    _string(state.get("mode_id"), f"{path}.mode_id")
    for field in (
        "ply",
        "quiet_ply",
        "actions_remaining",
        "shared_turn_count",
        "current_repetition_count",
    ):
        _unsigned(state.get(field), f"{path}.{field}")
    for field in ("cards_used_this_turn", "half_star_totals", "turns_taken"):
        values = _sequence(state.get(field), f"{path}.{field}")
        if len(values) != 2:
            raise SchemaError(f"{path}.{field} must contain self and opponent")
        for index, value in enumerate(values):
            _unsigned(value, f"{path}.{field}[{index}]")
    drafts = _sequence(state.get("completed_drafts"), f"{path}.completed_drafts")
    if len(drafts) != 3 or any(type(value) is not bool for value in drafts):
        raise SchemaError(f"{path}.completed_drafts must contain three booleans")

    status = _mapping(state.get("status"), f"{path}.status")
    status_kind = _string(status.get("kind"), f"{path}.status.kind")
    if status_kind == "ongoing":
        _exact_keys(status, {"kind"}, f"{path}.status")
    elif status_kind == "won":
        _exact_keys(status, {"kind", "winner", "reason", "detail"}, f"{path}.status")
        _player(status.get("winner"), f"{path}.status.winner")
        _string(status.get("reason"), f"{path}.status.reason")
        _optional_string(status.get("detail"), f"{path}.status.detail")
    elif status_kind == "draw":
        _exact_keys(status, {"kind", "reason", "detail"}, f"{path}.status")
        _string(status.get("reason"), f"{path}.status.reason")
        _optional_string(status.get("detail"), f"{path}.status.detail")
    else:
        raise SchemaError(f"{path}.status.kind has unknown value {status_kind!r}")

    deathmatch = state.get("deathmatch")
    if deathmatch is not None:
        deathmatch_path = f"{path}.deathmatch"
        deathmatch = _mapping(deathmatch, deathmatch_path)
        _exact_keys(
            deathmatch,
            {
                "age_shared_turns",
                "quiet_full_turns",
                "interval_full_turns",
                "progress_this_round",
            },
            deathmatch_path,
        )
        for field in ("age_shared_turns", "quiet_full_turns", "interval_full_turns"):
            _unsigned(deathmatch.get(field), f"{deathmatch_path}.{field}")
        if type(deathmatch.get("progress_this_round")) is not bool:
            raise SchemaError(f"{deathmatch_path}.progress_this_round must be bool")

    en_passant = state.get("en_passant")
    if en_passant is not None:
        en_passant_path = f"{path}.en_passant"
        en_passant = _mapping(en_passant, en_passant_path)
        _exact_keys(en_passant, {"target", "captured", "pawn_owner"}, en_passant_path)
        _square(en_passant.get("target"), f"{en_passant_path}.target")
        _square(en_passant.get("captured"), f"{en_passant_path}.captured")
        _player(en_passant.get("pawn_owner"), f"{en_passant_path}.pawn_owner")

    for index, rule in enumerate(_sequence(state.get("applied_rules"), f"{path}.applied_rules")):
        _string(rule, f"{path}.applied_rules[{index}]")


def _validate_phase(
    phase: dict[str, Any],
    kind: str,
    path: str,
    piece_slot_count: int,
    card_slot_count: int,
) -> None:
    if kind == "playing":
        _exact_keys(phase, {"kind"}, path)
    elif kind == "forced_move":
        _exact_keys(
            phase,
            {"kind", "piece", "source_kind", "source_id", "continuation"},
            path,
        )
        _slot(phase.get("piece"), piece_slot_count, f"{path}.piece")
        source_kind = _string(phase.get("source_kind"), f"{path}.source_kind")
        if source_kind not in {"platform", "effect"}:
            raise SchemaError(f"{path}.source_kind has unknown value {source_kind!r}")
        _optional_string(phase.get("source_id"), f"{path}.source_id")
        _continuation(phase.get("continuation"), f"{path}.continuation")
    elif kind == "draft":
        _exact_keys(phase, {"kind", "draft_phase", "offers", "resume_turn"}, path)
        draft_phase = _string(phase.get("draft_phase"), f"{path}.draft_phase")
        if draft_phase not in {"opening", "middle", "end"}:
            raise SchemaError(f"{path}.draft_phase has unknown value {draft_phase!r}")
        for index, offer in enumerate(_sequence(phase.get("offers"), f"{path}.offers")):
            _string(offer, f"{path}.offers[{index}]")
        _player(phase.get("resume_turn"), f"{path}.resume_turn")
    elif kind == "awaiting_choice":
        _exact_keys(
            phase,
            {
                "kind",
                "choice_slot",
                "owner",
                "source_card",
                "handler_id",
                "options",
                "continuation",
            },
            path,
        )
        _equal(phase.get("choice_slot"), 0, f"{path}.choice_slot")
        _player(phase.get("owner"), f"{path}.owner")
        _slot(phase.get("source_card"), card_slot_count, f"{path}.source_card")
        handler_id = _string(phase.get("handler_id"), f"{path}.handler_id")
        for index, option in enumerate(_sequence(phase.get("options"), f"{path}.options")):
            option = _string(option, f"{path}.options[{index}]")
            expected_option = _model_choice_option_token(handler_id, index)
            if option != expected_option:
                raise SchemaError(
                    f"{path}.options[{index}] must be stable token "
                    f"{expected_option!r}, got {option!r}"
                )
        _continuation(phase.get("continuation"), f"{path}.continuation")
    else:
        raise SchemaError(f"{path}.kind has unknown value {kind!r}")


def _continuation(payload: Any, path: str) -> str:
    continuation = _string(payload, path)
    if continuation not in {"resume_turn", "finish_turn"}:
        raise SchemaError(f"{path} has unknown continuation {continuation!r}")
    return continuation


def _model_choice_option_token(handler_id: str, option_index: int) -> str:
    return f"choice:{handler_id}:{option_index}"


def _optional_string(payload: Any, path: str) -> str | None:
    if payload is None:
        return None
    return _string(payload, path)


def _slot(payload: Any, count: int, path: str) -> int:
    slot = _unsigned(payload, path)
    if slot >= count:
        raise SchemaError(f"{path} local slot {slot} is outside 0..{count}")
    return slot


def _piece_kind(payload: Any, path: str) -> int:
    code = _unsigned(payload, path)
    if not 1 <= code <= 51:
        raise SchemaError(f"{path} piece kind {code} is outside 1..51")
    return code


def _square(payload: Any, path: str) -> int:
    square = _unsigned(payload, path)
    if square >= 64:
        raise SchemaError(f"{path} square {square} is outside 0..64")
    return square


def _player(payload: Any, path: str) -> str:
    player = _string(payload, path)
    if player not in _PLAYERS:
        raise SchemaError(f"{path} has unknown relative player {player!r}")
    return player


def _unsigned(payload: Any, path: str) -> int:
    value = _integer(payload, path)
    if value < 0:
        raise SchemaError(f"{path} must be non-negative")
    return value


def _integer(payload: Any, path: str) -> int:
    if type(payload) is not int:
        raise SchemaError(f"{path} must be an integer")
    return payload


def _string(payload: Any, path: str) -> str:
    if not isinstance(payload, str) or not payload:
        raise SchemaError(f"{path} must be a non-empty string")
    return payload


def _mapping(payload: Any, path: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SchemaError(f"{path} must be an object")
    return payload


def _sequence(payload: Any, path: str) -> list[Any]:
    if not isinstance(payload, list):
        raise SchemaError(f"{path} must be an array")
    return payload


def _equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise SchemaError(f"{path} is {actual!r}; expected {expected!r}")


def _exact_keys(payload: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SchemaError(f"{path} fields differ; missing={missing}, extra={extra}")

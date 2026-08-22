"""Tensorization contract for the first ObservationV3/ActionV3 network."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from .schema import GLOBAL_SHAPE, RUNTIME_FIELD_COUNT, SPATIAL_SHAPE, SchemaError

PLAYER_IDS = {"self_player": 1, "opponent": 2}
PHASE_IDS = {"playing": 1, "forced_move": 2, "draft": 3, "awaiting_choice": 4}
RUNTIME_TYPE_IDS = {
    "none": 1,
    "bool": 2,
    "unsigned": 3,
    "signed": 4,
    "player": 5,
    "piece": 6,
    "card": 7,
    "piece_kind": 8,
    "square": 9,
    "text": 10,
    "list": 11,
    "record": 12,
}
ACTION_KIND_IDS = {
    "move": 1,
    "piece_ability": 2,
    "card": 3,
    "draft_pick": 4,
    "resolve_choice": 5,
}
TARGET_KIND_IDS = {"none": 1, "square": 2, "piece_at": 3, "squares": 4}
ABILITY_IDS = {
    "wizard_lightning": 1,
    "wizard_shield": 2,
    "wizard_meteor": 3,
    "wizard_time_stop": 4,
    "colossus_sector": 5,
    "merchant_buy": 6,
}


def _default_catalog_path() -> Path:
    return (
        Path(__file__).parents[2]
        / "rulesets"
        / "augment_chess_20260807"
        / "cards"
        / "catalog.json"
    )


@dataclass(frozen=True)
class V3Vocabulary:
    """Stable exact-string vocabulary; index zero is reserved for absence."""

    tokens: tuple[str, ...]
    _lookup: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.tokens or self.tokens[0] != "<none>":
            raise ValueError("V3 vocabulary index zero must be <none>")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("V3 vocabulary contains duplicate tokens")
        object.__setattr__(
            self,
            "_lookup",
            MappingProxyType({token: index for index, token in enumerate(self.tokens)}),
        )

    @classmethod
    def from_cases(
        cls,
        cases: Iterable[Mapping[str, Any]],
        catalog_path: str | Path | None = None,
        extra_tokens: Iterable[str] = (),
    ) -> V3Vocabulary:
        """Build the immutable run vocabulary from every training example."""
        tokens: set[str] = set()
        for case in cases:
            collect_case_text_tokens(case, tokens)
        return cls.from_tokens(tokens, catalog_path, extra_tokens)

    @classmethod
    def from_tokens(
        cls,
        tokens: Iterable[str],
        catalog_path: str | Path | None = None,
        extra_tokens: Iterable[str] = (),
    ) -> V3Vocabulary:
        """Build the run vocabulary from tokens collected during replay validation."""
        collected = set(tokens)
        collected.update(extra_tokens)

        path = (
            Path(catalog_path) if catalog_path is not None else _default_catalog_path()
        )
        catalog = json.loads(path.read_text(encoding="utf-8"))
        for card in catalog["cards"]:
            collected.add(card["id"])
            for optional_key in ("effect", "target"):
                value = card.get(optional_key)
                if isinstance(value, str):
                    collected.add(value)
        return cls(("<none>", *sorted(token for token in collected if token)))

    @classmethod
    def from_golden_suite(
        cls,
        suite: Mapping[str, Any],
        catalog_path: str | Path | None = None,
        extra_tokens: Iterable[str] = (),
    ) -> V3Vocabulary:
        return cls.from_cases(suite["cases"], catalog_path, extra_tokens)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> V3Vocabulary:
        if set(payload) != {"schema_id", "tokens"}:
            raise SchemaError(
                "model vocabulary must contain exactly schema_id and tokens"
            )
        if payload["schema_id"] != "augmentia.model-vocabulary.v3":
            raise SchemaError(
                f"unsupported model vocabulary schema {payload['schema_id']!r}"
            )
        tokens = payload["tokens"]
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise SchemaError("model vocabulary tokens must be a string list")
        return cls(tuple(tokens))

    @property
    def size(self) -> int:
        return len(self.tokens)

    def index(self, token: str | None, path: str) -> int:
        if token is None:
            return 0
        try:
            return self._lookup[token]
        except KeyError as error:
            raise SchemaError(
                f"{path} contains unknown model vocabulary token {token!r}; "
                "regenerate the run vocabulary instead of hashing or using an unknown bucket"
            ) from error

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": "augmentia.model-vocabulary.v3",
            "tokens": list(self.tokens),
        }


def collect_case_text_tokens(case: Mapping[str, Any], tokens: set[str]) -> None:
    """Collect exact-string features without retaining a training example."""
    _collect_observation_text(case["observation"], tokens)
    for action in case["legal_actions"]["actions"]:
        _collect_action_text(action, tokens)


def _collect_observation_text(observation: Mapping[str, Any], tokens: set[str]) -> None:
    state = observation["state"]
    tokens.add(state["mode_id"])
    tokens.update(state["applied_rules"])
    status = state["status"]
    tokens.add(status["kind"])
    if status.get("reason"):
        tokens.add(status["reason"])
    if status.get("detail"):
        tokens.add(status["detail"])
    for card in observation["cards"]:
        tokens.add(card["definition_id"])
    phase = observation["phase"]
    for key in (
        "source_kind",
        "source_id",
        "continuation",
        "draft_phase",
        "handler_id",
    ):
        value = phase.get(key)
        if isinstance(value, str):
            tokens.add(value)
    tokens.update(phase.get("offers", ()))
    tokens.update(phase.get("options", ()))
    for feature in observation["runtime"]:
        _collect_runtime_text(feature["value"], tokens)


def _collect_runtime_text(value: Mapping[str, Any], tokens: set[str]) -> None:
    value_type = value["type"]
    if value_type == "text":
        tokens.add(value["value"])
    elif value_type == "list":
        for child in value["value"]:
            _collect_runtime_text(child, tokens)
    elif value_type == "record":
        for child in value["value"]:
            tokens.add(child["name"])
            _collect_runtime_text(child["value"], tokens)


def _collect_action_text(action: Mapping[str, Any], tokens: set[str]) -> None:
    for key in ("definition_id", "handler_id", "option"):
        value = action.get(key)
        if isinstance(value, str):
            tokens.add(value)


@dataclass
class V3Batch:
    spatial: torch.Tensor
    global_features: torch.Tensor
    state_text: torch.Tensor
    state_text_position: torch.Tensor
    state_text_mask: torch.Tensor
    entity_counts: torch.Tensor
    piece_kind: torch.Tensor
    piece_square: torch.Tensor
    piece_owner: torch.Tensor
    piece_slot: torch.Tensor
    piece_numeric: torch.Tensor
    piece_mask: torch.Tensor
    card_definition: torch.Tensor
    card_owner: torch.Tensor
    card_slot: torch.Tensor
    card_numeric: torch.Tensor
    card_mask: torch.Tensor
    phase_kind: torch.Tensor
    phase_owner: torch.Tensor
    phase_piece_ref: torch.Tensor
    phase_card_ref: torch.Tensor
    phase_numeric: torch.Tensor
    phase_text: torch.Tensor
    phase_text_position: torch.Tensor
    phase_text_mask: torch.Tensor
    runtime_field: torch.Tensor
    runtime_type: torch.Tensor
    runtime_text: torch.Tensor
    runtime_player: torch.Tensor
    runtime_piece_kind: torch.Tensor
    runtime_square: torch.Tensor
    runtime_piece_ref: torch.Tensor
    runtime_card_ref: torch.Tensor
    runtime_parent: torch.Tensor
    runtime_numeric: torch.Tensor
    runtime_mask: torch.Tensor
    action_kind: torch.Tensor
    action_piece_ref: torch.Tensor
    action_piece_kind: torch.Tensor
    action_from: torch.Tensor
    action_to: torch.Tensor
    action_promotion: torch.Tensor
    action_ability: torch.Tensor
    action_card_ref: torch.Tensor
    action_definition: torch.Tensor
    action_handler: torch.Tensor
    action_option: torch.Tensor
    action_target_kind: torch.Tensor
    action_target_squares: torch.Tensor
    action_target_sequence: torch.Tensor
    action_target_sequence_position: torch.Tensor
    action_target_sequence_mask: torch.Tensor
    action_numeric: torch.Tensor
    action_mask: torch.Tensor

    def to(self, device: torch.device | str) -> V3Batch:
        return V3Batch(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )

    @property
    def batch_size(self) -> int:
        return int(self.spatial.shape[0])


@dataclass
class _RuntimeNode:
    field_id: int
    value_type: int
    text: int = 0
    player: int = 0
    piece_kind: int = 0
    square: int = 0
    piece_ref: int = -1
    card_ref: int = -1
    parent: int = -1
    numeric: tuple[float, ...] = (0.0,) * 7


def tensorize_cases(
    cases: Sequence[Mapping[str, Any]],
    vocabulary: V3Vocabulary,
    device: torch.device | str | None = None,
) -> V3Batch:
    if not cases:
        raise ValueError("cannot tensorize an empty case batch")
    batch_size = len(cases)
    max_pieces = max(1, max(len(case["observation"]["pieces"]) for case in cases))
    max_cards = max(1, max(len(case["observation"]["cards"]) for case in cases))
    max_actions = max(1, max(len(case["legal_actions"]["actions"]) for case in cases))
    max_target_squares = max(
        1,
        max(
            (
                len(_action_target_square_list(action))
                for case in cases
                for action in case["legal_actions"]["actions"]
            ),
            default=0,
        ),
    )

    runtime_nodes = [
        _flatten_runtime(observation=case["observation"], vocabulary=vocabulary)
        for case in cases
    ]
    max_runtime = max(1, max(len(nodes) for nodes in runtime_nodes))
    phase_tokens = [
        _phase_tokens(case["observation"]["phase"], vocabulary) for case in cases
    ]
    max_phase_text = max(1, max(len(tokens) for tokens in phase_tokens))
    state_tokens = [
        _state_tokens(case["observation"]["state"], vocabulary) for case in cases
    ]
    max_state_text = max(1, max(len(tokens) for tokens in state_tokens))

    spatial = torch.zeros((batch_size, *SPATIAL_SHAPE), dtype=torch.float32)
    global_features = torch.zeros((batch_size, GLOBAL_SHAPE[0]), dtype=torch.float32)
    state_text = torch.zeros((batch_size, max_state_text), dtype=torch.long)
    state_text_position = torch.zeros(
        (batch_size, max_state_text, 1), dtype=torch.float32
    )
    state_text_mask = torch.zeros((batch_size, max_state_text), dtype=torch.bool)
    entity_counts = torch.zeros((batch_size, 2), dtype=torch.float32)
    piece_kind = torch.zeros((batch_size, max_pieces), dtype=torch.long)
    piece_square = torch.zeros_like(piece_kind)
    piece_owner = torch.zeros_like(piece_kind)
    piece_slot = torch.full_like(piece_kind, -1)
    piece_numeric = torch.zeros((batch_size, max_pieces, 2), dtype=torch.float32)
    piece_mask = torch.zeros((batch_size, max_pieces), dtype=torch.bool)
    card_definition = torch.zeros((batch_size, max_cards), dtype=torch.long)
    card_owner = torch.zeros_like(card_definition)
    card_slot = torch.full_like(card_definition, -1)
    card_numeric = torch.zeros((batch_size, max_cards, 5), dtype=torch.float32)
    card_mask = torch.zeros((batch_size, max_cards), dtype=torch.bool)
    phase_kind = torch.zeros(batch_size, dtype=torch.long)
    phase_owner = torch.zeros(batch_size, dtype=torch.long)
    phase_piece_ref = torch.full((batch_size,), -1, dtype=torch.long)
    phase_card_ref = torch.full((batch_size,), -1, dtype=torch.long)
    phase_numeric = torch.zeros((batch_size, 2), dtype=torch.float32)
    phase_text = torch.zeros((batch_size, max_phase_text), dtype=torch.long)
    phase_text_position = torch.zeros(
        (batch_size, max_phase_text, 1), dtype=torch.float32
    )
    phase_text_mask = torch.zeros((batch_size, max_phase_text), dtype=torch.bool)

    runtime_field = torch.zeros((batch_size, max_runtime), dtype=torch.long)
    runtime_type = torch.zeros_like(runtime_field)
    runtime_text = torch.zeros_like(runtime_field)
    runtime_player = torch.zeros_like(runtime_field)
    runtime_piece_kind = torch.zeros_like(runtime_field)
    runtime_square = torch.zeros_like(runtime_field)
    runtime_piece_ref = torch.full_like(runtime_field, -1)
    runtime_card_ref = torch.full_like(runtime_field, -1)
    runtime_parent = torch.full_like(runtime_field, -1)
    runtime_numeric = torch.zeros((batch_size, max_runtime, 7), dtype=torch.float32)
    runtime_mask = torch.zeros((batch_size, max_runtime), dtype=torch.bool)

    action_kind = torch.zeros((batch_size, max_actions), dtype=torch.long)
    action_piece_ref = torch.full_like(action_kind, -1)
    action_piece_kind = torch.zeros_like(action_kind)
    action_from = torch.zeros_like(action_kind)
    action_to = torch.zeros_like(action_kind)
    action_promotion = torch.zeros_like(action_kind)
    action_ability = torch.zeros_like(action_kind)
    action_card_ref = torch.full_like(action_kind, -1)
    action_definition = torch.zeros_like(action_kind)
    action_handler = torch.zeros_like(action_kind)
    action_option = torch.zeros_like(action_kind)
    action_target_kind = torch.zeros_like(action_kind)
    action_target_squares = torch.zeros(
        (batch_size, max_actions, 64), dtype=torch.float32
    )
    action_target_sequence = torch.zeros(
        (batch_size, max_actions, max_target_squares), dtype=torch.long
    )
    action_target_sequence_position = torch.zeros(
        (batch_size, max_actions, max_target_squares, 1), dtype=torch.float32
    )
    action_target_sequence_mask = torch.zeros(
        (batch_size, max_actions, max_target_squares), dtype=torch.bool
    )
    action_numeric = torch.zeros((batch_size, max_actions, 6), dtype=torch.float32)
    action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)

    for batch_index, case in enumerate(cases):
        observation = case["observation"]
        spatial[batch_index] = torch.tensor(
            observation["spatial"], dtype=torch.float32
        ).reshape(SPATIAL_SHAPE)
        global_features[batch_index] = torch.tensor(
            observation["global"], dtype=torch.float32
        )
        for index, token in enumerate(state_tokens[batch_index]):
            state_text[batch_index, index] = token
            state_text_position[batch_index, index, 0] = _signed_log(index)
            state_text_mask[batch_index, index] = True
        entity_counts[batch_index] = torch.tensor(
            [
                _signed_log(observation["piece_slot_count"]),
                _signed_log(observation["card_slot_count"]),
            ],
            dtype=torch.float32,
        )
        for index, piece in enumerate(observation["pieces"]):
            piece_kind[batch_index, index] = piece["kind_code"]
            piece_square[batch_index, index] = piece["square"] + 1
            piece_owner[batch_index, index] = PLAYER_IDS[piece["owner"]]
            piece_slot[batch_index, index] = piece["slot"]
            piece_numeric[batch_index, index] = torch.tensor(
                [_signed_log(piece["capture_count"]), _signed_log(piece["slot"])],
                dtype=torch.float32,
            )
            piece_mask[batch_index, index] = True
        for index, card in enumerate(observation["cards"]):
            card_definition[batch_index, index] = vocabulary.index(
                card["definition_id"],
                f"cases[{batch_index}].observation.cards[{index}]",
            )
            card_owner[batch_index, index] = PLAYER_IDS[card["owner"]]
            card_slot[batch_index, index] = card["slot"]
            available = card["available_in_owner_turns"]
            card_numeric[batch_index, index] = torch.tensor(
                [
                    _signed_log(card["half_stars"]),
                    _signed_log(card["charges"]),
                    float(available is not None),
                    _signed_log(available or 0),
                    _signed_log(card["slot"]),
                ],
                dtype=torch.float32,
            )
            card_mask[batch_index, index] = True

        _write_phase(
            batch_index,
            observation["phase"],
            phase_tokens[batch_index],
            phase_kind,
            phase_owner,
            phase_piece_ref,
            phase_card_ref,
            phase_numeric,
            phase_text,
            phase_text_position,
            phase_text_mask,
        )
        for index, node in enumerate(runtime_nodes[batch_index]):
            runtime_field[batch_index, index] = node.field_id
            runtime_type[batch_index, index] = node.value_type
            runtime_text[batch_index, index] = node.text
            runtime_player[batch_index, index] = node.player
            runtime_piece_kind[batch_index, index] = node.piece_kind
            runtime_square[batch_index, index] = node.square
            runtime_piece_ref[batch_index, index] = node.piece_ref
            runtime_card_ref[batch_index, index] = node.card_ref
            runtime_parent[batch_index, index] = node.parent
            runtime_numeric[batch_index, index] = torch.tensor(node.numeric)
            runtime_mask[batch_index, index] = True

        for index, action in enumerate(case["legal_actions"]["actions"]):
            _write_action(
                batch_index,
                index,
                action,
                vocabulary,
                action_kind,
                action_piece_ref,
                action_piece_kind,
                action_from,
                action_to,
                action_promotion,
                action_ability,
                action_card_ref,
                action_definition,
                action_handler,
                action_option,
                action_target_kind,
                action_target_squares,
                action_target_sequence,
                action_target_sequence_position,
                action_target_sequence_mask,
                action_numeric,
            )
            action_mask[batch_index, index] = True

    batch = V3Batch(
        spatial=spatial,
        global_features=global_features,
        state_text=state_text,
        state_text_position=state_text_position,
        state_text_mask=state_text_mask,
        entity_counts=entity_counts,
        piece_kind=piece_kind,
        piece_square=piece_square,
        piece_owner=piece_owner,
        piece_slot=piece_slot,
        piece_numeric=piece_numeric,
        piece_mask=piece_mask,
        card_definition=card_definition,
        card_owner=card_owner,
        card_slot=card_slot,
        card_numeric=card_numeric,
        card_mask=card_mask,
        phase_kind=phase_kind,
        phase_owner=phase_owner,
        phase_piece_ref=phase_piece_ref,
        phase_card_ref=phase_card_ref,
        phase_numeric=phase_numeric,
        phase_text=phase_text,
        phase_text_position=phase_text_position,
        phase_text_mask=phase_text_mask,
        runtime_field=runtime_field,
        runtime_type=runtime_type,
        runtime_text=runtime_text,
        runtime_player=runtime_player,
        runtime_piece_kind=runtime_piece_kind,
        runtime_square=runtime_square,
        runtime_piece_ref=runtime_piece_ref,
        runtime_card_ref=runtime_card_ref,
        runtime_parent=runtime_parent,
        runtime_numeric=runtime_numeric,
        runtime_mask=runtime_mask,
        action_kind=action_kind,
        action_piece_ref=action_piece_ref,
        action_piece_kind=action_piece_kind,
        action_from=action_from,
        action_to=action_to,
        action_promotion=action_promotion,
        action_ability=action_ability,
        action_card_ref=action_card_ref,
        action_definition=action_definition,
        action_handler=action_handler,
        action_option=action_option,
        action_target_kind=action_target_kind,
        action_target_squares=action_target_squares,
        action_target_sequence=action_target_sequence,
        action_target_sequence_position=action_target_sequence_position,
        action_target_sequence_mask=action_target_sequence_mask,
        action_numeric=action_numeric,
        action_mask=action_mask,
    )
    return batch if device is None else batch.to(device)


def _flatten_runtime(
    observation: Mapping[str, Any], vocabulary: V3Vocabulary
) -> list[_RuntimeNode]:
    nodes: list[_RuntimeNode] = []
    runtime = observation["runtime"]
    if len(runtime) != RUNTIME_FIELD_COUNT:
        raise SchemaError(
            f"runtime has {len(runtime)} fields; expected {RUNTIME_FIELD_COUNT}"
        )
    for feature in runtime:
        _append_runtime_node(
            nodes,
            feature["field_id"],
            feature["value"],
            vocabulary,
            parent=-1,
            depth=0,
            sibling=feature["field_id"] - 1,
            name=None,
        )
    return nodes


def _append_runtime_node(
    nodes: list[_RuntimeNode],
    field_id: int,
    value: Mapping[str, Any],
    vocabulary: V3Vocabulary,
    parent: int,
    depth: int,
    sibling: int,
    name: str | None,
) -> None:
    value_type = value["type"]
    node_index = len(nodes)
    node = _RuntimeNode(
        field_id=field_id,
        value_type=RUNTIME_TYPE_IDS[value_type],
        text=vocabulary.index(name, f"runtime[{field_id}].record_name"),
        parent=parent,
        numeric=(0.0, 0.0, _signed_log(depth), _signed_log(sibling), 0.0, 0.0, 0.0),
    )
    payload = value.get("value")
    numeric = list(node.numeric)
    if value_type == "bool":
        numeric[0], numeric[1] = float(payload), 1.0
    elif value_type in {"unsigned", "signed"}:
        numeric[0], numeric[1] = _signed_log(payload), 1.0
    elif value_type == "player":
        node.player = PLAYER_IDS[payload]
    elif value_type == "piece":
        node.piece_ref = payload
        numeric[4] = _signed_log(payload)
    elif value_type == "card":
        node.card_ref = payload
        numeric[5] = _signed_log(payload)
    elif value_type == "piece_kind":
        node.piece_kind = payload
    elif value_type == "square":
        node.square = payload + 1
    elif value_type == "text":
        node.text = vocabulary.index(payload, f"runtime[{field_id}].text")
    node.numeric = tuple(numeric)
    nodes.append(node)

    if value_type == "list":
        for child_index, child in enumerate(payload):
            _append_runtime_node(
                nodes,
                field_id,
                child,
                vocabulary,
                parent=node_index,
                depth=depth + 1,
                sibling=child_index,
                name=None,
            )
    elif value_type == "record":
        for child_index, child in enumerate(payload):
            _append_runtime_node(
                nodes,
                field_id,
                child["value"],
                vocabulary,
                parent=node_index,
                depth=depth + 1,
                sibling=child_index,
                name=child["name"],
            )


def _phase_tokens(phase: Mapping[str, Any], vocabulary: V3Vocabulary) -> list[int]:
    values: list[str] = []
    for key in (
        "source_kind",
        "source_id",
        "continuation",
        "draft_phase",
        "handler_id",
    ):
        value = phase.get(key)
        if isinstance(value, str):
            values.append(value)
    values.extend(phase.get("offers", ()))
    values.extend(phase.get("options", ()))
    return [
        vocabulary.index(value, f"phase.{index}") for index, value in enumerate(values)
    ]


def _state_tokens(state: Mapping[str, Any], vocabulary: V3Vocabulary) -> list[int]:
    values = [state["mode_id"], *state["applied_rules"]]
    return [
        vocabulary.index(value, f"state.text[{index}]")
        for index, value in enumerate(values)
    ]


def _write_phase(
    batch_index: int,
    phase: Mapping[str, Any],
    tokens: Sequence[int],
    phase_kind: torch.Tensor,
    phase_owner: torch.Tensor,
    phase_piece_ref: torch.Tensor,
    phase_card_ref: torch.Tensor,
    phase_numeric: torch.Tensor,
    phase_text: torch.Tensor,
    phase_text_position: torch.Tensor,
    phase_text_mask: torch.Tensor,
) -> None:
    phase_kind[batch_index] = PHASE_IDS[phase["kind"]]
    if "owner" in phase:
        phase_owner[batch_index] = PLAYER_IDS[phase["owner"]]
    elif "resume_turn" in phase:
        phase_owner[batch_index] = PLAYER_IDS[phase["resume_turn"]]
    if "piece" in phase:
        phase_piece_ref[batch_index] = phase["piece"]
    if "source_card" in phase:
        phase_card_ref[batch_index] = phase["source_card"]
    choice_slot = phase.get("choice_slot")
    phase_numeric[batch_index] = torch.tensor(
        [_signed_log(choice_slot or 0), float(choice_slot is not None)],
        dtype=torch.float32,
    )
    for index, token in enumerate(tokens):
        phase_text[batch_index, index] = token
        phase_text_position[batch_index, index, 0] = _signed_log(index)
        phase_text_mask[batch_index, index] = True


def _write_action(
    batch_index: int,
    action_index: int,
    action: Mapping[str, Any],
    vocabulary: V3Vocabulary,
    action_kind: torch.Tensor,
    action_piece_ref: torch.Tensor,
    action_piece_kind: torch.Tensor,
    action_from: torch.Tensor,
    action_to: torch.Tensor,
    action_promotion: torch.Tensor,
    action_ability: torch.Tensor,
    action_card_ref: torch.Tensor,
    action_definition: torch.Tensor,
    action_handler: torch.Tensor,
    action_option: torch.Tensor,
    action_target_kind: torch.Tensor,
    action_target_squares: torch.Tensor,
    action_target_sequence: torch.Tensor,
    action_target_sequence_position: torch.Tensor,
    action_target_sequence_mask: torch.Tensor,
    action_numeric: torch.Tensor,
) -> None:
    kind = action["kind"]
    action_kind[batch_index, action_index] = ACTION_KIND_IDS[kind]
    piece_ref = action.get("piece")
    card_ref = action.get("card")
    if piece_ref is not None:
        action_piece_ref[batch_index, action_index] = piece_ref
        action_piece_kind[batch_index, action_index] = action["piece_kind"]
    if card_ref is not None:
        action_card_ref[batch_index, action_index] = card_ref
    if "from" in action:
        action_from[batch_index, action_index] = action["from"] + 1
    if "to" in action:
        action_to[batch_index, action_index] = action["to"] + 1
    if action.get("promotion") is not None:
        action_promotion[batch_index, action_index] = action["promotion"]
    if "ability" in action:
        action_ability[batch_index, action_index] = ABILITY_IDS[action["ability"]]
    if "definition_id" in action:
        action_definition[batch_index, action_index] = vocabulary.index(
            action["definition_id"], f"action[{action_index}].definition_id"
        )
    if "handler_id" in action:
        action_handler[batch_index, action_index] = vocabulary.index(
            action["handler_id"], f"action[{action_index}].handler_id"
        )
    if "option" in action:
        action_option[batch_index, action_index] = vocabulary.index(
            action["option"], f"action[{action_index}].option"
        )

    target = action.get("target", {"kind": "none"})
    target_kind = target["kind"]
    action_target_kind[batch_index, action_index] = TARGET_KIND_IDS[target_kind]
    squares = _action_target_square_list(action)
    for sequence_index, square in enumerate(squares):
        action_target_squares[batch_index, action_index, square] = 1.0
        action_target_sequence[batch_index, action_index, sequence_index] = square + 1
        action_target_sequence_position[
            batch_index, action_index, sequence_index, 0
        ] = _signed_log(sequence_index)
        action_target_sequence_mask[batch_index, action_index, sequence_index] = True

    action_numeric[batch_index, action_index] = torch.tensor(
        [
            _signed_log(action.get("offer_index", 0)),
            _signed_log(action.get("choice_slot", 0)),
            _signed_log(action.get("option_index", 0)),
            _signed_log(len(squares)),
            _signed_log(piece_ref or 0),
            _signed_log(card_ref or 0),
        ],
        dtype=torch.float32,
    )


def _signed_log(value: float) -> float:
    numeric = float(value)
    return math.copysign(math.log1p(abs(numeric)), numeric)


def _action_target_square_list(action: Mapping[str, Any]) -> list[int]:
    target = action.get("target", {"kind": "none"})
    if target["kind"] in {"square", "piece_at"}:
        return [target["square"]]
    if target["kind"] == "squares":
        return list(target["squares"])
    return []

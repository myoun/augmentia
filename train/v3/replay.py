"""Verified V3 replay loading, deterministic sampling, and target collation."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

import torch
import zstandard

from .schema import (
    ACTION_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    SchemaError,
    validate_golden_suite,
)
from .tensorize import V3Batch, V3Vocabulary, tensorize_cases

MANIFEST_SCHEMA_ID = "augmentia.dataset.manifest.v3"
DECISION_SCHEMA_ID = "augmentia.training.decision.v3"


@dataclass(frozen=True)
class DatasetIdentity:
    manifest_path: str
    dataset_id: str
    manifest_sha256: str
    shard_sha256: tuple[tuple[str, str], ...]
    ruleset_id: str
    ruleset_manifest_sha256: str
    observation_schema_id: str = OBSERVATION_SCHEMA_ID
    action_schema_id: str = ACTION_SCHEMA_ID

    def to_json(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "dataset_id": self.dataset_id,
            "manifest_sha256": self.manifest_sha256,
            "shard_sha256": [list(item) for item in self.shard_sha256],
            "ruleset_id": self.ruleset_id,
            "ruleset_manifest_sha256": self.ruleset_manifest_sha256,
            "observation_schema_id": self.observation_schema_id,
            "action_schema_id": self.action_schema_id,
        }


@dataclass(frozen=True)
class TrainingExampleV3:
    decision_id: str
    observation: Mapping[str, Any]
    legal_actions: Mapping[str, Any]
    policy: tuple[float, ...] | None
    value: float | None
    forced_win_distance: float | None = None

    @property
    def case(self) -> dict[str, Any]:
        return {"observation": self.observation, "legal_actions": self.legal_actions}


@dataclass
class TrainingBatchV3:
    inputs: V3Batch
    policy_targets: torch.Tensor
    policy_supervision: torch.Tensor
    value_targets: torch.Tensor
    value_supervision: torch.Tensor
    auxiliary_targets: torch.Tensor
    auxiliary_supervision: torch.Tensor
    decision_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> TrainingBatchV3:
        return TrainingBatchV3(
            inputs=self.inputs.to(device),
            policy_targets=self.policy_targets.to(device),
            policy_supervision=self.policy_supervision.to(device),
            value_targets=self.value_targets.to(device),
            value_supervision=self.value_supervision.to(device),
            auxiliary_targets=self.auxiliary_targets.to(device),
            auxiliary_supervision=self.auxiliary_supervision.to(device),
            decision_ids=self.decision_ids,
        )


class ReplayDatasetV3(Sequence[TrainingExampleV3]):
    """A verified immutable view of one manifest split.

    Records are materialized deliberately: the sampler cursor then identifies an exact
    record independent of zstd buffering or DataLoader worker scheduling.
    """

    def __init__(self, manifest_path: str | Path, split: str = "train") -> None:
        path = Path(manifest_path).resolve()
        raw_manifest = path.read_bytes()
        try:
            manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as error:
            raise SchemaError(f"invalid dataset manifest JSON: {error}") from error
        _validate_manifest(manifest)

        examples: list[TrainingExampleV3] = []
        shard_hashes: list[tuple[str, str]] = []
        rulesets: set[tuple[str, str]] = set()
        for shard in manifest["shards"]:
            if shard["split"] != split:
                continue
            shard_path = _resolve_shard(path.parent, shard["path"])
            compressed = shard_path.read_bytes()
            digest = hashlib.sha256(compressed).hexdigest()
            if digest != shard["sha256"]:
                raise SchemaError(
                    f"shard checksum mismatch for {shard['path']}: {digest} != {shard['sha256']}"
                )
            if len(compressed) != shard["compressed_bytes"]:
                raise SchemaError(f"shard byte count mismatch for {shard['path']}")
            shard_hashes.append((shard["path"], digest))
            record_count, loaded = _read_shard(compressed, shard["path"], rulesets)
            if record_count != shard["records"]:
                raise SchemaError(f"shard record count mismatch for {shard['path']}")
            examples.extend(loaded)

        if not examples:
            raise SchemaError(f"dataset split {split!r} contains no decision records")
        if len(rulesets) != 1:
            raise SchemaError(
                "one training run must contain exactly one ruleset artifact"
            )
        ruleset_id, ruleset_hash = next(iter(rulesets))
        self.manifest = manifest
        self.examples = tuple(examples)
        self.identity = DatasetIdentity(
            manifest_path=str(path),
            dataset_id=manifest["dataset_id"],
            manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
            shard_sha256=tuple(shard_hashes),
            ruleset_id=ruleset_id,
            ruleset_manifest_sha256=ruleset_hash,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExampleV3:
        return self.examples[index]


@dataclass
class DeterministicSamplerV3:
    dataset_size: int
    seed: int
    epoch: int = 0
    cursor: int = 0

    def __post_init__(self) -> None:
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if not 0 <= self.cursor <= self.dataset_size:
            raise ValueError("sampler cursor is outside the epoch")

    def remaining_indices(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.dataset_size, generator=generator).tolist()
        yield from order[self.cursor :]

    def take(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("batch size must be positive")
        indices = list(self.remaining_indices())[:count]
        self.cursor += len(indices)
        return indices

    def advance_epoch_if_complete(self) -> None:
        if self.cursor == self.dataset_size:
            self.epoch += 1
            self.cursor = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("dataset_size") != self.dataset_size
            or state.get("seed") != self.seed
        ):
            raise ValueError(
                "checkpoint sampler does not match dataset size or shuffle seed"
            )
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self.__post_init__()


def collate_examples(
    examples: Sequence[TrainingExampleV3], vocabulary: V3Vocabulary
) -> TrainingBatchV3:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    inputs = tensorize_cases([example.case for example in examples], vocabulary)
    policy_targets = torch.zeros_like(inputs.action_mask, dtype=torch.float32)
    policy_supervision = torch.zeros(len(examples), dtype=torch.bool)
    value_targets = torch.zeros(len(examples), dtype=torch.float32)
    value_supervision = torch.zeros(len(examples), dtype=torch.bool)
    auxiliary_targets = torch.zeros(len(examples), dtype=torch.float32)
    auxiliary_supervision = torch.zeros(len(examples), dtype=torch.bool)
    for row, example in enumerate(examples):
        if example.policy is not None:
            policy_targets[row, : len(example.policy)] = torch.tensor(example.policy)
            policy_supervision[row] = True
        if example.value is not None:
            value_targets[row] = example.value
            value_supervision[row] = True
        if example.forced_win_distance is not None:
            auxiliary_targets[row] = example.forced_win_distance
            auxiliary_supervision[row] = True
    return TrainingBatchV3(
        inputs,
        policy_targets,
        policy_supervision,
        value_targets,
        value_supervision,
        auxiliary_targets,
        auxiliary_supervision,
        tuple(example.decision_id for example in examples),
    )


def _validate_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise SchemaError("dataset manifest must be an object")
    required = {
        "manifest_schema_id",
        "dataset_id",
        "created_at_utc",
        "observation_schema_id",
        "action_schema_id",
        "split_policy",
        "compression",
        "shards",
        "totals",
    }
    if set(manifest) != required:
        raise SchemaError("dataset manifest fields do not match the V3 contract")
    if manifest["manifest_schema_id"] != MANIFEST_SCHEMA_ID:
        raise SchemaError("unsupported dataset manifest schema")
    if manifest["observation_schema_id"] != OBSERVATION_SCHEMA_ID:
        raise SchemaError("observation schema mismatch")
    if manifest["action_schema_id"] != ACTION_SCHEMA_ID:
        raise SchemaError("action schema mismatch")
    if manifest["compression"].get("codec") != "zstd":
        raise SchemaError("V3 replay shards must use zstd")
    if not isinstance(manifest["shards"], list) or not manifest["shards"]:
        raise SchemaError("dataset manifest must list at least one shard")
    seen: set[str] = set()
    for shard in manifest["shards"]:
        relative = shard.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise SchemaError("shard paths must be unique strings")
        seen.add(relative)
        if shard.get("split") not in {"train", "validation", "holdout"}:
            raise SchemaError(f"invalid split for shard {relative}")
        if shard.get("record_schema_id") not in {
            DECISION_SCHEMA_ID,
            "augmentia.replay.game.v3",
        }:
            raise SchemaError(f"unsupported record schema for shard {relative}")
        if not isinstance(shard.get("sha256"), str) or len(shard["sha256"]) != 64:
            raise SchemaError(f"invalid checksum for shard {relative}")


def _resolve_shard(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise SchemaError(f"unsafe shard path {relative!r}")
    result = (root / Path(*parsed.parts)).resolve()
    if root.resolve() not in result.parents:
        raise SchemaError(f"shard escapes manifest directory: {relative!r}")
    return result


def _read_shard(
    compressed: bytes, name: str, rulesets: set[tuple[str, str]]
) -> tuple[int, list[TrainingExampleV3]]:
    try:
        with zstandard.ZstdDecompressor().stream_reader(
            io.BytesIO(compressed)
        ) as reader:
            payload = reader.read()
    except zstandard.ZstdError as error:
        raise SchemaError(f"cannot decompress shard {name}: {error}") from error
    examples = []
    lines = payload.splitlines()
    for line_number, raw in enumerate(lines, 1):
        if not raw:
            raise SchemaError(f"{name}:{line_number}: empty JSON line")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SchemaError(f"{name}:{line_number}: invalid JSON: {error}") from error
        path = f"{name}:{line_number}"
        schema = record.get("record_schema_id") if isinstance(record, dict) else None
        if schema == DECISION_SCHEMA_ID:
            examples.append(_parse_decision(record, path, rulesets))
        elif schema == "augmentia.replay.game.v3":
            examples.extend(_parse_game(record, path, rulesets))
        else:
            raise SchemaError(f"{path}: unsupported record schema {schema!r}")
    return len(lines), examples


def _parse_game(
    record: Mapping[str, Any], path: str, rulesets: set[tuple[str, str]]
) -> list[TrainingExampleV3]:
    if (
        record.get("observation_schema_id") != OBSERVATION_SCHEMA_ID
        or record.get("action_schema_id") != ACTION_SCHEMA_ID
    ):
        raise SchemaError(f"{path}: game schema mismatch")
    game_id = record.get("game_id")
    ruleset = record.get("ruleset", {})
    ruleset_id, ruleset_hash = ruleset.get("id"), ruleset.get("manifest_sha256")
    if not isinstance(game_id, str) or not game_id:
        raise SchemaError(f"{path}: invalid game_id")
    if (
        not isinstance(ruleset_id, str)
        or not isinstance(ruleset_hash, str)
        or len(ruleset_hash) != 64
    ):
        raise SchemaError(f"{path}: invalid ruleset reference")
    rulesets.add((ruleset_id, ruleset_hash))
    termination = record.get("termination", {})
    result = (
        termination.get("result")
        if termination.get("kind") == "engine_terminal"
        else None
    )
    examples = []
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise SchemaError(f"{path}: game must contain steps")
    for index, step in enumerate(steps):
        decision_id = f"{game_id}:{index}"
        case = {
            "observation": step.get("observation"),
            "legal_actions": step.get("legal_actions"),
        }
        validate_golden_suite(
            {
                "fixture_schema_id": "augmentia.golden.observation-action.v3",
                "observation_schema_id": OBSERVATION_SCHEMA_ID,
                "action_schema_id": ACTION_SCHEMA_ID,
                "ruleset_id": ruleset_id,
                "cases": [{"name": decision_id, **case}],
            }
        )
        action_count = len(case["legal_actions"]["actions"])
        policy_payload = step.get("policy_target")
        policy = None
        if isinstance(policy_payload, dict) and policy_payload.get("kind") != "none":
            policy = _policy_target(
                {"scope": "augment_policy", "target": policy_payload},
                action_count,
                path,
            )
        actor = step.get("actor")
        value = None
        if result == "draw":
            value = 0.0
        elif result in {"white_win", "black_win"} and actor in {"white", "black"}:
            value = 1.0 if result.removesuffix("_win") == actor else -1.0
        if policy is not None or value is not None:
            examples.append(
                TrainingExampleV3(
                    decision_id,
                    case["observation"],
                    case["legal_actions"],
                    policy,
                    value,
                )
            )
    return examples


def _parse_decision(
    record: Any, path: str, rulesets: set[tuple[str, str]]
) -> TrainingExampleV3:
    if (
        not isinstance(record, dict)
        or record.get("record_schema_id") != DECISION_SCHEMA_ID
    ):
        raise SchemaError(f"{path}: expected {DECISION_SCHEMA_ID}")
    if record.get("observation_schema_id") != OBSERVATION_SCHEMA_ID:
        raise SchemaError(f"{path}: observation schema mismatch")
    if record.get("action_schema_id") != ACTION_SCHEMA_ID:
        raise SchemaError(f"{path}: action schema mismatch")
    decision_id = record.get("decision_id")
    if not isinstance(decision_id, str) or not decision_id:
        raise SchemaError(f"{path}: decision_id must be non-empty")
    ruleset = record.get("ruleset", {})
    ruleset_id, ruleset_hash = ruleset.get("id"), ruleset.get("manifest_sha256")
    if (
        not isinstance(ruleset_id, str)
        or not isinstance(ruleset_hash, str)
        or len(ruleset_hash) != 64
    ):
        raise SchemaError(f"{path}: invalid ruleset reference")
    rulesets.add((ruleset_id, ruleset_hash))
    case = {
        "observation": record.get("observation"),
        "legal_actions": record.get("legal_actions"),
    }
    validate_golden_suite(
        {
            "fixture_schema_id": "augmentia.golden.observation-action.v3",
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "ruleset_id": ruleset_id,
            "cases": [{"name": decision_id, **case}],
        }
    )
    action_count = len(case["legal_actions"]["actions"])
    if action_count == 0:
        raise SchemaError(f"{path}: a training decision must have legal actions")
    targets = record.get("targets")
    if not isinstance(targets, dict):
        raise SchemaError(f"{path}: targets must be an object")
    policy = _policy_target(targets.get("policy"), action_count, path)
    value = _value_target(targets.get("value"), path)
    auxiliary = _auxiliary_target(targets.get("auxiliary"), path)
    if policy is None and value is None and auxiliary is None:
        raise SchemaError(f"{path}: decision has no trainable target")
    return TrainingExampleV3(
        decision_id,
        case["observation"],
        case["legal_actions"],
        policy,
        value,
        auxiliary,
    )


def _policy_target(target: Any, count: int, path: str) -> tuple[float, ...] | None:
    if target is None:
        return None
    if not isinstance(target, dict) or target.get("scope") not in {
        "augment_policy",
        "board_pretraining",
    }:
        raise SchemaError(f"{path}: invalid policy supervision")
    value = target.get("target", {})
    kind = value.get("kind")
    if kind == "behavior_distribution":
        probabilities = value.get("probabilities")
    elif kind == "mcts_visits":
        visits = value.get("visit_counts")
        if not isinstance(visits, list) or sum(visits) <= 0:
            raise SchemaError(f"{path}: MCTS visits must have positive mass")
        probabilities = [item / sum(visits) for item in visits]
    elif kind == "optimal_action_set":
        indices = value.get("action_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or len(set(indices)) != len(indices)
        ):
            raise SchemaError(
                f"{path}: optimal action set must be unique and non-empty"
            )
        probabilities = [0.0] * count
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < count:
                raise SchemaError(f"{path}: optimal action index is illegal")
            probabilities[index] = 1.0 / len(indices)
    else:
        raise SchemaError(f"{path}: unsupported policy target {kind!r}")
    if not isinstance(probabilities, list) or len(probabilities) != count:
        raise SchemaError(f"{path}: policy width does not match legal actions")
    if any(
        not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0
        for item in probabilities
    ):
        raise SchemaError(
            f"{path}: policy probabilities must be finite and non-negative"
        )
    total = float(sum(probabilities))
    if not math.isclose(total, 1.0, abs_tol=1e-5):
        raise SchemaError(f"{path}: policy probabilities must sum to one, got {total}")
    return tuple(float(item) for item in probabilities)


def _value_target(target: Any, path: str) -> float | None:
    if target is None:
        return None
    if not isinstance(target, dict) or target.get("kind") not in {
        "terminal_outcome",
        "proven",
    }:
        raise SchemaError(f"{path}: invalid value target")
    value = target.get("value")
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not -1 <= value <= 1
    ):
        raise SchemaError(f"{path}: value target must be finite and in [-1, 1]")
    return float(value)


def _auxiliary_target(targets: Any, path: str) -> float | None:
    if not isinstance(targets, list):
        raise SchemaError(f"{path}: auxiliary targets must be a list")
    result = None
    for target in targets:
        if (
            not isinstance(target, dict)
            or target.get("name") != "forced_win_distance_decisions"
        ):
            raise SchemaError(f"{path}: unsupported auxiliary target {target!r}")
        value = target.get("value", {})
        if (
            value.get("type") != "integer"
            or not isinstance(value.get("value"), int)
            or value["value"] < 0
        ):
            raise SchemaError(
                f"{path}: forced-win distance must be a non-negative integer"
            )
        if result is not None:
            raise SchemaError(f"{path}: duplicate forced-win distance target")
        result = math.log1p(value["value"])
    return result

"""Verified V3 replay loading, deterministic sampling, and target collation."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import uuid
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
import zstandard

from .schema import (
    ACTION_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    SchemaError,
    validate_golden_suite,
)
from .tensorize import (
    V3Batch,
    V3Vocabulary,
    collect_case_text_tokens,
    tensorize_cases,
)

MANIFEST_SCHEMA_ID = "augmentia.dataset.manifest.v3"
DECISION_SCHEMA_ID = "augmentia.training.decision.v3"
REPLAY_INDEX_SCHEMA_ID = "augmentia.replay-index.v3.1"
REPLAY_PARSER_VERSION = "augmentia.replay-parser.v3.1"
REPLAY_VERIFIER_VERSION = "augmentia.python-schema-verifier.v3.1"


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


@dataclass(frozen=True)
class VerifiedReplayShardV3:
    path: Path
    relative_path: str
    expected_sha256: str
    expected_bytes: int
    expected_records: int


@dataclass(frozen=True)
class ReplayShardV3:
    path: Path
    relative_path: str
    expected_sha256: str
    expected_bytes: int
    expected_records: int
    example_count: int
    start_index: int


@dataclass(frozen=True)
class ReplayShardScanV3:
    shard_index: int
    record_count: int
    example_count: int
    rulesets: frozenset[tuple[str, str]]
    vocabulary_tokens: frozenset[str]


@dataclass
class VocabularyTokenCollectorV3:
    tokens: set[str]
    standard_chess_collected: bool = False

    def observe(
        self, record: Mapping[str, Any], examples: Sequence[TrainingExampleV3]
    ) -> None:
        provenance = record.get("provenance")
        standard_chess = (
            isinstance(provenance, dict)
            and provenance.get("source_domain") == "standard_chess"
        )
        if standard_chess and self.standard_chess_collected:
            return
        for example in examples:
            collect_case_text_tokens(example.case, self.tokens)
        if standard_chess:
            self.standard_chess_collected = True


class ReplayDatasetV3(Sequence[TrainingExampleV3]):
    """A verified immutable view with bounded, shard-local materialization."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        *,
        max_cached_shards: int = 1,
        verification_workers: int | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        if max_cached_shards <= 0:
            raise ValueError("max_cached_shards must be positive")
        if verification_workers is not None and verification_workers <= 0:
            raise ValueError("verification_workers must be positive")
        path = Path(manifest_path).resolve()
        raw_manifest = path.read_bytes()
        try:
            manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as error:
            raise SchemaError(f"invalid dataset manifest JSON: {error}") from error
        _validate_manifest(manifest)

        manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
        verified_shards: list[VerifiedReplayShardV3] = []
        shard_hashes: list[tuple[str, str]] = []
        for shard in manifest["shards"]:
            if shard["split"] != split:
                continue
            shard_path = _resolve_shard(path.parent, shard["path"])
            digest = _sha256_file(shard_path)
            if digest != shard["sha256"]:
                raise SchemaError(
                    f"shard checksum mismatch for {shard['path']}: {digest} != {shard['sha256']}"
                )
            if shard_path.stat().st_size != shard["compressed_bytes"]:
                raise SchemaError(f"shard byte count mismatch for {shard['path']}")
            shard_hashes.append((shard["path"], digest))
            verified_shards.append(
                VerifiedReplayShardV3(
                    path=shard_path,
                    relative_path=shard["path"],
                    expected_sha256=digest,
                    expected_bytes=shard["compressed_bytes"],
                    expected_records=shard["records"],
                )
            )

        index_path = _replay_index_path(path, split, manifest_sha256)
        cached_index = _load_replay_index(
            index_path, manifest_sha256, split, verified_shards
        )
        self.index_cache_hit = cached_index is not None
        self.verification_workers = 0
        if cached_index is None:
            self.verification_workers = _effective_verification_workers(
                verified_shards, verification_workers
            )
            shards, example_count, ruleset, vocabulary_tokens, index_rows = (
                _scan_verified_shards(
                    verified_shards,
                    progress_callback,
                    workers=self.verification_workers,
                )
            )
            _write_replay_index(
                index_path,
                manifest_sha256,
                split,
                index_rows,
                ruleset,
                vocabulary_tokens,
            )
        else:
            shards, example_count, ruleset, vocabulary_tokens = cached_index

        if example_count == 0:
            raise SchemaError(f"dataset split {split!r} contains no decision records")
        ruleset_id, ruleset_hash = ruleset
        self.manifest = manifest
        self.split = split
        self.shards = tuple(shards)
        self._example_count = example_count
        self._shard_ends = tuple(
            shard.start_index + shard.example_count for shard in self.shards
        )
        self._max_cached_shards = max_cached_shards
        self._vocabulary_tokens = frozenset(vocabulary_tokens)
        self.index_cache_path = index_path
        self._cache: OrderedDict[int, tuple[TrainingExampleV3, ...]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self.identity = DatasetIdentity(
            manifest_path=str(path),
            dataset_id=manifest["dataset_id"],
            manifest_sha256=manifest_sha256,
            shard_sha256=tuple(shard_hashes),
            ruleset_id=ruleset_id,
            ruleset_manifest_sha256=ruleset_hash,
        )

    def __len__(self) -> int:
        return self._example_count

    def __getitem__(self, index: int) -> TrainingExampleV3:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = self.shard_index(index)
        shard = self.shards[shard_index]
        return self._load_shard(shard_index)[index - shard.start_index]

    @property
    def sampler_group_sizes(self) -> tuple[int, ...]:
        return tuple(shard.example_count for shard in self.shards)

    @property
    def vocabulary_tokens(self) -> frozenset[str]:
        return self._vocabulary_tokens

    def build_vocabulary(self) -> V3Vocabulary:
        return V3Vocabulary.from_tokens(self._vocabulary_tokens)

    def shard_index(self, index: int) -> int:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return bisect_right(self._shard_ends, index)

    def cache_info(self) -> dict[str, int]:
        return {
            "capacity": self._max_cached_shards,
            "resident_shards": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    def action_count(self, index: int) -> int:
        example = self[index]
        actions = example.legal_actions.get("actions")
        if not isinstance(actions, list) or not actions:
            raise SchemaError(f"example {index} has no legal actions")
        return len(actions)

    def _load_shard(self, shard_index: int) -> tuple[TrainingExampleV3, ...]:
        cached = self._cache.pop(shard_index, None)
        if cached is not None:
            self._cache_hits += 1
            self._cache[shard_index] = cached
            return cached
        self._cache_misses += 1
        while len(self._cache) >= self._max_cached_shards:
            self._cache.popitem(last=False)
        shard = self.shards[shard_index]
        if (
            shard.path.stat().st_size != shard.expected_bytes
            or _sha256_file(shard.path) != shard.expected_sha256
        ):
            raise SchemaError(
                f"shard changed after verification: {shard.relative_path}"
            )
        rulesets: set[tuple[str, str]] = set()
        record_count, example_count, examples = _read_shard_path(
            shard.path,
            shard.relative_path,
            rulesets,
            collect=True,
            token_collector=None,
        )
        if (
            record_count != shard.expected_records
            or example_count != shard.example_count
        ):
            raise SchemaError(
                f"shard contents changed after verification: {shard.relative_path}"
            )
        loaded = tuple(examples)
        self._cache[shard_index] = loaded
        return loaded


@dataclass
class DeterministicSamplerV3:
    dataset_size: int
    seed: int
    group_sizes: tuple[int, ...] = ()
    epoch: int = 0
    cursor: int = 0

    def __post_init__(self) -> None:
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if not self.group_sizes:
            self.group_sizes = (self.dataset_size,)
        if any(size <= 0 for size in self.group_sizes):
            raise ValueError("sampler group sizes must be positive")
        if sum(self.group_sizes) != self.dataset_size:
            raise ValueError("sampler group sizes must cover the dataset exactly")
        if not 0 <= self.cursor <= self.dataset_size:
            raise ValueError("sampler cursor is outside the epoch")

    def remaining_indices(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        if len(self.group_sizes) == 1:
            order: Iterator[int] = iter(
                torch.randperm(self.dataset_size, generator=generator).tolist()
            )
        else:
            offsets = []
            offset = 0
            for size in self.group_sizes:
                offsets.append(offset)
                offset += size

            def grouped_order() -> Iterator[int]:
                for group_index in torch.randperm(
                    len(self.group_sizes), generator=generator
                ).tolist():
                    group_offset = offsets[group_index]
                    yield from (
                        group_offset + local
                        for local in torch.randperm(
                            self.group_sizes[group_index], generator=generator
                        ).tolist()
                    )

            order = grouped_order()
        yield from islice(order, self.cursor, None)

    def take(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("batch size must be positive")
        indices = list(islice(self.remaining_indices(), count))
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
            "group_sizes": self.group_sizes,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("dataset_size") != self.dataset_size
            or state.get("seed") != self.seed
            or tuple(state.get("group_sizes", ())) != self.group_sizes
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


def _read_shard_path(
    path: Path,
    name: str,
    rulesets: set[tuple[str, str]],
    *,
    collect: bool,
    token_collector: VocabularyTokenCollectorV3 | None,
) -> tuple[int, int, list[TrainingExampleV3]]:
    examples: list[TrainingExampleV3] = []
    record_count = 0
    example_count = 0
    try:
        with path.open("rb") as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                with io.BufferedReader(reader) as buffered:
                    for line_number, raw in enumerate(buffered, 1):
                        record_count += 1
                        if not raw.strip():
                            raise SchemaError(f"{name}:{line_number}: empty JSON line")
                        try:
                            record = json.loads(raw)
                        except json.JSONDecodeError as error:
                            raise SchemaError(
                                f"{name}:{line_number}: invalid JSON: {error}"
                            ) from error
                        record_examples = _parse_record(
                            record, f"{name}:{line_number}", rulesets
                        )
                        if token_collector is not None:
                            token_collector.observe(record, record_examples)
                        example_count += len(record_examples)
                        if collect:
                            examples.extend(record_examples)
    except (OSError, zstandard.ZstdError) as error:
        raise SchemaError(f"cannot decompress shard {name}: {error}") from error
    return record_count, example_count, examples


def _parse_record(
    record: Any, path: str, rulesets: set[tuple[str, str]]
) -> list[TrainingExampleV3]:
    schema = record.get("record_schema_id") if isinstance(record, dict) else None
    if schema == DECISION_SCHEMA_ID:
        return [_parse_decision(record, path, rulesets)]
    if schema == "augmentia.replay.game.v3":
        return _parse_game(record, path, rulesets)
    raise SchemaError(f"{path}: unsupported record schema {schema!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_index_path(manifest_path: Path, split: str, manifest_sha256: str) -> Path:
    return manifest_path.with_name(
        f"{manifest_path.name}.{split}.{manifest_sha256}.replay-index-v3.json"
    )


def _loader_code_version() -> str:
    digest = hashlib.sha256()
    for source in (
        Path(__file__),
        Path(__file__).with_name("schema.py"),
        Path(__file__).with_name("tensorize.py"),
    ):
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _index_schema_versions() -> dict[str, str]:
    return {
        "manifest": MANIFEST_SCHEMA_ID,
        "decision": DECISION_SCHEMA_ID,
        "game": "augmentia.replay.game.v3",
        "observation": OBSERVATION_SCHEMA_ID,
        "action": ACTION_SCHEMA_ID,
    }


def _scan_verified_shards(
    verified_shards: Sequence[VerifiedReplayShardV3],
    progress_callback: Callable[[int, int, str], None] | None = None,
    *,
    workers: int = 1,
) -> tuple[
    list[ReplayShardV3],
    int,
    tuple[str, str],
    frozenset[str],
    list[dict[str, Any]],
]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    scans: list[ReplayShardScanV3 | None] = [None] * len(verified_shards)
    completed = 0
    if workers == 1:
        for shard_index, shard in enumerate(verified_shards):
            scans[shard_index] = _scan_one_verified_shard(shard_index, shard)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(verified_shards), shard.relative_path)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_scan_one_verified_shard, shard_index, shard): (
                    shard_index,
                    shard,
                )
                for shard_index, shard in enumerate(verified_shards)
            }
            try:
                for future in as_completed(futures):
                    shard_index, shard = futures[future]
                    scans[shard_index] = future.result()
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(
                            completed, len(verified_shards), shard.relative_path
                        )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    shards: list[ReplayShardV3] = []
    index_rows: list[dict[str, Any]] = []
    rulesets: set[tuple[str, str]] = set()
    vocabulary_tokens: set[str] = set()
    example_count = 0
    for shard_index, (shard, scan) in enumerate(
        zip(verified_shards, scans, strict=True)
    ):
        if scan is None or scan.shard_index != shard_index:
            raise RuntimeError(
                "parallel shard verification returned an incomplete result"
            )
        if scan.record_count != shard.expected_records:
            raise SchemaError(f"shard record count mismatch for {shard.relative_path}")
        rulesets.update(scan.rulesets)
        vocabulary_tokens.update(scan.vocabulary_tokens)
        index_rows.append(
            {
                "path": shard.relative_path,
                "sha256": shard.expected_sha256,
                "compressed_bytes": shard.expected_bytes,
                "records": shard.expected_records,
                "examples": scan.example_count,
                "start_index": example_count,
            }
        )
        if scan.example_count > 0:
            shards.append(
                ReplayShardV3(
                    path=shard.path,
                    relative_path=shard.relative_path,
                    expected_sha256=shard.expected_sha256,
                    expected_bytes=shard.expected_bytes,
                    expected_records=scan.record_count,
                    example_count=scan.example_count,
                    start_index=example_count,
                )
            )
        example_count += scan.example_count
    if len(rulesets) != 1:
        raise SchemaError("one training run must contain exactly one ruleset artifact")
    return (
        shards,
        example_count,
        next(iter(rulesets)),
        frozenset(vocabulary_tokens),
        index_rows,
    )


def _effective_verification_workers(
    shards: Sequence[VerifiedReplayShardV3], requested: int | None
) -> int:
    if not shards:
        return 1
    if requested is not None:
        return min(requested, len(shards))
    total_bytes = sum(shard.expected_bytes for shard in shards)
    if len(shards) < 4 or total_bytes < 64 * 1024 * 1024:
        return 1
    return min(os.cpu_count() or 1, len(shards))


def _scan_one_verified_shard(
    shard_index: int, shard: VerifiedReplayShardV3
) -> ReplayShardScanV3:
    rulesets: set[tuple[str, str]] = set()
    token_collector = VocabularyTokenCollectorV3(set())
    record_count, example_count, _ = _read_shard_path(
        shard.path,
        shard.relative_path,
        rulesets,
        collect=False,
        token_collector=token_collector,
    )
    return ReplayShardScanV3(
        shard_index=shard_index,
        record_count=record_count,
        example_count=example_count,
        rulesets=frozenset(rulesets),
        vocabulary_tokens=frozenset(token_collector.tokens),
    )


def _load_replay_index(
    index_path: Path,
    manifest_sha256: str,
    split: str,
    verified_shards: Sequence[VerifiedReplayShardV3],
) -> tuple[list[ReplayShardV3], int, tuple[str, str], frozenset[str]] | None:
    try:
        payload = json.loads(index_path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "schema_id",
        "schema_versions",
        "parser_version",
        "verifier_version",
        "code_version",
        "manifest_sha256",
        "split",
        "shards",
        "ruleset",
        "vocabulary_tokens",
    }:
        return None
    if (
        payload["schema_id"] != REPLAY_INDEX_SCHEMA_ID
        or payload["schema_versions"] != _index_schema_versions()
        or payload["parser_version"] != REPLAY_PARSER_VERSION
        or payload["verifier_version"] != REPLAY_VERIFIER_VERSION
        or payload["code_version"] != _loader_code_version()
        or payload["manifest_sha256"] != manifest_sha256
        or payload["split"] != split
    ):
        return None
    rows = payload["shards"]
    if not isinstance(rows, list) or len(rows) != len(verified_shards):
        return None
    ruleset = payload["ruleset"]
    if (
        not isinstance(ruleset, dict)
        or set(ruleset) != {"id", "manifest_sha256"}
        or not isinstance(ruleset["id"], str)
        or not isinstance(ruleset["manifest_sha256"], str)
        or len(ruleset["manifest_sha256"]) != 64
    ):
        return None
    vocabulary_tokens = payload["vocabulary_tokens"]
    if (
        not isinstance(vocabulary_tokens, list)
        or not all(isinstance(token, str) and token for token in vocabulary_tokens)
        or len(set(vocabulary_tokens)) != len(vocabulary_tokens)
    ):
        return None

    shards: list[ReplayShardV3] = []
    example_count = 0
    required_row_fields = {
        "path",
        "sha256",
        "compressed_bytes",
        "records",
        "examples",
        "start_index",
    }
    for shard, row in zip(verified_shards, rows, strict=True):
        if not isinstance(row, dict) or set(row) != required_row_fields:
            return None
        if (
            row["path"] != shard.relative_path
            or row["sha256"] != shard.expected_sha256
            or row["compressed_bytes"] != shard.expected_bytes
            or row["records"] != shard.expected_records
            or not isinstance(row["examples"], int)
            or row["examples"] < 0
            or row["start_index"] != example_count
        ):
            return None
        if row["examples"] > 0:
            shards.append(
                ReplayShardV3(
                    path=shard.path,
                    relative_path=shard.relative_path,
                    expected_sha256=shard.expected_sha256,
                    expected_bytes=shard.expected_bytes,
                    expected_records=shard.expected_records,
                    example_count=row["examples"],
                    start_index=example_count,
                )
            )
        example_count += row["examples"]
    if example_count <= 0:
        return None
    return (
        shards,
        example_count,
        (ruleset["id"], ruleset["manifest_sha256"]),
        frozenset(vocabulary_tokens),
    )


def _write_replay_index(
    index_path: Path,
    manifest_sha256: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    ruleset: tuple[str, str],
    vocabulary_tokens: frozenset[str],
) -> None:
    payload = {
        "schema_id": REPLAY_INDEX_SCHEMA_ID,
        "schema_versions": _index_schema_versions(),
        "parser_version": REPLAY_PARSER_VERSION,
        "verifier_version": REPLAY_VERIFIER_VERSION,
        "code_version": _loader_code_version(),
        "manifest_sha256": manifest_sha256,
        "split": split,
        "shards": list(rows),
        "ruleset": {"id": ruleset[0], "manifest_sha256": ruleset[1]},
        "vocabulary_tokens": sorted(vocabulary_tokens),
    }
    temporary = index_path.with_name(
        f"{index_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, index_path)
    finally:
        temporary.unlink(missing_ok=True)


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

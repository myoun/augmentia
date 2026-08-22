"""Content-addressed, pre-tensorized V3 replay artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import uuid
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as functional
import zstandard

from .replay import (
    ReplayDatasetV3,
    ReplayShardV3,
    TrainingBatchV3,
    TrainingExampleV3,
    _parse_record,
    collate_examples,
)
from .schema import SchemaError
from .tensorize import V3Batch, V3Vocabulary

PRETENSOR_MANIFEST_SCHEMA_ID = "augmentia.pretensor-manifest.v3.1"
PRETENSOR_FRAME_SCHEMA_ID = "augmentia.pretensor-frame.v3.1"
PRETENSOR_FORMAT_VERSION = "augmentia.pretensor-packed-batch.v3.1"

_LONG_INPUT_FIELDS = {
    field.name
    for field in fields(V3Batch)
    if field.name
    not in {
        "spatial",
        "global_features",
        "state_text_position",
        "state_text_mask",
        "entity_counts",
        "piece_numeric",
        "piece_mask",
        "card_numeric",
        "card_mask",
        "phase_numeric",
        "phase_text_position",
        "phase_text_mask",
        "runtime_numeric",
        "runtime_mask",
        "action_target_squares",
        "action_target_sequence_position",
        "action_target_sequence_mask",
        "action_numeric",
        "action_mask",
    }
}
_NEGATIVE_PAD_FIELDS = {
    "piece_slot",
    "card_slot",
    "phase_piece_ref",
    "phase_card_ref",
    "runtime_piece_ref",
    "runtime_card_ref",
    "runtime_parent",
    "action_piece_ref",
    "action_card_ref",
}


@dataclass(frozen=True)
class PretensorizedIdentityV3:
    artifact_manifest_path: str
    artifact_manifest_sha256: str
    source: dict[str, Any]
    tensorizer_code_version: str
    vocabulary_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "identity_kind": "pretensorized_v3",
            "artifact_manifest_path": self.artifact_manifest_path,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "source": self.source,
            "tensorizer_code_version": self.tensorizer_code_version,
            "vocabulary_sha256": self.vocabulary_sha256,
        }


@dataclass(frozen=True)
class PretensorFrameV3:
    artifact_path: Path
    artifact_relative_path: str
    source_shard_sha256: str
    offset: int
    compressed_bytes: int
    sha256: str
    examples: int
    start_index: int


@dataclass(frozen=True)
class PretensorizedExampleRefV3:
    dataset: "PretensorizedReplayDatasetV3"
    index: int


class PretensorizedReplayDatasetV3(Sequence[PretensorizedExampleRefV3]):
    """Random-access replay dataset backed by independently compressed tensor frames."""

    def __init__(self, manifest_path: str | Path, *, max_cached_frames: int = 2):
        if max_cached_frames <= 0:
            raise ValueError("max_cached_frames must be positive")
        path = Path(manifest_path).resolve()
        raw = path.read_bytes()
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SchemaError(f"invalid pre-tensor manifest JSON: {error}") from error
        _validate_pretensor_manifest(manifest)
        if manifest["tensorizer_code_version"] != _tensorizer_code_version():
            raise SchemaError("pre-tensor artifact code version is stale")
        vocabulary = V3Vocabulary.from_json(manifest["vocabulary"])
        vocabulary_sha256 = _vocabulary_sha256(vocabulary)
        if vocabulary_sha256 != manifest["vocabulary_sha256"]:
            raise SchemaError("pre-tensor vocabulary checksum mismatch")

        frames: list[PretensorFrameV3] = []
        sampler_groups: list[int] = []
        example_count = 0
        for shard in manifest["shards"]:
            artifact_path = _resolve_artifact(path.parent, shard["path"])
            if artifact_path.stat().st_size != shard["bytes"]:
                raise SchemaError(f"pre-tensor byte count mismatch for {shard['path']}")
            if _sha256_file(artifact_path) != shard["sha256"]:
                raise SchemaError(f"pre-tensor checksum mismatch for {shard['path']}")
            previous_end = 0
            for frame in shard["frames"]:
                if frame["offset"] != previous_end:
                    raise SchemaError("pre-tensor frame offsets are not contiguous")
                if frame["start_index"] != example_count:
                    raise SchemaError("pre-tensor example offsets are not contiguous")
                frames.append(
                    PretensorFrameV3(
                        artifact_path=artifact_path,
                        artifact_relative_path=shard["path"],
                        source_shard_sha256=shard["source_shard_sha256"],
                        offset=frame["offset"],
                        compressed_bytes=frame["compressed_bytes"],
                        sha256=frame["sha256"],
                        examples=frame["examples"],
                        start_index=example_count,
                    )
                )
                sampler_groups.append(frame["examples"])
                example_count += frame["examples"]
                previous_end += frame["compressed_bytes"]
            if previous_end != shard["bytes"]:
                raise SchemaError("pre-tensor frames do not cover their artifact shard")
        if example_count != manifest["totals"]["examples"]:
            raise SchemaError("pre-tensor example total mismatch")

        self.manifest = manifest
        self.frames = tuple(frames)
        self._frame_ends = tuple(
            frame.start_index + frame.examples for frame in self.frames
        )
        self._sampler_group_sizes = tuple(sampler_groups)
        self._example_count = example_count
        self._cache: OrderedDict[int, TrainingBatchV3] = OrderedDict()
        self._max_cached_frames = max_cached_frames
        self.vocabulary = vocabulary
        self.vocabulary_tokens = frozenset(vocabulary.tokens[1:])
        self.identity = PretensorizedIdentityV3(
            artifact_manifest_path=str(path),
            artifact_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            source=dict(manifest["source_identity"]),
            tensorizer_code_version=manifest["tensorizer_code_version"],
            vocabulary_sha256=vocabulary_sha256,
        )

    def __len__(self) -> int:
        return self._example_count

    def __getitem__(self, index: int) -> PretensorizedExampleRefV3:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return PretensorizedExampleRefV3(self, index)

    @property
    def sampler_group_sizes(self) -> tuple[int, ...]:
        return self._sampler_group_sizes

    def shard_index(self, index: int) -> int:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return bisect_right(self._frame_ends, index)

    def build_vocabulary(self) -> V3Vocabulary:
        return self.vocabulary

    def clear_cache(self) -> None:
        self._cache.clear()

    def collate_indices(self, indices: Sequence[int]) -> TrainingBatchV3:
        if not indices:
            raise ValueError("cannot collate an empty batch")
        grouped: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
        for output_index, index in enumerate(indices):
            frame_index = self.shard_index(index)
            local_index = index - self.frames[frame_index].start_index
            grouped.setdefault(frame_index, []).append((output_index, local_index))
        pieces: list[tuple[list[int], TrainingBatchV3]] = []
        for frame_index, rows in grouped.items():
            batch = self._load_frame(frame_index)
            local_rows = torch.tensor([row[1] for row in rows], dtype=torch.long)
            pieces.append(
                ([row[0] for row in rows], _select_training_batch(batch, local_rows))
            )
        merged = _merge_training_batches([piece[1] for piece in pieces])
        current_order = [index for piece in pieces for index in piece[0]]
        restore = torch.tensor(
            [current_order.index(index) for index in range(len(indices))],
            dtype=torch.long,
        )
        return _select_training_batch(merged, restore)

    def _load_frame(self, frame_index: int) -> TrainingBatchV3:
        cached = self._cache.pop(frame_index, None)
        if cached is not None:
            self._cache[frame_index] = cached
            return cached
        while len(self._cache) >= self._max_cached_frames:
            self._cache.popitem(last=False)
        frame = self.frames[frame_index]
        with frame.artifact_path.open("rb") as source:
            source.seek(frame.offset)
            compressed = source.read(frame.compressed_bytes)
        if len(compressed) != frame.compressed_bytes:
            raise SchemaError("truncated pre-tensor frame")
        if hashlib.sha256(compressed).hexdigest() != frame.sha256:
            raise SchemaError("pre-tensor frame checksum mismatch")
        try:
            raw = zstandard.ZstdDecompressor().decompress(compressed)
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        except (zstandard.ZstdError, RuntimeError, EOFError) as error:
            raise SchemaError(f"invalid pre-tensor frame: {error}") from error
        batch = _decode_frame(payload, frame)
        self._cache[frame_index] = batch
        return batch


def build_pretensor_artifact(
    source_manifest: str | Path,
    output_root: str | Path,
    *,
    split: str = "train",
    chunk_size: int = 256,
    workers: int | None = None,
    progress_callback: Any = None,
) -> Path:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    dataset = ReplayDatasetV3(source_manifest, split=split)
    vocabulary = dataset.build_vocabulary()
    code_version = _tensorizer_code_version()
    vocabulary_sha256 = _vocabulary_sha256(vocabulary)
    source_identity = dataset.identity.to_json()
    content_key = hashlib.sha256(
        json.dumps(
            {
                "source": source_identity,
                "split": split,
                "chunk_size": chunk_size,
                "code_version": code_version,
                "vocabulary_sha256": vocabulary_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    safe_dataset_id = dataset.identity.dataset_id.replace("/", "-").replace(":", "-")
    target = (
        Path(output_root).resolve() / f"{safe_dataset_id}.{split}.{content_key[:16]}"
    )
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        PretensorizedReplayDatasetV3(manifest_path)
        return manifest_path

    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    selected_workers = min(workers or (os.cpu_count() or 1), len(dataset.shards))
    try:
        tasks = [
            (
                shard_index,
                shard,
                temporary,
                vocabulary.to_json(),
                vocabulary_sha256,
                chunk_size,
            )
            for shard_index, shard in enumerate(dataset.shards)
        ]
        rows: list[dict[str, Any] | None] = [None] * len(tasks)
        if selected_workers == 1:
            for completed, task in enumerate(tasks, 1):
                shard_index, row = _build_artifact_shard(task)
                rows[shard_index] = row
                if progress_callback is not None:
                    progress_callback(completed, len(tasks), row["path"])
        else:
            with ProcessPoolExecutor(max_workers=selected_workers) as executor:
                futures = {
                    executor.submit(_build_artifact_shard, task): task[0]
                    for task in tasks
                }
                for completed, future in enumerate(as_completed(futures), 1):
                    shard_index, row = future.result()
                    rows[shard_index] = row
                    if progress_callback is not None:
                        progress_callback(completed, len(tasks), row["path"])
        if any(row is None for row in rows):
            raise RuntimeError("pre-tensor generation returned incomplete shards")
        final_rows = [row for row in rows if row is not None]
        start_index = 0
        frame_count = 0
        total_bytes = 0
        for row in final_rows:
            for frame in row["frames"]:
                frame["start_index"] = start_index
                start_index += frame["examples"]
                frame_count += 1
            total_bytes += row["bytes"]
        if start_index != len(dataset):
            raise RuntimeError("pre-tensor generation changed the example count")
        manifest = {
            "artifact_schema_id": PRETENSOR_MANIFEST_SCHEMA_ID,
            "format_version": PRETENSOR_FORMAT_VERSION,
            "tensorizer_code_version": code_version,
            "source_identity": source_identity,
            "split": split,
            "chunk_size": chunk_size,
            "vocabulary": vocabulary.to_json(),
            "vocabulary_sha256": vocabulary_sha256,
            "shards": final_rows,
            "totals": {
                "examples": start_index,
                "frames": frame_count,
                "bytes": total_bytes,
            },
        }
        _atomic_json_write(temporary / "manifest.json", manifest)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    PretensorizedReplayDatasetV3(manifest_path)
    return manifest_path


def _build_artifact_shard(
    task: tuple[int, ReplayShardV3, Path, Mapping[str, Any], str, int],
) -> tuple[int, dict[str, Any]]:
    shard_index, shard, root, vocabulary_json, vocabulary_sha256, chunk_size = task
    torch.set_num_threads(1)
    vocabulary = V3Vocabulary.from_json(vocabulary_json)
    relative_path = f"shards/tensor-{shard_index:05d}.ptz"
    output_path = _resolve_artifact(root, relative_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rulesets: set[tuple[str, str]] = set()
    frames_json: list[dict[str, Any]] = []
    pending: list[TrainingExampleV3] = []
    offset = 0
    example_count = 0
    compressor = zstandard.ZstdCompressor(level=3)
    with output_path.open("xb") as output:
        for example in _iter_source_examples(shard, rulesets):
            pending.append(example)
            if len(pending) == chunk_size:
                frame = _encode_frame(pending, vocabulary, vocabulary_sha256, shard)
                compressed = compressor.compress(frame)
                output.write(compressed)
                frames_json.append(_frame_row(offset, compressed, len(pending)))
                offset += len(compressed)
                example_count += len(pending)
                pending.clear()
        if pending:
            frame = _encode_frame(pending, vocabulary, vocabulary_sha256, shard)
            compressed = compressor.compress(frame)
            output.write(compressed)
            frames_json.append(_frame_row(offset, compressed, len(pending)))
            offset += len(compressed)
            example_count += len(pending)
        output.flush()
        os.fsync(output.fileno())
    if example_count != shard.example_count:
        raise SchemaError(
            f"pre-tensor example count mismatch for {shard.relative_path}"
        )
    return shard_index, {
        "path": relative_path,
        "source_path": shard.relative_path,
        "source_shard_sha256": shard.expected_sha256,
        "bytes": offset,
        "sha256": _sha256_file(output_path),
        "examples": example_count,
        "frames": frames_json,
    }


def _iter_source_examples(
    shard: ReplayShardV3, rulesets: set[tuple[str, str]]
) -> Iterator[TrainingExampleV3]:
    try:
        with shard.path.open("rb") as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                with io.BufferedReader(reader) as buffered:
                    for line_number, raw in enumerate(buffered, 1):
                        try:
                            record = json.loads(raw)
                        except json.JSONDecodeError as error:
                            raise SchemaError(
                                f"{shard.relative_path}:{line_number}: invalid JSON"
                            ) from error
                        yield from _parse_record(
                            record, f"{shard.relative_path}:{line_number}", rulesets
                        )
    except (OSError, zstandard.ZstdError) as error:
        raise SchemaError(f"cannot read {shard.relative_path}: {error}") from error


def _encode_frame(
    examples: Sequence[TrainingExampleV3],
    vocabulary: V3Vocabulary,
    vocabulary_sha256: str,
    shard: ReplayShardV3,
) -> bytes:
    batch = collate_examples(examples, vocabulary)
    inputs = {}
    for field in fields(V3Batch):
        tensor = getattr(batch.inputs, field.name).contiguous()
        if field.name in _LONG_INPUT_FIELDS:
            if tensor.numel() and (
                int(tensor.min()) < -(2**31) or int(tensor.max()) >= 2**31
            ):
                raise SchemaError(f"{field.name} does not fit pre-tensor int32")
            tensor = tensor.to(torch.int32)
        elif field.name == "action_target_squares":
            if tensor.numel() and not torch.all((tensor == 0) | (tensor == 1)):
                raise SchemaError("action target bitset is not binary")
            tensor = tensor.to(torch.uint8)
        inputs[field.name] = tensor
    payload = {
        "frame_schema_id": PRETENSOR_FRAME_SCHEMA_ID,
        "format_version": PRETENSOR_FORMAT_VERSION,
        "source_shard_sha256": shard.expected_sha256,
        "vocabulary_sha256": vocabulary_sha256,
        "inputs": inputs,
        "policy_targets": batch.policy_targets,
        "policy_supervision": batch.policy_supervision,
        "value_targets": batch.value_targets,
        "value_supervision": batch.value_supervision,
        "auxiliary_targets": batch.auxiliary_targets,
        "auxiliary_supervision": batch.auxiliary_supervision,
        "decision_ids": batch.decision_ids,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _decode_frame(payload: Any, frame: PretensorFrameV3) -> TrainingBatchV3:
    required = {
        "frame_schema_id",
        "format_version",
        "source_shard_sha256",
        "vocabulary_sha256",
        "inputs",
        "policy_targets",
        "policy_supervision",
        "value_targets",
        "value_supervision",
        "auxiliary_targets",
        "auxiliary_supervision",
        "decision_ids",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise SchemaError("pre-tensor frame fields do not match the contract")
    if (
        payload["frame_schema_id"] != PRETENSOR_FRAME_SCHEMA_ID
        or payload["format_version"] != PRETENSOR_FORMAT_VERSION
        or payload["source_shard_sha256"] != frame.source_shard_sha256
    ):
        raise SchemaError("pre-tensor frame identity mismatch")
    encoded_inputs = payload["inputs"]
    if not isinstance(encoded_inputs, dict) or set(encoded_inputs) != {
        field.name for field in fields(V3Batch)
    }:
        raise SchemaError("pre-tensor input fields do not match V3Batch")
    decoded = {}
    for field in fields(V3Batch):
        tensor = encoded_inputs[field.name]
        if not isinstance(tensor, torch.Tensor):
            raise SchemaError(f"pre-tensor input {field.name} is not a tensor")
        if field.name in _LONG_INPUT_FIELDS:
            tensor = tensor.long()
        elif field.name == "action_target_squares":
            tensor = tensor.float()
        decoded[field.name] = tensor
    batch = TrainingBatchV3(
        inputs=V3Batch(**decoded),
        policy_targets=payload["policy_targets"],
        policy_supervision=payload["policy_supervision"],
        value_targets=payload["value_targets"],
        value_supervision=payload["value_supervision"],
        auxiliary_targets=payload["auxiliary_targets"],
        auxiliary_supervision=payload["auxiliary_supervision"],
        decision_ids=tuple(payload["decision_ids"]),
    )
    if (
        batch.inputs.batch_size != frame.examples
        or len(batch.decision_ids) != frame.examples
    ):
        raise SchemaError("pre-tensor frame example count mismatch")
    return batch


def _select_training_batch(
    batch: TrainingBatchV3, rows: torch.Tensor
) -> TrainingBatchV3:
    return TrainingBatchV3(
        inputs=V3Batch(
            **{
                field.name: getattr(batch.inputs, field.name).index_select(0, rows)
                for field in fields(V3Batch)
            }
        ),
        policy_targets=batch.policy_targets.index_select(0, rows),
        policy_supervision=batch.policy_supervision.index_select(0, rows),
        value_targets=batch.value_targets.index_select(0, rows),
        value_supervision=batch.value_supervision.index_select(0, rows),
        auxiliary_targets=batch.auxiliary_targets.index_select(0, rows),
        auxiliary_supervision=batch.auxiliary_supervision.index_select(0, rows),
        decision_ids=tuple(batch.decision_ids[int(row)] for row in rows),
    )


def _merge_training_batches(batches: Sequence[TrainingBatchV3]) -> TrainingBatchV3:
    if len(batches) == 1:
        return batches[0]
    inputs = {}
    for field in fields(V3Batch):
        tensors = [getattr(batch.inputs, field.name) for batch in batches]
        inputs[field.name] = _pad_and_cat(
            tensors, -1 if field.name in _NEGATIVE_PAD_FIELDS else 0
        )
    return TrainingBatchV3(
        inputs=V3Batch(**inputs),
        policy_targets=_pad_and_cat([batch.policy_targets for batch in batches], 0),
        policy_supervision=torch.cat([batch.policy_supervision for batch in batches]),
        value_targets=torch.cat([batch.value_targets for batch in batches]),
        value_supervision=torch.cat([batch.value_supervision for batch in batches]),
        auxiliary_targets=torch.cat([batch.auxiliary_targets for batch in batches]),
        auxiliary_supervision=torch.cat(
            [batch.auxiliary_supervision for batch in batches]
        ),
        decision_ids=tuple(
            identifier for batch in batches for identifier in batch.decision_ids
        ),
    )


def _pad_and_cat(tensors: Sequence[torch.Tensor], value: int | float) -> torch.Tensor:
    rank = tensors[0].ndim
    target = [
        max(tensor.shape[dimension] for tensor in tensors)
        for dimension in range(1, rank)
    ]
    padded = []
    for tensor in tensors:
        padding = []
        for dimension in range(rank - 1, 0, -1):
            padding.extend((0, target[dimension - 1] - tensor.shape[dimension]))
        padded.append(
            functional.pad(tensor, padding, value=value) if any(padding) else tensor
        )
    return torch.cat(padded, dim=0)


def _frame_row(offset: int, compressed: bytes, examples: int) -> dict[str, Any]:
    return {
        "offset": offset,
        "compressed_bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "examples": examples,
        "start_index": 0,
    }


def _tensorizer_code_version() -> str:
    digest = hashlib.sha256()
    for source in (
        Path(__file__),
        Path(__file__).with_name("tensorize.py"),
        Path(__file__).with_name("replay.py"),
    ):
        digest.update(source.name.encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _vocabulary_sha256(vocabulary: V3Vocabulary) -> str:
    return hashlib.sha256(
        json.dumps(vocabulary.to_json(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise SchemaError(f"artifact path escapes its root: {relative}") from error
    return candidate


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _validate_pretensor_manifest(payload: Any) -> None:
    required = {
        "artifact_schema_id",
        "format_version",
        "tensorizer_code_version",
        "source_identity",
        "split",
        "chunk_size",
        "vocabulary",
        "vocabulary_sha256",
        "shards",
        "totals",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise SchemaError("pre-tensor manifest fields do not match the contract")
    if (
        payload["artifact_schema_id"] != PRETENSOR_MANIFEST_SCHEMA_ID
        or payload["format_version"] != PRETENSOR_FORMAT_VERSION
    ):
        raise SchemaError("unsupported pre-tensor artifact format")
    if not isinstance(payload["shards"], list) or not payload["shards"]:
        raise SchemaError("pre-tensor manifest contains no shards")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    started = __import__("time").perf_counter()

    def progress(completed: int, total: int, shard: str) -> None:
        print(
            json.dumps(
                {
                    "completed_shards": completed,
                    "total_shards": total,
                    "shard": shard,
                    "elapsed_seconds": __import__("time").perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = build_pretensor_artifact(
        args.source_manifest,
        args.output_root,
        split=args.split,
        chunk_size=args.chunk_size,
        workers=args.workers,
        progress_callback=progress,
    )
    print(result)


if __name__ == "__main__":
    main()

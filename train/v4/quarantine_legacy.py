"""Fail-closed, non-destructive quarantine for legacy V4 target artifacts.

The scanner treats replay v4 / target v2 artifacts as historical evidence. It
never moves, deletes, copies, or overwrites artifact bytes. A central registry
can be created outside the artifact root, and an optional root marker is created
only when the caller explicitly requests it. Existing files are accepted only
when byte-identical, making repeated quarantine runs idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .dataset import (
    DATASET_INDEX_SCHEMA_ID,
    LEGACY_FACTORIZED_TARGET_SCHEMA_ID,
    LEGACY_REPLAY_SCHEMA_ID,
    RUN_MANIFEST_SCHEMA_ID,
    VALIDATION_SCHEMA_ID,
)
from .replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    GAME_RECORD_V5_SCHEMA_ID,
    LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
    LEGACY_GAME_RECORD_V4_SCHEMA_ID,
)

REGISTRY_SCHEMA_ID = "augmentia.v4-legacy-quarantine-registry.v1"
ROOT_MARKER_SCHEMA_ID = "augmentia.v4-legacy-quarantine-root-marker.v1"
ROOT_MARKER_NAME = ".augmentia-v4-legacy-quarantine.json"
QUARANTINE_LABEL = "legacy/non-promotable/contaminated-target-semantics"
LEGACY_DATASET_INDEX_SCHEMA_ID = "augmentia.self-play-v4-dataset-index.v1"
LEGACY_CHECKPOINT_SCHEMAS = {
    "augmentia.v4-trainer-checkpoint.v1",
    "augmentia.v4-trainer-checkpoint.v2",
}
CURRENT_CHECKPOINT_SCHEMA_ID = "augmentia.v4-trainer-checkpoint.v3"
BUNDLE_METADATA_SCHEMA_ID = "augmentia.onnx-factorized-policy-bundle.v1"
RULESET_ID = "augment_chess_20260915"


class LegacyQuarantineError(RuntimeError):
    """Raised when quarantine would be ambiguous, destructive, or incomplete."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LegacyQuarantineError(f"cannot read JSON artifact {source}: {error}") from error
    if not isinstance(payload, dict):
        raise LegacyQuarantineError(f"JSON artifact must be an object: {source}")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LegacyQuarantineError(f"{label} is not a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise LegacyQuarantineError(f"{label} is not hexadecimal SHA-256") from error
    return value.lower()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise LegacyQuarantineError(f"artifact escapes root: {path} not under {root}") from error


def _safe_path(root: Path, value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise LegacyQuarantineError(f"unsafe {label} path: {value}")
    resolved = (root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise LegacyQuarantineError(f"{label} path escapes artifact root: {value}") from error
    return resolved


def _file_record(
    path: Path,
    root: Path,
    *,
    declared_sha256: str | None = None,
    source: str,
    hash_content: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise LegacyQuarantineError(f"artifact is missing: {_relative(path, root)}")
    declared = (
        _require_sha256(declared_sha256, f"{_relative(path, root)} SHA-256")
        if declared_sha256 is not None
        else None
    )
    if source == "content" or hash_content:
        if declared is not None and source != "content":
            actual = sha256_file(path)
            if actual != declared:
                raise LegacyQuarantineError(
                    f"declared SHA-256 mismatch for {_relative(path, root)}"
                )
            digest = declared
        else:
            digest = sha256_file(path)
    else:
        if declared is None:
            raise LegacyQuarantineError(
                f"declared SHA-256 missing for {_relative(path, root)}"
            )
        digest = declared
    return {
        "path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "sha256_source": source,
    }


def _classification() -> dict[str, Any]:
    return {
        "label": QUARANTINE_LABEL,
        "lifecycle": "legacy",
        "promotable": False,
        "reason": "contaminated-target-semantics",
        "legacy_replay_schema_id": LEGACY_GAME_RECORD_V4_SCHEMA_ID,
        "legacy_factorized_target_schema_id": LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
        "allowed_uses": [
            "historical-analysis",
            "legacy-all-open-ab-control",
            "reproduction-audit",
        ],
        "forbidden_uses": [
            "promotion",
            "web-default",
            "self-play-producer",
            "opened-only-training",
            "opened-only-resume",
        ],
    }


def _manifest_lineage(payload: Mapping[str, Any], path: Path) -> str:
    if payload.get("manifest_schema_id") != RUN_MANIFEST_SCHEMA_ID:
        raise LegacyQuarantineError(f"unknown V4 run manifest schema: {path}")
    lineage = (
        payload.get("replay_schema_id"),
        payload.get("factorized_target_schema_id"),
    )
    if lineage == (
        LEGACY_GAME_RECORD_V4_SCHEMA_ID,
        LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
    ):
        return "legacy"
    if lineage == (
        GAME_RECORD_V5_SCHEMA_ID,
        FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    ):
        return "current"
    raise LegacyQuarantineError(f"unknown or mixed replay/target lineage: {path}")


def _game_merkle(games: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "game_index": item.get("game_index"),
                    "file_sha256": item.get("file_sha256"),
                }
                for item in games
            ]
        )
    )


def _scan_corpora(
    root: Path, *, hash_games: bool
) -> tuple[list[dict[str, Any]], dict[str, set[str]], set[str]]:
    legacy: list[dict[str, Any]] = []
    manifest_hashes: dict[str, set[str]] = {"legacy": set(), "current": set()}
    dataset_ids: dict[str, set[str]] = {"legacy": set(), "current": set()}
    for manifest_path in sorted(root.rglob("manifest.json")):
        payload = load_json(manifest_path)
        lineage = _manifest_lineage(payload, manifest_path)
        manifest_hashes[lineage].add(sha256_file(manifest_path))
        dataset_id = payload.get("run_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise LegacyQuarantineError(f"run manifest has no run_id: {manifest_path}")
        dataset_ids[lineage].add(dataset_id)
        if lineage == "current":
            continue
        games = payload.get("games_artifacts")
        if not isinstance(games, list) or not games:
            raise LegacyQuarantineError(f"legacy corpus has no games: {manifest_path}")
        game_records: list[dict[str, Any]] = []
        seen_indexes: set[int] = set()
        for item in games:
            if not isinstance(item, Mapping):
                raise LegacyQuarantineError(f"legacy game entry is invalid: {manifest_path}")
            index = item.get("game_index")
            relative = item.get("relative_path")
            declared = _require_sha256(
                item.get("file_sha256"), f"game {dataset_id}/{index} SHA-256"
            )
            if not isinstance(index, int) or isinstance(index, bool) or index in seen_indexes:
                raise LegacyQuarantineError(f"legacy game index is invalid: {manifest_path}")
            if not isinstance(relative, str) or not relative:
                raise LegacyQuarantineError(f"legacy game path is invalid: {manifest_path}")
            seen_indexes.add(index)
            game_path = _safe_path(root, f"{_relative(manifest_path.parent, root)}/{relative}", "game")
            game_records.append(
                _file_record(
                    game_path,
                    root,
                    declared_sha256=declared,
                    source="run-manifest",
                    hash_content=hash_games,
                )
            )
        ordered_indexes = sorted(seen_indexes)
        expected_indexes = set(
            range(ordered_indexes[0], ordered_indexes[0] + len(ordered_indexes))
        )
        if seen_indexes != expected_indexes:
            raise LegacyQuarantineError(f"legacy game indexes are not contiguous: {manifest_path}")
        if payload.get("game_index_start") not in (None, ordered_indexes[0]):
            raise LegacyQuarantineError(
                f"legacy game_index_start does not match the shard: {manifest_path}"
            )
        validation_path = manifest_path.parent / "validation.json"
        if not validation_path.is_file():
            raise LegacyQuarantineError(f"legacy corpus validation is missing: {validation_path}")
        validation = load_json(validation_path)
        if (
            validation.get("validation_schema_id") != VALIDATION_SCHEMA_ID
            or validation.get("run_id") != dataset_id
            or validation.get("games") != len(games)
        ):
            raise LegacyQuarantineError(f"legacy corpus validation mismatch: {validation_path}")
        legacy.append(
            {
                "run_id": dataset_id,
                "path": _relative(manifest_path.parent, root),
                "run_kind": payload.get("run_kind"),
                "legacy_reason": "replay-v4-target-v2-missing-opened-masks",
                "ruleset_id": payload.get("ruleset", {}).get("id"),
                "engine_commit": payload.get("engine_commit"),
                "source_seal_tag": payload.get("source_seal_tag"),
                "source_seal_commit": payload.get("source_seal_commit"),
                "replay_schema_id": payload.get("replay_schema_id"),
                "factorized_target_schema_id": payload.get(
                    "factorized_target_schema_id"
                ),
                "producer": {
                    "evaluator": payload.get("evaluator"),
                    "base_seed": payload.get("base_seed"),
                    "seed_derivation": payload.get("seed_derivation"),
                    "simulations": payload.get("simulations"),
                    "chance_outcome_samples": payload.get("chance_outcome_samples"),
                    "temperature": payload.get("temperature"),
                    "temperature_decisions": payload.get("temperature_decisions"),
                    "max_decisions": payload.get("max_decisions"),
                    "workers": payload.get("workers"),
                    "execution_provider": payload.get("execution_provider"),
                    "intra_threads": payload.get("intra_threads"),
                    "search_semantics_id": payload.get("search_semantics_id"),
                    "root_selection_id": payload.get("root_selection_id"),
                    "chance_sampling_id": payload.get("chance_sampling_id"),
                },
                "games": len(games),
                "game_hash_merkle_sha256": _game_merkle(games),
                "files": [
                    _file_record(
                        manifest_path,
                        root,
                        source="content",
                        hash_content=True,
                    ),
                    _file_record(
                        validation_path,
                        root,
                        source="content",
                        hash_content=True,
                    ),
                    *game_records,
                ],
            }
        )
    if not legacy:
        raise LegacyQuarantineError(f"no legacy V4 corpus found under {root}")
    return sorted(legacy, key=lambda item: item["run_id"]), manifest_hashes, dataset_ids


def _resolve_index_source(index_path: Path, value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise LegacyQuarantineError(f"dataset index source path is invalid: {index_path}")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (index_path.parent / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise LegacyQuarantineError(f"dataset index source escapes root: {resolved}") from error
    return resolved


def _scan_dataset_indexes(
    root: Path,
    *,
    manifest_hashes: Mapping[str, set[str]],
    dataset_ids: dict[str, set[str]],
) -> list[dict[str, Any]]:
    indexes: list[dict[str, Any]] = []
    for index_path in sorted(root.rglob("dataset.json")):
        payload = load_json(index_path)
        schema = payload.get("dataset_schema_id")
        if schema == LEGACY_DATASET_INDEX_SCHEMA_ID:
            lineage = "legacy"
        elif schema == DATASET_INDEX_SCHEMA_ID:
            current_pair = (
                payload.get("replay_schema_id"),
                payload.get("factorized_target_schema_id"),
            )
            if current_pair != (
                GAME_RECORD_V5_SCHEMA_ID,
                FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
            ):
                raise LegacyQuarantineError(f"unknown current dataset lineage: {index_path}")
            lineage = "current"
        else:
            raise LegacyQuarantineError(f"unknown dataset index schema: {index_path}")
        dataset_id = payload.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise LegacyQuarantineError(f"dataset index has no dataset_id: {index_path}")
        dataset_ids[lineage].add(dataset_id)
        sources = payload.get("manifests")
        if not isinstance(sources, list) or not sources:
            raise LegacyQuarantineError(f"dataset index has no sources: {index_path}")
        source_records: list[dict[str, Any]] = []
        for source in sources:
            if not isinstance(source, Mapping):
                raise LegacyQuarantineError(f"dataset source is invalid: {index_path}")
            source_path = _resolve_index_source(index_path, source.get("path"), root)
            declared = _require_sha256(source.get("sha256"), "dataset source SHA-256")
            if declared not in manifest_hashes["legacy"] | manifest_hashes["current"]:
                raise LegacyQuarantineError(
                    f"dataset source is not a scanned V4 manifest: {source_path}"
                )
            source_manifest = load_json(source_path)
            source_lineage = _manifest_lineage(source_manifest, source_path)
            if source_lineage != lineage:
                raise LegacyQuarantineError(
                    f"dataset index mixes {lineage} and {source_lineage} lineage: {index_path}"
                )
            source_records.append(
                {
                    "run_id": source_manifest.get("run_id"),
                    "path": _relative(source_path, root),
                    "sha256": declared,
                }
            )
        if lineage == "current":
            continue
        indexes.append(
            {
                "dataset_id": dataset_id,
                "path": _relative(index_path.parent, root),
                "schema_id": schema,
                "legacy_reason": "dataset-index-v1-with-only-replay-v4-target-v2-sources",
                "games": payload.get("games"),
                "game_index_start": payload.get("game_index_start"),
                "game_index_end": payload.get("game_index_end"),
                "source_manifests": source_records,
                "file": _file_record(
                    index_path,
                    root,
                    source="content",
                    hash_content=True,
                ),
            }
        )
    return sorted(indexes, key=lambda item: item["dataset_id"])


def _provenance_lineage(
    provenance: Mapping[str, Any],
    *,
    legacy_datasets: set[str],
    legacy_manifest_hashes: set[str],
) -> str:
    dataset_id = provenance.get("dataset_run_id")
    if isinstance(dataset_id, str) and dataset_id in legacy_datasets:
        return "legacy"
    sources = provenance.get("dataset_source_manifests", [])
    if isinstance(sources, list) and any(
        isinstance(item, Mapping) and item.get("sha256") in legacy_manifest_hashes
        for item in sources
    ):
        return "legacy"
    explicit = (
        provenance.get("replay_schema_id"),
        provenance.get("factorized_target_schema_id"),
    )
    if explicit == (
        GAME_RECORD_V5_SCHEMA_ID,
        FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    ):
        return "current"
    return "unknown"


def _checkpoint_path(sidecar_path: Path) -> Path:
    suffix = ".v4-checkpoint.json"
    if not sidecar_path.name.endswith(suffix):
        raise LegacyQuarantineError(f"invalid checkpoint sidecar name: {sidecar_path}")
    return sidecar_path.with_name(sidecar_path.name[: -len(suffix)] + ".pt")


def _scan_training_runs(
    root: Path,
    *,
    legacy_datasets: set[str],
    current_datasets: set[str],
    legacy_manifest_hashes: set[str],
    current_checkpoint_hashes: set[str],
    hash_binaries: bool,
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    grouped: dict[Path, list[tuple[Path, dict[str, Any]]]] = {}
    legacy_hashes: set[str] = set()
    for sidecar_path in sorted(root.rglob("*.v4-checkpoint.json")):
        payload = load_json(sidecar_path)
        schema = payload.get("schema_id")
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise LegacyQuarantineError(f"checkpoint provenance is missing: {sidecar_path}")
        ruleset_id = provenance.get("ruleset_id")
        if not isinstance(ruleset_id, str) or not ruleset_id:
            raise LegacyQuarantineError(f"checkpoint ruleset is missing: {sidecar_path}")
        if ruleset_id != RULESET_ID:
            continue
        if schema in LEGACY_CHECKPOINT_SCHEMAS:
            dataset_id = provenance.get("dataset_run_id")
            explicit = (
                provenance.get("replay_schema_id"),
                provenance.get("factorized_target_schema_id"),
            )
            if (
                isinstance(dataset_id, str)
                and dataset_id in current_datasets
            ) or explicit == (
                GAME_RECORD_V5_SCHEMA_ID,
                FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
            ):
                raise LegacyQuarantineError(
                    f"legacy checkpoint schema conflicts with current lineage: {sidecar_path}"
                )
            lineage = "legacy"
        elif schema == CURRENT_CHECKPOINT_SCHEMA_ID:
            lineage = _provenance_lineage(
                provenance,
                legacy_datasets=legacy_datasets,
                legacy_manifest_hashes=legacy_manifest_hashes,
            )
            if lineage != "current":
                raise LegacyQuarantineError(
                    f"current checkpoint schema has unknown dataset lineage: {sidecar_path}"
                )
            checkpoint_hash = payload.get("checkpoint_sha256")
            if isinstance(checkpoint_hash, str):
                current_checkpoint_hashes.add(checkpoint_hash)
            continue
        else:
            raise LegacyQuarantineError(f"unknown checkpoint schema: {sidecar_path}")
        checkpoint_hash = _require_sha256(
            payload.get("checkpoint_sha256"), f"{sidecar_path} checkpoint SHA-256"
        )
        legacy_hashes.add(checkpoint_hash)
        grouped.setdefault(sidecar_path.parent, []).append((sidecar_path, payload))

    if not grouped:
        raise LegacyQuarantineError(
            f"no legacy 0915 checkpoint sidecars found under {root}"
        )

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(grouped, key=lambda item: _relative(item, root)):
        checkpoints: list[dict[str, Any]] = []
        provenance: dict[str, Any] = {}
        for sidecar_path, payload in sorted(grouped[run_dir], key=lambda item: item[0].name):
            if not provenance:
                provenance = dict(payload["provenance"])
            checkpoint_path = _checkpoint_path(sidecar_path)
            checkpoint_hash = _require_sha256(
                payload.get("checkpoint_sha256"), "checkpoint SHA-256"
            )
            optimizer_name = payload.get("optimizer_path")
            if not isinstance(optimizer_name, str) or not optimizer_name:
                raise LegacyQuarantineError(f"optimizer path is missing: {sidecar_path}")
            optimizer_path = _safe_path(
                root,
                f"{_relative(run_dir, root)}/{optimizer_name}",
                "optimizer",
            )
            optimizer_hash = _require_sha256(
                payload.get("optimizer_sha256"), "optimizer SHA-256"
            )
            checkpoints.append(
                {
                    "sidecar_schema_id": payload.get("schema_id"),
                    "completed_epochs": payload.get("completed_epochs"),
                    "example_count": payload.get("example_count"),
                    "validation_example_count": payload.get("validation_example_count"),
                    "sidecar": _file_record(
                        sidecar_path,
                        root,
                        source="content",
                        hash_content=True,
                    ),
                    "checkpoint": _file_record(
                        checkpoint_path,
                        root,
                        declared_sha256=checkpoint_hash,
                        source="checkpoint-sidecar",
                        hash_content=hash_binaries,
                    ),
                    "optimizer": _file_record(
                        optimizer_path,
                        root,
                        declared_sha256=optimizer_hash,
                        source="checkpoint-sidecar",
                        hash_content=hash_binaries,
                    ),
                }
            )
        metadata_files = []
        for name in ("run.json", "metrics.json", "history.json"):
            path = run_dir / name
            if path.is_file():
                metadata_files.append(
                    _file_record(path, root, source="content", hash_content=True)
                )
        runs.append(
            {
                "run_id": run_dir.name,
                "path": _relative(run_dir, root),
                "legacy_reason": "trainer-checkpoint-schema-v1-or-v2-missing-v3-lineage",
                "dataset_run_id": provenance.get("dataset_run_id"),
                "dataset_manifest_sha256": provenance.get("dataset_manifest_sha256"),
                "dataset_input_schema_id": provenance.get("dataset_input_schema_id"),
                "ruleset_id": provenance.get("ruleset_id"),
                "encoder_kind": provenance.get("encoder_kind"),
                "training_config": provenance.get("training_config"),
                "checkpoints": checkpoints,
                "files": metadata_files,
            }
        )
    return sorted(runs, key=lambda item: item["path"]), legacy_hashes, current_checkpoint_hashes


def _scan_bundles(
    root: Path,
    *,
    legacy_datasets: set[str],
    current_datasets: set[str],
    legacy_checkpoints: set[str],
    current_checkpoints: set[str],
    hash_binaries: bool,
) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    for metadata_path in sorted(root.rglob("*.v4-bundle.metadata.json")):
        payload = load_json(metadata_path)
        if payload.get("metadata_schema_id") != BUNDLE_METADATA_SCHEMA_ID:
            raise LegacyQuarantineError(f"unknown V4 bundle metadata: {metadata_path}")
        if payload.get("ruleset_id") != RULESET_ID:
            continue
        dataset = payload.get("dataset")
        if not isinstance(dataset, Mapping):
            raise LegacyQuarantineError(f"bundle dataset provenance is missing: {metadata_path}")
        dataset_id = dataset.get("dataset_run_id")
        checkpoint_hash = _require_sha256(
            payload.get("checkpoint_sha256"), "bundle checkpoint SHA-256"
        )
        dataset_legacy = dataset_id in legacy_datasets
        dataset_current = dataset_id in current_datasets
        checkpoint_legacy = checkpoint_hash in legacy_checkpoints
        checkpoint_current = checkpoint_hash in current_checkpoints
        if checkpoint_legacy and dataset_current:
            raise LegacyQuarantineError(
                f"0915 bundle mixes current dataset and legacy checkpoint: {metadata_path}"
            )
        if dataset_legacy and checkpoint_current:
            raise LegacyQuarantineError(
                f"0915 bundle mixes legacy dataset and current checkpoint: {metadata_path}"
            )
        if checkpoint_legacy:
            lineage = "legacy"
            dataset_resolution = (
                "legacy-dataset-and-checkpoint"
                if dataset_legacy
                else "legacy-checkpoint-authoritative"
            )
        elif dataset_legacy or not (dataset_current and checkpoint_current):
            raise LegacyQuarantineError(
                f"0915 bundle has unknown dataset/checkpoint lineage: {metadata_path}"
            )
        else:
            continue
        graphs = []
        for section in ("state_encoder", "factorized_scorer"):
            graph = payload.get(section)
            if not isinstance(graph, Mapping):
                raise LegacyQuarantineError(f"bundle graph metadata is missing: {metadata_path}")
            graph_name = graph.get("path")
            if not isinstance(graph_name, str) or not graph_name:
                raise LegacyQuarantineError(f"bundle graph path is missing: {metadata_path}")
            graph_path = _safe_path(
                root,
                f"{_relative(metadata_path.parent, root)}/{graph_name}",
                "bundle graph",
            )
            graph_hash = _require_sha256(graph.get("sha256"), "bundle graph SHA-256")
            graphs.append(
                _file_record(
                    graph_path,
                    root,
                    declared_sha256=graph_hash,
                    source="bundle-metadata",
                    hash_content=hash_binaries,
                )
            )
        bundles.append(
            {
                "path": _relative(metadata_path.parent, root),
                "legacy_reason": "derived-from-legacy-dataset-or-checkpoint",
                "metadata_schema_id": payload.get("metadata_schema_id"),
                "dataset_run_id": dataset_id,
                "dataset_lineage_resolution": dataset_resolution,
                "dataset_manifest_sha256": dataset.get("dataset_manifest_sha256"),
                "checkpoint_sha256": checkpoint_hash,
                "code_commit": payload.get("code_commit"),
                "model_config": payload.get("model_config"),
                "metadata": _file_record(
                    metadata_path,
                    root,
                    source="content",
                    hash_content=True,
                ),
                "graphs": graphs,
            }
        )
    return sorted(bundles, key=lambda item: item["path"])


def _iter_file_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256_source"), str):
            yield value
        for child in value.values():
            yield from _iter_file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_file_records(child)


def _summary(core: Mapping[str, Any]) -> dict[str, int]:
    corpora = core.get("corpora", [])
    dataset_indexes = core.get("dataset_indexes", [])
    training_runs = core.get("training_runs", [])
    return {
        "corpora": len(corpora),
        "corpus_games": sum(int(item["games"]) for item in corpora),
        "dataset_indexes": len(dataset_indexes),
        "training_runs": len(training_runs),
        "checkpoints": sum(len(item["checkpoints"]) for item in training_runs),
        "bundles": len(core.get("bundles", [])),
    }


def build_registry(
    artifact_root: str | Path,
    *,
    hash_games: bool = False,
    hash_binaries: bool = False,
) -> dict[str, Any]:
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise LegacyQuarantineError(f"artifact root does not exist: {root}")
    corpora, manifest_hashes, dataset_ids = _scan_corpora(root, hash_games=hash_games)
    dataset_indexes = _scan_dataset_indexes(
        root,
        manifest_hashes=manifest_hashes,
        dataset_ids=dataset_ids,
    )
    legacy_datasets = set(dataset_ids["legacy"])
    training_runs, legacy_checkpoints, current_checkpoints = _scan_training_runs(
        root,
        legacy_datasets=legacy_datasets,
        current_datasets=set(dataset_ids["current"]),
        legacy_manifest_hashes=manifest_hashes["legacy"],
        current_checkpoint_hashes=set(),
        hash_binaries=hash_binaries,
    )
    bundles = _scan_bundles(
        root,
        legacy_datasets=legacy_datasets,
        current_datasets=set(dataset_ids["current"]),
        legacy_checkpoints=legacy_checkpoints,
        current_checkpoints=current_checkpoints,
        hash_binaries=hash_binaries,
    )
    core = {
        "schema_id": REGISTRY_SCHEMA_ID,
        "artifact_root": root.as_posix(),
        "classification": _classification(),
        "preservation_policy": {
            "move": "forbidden",
            "delete": "forbidden",
            "copy": "forbidden",
            "overwrite": "forbidden",
            "existing_identical_output": "accepted",
            "existing_different_output": "rejected",
        },
        "scan": {
            "hash_games": hash_games,
            "hash_binaries": hash_binaries,
            "unknown_lineage": "fail-closed",
        },
        "corpora": corpora,
        "dataset_indexes": dataset_indexes,
        "training_runs": training_runs,
        "bundles": bundles,
    }
    summary = _summary(core)
    sealed = {**core, "summary": summary}
    return {
        **sealed,
        "inventory_digest_sha256": sha256_bytes(canonical_json_bytes(sealed)),
    }


def _registry_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema_id",
            "artifact_root",
            "classification",
            "preservation_policy",
            "scan",
            "corpora",
            "dataset_indexes",
            "training_runs",
            "bundles",
            "summary",
        )
    }


def _json_file_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _json_file_bytes(payload).decode("utf-8")
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return True
    except FileExistsError:
        existing = load_json(path)
        if existing != payload:
            raise LegacyQuarantineError(
                f"refusing to overwrite different quarantine file: {path}"
            )
        return False


def root_marker_payload(registry_path: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": ROOT_MARKER_SCHEMA_ID,
        "classification": registry["classification"],
        "artifact_root": registry["artifact_root"],
        "registry_path": registry_path.resolve().as_posix(),
        "registry_sha256": sha256_bytes(_json_file_bytes(registry)),
        "inventory_digest_sha256": registry["inventory_digest_sha256"],
        "summary": registry["summary"],
    }


def quarantine(
    artifact_root: str | Path,
    registry_path: str | Path,
    *,
    write_root_marker: bool = False,
    hash_games: bool = False,
    hash_binaries: bool = False,
) -> dict[str, Any]:
    root = Path(artifact_root).resolve()
    output = Path(registry_path).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise LegacyQuarantineError("central registry must be outside the artifact root")
    registry = build_registry(
        root,
        hash_games=hash_games,
        hash_binaries=hash_binaries,
    )
    marker = root_marker_payload(output, registry) if write_root_marker else None
    # Preflight both existing destinations before creating either one.
    for path, expected in ((output, registry), ((root / ROOT_MARKER_NAME), marker)):
        if path.exists() and expected is not None and load_json(path) != expected:
            raise LegacyQuarantineError(
                f"refusing to overwrite different quarantine file: {path}"
            )
    registry_created = _write_new_or_identical(output, registry)
    marker_created = False
    if marker is not None:
        marker_created = _write_new_or_identical(root / ROOT_MARKER_NAME, marker)
    return {
        "registry": output,
        "registry_created": registry_created,
        "root_marker": root / ROOT_MARKER_NAME if marker is not None else None,
        "root_marker_created": marker_created,
        "summary": registry["summary"],
        "inventory_digest_sha256": registry["inventory_digest_sha256"],
    }


def verify_registry(
    registry_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    hash_games: bool = False,
    hash_binaries: bool = False,
) -> dict[str, Any]:
    source = Path(registry_path)
    payload = load_json(source)
    if payload.get("schema_id") != REGISTRY_SCHEMA_ID:
        raise LegacyQuarantineError("unsupported legacy quarantine registry schema")
    if payload.get("classification") != _classification():
        raise LegacyQuarantineError("quarantine classification drifted")
    core = _registry_core(payload)
    if payload.get("summary") != _summary(core):
        raise LegacyQuarantineError("quarantine registry summary is inconsistent")
    digest = sha256_bytes(canonical_json_bytes(core))
    if payload.get("inventory_digest_sha256") != digest:
        raise LegacyQuarantineError("quarantine inventory digest mismatch")
    root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else Path(str(payload.get("artifact_root")))
    )
    if not root.is_dir():
        raise LegacyQuarantineError(f"artifact root does not exist: {root}")
    checked = 0
    checked_declared = 0
    checked_games = 0
    for record in _iter_file_records(core):
        path = _safe_path(root, str(record["path"]), "registry")
        if not path.is_file():
            raise LegacyQuarantineError(f"registered artifact is missing: {record['path']}")
        if path.stat().st_size != record.get("bytes"):
            raise LegacyQuarantineError(f"registered artifact size changed: {record['path']}")
        source_kind = record.get("sha256_source")
        should_hash = source_kind == "content" or (
            hash_binaries and source_kind in {"checkpoint-sidecar", "bundle-metadata"}
        ) or (
            hash_games and source_kind == "run-manifest"
        )
        if should_hash and sha256_file(path) != record.get("sha256"):
            raise LegacyQuarantineError(f"registered artifact hash changed: {record['path']}")
        if should_hash and source_kind != "content":
            checked_declared += 1
        if source_kind == "run-manifest":
            checked_games += 1
        checked += 1
    for corpus in payload.get("corpora", []):
        manifest_path = _safe_path(
            root, f"{corpus['path']}/manifest.json", "corpus manifest"
        )
        manifest = load_json(manifest_path)
        if _manifest_lineage(manifest, manifest_path) != "legacy":
            raise LegacyQuarantineError(f"quarantined corpus lineage changed: {manifest_path}")
        if _game_merkle(manifest.get("games_artifacts", [])) != corpus.get(
            "game_hash_merkle_sha256"
        ):
            raise LegacyQuarantineError(f"quarantined game Merkle root changed: {manifest_path}")
    return {
        "schema_id": REGISTRY_SCHEMA_ID,
        "artifact_root": root.as_posix(),
        "inventory_digest_sha256": digest,
        "checked_files": checked,
        "checked_declared_files": checked_declared,
        "checked_game_files": checked_games,
        "hash_games": hash_games,
        "hash_binaries": hash_binaries,
        **payload["summary"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="scan and create/accept registry")
    build.add_argument("--artifact-root", type=Path, required=True)
    build.add_argument("--registry", type=Path, required=True)
    build.add_argument("--root-marker", action="store_true")
    build.add_argument("--hash-games", action="store_true")
    build.add_argument("--hash-binaries", action="store_true")
    verify = subparsers.add_parser("verify", help="verify a frozen registry")
    verify.add_argument("registry", type=Path)
    verify.add_argument("--artifact-root", type=Path)
    verify.add_argument("--hash-games", action="store_true")
    verify.add_argument("--hash-binaries", action="store_true")
    inventory = subparsers.add_parser("inventory", help="print registry summary")
    inventory.add_argument("registry", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            report = quarantine(
                args.artifact_root,
                args.registry,
                write_root_marker=args.root_marker,
                hash_games=args.hash_games,
                hash_binaries=args.hash_binaries,
            )
        elif args.command == "verify":
            report = verify_registry(
                args.registry,
                artifact_root=args.artifact_root,
                hash_games=args.hash_games,
                hash_binaries=args.hash_binaries,
            )
        else:
            payload = load_json(args.registry)
            report = {
                "summary": payload.get("summary"),
                "inventory_digest_sha256": payload.get("inventory_digest_sha256"),
            }
    except LegacyQuarantineError as error:
        print(f"V4 legacy quarantine failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

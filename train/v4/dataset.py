"""Validated dataset inputs for V4 training, including multi-shard indexes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from v3.factorized import FACTOR_POLICY_SCHEMA_ID
from v3.schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID

RUN_MANIFEST_SCHEMA_ID = "augmentia.self-play-v4-run.v1"
DATASET_INDEX_SCHEMA_ID = "augmentia.self-play-v4-dataset-index.v2"
VALIDATION_SCHEMA_ID = "augmentia.self-play-v4-validation.v1"
RULESET_ID = "augment_chess_20260915"
REPLAY_SCHEMA_ID = "augmentia.replay.game.v5"
FACTORIZED_TARGET_SCHEMA_ID = "augmentia.factorized-mcts-visits.v3"
LEGACY_REPLAY_SCHEMA_ID = "augmentia.replay.game.v4"
LEGACY_FACTORIZED_TARGET_SCHEMA_ID = "augmentia.factorized-mcts-visits.v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetGameV4:
    game_id: str
    game_index: int
    path: Path


@dataclass(frozen=True)
class DatasetInputV4:
    path: Path
    sha256: str
    schema_id: str
    dataset_id: str
    run_kind: str
    ruleset: Mapping[str, Any]
    factorized_policy_schema_id: str
    replay_schema_id: str
    factorized_target_schema_id: str
    observation_schema_id: str
    action_schema_id: str
    source_seal_tag: str | None
    source_seal_commit: str | None
    engine_commit: str | None
    producer_evaluator: Mapping[str, Any]
    games: tuple[DatasetGameV4, ...]
    source_manifests: tuple[Mapping[str, Any], ...]


def _load_run_manifest(
    path: Path, *, allow_legacy: bool = False
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_schema_id") != RUN_MANIFEST_SCHEMA_ID:
        raise ValueError(f"{path}: not a V4 self-play run manifest")
    if payload.get("ruleset", {}).get("id") != RULESET_ID:
        raise ValueError(f"{path}: not the 0915 ruleset")
    if payload.get("factorized_policy_schema_id") != FACTOR_POLICY_SCHEMA_ID:
        raise ValueError(f"{path}: factorized policy schema mismatch")
    expected = (
        (LEGACY_REPLAY_SCHEMA_ID, LEGACY_FACTORIZED_TARGET_SCHEMA_ID)
        if allow_legacy
        else (REPLAY_SCHEMA_ID, FACTORIZED_TARGET_SCHEMA_ID)
    )
    if (
        payload.get("replay_schema_id") != expected[0]
        or payload.get("factorized_target_schema_id") != expected[1]
    ):
        raise ValueError(
            f"{path}: replay/target lineage must be {expected[0]}/{expected[1]}"
        )
    if (
        payload.get("observation_schema_id") != OBSERVATION_SCHEMA_ID
        or payload.get("action_schema_id") != ACTION_SCHEMA_ID
    ):
        raise ValueError(f"{path}: observation/action schema lineage mismatch")
    artifacts = payload.get("games_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{path}: manifest contains no games")
    return payload


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _validated_index_manifest(
    index_path: Path,
    entry: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
) -> tuple[Path, dict[str, Any], Path]:
    if not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
        raise ValueError("dataset index manifest entries require path and sha256")
    manifest_path = _resolve(index_path.parent, entry["path"])
    if sha256_file(manifest_path) != entry["sha256"]:
        raise ValueError(f"dataset source manifest hash mismatch: {manifest_path}")
    manifest = _load_run_manifest(manifest_path, allow_legacy=allow_legacy)

    validation_value = entry.get("validation_path")
    validation_sha = entry.get("validation_sha256")
    if not isinstance(validation_value, str) or not isinstance(validation_sha, str):
        raise ValueError(
            "dataset index manifest entries require validation_path and validation_sha256"
        )
    validation_path = _resolve(index_path.parent, validation_value)
    if sha256_file(validation_path) != validation_sha:
        raise ValueError(f"dataset validation report hash mismatch: {validation_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("validation_schema_id") != VALIDATION_SCHEMA_ID:
        raise ValueError(f"{validation_path}: validation schema mismatch")
    if validation.get("run_id") != manifest.get("run_id"):
        raise ValueError(f"{validation_path}: validation run_id mismatch")
    if validation.get("games") != manifest.get("games"):
        raise ValueError(f"{validation_path}: validation game count mismatch")
    if validation.get("terminal_games", 0) + validation.get(
        "truncated_games", 0
    ) != validation.get("games"):
        raise ValueError(
            f"{validation_path}: validation terminal/truncated accounting mismatch"
        )
    return manifest_path, manifest, validation_path


def load_dataset_input(
    path: str | Path, *, allow_legacy: bool = False
) -> DatasetInputV4:
    input_path = Path(path).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    if payload.get("manifest_schema_id") == RUN_MANIFEST_SCHEMA_ID:
        manifest = _load_run_manifest(input_path, allow_legacy=allow_legacy)
        manifest_records = [(input_path, manifest, None)]
        schema_id = RUN_MANIFEST_SCHEMA_ID
        dataset_id = manifest["run_id"]
    elif payload.get("dataset_schema_id") == DATASET_INDEX_SCHEMA_ID:
        entries = payload.get("manifests")
        if not isinstance(entries, list) or not entries:
            raise ValueError("dataset index contains no manifests")
        dataset_id = payload.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset index requires a non-empty dataset_id")
        expected_lineage = (
            (LEGACY_REPLAY_SCHEMA_ID, LEGACY_FACTORIZED_TARGET_SCHEMA_ID)
            if allow_legacy
            else (REPLAY_SCHEMA_ID, FACTORIZED_TARGET_SCHEMA_ID)
        )
        if (
            payload.get("replay_schema_id") != expected_lineage[0]
            or payload.get("factorized_target_schema_id") != expected_lineage[1]
            or payload.get("observation_schema_id") != OBSERVATION_SCHEMA_ID
            or payload.get("action_schema_id") != ACTION_SCHEMA_ID
        ):
            raise ValueError("dataset index lineage metadata is missing or mismatched")
        manifest_records = [
            _validated_index_manifest(
                input_path, entry, allow_legacy=allow_legacy
            )
            for entry in entries
        ]
        schema_id = DATASET_INDEX_SCHEMA_ID
    else:
        raise ValueError(
            "training input is neither a V4 run manifest nor dataset index"
        )

    first = manifest_records[0][1]
    identity_fields = (
        "run_kind",
        "source_seal_tag",
        "source_seal_commit",
        "engine_commit",
        "factorized_policy_schema_id",
        "replay_schema_id",
        "factorized_target_schema_id",
        "observation_schema_id",
        "action_schema_id",
    )
    for _, manifest, _ in manifest_records[1:]:
        for field in identity_fields:
            if manifest.get(field) != first.get(field):
                raise ValueError(f"dataset source manifests disagree on {field}")
        if manifest.get("ruleset") != first.get("ruleset"):
            raise ValueError("dataset source manifests disagree on ruleset")
        if manifest.get("evaluator") != first.get("evaluator"):
            raise ValueError(
                "dataset source manifests disagree on evaluator provenance"
            )

    games: list[DatasetGameV4] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    source_manifests: list[Mapping[str, Any]] = []
    for manifest_path, manifest, validation_path in manifest_records:
        source_record: dict[str, Any] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "run_id": manifest["run_id"],
            "games": manifest["games"],
            "replay_schema_id": manifest["replay_schema_id"],
            "factorized_target_schema_id": manifest["factorized_target_schema_id"],
            "observation_schema_id": manifest["observation_schema_id"],
            "action_schema_id": manifest["action_schema_id"],
        }
        if validation_path is not None:
            source_record["validation_path"] = str(validation_path)
            source_record["validation_sha256"] = sha256_file(validation_path)
        source_manifests.append(source_record)
        for artifact in manifest["games_artifacts"]:
            game_id = artifact["game_id"]
            game_index = int(artifact["game_index"])
            if game_id in seen_ids:
                raise ValueError(
                    f"duplicate game_id across dataset manifests: {game_id}"
                )
            if game_index in seen_indices:
                raise ValueError(
                    f"duplicate game_index across dataset manifests: {game_index}"
                )
            seen_ids.add(game_id)
            seen_indices.add(game_index)
            games.append(
                DatasetGameV4(
                    game_id=game_id,
                    game_index=game_index,
                    path=(manifest_path.parent / artifact["relative_path"]).resolve(),
                )
            )
    games.sort(key=lambda game: game.game_index)
    if schema_id == DATASET_INDEX_SCHEMA_ID:
        expected_games = payload.get("games")
        if expected_games != len(games):
            raise ValueError("dataset index game count does not match source manifests")
        if not games:
            raise ValueError("dataset index contains no games")
        if payload.get("game_index_start") != games[0].game_index or payload.get(
            "game_index_end"
        ) != games[-1].game_index:
            raise ValueError("dataset index range does not match source manifests")
        expected_indices = list(range(games[0].game_index, games[-1].game_index + 1))
        if [game.game_index for game in games] != expected_indices:
            raise ValueError("dataset index source manifests contain a game_index gap")

    return DatasetInputV4(
        path=input_path,
        sha256=sha256_file(input_path),
        schema_id=schema_id,
        dataset_id=dataset_id,
        run_kind=first["run_kind"],
        ruleset=first["ruleset"],
        factorized_policy_schema_id=first["factorized_policy_schema_id"],
        replay_schema_id=first["replay_schema_id"],
        factorized_target_schema_id=first["factorized_target_schema_id"],
        observation_schema_id=first["observation_schema_id"],
        action_schema_id=first["action_schema_id"],
        source_seal_tag=first.get("source_seal_tag"),
        source_seal_commit=first.get("source_seal_commit"),
        engine_commit=first.get("engine_commit"),
        producer_evaluator=first["evaluator"],
        games=tuple(games),
        source_manifests=tuple(source_manifests),
    )

"""Controlled Phase 1 V4 opened-only versus legacy-all-open A/B training.

The two arms share one strict v5/v3 corpus split and one initialized baseline
model.  The legacy-all-open arm is an explicit, non-promotable experiment
*view* over the same v3 records: only root/prefix opening masks are replaced in
memory. Source replay records and their v5/v3 lineage are never rewritten, and
production export/resume rejects the experiment checkpoint semantics.

The module intentionally composes the Phase 0 trainer, exporter, and Rust parity
validator rather than maintaining a second training or export implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch.nn import functional as F

from v3.factorized import factorized_action_path
from v3.tensorize import ABILITY_IDS, ACTION_KIND_IDS, TARGET_KIND_IDS, V3Vocabulary

from .batch import FACTOR_FIELD_IDS, PrefixBatchV4, tensorize_factorized_cases
from .dataset import DatasetInputV4, load_dataset_input
from .loss import conditional_prefix_policy_metrics
from .replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    GAME_RECORD_V5_SCHEMA_ID,
    FactorizedTrainingExampleV4,
)
from .train import (
    POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT,
    POLICY_TARGET_SEMANTICS_OPENED_ONLY,
    RULESET_ID,
    SOURCE_SEAL_COMMIT,
    SOURCE_SEAL_TAG,
    V4Trainer,
    V4TrainingConfig,
    _dataset_provenance,
    _git_identity,
    _load_examples,
    _split_games,
    default_runtime_vocabulary_path,
    load_runtime_vocabulary,
    policy_target_semantics_lineage,
)

PHASE1_AB_PLAN_SCHEMA_ID = "augmentia.v4-phase1-ab-plan.v1"
PHASE1_AB_SUMMARY_SCHEMA_ID = "augmentia.v4-phase1-ab-summary.v1"
PHASE1_AB_ARM_RUN_SCHEMA_ID = "augmentia.v4-phase1-ab-arm-run.v1"
PHASE1_AB_PARITY_RECEIPT_SCHEMA_ID = "augmentia.v4-phase1-ab-parity-receipt.v1"

ARM_OPENED_ONLY = "opened-only"
ARM_LEGACY_ALL_OPEN_EXPERIMENT = "legacy-all-open-experiment"
# Compatibility aliases used by earlier Phase 1 callers.  The persisted arm
# and view IDs are explicit that this is an experiment, never a control model
# eligible for production export/resume.
ARM_ALL_OPEN_CONTROL = ARM_LEGACY_ALL_OPEN_EXPERIMENT
TRAINING_VIEW_OPENED_ONLY = "augmentia.v4-training-view.opened-only.v1"
TRAINING_VIEW_LEGACY_ALL_OPEN_EXPERIMENT = (
    "augmentia.v4-training-view.legacy-all-open-experiment.v1"
)
TRAINING_VIEW_ALL_OPEN_CONTROL = TRAINING_VIEW_LEGACY_ALL_OPEN_EXPERIMENT
ALL_OPEN_TRANSFORM_ID = "augmentia.v4-phase1-transform.all-legal-prefix-masks.v1"

TrainingViewName = Literal["opened_only", "all_open_control"]


class Phase1ABError(RuntimeError):
    """Raised when controlled A/B identity or execution proof is incomplete."""


@dataclass(frozen=True)
class Phase1ABConfig:
    """Immutable experiment configuration shared by both arms."""

    trainer: V4TrainingConfig
    validation_basis_points: int = 2000
    max_train_examples: int | None = None
    max_validation_examples: int | None = None
    checkpoint_every: int = 1
    runtime_vocabulary_path: str | None = None
    state_encoder_checkpoint: str | None = None
    ece_bins: int = 10
    bootstrap_samples: int = 1000
    bootstrap_seed: int | None = None
    parity_tolerance: float = 2e-4
    nll_non_degradation_tolerance: float = 0.02
    top1_non_degradation_tolerance: float = 0.01
    ece_non_degradation_tolerance: float = 0.01
    run_parity: bool = True
    rust_validator: str | None = None
    ort_dylib_path: str | None = None
    allow_smoke_corpus: bool = False
    allow_dirty_smoke: bool = False

    def validate(self) -> "Phase1ABConfig":
        self.trainer.validate()
        if self.trainer.encoder_kind != "baseline":
            raise Phase1ABError("Phase 1 A/B requires encoder_kind='baseline'")
        if not 1 <= self.validation_basis_points <= 9000:
            raise Phase1ABError("validation_basis_points must be in 1..=9000")
        for name in ("max_train_examples", "max_validation_examples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise Phase1ABError(f"{name} must be positive when supplied")
        if self.checkpoint_every < 0:
            raise Phase1ABError("checkpoint_every must be non-negative")
        if not 1 <= self.ece_bins <= 100:
            raise Phase1ABError("ece_bins must be in 1..=100")
        if self.bootstrap_samples < 0:
            raise Phase1ABError("bootstrap_samples must be non-negative")
        if not math.isfinite(self.parity_tolerance) or self.parity_tolerance <= 0.0:
            raise Phase1ABError("parity_tolerance must be finite and positive")
        for name in (
            "nll_non_degradation_tolerance",
            "top1_non_degradation_tolerance",
            "ece_non_degradation_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise Phase1ABError(f"{name} must be finite and non-negative")
        if not self.run_parity and not (
            self.allow_dirty_smoke or self.allow_smoke_corpus
        ):
            raise Phase1ABError("parity may only be skipped for an explicit smoke")
        if self.run_parity and not self.rust_validator:
            raise Phase1ABError("parity execution requires a Rust validator path")
        return self

    @property
    def vocabulary_path(self) -> Path:
        return Path(
            self.runtime_vocabulary_path or default_runtime_vocabulary_path()
        ).resolve()


@dataclass(frozen=True)
class ControlledCorpus:
    """One immutable load/split of the strict clean v5/v3 corpus."""

    dataset: DatasetInputV4
    train_paths: tuple[Path, ...]
    validation_paths: tuple[Path, ...]
    train_game_ids: tuple[str, ...]
    validation_game_ids: tuple[str, ...]
    train_examples: tuple[FactorizedTrainingExampleV4, ...]
    validation_examples: tuple[FactorizedTrainingExampleV4, ...]
    game_identities: tuple[Mapping[str, Any], ...]
    source_corpus_sha256: str
    split_sha256: str
    provenance: Mapping[str, Any]
    production_eligible: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset_input_schema_id": self.dataset.schema_id,
            "dataset_id": self.dataset.dataset_id,
            "dataset_manifest_path": str(self.dataset.path),
            "dataset_manifest_sha256": self.dataset.sha256,
            "source_corpus_sha256": self.source_corpus_sha256,
            "run_kind": self.dataset.run_kind,
            "production_eligible": self.production_eligible,
            "replay_schema_id": self.dataset.replay_schema_id,
            "factorized_target_schema_id": self.dataset.factorized_target_schema_id,
            "ruleset_id": self.dataset.ruleset.get("id"),
            "ruleset_manifest_sha256": self.dataset.ruleset.get("manifest_sha256"),
            "games": [dict(item) for item in self.game_identities],
        }


@dataclass(frozen=True)
class TrainingViewBundle:
    """A named mask transform over otherwise identical V3 examples."""

    view_id: str
    mode: TrainingViewName
    policy_target_semantics_id: str
    policy_target_semantics_lineage: Mapping[str, Any]
    examples: tuple[FactorizedTrainingExampleV4, ...]
    source_view_sha256: str
    view_sha256: str
    changed_decisions: int
    source_opened_action_count: int
    view_opened_action_count: int
    legal_action_count: int

    @property
    def is_control(self) -> bool:
        return self.mode == "all_open_control"

    def to_json(self) -> dict[str, Any]:
        return {
            "training_view_id": self.view_id,
            "mode": self.mode,
            "policy_target_semantics_id": self.policy_target_semantics_id,
            "policy_target_semantics_lineage": dict(
                self.policy_target_semantics_lineage
            ),
            "source_view_sha256": self.source_view_sha256,
            "training_view_sha256": self.view_sha256,
            "changed_decisions": self.changed_decisions,
            "source_opened_action_count": self.source_opened_action_count,
            "view_opened_action_count": self.view_opened_action_count,
            "legal_action_count": self.legal_action_count,
            "explicit_all_open_transform": ALL_OPEN_TRANSFORM_ID if self.is_control else None,
            "promotable": False,
        }


@dataclass(frozen=True)
class ArmRunResult:
    arm_id: str
    training_view: TrainingViewBundle
    initialization_sha256: str
    run_json_path: Path
    run_json_file_sha256: str
    run_json_seal_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    evaluation: Mapping[str, Any]
    evaluation_path: Path
    evaluation_file_sha256: str
    evaluation_seal_sha256: str
    parity: Mapping[str, Any] | None

    def to_json(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "training_view": self.training_view.to_json(),
            "initialization_sha256": self.initialization_sha256,
            "run_json_path": str(self.run_json_path),
            "run_json_file_sha256": self.run_json_file_sha256,
            "run_json_seal_sha256": self.run_json_seal_sha256,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "sidecar_path": str(self.sidecar_path),
            "sidecar_sha256": self.sidecar_sha256,
            "evaluation_path": str(self.evaluation_path),
            "evaluation_file_sha256": self.evaluation_file_sha256,
            "evaluation_seal_sha256": self.evaluation_seal_sha256,
            "parity": dict(self.parity) if self.parity is not None else None,
            "promotable": False,
        }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
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


def _atomic_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _seal_payload(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise Phase1ABError(f"sealed payload already contains {field}")
    result = dict(payload)
    result[field] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_payload_seal(payload: Mapping[str, Any], field: str) -> str:
    actual = payload.get(field)
    content = dict(payload)
    content.pop(field, None)
    expected = sha256_bytes(canonical_json_bytes(content))
    if actual != expected:
        raise Phase1ABError(f"{field} checksum mismatch")
    return str(actual)


def _write_idempotent_sealed_json(
    path: str | Path, payload: Mapping[str, Any], seal_field: str
) -> Path:
    verify_payload_seal(payload, seal_field)
    target = Path(path)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        verify_payload_seal(existing, seal_field)
        if existing != payload:
            raise Phase1ABError(f"refusing to overwrite different sealed artifact: {target}")
        return target
    return _atomic_json(target, payload)


def _example_payload(example: FactorizedTrainingExampleV4) -> dict[str, Any]:
    return asdict(example)


def _example_semantic_sha256(
    examples: Sequence[FactorizedTrainingExampleV4],
) -> str:
    payload = {
        "replay_schema_id": GAME_RECORD_V5_SCHEMA_ID,
        "factorized_target_schema_id": FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
        "examples": [_example_payload(item) for item in examples],
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _view_sha256(
    view_id: str,
    mode: TrainingViewName,
    examples: Sequence[FactorizedTrainingExampleV4],
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "training_view_id": view_id,
                "mode": mode,
                "examples": [_example_payload(item) for item in examples],
            }
        )
    )


def _assert_same_nonmask_payload(
    source: FactorizedTrainingExampleV4,
    transformed: FactorizedTrainingExampleV4,
) -> None:
    left = _example_payload(source)
    right = _example_payload(transformed)
    for payload in (left, right):
        payload.pop("opened_action_mask", None)
        payload.pop("opened_masks", None)
    if left != right:
        raise Phase1ABError(
            f"training view changed non-mask fields for {source.decision_id}"
        )


def make_training_view(
    examples: Sequence[FactorizedTrainingExampleV4],
    mode: TrainingViewName,
) -> TrainingViewBundle:
    """Return the opened-only or explicit all-open view without mutating source records."""

    source = tuple(examples)
    if not source:
        raise Phase1ABError("training view requires at least one example")
    source_sha = _view_sha256(TRAINING_VIEW_OPENED_ONLY, "opened_only", source)
    if mode == "opened_only":
        semantics_id = POLICY_TARGET_SEMANTICS_OPENED_ONLY
    elif mode == "all_open_control":
        semantics_id = POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT
    else:
        raise Phase1ABError(f"unknown training view mode: {mode}")
    semantics_lineage = policy_target_semantics_lineage(semantics_id)
    transformed: list[FactorizedTrainingExampleV4] = []
    changed = 0
    legal_count = 0
    for item in source:
        if (
            item.legacy
            or item.target_schema_id != FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID
            or item.replay_schema_id != GAME_RECORD_V5_SCHEMA_ID
        ):
            raise Phase1ABError(
                f"example {item.decision_id} is not strict replay.game.v5 / target.v3"
            )
        actions = item.legal_actions.get("actions", [])
        if not isinstance(actions, list) or not actions:
            raise Phase1ABError(f"example {item.decision_id} has no legal actions")
        if len(item.complete_visit_counts) != len(actions):
            raise Phase1ABError(f"example {item.decision_id} visits are not action-aligned")
        if len(item.opened_action_mask) != len(actions):
            raise Phase1ABError(f"example {item.decision_id} root mask is not action-aligned")
        if len(item.opened_masks) != len(item.prefix_targets):
            raise Phase1ABError(f"example {item.decision_id} prefix masks are not row-aligned")
        for masks, row in zip(item.opened_masks, item.prefix_targets):
            if len(masks) != len(row.children):
                raise Phase1ABError(
                    f"example {item.decision_id} prefix mask is not child-aligned"
                )
        legal_count += len(actions)
        if mode == "opened_only":
            transformed.append(item)
            continue
        all_open = tuple(True for _ in actions)
        all_prefix = tuple(
            tuple(True for _ in row.children) for row in item.prefix_targets
        )
        view = replace(
            item,
            opened_action_mask=all_open,
            opened_masks=all_prefix,
        )
        _assert_same_nonmask_payload(item, view)
        changed += int(view.opened_action_mask != item.opened_action_mask)
        transformed.append(view)
    if mode == "opened_only":
        view_id = TRAINING_VIEW_OPENED_ONLY
    elif mode == "all_open_control":
        view_id = TRAINING_VIEW_ALL_OPEN_CONTROL
    else:
        raise Phase1ABError(f"unknown training view mode: {mode}")
    result = tuple(transformed)
    return TrainingViewBundle(
        view_id=view_id,
        mode=mode,
        policy_target_semantics_id=semantics_id,
        policy_target_semantics_lineage=semantics_lineage,
        examples=result,
        source_view_sha256=source_sha,
        view_sha256=_view_sha256(view_id, mode, result),
        changed_decisions=changed,
        source_opened_action_count=sum(
            sum(item.opened_action_mask) for item in source
        ),
        view_opened_action_count=sum(
            sum(item.opened_action_mask) for item in result
        ),
        legal_action_count=legal_count,
    )


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash every state-dict tensor with dtype and shape, independent of device."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _game_identities(dataset: DatasetInputV4) -> tuple[Mapping[str, Any], ...]:
    expected: dict[str, Mapping[str, Any]] = {}
    for source in dataset.source_manifests:
        source_path = Path(str(source["path"]))
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        for artifact in source_payload.get("games_artifacts", []):
            game_id = str(artifact.get("game_id", ""))
            if not game_id or game_id in expected:
                raise Phase1ABError(
                    f"source manifest has missing/duplicate game_id: {game_id}"
                )
            expected[game_id] = artifact
    identities: list[Mapping[str, Any]] = []
    for game in dataset.games:
        if not game.path.is_file():
            raise Phase1ABError(f"dataset replay is missing: {game.path}")
        payload = json.loads(game.path.read_text(encoding="utf-8"))
        trajectory = payload.get("trajectory_sha256")
        if not isinstance(trajectory, str) or len(trajectory) != 64:
            raise Phase1ABError(f"replay {game.game_id} has no trajectory SHA-256")
        file_sha = sha256_file(game.path)
        artifact = expected.get(game.game_id)
        if artifact is None:
            raise Phase1ABError(
                f"dataset game {game.game_id} is absent from source manifests"
            )
        if artifact.get("file_sha256") != file_sha:
            raise Phase1ABError(
                f"dataset replay file hash mismatch for {game.game_id}"
            )
        if artifact.get("trajectory_sha256") != trajectory:
            raise Phase1ABError(
                f"dataset trajectory hash mismatch for {game.game_id}"
            )
        identities.append(
            {
                "game_index": game.game_index,
                "game_id": game.game_id,
                "path": str(game.path),
                "file_sha256": file_sha,
                "trajectory_sha256": trajectory,
            }
        )
    if set(expected) != {item.game_id for item in dataset.games}:
        raise Phase1ABError("source manifests and dataset game IDs differ")
    return tuple(identities)


def load_controlled_corpus(
    manifest: str | Path,
    config: Phase1ABConfig,
    vocabulary_identity: Mapping[str, Any],
) -> ControlledCorpus:
    """Load and split the corpus exactly once for both A/B arms."""

    dataset = load_dataset_input(manifest)
    if dataset.ruleset.get("id") != RULESET_ID:
        raise Phase1ABError(f"Phase 1 corpus must use {RULESET_ID}")
    if (
        dataset.replay_schema_id != GAME_RECORD_V5_SCHEMA_ID
        or dataset.factorized_target_schema_id != FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID
    ):
        raise Phase1ABError("Phase 1 corpus must use replay.game.v5 / target.v3")
    if (
        dataset.source_seal_tag != SOURCE_SEAL_TAG
        or dataset.source_seal_commit != SOURCE_SEAL_COMMIT
    ):
        raise Phase1ABError("Phase 1 corpus source seal differs from the V4 training base")
    production_eligible = dataset.run_kind == "production"
    if not production_eligible and not config.allow_smoke_corpus:
        raise Phase1ABError(
            f"corpus run_kind={dataset.run_kind!r} requires allow_smoke_corpus=True"
        )
    train_paths, validation_paths, train_ids, validation_ids = _split_games(
        dataset,
        config.validation_basis_points,
        config.trainer.seed,
    )
    train_examples = tuple(
        _load_examples(train_paths, config.max_train_examples)
    )
    validation_examples = tuple(
        _load_examples(validation_paths, config.max_validation_examples)
    )
    if not train_examples or not validation_examples:
        raise Phase1ABError("Phase 1 requires non-empty train and validation decisions")
    train_decisions = [item.decision_id for item in train_examples]
    validation_decisions = [item.decision_id for item in validation_examples]
    if len(set(train_decisions)) != len(train_decisions) or len(
        set(validation_decisions)
    ) != len(validation_decisions):
        raise Phase1ABError("decision IDs must be unique within each split")
    if set(train_decisions) & set(validation_decisions):
        raise Phase1ABError("train/validation decision split overlaps")
    game_identities = _game_identities(dataset)
    source_payload = {
        "dataset_manifest_sha256": dataset.sha256,
        "dataset_id": dataset.dataset_id,
        "games": list(game_identities),
    }
    source_sha = sha256_bytes(canonical_json_bytes(source_payload))
    split_payload = {
        "source_corpus_sha256": source_sha,
        "algorithm": "sha256(f'{split_seed}:{game_id}') modulo 10000",
        "split_seed": config.trainer.seed,
        "validation_basis_points": config.validation_basis_points,
        "train_game_ids": list(train_ids),
        "validation_game_ids": list(validation_ids),
        "max_train_examples": config.max_train_examples,
        "max_validation_examples": config.max_validation_examples,
        "train_example_semantic_sha256": _example_semantic_sha256(train_examples),
        "validation_example_semantic_sha256": _example_semantic_sha256(
            validation_examples
        ),
        "train_decision_ids": train_decisions,
        "validation_decision_ids": validation_decisions,
    }
    split_sha = sha256_bytes(canonical_json_bytes(split_payload))
    provenance = dict(
        _dataset_provenance(
            dataset,
            vocabulary_identity,
            train_ids,
            validation_ids,
            max_train_examples=config.max_train_examples,
            max_validation_examples=config.max_validation_examples,
        )
    )
    provenance.update(
        {
            "source_corpus_sha256": source_sha,
            "split_sha256": split_sha,
            "train_example_semantic_sha256": split_payload[
                "train_example_semantic_sha256"
            ],
            "validation_example_semantic_sha256": split_payload[
                "validation_example_semantic_sha256"
            ],
            "corpus_production_eligible": production_eligible,
        }
    )
    return ControlledCorpus(
        dataset=dataset,
        train_paths=tuple(train_paths),
        validation_paths=tuple(validation_paths),
        train_game_ids=tuple(train_ids),
        validation_game_ids=tuple(validation_ids),
        train_examples=train_examples,
        validation_examples=validation_examples,
        game_identities=game_identities,
        source_corpus_sha256=source_sha,
        split_sha256=split_sha,
        provenance=provenance,
        production_eligible=production_eligible,
    )


def _arm_provenance(
    corpus: ControlledCorpus,
    view: TrainingViewBundle,
    arm_id: str,
    plan_sha256: str,
    initialization_sha256: str,
) -> dict[str, Any]:
    provenance = dict(corpus.provenance)
    provenance.update(
        {
            "phase1_ab_plan_sha256": plan_sha256,
            "phase1_ab_arm_id": arm_id,
            "phase1_initialization_sha256": initialization_sha256,
            "training_view_id": view.view_id,
            "training_view_mode": view.mode,
            "training_view_sha256": view.view_sha256,
            "source_training_view_sha256": view.source_view_sha256,
            "policy_target_semantics_id": view.policy_target_semantics_id,
            "policy_target_semantics_lineage": dict(
                view.policy_target_semantics_lineage
            ),
            "policy_target_semantics_experiment_only": bool(
                view.policy_target_semantics_lineage["experiment_only"]
            ),
            "policy_target_semantics_production_eligible": not bool(
                view.policy_target_semantics_lineage["experiment_only"]
            ),
            "source_corpus_sha256": corpus.source_corpus_sha256,
            "split_sha256": corpus.split_sha256,
            "phase1_promotable": False,
        }
    )
    if view.is_control:
        provenance.update(
            {
                "legacy_all_open_experiment": True,
                "all_open_control": True,
                "all_open_transform_id": ALL_OPEN_TRANSFORM_ID,
                "phase1_export_eligible": False,
                "phase1_resume_eligible": False,
                "promotable": False,
                "non_promotable_reason": "legacy all-open training semantics",
            }
        )
    else:
        provenance.update(
            {
                "legacy_all_open_experiment": False,
                "all_open_control": False,
                "phase1_export_eligible": True,
                "phase1_resume_eligible": True,
            }
        )
    return provenance


def build_phase1_ab_plan(
    corpus: ControlledCorpus,
    opened_view: TrainingViewBundle,
    control_view: TrainingViewBundle,
    config: Phase1ABConfig,
    vocabulary_identity: Mapping[str, Any],
    initialization_sha256: str,
    git_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable pre-training identity for both arms."""

    if opened_view.source_view_sha256 != control_view.source_view_sha256:
        raise Phase1ABError("A/B views do not share the same source training view")
    evaluation_view = make_training_view(
        corpus.validation_examples, "opened_only"
    )
    state_encoder_sha = (
        sha256_file(config.state_encoder_checkpoint)
        if config.state_encoder_checkpoint
        else None
    )
    phase1_config = asdict(config)
    for field in (
        "runtime_vocabulary_path",
        "state_encoder_checkpoint",
        "rust_validator",
        "ort_dylib_path",
    ):
        if phase1_config.get(field) is not None:
            phase1_config[field] = str(phase1_config[field])
    plan = {
        "schema_id": PHASE1_AB_PLAN_SCHEMA_ID,
        "status": "planned",
        "source_corpus": corpus.to_json(),
        "split": {
            "split_sha256": corpus.split_sha256,
            "validation_basis_points": config.validation_basis_points,
            "split_seed": config.trainer.seed,
            "train_game_ids": list(corpus.train_game_ids),
            "validation_game_ids": list(corpus.validation_game_ids),
            "train_example_count": len(corpus.train_examples),
            "validation_example_count": len(corpus.validation_examples),
            "train_example_semantic_sha256": _example_semantic_sha256(
                corpus.train_examples
            ),
            "validation_example_semantic_sha256": _example_semantic_sha256(
                corpus.validation_examples
            ),
        },
        "phase1_config": phase1_config,
        "phase1_config_sha256": sha256_bytes(
            canonical_json_bytes(phase1_config)
        ),
        "training_config": asdict(config.trainer),
        "training_config_sha256": sha256_bytes(
            canonical_json_bytes(asdict(config.trainer))
        ),
        "policy_target_semantics": {
            ARM_OPENED_ONLY: policy_target_semantics_lineage(
                POLICY_TARGET_SEMANTICS_OPENED_ONLY
            ),
            ARM_LEGACY_ALL_OPEN_EXPERIMENT: policy_target_semantics_lineage(
                POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT
            ),
        },
        "model_gate": {
            "primary_rule": "conditional_kl_or_visited_tail_recall_at_3_consistent_improvement",
            "nll_non_degradation_tolerance": config.nll_non_degradation_tolerance,
            "top1_non_degradation_tolerance": config.top1_non_degradation_tolerance,
            "ece_non_degradation_tolerance": config.ece_non_degradation_tolerance,
            "promotion_effect": False,
        },
        "runtime_vocabulary": dict(vocabulary_identity),
        "state_encoder_checkpoint_path": (
            str(config.state_encoder_checkpoint)
            if config.state_encoder_checkpoint is not None
            else None
        ),
        "state_encoder_checkpoint_sha256": state_encoder_sha,
        "initialization_sha256": initialization_sha256,
        "views": {
            ARM_OPENED_ONLY: opened_view.to_json(),
            ARM_ALL_OPEN_CONTROL: control_view.to_json(),
        },
        "training_validation_views": {
            ARM_OPENED_ONLY: make_training_view(
                corpus.validation_examples, "opened_only"
            ).to_json(),
            ARM_ALL_OPEN_CONTROL: make_training_view(
                corpus.validation_examples, "all_open_control"
            ).to_json(),
        },
        "evaluation_view_id": TRAINING_VIEW_OPENED_ONLY,
        "evaluation_view_sha256": evaluation_view.view_sha256,
        "parity": {
            "enabled": config.run_parity,
            "scope": "opened-only-production-semantics",
            "experiment_arm_export_blocked": True,
            "tolerance": config.parity_tolerance,
            "rust_validator": (
                str(config.rust_validator)
                if config.rust_validator is not None
                else None
            ),
            "ort_dylib_path": (
                str(config.ort_dylib_path)
                if config.ort_dylib_path is not None
                else None
            ),
        },
        "git": dict(git_identity),
        "promotable": False,
        "promotion_note": "Phase 1 A/B integrity evidence is not a Phase 2 strength gate",
    }
    return _seal_payload(plan, "plan_sha256")


def _build_trainer(
    examples: Sequence[FactorizedTrainingExampleV4],
    validation_examples: Sequence[FactorizedTrainingExampleV4],
    run_dir: Path,
    config: Phase1ABConfig,
    vocabulary: V3Vocabulary,
    provenance: Mapping[str, Any],
) -> V4Trainer:
    return V4Trainer(
        examples,
        run_dir,
        config.trainer,
        config.state_encoder_checkpoint,
        vocabulary=vocabulary,
        validation_examples=validation_examples,
        provenance=provenance,
        policy_target_semantics_id=provenance["policy_target_semantics_id"],
    )


def prove_shared_initialization(
    corpus: ControlledCorpus,
    opened_view: TrainingViewBundle,
    control_view: TrainingViewBundle,
    config: Phase1ABConfig,
    vocabulary: V3Vocabulary,
    probe_dir: Path,
) -> str:
    """Build both views before training and require bit-identical initialization."""

    hashes: list[str] = []
    for view in (opened_view, control_view):
        trainer = _build_trainer(
            view.examples,
            (),
            probe_dir,
            config,
            vocabulary,
            {
                "policy_target_semantics_id": view.policy_target_semantics_id,
                "policy_target_semantics_lineage": dict(
                    view.policy_target_semantics_lineage
                ),
            },
        )
        hashes.append(model_state_sha256(trainer.model))
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if hashes[0] != hashes[1]:
        raise Phase1ABError(
            f"A/B initial weights differ: opened={hashes[0]} control={hashes[1]}"
        )
    return hashes[0]


def _field_name_by_id() -> dict[int, str]:
    return {value: key for key, value in FACTOR_FIELD_IDS.items()}


def _category_value(field: str, category: int) -> str:
    table = {
        "family": ACTION_KIND_IDS,
        "ability": ABILITY_IDS,
        "target_kind": TARGET_KIND_IDS,
    }
    for name, identifier in table[field].items():
        if identifier == category:
            return name
    raise Phase1ABError(f"unknown factor category {field}:{category}")


def _edge_tokens(edges: Any, row: int) -> tuple[str, ...]:
    field_names = _field_name_by_id()
    tokens: list[str] = []
    for column in range(edges.field.shape[1]):
        if not bool(edges.mask[row, column]):
            continue
        field = field_names[int(edges.field[row, column])]
        category = int(edges.category[row, column])
        index = int(edges.index[row, column])
        if field == "target_sequence":
            ordinal = int(edges.ordinal[row, column])
            value = str(index)
            tokens.append(f"target_{ordinal - 1}:{value}")
        elif field in ("family", "ability", "target_kind"):
            tokens.append(f"{field}:{_category_value(field, category)}")
        elif index < 0:
            value = "none" if field == "promotion" else "action"
            tokens.append(f"{field}:{value}")
        else:
            tokens.append(f"{field}:{index}")
    return tuple(tokens)


def complete_action_log_probabilities(
    actions_by_decision: Sequence[Sequence[Mapping[str, Any]]],
    prefixes: PrefixBatchV4,
    prefix_logits: torch.Tensor,
) -> torch.Tensor:
    """Return all-legal complete-action log probabilities as path products."""

    if len(actions_by_decision) != prefixes.complete_action_mask.shape[0]:
        raise Phase1ABError("action decisions and prefix batch are not aligned")
    if prefix_logits.shape != prefixes.visit_counts.shape:
        raise Phase1ABError("prefix logits and visit tensor shapes differ")
    row_index: dict[tuple[int, tuple[str, ...]], int] = {}
    for row in range(prefixes.row_count):
        key = (int(prefixes.state_index[row]), _edge_tokens(prefixes.prefix, row))
        if key in row_index:
            raise Phase1ABError(f"duplicate prefix row key: {key}")
        row_index[key] = row
    maximum = max(len(actions) for actions in actions_by_decision)
    result = torch.zeros(
        (len(actions_by_decision), maximum),
        dtype=prefix_logits.dtype,
        device=prefix_logits.device,
    )
    minimum = torch.finfo(prefix_logits.dtype).min
    for decision, actions in enumerate(actions_by_decision):
        for action_index, action in enumerate(actions):
            path = tuple(
                f"{field}:{value}" for field, value in factorized_action_path(action)
            )
            log_probability = prefix_logits.new_zeros(())
            for depth, token in enumerate(path):
                key = (decision, path[:depth])
                row = row_index.get(key)
                if row is None:
                    raise Phase1ABError(f"missing prefix row for decision {decision}: {key}")
                children = _edge_tokens(prefixes.children, row)
                try:
                    column = children.index(token)
                except ValueError as error:
                    raise Phase1ABError(
                        f"action path token {token} is absent from decision {decision}"
                    ) from error
                legal_logits = prefix_logits[row].masked_fill(
                    ~prefixes.legal_mask[row], minimum
                )
                log_probability = log_probability + F.log_softmax(
                    legal_logits, dim=-1
                )[column]
            result[decision, action_index] = log_probability
    valid = prefixes.complete_action_mask
    normalized = torch.where(valid, result, torch.full_like(result, minimum))
    mass_error = (torch.logsumexp(normalized, dim=1) - torch.zeros(result.shape[0], device=result.device)).abs()
    if torch.any(~torch.isfinite(result.masked_select(valid))) or torch.any(
        mass_error > 5e-4
    ):
        raise Phase1ABError("complete-action path products are not normalized")
    return result


def _decision_id_parts(decision_id: str) -> tuple[str, str]:
    game_id, separator, index = decision_id.rpartition(":")
    if not separator or not game_id or not index.isdigit():
        return decision_id, "0"
    return game_id, index


def evaluate_prefix_batch(
    prefix_logits: torch.Tensor,
    prefixes: PrefixBatchV4,
    actions_by_decision: Sequence[Sequence[Mapping[str, Any]]],
    decision_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Compute per-decision conditional and complete-action quality metrics."""

    if len(decision_ids) != len(actions_by_decision):
        raise Phase1ABError("decision IDs and action decisions are not aligned")
    cross_entropy, entropy, kl = conditional_prefix_policy_metrics(
        prefix_logits, prefixes, reduction="none"
    )
    decision_count = len(decision_ids)
    device = prefix_logits.device
    ce_decision = torch.zeros(decision_count, dtype=torch.float64, device=device)
    entropy_decision = torch.zeros_like(ce_decision)
    kl_decision = torch.zeros_like(ce_decision)
    ce_decision.index_add_(0, prefixes.state_index.long(), cross_entropy.double())
    entropy_decision.index_add_(0, prefixes.state_index.long(), entropy.double())
    kl_decision.index_add_(0, prefixes.state_index.long(), kl.double())
    log_probabilities = complete_action_log_probabilities(
        actions_by_decision, prefixes, prefix_logits
    )
    visits = prefixes.complete_action_visits
    valid = prefixes.complete_action_mask
    row_visits = (visits * valid.long()).sum(dim=1)
    if torch.any(row_visits <= 0):
        raise Phase1ABError("every evaluated decision must have positive visit mass")
    target = visits / row_visits.clamp_min(1).unsqueeze(1)
    soft_nll = -(target * log_probabilities).sum(dim=1)
    modal_index = torch.argmax(visits.masked_fill(~valid, -1), dim=1)
    modal_nll = -log_probabilities.gather(1, modal_index[:, None]).squeeze(1)
    probabilities = torch.softmax(
        log_probabilities.masked_fill(~valid, torch.finfo(log_probabilities.dtype).min),
        dim=1,
    )
    prediction = torch.argmax(probabilities, dim=1)
    confidence = probabilities.max(dim=1).values
    correctness = prediction.eq(modal_index)
    ranked_actions = torch.topk(
        probabilities, k=min(3, probabilities.shape[1]), dim=1
    ).indices
    rows: list[dict[str, Any]] = []
    for decision, decision_id in enumerate(decision_ids):
        decision_rows = prefixes.state_index == decision
        legal = prefixes.legal_mask[decision_rows]
        opened = prefixes.opened_mask[decision_rows]
        ce = float(ce_decision[decision].cpu())
        target_entropy = float(entropy_decision[decision].cpu())
        conditional_kl = float(kl_decision[decision].cpu())
        if not math.isclose(
            ce,
            target_entropy + conditional_kl,
            rel_tol=1e-5,
            abs_tol=2e-5,
        ):
            raise Phase1ABError("conditional CE = entropy + KL identity failed")
        game_id, decision_index = _decision_id_parts(decision_id)
        visited_actions = [
            int(index)
            for index in torch.nonzero(visits[decision] > 0, as_tuple=False)
            .flatten()
            .cpu()
            .tolist()
        ]
        visited_tail = [
            action for action in visited_actions if action != int(modal_index[decision])
        ]
        ranked = [int(index) for index in ranked_actions[decision].cpu().tolist()]
        visited_tail_hits = sum(action in ranked for action in visited_tail)
        visited_tail_recall = (
            visited_tail_hits / len(visited_tail) if visited_tail else None
        )
        rows.append(
            {
                "decision_id": decision_id,
                "game_id": game_id,
                "decision_index": decision_index,
                "conditional_cross_entropy": ce,
                "conditional_target_entropy": target_entropy,
                "conditional_kl": conditional_kl,
                "complete_action_soft_nll": float(soft_nll[decision].cpu()),
                "complete_action_modal_nll": float(modal_nll[decision].cpu()),
                "mcts_modal_top1": float(correctness[decision].cpu()),
                "mcts_modal_confidence": float(confidence[decision].cpu()),
                "visited_actions": len(visited_actions),
                "visited_tail_actions": len(visited_tail),
                "visited_tail_hits_at_3": visited_tail_hits,
                "visited_tail_recall_at_3": visited_tail_recall,
                "calibration_gap": abs(
                    float(correctness[decision].cpu())
                    - float(confidence[decision].cpu())
                ),
                "prefix_rows": int(decision_rows.sum().cpu()),
                "opened_children": int((legal & opened).sum().cpu()),
                "closed_children_excluded": int((legal & ~opened).sum().cpu()),
                "complete_actions": len(actions_by_decision[decision]),
            }
        )
    return rows


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(float(item[key]) for item in rows) / max(1, len(rows))


def _calibration_bins(
    per_decision: Sequence[Mapping[str, Any]], ece_bins: int
) -> tuple[float, list[dict[str, Any]]]:
    if not per_decision or not 1 <= ece_bins <= 100:
        raise Phase1ABError("ECE requires decisions and 1..=100 bins")
    bins = [
        {
            "lower": index / ece_bins,
            "upper": (index + 1) / ece_bins,
            "count": 0,
            "confidence_sum": 0.0,
            "correct_sum": 0.0,
        }
        for index in range(ece_bins)
    ]
    for row in per_decision:
        confidence = min(1.0, max(0.0, float(row["mcts_modal_confidence"])))
        index = min(ece_bins - 1, int(confidence * ece_bins))
        bins[index]["count"] += 1
        bins[index]["confidence_sum"] += confidence
        bins[index]["correct_sum"] += float(row["mcts_modal_top1"])
    ece = 0.0
    for item in bins:
        item_count = int(item["count"])
        if item_count:
            average_confidence = float(item["confidence_sum"]) / item_count
            accuracy = float(item["correct_sum"]) / item_count
            item["average_confidence"] = average_confidence
            item["accuracy"] = accuracy
            item["absolute_calibration_gap"] = abs(accuracy - average_confidence)
            ece += item_count * item["absolute_calibration_gap"]
        else:
            item["average_confidence"] = None
            item["accuracy"] = None
            item["absolute_calibration_gap"] = None
    return ece / len(per_decision), bins


def aggregate_evaluation(
    per_decision: Sequence[Mapping[str, Any]],
    *,
    ece_bins: int,
) -> dict[str, Any]:
    if not per_decision:
        raise Phase1ABError("evaluation requires at least one decision")
    ece, bins = _calibration_bins(per_decision, ece_bins)
    count = len(per_decision)
    tail_rows = [
        row for row in per_decision if row.get("visited_tail_recall_at_3") is not None
    ]
    tail_actions = sum(int(row["visited_tail_actions"]) for row in tail_rows)
    tail_hits = sum(int(row["visited_tail_hits_at_3"]) for row in tail_rows)
    tail_recall_macro = (
        sum(float(row["visited_tail_recall_at_3"]) for row in tail_rows)
        / len(tail_rows)
        if tail_rows
        else None
    )
    return {
        "schema_id": "augmentia.v4-phase1-ab-evaluation.v1",
        "decision_count": count,
        "game_count": len({str(item["game_id"]) for item in per_decision}),
        "conditional_cross_entropy": _mean(
            per_decision, "conditional_cross_entropy"
        ),
        "conditional_target_entropy": _mean(
            per_decision, "conditional_target_entropy"
        ),
        "conditional_kl": _mean(per_decision, "conditional_kl"),
        "complete_action_soft_nll": _mean(
            per_decision, "complete_action_soft_nll"
        ),
        "complete_action_modal_nll": _mean(
            per_decision, "complete_action_modal_nll"
        ),
        "mcts_modal_top1": _mean(per_decision, "mcts_modal_top1"),
        "visited_tail_recall_at_3": tail_recall_macro,
        "visited_tail_recall_at_3_micro": (
            tail_hits / tail_actions if tail_actions else None
        ),
        "visited_tail_eligible_decisions": len(tail_rows),
        "visited_tail_actions": tail_actions,
        "visited_tail_hits_at_3": tail_hits,
        "mcts_modal_ece": ece,
        "mean_calibration_gap": _mean(per_decision, "calibration_gap"),
        "opened_children": sum(int(item["opened_children"]) for item in per_decision),
        "closed_children_excluded": sum(
            int(item["closed_children_excluded"]) for item in per_decision
        ),
        "metric_definitions": {
            "conditional_kl": "opened-only prefix KL, micro-averaged per decision",
            "complete_action_soft_nll": "cross entropy against MCTS visit distribution over all legal actions",
            "complete_action_modal_nll": "NLL of the deterministic MCTS visit-modal action",
            "mcts_modal_top1": "model argmax equals the MCTS visit-modal action",
            "visited_tail_recall_at_3": (
                "macro recall of positive-visit non-modal complete actions in the "
                "model top-3 over all legal complete actions"
            ),
            "mcts_modal_ece": f"{ece_bins}-bin equal-width MCTS-modal calibration error",
        },
        "ece_bins": bins,
        "per_decision": [dict(item) for item in per_decision],
    }


@torch.no_grad()
def evaluate_common_view(
    model: torch.nn.Module,
    examples: Sequence[FactorizedTrainingExampleV4],
    vocabulary: V3Vocabulary,
    device: torch.device | str,
    *,
    batch_size: int,
    ece_bins: int,
) -> dict[str, Any]:
    """Evaluate both trained arms on the same opened-only held-out view."""

    if not examples or batch_size <= 0:
        raise Phase1ABError("common evaluation requires examples and positive batch size")
    model.eval()
    rows: list[dict[str, Any]] = []
    cases = [
        {"observation": item.observation, "legal_actions": item.legal_actions}
        for item in examples
    ]
    visits = [list(item.complete_visit_counts) for item in examples]
    opened = [list(item.opened_action_mask) for item in examples]
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start : start + batch_size])
        batch = tensorize_factorized_cases(
            cases[start : start + batch_size],
            vocabulary,
            visits[start : start + batch_size],
            opened_action_masks=opened[start : start + batch_size],
        ).to(device)
        output = model(batch)
        rows.extend(
            evaluate_prefix_batch(
                output.prefix_logits,
                batch.prefixes,
                [item.legal_actions["actions"] for item in selected],
                [item.decision_id for item in selected],
            )
        )
    result = aggregate_evaluation(rows, ece_bins=ece_bins)
    result["evaluation_view_id"] = TRAINING_VIEW_OPENED_ONLY
    result["evaluation_view_sha256"] = _view_sha256(
        TRAINING_VIEW_OPENED_ONLY, "opened_only", examples
    )
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise Phase1ABError("cannot take a quantile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_game: dict[str, list[float]] = {}
    for row in rows:
        by_game.setdefault(str(row["game_id"]), []).append(float(row[key]))
    game_ids = sorted(by_game)
    rng = random.Random(seed)
    estimates: list[float] = []
    game_count = len(game_ids)
    for _ in range(samples):
        chosen = [rng.choice(game_ids) for _ in range(game_count)]
        values = [value for game_id in chosen for value in by_game[game_id]]
        estimates.append(sum(values) / len(values))
    return {
        "mean": sum(estimates) / len(estimates) if estimates else _mean(rows, key),
        "ci95_low": _quantile(estimates, 0.025) if estimates else None,
        "ci95_high": _quantile(estimates, 0.975) if estimates else None,
        "samples": samples,
        "seed": seed,
        "blocks": game_count,
    }


def _paired_ece_bootstrap(
    opened_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    ece_bins: int,
) -> dict[str, Any]:
    opened_by_game: dict[str, list[Mapping[str, Any]]] = {}
    control_by_game: dict[str, list[Mapping[str, Any]]] = {}
    for row in opened_rows:
        opened_by_game.setdefault(str(row["game_id"]), []).append(row)
    for row in control_rows:
        control_by_game.setdefault(str(row["game_id"]), []).append(row)
    game_ids = sorted(opened_by_game)
    if game_ids != sorted(control_by_game):
        raise Phase1ABError("A/B ECE game blocks differ")
    observed = (
        _calibration_bins(control_rows, ece_bins)[0]
        - _calibration_bins(opened_rows, ece_bins)[0]
    )
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        chosen = [rng.choice(game_ids) for _ in range(len(game_ids))]
        sampled_opened = [row for game_id in chosen for row in opened_by_game[game_id]]
        sampled_control = [row for game_id in chosen for row in control_by_game[game_id]]
        estimates.append(
            _calibration_bins(sampled_control, ece_bins)[0]
            - _calibration_bins(sampled_opened, ece_bins)[0]
        )
    return {
        "mean": sum(estimates) / len(estimates) if estimates else observed,
        "observed_control_minus_opened": observed,
        "ci95_low": _quantile(estimates, 0.025) if estimates else None,
        "ci95_high": _quantile(estimates, 0.975) if estimates else None,
        "samples": samples,
        "seed": seed,
        "blocks": len(game_ids),
        "bins": ece_bins,
    }


def compare_evaluations(
    opened: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Pair held-out decisions and report improvements with game-block CIs."""

    opened_rows = list(opened["per_decision"])
    control_rows = list(control["per_decision"])
    if [item["decision_id"] for item in opened_rows] != [
        item["decision_id"] for item in control_rows
    ]:
        raise Phase1ABError("A/B evaluation decision order or identity differs")
    differences: list[dict[str, Any]] = []
    for left, right in zip(opened_rows, control_rows):
        row = dict(left)
        row.update(
            {
                f"delta_{key}": float(left[key]) - float(right[key])
                for key in (
                    "conditional_cross_entropy",
                    "conditional_kl",
                    "complete_action_soft_nll",
                    "complete_action_modal_nll",
                    "mcts_modal_top1",
                    "calibration_gap",
                )
            }
        )
        if (
            left.get("visited_tail_recall_at_3") is not None
            and right.get("visited_tail_recall_at_3") is not None
        ):
            row["delta_visited_tail_recall_at_3"] = (
                float(left["visited_tail_recall_at_3"])
                - float(right["visited_tail_recall_at_3"])
            )
        differences.append(row)
    specifications = {
        "conditional_cross_entropy_improvement": "control_ce - opened_ce",
        "conditional_kl_improvement": "control_kl - opened_kl",
        "complete_action_soft_nll_improvement": "control_nll - opened_nll",
        "complete_action_modal_nll_improvement": "control_nll - opened_nll",
        "mcts_modal_top1_improvement": "opened_top1 - control_top1",
    }
    deltas = {
        "conditional_cross_entropy_improvement": "conditional_cross_entropy",
        "conditional_kl_improvement": "conditional_kl",
        "complete_action_soft_nll_improvement": "complete_action_soft_nll",
        "complete_action_modal_nll_improvement": "complete_action_modal_nll",
        "mcts_modal_top1_improvement": "mcts_modal_top1",
    }
    summary: dict[str, Any] = {}
    for output_name, source_key in deltas.items():
        lower_is_better = (
            output_name
            in {"conditional_cross_entropy_improvement", "conditional_kl_improvement"}
            or output_name.endswith("nll_improvement")
        )
        if lower_is_better:
            values = [
                -float(item[f"delta_{source_key}"]) for item in differences
            ]
        else:
            values = [float(item[f"delta_{source_key}"]) for item in differences]
        rows = [{**item, "improvement": value} for item, value in zip(differences, values)]
        summary[output_name] = {
            "definition": specifications[output_name],
            "positive_is_better": True,
            "mean": _mean(rows, "improvement"),
            "bootstrap": _paired_bootstrap(
                rows,
                "improvement",
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
        }
    tail_rows = [
        row
        for row in differences
        if row.get("delta_visited_tail_recall_at_3") is not None
    ]
    summary["visited_tail_recall_at_3_improvement"] = {
        "definition": "opened visited-tail recall@3 - experiment visited-tail recall@3",
        "positive_is_better": True,
        "mean": (
            _mean(tail_rows, "delta_visited_tail_recall_at_3")
            if tail_rows
            else None
        ),
        "eligible_decisions": len(tail_rows),
        "bootstrap": (
            _paired_bootstrap(
                tail_rows,
                "delta_visited_tail_recall_at_3",
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            if tail_rows
            else None
        ),
    }
    ece_bin_count = len(opened.get("ece_bins", ()))
    if ece_bin_count == 0 or len(control.get("ece_bins", ())) != ece_bin_count:
        raise Phase1ABError("A/B ECE bin definitions differ")
    opened_ece, _ = _calibration_bins(opened_rows, ece_bin_count)
    control_ece, _ = _calibration_bins(control_rows, ece_bin_count)
    if not math.isclose(opened_ece, float(opened["mcts_modal_ece"]), abs_tol=1e-12):
        raise Phase1ABError("opened aggregate ECE does not match per-decision rows")
    if not math.isclose(control_ece, float(control["mcts_modal_ece"]), abs_tol=1e-12):
        raise Phase1ABError("control aggregate ECE does not match per-decision rows")
    summary["mcts_modal_ece_reduction"] = {
        "definition": "control aggregate ECE - opened aggregate ECE",
        "positive_is_better": True,
        "mean": control_ece - opened_ece,
        "bootstrap": _paired_ece_bootstrap(
            opened_rows,
            control_rows,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            ece_bins=ece_bin_count,
        ),
    }
    return {
        "paired_decisions": len(differences),
        "game_blocks": len({str(item["game_id"]) for item in differences}),
        "improvements": summary,
        "per_decision": differences,
    }


def build_phase1_model_gate(
    opened: Mapping[str, Any],
    control: Mapping[str, Any],
    comparison: Mapping[str, Any],
    config: Phase1ABConfig,
    *,
    parity_complete: bool,
) -> dict[str, Any]:
    """Apply the pre-declared Phase 1 model/evaluation rule without promotion."""

    improvements = comparison["improvements"]

    def consistent_improvement(name: str) -> dict[str, Any]:
        metric = improvements[name]
        bootstrap = metric.get("bootstrap")
        mean = metric.get("mean")
        ci95_low = bootstrap.get("ci95_low") if bootstrap else None
        sufficient = mean is not None and ci95_low is not None
        passed = bool(sufficient and float(mean) > 0.0 and float(ci95_low) > 0.0)
        return {
            "metric": name,
            "mean": mean,
            "ci95_low": ci95_low,
            "consistent_improvement": passed,
            "evidence_sufficient": sufficient,
        }

    conditional_kl = consistent_improvement("conditional_kl_improvement")
    tail_recall = consistent_improvement("visited_tail_recall_at_3_improvement")
    primary_passed = bool(
        conditional_kl["consistent_improvement"]
        or tail_recall["consistent_improvement"]
    )
    primary_sufficient = bool(
        conditional_kl["evidence_sufficient"] or tail_recall["evidence_sufficient"]
    )
    nll_margin = float(opened["complete_action_soft_nll"]) - float(
        control["complete_action_soft_nll"]
    )
    top1_margin = float(opened["mcts_modal_top1"]) - float(
        control["mcts_modal_top1"]
    )
    ece_margin = float(opened["mcts_modal_ece"]) - float(
        control["mcts_modal_ece"]
    )
    non_degradation = {
        "complete_action_nll": {
            "opened_minus_control": nll_margin,
            "maximum_allowed_regression": config.nll_non_degradation_tolerance,
            "passed": nll_margin <= config.nll_non_degradation_tolerance,
        },
        "mcts_modal_top1": {
            "opened_minus_control": top1_margin,
            "maximum_allowed_regression": config.top1_non_degradation_tolerance,
            "passed": top1_margin >= -config.top1_non_degradation_tolerance,
        },
        "mcts_modal_ece": {
            "opened_minus_control": ece_margin,
            "maximum_allowed_regression": config.ece_non_degradation_tolerance,
            "passed": ece_margin <= config.ece_non_degradation_tolerance,
        },
    }
    non_degradation_passed = all(
        bool(item["passed"]) for item in non_degradation.values()
    )
    parity_passed = (not config.run_parity) or bool(parity_complete)
    if not primary_sufficient:
        outcome = "insufficient_evidence"
    elif primary_passed and non_degradation_passed and parity_passed:
        outcome = "pass"
    else:
        outcome = "fail"
    return {
        "schema_id": "augmentia.v4-phase1-model-gate.v1",
        "outcome": outcome,
        "primary_rule": (
            "opened-only consistently improves conditional KL or visited-tail "
            "recall@3 by paired game-bootstrap mean and 95% CI lower bound > 0"
        ),
        "primary": {
            "conditional_kl": conditional_kl,
            "visited_tail_recall_at_3": tail_recall,
            "passed": primary_passed,
        },
        "non_degradation": non_degradation,
        "non_degradation_passed": non_degradation_passed,
        "parity_required": config.run_parity,
        "parity_complete": parity_complete,
        "parity_passed": parity_passed,
        "legacy_all_open_experiment_export_blocked": True,
        "promotable": False,
        "promotion_note": (
            "This gate seals only the Phase 1 model/evaluation requirement. "
            "Phase 2 paired strength and CI evidence is still required."
        ),
    }


def build_export_command(
    checkpoint: str | Path,
    manifest: str | Path,
    output: str | Path,
    runtime_vocabulary: str | Path,
    *,
    python_executable: str | None = None,
) -> list[str]:
    return [
        python_executable or sys.executable,
        "-m",
        "v4.export_checkpoint",
        str(Path(checkpoint).resolve()),
        str(Path(manifest).resolve()),
        str(Path(output).resolve()),
        "--runtime-vocabulary",
        str(Path(runtime_vocabulary).resolve()),
    ]


def build_rust_parity_command(
    validator: str | Path,
    bundle_metadata: str | Path,
    parity_sample: str | Path,
    tolerance: float,
) -> list[str]:
    return [
        str(Path(validator).resolve()),
        "--bundle",
        str(Path(bundle_metadata).resolve()),
        "--parity",
        str(Path(parity_sample).resolve()),
        "--tolerance",
        str(tolerance),
    ]


def _run_checked(command: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise Phase1ABError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def run_export_and_rust_parity(
    *,
    arm_id: str,
    checkpoint: Path,
    manifest: Path,
    output: Path,
    runtime_vocabulary: Path,
    rust_validator: str | Path | None,
    tolerance: float,
    repo_root: Path,
    ort_dylib_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export the opened-only arm and execute eager/ONNX/Rust parity."""

    if arm_id != ARM_OPENED_ONLY:
        raise Phase1ABError(
            "legacy-all-open experiment checkpoints are blocked from production export"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    export_command = build_export_command(
        checkpoint, manifest, output, runtime_vocabulary
    )
    export_result = _run_checked(export_command, cwd=repo_root / "train")
    bundle = output.with_suffix(".v4-bundle.metadata.json")
    parity = output.with_suffix(".parity.json")
    export_report = output.with_suffix(".export-report.json")
    for path in (bundle, parity, export_report):
        if not path.is_file():
            raise Phase1ABError(f"export did not create required artifact: {path}")
    rust_path = output.with_suffix(".rust-parity.json")
    if rust_validator is None:
        raise Phase1ABError("Rust parity validator path is required")
    validator_path = Path(rust_validator)
    if not validator_path.is_absolute():
        validator_path = repo_root / validator_path
    validator_path = validator_path.resolve()
    if not validator_path.is_file():
        raise Phase1ABError(
            f"Rust parity validator is missing: {validator_path}"
        )
    rust_command = build_rust_parity_command(
        validator_path, bundle, parity, tolerance
    )
    env = dict(os.environ)
    if ort_dylib_path is not None:
        env["ORT_DYLIB_PATH"] = str(Path(ort_dylib_path).resolve())
    rust_result = _run_checked(rust_command, cwd=repo_root, env=env)
    try:
        rust_payload = json.loads(rust_result.stdout)
    except json.JSONDecodeError as error:
        raise Phase1ABError("Rust parity validator did not emit JSON") from error
    _atomic_json(rust_path, rust_payload)
    receipt_path = output.with_suffix(".parity-receipt.json")
    receipt = {
        "schema_id": PHASE1_AB_PARITY_RECEIPT_SCHEMA_ID,
        "receipt_path": str(receipt_path),
        "arm_id": arm_id,
        "export_command": export_command,
        "export_returncode": export_result.returncode,
        "export_report_path": str(export_report),
        "export_report_sha256": sha256_file(export_report),
        "bundle_metadata_path": str(bundle),
        "bundle_metadata_sha256": sha256_file(bundle),
        "parity_sample_path": str(parity),
        "parity_sample_sha256": sha256_file(parity),
        "rust_validator": str(validator_path),
        "rust_validator_sha256": sha256_file(validator_path),
        "rust_command": rust_command,
        "rust_returncode": rust_result.returncode,
        "rust_report_path": str(rust_path),
        "rust_report_sha256": sha256_file(rust_path),
        "tolerance": tolerance,
    }
    receipt = _seal_payload(receipt, "receipt_sha256")
    _write_idempotent_sealed_json(receipt_path, receipt, "receipt_sha256")
    return {
        **receipt,
        "receipt_file_sha256": sha256_file(receipt_path),
    }


def _run_one_arm(
    *,
    arm_id: str,
    view: TrainingViewBundle,
    corpus: ControlledCorpus,
    config: Phase1ABConfig,
    vocabulary: V3Vocabulary,
    plan_sha256: str,
    initialization_sha256: str,
    output_dir: Path,
    manifest_path: Path,
    repo_root: Path,
    rust_validator: str | Path | None,
    ort_dylib_path: str | Path | None,
) -> ArmRunResult:
    run_dir = output_dir / arm_id
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = _arm_provenance(
        corpus, view, arm_id, plan_sha256, initialization_sha256
    )
    training_validation_view = make_training_view(
        corpus.validation_examples,
        "all_open_control" if view.is_control else "opened_only",
    )
    provenance.update(
        {
            "training_validation_view_id": training_validation_view.view_id,
            "training_validation_view_sha256": training_validation_view.view_sha256,
            "source_validation_view_sha256": training_validation_view.source_view_sha256,
        }
    )
    trainer = _build_trainer(
        view.examples,
        training_validation_view.examples,
        run_dir,
        config,
        vocabulary,
        provenance,
    )
    actual_initialization = model_state_sha256(trainer.model)
    if actual_initialization != initialization_sha256:
        raise Phase1ABError(f"{arm_id} initialization drifted before training")
    run_payload = _seal_payload(
        {
            "schema_id": PHASE1_AB_ARM_RUN_SCHEMA_ID,
            "arm_id": arm_id,
            "plan_sha256": plan_sha256,
            "source_corpus_sha256": corpus.source_corpus_sha256,
            "split_sha256": corpus.split_sha256,
            "training_view": view.to_json(),
            "policy_target_semantics_id": view.policy_target_semantics_id,
            "policy_target_semantics_lineage": dict(
                view.policy_target_semantics_lineage
            ),
            "initialization_sha256": initialization_sha256,
            "training_config": asdict(config.trainer),
            "provenance": provenance,
            "promotable": False,
        },
        "run_sha256",
    )
    run_path = _write_idempotent_sealed_json(
        run_dir / "run.json", run_payload, "run_sha256"
    )
    trainer.fit(checkpoint_every=config.checkpoint_every)
    checkpoint, sidecar = trainer.save_checkpoint(run_dir / "final.pt")
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    for key, expected in (
        ("phase1_ab_plan_sha256", plan_sha256),
        ("phase1_ab_arm_id", arm_id),
        ("training_view_id", view.view_id),
        ("training_view_sha256", view.view_sha256),
        ("policy_target_semantics_id", view.policy_target_semantics_id),
        ("policy_target_semantics_experiment_only", bool(
            view.policy_target_semantics_lineage["experiment_only"]
        )),
        ("source_corpus_sha256", corpus.source_corpus_sha256),
        ("split_sha256", corpus.split_sha256),
        ("phase1_initialization_sha256", initialization_sha256),
    ):
        actual = sidecar_payload.get("provenance", {}).get(key)
        if actual != expected:
            raise Phase1ABError(f"{arm_id} checkpoint provenance {key} drifted")
    if sidecar_payload["provenance"]["policy_target_semantics_lineage"] != dict(
        view.policy_target_semantics_lineage
    ):
        raise Phase1ABError(
            f"{arm_id} checkpoint policy target semantics lineage drifted"
        )
    evaluation = evaluate_common_view(
        trainer.model,
        corpus.validation_examples,
        vocabulary,
        trainer.device,
        batch_size=config.trainer.batch_size,
        ece_bins=config.ece_bins,
    )
    evaluation.update(
        {
            "arm_id": arm_id,
            "training_view_id": view.view_id,
            "policy_target_semantics_id": view.policy_target_semantics_id,
            "policy_target_semantics_experiment_only": bool(
                view.policy_target_semantics_lineage["experiment_only"]
            ),
            "checkpoint_sha256": sha256_file(checkpoint),
            "source_corpus_sha256": corpus.source_corpus_sha256,
            "split_sha256": corpus.split_sha256,
            "initialization_sha256": initialization_sha256,
        }
    )
    evaluation = _seal_payload(evaluation, "evaluation_sha256")
    evaluation_path = _write_idempotent_sealed_json(
        run_dir / "heldout-evaluation.json", evaluation, "evaluation_sha256"
    )
    parity = None
    if config.run_parity and not view.is_control:
        parity = run_export_and_rust_parity(
            arm_id=arm_id,
            checkpoint=checkpoint,
            manifest=manifest_path,
            output=run_dir / "parity" / "model",
            runtime_vocabulary=config.vocabulary_path,
            rust_validator=rust_validator,
            tolerance=config.parity_tolerance,
            repo_root=repo_root,
            ort_dylib_path=ort_dylib_path,
        )
    result = ArmRunResult(
        arm_id=arm_id,
        training_view=view,
        initialization_sha256=initialization_sha256,
        run_json_path=run_path,
        run_json_file_sha256=sha256_file(run_path),
        run_json_seal_sha256=run_payload["run_sha256"],
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        sidecar_path=sidecar,
        sidecar_sha256=sha256_file(sidecar),
        evaluation=evaluation,
        evaluation_path=evaluation_path,
        evaluation_file_sha256=sha256_file(evaluation_path),
        evaluation_seal_sha256=evaluation["evaluation_sha256"],
        parity=parity,
    )
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_phase1_ab(
    manifest: str | Path,
    output_dir: str | Path,
    config: Phase1ABConfig,
    *,
    rust_validator: str | Path | None = None,
    ort_dylib_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the controlled two-arm experiment and write immutable evidence."""

    effective_rust_validator = (
        str(rust_validator) if rust_validator is not None else config.rust_validator
    )
    effective_ort_dylib = (
        str(ort_dylib_path)
        if ort_dylib_path is not None
        else config.ort_dylib_path
    )
    config = replace(
        config,
        rust_validator=effective_rust_validator,
        ort_dylib_path=effective_ort_dylib,
    ).validate()
    manifest_path = Path(manifest).resolve()
    root = Path(output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    commit, dirty = _git_identity(repo_root)
    if dirty and not config.allow_dirty_smoke:
        raise Phase1ABError("dirty worktree requires allow_dirty_smoke=True")
    if dirty and config.allow_dirty_smoke and "smoke" not in root.name.lower():
        raise Phase1ABError("dirty smoke output directory name must contain 'smoke'")
    vocabulary, vocabulary_identity = load_runtime_vocabulary(
        config.vocabulary_path
    )
    corpus = load_controlled_corpus(manifest_path, config, vocabulary_identity)
    corpus = replace(
        corpus,
        provenance={
            **corpus.provenance,
            "trainer_git_commit": commit,
            "trainer_dirty_worktree": dirty,
        },
    )
    opened = make_training_view(corpus.train_examples, "opened_only")
    control = make_training_view(corpus.train_examples, "all_open_control")
    validation_view = make_training_view(
        corpus.validation_examples, "opened_only"
    )
    if opened.view_sha256 != _view_sha256(
        TRAINING_VIEW_OPENED_ONLY, "opened_only", corpus.train_examples
    ):
        raise Phase1ABError("opened-only source view hash drifted")
    initialization = prove_shared_initialization(
        corpus, opened, control, config, vocabulary, root / ".init-probe"
    )
    plan = build_phase1_ab_plan(
        corpus,
        opened,
        control,
        config,
        vocabulary_identity,
        initialization,
        {"commit": commit, "dirty": dirty},
    )
    plan_path = _write_idempotent_sealed_json(
        root / "ab-plan.json", plan, "plan_sha256"
    )
    arms = {
        ARM_OPENED_ONLY: _run_one_arm(
            arm_id=ARM_OPENED_ONLY,
            view=opened,
            corpus=corpus,
            config=config,
            vocabulary=vocabulary,
            plan_sha256=plan["plan_sha256"],
            initialization_sha256=initialization,
            output_dir=root,
            manifest_path=manifest_path,
            repo_root=repo_root,
            rust_validator=effective_rust_validator,
            ort_dylib_path=effective_ort_dylib,
        ),
        ARM_ALL_OPEN_CONTROL: _run_one_arm(
            arm_id=ARM_ALL_OPEN_CONTROL,
            view=control,
            corpus=corpus,
            config=config,
            vocabulary=vocabulary,
            plan_sha256=plan["plan_sha256"],
            initialization_sha256=initialization,
            output_dir=root,
            manifest_path=manifest_path,
            repo_root=repo_root,
            rust_validator=effective_rust_validator,
            ort_dylib_path=effective_ort_dylib,
        ),
    }
    bootstrap_seed = (
        config.bootstrap_seed
        if config.bootstrap_seed is not None
        else config.trainer.seed + 0xA51B
    )
    comparison = compare_evaluations(
        arms[ARM_OPENED_ONLY].evaluation,
        arms[ARM_ALL_OPEN_CONTROL].evaluation,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    parity_complete = arms[ARM_OPENED_ONLY].parity is not None
    model_gate = build_phase1_model_gate(
        arms[ARM_OPENED_ONLY].evaluation,
        arms[ARM_ALL_OPEN_CONTROL].evaluation,
        comparison,
        config,
        parity_complete=parity_complete,
    )
    summary = {
        "schema_id": PHASE1_AB_SUMMARY_SCHEMA_ID,
        "status": "complete",
        "plan_path": str(plan_path),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "source_corpus_sha256": corpus.source_corpus_sha256,
        "split_sha256": corpus.split_sha256,
        "initialization_sha256": initialization,
        "evaluation_view_id": TRAINING_VIEW_OPENED_ONLY,
        "evaluation_view_sha256": validation_view.view_sha256,
        "arms": {arm_id: result.to_json() for arm_id, result in arms.items()},
        "comparison": comparison,
        "model_gate": model_gate,
        "parity_scope": "opened-only-production-semantics",
        "parity_complete": parity_complete,
        "legacy_all_open_experiment_export_blocked": True,
        "promotable": False,
        "promotion_note": (
            "All arms remain non-promotable; the legacy-all-open arm is an "
            "explicit experiment blocked from production export/resume, and Phase 1 "
            "does not seal Phase 2 strength gates."
        ),
    }
    summary = _seal_payload(summary, "summary_sha256")
    _write_idempotent_sealed_json(
        root / "ab-summary.json", summary, "summary_sha256"
    )
    return summary


def _default_rust_validator(repo_root: Path) -> Path:
    name = "validate_v4_bundle.exe" if os.name == "nt" else "validate_v4_bundle"
    return repo_root / "target" / "release" / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=2026091401)
    parser.add_argument("--validation-bp", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=6)
    parser.add_argument("--runtime-message-passes", type=int, default=2)
    parser.add_argument(
        "--batching",
        dest="batching_kind",
        choices=("random", "bucketed", "cached_bucketed"),
        default="random",
    )
    parser.add_argument("--tensor-cache-max-batches", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument(
        "--runtime-vocabulary",
        type=Path,
        default=default_runtime_vocabulary_path(),
    )
    parser.add_argument("--state-encoder-checkpoint", type=Path)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--parity-tolerance", type=float, default=2e-4)
    parser.add_argument("--nll-non-degradation-tolerance", type=float, default=0.02)
    parser.add_argument("--top1-non-degradation-tolerance", type=float, default=0.01)
    parser.add_argument("--ece-non-degradation-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--rust-validator",
        type=Path,
        default=_default_rust_validator(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument("--ort-dylib", type=Path)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--allow-smoke-corpus", action="store_true")
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()

    trainer = V4TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        value_weight=args.value_weight,
        seed=args.seed,
        device=args.device,
        model_dim=args.model_dim,
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        runtime_message_passes=args.runtime_message_passes,
        encoder_kind="baseline",
        batching_kind=args.batching_kind,
        tensor_cache_max_batches=args.tensor_cache_max_batches,
    )
    config = Phase1ABConfig(
        trainer=trainer,
        validation_basis_points=args.validation_bp,
        max_train_examples=args.max_train_examples,
        max_validation_examples=args.max_validation_examples,
        checkpoint_every=args.checkpoint_every,
        runtime_vocabulary_path=str(args.runtime_vocabulary),
        state_encoder_checkpoint=(
            str(args.state_encoder_checkpoint) if args.state_encoder_checkpoint else None
        ),
        ece_bins=args.ece_bins,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        parity_tolerance=args.parity_tolerance,
        nll_non_degradation_tolerance=args.nll_non_degradation_tolerance,
        top1_non_degradation_tolerance=args.top1_non_degradation_tolerance,
        ece_non_degradation_tolerance=args.ece_non_degradation_tolerance,
        run_parity=not args.skip_parity,
        rust_validator=str(args.rust_validator),
        ort_dylib_path=(str(args.ort_dylib) if args.ort_dylib else None),
        allow_smoke_corpus=args.allow_smoke_corpus,
        allow_dirty_smoke=args.allow_dirty_smoke,
    )
    summary = run_phase1_ab(
        args.manifest,
        args.output_dir,
        config,
        rust_validator=args.rust_validator,
        ort_dylib_path=args.ort_dylib,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

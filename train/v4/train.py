"""Production-oriented trainer for the V4 factorized conditional policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from v3.factorized import FACTOR_POLICY_SCHEMA_ID, factorized_action_path
from v3.tensorize import V3Vocabulary

from .batch import tensorize_factorized_cases
from .dataset import DatasetInputV4, load_dataset_input
from .loss import conditional_prefix_cross_entropy
from .model import AugmentNetV4, V4ModelConfig
from .replay import FactorizedTrainingExampleV4, load_game_v4

V4_TRAINER_SCHEMA_ID = "augmentia.v4-trainer-checkpoint.v2"
V4_TRAINING_RUN_SCHEMA_ID = "augmentia.v4-training-run.v1"
RUNTIME_VOCABULARY_SCHEMA_ID = "augmentia.runtime-vocabulary.v3"
RULESET_ID = "augment_chess_20260915"
SOURCE_SEAL_TAG = "ai/v4-training-base-20260919"
SOURCE_SEAL_COMMIT = "f4e64c35f5adcb7f361ed845fb55eb4d123b370b"


@dataclass(frozen=True)
class V4TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 1
    value_weight: float = 1.0
    seed: int = 2026091401
    device: str = "cpu"
    model_dim: int = 128
    channels: int = 128
    residual_blocks: int = 6
    runtime_message_passes: int = 2
    encoder_kind: str = "baseline"
    hybrid_layers: int = 2
    hybrid_heads: int = 4
    batching_kind: str = "random"

    def validate(self) -> "V4TrainingConfig":
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or self.batch_size <= 0
            or self.epochs <= 0
        ):
            raise ValueError("learning rate, batch size, and epochs must be positive")
        if not math.isfinite(self.value_weight) or self.value_weight < 0.0:
            raise ValueError("value weight must be finite and non-negative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.encoder_kind not in {"baseline", "hybrid"}:
            raise ValueError("encoder_kind must be baseline or hybrid")
        if self.batching_kind not in {"random", "bucketed"}:
            raise ValueError("batching_kind must be random or bucketed")
        if min(
            self.model_dim,
            self.channels,
            self.residual_blocks,
            self.runtime_message_passes,
            self.hybrid_layers,
            self.hybrid_heads,
        ) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        return self


def default_runtime_vocabulary_path() -> Path:
    return Path(__file__).parents[1] / "v3" / "fixtures" / "0915" / "runtime_vocabulary_v3.json"


def load_runtime_vocabulary(path: str | Path) -> tuple[V3Vocabulary, dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_id") != RUNTIME_VOCABULARY_SCHEMA_ID:
        raise ValueError("runtime vocabulary schema mismatch")
    if payload.get("ruleset_id") != RULESET_ID:
        raise ValueError("runtime vocabulary ruleset mismatch")
    tokens = payload.get("tokens")
    piece_kind_codes = payload.get("piece_kind_codes")
    if (
        not isinstance(tokens, list)
        or not tokens
        or not all(isinstance(token, str) and token for token in tokens)
        or len(set(tokens)) != len(tokens)
        or "<none>" in tokens
    ):
        raise ValueError("runtime vocabulary tokens are invalid")
    if (
        not isinstance(piece_kind_codes, list)
        or piece_kind_codes != list(range(1, 69))
    ):
        raise ValueError("0915 piece-kind vocabulary must contain stable codes 1..=68")
    vocabulary = V3Vocabulary(("<none>", *tokens))
    identity = {
        "path": str(source),
        "sha256": _sha256_file(source),
        "schema_id": payload["schema_id"],
        "ruleset_id": payload["ruleset_id"],
        "token_count": len(tokens),
        "model_vocabulary_size": vocabulary.size,
        "piece_kind_codes": piece_kind_codes,
    }
    return vocabulary, identity


def _factorized_shape_key(
    example: FactorizedTrainingExampleV4,
) -> tuple[int, int, int, int]:
    """Return deterministic padding drivers for one factorized decision."""
    actions = example.legal_actions["actions"]
    children_by_prefix: dict[tuple[Any, ...], set[Any]] = {}
    for action in actions:
        path = factorized_action_path(action)
        for depth, token in enumerate(path):
            prefix = tuple(path[:depth])
            children_by_prefix.setdefault(prefix, set()).add(token)
    max_children = max(
        (len(children) for children in children_by_prefix.values()), default=0
    )
    max_prefix = max((len(prefix) for prefix in children_by_prefix), default=0)
    return max_children, len(actions), max_prefix, len(children_by_prefix)


class V4Trainer:
    """Train the conditional scorer and value head with a fixed runtime vocabulary."""

    def __init__(
        self,
        examples: Sequence[FactorizedTrainingExampleV4],
        run_dir: str | Path,
        config: V4TrainingConfig | None = None,
        state_encoder_checkpoint: str | Path | None = None,
        *,
        vocabulary: V3Vocabulary | None = None,
        model_config: V4ModelConfig | None = None,
        validation_examples: Sequence[FactorizedTrainingExampleV4] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = (config or V4TrainingConfig()).validate()
        self.run_dir = Path(run_dir)
        self.examples = list(examples)
        self.validation_examples = list(validation_examples)
        self._training_shape_keys = (
            [_factorized_shape_key(example) for example in self.examples]
            if self.config.batching_kind == "bucketed"
            else None
        )
        if not self.examples:
            raise ValueError("V4 training requires at least one decision example")
        self.vocabulary = vocabulary or V3Vocabulary.from_cases(
            [
                {
                    "observation": example.observation,
                    "legal_actions": example.legal_actions,
                }
                for example in self.examples
            ]
        )
        if model_config is None:
            model_config = V4ModelConfig(
                text_vocabulary_size=self.vocabulary.size,
                channels=self.config.channels,
                residual_blocks=self.config.residual_blocks,
                model_dim=self.config.model_dim,
                runtime_message_passes=self.config.runtime_message_passes,
                encoder_kind=self.config.encoder_kind,
                hybrid_layers=self.config.hybrid_layers,
                hybrid_heads=self.config.hybrid_heads,
            )
        if model_config.text_vocabulary_size != self.vocabulary.size:
            raise ValueError("model config and vocabulary size disagree")
        if model_config.encoder_kind != self.config.encoder_kind:
            raise ValueError("model config and training encoder_kind disagree")

        _seed_everything(self.config.seed)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA training requested but torch.cuda.is_available() is false")
        self.model = AugmentNetV4(model_config)
        if state_encoder_checkpoint is not None:
            state = torch.load(state_encoder_checkpoint, map_location="cpu", weights_only=True)
            _missing, unexpected = self.model.state_encoder.load_state_dict(state, strict=False)
            if unexpected:
                raise ValueError(
                    f"state encoder checkpoint has unexpected keys: {sorted(unexpected)}"
                )
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.completed_epochs = 0
        self.history: list[dict[str, Any]] = []
        self.provenance = dict(provenance or {})
        self.source_checkpoint_sha256: str | None = None
        self.stop_requested = False
        self.paused = False
        self.pause_checkpoint: Path | None = None
        self.epoch_progress: dict[str, Any] | None = None
        self.pending_validation: dict[str, Any] | None = None
        self._resume_target_epochs: int | None = None

    def batch_count(self, examples: Sequence[FactorizedTrainingExampleV4] | None = None) -> int:
        selected = self.examples if examples is None else examples
        size = self.config.batch_size
        return (len(selected) + size - 1) // size

    def _batches(
        self,
        examples: Sequence[FactorizedTrainingExampleV4],
        *,
        shuffle: bool,
        epoch: int,
        start_batch: int = 0,
    ):
        order = list(range(len(examples)))
        if shuffle:
            rng = random.Random(self.config.seed + epoch)
            if self.config.batching_kind == "bucketed" and examples is self.examples:
                if self._training_shape_keys is None:
                    raise RuntimeError("bucketed batching shape keys are unavailable")
                order.sort(key=lambda index: (*self._training_shape_keys[index], index))
                size = self.config.batch_size
                batches = [
                    order[start : start + size]
                    for start in range(0, len(order), size)
                ]
                for indexes in batches:
                    rng.shuffle(indexes)
                rng.shuffle(batches)
                order = [index for indexes in batches for index in indexes]
            else:
                rng.shuffle(order)
        size = self.config.batch_size
        for batch_index, start in enumerate(range(0, len(order), size)):
            if batch_index < start_batch:
                continue
            selected = [examples[index] for index in order[start : start + size]]
            cases = [
                {"observation": item.observation, "legal_actions": item.legal_actions}
                for item in selected
            ]
            visits = [list(item.complete_visit_counts) for item in selected]
            batch = tensorize_factorized_cases(cases, self.vocabulary, visits)
            values = [item.value for item in selected]
            yield batch_index, batch, values

    def _loss_for_batch(
        self,
        batch: Any,
        values: Sequence[float | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        output = self.model(batch)
        policy_loss = conditional_prefix_cross_entropy(
            output.prefix_logits, batch.prefixes
        )
        targets = [item for item in values if item is not None]
        if not targets:
            return policy_loss, policy_loss, None, 0
        value_tensor = torch.tensor(
            targets, dtype=torch.float32, device=self.device
        )
        mask = torch.tensor(
            [item is not None for item in values],
            dtype=torch.bool,
            device=self.device,
        )
        value_loss = torch.nn.functional.mse_loss(output.value[mask], value_tensor)
        loss = policy_loss + self.config.value_weight * value_loss
        return loss, policy_loss, value_loss, len(targets)

    def train_epoch(self) -> dict[str, float | int] | None:
        self.model.train()
        epoch = self.completed_epochs
        if self.epoch_progress is None:
            self.epoch_progress = {
                "epoch": epoch,
                "next_batch": 0,
                "total_total": 0.0,
                "policy_total": 0.0,
                "value_total": 0.0,
                "value_batches": 0,
                "value_samples": 0,
                "batches": 0,
                "elapsed_seconds": 0.0,
            }
        progress = self.epoch_progress
        if int(progress.get("epoch", -1)) != epoch:
            raise ValueError("checkpoint epoch progress does not match completed_epochs")
        started = time.perf_counter()
        try:
            if self.stop_requested:
                return None
            for batch_index, batch, values in self._batches(
                self.examples,
                shuffle=True,
                epoch=epoch,
                start_batch=int(progress["next_batch"]),
            ):
                batch = batch.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss, policy_loss, value_loss, supervised = self._loss_for_batch(
                    batch, values
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("training loss became NaN or infinity")
                loss.backward()
                self.optimizer.step()
                progress["total_total"] += float(loss.detach())
                progress["policy_total"] += float(policy_loss.detach())
                if value_loss is not None:
                    progress["value_total"] += float(value_loss.detach())
                    progress["value_batches"] += 1
                    progress["value_samples"] += supervised
                progress["batches"] += 1
                progress["next_batch"] = batch_index + 1
                if self.stop_requested:
                    return None
        finally:
            progress["elapsed_seconds"] += time.perf_counter() - started
        batches = int(progress["batches"])
        value_batches = int(progress["value_batches"])
        metrics = {
            "loss": float(progress["total_total"]) / max(1, batches),
            "policy_loss": float(progress["policy_total"]) / max(1, batches),
            "value_loss": float(progress["value_total"]) / max(1, value_batches),
            "batches": batches,
            "examples": len(self.examples),
            "value_supervised_examples": int(progress["value_samples"]),
        }
        train_elapsed_seconds = float(progress["elapsed_seconds"])
        self.completed_epochs += 1
        self.epoch_progress = None
        self.pending_validation = {
            "epoch": self.completed_epochs,
            "train": metrics,
            "train_elapsed_seconds": train_elapsed_seconds,
        }
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        examples: Sequence[FactorizedTrainingExampleV4] | None = None,
    ) -> dict[str, float | int] | None:
        selected = self.validation_examples if examples is None else list(examples)
        if not selected:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "batches": 0,
                "examples": 0,
                "value_supervised_examples": 0,
            }
        self.model.eval()
        total_total = 0.0
        policy_total = 0.0
        value_total = 0.0
        value_batches = 0
        value_samples = 0
        batches = 0
        if self.stop_requested:
            return None
        for _batch_index, batch, values in self._batches(
            selected, shuffle=False, epoch=self.completed_epochs
        ):
            batch = batch.to(self.device)
            loss, policy_loss, value_loss, supervised = self._loss_for_batch(
                batch, values
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("validation loss became NaN or infinity")
            total_total += float(loss.detach())
            policy_total += float(policy_loss.detach())
            if value_loss is not None:
                value_total += float(value_loss.detach())
                value_batches += 1
                value_samples += supervised
            batches += 1
            if self.stop_requested:
                return None
        return {
            "loss": total_total / max(1, batches),
            "policy_loss": policy_total / max(1, batches),
            "value_loss": value_total / max(1, value_batches),
            "batches": batches,
            "examples": len(selected),
            "value_supervised_examples": value_samples,
        }

    def fit(
        self,
        *,
        checkpoint_every: int = 0,
        checkpoint_prefix: str = "checkpoint",
    ) -> list[dict[str, Any]]:
        target_epochs = (
            self._resume_target_epochs
            if self._resume_target_epochs is not None
            else self.completed_epochs + self.config.epochs
        )
        while self.pending_validation is not None or self.completed_epochs < target_epochs:
            if self.pending_validation is None:
                train_metrics = self.train_epoch()
                if train_metrics is None:
                    self._save_interrupt_checkpoint()
                    return self.history
            pending = self.pending_validation
            if pending is None:
                raise RuntimeError("training epoch completed without pending validation state")
            validation_started = time.perf_counter()
            validation_metrics = self.evaluate()
            if validation_metrics is None:
                self._save_interrupt_checkpoint()
                return self.history
            event = {
                "epoch": int(pending["epoch"]),
                "elapsed_seconds": float(pending["train_elapsed_seconds"])
                + (time.perf_counter() - validation_started),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": dict(pending["train"]),
                "validation": validation_metrics,
                "cuda_memory_allocated_bytes": (
                    torch.cuda.memory_allocated(self.device)
                    if self.device.type == "cuda"
                    else 0
                ),
                "cuda_memory_reserved_bytes": (
                    torch.cuda.memory_reserved(self.device)
                    if self.device.type == "cuda"
                    else 0
                ),
                "cuda_max_memory_allocated_bytes": (
                    torch.cuda.max_memory_allocated(self.device)
                    if self.device.type == "cuda"
                    else 0
                ),
            }
            self.pending_validation = None
            self.history.append(event)
            self._write_history()
            print(json.dumps(event, sort_keys=True), flush=True)
            if checkpoint_every and self.completed_epochs % checkpoint_every == 0:
                self.save_checkpoint(
                    self.run_dir
                    / f"{checkpoint_prefix}-epoch-{self.completed_epochs:04d}.pt"
                )
            if self.stop_requested:
                self._save_interrupt_checkpoint()
                return self.history
        return self.history

    def _save_interrupt_checkpoint(self) -> tuple[Path, Path]:
        self.paused = True
        self.run_dir.mkdir(parents=True, exist_ok=True)
        target = self.run_dir / "interrupt.pt"
        checkpoint, sidecar = self.save_checkpoint(target)
        self.pause_checkpoint = checkpoint
        print(
            json.dumps(
                {
                    "event": "v4_training_paused",
                    "checkpoint": str(checkpoint),
                    "completed_epochs": self.completed_epochs,
                    "next_batch": (
                        None
                        if self.epoch_progress is None
                        else int(self.epoch_progress["next_batch"])
                    ),
                    "pending_validation": self.pending_validation is not None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return checkpoint, sidecar

    def _write_history(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.run_dir / "history.json", self.history)

    def save_checkpoint(self, path: str | Path) -> tuple[Path, Path]:
        """Write model weights, optimizer state, and the immutable resume sidecar."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), target)
        optimizer_path = target.with_suffix(".optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        checkpoint_sha256 = _sha256_file(target)
        optimizer_sha256 = _sha256_file(optimizer_path)
        sidecar = target.with_suffix(".v4-checkpoint.json")
        payload = {
            "schema_id": V4_TRAINER_SCHEMA_ID,
            "checkpoint_sha256": checkpoint_sha256,
            "optimizer_path": optimizer_path.name,
            "optimizer_sha256": optimizer_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "vocabulary": self.vocabulary.to_json(),
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.config),
            "completed_epochs": self.completed_epochs,
            "example_count": len(self.examples),
            "validation_example_count": len(self.validation_examples),
            "history": self.history,
            "provenance": self.provenance,
            "epoch_progress": self.epoch_progress,
            "pending_validation": self.pending_validation,
            "paused": self.paused,
            "rng_state": _capture_rng_state(),
            "environment": _environment_identity(self.device),
        }
        _atomic_json(sidecar, payload)
        return target, sidecar

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        examples: Sequence[FactorizedTrainingExampleV4],
        config: V4TrainingConfig | None = None,
        *,
        validation_examples: Sequence[FactorizedTrainingExampleV4] = (),
        expected_vocabulary: V3Vocabulary | None = None,
        expected_provenance: Mapping[str, Any] | None = None,
    ) -> "V4Trainer":
        """Resume using the checkpoint sidecar vocabulary and model config."""
        checkpoint_path = Path(checkpoint_path)
        sidecar = checkpoint_path.with_suffix(".v4-checkpoint.json")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("schema_id") != V4_TRAINER_SCHEMA_ID:
            raise ValueError("checkpoint sidecar is not a production V4 trainer checkpoint")
        actual_checkpoint_hash = _sha256_file(checkpoint_path)
        if payload.get("checkpoint_sha256") != actual_checkpoint_hash:
            raise ValueError("checkpoint SHA-256 differs from its sidecar")
        vocabulary = V3Vocabulary.from_json(payload["vocabulary"])
        if expected_vocabulary is not None and vocabulary != expected_vocabulary:
            raise ValueError("resume vocabulary differs from the required runtime vocabulary")
        model_config = V4ModelConfig(**payload["model_config"])

        saved_training = payload["training_config"]
        if config is None:
            config = V4TrainingConfig(
                **{
                    key: value
                    for key, value in saved_training.items()
                    if key in V4TrainingConfig.__dataclass_fields__
                }
            )
        else:
            for field in (
                "model_dim",
                "channels",
                "residual_blocks",
                "runtime_message_passes",
                "encoder_kind",
                "hybrid_layers",
                "hybrid_heads",
                "batching_kind",
            ):
                saved_value = (
                    saved_training.get(field, "random")
                    if field == "batching_kind"
                    else saved_training[field]
                )
                if getattr(config, field) != saved_value:
                    raise ValueError(f"resume model/training field {field} changed")
            if payload.get("paused"):
                for field in (
                    "batch_size",
                    "learning_rate",
                    "value_weight",
                    "seed",
                    "device",
                    "batching_kind",
                ):
                    saved_value = (
                        saved_training.get(field, "random")
                        if field == "batching_kind"
                        else saved_training[field]
                    )
                    if getattr(config, field) != saved_value:
                        raise ValueError(
                            f"paused resume training field {field} changed"
                        )
        provenance = dict(payload.get("provenance", {}))
        if expected_provenance is not None:
            fields = [
                "ruleset_id",
                "ruleset_manifest_sha256",
                "factorized_policy_schema_id",
                "dataset_manifest_sha256",
            ]
            if payload.get("paused"):
                fields.extend(
                    [
                        "max_train_examples",
                        "max_validation_examples",
                        "train_game_ids",
                        "validation_game_ids",
                        "trainer_git_commit",
                        "trainer_dirty_worktree",
                    ]
                )
            for field in fields:
                if provenance.get(field) != expected_provenance.get(field):
                    raise ValueError(f"resume provenance field {field} changed")

        trainer = cls(
            examples,
            checkpoint_path.parent,
            config,
            vocabulary=vocabulary,
            model_config=model_config,
            validation_examples=validation_examples,
            provenance=provenance,
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        trainer.model.load_state_dict(state, strict=True)
        optimizer_path = checkpoint_path.parent / payload["optimizer_path"]
        if not optimizer_path.exists():
            raise ValueError("resume optimizer state is missing")
        if _sha256_file(optimizer_path) != payload.get("optimizer_sha256"):
            raise ValueError("resume optimizer SHA-256 differs from sidecar")
        optimizer_state = torch.load(
            optimizer_path, map_location="cpu", weights_only=True
        )
        trainer.optimizer.load_state_dict(optimizer_state)
        _optimizer_to(trainer.optimizer, trainer.device)
        trainer.completed_epochs = int(payload.get("completed_epochs", 0))
        trainer.history = list(payload.get("history", []))
        trainer.epoch_progress = payload.get("epoch_progress")
        trainer.pending_validation = payload.get("pending_validation")
        if len(examples) != int(payload.get("example_count", len(examples))):
            raise ValueError("resume training example count changed")
        if len(validation_examples) != int(
            payload.get("validation_example_count", len(validation_examples))
        ):
            raise ValueError("resume validation example count changed")
        if trainer.epoch_progress is not None and trainer.pending_validation is not None:
            raise ValueError("checkpoint cannot contain both epoch progress and pending validation")
        if trainer.epoch_progress is not None:
            if int(trainer.epoch_progress.get("epoch", -1)) != trainer.completed_epochs:
                raise ValueError("checkpoint epoch progress is inconsistent")
            next_batch = int(trainer.epoch_progress.get("next_batch", -1))
            if not 0 <= next_batch <= trainer.batch_count():
                raise ValueError("checkpoint next batch is out of range")
        if trainer.pending_validation is not None and int(
            trainer.pending_validation.get("epoch", -1)
        ) != trainer.completed_epochs:
            raise ValueError("checkpoint pending validation epoch is inconsistent")
        if payload.get("paused"):
            if trainer.epoch_progress is not None and config.epochs <= trainer.completed_epochs:
                raise ValueError(
                    "paused mid-epoch resume requires --epochs above completed_epochs"
                )
            trainer._resume_target_epochs = config.epochs
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, dict):
            _restore_rng_state(rng_state)
        trainer.source_checkpoint_sha256 = actual_checkpoint_hash
        return trainer


def train_from_games(
    game_paths: Sequence[str | Path],
    run_dir: str | Path,
    config: V4TrainingConfig | None = None,
    *,
    vocabulary: V3Vocabulary | None = None,
) -> V4Trainer:
    examples: list[FactorizedTrainingExampleV4] = []
    for path in game_paths:
        examples.extend(load_game_v4(path))
    trainer = V4Trainer(examples, run_dir, config, vocabulary=vocabulary)
    trainer.fit()
    return trainer


def _split_games(
    dataset: DatasetInputV4,
    validation_basis_points: int,
    split_seed: int,
) -> tuple[list[Path], list[Path], list[str], list[str]]:
    if not 1 <= validation_basis_points <= 9000:
        raise ValueError("validation_basis_points must be in 1..=9000")
    train: list[Path] = []
    validation: list[Path] = []
    train_ids: list[str] = []
    validation_ids: list[str] = []
    scored: list[tuple[int, Any]] = []
    for game in dataset.games:
        digest = hashlib.sha256(f"{split_seed}:{game.game_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 10_000
        scored.append((bucket, game))
        target_paths = (
            (validation, validation_ids)
            if bucket < validation_basis_points
            else (train, train_ids)
        )
        target_paths[0].append(game.path)
        target_paths[1].append(game.game_id)
    if not validation:
        _bucket, game = min(scored, key=lambda item: item[0])
        train.remove(game.path)
        train_ids.remove(game.game_id)
        validation.append(game.path)
        validation_ids.append(game.game_id)
    if not train:
        _bucket, game = max(scored, key=lambda item: item[0])
        validation.remove(game.path)
        validation_ids.remove(game.game_id)
        train.append(game.path)
        train_ids.append(game.game_id)
    return train, validation, train_ids, validation_ids


def _load_examples(
    paths: Sequence[Path], max_examples: int | None = None
) -> list[FactorizedTrainingExampleV4]:
    result: list[FactorizedTrainingExampleV4] = []
    for path in paths:
        remaining = None if max_examples is None else max_examples - len(result)
        if remaining is not None and remaining <= 0:
            break
        examples = load_game_v4(path)
        result.extend(examples if remaining is None else examples[:remaining])
    return result


def _dataset_provenance(
    dataset: DatasetInputV4,
    vocabulary_identity: Mapping[str, Any],
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    *,
    max_train_examples: int | None,
    max_validation_examples: int | None,
) -> dict[str, Any]:
    return {
        "source_seal_tag": dataset.source_seal_tag,
        "source_seal_commit": dataset.source_seal_commit,
        "source_engine_commit": dataset.engine_commit,
        "ruleset_id": dataset.ruleset["id"],
        "ruleset_manifest_sha256": dataset.ruleset["manifest_sha256"],
        "factorized_policy_schema_id": dataset.factorized_policy_schema_id,
        "dataset_input_schema_id": dataset.schema_id,
        "dataset_manifest_path": str(dataset.path),
        "dataset_manifest_sha256": dataset.sha256,
        "dataset_run_id": dataset.dataset_id,
        "dataset_run_kind": dataset.run_kind,
        "dataset_source_manifests": [dict(item) for item in dataset.source_manifests],
        "producer_evaluator": dict(dataset.producer_evaluator),
        "train_game_ids": list(train_ids),
        "validation_game_ids": list(validation_ids),
        "runtime_vocabulary": dict(vocabulary_identity),
        "max_train_examples": max_train_examples,
        "max_validation_examples": max_validation_examples,
    }


def _environment_identity(device: torch.device) -> dict[str, Any]:
    try:
        import onnxruntime as ort

        ort_version = ort.__version__
    except ImportError:
        ort_version = None
    cuda_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(device)
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "onnxruntime": ort_version,
        "device": str(device),
        "cuda_device_name": cuda_name,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": (
            [state.tolist() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    python_state = payload.get("python")
    if python_state is not None:
        random.setstate(_nested_tuple(python_state))
    torch_cpu = payload.get("torch_cpu")
    if isinstance(torch_cpu, list):
        torch.set_rng_state(torch.tensor(torch_cpu, dtype=torch.uint8))
    torch_cuda = payload.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(torch_cuda, list) and torch_cuda:
        states = [torch.tensor(state, dtype=torch.uint8) for state in torch_cuda]
        if len(states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(states)


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_identity(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AugmentNet V4")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026091401)
    parser.add_argument("--encoder", choices=("baseline", "hybrid"), default="baseline")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=6)
    parser.add_argument("--runtime-message-passes", type=int, default=2)
    parser.add_argument("--hybrid-layers", type=int, default=2)
    parser.add_argument("--hybrid-heads", type=int, default=4)
    parser.add_argument(
        "--batching",
        dest="batching_kind",
        choices=("random", "bucketed"),
        default="random",
    )
    parser.add_argument("--validation-bp", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--runtime-vocabulary",
        type=Path,
        default=default_runtime_vocabulary_path(),
    )
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()

    for name in (
        "batch_size",
        "epochs",
        "channels",
        "model_dim",
        "residual_blocks",
        "runtime_message_passes",
        "hybrid_layers",
        "hybrid_heads",
        "checkpoint_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_train_examples", "max_validation_examples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.value_weight) or args.value_weight < 0:
        parser.error("--value-weight must be finite and non-negative")

    dataset_path = args.manifest.resolve()
    run_dir = args.run_dir.resolve()
    repo_root = Path(__file__).parents[2]
    commit, dirty = _git_identity(repo_root)
    if dirty and not args.allow_dirty_smoke:
        parser.error(
            "refusing training from a dirty worktree; use --allow-dirty-smoke only for non-promotable smoke runs"
        )
    if dirty and "smoke" not in run_dir.name.lower():
        parser.error("--allow-dirty-smoke requires run_dir name to contain 'smoke'")

    dataset = load_dataset_input(dataset_path)
    if (
        dataset.source_seal_tag != SOURCE_SEAL_TAG
        or dataset.source_seal_commit != SOURCE_SEAL_COMMIT
    ):
        parser.error("dataset source seal does not match the V4 training base")
    vocabulary, vocabulary_identity = load_runtime_vocabulary(args.runtime_vocabulary)
    train_paths, validation_paths, train_ids, validation_ids = _split_games(
        dataset,
        args.validation_bp,
        args.seed,
    )
    train_examples = _load_examples(train_paths, args.max_train_examples)
    validation_examples = _load_examples(
        validation_paths, args.max_validation_examples
    )
    provenance = _dataset_provenance(
        dataset,
        vocabulary_identity,
        train_ids,
        validation_ids,
        max_train_examples=args.max_train_examples,
        max_validation_examples=args.max_validation_examples,
    )
    provenance["trainer_git_commit"] = commit
    provenance["trainer_dirty_worktree"] = dirty

    config = V4TrainingConfig(
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
        encoder_kind=args.encoder,
        hybrid_layers=args.hybrid_layers,
        hybrid_heads=args.hybrid_heads,
        batching_kind=args.batching_kind,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    limited_dataset = (
        args.max_train_examples is not None or args.max_validation_examples is not None
    )
    run_identity = {
        "schema_id": V4_TRAINING_RUN_SCHEMA_ID,
        "run_kind": (
            "smoke"
            if dirty or dataset.run_kind == "smoke" or limited_dataset
            else "production"
        ),
        "config": asdict(config),
        "provenance": provenance,
        "environment": _environment_identity(torch.device(args.device)),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
    }
    _atomic_json(run_dir / "run.json", run_identity)

    if args.resume is None:
        trainer = V4Trainer(
            train_examples,
            run_dir,
            config,
            vocabulary=vocabulary,
            validation_examples=validation_examples,
            provenance=provenance,
        )
    else:
        trainer = V4Trainer.from_checkpoint(
            args.resume,
            train_examples,
            config,
            validation_examples=validation_examples,
            expected_vocabulary=vocabulary,
            expected_provenance=provenance,
        )
        trainer.run_dir = run_dir

    interrupt_count = 0

    def request_pause(signum: int, _frame: Any) -> None:
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count == 1:
            trainer.stop_requested = True
            print(
                json.dumps(
                    {
                        "event": "v4_training_pause_requested",
                        "signal": signum,
                        "message": "finishing the current batch before checkpointing",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        raise KeyboardInterrupt

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = (
        signal.getsignal(signal.SIGTERM) if hasattr(signal, "SIGTERM") else None
    )
    signal.signal(signal.SIGINT, request_pause)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_pause)
    try:
        trainer.fit(checkpoint_every=args.checkpoint_every)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if hasattr(signal, "SIGTERM") and previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)

    if trainer.paused:
        metrics = {
            "schema_id": V4_TRAINING_RUN_SCHEMA_ID,
            "paused": True,
            "completed_epochs": trainer.completed_epochs,
            "history": trainer.history,
            "resume_checkpoint": str(trainer.pause_checkpoint),
            "resume_checkpoint_sha256": (
                None
                if trainer.pause_checkpoint is None
                else _sha256_file(trainer.pause_checkpoint)
            ),
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "train_games": len(train_ids),
            "validation_games": len(validation_ids),
        }
        _atomic_json(run_dir / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    final_checkpoint, final_sidecar = trainer.save_checkpoint(run_dir / "final.pt")
    metrics = {
        "schema_id": V4_TRAINING_RUN_SCHEMA_ID,
        "paused": False,
        "completed_epochs": trainer.completed_epochs,
        "history": trainer.history,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": _sha256_file(final_checkpoint),
        "final_sidecar": str(final_sidecar),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_games": len(train_ids),
        "validation_games": len(validation_ids),
    }
    _atomic_json(run_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

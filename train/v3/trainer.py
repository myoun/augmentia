"""Deterministic V3 training with atomic, exact-step checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from .model import AugmentNetV3, V3ModelConfig, compute_v3_loss
from .mixture import load_training_dataset
from .replay import (
    DeterministicSamplerV3,
    ReplayDatasetV3,
    TrainingBatchV3,
    collate_examples,
)
from .tensorize import V3Vocabulary

CHECKPOINT_SCHEMA_ID = "augmentia.training-checkpoint.v3"
METRICS_SCHEMA_ID = "augmentia.training-metrics.v3"
MAX_AMP_OVERFLOW_RETRIES = 8


@dataclass(frozen=True)
class TrainingConfigV3:
    seed: int = 20260821
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    total_steps: int = 100_000
    checkpoint_interval: int = 500
    checkpoint_keep: int = 4
    value_weight: float = 1.0
    amp: bool = True
    device: str = "cuda"
    prefetch_batches: int = 1
    max_padded_actions: int | None = None

    def __post_init__(self) -> None:
        positive = (
            self.batch_size,
            self.gradient_accumulation_steps,
            self.learning_rate,
            self.total_steps,
            self.checkpoint_interval,
            self.checkpoint_keep,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("training sizes, rates and intervals must be positive")
        if self.prefetch_batches not in {0, 1}:
            raise ValueError("prefetch_batches must be zero or one")
        if self.max_padded_actions is not None and self.max_padded_actions <= 0:
            raise ValueError("max_padded_actions must be positive when configured")


@dataclass
class StepMetricsV3:
    step: int
    samples: int
    loss: float
    policy_loss: float
    value_loss: float
    auxiliary_loss: float
    decision_ids: tuple[str, ...]
    amp_overflow_retries: int = 0


@dataclass(frozen=True)
class PolicyValidationMetricsV3:
    step: int
    examples: int
    cross_entropy: float
    perplexity: float
    top1_accuracy: float
    top3_accuracy: float
    top1_correct: int
    top3_correct: int
    elapsed_seconds: float
    examples_per_second: float


@dataclass
class EarlyStoppingV3:
    patience: int
    min_delta: float = 0.0
    best_cross_entropy: float = math.inf
    checks_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("early-stopping patience must be non-negative")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("early-stopping min-delta must be finite and non-negative")

    def observe(self, cross_entropy: float) -> tuple[bool, bool]:
        if not math.isfinite(cross_entropy):
            raise ValueError("validation cross-entropy must be finite")
        improved = cross_entropy < self.best_cross_entropy - self.min_delta
        if improved:
            self.best_cross_entropy = cross_entropy
            self.checks_without_improvement = 0
        else:
            self.checks_without_improvement += 1
        should_stop = (
            self.patience > 0 and self.checks_without_improvement >= self.patience
        )
        return improved, should_stop


class TrainingMetricsJournalV3:
    """Atomically persisted, resume-aware train and validation history."""

    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        run_identity: Mapping[str, Any],
    ) -> None:
        self.path = Path(run_dir).resolve() / "metrics.json"
        self.run_id = run_id
        self.run_identity_sha256 = _json_sha256(run_identity)
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_bytes())
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"invalid training metrics journal: {error}"
                ) from error
            if (
                not isinstance(payload, dict)
                or payload.get("schema_id") != METRICS_SCHEMA_ID
                or payload.get("run_id") != run_id
                or payload.get("run_identity_sha256") != self.run_identity_sha256
                or not isinstance(payload.get("events"), list)
            ):
                raise CheckpointMismatchError(
                    "training metrics journal does not match the resumed run"
                )
            self._payload = payload
        else:
            self._payload = {
                "schema_id": METRICS_SCHEMA_ID,
                "run_id": run_id,
                "run_identity_sha256": self.run_identity_sha256,
                "events": [],
            }
            self._write()

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._payload["events"])

    @property
    def validation_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(event)
            for event in self._payload["events"]
            if event.get("kind") == "validation"
        )

    def event_at(self, kind: str, step: int) -> dict[str, Any] | None:
        for event in self._payload["events"]:
            if event.get("kind") == kind and event.get("step") == step:
                return dict(event)
        return None

    def append(self, event: Mapping[str, Any]) -> None:
        normalized = dict(event)
        if normalized.get("kind") not in {"train", "validation"}:
            raise ValueError("metric event kind must be train or validation")
        step = normalized.get("step")
        if not isinstance(step, int) or step < 0:
            raise ValueError("metric event step must be a non-negative integer")
        if self.event_at(str(normalized["kind"]), step) is not None:
            raise RuntimeError(
                f"duplicate {normalized['kind']} metric event at step {step}"
            )
        self._payload["events"].append(normalized)
        self._write()

    def truncate_after(self, step: int) -> None:
        retained = [
            event
            for event in self._payload["events"]
            if isinstance(event.get("step"), int) and event["step"] <= step
        ]
        if len(retained) != len(self._payload["events"]):
            self._payload["events"] = retained
            self._write()

    def _write(self) -> None:
        _atomic_json_write(self.path, self._payload)


@dataclass
class _PrefetchedBatchV3:
    base_sampler_state: dict[str, Any]
    committed_sampler_state: dict[str, Any]
    future: Future[tuple[TrainingBatchV3, ...]]


class CheckpointMismatchError(RuntimeError):
    pass


class CheckpointManagerV3:
    def __init__(self, run_dir: str | Path, keep: int) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    def save(self, payload: Mapping[str, Any], step: int) -> Path:
        target = self.checkpoint_dir / f"step-{step:012d}.pt"
        temporary = target.with_suffix(f".pt.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        checksum_path = target.with_suffix(".pt.sha256")
        checksum_temporary = checksum_path.with_suffix(
            f".sha256.tmp-{uuid.uuid4().hex}"
        )
        try:
            torch.save(dict(payload), temporary)
            _fsync_file(temporary)
            digest = _sha256_file(temporary)
            checksum_temporary.write_text(digest + "\n", encoding="ascii")
            _fsync_file(checksum_temporary)
            os.replace(temporary, target)
            os.replace(checksum_temporary, checksum_path)
            _fsync_directory(self.checkpoint_dir)
            self._write_latest(target.name, digest)
            self._prune()
            return target
        finally:
            temporary.unlink(missing_ok=True)
            checksum_temporary.unlink(missing_ok=True)

    def load_latest_valid(self) -> tuple[Path, dict[str, Any]] | None:
        for path in sorted(self.checkpoint_dir.glob("step-*.pt"), reverse=True):
            checksum_path = path.with_suffix(".pt.sha256")
            try:
                expected = checksum_path.read_text(encoding="ascii").strip()
                if len(expected) != 64 or _sha256_file(path) != expected:
                    continue
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("checkpoint_schema_id") != CHECKPOINT_SCHEMA_ID:
                    continue
                return path, payload
            except (OSError, RuntimeError, ValueError, EOFError):
                continue
        return None

    def save_best(
        self,
        payload: Mapping[str, Any],
        validation: Mapping[str, Any],
    ) -> Path:
        """Publish a validation-best checkpoint outside rotating retention."""
        target = self.checkpoint_dir / "best-validation.pt"
        temporary = target.with_suffix(f".pt.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        checksum_path = target.with_suffix(".pt.sha256")
        checksum_temporary = checksum_path.with_suffix(
            f".sha256.tmp-{uuid.uuid4().hex}"
        )
        metadata_path = self.checkpoint_dir / "best-validation.json"
        try:
            torch.save(dict(payload), temporary)
            _fsync_file(temporary)
            digest = _sha256_file(temporary)
            checksum_temporary.write_text(digest + "\n", encoding="ascii")
            _fsync_file(checksum_temporary)
            os.replace(temporary, target)
            os.replace(checksum_temporary, checksum_path)
            _atomic_json_write(
                metadata_path,
                {
                    "checkpoint": target.name,
                    "checkpoint_sha256": digest,
                    "global_step": int(payload["global_step"]),
                    "primary_metric": "cross_entropy",
                    "validation": dict(validation),
                },
            )
            _fsync_directory(self.checkpoint_dir)
            return target
        finally:
            temporary.unlink(missing_ok=True)
            checksum_temporary.unlink(missing_ok=True)

    def _write_latest(self, filename: str, digest: str) -> None:
        target = self.checkpoint_dir / "latest.json"
        temporary = target.with_suffix(f".json.tmp-{uuid.uuid4().hex}")
        temporary.write_text(
            json.dumps({"checkpoint": filename, "sha256": digest}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        _fsync_file(temporary)
        os.replace(temporary, target)
        _fsync_directory(self.checkpoint_dir)

    def _prune(self) -> None:
        valid = sorted(self.checkpoint_dir.glob("step-*.pt"), reverse=True)
        for path in valid[self.keep :]:
            path.unlink(missing_ok=True)
            path.with_suffix(".pt.sha256").unlink(missing_ok=True)


class TrainerV3:
    def __init__(
        self,
        dataset: ReplayDatasetV3,
        vocabulary: V3Vocabulary,
        model_config: V3ModelConfig,
        training_config: TrainingConfigV3,
        run_dir: str | Path,
        run_id: str | None = None,
        parent_checkpoint: str | None = None,
        parent_metadata: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.vocabulary = vocabulary
        self.model_config = model_config
        self.config = training_config
        self.device = _resolve_device(training_config.device)
        _seed_everything(training_config.seed)
        if (parent_checkpoint is None) != (parent_metadata is None):
            raise ValueError(
                "parent_checkpoint and parent_metadata must be provided together"
            )
        self.model = AugmentNetV3(model_config)
        self.parent_checkpoint = None
        self.parent_metadata = None
        self.parent_identity: dict[str, Any] | None = None
        self.parent_migrations: tuple[dict[str, Any], ...] = ()
        if parent_checkpoint is not None and parent_metadata is not None:
            (
                parent_state,
                self.parent_identity,
                self.parent_migrations,
            ) = _load_parent_checkpoint(
                Path(parent_checkpoint),
                Path(parent_metadata),
                self.model,
                vocabulary,
            )
            self.model.load_state_dict(parent_state)
            self.parent_checkpoint = str(Path(parent_checkpoint).resolve())
            self.parent_metadata = str(Path(parent_metadata).resolve())
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=training_config.total_steps
        )
        amp_enabled = training_config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.sampler = DeterministicSamplerV3(
            len(dataset), training_config.seed, dataset.sampler_group_sizes
        )
        self.checkpoints = CheckpointManagerV3(run_dir, training_config.checkpoint_keep)
        self.run_id = run_id or uuid.uuid4().hex
        self.global_step = 0
        self.samples_seen = 0
        self.last_metrics: dict[str, float] = {}
        self.stop_requested = False
        self.code_commit = _git_commit()
        self._identity = self._run_identity()
        self._prefetch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="v3-batch-prefetch")
            if training_config.prefetch_batches
            else None
        )
        self._prefetched_batch: _PrefetchedBatchV3 | None = None

    @property
    def run_identity(self) -> dict[str, Any]:
        return dict(self._identity)

    def train_step(self) -> StepMetricsV3:
        if self.global_step >= self.config.total_steps:
            raise StopIteration("configured optimizer step limit reached")
        self.model.train()
        cached_batch_groups: list[tuple[TrainingBatchV3, ...]] = []
        overflow_retries = 0
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_ids: list[str] = []
            totals = torch.zeros(4, dtype=torch.float64)
            logical_batches = 0
            samples = 0
            for accumulation_index in range(self.config.gradient_accumulation_steps):
                if overflow_retries == 0:
                    source_batches = self._next_training_batches()
                    cached_batch_groups.append(source_batches)
                else:
                    source_batches = cached_batch_groups[accumulation_index]
                supervision_totals = _supervision_totals(source_batches)
                logical_losses = torch.zeros(4, dtype=torch.float64)
                for source_batch in source_batches:
                    batch = _move_training_batch(
                        source_batch,
                        self.device,
                        non_blocking=self._prefetch_executor is not None,
                    )
                    accumulated_ids.extend(batch.decision_ids)
                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=self.scaler.is_enabled(),
                        dtype=torch.float16,
                    ):
                        output = self.model(batch.inputs)
                        loss = compute_v3_loss(
                            output,
                            batch.inputs.action_mask,
                            batch.policy_targets,
                            batch.policy_supervision,
                            batch.value_targets,
                            batch.value_supervision,
                            self.config.value_weight,
                            batch.auxiliary_targets,
                            batch.auxiliary_supervision,
                        )
                        policy_weight = _supervision_weight(
                            loss.policy_samples, supervision_totals[0]
                        )
                        value_weight = _supervision_weight(
                            loss.value_samples, supervision_totals[1]
                        )
                        auxiliary_weight = _supervision_weight(
                            loss.auxiliary_samples, supervision_totals[2]
                        )
                        combined_loss = (
                            loss.policy * policy_weight
                            + self.config.value_weight * loss.value * value_weight
                            + 0.25 * loss.auxiliary * auxiliary_weight
                        )
                        scaled_loss = (
                            combined_loss / self.config.gradient_accumulation_steps
                        )
                    if not torch.isfinite(combined_loss):
                        raise FloatingPointError("training loss became NaN or infinite")
                    self.scaler.scale(scaled_loss).backward()
                    logical_losses += torch.tensor(
                        [
                            combined_loss.item(),
                            loss.policy.item() * policy_weight,
                            loss.value.item() * value_weight,
                            loss.auxiliary.item() * auxiliary_weight,
                        ],
                        dtype=torch.float64,
                    )
                    samples += len(batch.decision_ids)
                totals += logical_losses
                logical_batches += 1
            self.scaler.unscale_(self.optimizer)
            invalid_gradient = next(
                (
                    name
                    for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ),
                None,
            )
            if invalid_gradient is None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                break
            if (
                not self.scaler.is_enabled()
                or overflow_retries >= MAX_AMP_OVERFLOW_RETRIES
            ):
                raise FloatingPointError(
                    f"gradient {invalid_gradient} became NaN or infinite"
                )
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= previous_scale:
                raise FloatingPointError(
                    f"gradient {invalid_gradient} became NaN or infinite without AMP backoff"
                )
            overflow_retries += 1
        self.scheduler.step()
        self.global_step += 1
        self.samples_seen += samples
        averages = totals / logical_batches
        result = StepMetricsV3(
            self.global_step,
            samples,
            float(averages[0]),
            float(averages[1]),
            float(averages[2]),
            float(averages[3]),
            tuple(accumulated_ids),
            overflow_retries,
        )
        self.last_metrics = {
            "loss": result.loss,
            "policy_loss": result.policy_loss,
            "value_loss": result.value_loss,
            "auxiliary_loss": result.auxiliary_loss,
        }
        return result

    def iter_train(self, steps: int | None = None) -> Iterator[StepMetricsV3]:
        limit = (
            self.config.total_steps
            if steps is None
            else min(self.config.total_steps, self.global_step + steps)
        )
        while self.global_step < limit and not self.stop_requested:
            metrics = self.train_step()
            if self.global_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
            yield metrics
        if (self.stop_requested or self.global_step == limit) and (
            self.global_step % self.config.checkpoint_interval != 0
        ):
            self.save_checkpoint()

    def train(self, steps: int | None = None) -> list[StepMetricsV3]:
        return list(self.iter_train(steps))

    def save_checkpoint(self) -> Path:
        return self.checkpoints.save(self._checkpoint_payload(), self.global_step)

    def save_best_checkpoint(self, validation: Mapping[str, Any]) -> Path:
        return self.checkpoints.save_best(self._checkpoint_payload(), validation)

    def resume_latest(self) -> Path:
        loaded = self.checkpoints.load_latest_valid()
        if loaded is None:
            raise FileNotFoundError("no complete V3 checkpoint was found")
        path, payload = loaded
        if payload.get("run_identity") != self._identity:
            raise CheckpointMismatchError(
                "checkpoint dataset/schema/model/training identity differs; fork a new run intentionally"
            )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["amp_scaler"])
        self.sampler.load_state_dict(payload["sampler"])
        if payload.get("gradient_accumulation_boundary") != 0:
            raise CheckpointMismatchError(
                "only completed optimizer-step checkpoints are resumable"
            )
        self.global_step = int(payload["global_step"])
        self.samples_seen = int(payload["samples_seen"])
        self.last_metrics = dict(payload["last_metrics"])
        self.run_id = str(payload["run_id"])
        _restore_rng(payload["rng"])
        return path

    def _next_index_batches(
        self, sampler: DeterministicSamplerV3
    ) -> list[list[int]]:
        return _take_action_budgeted_sampler_indices(
            sampler,
            self.config.batch_size,
            self.dataset,
            self.config.max_padded_actions,
        )

    def _next_training_batches(self) -> tuple[TrainingBatchV3, ...]:
        prefetched = self._prefetched_batch
        if prefetched is None:
            index_batches = self._next_index_batches(self.sampler)
            batches = _collate_dataset_index_batches(
                self.dataset,
                index_batches,
                self.vocabulary,
                pin_memory=self._prefetch_executor is not None
                and self.device.type == "cuda",
            )
        else:
            if self.sampler.state_dict() != prefetched.base_sampler_state:
                raise RuntimeError("prefetched batch sampler state became stale")
            batches = prefetched.future.result()
            self.sampler.load_state_dict(prefetched.committed_sampler_state)
            self._prefetched_batch = None
        self._schedule_prefetch()
        return batches

    def _schedule_prefetch(self) -> None:
        if self._prefetch_executor is None or self._prefetched_batch is not None:
            return
        base_state = self.sampler.state_dict()
        preview = DeterministicSamplerV3(
            len(self.dataset),
            self.config.seed,
            self.dataset.sampler_group_sizes,
        )
        preview.load_state_dict(base_state)
        index_batches = self._next_index_batches(preview)
        self._prefetched_batch = _PrefetchedBatchV3(
            base_sampler_state=base_state,
            committed_sampler_state=preview.state_dict(),
            future=self._prefetch_executor.submit(
                _collate_dataset_index_batches,
                self.dataset,
                index_batches,
                self.vocabulary,
                self.device.type == "cuda",
            ),
        )

    def close(self) -> None:
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
            self._prefetch_executor = None
            self._prefetched_batch = None

    def __del__(self) -> None:
        executor = getattr(self, "_prefetch_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _run_identity(self) -> dict[str, Any]:
        critical_config = asdict(self.config)
        critical_config.pop("checkpoint_interval")
        critical_config.pop("checkpoint_keep")
        critical_config.pop("device")
        critical_config.pop("prefetch_batches")
        return {
            "dataset": self.dataset.identity.to_json(),
            "model_config": asdict(self.model_config),
            "training_config": critical_config,
            "vocabulary": self.vocabulary.to_json(),
            "parent": self.parent_identity,
        }

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "checkpoint_schema_id": CHECKPOINT_SCHEMA_ID,
            "run_id": self.run_id,
            "parent_checkpoint": self.parent_checkpoint,
            "parent_metadata": self.parent_metadata,
            "parent_migrations": list(self.parent_migrations),
            "code_commit": self.code_commit,
            "run_identity": self._identity,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "amp_scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.sampler.epoch,
            "gradient_accumulation_boundary": 0,
            "sampler": self.sampler.state_dict(),
            "rng": _capture_rng(),
            "shuffle_seed": self.config.seed,
            "samples_seen": self.samples_seen,
            "last_metrics": self.last_metrics,
        }


def evaluate_policy_v3(
    model: AugmentNetV3,
    dataset: ReplayDatasetV3,
    vocabulary: V3Vocabulary,
    device: torch.device,
    *,
    batch_size: int,
    step: int,
    max_padded_actions: int | None = None,
) -> PolicyValidationMetricsV3:
    """Evaluate policy calibration and ranking on a stable sequential split."""
    if batch_size <= 0:
        raise ValueError("validation batch size must be positive")
    started = time.perf_counter()
    previous_training = model.training
    model.eval()
    cross_entropy_sum = 0.0
    supervised_examples = 0
    top1_correct = 0
    top3_correct = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                logical_indices = list(
                    range(start, min(start + batch_size, len(dataset)))
                )
                for indices in _partition_indices_by_action_budget(
                    dataset, logical_indices, max_padded_actions
                ):
                    source = _collate_dataset_indices(
                        dataset, indices, vocabulary, pin_memory=False
                    )
                    batch = _move_training_batch(source, device, non_blocking=False)
                    output = model(batch.inputs)
                    logits = output.policy_logits.float()
                    log_probabilities = torch.log_softmax(logits, dim=1)
                    per_sample = -(batch.policy_targets * log_probabilities).sum(dim=1)
                    supervised = batch.policy_supervision
                    targets = batch.policy_targets.argmax(dim=1)
                    top1 = logits.argmax(dim=1)
                    top3 = logits.topk(min(3, logits.shape[1]), dim=1).indices
                    cross_entropy_sum += (
                        per_sample.masked_select(supervised).double().sum().item()
                    )
                    supervised_examples += int(supervised.sum().item())
                    top1_correct += int(
                        top1.eq(targets).logical_and(supervised).sum().item()
                    )
                    top3_correct += int(
                        top3.eq(targets.unsqueeze(1))
                        .any(dim=1)
                        .logical_and(supervised)
                        .sum()
                        .item()
                    )
    finally:
        model.train(previous_training)
    if supervised_examples == 0:
        raise ValueError("validation split contains no policy-supervised examples")
    elapsed = time.perf_counter() - started
    cross_entropy = cross_entropy_sum / supervised_examples
    return PolicyValidationMetricsV3(
        step=step,
        examples=supervised_examples,
        cross_entropy=cross_entropy,
        perplexity=math.exp(cross_entropy),
        top1_accuracy=top1_correct / supervised_examples,
        top3_accuracy=top3_correct / supervised_examples,
        top1_correct=top1_correct,
        top3_correct=top3_correct,
        elapsed_seconds=elapsed,
        examples_per_second=supervised_examples / elapsed,
    )


def load_validation_dataset(
    path: str | Path, split: str = "validation"
) -> ReplayDatasetV3:
    if split not in {"validation", "holdout"}:
        raise ValueError("evaluation split must be validation or holdout")
    candidate = Path(path).resolve()
    try:
        payload = json.loads(candidate.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid validation dataset JSON: {error}") from error
    if (
        isinstance(payload, dict)
        and payload.get("artifact_schema_id") == "augmentia.pretensor-manifest.v3.1"
    ):
        if payload.get("split") != split:
            raise ValueError(
                f"validation artifact split is {payload.get('split')!r}, expected {split!r}"
            )
        from .pretensor import PretensorizedReplayDatasetV3

        return PretensorizedReplayDatasetV3(candidate)
    if isinstance(payload, dict) and payload.get("concat_schema_id"):
        dataset = load_training_dataset(candidate)
        if not _dataset_uses_split(dataset, split):
            raise ValueError(f"concat validation sources must all use split {split!r}")
        return dataset
    return ReplayDatasetV3(candidate, split=split)


def _dataset_uses_split(dataset: Any, split: str) -> bool:
    dataset_split = getattr(dataset, "split", None)
    if dataset_split is not None:
        return dataset_split == split
    sources = getattr(dataset, "datasets", None)
    return bool(sources) and all(_dataset_uses_split(source, split) for source in sources)


def fine_tuning_vocabulary(
    dataset: ReplayDatasetV3, parent_metadata: str | Path | None
) -> V3Vocabulary:
    """Keep parent token indices stable and append only newly observed tokens."""
    dataset_vocabulary = dataset.build_vocabulary()
    if parent_metadata is None:
        return dataset_vocabulary
    metadata = json.loads(Path(parent_metadata).read_text(encoding="utf-8"))
    parent_vocabulary = V3Vocabulary.from_json(metadata["vocabulary"])
    parent_tokens = set(parent_vocabulary.tokens)
    appended = sorted(
        token for token in dataset_vocabulary.tokens if token not in parent_tokens
    )
    return V3Vocabulary((*parent_vocabulary.tokens, *appended))


def assert_validation_compatible(
    training_dataset: ReplayDatasetV3,
    validation_dataset: ReplayDatasetV3,
    vocabulary: V3Vocabulary,
) -> None:
    validation_vocabulary = validation_dataset.build_vocabulary()
    if not set(validation_vocabulary.tokens).issubset(vocabulary.tokens):
        raise CheckpointMismatchError(
            "validation vocabulary contains tokens absent from training"
        )
    training_identity = _base_dataset_identity(training_dataset.identity.to_json())
    validation_identity = _base_dataset_identity(validation_dataset.identity.to_json())
    keys = (
        "ruleset_id",
        "ruleset_manifest_sha256",
        "observation_schema_id",
        "action_schema_id",
    )
    if any(training_identity.get(key) != validation_identity.get(key) for key in keys):
        raise CheckpointMismatchError(
            "validation ruleset or schema differs from the training dataset"
        )


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def _take_sampler_indices(sampler: DeterministicSamplerV3, count: int) -> list[int]:
    indices: list[int] = []
    while len(indices) < count:
        indices.extend(sampler.take(count - len(indices)))
        sampler.advance_epoch_if_complete()
    return indices


def _take_action_budgeted_sampler_indices(
    sampler: DeterministicSamplerV3,
    count: int,
    dataset: Any,
    max_padded_actions: int | None,
) -> list[list[int]]:
    if max_padded_actions is None:
        return [_take_sampler_indices(sampler, count)]
    batches: list[list[int]] = []
    remaining = count
    while remaining > 0:
        indices: list[int] = []
        maximum_actions = 0
        for index in sampler.remaining_indices():
            action_count = _dataset_action_count(dataset, index)
            projected = (len(indices) + 1) * max(maximum_actions, action_count)
            if indices and projected > max_padded_actions:
                break
            indices.append(index)
            maximum_actions = max(maximum_actions, action_count)
            if len(indices) == remaining:
                break
        if not indices:
            raise RuntimeError("action-budgeted sampler made no progress")
        sampler.cursor += len(indices)
        sampler.advance_epoch_if_complete()
        batches.append(indices)
        remaining -= len(indices)
    return batches


def _dataset_action_count(dataset: Any, index: int) -> int:
    if hasattr(dataset, "action_count"):
        count = int(dataset.action_count(index))
    else:
        example = dataset[index]
        count = len(example.legal_actions["actions"])
    if count <= 0:
        raise ValueError(f"dataset example {index} has no legal actions")
    return count


def _partition_indices_by_action_budget(
    dataset: Any,
    indices: list[int],
    max_padded_actions: int | None,
) -> list[list[int]]:
    if max_padded_actions is None:
        return [indices]
    batches: list[list[int]] = []
    pending: list[int] = []
    maximum_actions = 0
    for index in indices:
        action_count = _dataset_action_count(dataset, index)
        projected = (len(pending) + 1) * max(maximum_actions, action_count)
        if pending and projected > max_padded_actions:
            batches.append(pending)
            pending = []
            maximum_actions = 0
        pending.append(index)
        maximum_actions = max(maximum_actions, action_count)
    if pending:
        batches.append(pending)
    return batches


def _collate_dataset_index_batches(
    dataset: ReplayDatasetV3,
    index_batches: list[list[int]],
    vocabulary: V3Vocabulary,
    pin_memory: bool,
) -> tuple[TrainingBatchV3, ...]:
    return tuple(
        _collate_dataset_indices(dataset, indices, vocabulary, pin_memory)
        for indices in index_batches
    )


def _supervision_totals(
    batches: tuple[TrainingBatchV3, ...],
) -> tuple[int, int, int]:
    return (
        sum(int(batch.policy_supervision.sum().item()) for batch in batches),
        sum(int(batch.value_supervision.sum().item()) for batch in batches),
        sum(int(batch.auxiliary_supervision.sum().item()) for batch in batches),
    )


def _supervision_weight(samples: int, total: int) -> float:
    return 0.0 if total == 0 else samples / total


def _collate_dataset_indices(
    dataset: ReplayDatasetV3,
    indices: list[int],
    vocabulary: V3Vocabulary,
    pin_memory: bool,
) -> TrainingBatchV3:
    if hasattr(dataset, "collate_indices"):
        batch = dataset.collate_indices(indices)
    else:
        examples = [dataset[index] for index in indices]
        batch = collate_examples(examples, vocabulary)
    return _pin_training_batch(batch) if pin_memory else batch


def _pin_training_batch(batch: TrainingBatchV3) -> TrainingBatchV3:
    inputs = type(batch.inputs)(
        **{
            field.name: getattr(batch.inputs, field.name).pin_memory()
            for field in fields(batch.inputs)
        }
    )
    return TrainingBatchV3(
        inputs=inputs,
        policy_targets=batch.policy_targets.pin_memory(),
        policy_supervision=batch.policy_supervision.pin_memory(),
        value_targets=batch.value_targets.pin_memory(),
        value_supervision=batch.value_supervision.pin_memory(),
        auxiliary_targets=batch.auxiliary_targets.pin_memory(),
        auxiliary_supervision=batch.auxiliary_supervision.pin_memory(),
        decision_ids=batch.decision_ids,
    )


def _move_training_batch(
    batch: TrainingBatchV3, device: torch.device, *, non_blocking: bool
) -> TrainingBatchV3:
    if not non_blocking:
        return batch.to(device)
    inputs = type(batch.inputs)(
        **{
            field.name: getattr(batch.inputs, field.name).to(device, non_blocking=True)
            for field in fields(batch.inputs)
        }
    )
    return TrainingBatchV3(
        inputs=inputs,
        policy_targets=batch.policy_targets.to(device, non_blocking=True),
        policy_supervision=batch.policy_supervision.to(device, non_blocking=True),
        value_targets=batch.value_targets.to(device, non_blocking=True),
        value_supervision=batch.value_supervision.to(device, non_blocking=True),
        auxiliary_targets=batch.auxiliary_targets.to(device, non_blocking=True),
        auxiliary_supervision=batch.auxiliary_supervision.to(device, non_blocking=True),
        decision_ids=batch.decision_ids,
    )


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; pass --device cpu explicitly"
        )
    return torch.device(requested)


def _base_dataset_identity(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    source = payload.get("source")
    return source if isinstance(source, Mapping) else payload


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_parent_checkpoint(
    checkpoint_path: Path,
    metadata_path: Path,
    target_model: AugmentNetV3,
    target_vocabulary: V3Vocabulary,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], tuple[dict[str, Any], ...]]:
    """Load a verified parent while preserving every learned embedding row."""
    checkpoint_path = checkpoint_path.resolve()
    metadata_path = metadata_path.resolve()
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    metadata_sha256 = _sha256_file(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise CheckpointMismatchError(
            "parent checkpoint SHA-256 does not match its model metadata"
        )
    checksum_sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    if checksum_sidecar.exists():
        sidecar_sha256 = checksum_sidecar.read_text(encoding="utf-8").strip().split()[0]
        if sidecar_sha256 != checkpoint_sha256:
            raise CheckpointMismatchError(
                "parent checkpoint SHA-256 does not match its checksum sidecar"
            )

    parent_vocabulary = V3Vocabulary.from_json(metadata["vocabulary"])
    parent_size = parent_vocabulary.size
    if target_vocabulary.tokens[:parent_size] != parent_vocabulary.tokens:
        raise CheckpointMismatchError(
            "fine-tuning vocabulary must preserve every parent token index"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema_id") != CHECKPOINT_SCHEMA_ID:
        raise CheckpointMismatchError("unsupported parent checkpoint schema")
    parent_model_config = checkpoint.get("run_identity", {}).get("model_config")
    if not isinstance(parent_model_config, Mapping):
        raise CheckpointMismatchError("parent checkpoint has no model identity")
    for name in ("channels", "residual_blocks", "model_dim", "runtime_message_passes"):
        if parent_model_config.get(name) != getattr(target_model.config, name):
            raise CheckpointMismatchError(
                f"parent and target model structures differ at {name}"
            )

    source_state = checkpoint.get("model")
    if not isinstance(source_state, Mapping):
        raise CheckpointMismatchError("parent checkpoint has no model state")
    target_state = target_model.state_dict()
    if set(source_state) != set(target_state):
        missing = sorted(set(target_state) - set(source_state))
        extra = sorted(set(source_state) - set(target_state))
        raise CheckpointMismatchError(
            f"parent model state keys differ (missing={missing}, extra={extra})"
        )

    text_rows = int(source_state["text_embedding.weight"].shape[0])
    extension = metadata.get("vocabulary_extension")
    expected_text_rows = (
        int(extension["base_size"])
        if isinstance(extension, Mapping) and "base_size" in extension
        else parent_size
    )
    if text_rows != expected_text_rows:
        raise CheckpointMismatchError(
            "parent text embedding rows do not match metadata vocabulary provenance"
        )

    migrations: list[dict[str, Any]] = []
    migrated: dict[str, torch.Tensor] = {}
    expandable = {
        "text_embedding.weight": "expand_zero_initialized_text_embedding",
        "piece_kind_embedding.weight": "expand_zero_initialized_piece_kind_embedding",
    }
    for name, target in target_state.items():
        source = source_state[name]
        if not isinstance(source, torch.Tensor):
            raise CheckpointMismatchError(f"parent state {name} is not a tensor")
        if source.shape == target.shape:
            migrated[name] = source
            continue
        if (
            name not in expandable
            or source.ndim != target.ndim
            or source.shape[1:] != target.shape[1:]
            or source.shape[0] >= target.shape[0]
        ):
            raise CheckpointMismatchError(
                f"parent tensor shape differs at {name}: {tuple(source.shape)} != {tuple(target.shape)}"
            )
        expanded = source.new_zeros(target.shape)
        expanded[: source.shape[0]].copy_(source)
        migrated[name] = expanded
        migrations.append(
            {
                "kind": expandable[name],
                "source_rows": source.shape[0],
                "target_rows": target.shape[0],
            }
        )

    identity = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha256,
        "global_step": int(checkpoint.get("global_step", 0)),
        "run_id": str(checkpoint.get("run_id", "")),
    }
    return migrated, identity, tuple(migrations)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AugmentNet V3")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-padded-actions", type=int)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--max-steps-this-run", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--prefetch-batches", type=int, choices=(0, 1), default=1)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--parent-metadata", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument(
        "--validation-split",
        choices=("validation", "holdout"),
        default="validation",
    )
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--validation-max-padded-actions", type=int)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_steps_this_run is not None and args.max_steps_this_run <= 0:
        parser.error("--max-steps-this-run must be positive")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")
    if args.validation_interval <= 0:
        parser.error("--validation-interval must be positive")
    if args.validation_batch_size is not None and args.validation_batch_size <= 0:
        parser.error("--validation-batch-size must be positive")
    if args.max_padded_actions is not None and args.max_padded_actions <= 0:
        parser.error("--max-padded-actions must be positive")
    if (
        args.validation_max_padded_actions is not None
        and args.validation_max_padded_actions <= 0
    ):
        parser.error("--validation-max-padded-actions must be positive")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative")
    if (args.parent_checkpoint is None) != (args.parent_metadata is None):
        parser.error(
            "--parent-checkpoint and --parent-metadata must be provided together"
        )
    if (
        not math.isfinite(args.early_stopping_min_delta)
        or args.early_stopping_min_delta < 0
    ):
        parser.error("--early-stopping-min-delta must be finite and non-negative")
    dataset = load_training_dataset(args.manifest)
    vocabulary = fine_tuning_vocabulary(dataset, args.parent_metadata)
    if hasattr(dataset, "set_collation_vocabulary"):
        dataset.set_collation_vocabulary(vocabulary)
    config = TrainingConfigV3(
        batch_size=args.batch_size,
        total_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        prefetch_batches=args.prefetch_batches,
        max_padded_actions=args.max_padded_actions,
    )
    trainer = TrainerV3(
        dataset,
        vocabulary,
        V3ModelConfig(text_vocabulary_size=vocabulary.size),
        config,
        args.run_dir,
        parent_checkpoint=(
            None if args.parent_checkpoint is None else str(args.parent_checkpoint)
        ),
        parent_metadata=(
            None if args.parent_metadata is None else str(args.parent_metadata)
        ),
    )
    if args.resume:
        print(f"resumed {trainer.resume_latest()}")

    journal = TrainingMetricsJournalV3(
        args.run_dir, trainer.run_id, trainer.run_identity
    )
    journal.truncate_after(trainer.global_step)
    validation_dataset = None
    validation_identity_sha256 = None
    validation_batch_size = args.validation_batch_size or args.batch_size
    early_stopping = EarlyStoppingV3(
        args.early_stopping_patience, args.early_stopping_min_delta
    )
    if args.validation_manifest is not None:
        validation_dataset = load_validation_dataset(
            args.validation_manifest, args.validation_split
        )
        assert_validation_compatible(dataset, validation_dataset, vocabulary)
        if hasattr(validation_dataset, "set_collation_vocabulary"):
            validation_dataset.set_collation_vocabulary(vocabulary)
        validation_identity_sha256 = _json_sha256(validation_dataset.identity.to_json())
        for event in journal.validation_events:
            early_stopping.observe(float(event["cross_entropy"]))

    def request_stop(_signum: int, _frame: Any) -> None:
        trainer.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    started = time.perf_counter()
    window_started = started
    window_samples = 0
    window_steps = 0
    window_loss = 0.0
    window_policy_loss = 0.0
    window_value_loss = 0.0
    window_auxiliary_loss = 0.0
    window_amp_overflows = 0
    last_logged_step = trainer.global_step
    last_metrics = None

    def emit_train_metrics(metrics: StepMetricsV3) -> None:
        nonlocal window_started
        nonlocal window_samples
        nonlocal window_steps
        nonlocal window_loss
        nonlocal window_policy_loss
        nonlocal window_value_loss
        nonlocal window_auxiliary_loss
        nonlocal window_amp_overflows
        nonlocal last_logged_step
        now = time.perf_counter()
        event = {
            "kind": "train",
            "step": metrics.step,
            "samples_seen": trainer.samples_seen,
            "window_steps": window_steps,
            "samples": window_samples,
            "decision_count": window_samples,
            "loss": window_loss / window_steps,
            "policy_loss": window_policy_loss / window_steps,
            "value_loss": window_value_loss / window_steps,
            "auxiliary_loss": window_auxiliary_loss / window_steps,
            "last_step_loss": metrics.loss,
            "amp_overflow_retries": window_amp_overflows,
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            "amp_scale": trainer.scaler.get_scale(),
            "examples_per_second": window_samples / (now - window_started),
        }
        journal.append(event)
        print(json.dumps(event, sort_keys=True), flush=True)
        window_started = now
        window_samples = 0
        window_steps = 0
        window_loss = 0.0
        window_policy_loss = 0.0
        window_value_loss = 0.0
        window_auxiliary_loss = 0.0
        window_amp_overflows = 0
        last_logged_step = metrics.step

    def run_validation() -> bool:
        if validation_dataset is None:
            return False
        existing = journal.event_at("validation", trainer.global_step)
        if existing is not None:
            return (
                early_stopping.patience > 0
                and early_stopping.checks_without_improvement >= early_stopping.patience
            )
        metrics = evaluate_policy_v3(
            trainer.model,
            validation_dataset,
            vocabulary,
            trainer.device,
            batch_size=validation_batch_size,
            step=trainer.global_step,
            max_padded_actions=(
                args.validation_max_padded_actions or args.max_padded_actions
            ),
        )
        improved, should_stop = early_stopping.observe(metrics.cross_entropy)
        event = {
            "kind": "validation",
            **asdict(metrics),
            "validation_identity_sha256": validation_identity_sha256,
            "is_best": improved,
            "best_cross_entropy": early_stopping.best_cross_entropy,
            "checks_without_improvement": (early_stopping.checks_without_improvement),
        }
        if improved:
            trainer.save_best_checkpoint(event)
        journal.append(event)
        print(json.dumps(event, sort_keys=True), flush=True)
        return should_stop

    early_stopped = False
    if (
        validation_dataset is not None
        and journal.event_at("validation", trainer.global_step) is None
    ):
        early_stopped = run_validation()
    elif validation_dataset is not None:
        early_stopped = (
            early_stopping.patience > 0
            and early_stopping.checks_without_improvement >= early_stopping.patience
        )

    try:
        if not early_stopped:
            for metrics in trainer.iter_train(steps=args.max_steps_this_run):
                last_metrics = metrics
                window_samples += metrics.samples
                window_steps += 1
                window_loss += metrics.loss
                window_policy_loss += metrics.policy_loss
                window_value_loss += metrics.value_loss
                window_auxiliary_loss += metrics.auxiliary_loss
                window_amp_overflows += metrics.amp_overflow_retries
                if metrics.step == 1 or metrics.step % args.log_interval == 0:
                    emit_train_metrics(metrics)
                if (
                    validation_dataset is not None
                    and metrics.step % args.validation_interval == 0
                    and run_validation()
                ):
                    early_stopped = True
                    if trainer.global_step % trainer.config.checkpoint_interval != 0:
                        trainer.save_checkpoint()
                    break
        if last_metrics is not None and last_metrics.step != last_logged_step:
            emit_train_metrics(last_metrics)
    finally:
        trainer.close()
    print(
        json.dumps(
            {
                "status": (
                    "stopped"
                    if trainer.stop_requested
                    else "early_stopped"
                    if early_stopped
                    else "complete"
                    if trainer.global_step == trainer.config.total_steps
                    else "paused"
                ),
                "global_step": trainer.global_step,
                "samples_seen": trainer.samples_seen,
                "elapsed_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

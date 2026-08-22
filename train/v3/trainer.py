"""Deterministic V3 training with atomic, exact-step checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
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


@dataclass
class _PrefetchedBatchV3:
    base_sampler_state: dict[str, Any]
    committed_sampler_state: dict[str, Any]
    future: Future[TrainingBatchV3]


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
    ) -> None:
        self.dataset = dataset
        self.vocabulary = vocabulary
        self.model_config = model_config
        self.config = training_config
        self.device = _resolve_device(training_config.device)
        _seed_everything(training_config.seed)
        self.model = AugmentNetV3(model_config).to(self.device)
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
        self.parent_checkpoint = parent_checkpoint
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

    def train_step(self) -> StepMetricsV3:
        if self.global_step >= self.config.total_steps:
            raise StopIteration("configured optimizer step limit reached")
        self.model.train()
        cached_batches: list[TrainingBatchV3] = []
        overflow_retries = 0
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_ids: list[str] = []
            totals = torch.zeros(4, dtype=torch.float64)
            microbatches = 0
            samples = 0
            for microbatch_index in range(self.config.gradient_accumulation_steps):
                if overflow_retries == 0:
                    source_batch = self._next_training_batch()
                    cached_batches.append(source_batch)
                else:
                    source_batch = cached_batches[microbatch_index]
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
                    scaled_loss = loss.total / self.config.gradient_accumulation_steps
                if not torch.isfinite(loss.total):
                    raise FloatingPointError("training loss became NaN or infinite")
                self.scaler.scale(scaled_loss).backward()
                totals += torch.tensor(
                    [
                        loss.total.item(),
                        loss.policy.item(),
                        loss.value.item(),
                        loss.auxiliary.item(),
                    ],
                    dtype=torch.float64,
                )
                microbatches += 1
                samples += len(batch.decision_ids)
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
            if not self.scaler.is_enabled() or overflow_retries >= MAX_AMP_OVERFLOW_RETRIES:
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
        averages = totals / microbatches
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

    def _next_indices(self, count: int) -> list[int]:
        return _take_sampler_indices(self.sampler, count)

    def _next_training_batch(self) -> TrainingBatchV3:
        prefetched = self._prefetched_batch
        if prefetched is None:
            indices = self._next_indices(self.config.batch_size)
            batch = _collate_dataset_indices(
                self.dataset,
                indices,
                self.vocabulary,
                pin_memory=self._prefetch_executor is not None
                and self.device.type == "cuda",
            )
        else:
            if self.sampler.state_dict() != prefetched.base_sampler_state:
                raise RuntimeError("prefetched batch sampler state became stale")
            batch = prefetched.future.result()
            self.sampler.load_state_dict(prefetched.committed_sampler_state)
            self._prefetched_batch = None
        self._schedule_prefetch()
        return batch

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
        indices = _take_sampler_indices(preview, self.config.batch_size)
        self._prefetched_batch = _PrefetchedBatchV3(
            base_sampler_state=base_state,
            committed_sampler_state=preview.state_dict(),
            future=self._prefetch_executor.submit(
                _collate_dataset_indices,
                self.dataset,
                indices,
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
        }

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "checkpoint_schema_id": CHECKPOINT_SCHEMA_ID,
            "run_id": self.run_id,
            "parent_checkpoint": self.parent_checkpoint,
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
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--max-steps-this-run", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--prefetch-batches", type=int, choices=(0, 1), default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_steps_this_run is not None and args.max_steps_this_run <= 0:
        parser.error("--max-steps-this-run must be positive")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")
    dataset = load_training_dataset(args.manifest)
    vocabulary = dataset.build_vocabulary()
    config = TrainingConfigV3(
        batch_size=args.batch_size,
        total_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        prefetch_batches=args.prefetch_batches,
    )
    trainer = TrainerV3(
        dataset,
        vocabulary,
        V3ModelConfig(text_vocabulary_size=vocabulary.size),
        config,
        args.run_dir,
    )
    if args.resume:
        print(f"resumed {trainer.resume_latest()}")

    def request_stop(_signum: int, _frame: Any) -> None:
        trainer.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    started = time.perf_counter()
    window_started = started
    window_samples = 0
    last_logged_step = trainer.global_step
    last_metrics = None
    for metrics in trainer.iter_train(steps=args.max_steps_this_run):
        last_metrics = metrics
        window_samples += metrics.samples
        if metrics.step == 1 or metrics.step % args.log_interval == 0:
            now = time.perf_counter()
            payload = asdict(metrics)
            payload["decision_count"] = len(payload.pop("decision_ids"))
            payload["examples_per_second"] = window_samples / (now - window_started)
            print(json.dumps(payload, sort_keys=True), flush=True)
            window_started = now
            window_samples = 0
            last_logged_step = metrics.step
    if last_metrics is not None and last_metrics.step != last_logged_step:
        now = time.perf_counter()
        payload = asdict(last_metrics)
        payload["decision_count"] = len(payload.pop("decision_ids"))
        payload["examples_per_second"] = window_samples / (now - window_started)
        print(json.dumps(payload, sort_keys=True), flush=True)
    trainer.close()
    print(
        json.dumps(
            {
                "status": (
                    "stopped"
                    if trainer.stop_requested
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

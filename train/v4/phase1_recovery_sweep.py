"""Bounded, reproducible Phase 1 recovery sweep for opened-only training.

The sweep reuses one immutable clean corpus split and one canonical initial model
state across every seed/learning-rate cell.  It never rewrites replay records,
exports a production bundle, or promotes a checkpoint.  A fixed optimizer-update
budget is translated into whole trainer epochs; early stopping may end a cell
early, and every completed epoch has its own resumable checkpoint.

The command supports ``--plan-only`` so the complete matrix and corpus identity
can be reviewed and sealed before any training is executed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from v3.tensorize import V3Vocabulary

from .phase1_ab import (
    PHASE1_AB_METRIC_CONTRACT_ID,
    TRAINING_VIEW_OPENED_ONLY,
    ControlledCorpus,
    Phase1ABConfig,
    Phase1ABError,
    TrainingViewBundle,
    _seal_payload,
    _write_idempotent_sealed_json,
    evaluate_common_view,
    load_controlled_corpus,
    make_training_view,
    model_state_sha256,
    sha256_file,
    verify_payload_seal,
)
from .replay import FactorizedTrainingExampleV4
from .train import (
    POLICY_TARGET_SEMANTICS_OPENED_ONLY,
    V4Trainer,
    V4TrainingConfig,
    _git_identity,
    default_runtime_vocabulary_path,
    load_runtime_vocabulary,
    mixed_row_shadow_provenance,
    policy_target_semantics_lineage,
)

RECOVERY_SWEEP_PLAN_SCHEMA_ID = "augmentia.v4-phase1-recovery-sweep-plan.v1"
RECOVERY_SWEEP_RUN_SCHEMA_ID = "augmentia.v4-phase1-recovery-sweep-run.v1"
RECOVERY_SWEEP_RESULT_SCHEMA_ID = "augmentia.v4-phase1-recovery-sweep-result.v1"
RECOVERY_SWEEP_SUMMARY_SCHEMA_ID = "augmentia.v4-phase1-recovery-sweep-summary.v1"
RECOVERY_SWEEP_EVALUATION_SCHEMA_ID = "augmentia.v4-phase1-recovery-sweep-evaluation.v1"
RECOVERY_SWEEP_CONTRACT_ID = "augmentia.v4-phase1-recovery-sweep.v1"
REQUIRED_LEARNING_RATES = (3e-4, 1e-3)
MINIMUM_SWEEP_SEEDS = 4
EARLY_STOP_METRICS = (
    "validation_policy_kl",
    "validation_policy_loss",
    "validation_loss",
)


class RecoverySweepError(RuntimeError):
    """Raised when a recovery sweep would be non-reproducible or unsafe."""


@dataclass(frozen=True)
class RecoveryConfig:
    """Immutable bounded sweep configuration.

    ``trainer.seed`` is a template only.  The split, canonical initialization,
    and each run receive the explicit seeds below.  ``fixed_updates`` is an
    upper bound: without early stopping every cell executes it exactly.
    """

    trainer: V4TrainingConfig
    seeds: tuple[int, ...] = (
        2026091501,
        2026091502,
        2026091503,
        2026091504,
    )
    learning_rates: tuple[float, ...] = REQUIRED_LEARNING_RATES
    fixed_updates: int = 42
    split_seed: int = 2026091501
    initialization_seed: int = 2026091500
    validation_basis_points: int = 2000
    max_train_examples: int | None = None
    max_validation_examples: int | None = None
    runtime_vocabulary_path: str | None = None
    state_encoder_checkpoint: str | None = None
    ece_bins: int = 10
    minimum_game_blocks: int = 30
    lr_schedule: str = "cosine"
    lr_min_ratio: float = 0.1
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    early_stopping_metric: str = "validation_policy_kl"
    allow_smoke_corpus: bool = False
    allow_dirty_smoke: bool = False

    def validate(self) -> "RecoverySweepConfig":
        self.trainer.validate()
        if self.trainer.encoder_kind != "baseline":
            raise RecoverySweepError("recovery sweep requires encoder_kind='baseline'")
        if len(self.seeds) < MINIMUM_SWEEP_SEEDS:
            raise RecoverySweepError(
                f"recovery sweep requires at least {MINIMUM_SWEEP_SEEDS} seeds"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise RecoverySweepError("recovery sweep seeds must be unique")
        if any(seed < 0 or seed >= 2**63 for seed in self.seeds):
            raise RecoverySweepError("recovery sweep seeds must fit signed int63")
        if self.split_seed < 0 or self.initialization_seed < 0:
            raise RecoverySweepError(
                "split and initialization seeds must be non-negative"
            )
        if len(set(self.learning_rates)) != len(self.learning_rates):
            raise RecoverySweepError("recovery sweep learning rates must be unique")
        if any(not math.isfinite(rate) or rate <= 0.0 for rate in self.learning_rates):
            raise RecoverySweepError("learning rates must be finite and positive")
        for required in REQUIRED_LEARNING_RATES:
            if not any(
                math.isclose(rate, required, rel_tol=0.0, abs_tol=1e-15)
                for rate in self.learning_rates
            ):
                raise RecoverySweepError(
                    f"recovery sweep must include learning rate {required}"
                )
        if self.fixed_updates <= 0:
            raise RecoverySweepError("fixed_updates must be positive")
        if not 1 <= self.validation_basis_points <= 9000:
            raise RecoverySweepError("validation_basis_points must be in 1..=9000")
        for name in ("max_train_examples", "max_validation_examples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise RecoverySweepError(f"{name} must be positive when supplied")
        if not 1 <= self.ece_bins <= 100:
            raise RecoverySweepError("ece_bins must be in 1..=100")
        if self.minimum_game_blocks < 8:
            raise RecoverySweepError("minimum_game_blocks must be at least 8")
        if self.lr_schedule not in {"none", "cosine"}:
            raise RecoverySweepError("lr_schedule must be 'none' or 'cosine'")
        if not math.isfinite(self.lr_min_ratio) or not 0.0 < self.lr_min_ratio <= 1.0:
            raise RecoverySweepError("lr_min_ratio must be in (0, 1]")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise RecoverySweepError("early_stopping_patience must be positive or null")
        if (
            not math.isfinite(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0.0
        ):
            raise RecoverySweepError(
                "early_stopping_min_delta must be finite and non-negative"
            )
        if self.early_stopping_metric not in EARLY_STOP_METRICS:
            raise RecoverySweepError(
                f"early_stopping_metric must be one of {EARLY_STOP_METRICS}"
            )
        return self

    @property
    def vocabulary_path(self) -> Path:
        return Path(
            self.runtime_vocabulary_path or default_runtime_vocabulary_path()
        ).resolve()

    def to_json(self) -> dict[str, Any]:
        return _jsonable(
            {
                "schema_id": RECOVERY_SWEEP_CONTRACT_ID,
                "trainer_template": asdict(self.trainer),
                "seeds": list(self.seeds),
                "learning_rates": [float(rate) for rate in self.learning_rates],
                "fixed_updates": self.fixed_updates,
                "split_seed": self.split_seed,
                "initialization_seed": self.initialization_seed,
                "validation_basis_points": self.validation_basis_points,
                "max_train_examples": self.max_train_examples,
                "max_validation_examples": self.max_validation_examples,
                "runtime_vocabulary_path": str(self.vocabulary_path),
                "state_encoder_checkpoint": self.state_encoder_checkpoint,
                "ece_bins": self.ece_bins,
                "minimum_game_blocks": self.minimum_game_blocks,
                "lr_schedule": self.lr_schedule,
                "lr_min_ratio": self.lr_min_ratio,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "early_stopping_metric": self.early_stopping_metric,
                "allow_smoke_corpus": self.allow_smoke_corpus,
                "allow_dirty_smoke": self.allow_dirty_smoke,
            }
        )


# Descriptive compatibility name used by the implementation and tests.
RecoverySweepConfig = RecoveryConfig


@dataclass(frozen=True)
class EarlyStopState:
    best_epoch: int
    best_metric: float
    non_improving_epochs: int
    stopped_early: bool


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_id_for(seed: int, learning_rate: float) -> str:
    """Return a stable filesystem-safe cell identity."""

    return f"seed-{seed:010d}-lr-{learning_rate:.8e}"


def validate_update_budget(fixed_updates: int, batches_per_epoch: int) -> int:
    """Require a fixed update count to align exactly with epoch checkpoints."""

    if fixed_updates <= 0 or batches_per_epoch <= 0:
        raise RecoverySweepError("fixed_updates and batches_per_epoch must be positive")
    if fixed_updates % batches_per_epoch:
        raise RecoverySweepError(
            "fixed_updates must be an exact multiple of batches_per_epoch; "
            f"got {fixed_updates} updates and {batches_per_epoch} batches/epoch"
        )
    return fixed_updates // batches_per_epoch


def learning_rate_at_epoch(
    base_learning_rate: float,
    *,
    schedule: str,
    min_ratio: float,
    epoch_index: int,
    total_epochs: int,
) -> float:
    """Calculate the deterministic pre-epoch learning rate."""

    if total_epochs <= 0 or not 0 <= epoch_index < total_epochs:
        raise RecoverySweepError("invalid epoch index for learning-rate schedule")
    if not math.isfinite(base_learning_rate) or base_learning_rate <= 0.0:
        raise RecoverySweepError("base learning rate must be finite and positive")
    if not 0.0 < min_ratio <= 1.0:
        raise RecoverySweepError("learning-rate minimum ratio must be in (0, 1]")
    if schedule == "none":
        return float(base_learning_rate)
    if schedule != "cosine":
        raise RecoverySweepError(f"unsupported learning-rate schedule: {schedule}")
    if total_epochs == 1:
        return float(base_learning_rate)
    floor = base_learning_rate * min_ratio
    progress = epoch_index / (total_epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (base_learning_rate - floor) * cosine


def _early_stop_metric(event: Mapping[str, Any], name: str) -> float:
    validation = event.get("validation")
    if not isinstance(validation, Mapping):
        raise RecoverySweepError("epoch history entry is missing validation metrics")
    field = {
        "validation_policy_kl": "policy_kl",
        "validation_policy_loss": "policy_loss",
        "validation_loss": "loss",
    }[name]
    value = validation.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RecoverySweepError(f"epoch history entry is missing {field}")
    result = float(value)
    if not math.isfinite(result):
        raise RecoverySweepError(f"early-stop metric {field} is not finite")
    return result


def update_early_stop(
    previous: EarlyStopState | None,
    *,
    epoch: int,
    metric: float,
    patience: int | None,
    min_delta: float,
) -> EarlyStopState:
    """Update best-metric state with standard no-increase early stopping."""

    if previous is None:
        return EarlyStopState(epoch, metric, 0, False)
    if metric < previous.best_metric - min_delta:
        return EarlyStopState(epoch, metric, 0, False)
    if patience is None:
        return EarlyStopState(
            previous.best_epoch,
            previous.best_metric,
            0,
            False,
        )
    non_improving = previous.non_improving_epochs + 1
    return EarlyStopState(
        previous.best_epoch,
        previous.best_metric,
        non_improving,
        non_improving >= patience,
    )


def _run_provenance(
    corpus: ControlledCorpus,
    training_view: TrainingViewBundle,
    *,
    plan_sha256: str,
    run_id: str,
    seed: int,
    learning_rate: float,
    initialization_sha256: str,
) -> dict[str, Any]:
    provenance = dict(corpus.provenance)
    provenance.update(
        {
            "phase1_recovery_sweep_contract_id": RECOVERY_SWEEP_CONTRACT_ID,
            "phase1_recovery_sweep_plan_sha256": plan_sha256,
            "phase1_recovery_sweep_run_id": run_id,
            "phase1_recovery_sweep_seed": seed,
            "phase1_recovery_sweep_learning_rate": float(learning_rate),
            "phase1_initialization_sha256": initialization_sha256,
            "training_view_id": training_view.view_id,
            "training_view_sha256": training_view.view_sha256,
            "source_corpus_sha256": corpus.source_corpus_sha256,
            "split_sha256": corpus.split_sha256,
            "phase1_promotable": False,
            "promotable": False,
        }
    )
    return _jsonable(provenance)


def _phase1_loader_config(config: RecoverySweepConfig) -> Phase1ABConfig:
    trainer = replace(
        config.trainer,
        learning_rate=config.learning_rates[0],
        seed=config.split_seed,
        epochs=1,
    )
    # These two flags only satisfy the Phase 1 loader's parity bypass.  The
    # recovery runner independently checks run_kind and dirty state below.
    return Phase1ABConfig(
        trainer=trainer,
        validation_basis_points=config.validation_basis_points,
        max_train_examples=config.max_train_examples,
        max_validation_examples=config.max_validation_examples,
        checkpoint_every=1,
        runtime_vocabulary_path=str(config.vocabulary_path),
        state_encoder_checkpoint=config.state_encoder_checkpoint,
        ece_bins=config.ece_bins,
        bootstrap_samples=0,
        minimum_game_blocks=config.minimum_game_blocks,
        run_parity=False,
        allow_smoke_corpus=True,
        allow_dirty_smoke=True,
    ).validate()


def _capture_initial_state(
    corpus: ControlledCorpus,
    training_view: TrainingViewBundle,
    config: RecoverySweepConfig,
    vocabulary: V3Vocabulary,
    probe_dir: Path,
) -> tuple[dict[str, torch.Tensor], str]:
    trainer_config = replace(
        config.trainer,
        learning_rate=config.learning_rates[0],
        seed=config.initialization_seed,
        epochs=1,
    )
    trainer = V4Trainer(
        training_view.examples,
        probe_dir,
        trainer_config,
        config.state_encoder_checkpoint,
        vocabulary=vocabulary,
        validation_examples=(),
        provenance={
            "policy_target_semantics_id": POLICY_TARGET_SEMANTICS_OPENED_ONLY,
            "policy_target_semantics_lineage": policy_target_semantics_lineage(
                POLICY_TARGET_SEMANTICS_OPENED_ONLY
            ),
        },
        policy_target_semantics_id=POLICY_TARGET_SEMANTICS_OPENED_ONLY,
    )
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in trainer.model.state_dict().items()
    }
    digest = model_state_sha256(trainer.model)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, digest


def build_sweep_plan(
    *,
    corpus: ControlledCorpus,
    training_view: TrainingViewBundle,
    evaluation_view: TrainingViewBundle,
    config: RecoverySweepConfig,
    batches_per_epoch: int,
    initialization_sha256: str,
    vocabulary_identity: Mapping[str, Any],
    git_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and seal the complete pre-execution sweep matrix."""

    planned_epochs = validate_update_budget(config.fixed_updates, batches_per_epoch)
    combinations = [
        {
            "run_id": run_id_for(seed, learning_rate),
            "seed": seed,
            "learning_rate": float(learning_rate),
        }
        for seed in config.seeds
        for learning_rate in config.learning_rates
    ]
    config_payload = config.to_json()
    config_sha = _seal_payload(
        {"schema_id": RECOVERY_SWEEP_CONTRACT_ID, **config_payload},
        "config_sha256",
    )["config_sha256"]
    state_encoder_sha256 = (
        sha256_file(config.state_encoder_checkpoint)
        if config.state_encoder_checkpoint is not None
        else None
    )
    payload = {
        "schema_id": RECOVERY_SWEEP_PLAN_SCHEMA_ID,
        "contract_id": RECOVERY_SWEEP_CONTRACT_ID,
        "metric_contract_id": PHASE1_AB_METRIC_CONTRACT_ID,
        "status": "planned",
        "source_corpus": corpus.to_json(),
        "source_corpus_sha256": corpus.source_corpus_sha256,
        "split": {
            "split_sha256": corpus.split_sha256,
            "split_seed": config.split_seed,
            "validation_basis_points": config.validation_basis_points,
            "train_game_ids": list(corpus.train_game_ids),
            "validation_game_ids": list(corpus.validation_game_ids),
            "train_examples": len(training_view.examples),
            "validation_examples": len(evaluation_view.examples),
        },
        "training_view": training_view.to_json(),
        "evaluation_view": evaluation_view.to_json(),
        "initialization_seed": config.initialization_seed,
        "initialization_sha256": initialization_sha256,
        "same_initialization_every_run": True,
        "trainer_template": config_payload["trainer_template"],
        "state_encoder_checkpoint_sha256": state_encoder_sha256,
        "sweep_config": config_payload,
        "sweep_config_sha256": config_sha,
        "batches_per_epoch": batches_per_epoch,
        "fixed_update_budget": config.fixed_updates,
        "planned_epochs_without_early_stop": planned_epochs,
        "combinations": combinations,
        "run_count": len(combinations),
        "runtime_vocabulary": _jsonable(dict(vocabulary_identity)),
        "git": _jsonable(dict(git_identity)),
        "source_policy": {
            "read_only": True,
            "rewrite_replay": False,
            "rewrite_dataset_index": False,
            "modify_existing_artifacts": False,
            "new_output_only": True,
        },
        "promotable": False,
        "promotion_note": (
            "Optimization sweep evidence is diagnostic and non-promotable; Phase 2 "
            "paired strength and CI gates remain mandatory."
        ),
    }
    return _seal_payload(payload, "plan_sha256")


def _load_existing_sealed(path: Path, seal_field: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        verify_payload_seal(payload, seal_field)
    except Phase1ABError as error:
        raise RecoverySweepError(
            f"invalid sealed recovery artifact {path}: {error}"
        ) from error
    return payload


def _set_optimizer_learning_rate(trainer: V4Trainer, learning_rate: float) -> None:
    for group in trainer.optimizer.param_groups:
        group["lr"] = float(learning_rate)


def run_bounded_training(
    trainer: V4Trainer,
    *,
    fixed_updates: int,
    planned_epochs: int,
    base_learning_rate: float,
    lr_schedule: str,
    lr_min_ratio: float,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    early_stopping_metric: str,
) -> dict[str, Any]:
    """Execute a fixed-update budget in whole epochs with epoch checkpoints."""

    if (
        trainer.history
        or trainer.completed_epochs
        or trainer.epoch_progress is not None
    ):
        raise RecoverySweepError("bounded sweep requires a fresh trainer")
    batches_per_epoch = trainer.batch_count()
    if planned_epochs * batches_per_epoch != fixed_updates:
        raise RecoverySweepError("planned epochs do not equal the fixed update budget")
    best: EarlyStopState | None = None
    updates = 0
    stop_requested = False
    for epoch_index in range(planned_epochs):
        learning_rate = learning_rate_at_epoch(
            base_learning_rate,
            schedule=lr_schedule,
            min_ratio=lr_min_ratio,
            epoch_index=epoch_index,
            total_epochs=planned_epochs,
        )
        _set_optimizer_learning_rate(trainer, learning_rate)
        before = trainer.completed_epochs
        trainer.fit(checkpoint_every=1, checkpoint_prefix="checkpoint")
        if trainer.completed_epochs != before + 1 or len(trainer.history) != before + 1:
            raise RecoverySweepError("trainer did not complete exactly one epoch")
        event = trainer.history[-1]
        epoch_batches = int(event["train"]["batches"])
        if epoch_batches != batches_per_epoch:
            raise RecoverySweepError("epoch batch count differs from the fixed budget")
        updates += epoch_batches
        metric = _early_stop_metric(event, early_stopping_metric)
        best = update_early_stop(
            best,
            epoch=int(event["epoch"]),
            metric=metric,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )
        if best.stopped_early:
            stop_requested = True
            break
    assert best is not None
    selected = trainer.run_dir / f"checkpoint-epoch-{best.best_epoch:04d}.pt"
    sidecar = selected.with_suffix(".v4-checkpoint.json")
    if not selected.is_file() or not sidecar.is_file():
        raise RecoverySweepError("selected epoch checkpoint is missing")
    return {
        "fixed_update_budget": fixed_updates,
        "updates_executed": updates,
        "batches_per_epoch": batches_per_epoch,
        "epochs_executed": trainer.completed_epochs,
        "planned_epochs_without_early_stop": planned_epochs,
        "early_stopped": stop_requested,
        "early_stopping_metric": early_stopping_metric,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "best_epoch": best.best_epoch,
        "best_validation_metric": best.best_metric,
        "selected_checkpoint_path": str(selected.resolve()),
        "selected_sidecar_path": str(sidecar.resolve()),
        "history_path": str((trainer.run_dir / "history.json").resolve()),
    }


def _load_or_run_result(
    *,
    run_id: str,
    seed: int,
    learning_rate: float,
    run_dir: Path,
    result_path: Path,
    plan_sha256: str,
) -> dict[str, Any] | None:
    existing = _load_existing_sealed(result_path, "result_sha256")
    if existing is None:
        return None
    if (
        existing.get("plan_sha256") != plan_sha256
        or existing.get("run_id") != run_id
        or existing.get("seed") != seed
        or existing.get("learning_rate") != float(learning_rate)
    ):
        raise RecoverySweepError(
            f"completed sweep result identity drifted: {result_path}"
        )
    return existing


def run_one(
    *,
    config: RecoverySweepConfig,
    corpus: ControlledCorpus,
    training_view: TrainingViewBundle,
    validation_examples: Sequence[FactorizedTrainingExampleV4],
    vocabulary: V3Vocabulary,
    initial_state: Mapping[str, torch.Tensor],
    plan: Mapping[str, Any],
    run_id: str,
    seed: int,
    learning_rate: float,
    batches_per_epoch: int,
    planned_epochs: int,
    output_dir: Path,
) -> dict[str, Any]:
    run_dir = output_dir / run_id
    result_path = run_dir / "run-result.json"
    existing = _load_or_run_result(
        run_id=run_id,
        seed=seed,
        learning_rate=learning_rate,
        run_dir=run_dir,
        result_path=result_path,
        plan_sha256=str(plan["plan_sha256"]),
    )
    if existing is not None:
        return existing
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RecoverySweepError(
            f"incomplete or different sweep run directory exists: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = _run_provenance(
        corpus,
        training_view,
        plan_sha256=str(plan["plan_sha256"]),
        run_id=run_id,
        seed=seed,
        learning_rate=learning_rate,
        initialization_sha256=str(plan["initialization_sha256"]),
    )
    trainer_config = replace(
        config.trainer,
        learning_rate=float(learning_rate),
        seed=seed,
        epochs=1,
    )
    run_payload = _seal_payload(
        {
            "schema_id": RECOVERY_SWEEP_RUN_SCHEMA_ID,
            "contract_id": RECOVERY_SWEEP_CONTRACT_ID,
            "run_id": run_id,
            "seed": seed,
            "learning_rate": float(learning_rate),
            "plan_sha256": plan["plan_sha256"],
            "source_corpus_sha256": corpus.source_corpus_sha256,
            "split_sha256": corpus.split_sha256,
            "training_view": training_view.to_json(),
            "initialization_sha256": plan["initialization_sha256"],
            "fixed_update_budget": config.fixed_updates,
            "batches_per_epoch": batches_per_epoch,
            "planned_epochs_without_early_stop": planned_epochs,
            "training_config": _jsonable(asdict(trainer_config)),
            "provenance": provenance,
            "promotable": False,
        },
        "run_sha256",
    )
    _write_idempotent_sealed_json(run_dir / "run.json", run_payload, "run_sha256")
    trainer = V4Trainer(
        training_view.examples,
        run_dir,
        trainer_config,
        config.state_encoder_checkpoint,
        vocabulary=vocabulary,
        validation_examples=validation_examples,
        provenance=provenance,
        policy_target_semantics_id=POLICY_TARGET_SEMANTICS_OPENED_ONLY,
    )
    trainer.model.load_state_dict(dict(initial_state), strict=True)
    if model_state_sha256(trainer.model) != plan["initialization_sha256"]:
        raise RecoverySweepError(f"{run_id} initialization drifted before training")
    training = run_bounded_training(
        trainer,
        fixed_updates=config.fixed_updates,
        planned_epochs=planned_epochs,
        base_learning_rate=float(learning_rate),
        lr_schedule=config.lr_schedule,
        lr_min_ratio=config.lr_min_ratio,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_metric=config.early_stopping_metric,
    )
    selected = Path(training["selected_checkpoint_path"])
    selected_sidecar = Path(training["selected_sidecar_path"])
    state = torch.load(selected, map_location=trainer.device, weights_only=True)
    trainer.model.load_state_dict(state, strict=True)
    evaluation = evaluate_common_view(
        trainer.model,
        corpus.validation_examples,
        vocabulary,
        trainer.device,
        batch_size=trainer_config.batch_size,
        ece_bins=config.ece_bins,
    )
    evaluation.update(
        {
            "schema_id": RECOVERY_SWEEP_EVALUATION_SCHEMA_ID,
            "contract_id": RECOVERY_SWEEP_CONTRACT_ID,
            "metric_contract_id": PHASE1_AB_METRIC_CONTRACT_ID,
            "run_id": run_id,
            "seed": seed,
            "learning_rate": float(learning_rate),
            "training_view_id": training_view.view_id,
            "evaluation_view_id": TRAINING_VIEW_OPENED_ONLY,
            "source_corpus_sha256": corpus.source_corpus_sha256,
            "split_sha256": corpus.split_sha256,
            "initialization_sha256": plan["initialization_sha256"],
            "selected_checkpoint_sha256": sha256_file(selected),
            "selected_epoch": training["best_epoch"],
            "updates_executed": training["updates_executed"],
            "early_stopped": training["early_stopped"],
            "promotable": False,
        }
    )
    evaluation = _seal_payload(evaluation, "evaluation_sha256")
    evaluation_path = _write_idempotent_sealed_json(
        run_dir / "heldout-evaluation.json", evaluation, "evaluation_sha256"
    )
    sidecar_payload = json.loads(selected_sidecar.read_text(encoding="utf-8"))
    if (
        sidecar_payload.get("provenance", {}).get("phase1_recovery_sweep_plan_sha256")
        != plan["plan_sha256"]
    ):
        raise RecoverySweepError("selected checkpoint sweep provenance drifted")
    result = {
        "schema_id": RECOVERY_SWEEP_RESULT_SCHEMA_ID,
        "contract_id": RECOVERY_SWEEP_CONTRACT_ID,
        "status": "complete",
        "run_id": run_id,
        "seed": seed,
        "learning_rate": float(learning_rate),
        "plan_sha256": plan["plan_sha256"],
        "source_corpus_sha256": corpus.source_corpus_sha256,
        "split_sha256": corpus.split_sha256,
        "initialization_sha256": plan["initialization_sha256"],
        "training": _jsonable(training),
        "selected_checkpoint_path": str(selected),
        "selected_checkpoint_sha256": sha256_file(selected),
        "selected_sidecar_path": str(selected_sidecar),
        "selected_sidecar_sha256": sha256_file(selected_sidecar),
        "history_path": training["history_path"],
        "history_sha256": sha256_file(Path(training["history_path"])),
        "evaluation_path": str(evaluation_path),
        "evaluation_file_sha256": sha256_file(evaluation_path),
        "evaluation_seal_sha256": evaluation["evaluation_sha256"],
        "metrics": {
            "conditional_kl": evaluation["conditional_kl"],
            "opened_support_complete_action_soft_nll": evaluation[
                "opened_support_complete_action_soft_nll"
            ],
            "opened_support_all_modal_top1": evaluation[
                "opened_support_all_modal_top1"
            ],
            "opened_support_fractional_modal_score": evaluation[
                "opened_support_fractional_modal_score"
            ],
            "opened_support_visited_tail_recall_at_3": evaluation[
                "opened_support_visited_tail_recall_at_3"
            ],
            "opened_support_mcts_modal_ece": evaluation[
                "opened_support_mcts_modal_ece"
            ],
        },
        "promotable": False,
    }
    result = _seal_payload(result, "result_sha256")
    _write_idempotent_sealed_json(result_path, result, "result_sha256")
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _metric_stats(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_sweep_summary(
    plan: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    source_immutability_verified: bool,
) -> dict[str, Any]:
    """Aggregate and seal diagnostic ranking without promotion semantics."""

    expected_ids = {item["run_id"] for item in plan["combinations"]}
    result_ids = {item["run_id"] for item in results}
    if result_ids != expected_ids or len(results) != len(expected_ids):
        raise RecoverySweepError("sweep results do not exactly cover the sealed matrix")
    ranked = sorted(
        results,
        key=lambda item: (
            float(item["metrics"]["conditional_kl"]),
            float(item["metrics"]["opened_support_complete_action_soft_nll"]),
            -float(item["metrics"]["opened_support_all_modal_top1"]),
            item["run_id"],
        ),
    )
    learning_rate_groups = []
    for learning_rate in sorted({float(item["learning_rate"]) for item in results}):
        group = [item for item in results if item["learning_rate"] == learning_rate]
        learning_rate_groups.append(
            {
                "learning_rate": learning_rate,
                "seed_count": len(group),
                "conditional_kl": _metric_stats(
                    [float(item["metrics"]["conditional_kl"]) for item in group]
                ),
                "opened_support_complete_action_soft_nll": _metric_stats(
                    [
                        float(
                            item["metrics"]["opened_support_complete_action_soft_nll"]
                        )
                        for item in group
                    ]
                ),
                "opened_support_all_modal_top1": _metric_stats(
                    [
                        float(item["metrics"]["opened_support_all_modal_top1"])
                        for item in group
                    ]
                ),
            }
        )
    validation_blocks = len(plan["split"]["validation_game_ids"])
    minimum_blocks = int(plan["sweep_config"]["minimum_game_blocks"])
    summary = {
        "schema_id": RECOVERY_SWEEP_SUMMARY_SCHEMA_ID,
        "contract_id": RECOVERY_SWEEP_CONTRACT_ID,
        "metric_contract_id": PHASE1_AB_METRIC_CONTRACT_ID,
        "status": "complete",
        "plan_sha256": plan["plan_sha256"],
        "source_corpus_sha256": plan["source_corpus_sha256"],
        "split_sha256": plan["split"]["split_sha256"],
        "initialization_sha256": plan["initialization_sha256"],
        "source_immutability_verified": source_immutability_verified,
        "run_count": len(results),
        "ranking_metric": (
            "ascending conditional_kl, then opened-support soft NLL, then descending "
            "all-modal top-1, then run_id"
        ),
        "ranked_run_ids": [item["run_id"] for item in ranked],
        "best_run_id": ranked[0]["run_id"],
        "learning_rate_groups": learning_rate_groups,
        "runs": [_jsonable(item) for item in ranked],
        "evidence": {
            "validation_game_blocks": validation_blocks,
            "minimum_game_blocks": minimum_blocks,
            "evidence_sufficient": validation_blocks >= minimum_blocks,
            "promotion_effect": False,
        },
        "early_stopped_runs": sum(
            bool(item["training"]["early_stopped"]) for item in results
        ),
        "total_updates_executed": sum(
            int(item["training"]["updates_executed"]) for item in results
        ),
        "promotable": False,
        "promotion_note": (
            "A sweep winner is diagnostic only. It cannot be exported/promoted without "
            "the separate Phase 2 paired strength and CI gates."
        ),
    }
    return _seal_payload(summary, "summary_sha256")


def verify_source_unchanged(
    manifest_path: str | Path,
    manifest_sha256: str,
    game_identities: Sequence[Mapping[str, Any]],
) -> None:
    """Fail if any source manifest or replay byte changed during the sweep."""

    manifest = Path(manifest_path)
    if not manifest.is_file() or sha256_file(manifest) != manifest_sha256:
        raise RecoverySweepError(
            "source dataset manifest changed during recovery sweep"
        )
    for identity in game_identities:
        path = Path(str(identity["path"]))
        if not path.is_file() or sha256_file(path) != identity["file_sha256"]:
            raise RecoverySweepError(
                f"source replay changed during recovery sweep: {path}"
            )


def run_sweep(
    manifest: str | Path,
    output_dir: str | Path,
    config: RecoverySweepConfig,
    *,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Build a sealed plan and optionally execute the bounded sweep."""

    config.validate()
    manifest_path = Path(manifest).resolve()
    root = Path(output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    commit, dirty = _git_identity(repo_root)
    if dirty and not config.allow_dirty_smoke:
        raise RecoverySweepError("dirty worktree requires allow_dirty_smoke=True")
    if dirty and "smoke" not in root.name.lower():
        raise RecoverySweepError(
            "dirty smoke output directory name must contain 'smoke'"
        )
    root.mkdir(parents=True, exist_ok=True)
    vocabulary, vocabulary_identity = load_runtime_vocabulary(config.vocabulary_path)
    loader_config = _phase1_loader_config(config)
    corpus = load_controlled_corpus(manifest_path, loader_config, vocabulary_identity)
    if corpus.dataset.run_kind != "production" and not config.allow_smoke_corpus:
        raise RecoverySweepError(
            "non-production corpus requires allow_smoke_corpus=True"
        )
    provenance = dict(corpus.provenance)
    provenance["mixed_row_shadow"] = mixed_row_shadow_provenance(
        config.trainer.mixed_row_shadow_lambda
    )
    provenance.update(
        {
            "trainer_git_commit": commit,
            "trainer_dirty_worktree": dirty,
        }
    )
    corpus = replace(corpus, provenance=provenance)
    training_view = make_training_view(corpus.train_examples, "opened_only")
    evaluation_view = make_training_view(corpus.validation_examples, "opened_only")
    batches_per_epoch = (
        len(training_view.examples) + config.trainer.batch_size - 1
    ) // config.trainer.batch_size
    planned_epochs = validate_update_budget(config.fixed_updates, batches_per_epoch)
    initial_state, initialization_sha256 = _capture_initial_state(
        corpus,
        training_view,
        config,
        vocabulary,
        root / ".init-probe",
    )
    plan = build_sweep_plan(
        corpus=corpus,
        training_view=training_view,
        evaluation_view=evaluation_view,
        config=config,
        batches_per_epoch=batches_per_epoch,
        initialization_sha256=initialization_sha256,
        vocabulary_identity=vocabulary_identity,
        git_identity={"commit": commit, "dirty": dirty},
    )
    plan_path = _write_idempotent_sealed_json(
        root / "sweep-plan.json", plan, "plan_sha256"
    )
    if plan_only:
        del initial_state
        gc.collect()
        return {**plan, "plan_path": str(plan_path)}
    summary_path = root / "sweep-summary.json"
    existing_summary = _load_existing_sealed(summary_path, "summary_sha256")
    if existing_summary is not None:
        if existing_summary.get("plan_sha256") != plan["plan_sha256"]:
            raise RecoverySweepError(
                "existing sweep summary references a different plan"
            )
        verify_source_unchanged(
            corpus.dataset.path,
            corpus.dataset.sha256,
            corpus.game_identities,
        )
        del initial_state
        gc.collect()
        return existing_summary
    results: list[dict[str, Any]] = []
    for cell in plan["combinations"]:
        results.append(
            run_one(
                config=config,
                corpus=corpus,
                training_view=training_view,
                validation_examples=evaluation_view.examples,
                vocabulary=vocabulary,
                initial_state=initial_state,
                plan=plan,
                run_id=str(cell["run_id"]),
                seed=int(cell["seed"]),
                learning_rate=float(cell["learning_rate"]),
                batches_per_epoch=batches_per_epoch,
                planned_epochs=planned_epochs,
                output_dir=root,
            )
        )
    verify_source_unchanged(
        corpus.dataset.path,
        corpus.dataset.sha256,
        corpus.game_identities,
    )
    summary = build_sweep_summary(plan, results, source_immutability_verified=True)
    _write_idempotent_sealed_json(summary_path, summary, "summary_sha256")
    del initial_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# Backward-compatible descriptive alias.
run_recovery_sweep = run_sweep


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--seed",
        dest="seeds",
        type=int,
        action="append",
        help="repeat at least four times; defaults to four fixed recovery seeds",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rates",
        type=float,
        action="append",
        help="repeat for each LR; 3e-4 and 1e-3 are mandatory defaults",
    )
    parser.add_argument("--fixed-updates", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2026091501)
    parser.add_argument("--initialization-seed", type=int, default=2026091500)
    parser.add_argument("--validation-bp", type=int, default=2000)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--value-weight", type=float, default=0.0)
    parser.add_argument("--mixed-row-shadow-lambda", type=float, default=0.0)
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
    parser.add_argument("--tensor-cache-max-batches", type=int, default=0)
    parser.add_argument(
        "--runtime-vocabulary",
        type=Path,
        default=default_runtime_vocabulary_path(),
    )
    parser.add_argument("--state-encoder-checkpoint", type=Path)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--minimum-game-blocks", type=int, default=30)
    parser.add_argument("--lr-schedule", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-metric",
        choices=EARLY_STOP_METRICS,
        default="validation_policy_kl",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-smoke-corpus", action="store_true")
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    seeds = tuple(
        args.seeds or RecoverySweepConfig.__dataclass_fields__["seeds"].default
    )
    learning_rates = tuple(
        args.learning_rates
        or RecoverySweepConfig.__dataclass_fields__["learning_rates"].default
    )
    trainer = V4TrainingConfig(
        learning_rate=learning_rates[0],
        batch_size=args.batch_size,
        epochs=1,
        value_weight=args.value_weight,
        mixed_row_shadow_lambda=args.mixed_row_shadow_lambda,
        seed=args.split_seed,
        device=args.device,
        model_dim=args.model_dim,
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        runtime_message_passes=args.runtime_message_passes,
        encoder_kind="baseline",
        batching_kind=args.batching_kind,
        tensor_cache_max_batches=args.tensor_cache_max_batches,
    )
    config = RecoverySweepConfig(
        trainer=trainer,
        seeds=seeds,
        learning_rates=learning_rates,
        fixed_updates=args.fixed_updates,
        split_seed=args.split_seed,
        initialization_seed=args.initialization_seed,
        validation_basis_points=args.validation_bp,
        max_train_examples=args.max_train_examples,
        max_validation_examples=args.max_validation_examples,
        runtime_vocabulary_path=str(args.runtime_vocabulary),
        state_encoder_checkpoint=(
            str(args.state_encoder_checkpoint)
            if args.state_encoder_checkpoint is not None
            else None
        ),
        ece_bins=args.ece_bins,
        minimum_game_blocks=args.minimum_game_blocks,
        lr_schedule=args.lr_schedule,
        lr_min_ratio=args.lr_min_ratio,
        early_stopping_patience=(args.early_stop_patience if args.early_stop else None),
        early_stopping_min_delta=args.early_stop_min_delta,
        early_stopping_metric=args.early_stop_metric,
        allow_smoke_corpus=args.allow_smoke_corpus,
        allow_dirty_smoke=args.allow_dirty_smoke,
    )
    result = run_sweep(
        args.manifest,
        args.output_dir,
        config,
        plan_only=args.plan_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

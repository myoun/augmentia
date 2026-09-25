"""Bounded Phase 1 mixed-row shadow-negative ablation.

This runner deliberately trains only the opened-only production arm.  The
legacy all-open arm is not a promotion candidate and is not needed to test the
causal hypothesis that a small mixed-row negative term changes shared scorer
calibration.  It uses the P0 opened-support evaluation contract and seals every
cell; it never modifies source replay or exports a production bundle.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from v3.tensorize import V3Vocabulary

from .phase1_ab import (
    PHASE1_AB_METRIC_CONTRACT_ID,
    ControlledCorpus,
    Phase1ABConfig,
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

SHADOW_ABLATION_PLAN_SCHEMA_ID = "augmentia.v4-phase1-shadow-ablation-plan.v1"
SHADOW_ABLATION_RESULT_SCHEMA_ID = "augmentia.v4-phase1-shadow-ablation-result.v1"
SHADOW_ABLATION_SUMMARY_SCHEMA_ID = "augmentia.v4-phase1-shadow-ablation-summary.v1"
SHADOW_ABLATION_CONTRACT_ID = "augmentia.v4-phase1-shadow-ablation.v1"
DEFAULT_LAMBDAS = (0.0, 0.01, 0.05, 0.1)


class ShadowAblationError(RuntimeError):
    """Raised when a shadow ablation is not reproducible or fail-closed."""


@dataclass(frozen=True)
class ShadowAblationConfig:
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    seed: int = 2026091501
    split_seed: int = 2026091501
    initialization_seed: int = 2026091500
    learning_rate: float = 1e-3
    epochs: int = 1
    batch_size: int = 16
    value_weight: float = 0.0
    device: str = "cuda"
    model_dim: int = 128
    channels: int = 128
    residual_blocks: int = 6
    runtime_message_passes: int = 2
    tensor_cache_max_batches: int = 0
    validation_basis_points: int = 2000
    minimum_game_blocks: int = 30
    ece_bins: int = 10
    runtime_vocabulary_path: str | None = None
    state_encoder_checkpoint: str | None = None
    allow_smoke_corpus: bool = False
    allow_dirty_smoke: bool = False

    def validate(self) -> "ShadowAblationConfig":
        if not self.lambdas:
            raise ShadowAblationError("at least one shadow lambda is required")
        if any(
            not math.isfinite(value) or value < 0.0 for value in self.lambdas
        ):
            raise ShadowAblationError("shadow lambdas must be finite and non-negative")
        if len(set(self.lambdas)) != len(self.lambdas):
            raise ShadowAblationError("shadow lambdas must be unique")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ShadowAblationError("epochs and batch size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ShadowAblationError("learning rate must be finite and positive")
        if not math.isfinite(self.value_weight) or self.value_weight < 0.0:
            raise ShadowAblationError("value weight must be finite and non-negative")
        if self.device not in {"cpu", "cuda"}:
            raise ShadowAblationError("device must be cpu or cuda")
        if not 1 <= self.validation_basis_points <= 9000:
            raise ShadowAblationError("validation basis points must be in 1..=9000")
        if not 1 <= self.ece_bins <= 100:
            raise ShadowAblationError("ECE bins must be in 1..=100")
        if self.minimum_game_blocks < 8:
            raise ShadowAblationError("minimum game blocks must be at least 8")
        return self

    @property
    def vocabulary_path(self) -> Path:
        return Path(
            self.runtime_vocabulary_path or default_runtime_vocabulary_path()
        ).resolve()

    def trainer_config(self, lambda_value: float, *, seed: int | None = None) -> V4TrainingConfig:
        return V4TrainingConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            value_weight=self.value_weight,
            mixed_row_shadow_lambda=float(lambda_value),
            seed=self.seed if seed is None else seed,
            device=self.device,
            model_dim=self.model_dim,
            channels=self.channels,
            residual_blocks=self.residual_blocks,
            runtime_message_passes=self.runtime_message_passes,
            encoder_kind="baseline",
            batching_kind="random",
            tensor_cache_max_batches=self.tensor_cache_max_batches,
        ).validate()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": SHADOW_ABLATION_CONTRACT_ID,
            "lambdas": [float(value) for value in self.lambdas],
            "seed": self.seed,
            "split_seed": self.split_seed,
            "initialization_seed": self.initialization_seed,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "value_weight": self.value_weight,
            "device": self.device,
            "model_dim": self.model_dim,
            "channels": self.channels,
            "residual_blocks": self.residual_blocks,
            "runtime_message_passes": self.runtime_message_passes,
            "tensor_cache_max_batches": self.tensor_cache_max_batches,
            "validation_basis_points": self.validation_basis_points,
            "minimum_game_blocks": self.minimum_game_blocks,
            "ece_bins": self.ece_bins,
            "runtime_vocabulary_path": str(self.vocabulary_path),
            "state_encoder_checkpoint": self.state_encoder_checkpoint,
            "allow_smoke_corpus": self.allow_smoke_corpus,
            "allow_dirty_smoke": self.allow_dirty_smoke,
        }


def _loader_config(config: ShadowAblationConfig) -> Phase1ABConfig:
    return Phase1ABConfig(
        trainer=config.trainer_config(0.0, seed=config.split_seed),
        validation_basis_points=config.validation_basis_points,
        runtime_vocabulary_path=str(config.vocabulary_path),
        state_encoder_checkpoint=config.state_encoder_checkpoint,
        ece_bins=config.ece_bins,
        bootstrap_samples=0,
        minimum_game_blocks=config.minimum_game_blocks,
        run_parity=False,
        allow_smoke_corpus=True,
        allow_dirty_smoke=True,
    ).validate()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _capture_initial_state(
    corpus: ControlledCorpus,
    view: TrainingViewBundle,
    config: ShadowAblationConfig,
    vocabulary: V3Vocabulary,
    probe_dir: Path,
) -> tuple[dict[str, torch.Tensor], str]:
    trainer = V4Trainer(
        view.examples,
        probe_dir,
        config.trainer_config(0.0, seed=config.initialization_seed),
        config.state_encoder_checkpoint,
        vocabulary=vocabulary,
        provenance={
            "policy_target_semantics_id": POLICY_TARGET_SEMANTICS_OPENED_ONLY,
            "policy_target_semantics_lineage": policy_target_semantics_lineage(
                POLICY_TARGET_SEMANTICS_OPENED_ONLY
            ),
            "mixed_row_shadow": mixed_row_shadow_provenance(0.0),
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


def _cell_id(lambda_value: float) -> str:
    return f"lambda-{lambda_value:.8e}"


def _metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    primary = evaluation["opened_support_primary"]
    tail = primary["visited_tail_recall_at_3"]
    return {
        "conditional_kl": float(evaluation["conditional_kl"]),
        "opened_support_soft_nll": float(
            evaluation["opened_support_complete_action_soft_nll"]
        ),
        "opened_support_all_modal_top1": float(
            evaluation["opened_support_all_modal_top1"]
        ),
        "opened_support_fractional_modal_score": float(
            evaluation["opened_support_fractional_modal_score"]
        ),
        "opened_support_tail_macro": float(tail["macro"]),
        "opened_support_tail_micro": float(tail["micro"]),
        "opened_support_all_modal_ece": float(
            evaluation["opened_support_mcts_modal_ece"]
        ),
        "opened_support_fractional_ece": float(
            evaluation["opened_support_mcts_modal_fractional_ece"]
        ),
    }


def _improvement(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, float]:
    return {
        "conditional_kl": float(baseline["conditional_kl"] - current["conditional_kl"]),
        "opened_support_soft_nll": float(
            baseline["opened_support_soft_nll"] - current["opened_support_soft_nll"]
        ),
        "opened_support_all_modal_top1": float(
            current["opened_support_all_modal_top1"]
            - baseline["opened_support_all_modal_top1"]
        ),
        "opened_support_fractional_modal_score": float(
            current["opened_support_fractional_modal_score"]
            - baseline["opened_support_fractional_modal_score"]
        ),
        "opened_support_tail_macro": float(
            current["opened_support_tail_macro"] - baseline["opened_support_tail_macro"]
        ),
        "opened_support_tail_micro": float(
            current["opened_support_tail_micro"] - baseline["opened_support_tail_micro"]
        ),
        "opened_support_all_modal_ece": float(
            baseline["opened_support_all_modal_ece"]
            - current["opened_support_all_modal_ece"]
        ),
        "opened_support_fractional_ece": float(
            baseline["opened_support_fractional_ece"]
            - current["opened_support_fractional_ece"]
        ),
    }


def _build_plan(
    corpus: ControlledCorpus,
    training_view: TrainingViewBundle,
    evaluation_view: TrainingViewBundle,
    config: ShadowAblationConfig,
    initialization_sha256: str,
    vocabulary_identity: Mapping[str, Any],
    git_identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_id": SHADOW_ABLATION_PLAN_SCHEMA_ID,
        "contract_id": SHADOW_ABLATION_CONTRACT_ID,
        "metric_contract_id": PHASE1_AB_METRIC_CONTRACT_ID,
        "status": "planned",
        "source_corpus": corpus.to_json(),
        "source_corpus_sha256": corpus.source_corpus_sha256,
        "split_sha256": corpus.split_sha256,
        "training_view": training_view.to_json(),
        "evaluation_view": evaluation_view.to_json(),
        "initialization_seed": config.initialization_seed,
        "initialization_sha256": initialization_sha256,
        "config": config.to_json(),
        "lambdas": [float(value) for value in config.lambdas],
        "runtime_vocabulary": _jsonable(dict(vocabulary_identity)),
        "git": _jsonable(dict(git_identity)),
        "source_policy": {
            "read_only": True,
            "modify_existing_artifacts": False,
            "production_export": False,
        },
        "promotable": False,
        "promotion_note": "Diagnostic shadow ablation only; Phase 2 strength gate remains mandatory.",
    }
    return _seal_payload(payload, "plan_sha256")


def _run_cell(
    *,
    config: ShadowAblationConfig,
    corpus: ControlledCorpus,
    training_view: TrainingViewBundle,
    evaluation_view: TrainingViewBundle,
    vocabulary: V3Vocabulary,
    initial_state: Mapping[str, torch.Tensor],
    plan: Mapping[str, Any],
    lambda_value: float,
    output_dir: Path,
) -> dict[str, Any]:
    cell_id = _cell_id(lambda_value)
    cell_dir = output_dir / cell_id
    result_path = cell_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        verify_payload_seal(result, "result_sha256")
        if result.get("plan_sha256") != plan["plan_sha256"]:
            raise ShadowAblationError("existing shadow result has a different plan")
        return result
    if cell_dir.exists() and any(cell_dir.iterdir()):
        raise ShadowAblationError(f"incomplete shadow cell exists: {cell_dir}")
    cell_dir.mkdir(parents=True, exist_ok=True)
    trainer_config = config.trainer_config(lambda_value, seed=config.seed)
    provenance = dict(corpus.provenance)
    provenance.update(
        {
            "policy_target_semantics_id": POLICY_TARGET_SEMANTICS_OPENED_ONLY,
            "policy_target_semantics_lineage": policy_target_semantics_lineage(
                POLICY_TARGET_SEMANTICS_OPENED_ONLY
            ),
            "mixed_row_shadow": mixed_row_shadow_provenance(lambda_value),
            "phase1_shadow_ablation_plan_sha256": plan["plan_sha256"],
            "phase1_shadow_ablation_lambda": float(lambda_value),
            "phase1_initialization_sha256": plan["initialization_sha256"],
            "promotable": False,
        }
    )
    run_payload = _seal_payload(
        {
            "schema_id": "augmentia.v4-phase1-shadow-ablation-run.v1",
            "contract_id": SHADOW_ABLATION_CONTRACT_ID,
            "cell_id": cell_id,
            "lambda": float(lambda_value),
            "plan_sha256": plan["plan_sha256"],
            "training_view_id": training_view.view_id,
            "training_view_sha256": training_view.view_sha256,
            "evaluation_view_id": evaluation_view.view_id,
            "evaluation_view_sha256": evaluation_view.view_sha256,
            "initialization_sha256": plan["initialization_sha256"],
            "source_corpus_sha256": corpus.source_corpus_sha256,
            "split_sha256": corpus.split_sha256,
            "training_config": asdict(trainer_config),
            "provenance": _jsonable(provenance),
            "promotable": False,
        },
        "run_sha256",
    )
    _write_idempotent_sealed_json(cell_dir / "run.json", run_payload, "run_sha256")
    trainer = V4Trainer(
        training_view.examples,
        cell_dir,
        trainer_config,
        config.state_encoder_checkpoint,
        vocabulary=vocabulary,
        validation_examples=evaluation_view.examples,
        provenance=provenance,
        policy_target_semantics_id=POLICY_TARGET_SEMANTICS_OPENED_ONLY,
    )
    trainer.model.load_state_dict(dict(initial_state), strict=True)
    if model_state_sha256(trainer.model) != plan["initialization_sha256"]:
        raise ShadowAblationError(f"initialization drifted for {cell_id}")
    trainer.fit(checkpoint_every=1)
    checkpoint, sidecar = trainer.save_checkpoint(cell_dir / "final.pt")
    evaluation = evaluate_common_view(
        trainer.model,
        corpus.validation_examples,
        vocabulary,
        trainer.device,
        batch_size=config.batch_size,
        ece_bins=config.ece_bins,
    )
    evaluation.update(
        {
            "schema_id": "augmentia.v4-phase1-shadow-ablation-evaluation.v1",
            "contract_id": SHADOW_ABLATION_CONTRACT_ID,
            "metric_contract_id": PHASE1_AB_METRIC_CONTRACT_ID,
            "cell_id": cell_id,
            "lambda": float(lambda_value),
            "checkpoint_sha256": sha256_file(checkpoint),
            "sidecar_sha256": sha256_file(sidecar),
            "source_corpus_sha256": corpus.source_corpus_sha256,
            "split_sha256": corpus.split_sha256,
            "initialization_sha256": plan["initialization_sha256"],
            "promotable": False,
        }
    )
    evaluation = _seal_payload(evaluation, "evaluation_sha256")
    evaluation_path = _write_idempotent_sealed_json(
        cell_dir / "heldout-evaluation.json", evaluation, "evaluation_sha256"
    )
    result = _seal_payload(
        {
            "schema_id": SHADOW_ABLATION_RESULT_SCHEMA_ID,
            "contract_id": SHADOW_ABLATION_CONTRACT_ID,
            "status": "complete",
            "cell_id": cell_id,
            "lambda": float(lambda_value),
            "plan_sha256": plan["plan_sha256"],
            "initialization_sha256": plan["initialization_sha256"],
            "source_corpus_sha256": corpus.source_corpus_sha256,
            "split_sha256": corpus.split_sha256,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "sidecar_path": str(sidecar),
            "sidecar_sha256": sha256_file(sidecar),
            "evaluation_path": str(evaluation_path),
            "evaluation_file_sha256": sha256_file(evaluation_path),
            "evaluation_seal_sha256": evaluation["evaluation_sha256"],
            "history_path": str(cell_dir / "history.json"),
            "history_sha256": sha256_file(cell_dir / "history.json"),
            "metrics": _metrics(evaluation),
            "promotable": False,
        },
        "result_sha256",
    )
    _write_idempotent_sealed_json(result_path, result, "result_sha256")
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _summary(
    plan: Mapping[str, Any], results: Sequence[Mapping[str, Any]], *, evidence: bool
) -> dict[str, Any]:
    if 0.0 not in {float(item["lambda"]) for item in results}:
        raise ShadowAblationError("lambda=0 baseline is required")
    baseline = next(item for item in results if float(item["lambda"]) == 0.0)
    rows = []
    for result in results:
        metrics = result["metrics"]
        row = {
            "cell_id": result["cell_id"],
            "lambda": float(result["lambda"]),
            "metrics": metrics,
            "improvement_vs_lambda0": _improvement(baseline["metrics"], metrics),
        }
        rows.append(row)
    return _seal_payload(
        {
            "schema_id": SHADOW_ABLATION_SUMMARY_SCHEMA_ID,
            "contract_id": SHADOW_ABLATION_CONTRACT_ID,
            "metric_contract_id": PHASE1_AB_METRIC_CONTRACT_ID,
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "source_corpus_sha256": plan["source_corpus_sha256"],
            "split_sha256": plan["split_sha256"],
            "initialization_sha256": plan["initialization_sha256"],
            "baseline_lambda": 0.0,
            "rows": rows,
            "evidence_sufficient": evidence,
            "promotable": False,
            "promotion_note": "Diagnostic only; no shadow ablation cell is a promotion candidate.",
        },
        "summary_sha256",
    )


def run_shadow_ablation(
    manifest: str | Path,
    output_dir: str | Path,
    config: ShadowAblationConfig,
) -> dict[str, Any]:
    config.validate()
    manifest_path = Path(manifest).resolve()
    root = Path(output_dir).resolve()
    if "smoke" not in root.name.lower():
        raise ShadowAblationError("shadow output directory name must contain smoke")
    repo_root = Path(__file__).resolve().parents[2]
    commit, dirty = _git_identity(repo_root)
    if dirty and not config.allow_dirty_smoke:
        raise ShadowAblationError("dirty worktree requires allow_dirty_smoke=True")
    vocabulary, vocabulary_identity = load_runtime_vocabulary(config.vocabulary_path)
    corpus = load_controlled_corpus(manifest_path, _loader_config(config), vocabulary_identity)
    if corpus.dataset.run_kind != "production" and not config.allow_smoke_corpus:
        raise ShadowAblationError("non-production corpus requires allow_smoke_corpus=True")
    training_view = make_training_view(corpus.train_examples, "opened_only")
    evaluation_view = make_training_view(corpus.validation_examples, "opened_only")
    initial_state, initialization_sha256 = _capture_initial_state(
        corpus, training_view, config, vocabulary, root / ".init-probe"
    )
    plan = _build_plan(
        corpus,
        training_view,
        evaluation_view,
        config,
        initialization_sha256,
        vocabulary_identity,
        {"commit": commit, "dirty": dirty},
    )
    _write_idempotent_sealed_json(root / "shadow-plan.json", plan, "plan_sha256")
    summary_path = root / "shadow-summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        verify_payload_seal(summary, "summary_sha256")
        if summary.get("plan_sha256") != plan["plan_sha256"]:
            raise ShadowAblationError("existing summary references a different plan")
        del initial_state
        return summary
    results = [
        _run_cell(
            config=config,
            corpus=corpus,
            training_view=training_view,
            evaluation_view=evaluation_view,
            vocabulary=vocabulary,
            initial_state=initial_state,
            plan=plan,
            lambda_value=float(lambda_value),
            output_dir=root,
        )
        for lambda_value in config.lambdas
    ]
    summary = _summary(
        plan,
        results,
        evidence=len(corpus.validation_game_ids) >= config.minimum_game_blocks,
    )
    _write_idempotent_sealed_json(summary_path, summary, "summary_sha256")
    del initial_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--lambda", dest="lambdas", type=float, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=2026091501)
    parser.add_argument("--split-seed", type=int, default=2026091501)
    parser.add_argument("--initialization-seed", type=int, default=2026091500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--value-weight", type=float, default=0.0)
    parser.add_argument("--minimum-game-blocks", type=int, default=30)
    parser.add_argument("--validation-bp", type=int, default=2000)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--allow-smoke-corpus", action="store_true")
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = ShadowAblationConfig(
        lambdas=tuple(args.lambdas or DEFAULT_LAMBDAS),
        seed=args.seed,
        split_seed=args.split_seed,
        initialization_seed=args.initialization_seed,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        value_weight=args.value_weight,
        device=args.device,
        minimum_game_blocks=args.minimum_game_blocks,
        validation_basis_points=args.validation_bp,
        ece_bins=args.ece_bins,
        allow_smoke_corpus=args.allow_smoke_corpus,
        allow_dirty_smoke=args.allow_dirty_smoke,
    )
    result = run_shadow_ablation(args.manifest, args.output_dir, config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

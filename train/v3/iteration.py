"""Run one reproducible V3 self-play -> train -> export iteration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from .mixture import MIXTURE_SCHEMA_ID, load_training_dataset
from .model import V3ModelConfig, compute_v3_loss
from .onnx_export import assert_onnx_equivalent, export_onnx_v3
from .replay import collate_examples
from .retention import ITERATION_MARKER_SCHEMA_ID, apply_iteration_retention
from .selfplay_parallel import ParallelSelfPlayConfig, run_parallel_selfplay
from .tensorize import V3Vocabulary
from .trainer import TrainerV3, TrainingConfigV3, TrainingMetricsJournalV3

ITERATION_CONFIG_SCHEMA_ID = "augmentia.selfplay-iteration-config.v3"


def run_iteration(
    config_path: str | Path, iteration_root: str | Path, *, resume: bool = False
) -> Path:
    config_path = Path(config_path).resolve()
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    _validate_config(config)
    iteration_id = config["iteration_id"]
    root = Path(iteration_root).resolve()
    iteration_dir = _direct_child(root, iteration_id)
    manifest_path = iteration_dir / "iteration-manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"iteration {iteration_id} is already complete")
    iteration_dir.mkdir(parents=True, exist_ok=True)
    _write_once_or_match(iteration_dir / "input-config.json", raw_config, resume)
    repository = Path(__file__).parents[2]
    ruleset_hash = config["ruleset_manifest_sha256"]

    configured_selfplay = config["selfplay"].get("manifest")
    if configured_selfplay is not None:
        selfplay_manifest = _resolve(config_path, configured_selfplay)
    else:
        selfplay_dir = iteration_dir / "selfplay"
        selfplay_manifest = selfplay_dir / "manifest.json"
        if not selfplay_manifest.exists():
            _run_selfplay(
                repository,
                config_path,
                config,
                selfplay_dir,
                ruleset_hash,
                resume=resume,
            )
        elif not resume:
            raise FileExistsError(
                "partial self-play artifacts already exist; use --resume"
            )
    _run_checked(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "augment-cli",
            "--bin",
            "validate_v3_dataset",
            "--",
            "--manifest",
            str(selfplay_manifest),
        ],
        cwd=repository,
    )

    mixture_config = _build_mixture_config(config_path, config, selfplay_manifest)
    mixture_path = iteration_dir / "mixture.json"
    mixture_bytes = _json_bytes(mixture_config)
    _write_once_or_match(mixture_path, mixture_bytes, resume)
    dataset = load_training_dataset(mixture_path)
    parent_checkpoint = config["training"].get("parent_checkpoint")
    parent_metadata = config["training"].get("parent_metadata")
    vocabulary = _fine_tuning_vocabulary(
        dataset,
        None if parent_metadata is None else _resolve(config_path, parent_metadata),
    )
    training_config = _training_config(config["training"])
    model_config = V3ModelConfig(
        text_vocabulary_size=vocabulary.size,
        **config["training"].get("model", {}),
    )
    trainer = TrainerV3(
        dataset,
        vocabulary,
        model_config,
        training_config,
        iteration_dir / "training",
        run_id=iteration_id,
        parent_checkpoint=(
            None
            if parent_checkpoint is None
            else str(_resolve(config_path, parent_checkpoint))
        ),
        parent_metadata=(
            None if parent_metadata is None else str(_resolve(config_path, parent_metadata))
        ),
    )
    if resume and trainer.checkpoints.load_latest_valid() is not None:
        trainer.resume_latest()
    journal = TrainingMetricsJournalV3(
        iteration_dir / "training", trainer.run_id, trainer.run_identity
    )
    journal.truncate_after(trainer.global_step)
    metrics_interval = _positive_int(
        config["training"].get("metrics_interval", 10), "metrics_interval"
    )
    steps_executed = 0
    last_logged_step = None
    last_metrics = None
    for metrics in trainer.iter_train():
        steps_executed += 1
        last_metrics = metrics
        if metrics.step == 1 or metrics.step % metrics_interval == 0:
            journal.append(_training_metric_event(trainer, metrics))
            last_logged_step = metrics.step
    if last_metrics is not None and last_metrics.step != last_logged_step:
        journal.append(_training_metric_event(trainer, last_metrics))
    trainer.close()
    loaded = trainer.checkpoints.load_latest_valid()
    if loaded is None:
        checkpoint = trainer.save_checkpoint()
    else:
        checkpoint = loaded[0]

    candidate_dir = iteration_dir / "candidate"
    candidate_dir.mkdir(exist_ok=True)
    candidate = candidate_dir / f"{iteration_id}.onnx"
    metadata = candidate.with_suffix(".metadata.json")
    sample = _collate_dataset_indices(
        dataset, list(range(min(8, len(dataset)))), vocabulary
    ).inputs
    if candidate.exists() or metadata.exists():
        if not resume or not (candidate.exists() and metadata.exists()):
            raise FileExistsError("candidate artifact is incomplete or already exists")
    else:
        export_onnx_v3(
            trainer.model,
            sample,
            candidate,
            vocabulary,
            dataset.identity,
            checkpoint,
            trainer.code_commit,
            checkpoint_migrations=trainer.parent_migrations,
        )
    equivalence = assert_onnx_equivalent(trainer.model, sample, candidate)
    trainer.model.to(trainer.device)
    evaluation = _evaluate_training_slice(
        trainer, dataset, vocabulary, config["training"].get("evaluation_examples", 64)
    )

    lineage = {
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "parent_models": [
            _artifact_identity(config_path, model)
            for model in config["models"]
        ],
        "source_manifests": {
            name: _file_identity(_resolve(config_path, path))
            for name, path in config["mixture"]["sources"].items()
        },
        "selfplay_manifest": _file_identity(selfplay_manifest),
        "mixture_identity": dataset.identity.to_json(),
        "checkpoint": _file_identity(checkpoint),
        "training_metrics": _file_identity(journal.path),
        "candidate": _file_identity(candidate),
        "candidate_metadata": _file_identity(metadata),
        "parent_training": trainer.parent_identity,
    }
    manifest = {
        "iteration_schema_id": ITERATION_MARKER_SCHEMA_ID,
        "iteration_id": iteration_id,
        "created_at_utc": config["created_at_utc"],
        "status": "complete",
        "lineage": lineage,
        "training": {
            "global_step": trainer.global_step,
            "samples_seen": trainer.samples_seen,
            "last_metrics": trainer.last_metrics,
            "steps_executed_this_invocation": steps_executed,
            "metrics_interval": metrics_interval,
            "source_quotas": dataset.quotas,
            "source_weights": dataset.weights,
        },
        "evaluation": {**evaluation, **equivalence},
        "retention": None,
    }
    _atomic_json(manifest_path, manifest)
    retention = config["retention"]
    retention_result = apply_iteration_retention(
        root,
        keep_iterations=retention["keep_iterations"],
        max_bytes=retention["max_bytes"],
        protected=frozenset({iteration_id}),
    )
    manifest["retention"] = asdict(retention_result)
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _run_selfplay(
    repository: Path,
    config_path: Path,
    config: Mapping[str, Any],
    output: Path,
    ruleset_hash: str,
    *,
    resume: bool,
) -> None:
    _run_checked(
        [
            "cargo",
            "build",
            "-q",
            "--release",
            "-p",
            "augment-cli",
            "--bin",
            "self_play_v3",
            "--bin",
            "validate_v3_dataset",
        ],
        cwd=repository,
    )
    models = config["models"]
    selfplay = config["selfplay"]
    options = {
        "games": 32,
        "workers": 4,
        "simulations": 400,
        "leaf_batch_size": 16,
        "max_decisions": 512,
        "temperature_decisions": 30,
        "temperature": 1.0,
        "noise_alpha": 0.3,
        "noise_epsilon": 0.25,
        "train_bp": 9000,
        "validation_bp": 500,
        "holdout_bp": 500,
        "compression_level": 3,
        "execution_provider": "cpu",
        "intra_threads": 1,
    }
    values = {name: selfplay.get(name, default) for name, default in options.items()}
    executable_suffix = ".exe" if os.name == "nt" else ""
    ort_dylib = (
        Path(sys.prefix)
        / "Lib"
        / "site-packages"
        / "onnxruntime"
        / "capi"
        / ("onnxruntime.dll" if os.name == "nt" else "libonnxruntime.so")
    )
    run_parallel_selfplay(
        ParallelSelfPlayConfig(
            repository=repository,
            binary=repository
            / "target"
            / "release"
            / f"self_play_v3{executable_suffix}",
            validator=repository
            / "target"
            / "release"
            / f"validate_v3_dataset{executable_suffix}",
            output_dir=output,
            model_a=_resolve(config_path, models[0]["onnx"]),
            metadata_a=_resolve(config_path, models[0]["metadata"]),
            model_b=_resolve(config_path, models[1]["onnx"]),
            metadata_b=_resolve(config_path, models[1]["metadata"]),
            ruleset_hash=ruleset_hash,
            run_id=config["iteration_id"],
            created_at_utc=config["created_at_utc"],
            base_seed=config["seed"],
            ort_dylib=ort_dylib,
            **values,
        ),
        resume=resume,
    )


def _build_mixture_config(
    config_path: Path,
    config: Mapping[str, Any],
    selfplay_manifest: Path,
) -> dict[str, Any]:
    mixture = config["mixture"]
    sources = {
        name: str(_resolve(config_path, path))
        for name, path in mixture["sources"].items()
    }
    selfplay_source = config["selfplay"]["source"]
    if selfplay_source in sources:
        raise ValueError(f"generated self-play would overwrite source {selfplay_source}")
    sources[selfplay_source] = str(selfplay_manifest)
    return {
        "mixture_schema_id": MIXTURE_SCHEMA_ID,
        "profile": mixture["profile"],
        "seed": config["seed"],
        "epoch_size": mixture["epoch_size"],
        "sources": sources,
    }


def _training_config(payload: Mapping[str, Any]) -> TrainingConfigV3:
    names = {
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "total_steps",
        "checkpoint_interval",
        "checkpoint_keep",
        "value_weight",
        "amp",
        "device",
        "prefetch_batches",
    }
    values = {name: payload[name] for name in names if name in payload}
    return TrainingConfigV3(**values)


def _fine_tuning_vocabulary(
    dataset: Any, parent_metadata: Path | None
) -> V3Vocabulary:
    dataset_vocabulary = dataset.build_vocabulary()
    if parent_metadata is None:
        return dataset_vocabulary
    metadata = json.loads(parent_metadata.read_text(encoding="utf-8"))
    parent_vocabulary = V3Vocabulary.from_json(metadata["vocabulary"])
    parent_tokens = set(parent_vocabulary.tokens)
    appended = sorted(
        token for token in dataset_vocabulary.tokens if token not in parent_tokens
    )
    return V3Vocabulary((*parent_vocabulary.tokens, *appended))


def _evaluate_training_slice(
    trainer: TrainerV3,
    dataset: Any,
    vocabulary: V3Vocabulary,
    count: int,
) -> dict[str, Any]:
    count = min(_positive_int(count, "evaluation_examples"), len(dataset))
    batch = _collate_dataset_indices(dataset, list(range(count)), vocabulary).to(
        trainer.device
    )
    trainer.model.eval()
    with torch.inference_mode():
        output = trainer.model(batch.inputs)
        losses = compute_v3_loss(
            output,
            batch.inputs.action_mask,
            batch.policy_targets,
            batch.policy_supervision,
            batch.value_targets,
            batch.value_supervision,
            trainer.config.value_weight,
            batch.auxiliary_targets,
            batch.auxiliary_supervision,
        )
    return {
        "examples": count,
        "loss": float(losses.total),
        "policy_loss": float(losses.policy),
        "value_loss": float(losses.value),
        "auxiliary_loss": float(losses.auxiliary),
        "policy_samples": losses.policy_samples,
        "value_samples": losses.value_samples,
        "auxiliary_samples": losses.auxiliary_samples,
        "scope": "deterministic_training_slice_not_promotion_gate",
    }


def _collate_dataset_indices(
    dataset: Any, indices: list[int], vocabulary: V3Vocabulary
) -> Any:
    if hasattr(dataset, "collate_indices"):
        return dataset.collate_indices(indices)
    return collate_examples([dataset[index] for index in indices], vocabulary)


def _training_metric_event(trainer: TrainerV3, metrics: Any) -> dict[str, Any]:
    return {
        "kind": "train",
        "step": metrics.step,
        "samples_seen": trainer.samples_seen,
        "samples": metrics.samples,
        "decision_count": metrics.samples,
        "loss": metrics.loss,
        "policy_loss": metrics.policy_loss,
        "value_loss": metrics.value_loss,
        "auxiliary_loss": metrics.auxiliary_loss,
        "amp_overflow_retries": metrics.amp_overflow_retries,
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "amp_scale": trainer.scaler.get_scale(),
    }


def _validate_config(config: Any) -> None:
    required = {
        "iteration_config_schema_id",
        "iteration_id",
        "created_at_utc",
        "seed",
        "ruleset_manifest_sha256",
        "models",
        "selfplay",
        "mixture",
        "training",
        "retention",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("iteration config fields do not match the V3 contract")
    if config["iteration_config_schema_id"] != ITERATION_CONFIG_SCHEMA_ID:
        raise ValueError("unsupported iteration config schema")
    iteration_id = config["iteration_id"]
    if (
        not isinstance(iteration_id, str)
        or not iteration_id
        or Path(iteration_id).name != iteration_id
        or iteration_id in {".", ".."}
    ):
        raise ValueError("iteration_id must be one safe path component")
    if not config["created_at_utc"].endswith("Z"):
        raise ValueError("created_at_utc must end with Z")
    digest = config["ruleset_manifest_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("ruleset manifest hash must be SHA-256")
    if not isinstance(config["models"], list) or len(config["models"]) != 2:
        raise ValueError("exactly two seat-balanced model entries are required")
    for model in config["models"]:
        if not isinstance(model, dict) or set(model) != {"onnx", "metadata"}:
            raise ValueError("model entries require onnx and metadata paths")
    source = config["selfplay"].get("source")
    if source not in {"weak_selfplay", "champion_selfplay"}:
        raise ValueError("selfplay source must identify weak or champion data")
    if "manifest" in config["selfplay"] and not isinstance(
        config["selfplay"]["manifest"], str
    ):
        raise ValueError("selfplay manifest must be a path string")
    parent_checkpoint = config["training"].get("parent_checkpoint")
    parent_metadata = config["training"].get("parent_metadata")
    if (parent_checkpoint is None) != (parent_metadata is None):
        raise ValueError(
            "training parent_checkpoint and parent_metadata must be provided together"
        )
    retention = config["retention"]
    _positive_int(retention.get("keep_iterations"), "keep_iterations")
    _positive_int(retention.get("max_bytes"), "max_bytes")


def _artifact_identity(config_path: Path, model: Mapping[str, str]) -> dict[str, Any]:
    return {
        "onnx": _file_identity(_resolve(config_path, model["onnx"])),
        "metadata": _file_identity(_resolve(config_path, model["metadata"])),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _resolve(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def _direct_child(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    child = (root / name).resolve()
    if child.parent != root:
        raise ValueError("iteration path escaped its configured root")
    return child


def _run_checked(
    command: list[str], *, cwd: Path, rust_onnx: bool = False
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if rust_onnx:
        environment["ORT_DYLIB_PATH"] = str(
            Path(sys.prefix)
            / "Lib"
            / "site-packages"
            / "onnxruntime"
            / "capi"
            / "onnxruntime.dll"
        )
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command exited {completed.returncode}: {' '.join(command)}\n{completed.stderr}"
        )
    return completed


def _write_once_or_match(path: Path, content: bytes, resume: bool) -> None:
    if path.exists():
        if not resume or path.read_bytes() != content:
            raise RuntimeError(f"existing {path} differs from the requested iteration")
        return
    path.write_bytes(content)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_bytes(_json_bytes(payload))
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one V3 self-play iteration")
    parser.add_argument("config", type=Path)
    parser.add_argument("iteration_root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(run_iteration(args.config, args.iteration_root, resume=args.resume))


if __name__ == "__main__":
    main()

"""Validate a V4 checkpoint by reconstructing its exact saved split and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import load_dataset_input
from .replay import FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID, GAME_RECORD_V5_SCHEMA_ID
from .train import (
    V4Trainer,
    V4_TRAINER_SCHEMA_ID,
    _load_examples,
    assert_production_policy_target_semantics,
    _sha256_file,
    load_runtime_vocabulary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    dataset_path = args.manifest.resolve()
    sidecar_path = checkpoint.with_suffix(".v4-checkpoint.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("schema_id") != V4_TRAINER_SCHEMA_ID:
        raise ValueError("checkpoint sidecar schema mismatch")
    assert_production_policy_target_semantics(sidecar)
    if (
        sidecar.get("factorized_target_schema_id")
        != FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID
        or sidecar.get("replay_schema_id") != GAME_RECORD_V5_SCHEMA_ID
    ):
        raise ValueError("checkpoint sidecar replay/target lineage mismatch")
    if sidecar.get("checkpoint_sha256") != _sha256_file(checkpoint):
        raise ValueError("checkpoint hash mismatch")

    dataset = load_dataset_input(dataset_path)
    artifacts = {game.game_id: game.path for game in dataset.games}
    provenance = sidecar.get("provenance", {})
    if provenance.get("dataset_manifest_sha256") != dataset.sha256:
        raise ValueError("dataset input hash differs from checkpoint provenance")

    train_paths = [artifacts[game_id] for game_id in provenance["train_game_ids"]]
    validation_paths = [
        artifacts[game_id] for game_id in provenance["validation_game_ids"]
    ]
    max_train = provenance.get("max_train_examples")
    max_validation = provenance.get("max_validation_examples")
    train_examples = _load_examples(
        train_paths, None if max_train is None else int(max_train)
    )
    validation_examples = _load_examples(
        validation_paths, None if max_validation is None else int(max_validation)
    )

    vocabulary, _identity = load_runtime_vocabulary(
        provenance["runtime_vocabulary"]["path"]
    )
    trainer = V4Trainer.from_checkpoint(
        checkpoint,
        train_examples,
        validation_examples=validation_examples,
        expected_vocabulary=vocabulary,
        expected_provenance=provenance,
    )
    validation = trainer.evaluate()
    expected = sidecar["history"][-1]["validation"]
    metric_keys = ["loss", "policy_loss", "value_loss"]
    metric_keys.extend(
        key
        for key in ("policy_target_entropy", "policy_kl")
        if key in expected
    )
    errors = {
        key: abs(float(validation[key]) - float(expected[key]))
        for key in metric_keys
    }
    if max(errors.values(), default=0.0) > 1e-7:
        raise ValueError(f"reloaded validation metrics drifted: {errors}")

    report = {
        "schema_id": "augmentia.v4-checkpoint-validation.v1",
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "policy_target_semantics_id": sidecar["policy_target_semantics_id"],
        "policy_target_semantics_lineage": sidecar[
            "policy_target_semantics_lineage"
        ],
        "policy_target_semantics_production_eligible": True,
        "completed_epochs": trainer.completed_epochs,
        "vocabulary_equal": trainer.vocabulary.to_json() == sidecar["vocabulary"],
        "model_config_equal": trainer.model.config.__dict__ == sidecar["model_config"],
        "validation": validation,
        "expected_validation": expected,
        "validation_absolute_errors": errors,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

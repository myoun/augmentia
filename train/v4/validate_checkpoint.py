"""Validate a V4 checkpoint by reconstructing its exact saved split and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .replay import load_game_v4
from .train import (
    V4Trainer,
    V4_TRAINER_SCHEMA_ID,
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
    manifest_path = args.manifest.resolve()
    sidecar_path = checkpoint.with_suffix(".v4-checkpoint.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("schema_id") != V4_TRAINER_SCHEMA_ID:
        raise ValueError("checkpoint sidecar schema mismatch")
    if sidecar.get("checkpoint_sha256") != _sha256_file(checkpoint):
        raise ValueError("checkpoint hash mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {
        item["game_id"]: manifest_path.parent / item["relative_path"]
        for item in manifest["games_artifacts"]
    }
    provenance = sidecar.get("provenance", {})
    if provenance.get("dataset_manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("dataset manifest hash differs from checkpoint provenance")

    train_paths = [artifacts[game_id] for game_id in provenance["train_game_ids"]]
    validation_paths = [
        artifacts[game_id] for game_id in provenance["validation_game_ids"]
    ]
    train_examples = [
        example for path in train_paths for example in load_game_v4(path)
    ]
    validation_examples = [
        example for path in validation_paths for example in load_game_v4(path)
    ]
    max_train = provenance.get("max_train_examples")
    max_validation = provenance.get("max_validation_examples")
    if max_train is not None:
        train_examples = train_examples[: int(max_train)]
    if max_validation is not None:
        validation_examples = validation_examples[: int(max_validation)]

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
    errors = {
        key: abs(float(validation[key]) - float(expected[key]))
        for key in ("loss", "policy_loss", "value_loss")
    }
    if max(errors.values(), default=0.0) > 1e-7:
        raise ValueError(f"reloaded validation metrics drifted: {errors}")

    report = {
        "schema_id": "augmentia.v4-checkpoint-validation.v1",
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
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

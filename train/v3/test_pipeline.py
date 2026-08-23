from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
import zstandard

from v3.model import AugmentNetV3, V3ModelConfig
from v3.mixture import MIXTURE_SCHEMA_ID, ReplayMixtureV3, load_training_dataset
from v3.onnx_export import (
    assert_onnx_equivalent,
    export_onnx_v3,
    extend_model_for_runtime_vocabulary,
    migrate_piece_kind_embedding,
)
from v3.onnx_runtime import OnnxRuntimeV3
from v3.pretensor import PretensorizedReplayDatasetV3, build_pretensor_artifact
from v3.replay import DeterministicSamplerV3, ReplayDatasetV3, collate_examples
from v3.schema import SchemaError, load_golden_suite
from v3.tensorize import V3Vocabulary
from v3.trainer import (
    CheckpointMismatchError,
    EarlyStoppingV3,
    TrainerV3,
    TrainingConfigV3,
    TrainingMetricsJournalV3,
    evaluate_policy_v3,
)


class PipelineV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_replay_loader_and_target_collation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_dataset(Path(directory))
            dataset = ReplayDatasetV3(manifest)
            vocabulary = V3Vocabulary.from_cases(item.case for item in dataset)
            batch = collate_examples(list(dataset), vocabulary)
            self.assertEqual(len(dataset), 3)
            self.assertTrue(batch.policy_supervision.all())
            self.assertTrue(batch.value_supervision.all())
            self.assertEqual(int(batch.auxiliary_supervision.sum()), 1)
            torch.testing.assert_close(batch.policy_targets.sum(dim=1), torch.ones(3))

    def test_pretensor_artifact_matches_raw_collation_across_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            manifest = _write_dataset(root / "source")
            artifact = build_pretensor_artifact(
                manifest, root / "tensorized", chunk_size=2, workers=1
            )
            dataset = PretensorizedReplayDatasetV3(artifact, max_cached_frames=1)
            loaded = load_training_dataset(artifact)
            raw = ReplayDatasetV3(manifest)
            indices = [2, 0, 1]
            expected = collate_examples(
                [raw[index] for index in indices], raw.build_vocabulary()
            )
            actual = dataset.collate_indices(indices)
            self.assertIsInstance(loaded, PretensorizedReplayDatasetV3)
            self.assertEqual(dataset.sampler_group_sizes, (2, 1))
            self.assertEqual(expected.decision_ids, actual.decision_ids)
            for field in fields(expected.inputs):
                left = getattr(expected.inputs, field.name)
                right = getattr(actual.inputs, field.name)
                slices = tuple(slice(0, size) for size in left.shape)
                torch.testing.assert_close(left, right[slices], rtol=0, atol=0)
            torch.testing.assert_close(
                expected.policy_targets,
                actual.policy_targets[:, : expected.policy_targets.shape[1]],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(expected.value_targets, actual.value_targets)

    def test_full_game_records_expand_to_actor_relative_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_dataset(root)
            _rewrite_as_game(root, manifest)
            dataset = ReplayDatasetV3(manifest)
            self.assertEqual(len(dataset), 3)
            self.assertEqual([item.value for item in dataset], [1.0, -1.0, 1.0])

    def test_replay_loader_uses_one_shard_cache_and_shard_local_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_dataset(root)
            _split_decisions_across_two_shards(root, manifest)
            dataset = ReplayDatasetV3(manifest, max_cached_shards=1)
            self.assertFalse(dataset.index_cache_hit)
            cached_dataset = ReplayDatasetV3(manifest, max_cached_shards=1)
            self.assertTrue(cached_dataset.index_cache_hit)
            self.assertEqual(
                cached_dataset.vocabulary_tokens, dataset.vocabulary_tokens
            )
            index_payload = json.loads(dataset.index_cache_path.read_text())
            self.assertIn("code_version", index_payload)
            self.assertIn("schema_versions", index_payload)
            index_payload["verifier_version"] = "stale"
            dataset.index_cache_path.write_text(
                json.dumps(index_payload), encoding="utf-8"
            )
            rebuilt_dataset = ReplayDatasetV3(manifest, max_cached_shards=1)
            self.assertFalse(rebuilt_dataset.index_cache_hit)
            self.assertEqual(dataset.sampler_group_sizes, (2, 1))
            self.assertEqual(dataset.cache_info()["resident_shards"], 0)
            collected_vocabulary = dataset.build_vocabulary()
            expected_vocabulary = V3Vocabulary.from_cases(item.case for item in dataset)
            self.assertEqual(collected_vocabulary, expected_vocabulary)
            dataset.clear_cache()
            baseline = dataset.cache_info()

            first = dataset[0]
            self.assertEqual(dataset.cache_info()["misses"], baseline["misses"] + 1)
            self.assertEqual(dataset[1].decision_id, "decision-1")
            self.assertEqual(dataset.cache_info()["hits"], baseline["hits"] + 1)
            self.assertEqual(dataset[2].decision_id, "decision-2")
            self.assertEqual(dataset.cache_info()["resident_shards"], 1)
            self.assertEqual(dataset[0].decision_id, first.decision_id)
            self.assertEqual(dataset.cache_info()["misses"], baseline["misses"] + 3)

            sampler = DeterministicSamplerV3(
                len(dataset), seed=71, group_sizes=dataset.sampler_group_sizes
            )
            order = sampler.take(len(dataset))
            positions = {index: offset for offset, index in enumerate(order)}
            self.assertEqual(abs(positions[0] - positions[1]), 1)
            state = sampler.state_dict()
            restored = DeterministicSamplerV3(
                len(dataset), seed=71, group_sizes=dataset.sampler_group_sizes
            )
            restored.load_state_dict(state)
            self.assertEqual(restored.state_dict(), state)

    def test_bootstrap_mixture_has_exact_reproducible_source_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_dataset(root)
            config = root / "mixture.json"
            config.write_text(
                json.dumps(
                    {
                        "mixture_schema_id": MIXTURE_SCHEMA_ID,
                        "profile": "bootstrap",
                        "seed": 41,
                        "epoch_size": 20,
                        "sources": {
                            "lichess": str(manifest),
                            "solver": str(manifest),
                            "weak_selfplay": str(manifest),
                        },
                    }
                ),
                encoding="utf-8",
            )
            left = load_training_dataset(config)
            right = ReplayMixtureV3.from_config(config)
            self.assertEqual(
                left.quotas, {"lichess": 10, "solver": 6, "weak_selfplay": 4}
            )
            self.assertEqual(
                [item.decision_id for item in left],
                [item.decision_id for item in right],
            )
            self.assertEqual(left.identity.to_json(), right.identity.to_json())

    def test_mixture_keeps_virtual_references_instead_of_materializing_examples(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_dataset(root)
            _split_decisions_across_two_shards(root, manifest)
            datasets = {
                name: ReplayDatasetV3(manifest)
                for name in ("lichess", "solver", "weak_selfplay")
            }
            mixture = ReplayMixtureV3(
                datasets,
                {"lichess": 0.5, "solver": 0.3, "weak_selfplay": 0.2},
                epoch_size=20,
                seed=43,
                profile="bootstrap",
            )
            self.assertEqual(len(mixture.entries), 20)
            self.assertTrue(
                all(
                    not item.cache_info()["resident_shards"]
                    for item in datasets.values()
                )
            )
            self.assertEqual(sum(mixture.sampler_group_sizes), len(mixture))
            self.assertTrue(
                mixture[0].decision_id.startswith(
                    ("lichess:", "solver:", "weak_selfplay:")
                )
            )
            self.assertEqual(
                sum(item.cache_info()["resident_shards"] for item in datasets.values()),
                1,
            )

    def test_mixture_resume_preserves_the_exact_cross_source_batch_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_dataset(root)
            datasets = {
                name: ReplayDatasetV3(manifest)
                for name in ("lichess", "solver", "weak_selfplay")
            }
            mixture = ReplayMixtureV3(
                datasets,
                {"lichess": 0.5, "solver": 0.3, "weak_selfplay": 0.2},
                epoch_size=20,
                seed=53,
                profile="bootstrap",
            )
            vocabulary = V3Vocabulary.from_cases(item.case for item in mixture)
            config = _training_config()
            model_config = _model_config(vocabulary)
            continuous = TrainerV3(
                mixture, vocabulary, model_config, config, root / "mix-continuous"
            )
            expected = continuous.train(steps=3)
            interrupted = TrainerV3(
                mixture, vocabulary, model_config, config, root / "mix-resumed"
            )
            actual = interrupted.train(steps=1)
            interrupted.save_checkpoint()
            resumed = TrainerV3(
                mixture, vocabulary, model_config, config, root / "mix-resumed"
            )
            resumed.resume_latest()
            actual.extend(resumed.train(steps=2))
            self.assertEqual(
                [item.decision_ids for item in expected],
                [item.decision_ids for item in actual],
            )
            np.testing.assert_allclose(
                [item.loss for item in expected],
                [item.loss for item in actual],
                rtol=0,
                atol=0,
            )

    def test_invalid_policy_is_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_dataset(root)
            _rewrite_first_probability(root, float("nan"))
            _refresh_manifest_hashes(manifest)
            with self.assertRaisesRegex(SchemaError, "finite"):
                ReplayDatasetV3(manifest)

    def test_uninterrupted_and_resumed_training_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = ReplayDatasetV3(_write_dataset(root))
            vocabulary = V3Vocabulary.from_cases(item.case for item in dataset)
            config = _training_config()
            model_config = _model_config(vocabulary)

            continuous = TrainerV3(
                dataset, vocabulary, model_config, config, root / "continuous"
            )
            continuous_history = continuous.train(steps=4)

            interrupted = TrainerV3(
                dataset, vocabulary, model_config, config, root / "resumed"
            )
            first_history = interrupted.train(steps=2)
            interrupted.save_checkpoint()
            resumed = TrainerV3(
                dataset, vocabulary, model_config, config, root / "resumed"
            )
            resumed.resume_latest()
            second_history = resumed.train(steps=2)

            combined = first_history + second_history
            self.assertEqual(
                [item.decision_ids for item in continuous_history],
                [item.decision_ids for item in combined],
            )
            np.testing.assert_allclose(
                [item.loss for item in continuous_history],
                [item.loss for item in combined],
                rtol=0,
                atol=0,
            )
            _assert_nested_equal(
                self, continuous.model.state_dict(), resumed.model.state_dict()
            )
            _assert_nested_equal(
                self, continuous.optimizer.state_dict(), resumed.optimizer.state_dict()
            )

    def test_corrupt_latest_checkpoint_is_ignored_and_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = ReplayDatasetV3(_write_dataset(root))
            vocabulary = V3Vocabulary.from_cases(item.case for item in dataset)
            config = _training_config()
            model_config = _model_config(vocabulary)
            trainer = TrainerV3(dataset, vocabulary, model_config, config, root / "run")
            trainer.train_step()
            first = trainer.save_checkpoint()
            trainer.train_step()
            latest = trainer.save_checkpoint()
            latest.write_bytes(b"partial checkpoint")

            recovered = TrainerV3(
                dataset, vocabulary, model_config, config, root / "run"
            )
            self.assertEqual(recovered.resume_latest(), first)
            self.assertEqual(recovered.global_step, 1)

            changed = TrainingConfigV3(
                **{**config.__dict__, "learning_rate": config.learning_rate * 2}
            )
            mismatch = TrainerV3(
                dataset, vocabulary, model_config, changed, root / "run"
            )
            with self.assertRaises(CheckpointMismatchError):
                mismatch.resume_latest()

    def test_validation_history_best_checkpoint_and_early_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = ReplayDatasetV3(_write_dataset(root))
            vocabulary = dataset.build_vocabulary()
            config = TrainingConfigV3(
                **{**_training_config().__dict__, "checkpoint_keep": 1}
            )
            trainer = TrainerV3(
                dataset,
                vocabulary,
                _model_config(vocabulary),
                config,
                root / "run",
            )
            trainer.train_step()
            rotating_first = trainer.save_checkpoint()
            metrics = evaluate_policy_v3(
                trainer.model,
                dataset,
                vocabulary,
                trainer.device,
                batch_size=2,
                step=trainer.global_step,
            )
            self.assertTrue(trainer.model.training)
            self.assertEqual(metrics.examples, len(dataset))
            self.assertTrue(math.isfinite(metrics.cross_entropy))
            self.assertGreaterEqual(metrics.top1_accuracy, 0.0)
            self.assertLessEqual(metrics.top1_accuracy, 1.0)

            event = {"kind": "validation", **asdict(metrics), "is_best": True}
            journal = TrainingMetricsJournalV3(
                root / "run", trainer.run_id, trainer.run_identity
            )
            journal.append(event)
            trainer.save_best_checkpoint(event)
            reloaded = TrainingMetricsJournalV3(
                root / "run", trainer.run_id, trainer.run_identity
            )
            self.assertEqual(reloaded.event_at("validation", 1), event)

            trainer.train_step()
            rotating_latest = trainer.save_checkpoint()
            self.assertFalse(rotating_first.exists())
            self.assertTrue(rotating_latest.exists())
            best = root / "run" / "checkpoints" / "best-validation.pt"
            self.assertTrue(best.exists())
            best_payload = torch.load(best, map_location="cpu", weights_only=False)
            self.assertEqual(best_payload["global_step"], 1)
            best_metadata = json.loads(
                (root / "run" / "checkpoints" / "best-validation.json").read_text()
            )
            self.assertEqual(best_metadata["global_step"], 1)
            self.assertEqual(
                best_metadata["validation"]["cross_entropy"], metrics.cross_entropy
            )

            stopper = EarlyStoppingV3(patience=2)
            self.assertEqual(stopper.observe(2.0), (True, False))
            self.assertEqual(stopper.observe(2.1), (False, False))
            self.assertEqual(stopper.observe(2.2), (False, True))

    def test_onnx_export_runtime_batch_and_schema_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = ReplayDatasetV3(_write_dataset(root))
            vocabulary = V3Vocabulary.from_cases(item.case for item in dataset)
            trainer = TrainerV3(
                dataset,
                vocabulary,
                _model_config(vocabulary),
                _training_config(),
                root / "run",
            )
            sample = collate_examples(list(dataset), vocabulary).inputs
            model_path, metadata_path = export_onnx_v3(
                trainer.model,
                sample,
                root / "model.onnx",
                vocabulary,
                dataset.identity,
            )
            errors = assert_onnx_equivalent(trainer.model, sample, model_path)
            self.assertLess(errors["policy_max_abs_error"], 2e-4)
            runtime = OnnxRuntimeV3(
                model_path,
                expected_ruleset_id=dataset.identity.ruleset_id,
            )
            evaluations = runtime.evaluate_cases([dataset[0].case, dataset[1].case])
            self.assertEqual(len(evaluations), 2)
            self.assertEqual(runtime.session_runs, 1)
            for result in evaluations:
                self.assertAlmostEqual(sum(result.probabilities), 1.0)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["observation_schema_id"] = "stale"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "observation schema"):
                OnnxRuntimeV3(model_path, metadata_path)

    def test_runtime_vocabulary_extension_preserves_trained_rows(self) -> None:
        vocabulary = V3Vocabulary(("<none>", "standard"))
        model = AugmentNetV3(_model_config(vocabulary))
        original = model.text_embedding.weight.detach().clone()
        extended, extended_vocabulary, appended = extend_model_for_runtime_vocabulary(
            model, vocabulary
        )
        self.assertTrue(appended)
        self.assertEqual(extended_vocabulary.tokens[:2], vocabulary.tokens)
        self.assertIn("square", appended)
        self.assertIn("square", extended_vocabulary.tokens)
        torch.testing.assert_close(extended.text_embedding.weight[:2], original)
        torch.testing.assert_close(
            extended.text_embedding.weight[2:],
            torch.zeros_like(extended.text_embedding.weight[2:]),
        )

    def test_legacy_piece_kind_embedding_expands_without_changing_trained_rows(self) -> None:
        legacy = torch.arange(52 * 4, dtype=torch.float32).reshape(52, 4)
        state, migration = migrate_piece_kind_embedding(
            {"piece_kind_embedding.weight": legacy}, 59
        )
        self.assertEqual(migration["source_rows"], 52)
        self.assertEqual(migration["target_rows"], 59)
        torch.testing.assert_close(state["piece_kind_embedding.weight"][:52], legacy)
        torch.testing.assert_close(
            state["piece_kind_embedding.weight"][52:],
            torch.zeros(7, 4),
        )


def _model_config(vocabulary: V3Vocabulary) -> V3ModelConfig:
    return V3ModelConfig(
        text_vocabulary_size=vocabulary.size,
        channels=8,
        residual_blocks=1,
        model_dim=16,
        runtime_message_passes=1,
    )


def _training_config() -> TrainingConfigV3:
    return TrainingConfigV3(
        seed=17,
        batch_size=2,
        learning_rate=1e-3,
        total_steps=8,
        checkpoint_interval=100,
        checkpoint_keep=4,
        amp=False,
        device="cpu",
    )


def _write_dataset(root: Path) -> Path:
    suite = load_golden_suite()
    cases = [case for case in suite["cases"] if case["legal_actions"]["actions"]][:3]
    records = []
    ruleset_hash = "a" * 64
    for index, case in enumerate(cases):
        action_count = len(case["legal_actions"]["actions"])
        if index == 0:
            policy_target = {
                "kind": "behavior_distribution",
                "probabilities": [1 / action_count] * action_count,
                "source": "test",
            }
        elif index == 1:
            policy_target = {
                "kind": "mcts_visits",
                "visit_counts": [1] * action_count,
                "completed_simulations": action_count,
                "root_value": 0.0,
            }
        else:
            policy_target = {
                "kind": "optimal_action_set",
                "action_indices": [0],
                "proof_source": "test",
            }
        records.append(
            {
                "record_schema_id": "augmentia.training.decision.v3",
                "decision_id": f"decision-{index}",
                "provenance": {
                    "source_kind": "manual_scenario",
                    "source_domain": "augment_chess",
                    "source_id": f"source-{index}",
                    "split_group_id": f"group-{index}",
                    "generator_commit": None,
                    "license": None,
                },
                "ruleset": {
                    "id": suite["ruleset_id"],
                    "manifest_sha256": ruleset_hash,
                },
                "observation_schema_id": suite["observation_schema_id"],
                "action_schema_id": suite["action_schema_id"],
                "actor": "white",
                "state_fingerprint": f"{index:016x}",
                "observation": case["observation"],
                "legal_actions": case["legal_actions"],
                "targets": {
                    "policy": {"scope": "augment_policy", "target": policy_target},
                    "value": {
                        "kind": "proven",
                        "value": float(index - 1),
                        "proof_source": "test",
                    },
                    "auxiliary": (
                        [
                            {
                                "name": "forced_win_distance_decisions",
                                "value": {"type": "integer", "value": 3},
                            }
                        ]
                        if index == 2
                        else []
                    ),
                },
            }
        )
    shard = root / "train.jsonl.zst"
    raw = (
        b"\n".join(
            json.dumps(record, separators=(",", ":"), allow_nan=True).encode()
            for record in records
        )
        + b"\n"
    )
    shard.write_bytes(zstandard.ZstdCompressor(level=1).compress(raw))
    manifest = root / "manifest.json"
    payload = {
        "manifest_schema_id": "augmentia.dataset.manifest.v3",
        "dataset_id": "test-dataset",
        "created_at_utc": "2026-08-21T00:00:00Z",
        "observation_schema_id": suite["observation_schema_id"],
        "action_schema_id": suite["action_schema_id"],
        "split_policy": {
            "algorithm": "sha256_split_group_v1",
            "salt": "test",
            "train_basis_points": 10000,
            "validation_basis_points": 0,
            "holdout_basis_points": 0,
        },
        "compression": {"codec": "zstd", "level": 1},
        "shards": [
            {
                "path": shard.name,
                "split": "train",
                "record_schema_id": "augmentia.training.decision.v3",
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "compressed_bytes": shard.stat().st_size,
                "records": len(records),
                "games": 0,
                "decisions": len(records),
                "game_steps": 0,
                "policy_eligible_decisions": len(records),
                "value_eligible_decisions": len(records),
            }
        ],
        "totals": {
            "shards": 1,
            "records": len(records),
            "games": 0,
            "decisions": len(records),
            "game_steps": 0,
            "policy_eligible_decisions": len(records),
            "value_eligible_decisions": len(records),
        },
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _rewrite_first_probability(root: Path, probability: float) -> None:
    shard = root / "train.jsonl.zst"
    raw = zstandard.ZstdDecompressor().decompress(shard.read_bytes())
    records = [json.loads(line) for line in raw.splitlines()]
    probabilities = records[0]["targets"]["policy"]["target"]["probabilities"]
    probabilities[0] = probability
    encoded = (
        b"\n".join(
            json.dumps(record, separators=(",", ":"), allow_nan=True).encode()
            for record in records
        )
        + b"\n"
    )
    shard.write_bytes(zstandard.ZstdCompressor(level=1).compress(encoded))


def _refresh_manifest_hashes(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    shard = manifest.parent / payload["shards"][0]["path"]
    payload["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    payload["shards"][0]["compressed_bytes"] = shard.stat().st_size
    manifest.write_text(json.dumps(payload), encoding="utf-8")


def _split_decisions_across_two_shards(root: Path, manifest: Path) -> None:
    original = root / "train.jsonl.zst"
    records = (
        zstandard.ZstdDecompressor().decompress(original.read_bytes()).splitlines()
    )
    shards = []
    for shard_index, shard_records in enumerate((records[:2], records[2:])):
        path = root / f"train-{shard_index}.jsonl.zst"
        raw = b"\n".join(shard_records) + b"\n"
        path.write_bytes(zstandard.ZstdCompressor(level=1).compress(raw))
        shards.append(
            {
                "path": path.name,
                "split": "train",
                "record_schema_id": "augmentia.training.decision.v3",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "compressed_bytes": path.stat().st_size,
                "records": len(shard_records),
                "games": 0,
                "decisions": len(shard_records),
                "game_steps": 0,
                "policy_eligible_decisions": len(shard_records),
                "value_eligible_decisions": len(shard_records),
            }
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["shards"] = shards
    payload["totals"]["shards"] = len(shards)
    manifest.write_text(json.dumps(payload), encoding="utf-8")


def _rewrite_as_game(root: Path, manifest: Path) -> None:
    shard = root / "train.jsonl.zst"
    records = [
        json.loads(line)
        for line in zstandard.ZstdDecompressor()
        .decompress(shard.read_bytes())
        .splitlines()
    ]
    steps = []
    for index, record in enumerate(records):
        action_count = len(record["legal_actions"]["actions"])
        steps.append(
            {
                "decision_index": index,
                "engine_ply": index,
                "actor": "white" if index % 2 == 0 else "black",
                "state_fingerprint": f"{index:016x}",
                "observation": record["observation"],
                "legal_actions": record["legal_actions"],
                "policy_target": {
                    "kind": "behavior_distribution",
                    "probabilities": [1 / action_count] * action_count,
                    "source": "test",
                },
                "played_action_index": 0,
                "next_state_fingerprint": f"{index + 1:016x}",
            }
        )
    game = {
        "record_schema_id": "augmentia.replay.game.v3",
        "game_id": "game-0",
        "ruleset": records[0]["ruleset"],
        "observation_schema_id": records[0]["observation_schema_id"],
        "action_schema_id": records[0]["action_schema_id"],
        "steps": steps,
        "termination": {
            "kind": "engine_terminal",
            "result": "white_win",
            "final_state_fingerprint": f"{len(steps):016x}",
        },
    }
    encoded = json.dumps(game, separators=(",", ":")).encode() + b"\n"
    shard.write_bytes(zstandard.ZstdCompressor(level=1).compress(encoded))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["shards"][0].update(
        {
            "record_schema_id": "augmentia.replay.game.v3",
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            "compressed_bytes": shard.stat().st_size,
            "records": 1,
            "games": 1,
            "decisions": 0,
            "game_steps": len(steps),
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")


def _assert_nested_equal(test: unittest.TestCase, left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(test, left_item, right_item)
    else:
        test.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()

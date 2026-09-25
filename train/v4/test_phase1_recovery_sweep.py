from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from v3.schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID
from v3.tensorize import V3Vocabulary

from v4.phase1_ab import (
    Phase1ABError,
    _seal_payload,
    make_training_view,
    model_state_sha256,
    verify_payload_seal,
)
from v4.phase1_recovery_sweep import (
    RECOVERY_SWEEP_PLAN_SCHEMA_ID,
    RECOVERY_SWEEP_SUMMARY_SCHEMA_ID,
    RecoveryConfig,
    RecoverySweepConfig,
    RecoverySweepError,
    _capture_initial_state,
    _load_or_run_result,
    _parser,
    build_sweep_plan,
    build_sweep_summary,
    learning_rate_at_epoch,
    run_bounded_training,
    run_id_for,
    run_one,
    run_sweep,
    update_early_stop,
    validate_update_budget,
    verify_source_unchanged,
)
from v4.replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    GAME_RECORD_V5_SCHEMA_ID,
    RULESET_ID_V4,
    make_factorized_target_v4,
    parse_game_v4,
)
from v4.train import V4Trainer, V4TrainingConfig

FIXTURE = (
    Path(__file__).parents[1]
    / "v3"
    / "fixtures"
    / "0915"
    / "observation_action_v3.json"
)


class RecoverySweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.case = next(
            case for case in payload["cases"] if case["name"] == "runtime_rich"
        )
        cls.actions = cls.case["legal_actions"]["actions"]
        cls.vocabulary = V3Vocabulary.from_cases([cls.case])

    def _example(self):
        visits = [1] * 15 + [0] * (len(self.actions) - 15)
        opened = [True] * 16 + [False] * (len(self.actions) - 16)
        target = make_factorized_target_v4(
            self.actions,
            visits,
            0.125,
            opened,
            search_semantics_id="augmentia.mcts-v4.factorized-gumbel-widening-public-chance.v2",
        )
        record = {
            "record_schema_id": GAME_RECORD_V5_SCHEMA_ID,
            "game_id": "recovery-sweep-regression",
            "ruleset": {"id": RULESET_ID_V4},
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "factorized_policy_schema_id": "augmentia.factorized-policy.v4.1",
            "factorized_target_schema_id": FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
            "steps": [
                {
                    "decision_index": 0,
                    "observation": self.case["observation"],
                    "legal_actions": self.case["legal_actions"],
                    "policy_target": target,
                    "played_action_index": 0,
                    "actor": "white",
                }
            ],
        }
        return parse_game_v4(record)[0]

    @staticmethod
    def _config(**overrides) -> RecoverySweepConfig:
        trainer = V4TrainingConfig(
            learning_rate=3e-4,
            batch_size=2,
            epochs=1,
            value_weight=0.0,
            seed=99,
            device="cpu",
            model_dim=16,
            channels=8,
            residual_blocks=1,
            runtime_message_passes=1,
            encoder_kind="baseline",
            tensor_cache_max_batches=0,
        )
        values = {
            "trainer": trainer,
            "seeds": (1, 2, 3, 4),
            "learning_rates": (3e-4, 1e-3),
            "fixed_updates": 9,
            "split_seed": 11,
            "initialization_seed": 10,
            "minimum_game_blocks": 8,
        }
        values.update(overrides)
        return RecoverySweepConfig(**values).validate()

    class _Corpus:
        train_game_ids = ("train-0", "train-1")
        validation_game_ids = ("validation-0",)
        source_corpus_sha256 = "a" * 64
        split_sha256 = "b" * 64
        provenance = {"source": "test"}

        @staticmethod
        def to_json():
            return {
                "dataset_id": "recovery-test",
                "source_corpus_sha256": "a" * 64,
                "run_kind": "smoke",
                "production_eligible": False,
            }

    def test_public_api_and_config_fail_closed(self) -> None:
        self.assertIs(RecoveryConfig, RecoverySweepConfig)
        self.assertTrue(callable(run_one))
        self.assertTrue(callable(run_sweep))
        self._config()
        with self.assertRaisesRegex(RecoverySweepError, "at least 4 seeds"):
            self._config(seeds=(1, 2, 3))
        with self.assertRaisesRegex(RecoverySweepError, "unique"):
            self._config(seeds=(1, 1, 2, 3))
        with self.assertRaisesRegex(RecoverySweepError, "must include learning rate"):
            self._config(learning_rates=(5e-4, 1e-3))
        with self.assertRaisesRegex(RecoverySweepError, "baseline"):
            hybrid = V4TrainingConfig(
                learning_rate=3e-4,
                epochs=1,
                encoder_kind="hybrid",
                model_dim=16,
                channels=8,
                residual_blocks=1,
                runtime_message_passes=1,
            )
            self._config(trainer=hybrid)
        with self.assertRaisesRegex(RecoverySweepError, "patience"):
            self._config(early_stopping_patience=0)
        with self.assertRaisesRegex(RecoverySweepError, "early_stopping_metric"):
            self._config(early_stopping_metric="not-a-metric")

    def test_fixed_budget_and_learning_rate_schedule(self) -> None:
        self.assertEqual(validate_update_budget(42, 14), 3)
        with self.assertRaisesRegex(RecoverySweepError, "exact multiple"):
            validate_update_budget(43, 14)
        self.assertEqual(
            learning_rate_at_epoch(
                1e-3, schedule="none", min_ratio=0.1, epoch_index=4, total_epochs=5
            ),
            1e-3,
        )
        values = [
            learning_rate_at_epoch(
                1e-3,
                schedule="cosine",
                min_ratio=0.1,
                epoch_index=index,
                total_epochs=3,
            )
            for index in range(3)
        ]
        self.assertAlmostEqual(values[0], 1e-3)
        self.assertAlmostEqual(values[1], 5.5e-4)
        self.assertAlmostEqual(values[2], 1e-4)
        self.assertEqual(run_id_for(7, 3e-4), run_id_for(7, 3e-4))
        self.assertNotEqual(run_id_for(7, 3e-4), run_id_for(8, 3e-4))

    def test_early_stop_state_tracks_best_epoch_and_patience(self) -> None:
        first = update_early_stop(None, epoch=1, metric=1.0, patience=2, min_delta=0.0)
        better = update_early_stop(
            first, epoch=2, metric=0.8, patience=2, min_delta=0.0
        )
        unchanged = update_early_stop(
            better, epoch=3, metric=0.8, patience=2, min_delta=0.0
        )
        stopped = update_early_stop(
            unchanged, epoch=4, metric=0.9, patience=2, min_delta=0.0
        )
        self.assertEqual(stopped.best_epoch, 2)
        self.assertEqual(stopped.best_metric, 0.8)
        self.assertEqual(stopped.non_improving_epochs, 2)
        self.assertTrue(stopped.stopped_early)
        no_early = update_early_stop(
            first, epoch=2, metric=2.0, patience=None, min_delta=0.0
        )
        self.assertEqual(no_early.best_epoch, 1)
        self.assertEqual(no_early.best_metric, 1.0)
        delta = update_early_stop(
            first, epoch=2, metric=0.97, patience=1, min_delta=0.05
        )
        self.assertEqual(delta.best_epoch, 1)
        self.assertTrue(delta.stopped_early)

    def test_canonical_initial_state_is_reused_independent_of_run_seed(self) -> None:
        example = self._example()
        view = make_training_view([example], "opened_only")
        config = self._config(seeds=(1, 2, 3, 4), initialization_seed=123)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state_a, hash_a = _capture_initial_state(
                SimpleNamespace(), view, config, self.vocabulary, root / "probe-a"
            )
            state_b, hash_b = _capture_initial_state(
                SimpleNamespace(), view, config, self.vocabulary, root / "probe-b"
            )
            self.assertEqual(hash_a, hash_b)
            for name in state_a:
                self.assertTrue(torch.equal(state_a[name], state_b[name]))
            changed = replace(config, initialization_seed=124)
            _state_c, hash_c = _capture_initial_state(
                SimpleNamespace(), view, changed, self.vocabulary, root / "probe-c"
            )
            self.assertNotEqual(hash_a, hash_c)
            self.assertEqual(
                hash_a,
                model_state_sha256(
                    V4Trainer(
                        [example],
                        root / "probe-d",
                        replace(config.trainer, seed=123, epochs=1),
                        vocabulary=self.vocabulary,
                    ).model
                ),
            )

    def test_bounded_training_executes_exact_updates_and_epoch_checkpoints(
        self,
    ) -> None:
        example = self._example()
        trainer_config = V4TrainingConfig(
            learning_rate=1e-3,
            batch_size=1,
            epochs=1,
            value_weight=0.0,
            seed=17,
            device="cpu",
            model_dim=16,
            channels=8,
            residual_blocks=1,
            runtime_message_passes=1,
            tensor_cache_max_batches=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            trainer = V4Trainer(
                [example],
                root,
                trainer_config,
                vocabulary=self.vocabulary,
                validation_examples=[example],
            )
            result = run_bounded_training(
                trainer,
                fixed_updates=3,
                planned_epochs=3,
                base_learning_rate=1e-3,
                lr_schedule="cosine",
                lr_min_ratio=0.1,
                early_stopping_patience=None,
                early_stopping_min_delta=0.0,
                early_stopping_metric="validation_policy_kl",
            )
            self.assertEqual(result["updates_executed"], 3)
            self.assertEqual(result["epochs_executed"], 3)
            self.assertFalse(result["early_stopped"])
            for epoch in range(1, 4):
                self.assertTrue((root / f"checkpoint-epoch-{epoch:04d}.pt").is_file())
                self.assertTrue(
                    (
                        root / f"checkpoint-epoch-{epoch:04d}.v4-checkpoint.json"
                    ).is_file()
                )
            self.assertAlmostEqual(trainer.history[0]["learning_rate"], 1e-3)
            self.assertAlmostEqual(trainer.history[1]["learning_rate"], 5.5e-4)
            self.assertAlmostEqual(trainer.history[2]["learning_rate"], 1e-4)
            selected = Path(result["selected_checkpoint_path"])
            self.assertTrue(selected.is_file())
            self.assertTrue(Path(result["history_path"]).is_file())

    def test_plan_is_sealed_covers_matrix_and_detects_tampering(self) -> None:
        config = self._config()
        corpus = self._Corpus()
        train_view = SimpleNamespace(
            examples=(object(), object(), object(), object(), object(), object()),
            view_id="train-view",
            view_sha256="c" * 64,
            to_json=lambda: {
                "training_view_id": "train-view",
                "training_view_sha256": "c" * 64,
            },
        )
        eval_view = SimpleNamespace(
            examples=(object(), object(), object()),
            view_id="eval-view",
            view_sha256="d" * 64,
            to_json=lambda: {
                "training_view_id": "eval-view",
                "training_view_sha256": "d" * 64,
            },
        )
        plan = build_sweep_plan(
            corpus=corpus,
            training_view=train_view,
            evaluation_view=eval_view,
            config=config,
            batches_per_epoch=3,
            initialization_sha256="e" * 64,
            vocabulary_identity={"sha256": "f" * 64},
            git_identity={"commit": "1" * 40, "dirty": False},
        )
        self.assertEqual(plan["schema_id"], RECOVERY_SWEEP_PLAN_SCHEMA_ID)
        self.assertEqual(plan["run_count"], 8)
        self.assertEqual(len(plan["combinations"]), 8)
        self.assertEqual(plan["planned_epochs_without_early_stop"], 3)
        verify_payload_seal(plan, "plan_sha256")
        tampered = dict(plan)
        tampered["run_count"] = 7
        with self.assertRaisesRegex(Phase1ABError, "checksum mismatch"):
            verify_payload_seal(tampered, "plan_sha256")

    def test_summary_ranks_runs_and_marks_insufficient_evidence_nonpromotable(
        self,
    ) -> None:
        config = self._config()
        corpus = self._Corpus()
        train_view = SimpleNamespace(
            examples=(object(),) * 6,
            view_id="train-view",
            view_sha256="c" * 64,
            to_json=lambda: {"training_view_id": "train-view"},
        )
        eval_view = SimpleNamespace(
            examples=(object(),) * 3,
            view_id="eval-view",
            view_sha256="d" * 64,
            to_json=lambda: {"training_view_id": "eval-view"},
        )
        plan = build_sweep_plan(
            corpus=corpus,
            training_view=train_view,
            evaluation_view=eval_view,
            config=config,
            batches_per_epoch=3,
            initialization_sha256="e" * 64,
            vocabulary_identity={},
            git_identity={"commit": "1" * 40, "dirty": False},
        )
        results = []
        for index, cell in enumerate(plan["combinations"]):
            results.append(
                {
                    "run_id": cell["run_id"],
                    "seed": cell["seed"],
                    "learning_rate": cell["learning_rate"],
                    "training": {"early_stopped": False, "updates_executed": 9},
                    "metrics": {
                        "conditional_kl": 1.0 - index * 0.01,
                        "opened_support_complete_action_soft_nll": 2.0 + index * 0.01,
                        "opened_support_all_modal_top1": index / 10.0,
                    },
                }
            )
        summary = build_sweep_summary(plan, results, source_immutability_verified=True)
        self.assertEqual(summary["schema_id"], RECOVERY_SWEEP_SUMMARY_SCHEMA_ID)
        self.assertEqual(summary["run_count"], 8)
        self.assertEqual(summary["best_run_id"], results[-1]["run_id"])
        self.assertFalse(summary["promotable"])
        self.assertFalse(summary["evidence"]["evidence_sufficient"])
        self.assertEqual(summary["total_updates_executed"], 72)
        verify_payload_seal(summary, "summary_sha256")

    def test_completed_run_result_is_sealed_and_idempotently_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result_path = root / "run-result.json"
            payload = _seal_payload(
                {
                    "schema_id": "result",
                    "plan_sha256": "a" * 64,
                    "run_id": "run",
                    "seed": 1,
                    "learning_rate": 0.0003,
                },
                "result_sha256",
            )
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = _load_or_run_result(
                run_id="run",
                seed=1,
                learning_rate=3e-4,
                run_dir=root,
                result_path=result_path,
                plan_sha256="a" * 64,
            )
            self.assertEqual(loaded, payload)
            payload["seed"] = 2
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RecoverySweepError):
                _load_or_run_result(
                    run_id="run",
                    seed=1,
                    learning_rate=3e-4,
                    run_dir=root,
                    result_path=result_path,
                    plan_sha256="a" * 64,
                )

    def test_source_immutability_verifier_detects_manifest_and_game_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            game = root / "game.json"
            manifest.write_text("manifest\n", encoding="utf-8")
            game.write_text("game\n", encoding="utf-8")
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            game_hash = hashlib.sha256(game.read_bytes()).hexdigest()
            verify_source_unchanged(
                manifest, manifest_hash, [{"path": str(game), "file_sha256": game_hash}]
            )
            game.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RecoverySweepError, "source replay changed"):
                verify_source_unchanged(
                    manifest,
                    manifest_hash,
                    [{"path": str(game), "file_sha256": game_hash}],
                )

    def test_cli_parses_bounded_recovery_options(self) -> None:
        args = _parser().parse_args(
            [
                "manifest.json",
                "out",
                "--seed",
                "1",
                "--seed",
                "2",
                "--seed",
                "3",
                "--seed",
                "4",
                "--learning-rate",
                "0.0003",
                "--learning-rate",
                "0.001",
                "--fixed-updates",
                "42",
                "--early-stop",
                "--early-stop-patience",
                "1",
                "--lr-schedule",
                "cosine",
                "--plan-only",
            ]
        )
        self.assertEqual(args.seeds, [1, 2, 3, 4])
        self.assertEqual(args.learning_rates, [0.0003, 0.001])
        self.assertEqual(args.fixed_updates, 42)
        self.assertTrue(args.early_stop)
        self.assertTrue(args.plan_only)


if __name__ == "__main__":
    unittest.main()

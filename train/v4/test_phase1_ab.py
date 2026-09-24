from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from v3.schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID
from v3.tensorize import V3Vocabulary

from v4.batch import tensorize_factorized_cases
from v4.phase1_ab import (
    ALL_OPEN_TRANSFORM_ID,
    ARM_ALL_OPEN_CONTROL,
    ARM_LEGACY_ALL_OPEN_EXPERIMENT,
    ARM_OPENED_ONLY,
    TRAINING_VIEW_ALL_OPEN_CONTROL,
    TRAINING_VIEW_OPENED_ONLY,
    Phase1ABConfig,
    Phase1ABError,
    _arm_provenance,
    _game_identities,
    aggregate_evaluation,
    build_export_command,
    build_phase1_model_gate,
    build_phase1_ab_plan,
    build_rust_parity_command,
    complete_action_log_probabilities,
    compare_evaluations,
    evaluate_prefix_batch,
    make_training_view,
    model_state_sha256,
    run_export_and_rust_parity,
    verify_payload_seal,
)
from v4.replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    GAME_RECORD_V5_SCHEMA_ID,
    LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
    RULESET_ID_V4,
    make_factorized_target_v4,
    parse_game_v4,
)
from v4.train import (
    POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT,
    POLICY_TARGET_SEMANTICS_OPENED_ONLY,
    V4Trainer,
    V4TrainingConfig,
    assert_production_policy_target_semantics,
    policy_target_semantics_lineage,
)


FIXTURE = Path(__file__).parents[1] / "v3" / "fixtures" / "0915" / "observation_action_v3.json"


class Phase1ABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.case = next(case for case in payload["cases"] if case["name"] == "runtime_rich")
        cls.actions = cls.case["legal_actions"]["actions"]
        cls.visits = [1] * 15 + [0] * (len(cls.actions) - 15)
        cls.opened = [True] * 16 + [False] * (len(cls.actions) - 16)
        cls.vocabulary = V3Vocabulary.from_cases([cls.case])

    def _example(self):
        target = make_factorized_target_v4(
            self.actions,
            self.visits,
            0.125,
            self.opened,
            search_semantics_id="augmentia.mcts-v4.factorized-gumbel-widening-public-chance.v2",
        )
        record = {
            "record_schema_id": GAME_RECORD_V5_SCHEMA_ID,
            "game_id": "phase1-ab-regression",
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
    def _config(seed: int = 7) -> V4TrainingConfig:
        return V4TrainingConfig(
            learning_rate=1e-3,
            batch_size=1,
            epochs=1,
            value_weight=0.0,
            seed=seed,
            device="cpu",
            model_dim=16,
            channels=8,
            residual_blocks=1,
            runtime_message_passes=1,
            encoder_kind="baseline",
        )

    def test_all_open_view_changes_masks_only_and_preserves_v3_lineage(self) -> None:
        source = self._example()
        source_snapshot = copy.deepcopy(source)
        opened = make_training_view([source], "opened_only")
        control = make_training_view([source], "all_open_control")

        self.assertEqual(opened.view_id, TRAINING_VIEW_OPENED_ONLY)
        self.assertEqual(control.view_id, TRAINING_VIEW_ALL_OPEN_CONTROL)
        self.assertEqual(opened.examples[0], source)
        self.assertEqual(source, source_snapshot)
        self.assertEqual(control.source_view_sha256, opened.source_view_sha256)
        self.assertNotEqual(control.view_sha256, opened.view_sha256)
        transformed = control.examples[0]
        self.assertEqual(transformed.target_schema_id, FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID)
        self.assertEqual(transformed.replay_schema_id, GAME_RECORD_V5_SCHEMA_ID)
        self.assertFalse(transformed.legacy)
        self.assertEqual(
            transformed.opened_action_mask,
            (True,) * len(self.actions),
        )
        self.assertTrue(
            all(all(row) for row in transformed.opened_masks)
        )
        self.assertEqual(control.changed_decisions, 1)
        self.assertEqual(control.source_opened_action_count, 16)
        self.assertEqual(control.view_opened_action_count, len(self.actions))
        self.assertTrue(control.is_control)
        self.assertEqual(
            control.policy_target_semantics_id,
            POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT,
        )
        self.assertTrue(
            control.policy_target_semantics_lineage["experiment_only"]
        )
        self.assertFalse(
            control.policy_target_semantics_lineage[
                "production_export_resume_eligible"
            ]
        )
        self.assertFalse(control.to_json()["promotable"])
        self.assertEqual(
            control.to_json()["explicit_all_open_transform"],
            ALL_OPEN_TRANSFORM_ID,
        )

    def test_all_open_view_rejects_legacy_or_mask_misalignment(self) -> None:
        source = self._example()
        with self.assertRaisesRegex(Phase1ABError, "strict replay.game.v5"):
            make_training_view(
                [
                    replace(
                        source,
                        target_schema_id=LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
                        legacy=True,
                    )
                ],
                "all_open_control",
            )
        with self.assertRaisesRegex(Phase1ABError, "root mask"):
            make_training_view(
                [replace(source, opened_action_mask=source.opened_action_mask[:-1])],
                "opened_only",
            )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "every root action opened"):
                V4Trainer(
                    [source],
                    Path(directory),
                    self._config(),
                    vocabulary=self.vocabulary,
                    policy_target_semantics_id=(
                        POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT
                    ),
                )

    def test_game_identity_binds_manifest_file_and_trajectory_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game_path = root / "game.json"
            game_path.write_text("{}\n", encoding="utf-8")
            trajectory = "a" * 64
            game_path.write_text(
                json.dumps({"trajectory_sha256": trajectory}) + "\n",
                encoding="utf-8",
            )
            file_sha = hashlib.sha256(game_path.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest = {
                "games_artifacts": [
                    {
                        "game_id": "g",
                        "file_sha256": file_sha,
                        "trajectory_sha256": trajectory,
                    }
                ]
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            dataset = SimpleNamespace(
                games=(
                    SimpleNamespace(
                        game_id="g",
                        game_index=0,
                        path=game_path,
                    ),
                ),
                source_manifests=({"path": str(manifest_path)},),
            )
            identities = _game_identities(dataset)
            self.assertEqual(identities[0]["file_sha256"], file_sha)
            manifest["games_artifacts"][0]["trajectory_sha256"] = "b" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(Phase1ABError, "trajectory hash mismatch"):
                _game_identities(dataset)

    def test_identical_config_produces_identical_initial_weights(self) -> None:
        source = self._example()
        opened = make_training_view([source], "opened_only")
        control = make_training_view([source], "all_open_control")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = V4Trainer(
                opened.examples,
                root / ARM_OPENED_ONLY,
                self._config(),
                vocabulary=self.vocabulary,
                validation_examples=[source],
            )
            second = V4Trainer(
                control.examples,
                root / ARM_ALL_OPEN_CONTROL,
                self._config(),
                vocabulary=self.vocabulary,
                validation_examples=[control.examples[0]],
                policy_target_semantics_id=(
                    POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT
                ),
            )
            different = V4Trainer(
                control.examples,
                root / "different-seed",
                self._config(seed=8),
                vocabulary=self.vocabulary,
                validation_examples=[control.examples[0]],
                policy_target_semantics_id=(
                    POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT
                ),
            )
            first_hash = model_state_sha256(first.model)
            self.assertEqual(first_hash, model_state_sha256(second.model))
            self.assertNotEqual(first_hash, model_state_sha256(different.model))

    def test_plan_binds_training_and_validation_views_independently(self) -> None:
        source = self._example()
        opened = make_training_view([source], "opened_only")
        control = make_training_view([source], "all_open_control")
        validation = make_training_view([replace(source, decision_id="validation:0")], "opened_only")
        corpus = SimpleNamespace(
            provenance={"dataset_manifest_sha256": "e" * 64},
            to_json=lambda: {
                "dataset_manifest_sha256": "e" * 64,
                "source_corpus_sha256": "f" * 64,
            },
            source_corpus_sha256="f" * 64,
            split_sha256="1" * 64,
            train_game_ids=("train",),
            validation_game_ids=("validation",),
            train_examples=opened.examples,
            validation_examples=validation.examples,
        )
        config = Phase1ABConfig(
            trainer=self._config(),
            run_parity=False,
            allow_smoke_corpus=True,
        )
        plan = build_phase1_ab_plan(
            corpus,
            opened,
            control,
            config,
            {"sha256": "2" * 64},
            "3" * 64,
            {"commit": "commit", "dirty": True},
        )
        verify_payload_seal(plan, "plan_sha256")
        self.assertNotEqual(
            plan["evaluation_view_sha256"], plan["views"][ARM_OPENED_ONLY]["training_view_sha256"]
        )
        self.assertEqual(
            plan["evaluation_view_sha256"], validation.view_sha256
        )
        self.assertEqual(
            plan["views"][ARM_ALL_OPEN_CONTROL]["view_opened_action_count"],
            len(self.actions),
        )
        self.assertEqual(
            plan["views"][ARM_OPENED_ONLY]["policy_target_semantics_id"],
            POLICY_TARGET_SEMANTICS_OPENED_ONLY,
        )
        self.assertEqual(
            plan["views"][ARM_ALL_OPEN_CONTROL]["policy_target_semantics_id"],
            POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT,
        )
        self.assertTrue(
            plan["training_validation_views"][ARM_ALL_OPEN_CONTROL][
                "policy_target_semantics_lineage"
            ]["experiment_only"]
        )
        self.assertTrue(plan["parity"]["experiment_arm_export_blocked"])

    def test_checkpoint_sidecar_records_arm_view_and_source_identity(self) -> None:
        source = self._example()
        view = make_training_view([source], "all_open_control")
        corpus = SimpleNamespace(
            provenance={"factorized_policy_schema_id": "augmentia.factorized-policy.v4.1"},
            source_corpus_sha256="a" * 64,
            split_sha256="b" * 64,
        )
        provenance = _arm_provenance(
            corpus, view, ARM_ALL_OPEN_CONTROL, "c" * 64, "d" * 64
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = V4Trainer(
                view.examples,
                Path(directory) / ARM_ALL_OPEN_CONTROL,
                self._config(),
                vocabulary=self.vocabulary,
                validation_examples=[view.examples[0]],
                provenance=provenance,
                policy_target_semantics_id=view.policy_target_semantics_id,
            )
            checkpoint, sidecar = trainer.save_checkpoint(Path(directory) / "final.pt")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            actual = payload["provenance"]
            self.assertEqual(actual["phase1_ab_arm_id"], ARM_ALL_OPEN_CONTROL)
            self.assertEqual(actual["training_view_id"], TRAINING_VIEW_ALL_OPEN_CONTROL)
            self.assertEqual(actual["training_view_sha256"], view.view_sha256)
            self.assertEqual(actual["source_training_view_sha256"], view.source_view_sha256)
            self.assertEqual(actual["source_corpus_sha256"], "a" * 64)
            self.assertEqual(actual["split_sha256"], "b" * 64)
            self.assertEqual(actual["phase1_initialization_sha256"], "d" * 64)
            self.assertTrue(actual["all_open_control"])
            self.assertTrue(actual["legacy_all_open_experiment"])
            self.assertFalse(actual["phase1_export_eligible"])
            self.assertFalse(actual["phase1_resume_eligible"])
            self.assertEqual(
                payload["policy_target_semantics_id"],
                POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT,
            )
            self.assertTrue(payload["policy_target_semantics_experiment_only"])
            self.assertFalse(
                payload["policy_target_semantics_production_eligible"]
            )
            self.assertEqual(
                actual["policy_target_semantics_lineage"],
                policy_target_semantics_lineage(
                    POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT
                ),
            )
            self.assertFalse(actual["phase1_promotable"])
            with self.assertRaisesRegex(ValueError, "opened-only"):
                assert_production_policy_target_semantics(payload)
            with self.assertRaisesRegex(ValueError, "opened-only"):
                V4Trainer.from_checkpoint(
                    checkpoint,
                    view.examples,
                    self._config(),
                    validation_examples=[view.examples[0]],
                    expected_vocabulary=self.vocabulary,
                )
            self.assertEqual(payload["checkpoint_sha256"], model_state_sha256_file(checkpoint))

    def test_opened_checkpoint_has_production_semantics_and_resume_gate_passes(self) -> None:
        source = self._example()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = V4Trainer(
                [source],
                root,
                self._config(),
                vocabulary=self.vocabulary,
                validation_examples=[source],
                policy_target_semantics_id=POLICY_TARGET_SEMANTICS_OPENED_ONLY,
            )
            checkpoint, sidecar = trainer.save_checkpoint(root / "opened.pt")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["policy_target_semantics_id"],
                POLICY_TARGET_SEMANTICS_OPENED_ONLY,
            )
            self.assertFalse(payload["policy_target_semantics_experiment_only"])
            self.assertTrue(payload["policy_target_semantics_production_eligible"])
            self.assertEqual(
                payload["provenance"]["policy_target_semantics_lineage"],
                policy_target_semantics_lineage(
                    POLICY_TARGET_SEMANTICS_OPENED_ONLY
                ),
            )
            self.assertEqual(
                assert_production_policy_target_semantics(payload), None
            )
            resumed = V4Trainer.from_checkpoint(
                checkpoint,
                [source],
                self._config(),
                validation_examples=[source],
                expected_vocabulary=self.vocabulary,
            )
            self.assertEqual(
                resumed.policy_target_semantics_id,
                POLICY_TARGET_SEMANTICS_OPENED_ONLY,
            )

    def test_conditional_metrics_ignore_closed_logits_but_complete_metrics_do_not(self) -> None:
        example = self._example()
        batch = tensorize_factorized_cases(
            [self.case],
            self.vocabulary,
            [self.visits],
            opened_action_masks=[self.opened],
        )
        base_logits = torch.zeros(batch.prefixes.visit_counts.shape, dtype=torch.float32)
        changed_logits = base_logits.clone()
        closed = batch.prefixes.legal_mask & ~batch.prefixes.opened_mask
        changed_logits[closed] = 25.0
        actions = [self.actions]
        decision_ids = ["phase1-ab-regression:0"]
        base_rows = evaluate_prefix_batch(
            base_logits, batch.prefixes, actions, decision_ids
        )
        changed_rows = evaluate_prefix_batch(
            changed_logits, batch.prefixes, actions, decision_ids
        )

        for key in (
            "conditional_cross_entropy",
            "conditional_target_entropy",
            "conditional_kl",
        ):
            self.assertAlmostEqual(base_rows[0][key], changed_rows[0][key], places=6)
        self.assertNotAlmostEqual(
            base_rows[0]["complete_action_soft_nll"],
            changed_rows[0]["complete_action_soft_nll"],
            places=5,
        )
        self.assertGreater(base_rows[0]["closed_children_excluded"], 0)
        self.assertTrue(
            math.isclose(
                base_rows[0]["conditional_cross_entropy"],
                base_rows[0]["conditional_target_entropy"]
                + base_rows[0]["conditional_kl"],
                rel_tol=1e-5,
                abs_tol=2e-5,
            )
        )

        log_probabilities = complete_action_log_probabilities(
            actions, batch.prefixes, changed_logits
        )
        self.assertTrue(
            torch.allclose(
                log_probabilities.exp().sum(dim=1),
                torch.ones(1),
                atol=5e-4,
                rtol=0.0,
            )
        )

    def test_evaluation_aggregate_and_paired_bootstrap_are_deterministic(self) -> None:
        opened = aggregate_evaluation(
            [
                {
                    "decision_id": "g:0",
                    "game_id": "g",
                    "conditional_cross_entropy": 1.5,
                    "conditional_target_entropy": 0.5,
                    "conditional_kl": 1.0,
                    "complete_action_soft_nll": 1.6,
                    "complete_action_modal_nll": 1.7,
                    "mcts_modal_top1": 1.0,
                    "mcts_modal_confidence": 0.8,
                    "calibration_gap": 0.2,
                    "opened_children": 3,
                    "closed_children_excluded": 1,
                }
            ],
            ece_bins=10,
        )
        control_row = copy.deepcopy(opened["per_decision"][0])
        control_row.update(
            {
                "conditional_cross_entropy": 2.0,
                "conditional_target_entropy": 0.5,
                "conditional_kl": 1.5,
                "complete_action_soft_nll": 2.0,
                "complete_action_modal_nll": 2.2,
                "mcts_modal_top1": 0.0,
                "mcts_modal_confidence": 0.6,
                "calibration_gap": 0.6,
            }
        )
        control = aggregate_evaluation([control_row], ece_bins=10)
        first = compare_evaluations(
            opened, control, bootstrap_samples=25, bootstrap_seed=19
        )
        second = compare_evaluations(
            opened, control, bootstrap_samples=25, bootstrap_seed=19
        )
        self.assertEqual(first, second)
        improvements = first["improvements"]
        self.assertGreater(improvements["conditional_kl_improvement"]["mean"], 0.0)
        self.assertGreater(
            improvements["complete_action_soft_nll_improvement"]["mean"], 0.0
        )
        self.assertEqual(improvements["mcts_modal_top1_improvement"]["mean"], 1.0)
        ece_improvement = improvements["mcts_modal_ece_reduction"]
        self.assertGreater(ece_improvement["mean"], 0.0)
        self.assertEqual(
            ece_improvement["definition"],
            "control aggregate ECE - opened aggregate ECE",
        )
        self.assertEqual(ece_improvement["bootstrap"]["bins"], 10)
        self.assertAlmostEqual(
            ece_improvement["bootstrap"]["observed_control_minus_opened"],
            0.4,
            places=12,
        )
        self.assertEqual(sum(item["count"] for item in opened["ece_bins"]), 1)

    def test_aggregate_ece_is_not_mislabeled_mean_calibration_gap(self) -> None:
        def row(decision_id: str, confidence: float, correct: float) -> dict:
            return {
                "decision_id": decision_id,
                "game_id": decision_id.split(":")[0],
                "conditional_cross_entropy": 1.0,
                "conditional_target_entropy": 0.4,
                "conditional_kl": 0.6,
                "complete_action_soft_nll": 1.0,
                "complete_action_modal_nll": 1.0,
                "mcts_modal_top1": correct,
                "mcts_modal_confidence": confidence,
                "calibration_gap": abs(correct - confidence),
                "opened_children": 1,
                "closed_children_excluded": 0,
            }

        evaluation = aggregate_evaluation(
            [row("same-game:0", 0.91, 1.0), row("same-game:1", 0.99, 0.0)],
            ece_bins=10,
        )
        self.assertAlmostEqual(evaluation["mcts_modal_ece"], 0.45, places=12)
        self.assertAlmostEqual(
            evaluation["mean_calibration_gap"], 0.54, places=12
        )
        self.assertNotAlmostEqual(
            evaluation["mcts_modal_ece"],
            evaluation["mean_calibration_gap"],
        )

    def test_model_gate_requires_primary_gain_and_complete_action_non_degradation(self) -> None:
        def row(
            decision_id: str,
            *,
            kl: float,
            nll: float,
            top1: float,
            confidence: float,
            tail_recall: float,
        ) -> dict:
            return {
                "decision_id": decision_id,
                "game_id": "gate-game",
                "conditional_cross_entropy": 0.5 + kl,
                "conditional_target_entropy": 0.5,
                "conditional_kl": kl,
                "complete_action_soft_nll": nll,
                "complete_action_modal_nll": nll + 0.1,
                "mcts_modal_top1": top1,
                "mcts_modal_confidence": confidence,
                "calibration_gap": abs(top1 - confidence),
                "opened_children": 3,
                "closed_children_excluded": 1,
                "visited_actions": 2,
                "visited_tail_actions": 1,
                "visited_tail_hits_at_3": int(tail_recall),
                "visited_tail_recall_at_3": tail_recall,
            }

        opened = aggregate_evaluation(
            [row("gate-game:0", kl=0.5, nll=1.0, top1=1.0, confidence=0.8, tail_recall=1.0)],
            ece_bins=10,
        )
        control = aggregate_evaluation(
            [row("gate-game:0", kl=1.0, nll=1.01, top1=0.0, confidence=0.6, tail_recall=0.0)],
            ece_bins=10,
        )
        comparison = compare_evaluations(
            opened, control, bootstrap_samples=10, bootstrap_seed=7
        )
        config = Phase1ABConfig(
            trainer=self._config(),
            run_parity=False,
            allow_smoke_corpus=True,
            nll_non_degradation_tolerance=0.02,
            top1_non_degradation_tolerance=0.01,
            ece_non_degradation_tolerance=0.01,
        )
        gate = build_phase1_model_gate(
            opened,
            control,
            comparison,
            config,
            parity_complete=False,
        )
        self.assertEqual(gate["outcome"], "pass")
        self.assertGreater(
            comparison["improvements"][
                "conditional_cross_entropy_improvement"
            ]["mean"],
            0.0,
        )
        self.assertTrue(gate["primary"]["conditional_kl"]["consistent_improvement"])
        self.assertTrue(gate["non_degradation_passed"])
        self.assertFalse(gate["promotable"])
        regressed_control_row = dict(control["per_decision"][0])
        regressed_control_row["complete_action_soft_nll"] = 0.5
        regressed_control_row["complete_action_modal_nll"] = 0.6
        regressed_control = aggregate_evaluation(
            [regressed_control_row], ece_bins=10
        )
        regressed_comparison = compare_evaluations(
            opened, regressed_control, bootstrap_samples=10, bootstrap_seed=7
        )
        failed = build_phase1_model_gate(
            opened,
            regressed_control,
            regressed_comparison,
            config,
            parity_complete=False,
        )
        self.assertEqual(failed["outcome"], "fail")
        self.assertFalse(failed["non_degradation_passed"])

    def test_baseline_and_skip_parity_gates_fail_closed(self) -> None:
        hybrid = replace(self._config(), encoder_kind="hybrid")
        with self.assertRaisesRegex(Phase1ABError, "baseline"):
            Phase1ABConfig(trainer=hybrid).validate()
        with self.assertRaisesRegex(Phase1ABError, "explicit smoke"):
            Phase1ABConfig(
                trainer=self._config(), run_parity=False
            ).validate()
        with self.assertRaisesRegex(Phase1ABError, "Rust validator"):
            Phase1ABConfig(trainer=self._config()).validate()

    def test_export_and_rust_parity_commands_are_explicit(self) -> None:
        export = build_export_command(
            Path("checkpoint.pt"),
            Path("manifest.json"),
            Path("output/model.onnx"),
            Path("vocabulary.json"),
            python_executable="python",
        )
        self.assertEqual(export[:4], ["python", "-m", "v4.export_checkpoint", str(Path("checkpoint.pt").resolve())])
        self.assertIn("--runtime-vocabulary", export)
        rust = build_rust_parity_command(
            Path("validator.exe"),
            Path("model.v4-bundle.metadata.json"),
            Path("model.parity.json"),
            2e-4,
        )
        self.assertEqual(rust[1], "--bundle")
        self.assertEqual(rust[3], "--parity")
        self.assertEqual(float(rust[-1]), 2e-4)
        with self.assertRaisesRegex(Phase1ABError, "blocked"):
            run_export_and_rust_parity(
                arm_id=ARM_LEGACY_ALL_OPEN_EXPERIMENT,
                checkpoint=Path("experiment.pt"),
                manifest=Path("manifest.json"),
                output=Path("output/model"),
                runtime_vocabulary=Path("vocabulary.json"),
                rust_validator=Path("validator.exe"),
                tolerance=2e-4,
                repo_root=Path("."),
            )

    def test_seal_detection_rejects_tampering(self) -> None:
        from v4.phase1_ab import _seal_payload

        sealed = _seal_payload({"schema_id": "test", "value": 1}, "test_sha256")
        self.assertEqual(verify_payload_seal(sealed, "test_sha256"), sealed["test_sha256"])
        sealed["value"] = 2
        with self.assertRaisesRegex(Phase1ABError, "checksum"):
            verify_payload_seal(sealed, "test_sha256")


def model_state_sha256_file(path: Path) -> str:
    from v4.phase1_ab import sha256_file

    return sha256_file(path)


if __name__ == "__main__":
    unittest.main()

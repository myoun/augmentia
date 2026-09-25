from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from v3.schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID, SchemaError
from v3.tensorize import V3Vocabulary

from v4.loss import (
    conditional_prefix_policy_metrics,
    mixed_row_all_legal_shadow_cross_entropy,
    mixed_row_shadow_mask,
)
from v4.replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    GAME_RECORD_V5_SCHEMA_ID,
    LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
    LEGACY_GAME_RECORD_V4_SCHEMA_ID,
    RULESET_ID_V4,
    make_factorized_target_v4,
    parse_game_v4,
)
from v4.train import (
    MIXED_ROW_SHADOW_LOSS_ID,
    MIXED_ROW_SHADOW_LOSS_SCHEMA_ID,
    V4Trainer,
    V4TrainingConfig,
)


FIXTURE = Path(__file__).parents[1] / "v3" / "fixtures" / "0915" / "observation_action_v3.json"


class ReplayTrainerOpenedMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.case = next(case for case in payload["cases"] if case["name"] == "runtime_rich")
        cls.actions = cls.case["legal_actions"]["actions"]
        cls.visits = [1] * 15 + [0] * (len(cls.actions) - 15)
        cls.opened = [True] * 16 + [False] * (len(cls.actions) - 16)
        cls.target = make_factorized_target_v4(
            cls.actions,
            cls.visits,
            0.125,
            cls.opened,
            search_semantics_id="augmentia.mcts-v4.factorized-gumbel-widening-public-chance.v2",
        )

    @staticmethod
    def _config() -> V4TrainingConfig:
        return V4TrainingConfig(
            learning_rate=1e-3,
            batch_size=1,
            epochs=1,
            value_weight=0.0,
            seed=7,
            device="cpu",
            model_dim=16,
            channels=8,
            residual_blocks=1,
            runtime_message_passes=1,
        )

    def _record(self, *, legacy: bool = False) -> dict:
        target = copy.deepcopy(self.target)
        record_schema = GAME_RECORD_V5_SCHEMA_ID
        target_lineage = FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID
        if legacy:
            record_schema = LEGACY_GAME_RECORD_V4_SCHEMA_ID
            target_lineage = LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID
            target["schema_id"] = target_lineage
            target.pop("opened_action_mask", None)
            for prefix in target["prefix_targets"]:
                prefix.pop("opened_masks", None)
            target["trie_sha256"] = hashlib.sha256(
                json.dumps(
                    target["prefix_targets"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        return {
            "record_schema_id": record_schema,
            "game_id": "opened-replay-regression",
            "ruleset": {"id": RULESET_ID_V4},
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "factorized_policy_schema_id": "augmentia.factorized-policy.v4.1",
            "factorized_target_schema_id": target_lineage,
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

    def test_replay_to_trainer_preserves_root_and_prefix_masks(self) -> None:
        example = parse_game_v4(self._record())[0]
        self.assertEqual(example.opened_action_mask, tuple(self.opened))
        self.assertEqual(
            example.opened_masks,
            tuple(tuple(item["opened_masks"]) for item in self.target["prefix_targets"]),
        )
        self.assertEqual(len(example.opened_masks), len(example.prefix_targets))
        self.assertEqual(
            example.target_schema_id, FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID
        )
        with tempfile.TemporaryDirectory() as temp:
            trainer = V4Trainer(
                [example],
                Path(temp),
                self._config(),
                vocabulary=V3Vocabulary.from_cases([self.case]),
            )
            batch, _values = trainer._tensorize_selected([example])
            root_opened = batch.prefixes.complete_action_opened_mask[0]
            self.assertTrue(
                torch.equal(root_opened[:16], torch.ones(16, dtype=torch.bool))
            )
            self.assertTrue(
                torch.equal(
                    root_opened[16:],
                    torch.zeros(len(self.actions) - 16, dtype=torch.bool),
                )
            )
            mixed_rows = torch.nonzero(
                (batch.prefixes.legal_mask.sum(dim=1) >= 2)
                & (
                    batch.prefixes.opened_mask.sum(dim=1)
                    < batch.prefixes.legal_mask.sum(dim=1)
                )
                & (batch.prefixes.opened_mask.sum(dim=1) > 0)
            ).flatten()
            self.assertGreaterEqual(len(mixed_rows), 1)

    def test_mixed_shadow_loss_excludes_all_closed_rows_from_gradient(self) -> None:
        example = parse_game_v4(self._record())[0]
        with tempfile.TemporaryDirectory() as temp:
            trainer = V4Trainer(
                [example],
                Path(temp),
                self._config(),
                vocabulary=V3Vocabulary.from_cases([self.case]),
            )
            batch, _values = trainer._tensorize_selected([example])
            logits = torch.randn(
                batch.prefixes.visit_counts.shape,
                dtype=torch.float32,
                requires_grad=True,
            )
            mixed = mixed_row_shadow_mask(batch.prefixes)
            all_closed = (
                batch.prefixes.legal_mask.any(dim=1)
                & ~batch.prefixes.opened_mask.any(dim=1)
            )
            self.assertGreater(int(mixed.sum()), 0)
            self.assertGreater(int(all_closed.sum()), 0)
            row_loss = mixed_row_all_legal_shadow_cross_entropy(
                logits, batch.prefixes, reduction="none"
            )
            self.assertTrue(torch.equal(row_loss[all_closed], torch.zeros_like(row_loss[all_closed])))
            loss = mixed_row_all_legal_shadow_cross_entropy(logits, batch.prefixes)
            loss.backward()
            self.assertIsNotNone(logits.grad)
            self.assertEqual(int(torch.count_nonzero(logits.grad[all_closed])), 0)
            self.assertGreater(int(torch.count_nonzero(logits.grad[mixed])), 0)
            self.assertEqual(
                int(torch.count_nonzero(logits.grad[~batch.prefixes.legal_mask])),
                0,
            )

    def test_lambda_zero_is_exact_opened_only_loss_and_gradient(self) -> None:
        example = parse_game_v4(self._record())[0]
        with tempfile.TemporaryDirectory() as temp:
            trainer = V4Trainer(
                [example],
                Path(temp),
                self._config(),
                vocabulary=V3Vocabulary.from_cases([self.case]),
            )
            trainer.model.eval()
            batch, values = trainer._tensorize_selected([example])
            output = trainer.model(batch)
            expected, _entropy, _kl = conditional_prefix_policy_metrics(
                output.prefix_logits, batch.prefixes
            )
            with mock.patch.object(trainer.model, "forward", return_value=output):
                total, policy, _entropy, _kl, shadow_ce, _weighted, _value, _count = (
                    trainer._loss_for_batch(batch, values)
                )
            self.assertEqual(total.item(), policy.item())
            self.assertEqual(total.item(), expected.item())
            self.assertGreater(shadow_ce.item(), 0.0)
            expected_gradient = torch.autograd.grad(
                expected, output.prefix_logits, retain_graph=True
            )[0]
            actual_gradient = torch.autograd.grad(total, output.prefix_logits)[0]
            self.assertTrue(
                torch.equal(expected_gradient, actual_gradient)
            )

    def test_shadow_lambda_provenance_history_and_resume_contract(self) -> None:
        example = parse_game_v4(self._record())[0]
        config = replace(self._config(), mixed_row_shadow_lambda=0.05)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trainer = V4Trainer(
                [example],
                root,
                config,
                vocabulary=V3Vocabulary.from_cases([self.case]),
                validation_examples=[example],
            )
            trainer.fit()
            for section in ("train", "validation"):
                metrics = trainer.history[0][section]
                self.assertEqual(metrics["mixed_row_shadow_lambda"], 0.05)
                self.assertGreater(metrics["mixed_row_shadow_ce"], 0.0)
                self.assertGreater(metrics["mixed_row_shadow_rows"], 0)
                self.assertAlmostEqual(
                    metrics["mixed_row_shadow_weighted_loss"],
                    0.05 * metrics["mixed_row_shadow_ce"],
                    places=7,
                )
                self.assertAlmostEqual(
                    metrics["loss"],
                    metrics["policy_loss"]
                    + metrics["mixed_row_shadow_weighted_loss"],
                    places=7,
                )
            checkpoint, sidecar = trainer.save_checkpoint(root / "final.pt")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["training_config"]["mixed_row_shadow_lambda"], 0.05
            )
            shadow = payload["provenance"]["mixed_row_shadow"]
            self.assertEqual(shadow["schema_id"], MIXED_ROW_SHADOW_LOSS_SCHEMA_ID)
            self.assertEqual(shadow["loss_id"], MIXED_ROW_SHADOW_LOSS_ID)
            self.assertEqual(shadow["lambda"], 0.05)
            self.assertFalse(shadow["all_closed_rows_contribute"])
            self.assertFalse(shadow["production_runtime_semantics_changed"])
            self.assertEqual(
                payload["policy_target_semantics_id"],
                "augmentia.policy-target.opened-mask.v1",
            )
            V4Trainer.from_checkpoint(
                checkpoint,
                [example],
                config,
                validation_examples=[example],
                expected_vocabulary=V3Vocabulary.from_cases([self.case]),
            )
            with self.assertRaisesRegex(
                ValueError, "mixed_row_shadow_lambda changed"
            ):
                V4Trainer.from_checkpoint(
                    checkpoint,
                    [example],
                    replace(config, mixed_row_shadow_lambda=0.1),
                    validation_examples=[example],
                    expected_vocabulary=V3Vocabulary.from_cases([self.case]),
                )

    def test_shadow_lambda_validation_and_opened_only_scope(self) -> None:
        for invalid in (-0.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "shadow lambda"):
                    replace(self._config(), mixed_row_shadow_lambda=invalid).validate()
        legacy = parse_game_v4(self._record(legacy=True), allow_legacy=True)[0]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "opened-only policy semantics"):
                V4Trainer(
                    [legacy],
                    Path(temp),
                    replace(self._config(), mixed_row_shadow_lambda=0.05),
                    vocabulary=V3Vocabulary.from_cases([self.case]),
                    allow_legacy=True,
                )

    def test_production_record_and_trainer_reject_missing_masks(self) -> None:
        record = self._record()
        missing_lineage = copy.deepcopy(record)
        missing_lineage.pop("factorized_target_schema_id")
        with self.assertRaisesRegex(SchemaError, "factorized target schema"):
            parse_game_v4(missing_lineage)

        missing_root_mask = copy.deepcopy(record)
        missing_root_mask["steps"][0]["policy_target"].pop("opened_action_mask")
        with self.assertRaisesRegex(SchemaError, "opened_action_mask"):
            parse_game_v4(missing_root_mask)

        example = parse_game_v4(record)[0]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "prefix opened_masks"):
                V4Trainer(
                    [replace(example, opened_masks=())],
                    Path(temp),
                    self._config(),
                    vocabulary=V3Vocabulary.from_cases([self.case]),
                )
            with self.assertRaisesRegex(ValueError, "opened_action_mask"):
                V4Trainer(
                    [replace(example, opened_action_mask=())],
                    Path(temp),
                    self._config(),
                    vocabulary=V3Vocabulary.from_cases([self.case]),
                )

    def test_legacy_requires_explicit_quarantine_and_cannot_mix(self) -> None:
        with self.assertRaisesRegex(SchemaError, "legacy replay schema"):
            parse_game_v4(self._record(legacy=True))
        legacy_example = parse_game_v4(
            self._record(legacy=True), allow_legacy=True
        )[0]
        self.assertTrue(legacy_example.legacy)
        self.assertEqual(
            legacy_example.target_schema_id,
            LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
        )
        v3_example = parse_game_v4(self._record())[0]
        with tempfile.TemporaryDirectory() as temp:
            legacy_trainer = V4Trainer(
                [legacy_example],
                Path(temp),
                self._config(),
                vocabulary=V3Vocabulary.from_cases([self.case]),
                allow_legacy=True,
            )
            legacy_batch, _values = legacy_trainer._tensorize_selected(
                [legacy_example]
            )
            self.assertTrue(
                bool(
                    (
                        legacy_batch.prefixes.complete_action_opened_mask
                        & legacy_batch.prefixes.complete_action_mask
                    ).all()
                )
            )
            with self.assertRaisesRegex(ValueError, "share one"):
                V4Trainer(
                    [legacy_example, v3_example],
                    Path(temp),
                    self._config(),
                    vocabulary=V3Vocabulary.from_cases([self.case]),
                )

    def test_checkpoint_records_and_rejects_legacy_lineage(self) -> None:
        example = parse_game_v4(self._record())[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trainer = V4Trainer(
                [example],
                root,
                self._config(),
                vocabulary=V3Vocabulary.from_cases([self.case]),
            )
            checkpoint, sidecar = trainer.save_checkpoint(root / "final.pt")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema_id"], "augmentia.v4-trainer-checkpoint.v3"
            )
            self.assertEqual(
                payload["factorized_target_schema_id"],
                FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
            )
            self.assertEqual(payload["replay_schema_id"], GAME_RECORD_V5_SCHEMA_ID)
            self.assertEqual(
                payload["provenance"]["factorized_target_schema_id"],
                FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
            )
            payload["schema_id"] = "augmentia.v4-trainer-checkpoint.v2"
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "production V4 trainer checkpoint"):
                V4Trainer.from_checkpoint(checkpoint, [example])


if __name__ == "__main__":
    unittest.main()

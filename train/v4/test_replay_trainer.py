from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from v3.schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID, SchemaError
from v3.tensorize import V3Vocabulary

from v4.replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    GAME_RECORD_V5_SCHEMA_ID,
    LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
    LEGACY_GAME_RECORD_V4_SCHEMA_ID,
    RULESET_ID_V4,
    make_factorized_target_v4,
    parse_game_v4,
)
from v4.train import V4Trainer, V4TrainingConfig


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

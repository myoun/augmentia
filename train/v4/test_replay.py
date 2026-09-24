from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from v3.schema import SchemaError

from v4.replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    make_factorized_target_v4,
    validate_factorized_target_v4,
)


class ReplayV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = json.loads(
            (
                Path(__file__).parent.parent
                / "v3"
                / "fixtures"
                / "factorized_policy_v4.json"
            ).read_text(encoding="utf-8")
        )
        self.actions = [case["action"] for case in fixture["cases"]]

    def test_target_round_trip_and_visit_conservation(self) -> None:
        visits = [1, 2, 3, 4, 5, 6]
        target = make_factorized_target_v4(
            self.actions,
            visits,
            0.25,
            synthetic_all_open=True,
        )
        actual_visits, prefixes = validate_factorized_target_v4(target, self.actions)
        self.assertEqual(actual_visits, tuple(visits))
        self.assertEqual(
            target["search_semantics_id"],
            "augmentia.mcts-v4.factorized-gumbel-widening-public-chance.v2",
        )
        root = next(item for item in prefixes if not item.prefix)
        self.assertEqual(sum(root.visit_counts), sum(visits))
        with self.assertRaisesRegex(SchemaError, "synthetic_all_open"):
            make_factorized_target_v4(self.actions, visits, 0.25)

    def test_wide_target_keeps_16_opened_including_zero_visit(self) -> None:
        actions = [
            {
                "kind": "card",
                "card": index,
                "definition_id": f"fixture-card-{index}",
                "target": {"kind": "none"},
            }
            for index in range(32)
        ]
        visits = [1] * 15 + [0] * 17
        opened = [True] * 16 + [False] * 16
        target = make_factorized_target_v4(actions, visits, 0.0, opened)
        validate_factorized_target_v4(target, actions)
        mixed = next(
            row
            for row in target["prefix_targets"]
            if row["prefix"] == ["family:card"]
        )
        self.assertEqual(len(mixed["children"]), 32)
        self.assertEqual(sum(mixed["opened_masks"]), 16)
        opened_zero = mixed["children"].index("card:15")
        closed_zero = mixed["children"].index("card:16")
        self.assertEqual(mixed["visit_counts"][opened_zero], 0)
        self.assertTrue(mixed["opened_masks"][opened_zero])
        self.assertEqual(mixed["visit_counts"][closed_zero], 0)
        self.assertFalse(mixed["opened_masks"][closed_zero])

    def test_shared_rust_v3_fixture_round_trips_with_opened_masks(self) -> None:
        fixture = json.loads(
            (
                Path(__file__).parent
                / "fixtures"
                / "factorized_mcts_target_v3.json"
            ).read_text(encoding="utf-8")
        )
        actions = fixture["actions"]
        target = fixture["target"]
        visits, prefixes = validate_factorized_target_v4(target, actions)
        self.assertEqual(target["schema_id"], FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID)
        self.assertEqual(visits, (3, 0, 0))
        self.assertEqual(target["opened_action_mask"], [True, True, False])
        root = next(item for item in prefixes if not item.prefix)
        self.assertEqual(root.visit_counts, (0, 3, 0))
        root_wire = next(
            item for item in target["prefix_targets"] if not item["prefix"]
        )
        self.assertEqual(root_wire["opened_masks"], [False, True, True])
        for row in target["prefix_targets"]:
            self.assertEqual(len(row["opened_masks"]), len(row["children"]))
        regenerated = make_factorized_target_v4(
            actions,
            target["complete_visit_counts"],
            target["root_value"],
            target["opened_action_mask"],
            search_semantics_id=target["search_semantics_id"],
            root_selection_id=target["root_selection_id"],
            chance_sampling_id=target["chance_sampling_id"],
            chance_outcome_samples=target["chance_outcome_samples"],
        )
        self.assertEqual(regenerated, target)
        self.assertEqual(
            target["trie_sha256"],
            "e2c30b93c35768e1364720b2aaefa5cfee33bdc59c3fbb0f7debfa83e9a39605",
        )
        changed = copy.deepcopy(target)
        changed["opened_action_mask"][1] = False
        with self.assertRaises(SchemaError):
            validate_factorized_target_v4(changed, actions)
        changed = copy.deepcopy(target)
        changed["prefix_targets"][0].pop("opened_masks")
        with self.assertRaises(SchemaError):
            validate_factorized_target_v4(changed, actions)

    def test_corrupted_prefix_or_checksum_is_rejected(self) -> None:
        target = make_factorized_target_v4(
            self.actions,
            [1] * len(self.actions),
            0.0,
            synthetic_all_open=True,
        )
        corrupted = copy.deepcopy(target)
        corrupted["prefix_targets"][0]["visit_counts"][0] += 1
        with self.assertRaisesRegex(SchemaError, "prefix targets"):
            validate_factorized_target_v4(corrupted, self.actions)
        corrupted = copy.deepcopy(target)
        corrupted["trie_sha256"] = "0" * 64
        with self.assertRaisesRegex(SchemaError, "checksum"):
            validate_factorized_target_v4(corrupted, self.actions)
        corrupted = copy.deepcopy(target)
        corrupted["chance_outcome_samples"] = 0
        with self.assertRaisesRegex(SchemaError, "chance_outcome_samples"):
            validate_factorized_target_v4(corrupted, self.actions)


if __name__ == "__main__":
    unittest.main()

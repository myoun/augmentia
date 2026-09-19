from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from v3.schema import SchemaError
from v4.replay import make_factorized_target_v4, validate_factorized_target_v4


class ReplayV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = json.loads(
            (Path(__file__).parent.parent / "v3" / "fixtures" / "factorized_policy_v4.json").read_text(
                encoding="utf-8"
            )
        )
        self.actions = [case["action"] for case in fixture["cases"]]

    def test_target_round_trip_and_visit_conservation(self) -> None:
        visits = [1, 2, 3, 4, 5, 6]
        target = make_factorized_target_v4(self.actions, visits, 0.25)
        actual_visits, prefixes = validate_factorized_target_v4(target, self.actions)
        self.assertEqual(actual_visits, tuple(visits))
        root = next(item for item in prefixes if not item.prefix)
        self.assertEqual(sum(root.visit_counts), sum(visits))

    def test_corrupted_prefix_or_checksum_is_rejected(self) -> None:
        target = make_factorized_target_v4(self.actions, [1] * len(self.actions), 0.0)
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

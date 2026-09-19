from __future__ import annotations

import json
import unittest
from pathlib import Path

from v3.factorized import (
    FactorizedPolicyError,
    aggregate_prefix_visits,
    build_factorized_trie,
    factorized_action_path,
    factorized_action_wire_path,
)


class FactorizedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "observation_action_v3.json").read_text(
                encoding="utf-8"
            )
        )
        cls.cases = fixture["cases"]

    def test_shared_python_rust_golden_paths(self) -> None:
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "factorized_policy_v4.json").read_text(
                encoding="utf-8"
            )
        )
        for case in fixture["cases"]:
            self.assertEqual(factorized_action_wire_path(case["action"]), tuple(case["path"]))

    def test_every_fixture_action_has_one_unique_complete_path(self) -> None:
        for case in self.cases:
            actions = case["legal_actions"]["actions"]
            if not actions:
                continue
            paths = [factorized_action_path(action) for action in actions]
            self.assertEqual(len(paths), len(set(paths)), case["name"])
            build_factorized_trie(actions)

    def test_prefix_visits_conserve_complete_visits(self) -> None:
        actions = next(
            case["legal_actions"]["actions"]
            for case in self.cases
            if case["name"] == "standard_playing"
        )
        visits = list(range(1, len(actions) + 1))
        targets = aggregate_prefix_visits(actions, visits)
        root = next(target for target in targets if not target.prefix)
        self.assertEqual(sum(root.visit_counts), sum(visits))
        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.children[0], ("family", "move"))
        self.assertTrue(all(sum(target.visit_counts) >= 0 for target in targets))

    def test_target_sequence_uses_canonical_order_and_completion_edge(self) -> None:
        first = {
            "kind": "card",
            "card": 2,
            "definition_id": "example",
            "target": {"kind": "squares", "squares": [1]},
        }
        second = {
            "kind": "card",
            "card": 2,
            "definition_id": "example",
            "target": {"kind": "squares", "squares": [1, 9]},
        }
        paths = [factorized_action_path(action) for action in (first, second)]
        self.assertNotEqual(paths[0], paths[1])
        build_factorized_trie([first, second])

    def test_duplicate_paths_are_rejected(self) -> None:
        action = {
            "kind": "draft_pick",
            "offer_index": 0,
            "definition_id": "siren",
        }
        with self.assertRaisesRegex(FactorizedPolicyError, "unique canonical paths"):
            build_factorized_trie([action, action])


if __name__ == "__main__":
    unittest.main()

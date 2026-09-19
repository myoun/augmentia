from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from v3.factorized import FactorizedPolicyError
from v3.tensorize import V3Vocabulary, tensorize_cases

from v4.batch import (
    FACTOR_FIELD_IDS,
    assert_visit_conservation,
    collate_factorized_batch,
    encode_factor_edge,
    tensorize_factorized_cases,
)


FIXTURE = Path(__file__).parents[1] / "v3" / "fixtures" / "observation_action_v3.json"


class FactorizedBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_serialized_v3_actions_collate_and_conserve_visits(self) -> None:
        cases = self.suite["cases"][:2]
        vocabulary = V3Vocabulary.from_cases(cases)
        visits = [
            [index + 1 for index in range(len(case["legal_actions"]["actions"]))]
            for case in cases
        ]
        batch = tensorize_factorized_cases(cases, vocabulary, visits)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.prefixes.prefix.field.ndim, 2)
        self.assertEqual(batch.prefixes.children.field.shape, batch.prefixes.visit_counts.shape)
        assert_visit_conservation(batch.prefixes)
        for decision, expected in enumerate(map(sum, visits)):
            rows = torch.nonzero(batch.prefixes.state_index == decision).flatten()
            roots = [int(row) for row in rows if not batch.prefixes.prefix.mask[row].any()]
            self.assertEqual(len(roots), 1)
            self.assertEqual(int(batch.prefixes.visit_counts[roots[0]].sum()), expected)

    def test_opened_zero_child_is_distinct_from_closed_child(self) -> None:
        case = self.suite["cases"][0]
        vocabulary = V3Vocabulary.from_cases([case])
        states = tensorize_cases([case], vocabulary)
        actions = case["legal_actions"]["actions"][:3]
        opened = collate_factorized_batch(states, [actions], [[7, 0, 1]])
        closed = collate_factorized_batch(
            states, [actions], [[7, 0, 1]], opened_action_masks=[[True, False, True]]
        )
        # The family prefix is the only three-way offer decision.
        row_a = int(torch.nonzero(opened.prefixes.legal_mask.sum(1) == 3).flatten()[0])
        row_b = int(torch.nonzero(closed.prefixes.legal_mask.sum(1) == 3).flatten()[0])
        self.assertEqual(int(opened.prefixes.opened_mask[row_a].sum()), 3)
        self.assertEqual(int(closed.prefixes.opened_mask[row_b].sum()), 2)
        self.assertTrue(torch.equal(opened.prefixes.visit_counts[row_a], closed.prefixes.visit_counts[row_b]))

    def test_structured_edge_encoding_preserves_type_and_target_order(self) -> None:
        first = encode_factor_edge(("target_0", 9))
        second = encode_factor_edge(("target_1", 9))
        square = encode_factor_edge(("target_square", 9))
        self.assertEqual(first[0], FACTOR_FIELD_IDS["target_sequence"])
        self.assertEqual((first[3], second[3]), (1, 2))
        self.assertEqual(square[0], FACTOR_FIELD_IDS["target_square"])
        self.assertNotEqual(first[0], square[0])

    def test_mixed_none_and_numeric_promotions_have_stable_rows(self) -> None:
        case = self.suite["cases"][0]
        states = tensorize_cases([case], V3Vocabulary.from_cases([case]))
        actions = [
            {"kind": "move", "piece": 0, "to": 63, "promotion": None},
            {"kind": "move", "piece": 0, "to": 63, "promotion": 5},
        ]
        batch = collate_factorized_batch(states, [actions], [[2, 3]])
        two_way = torch.nonzero(batch.prefixes.legal_mask.sum(1) == 2).flatten()
        self.assertEqual(len(two_way), 1)
        self.assertEqual(sorted(batch.prefixes.visit_counts[int(two_way[0]), :2].tolist()), [2.0, 3.0])

    def test_conservation_detects_corruption(self) -> None:
        case = self.suite["cases"][0]
        vocabulary = V3Vocabulary.from_cases([case])
        actions = case["legal_actions"]["actions"][:2]
        batch = collate_factorized_batch(tensorize_cases([case], vocabulary), [actions], [[2, 1]])
        bad_counts = batch.prefixes.visit_counts.clone()
        root = int(torch.nonzero(~batch.prefixes.prefix.mask.any(1)).flatten()[0])
        bad_counts[root, 0] += 1
        with self.assertRaises(FactorizedPolicyError):
            assert_visit_conservation(replace(batch.prefixes, visit_counts=bad_counts))


if __name__ == "__main__":
    unittest.main()

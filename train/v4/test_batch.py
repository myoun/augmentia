from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from v3.factorized import FactorizedPolicyError
from v3.tensorize import V3Vocabulary, tensorize_cases

import v4.loss as loss_module
from v4.batch import (
    FACTOR_FIELD_IDS,
    PrefixBatchV4,
    assert_visit_conservation,
    collate_factorized_batch,
    encode_factor_edge,
    tensorize_factorized_cases,
)
from v4.loss import (
    conditional_prefix_cross_entropy,
    mixed_row_all_legal_shadow_cross_entropy,
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
        batch = tensorize_factorized_cases(
            cases,
            vocabulary,
            visits,
            opened_action_masks=[[True] * len(visit) for visit in visits],
        )
        self.assertEqual(batch.batch_size, 2)
        self.assertTrue(batch.prefixes.validated)
        self.assertTrue(batch.prefixes.to("cpu").validated)
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
        opened = collate_factorized_batch(
            states, [actions], [[7, 0, 1]], opened_action_masks=[[True, True, True]]
        )
        closed = collate_factorized_batch(
            states, [actions], [[7, 0, 1]], opened_action_masks=[[True, False, True]]
        )
        # The family prefix is the only three-way offer decision.
        row_a = int(torch.nonzero(opened.prefixes.legal_mask.sum(1) == 3).flatten()[0])
        row_b = int(torch.nonzero(closed.prefixes.legal_mask.sum(1) == 3).flatten()[0])
        self.assertEqual(int(opened.prefixes.opened_mask[row_a].sum()), 3)
        self.assertEqual(int(closed.prefixes.opened_mask[row_b].sum()), 2)
        self.assertTrue(torch.equal(opened.prefixes.visit_counts[row_a], closed.prefixes.visit_counts[row_b]))

    def test_wide_mixed_row_excludes_closed_zero_visit_siblings(self) -> None:
        # Use a wide same-family action family so the mixed row has 32 direct
        # children (the production corpus has 28+ legal actions and 16 opened).
        case = self.suite["cases"][0]
        actions = [
            {
                "kind": "card",
                "card": index,
                "definition_id": f"fixture-card-{index}",
                "target": {"kind": "none"},
            }
            for index in range(32)
        ]
        visits = [1] * 15 + [0] * (len(actions) - 15)
        opened = [True] * 16 + [False] * (len(actions) - 16)
        vocabulary = V3Vocabulary.from_cases([case])
        states = tensorize_cases([case], vocabulary)
        batch = collate_factorized_batch(
            states,
            [actions],
            [visits],
            opened_action_masks=[opened],
        )
        row_candidates = torch.nonzero(
            (batch.prefixes.legal_mask.sum(dim=1) >= 28)
            & (batch.prefixes.opened_mask.sum(dim=1) == 16)
        ).flatten()
        self.assertGreaterEqual(len(row_candidates), 1)
        self.assertEqual(int(batch.prefixes.complete_action_opened_mask[0].sum()), 16)
        row = int(row_candidates[0])
        self.assertTrue(bool(batch.prefixes.opened_mask[row, 0]))
        self.assertTrue(bool((batch.prefixes.visit_counts[row] == 0).logical_and(
            batch.prefixes.opened_mask[row]
        ).any()))
        self.assertTrue(bool((batch.prefixes.visit_counts[row] == 0).logical_and(
            ~batch.prefixes.opened_mask[row]
        ).any()))
        logits = torch.randn_like(batch.prefixes.visit_counts, dtype=torch.float32)
        baseline = conditional_prefix_cross_entropy(logits, batch.prefixes)
        changed = logits.clone()
        closed_columns = ~batch.prefixes.opened_mask[row]
        changed[row, closed_columns] += 1000.0
        self.assertTrue(
            torch.equal(
                baseline,
                conditional_prefix_cross_entropy(changed, batch.prefixes),
            )
        )

    def test_missing_production_mask_fails_closed(self) -> None:
        case = self.suite["cases"][0]
        states = tensorize_cases([case], V3Vocabulary.from_cases([case]))
        with self.assertRaisesRegex(FactorizedPolicyError, "opened_action_masks"):
            collate_factorized_batch(states, [case["legal_actions"]["actions"]], [[1]])

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
            {"kind": "move", "piece": 0, "from": 55, "to": 63, "promotion": None},
            {"kind": "move", "piece": 0, "from": 55, "to": 63, "promotion": 5},
        ]
        batch = collate_factorized_batch(
            states, [actions], [[2, 3]], opened_action_masks=[[True, True]]
        )
        two_way = torch.nonzero(batch.prefixes.legal_mask.sum(1) == 2).flatten()
        self.assertEqual(len(two_way), 1)
        self.assertEqual(sorted(batch.prefixes.visit_counts[int(two_way[0]), :2].tolist()), [2.0, 3.0])

    def test_losses_reject_prefixes_that_bypass_collation_validation(self) -> None:
        self.assertIs(PrefixBatchV4.__dataclass_fields__["validated"].default, False)
        case = self.suite["cases"][0]
        vocabulary = V3Vocabulary.from_cases([case])
        actions = case["legal_actions"]["actions"][:2]
        batch = collate_factorized_batch(
            tensorize_cases([case], vocabulary),
            [actions],
            [[2, 1]],
            opened_action_masks=[[True, True]],
        )
        unvalidated = replace(batch.prefixes, validated=False)
        logits = torch.zeros_like(unvalidated.visit_counts, dtype=torch.float32)
        for loss in (
            conditional_prefix_cross_entropy,
            mixed_row_all_legal_shadow_cross_entropy,
        ):
            with self.subTest(loss=loss.__name__):
                with self.assertRaisesRegex(
                    FactorizedPolicyError,
                    "validated by collate_factorized_batch",
                ):
                    loss(logits, unvalidated)

    def test_losses_do_not_repeat_conservation_scan(self) -> None:
        case = self.suite["cases"][0]
        vocabulary = V3Vocabulary.from_cases([case])
        actions = case["legal_actions"]["actions"][:2]
        with patch(
            "v4.batch.assert_visit_conservation",
            wraps=assert_visit_conservation,
        ) as conservation_check:
            batch = collate_factorized_batch(
                tensorize_cases([case], vocabulary),
                [actions],
                [[2, 1]],
                opened_action_masks=[[True, True]],
            )
            logits = torch.zeros_like(
                batch.prefixes.visit_counts,
                dtype=torch.float32,
            )
            conditional_prefix_cross_entropy(logits, batch.prefixes)
            mixed_row_all_legal_shadow_cross_entropy(logits, batch.prefixes)

        self.assertEqual(conservation_check.call_count, 1)
        self.assertNotIn("assert_visit_conservation", vars(loss_module))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_losses_do_not_force_cuda_synchronization(self) -> None:
        case = self.suite["cases"][0]
        vocabulary = V3Vocabulary.from_cases([case])
        actions = case["legal_actions"]["actions"][:2]
        batch = collate_factorized_batch(
            tensorize_cases([case], vocabulary),
            [actions],
            [[2, 1]],
            opened_action_masks=[[True, True]],
        ).to("cuda")
        logits = torch.zeros_like(batch.prefixes.visit_counts, dtype=torch.float32)

        torch.cuda.set_sync_debug_mode("error")
        try:
            conditional_prefix_cross_entropy(logits, batch.prefixes)
            mixed_row_all_legal_shadow_cross_entropy(logits, batch.prefixes)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_conservation_detects_corruption(self) -> None:
        case = self.suite["cases"][0]
        vocabulary = V3Vocabulary.from_cases([case])
        actions = case["legal_actions"]["actions"][:2]
        batch = collate_factorized_batch(
            tensorize_cases([case], vocabulary),
            [actions],
            [[2, 1]],
            opened_action_masks=[[True, True]],
        )
        bad_counts = batch.prefixes.visit_counts.clone()
        root = int(torch.nonzero(~batch.prefixes.prefix.mask.any(1)).flatten()[0])
        bad_counts[root, 0] += 1
        with self.assertRaises(FactorizedPolicyError):
            assert_visit_conservation(replace(batch.prefixes, visit_counts=bad_counts))


if __name__ == "__main__":
    unittest.main()

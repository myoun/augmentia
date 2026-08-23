from __future__ import annotations

import copy
import unittest

from v3.schema import (
    SchemaError,
    load_golden_suite,
    load_runtime_vocabulary,
    validate_golden_suite,
)


class GoldenV3ParityTests(unittest.TestCase):
    def test_rust_runtime_vocabulary_covers_structural_record_names(self) -> None:
        vocabulary = load_runtime_vocabulary()
        self.assertIn("square", vocabulary["tokens"])
        self.assertIn("middle-extra-move", vocabulary["tokens"])
        self.assertEqual(vocabulary["piece_kind_codes"], list(range(1, 59)))

    def test_checked_in_rust_fixture_passes_python_validation(self) -> None:
        suite = load_golden_suite()
        self.assertEqual(
            [case["name"] for case in suite["cases"]],
            [
                "initial_draft",
                "standard_playing",
                "forced_move",
                "forced_effect_move",
                "quantum_shadow",
                "awaiting_choice",
                "runtime_rich",
                "promotion",
                "terminal_draw",
            ],
        )

    def test_fixture_covers_every_action_descriptor_kind(self) -> None:
        suite = load_golden_suite()
        kinds = {
            action["kind"]
            for case in suite["cases"]
            for action in case["legal_actions"]["actions"]
        }
        self.assertEqual(
            kinds,
            {"move", "piece_ability", "card", "draft_pick", "resolve_choice"},
        )

    def test_promotion_and_terminal_cases_are_explicit(self) -> None:
        suite = load_golden_suite()
        cases = {case["name"]: case for case in suite["cases"]}
        promotions = {
            action["promotion"]
            for action in cases["promotion"]["legal_actions"]["actions"]
            if action["kind"] == "move" and action["promotion"] is not None
        }
        self.assertEqual(promotions, {2, 3, 4, 5})
        self.assertEqual(cases["terminal_draw"]["legal_actions"]["actions"], [])

    def test_choice_display_labels_are_not_model_tokens(self) -> None:
        suite = load_golden_suite()
        choice = next(case for case in suite["cases"] if case["name"] == "awaiting_choice")
        self.assertEqual(
            choice["observation"]["phase"]["options"],
            ["choice:trolley:0", "choice:trolley:1"],
        )
        action_options = [
            action["option"]
            for action in choice["legal_actions"]["actions"]
            if action["kind"] == "resolve_choice"
        ]
        self.assertEqual(action_options, ["choice:trolley:0", "choice:trolley:1"])

        corrupted = copy.deepcopy(suite)
        corrupted_choice = next(
            case for case in corrupted["cases"] if case["name"] == "awaiting_choice"
        )
        corrupted_choice["observation"]["phase"]["options"][0] = "희생: Pawn + Knight"
        with self.assertRaisesRegex(SchemaError, "stable token"):
            validate_golden_suite(corrupted)

    def test_shape_drift_is_rejected(self) -> None:
        suite = load_golden_suite()
        corrupted = copy.deepcopy(suite)
        corrupted["cases"][0]["observation"]["spatial_shape"] = [64, 8, 8]
        with self.assertRaisesRegex(SchemaError, "spatial_shape"):
            validate_golden_suite(corrupted)

    def test_invalid_local_reference_is_rejected(self) -> None:
        suite = load_golden_suite()
        corrupted = copy.deepcopy(suite)
        runtime = corrupted["cases"][0]["observation"]["runtime"]
        runtime[0]["value"] = {"type": "piece", "value": 999999}
        with self.assertRaisesRegex(SchemaError, "local slot"):
            validate_golden_suite(corrupted)

    def test_unknown_fields_are_rejected_instead_of_ignored(self) -> None:
        suite = load_golden_suite()
        corrupted = copy.deepcopy(suite)
        corrupted["cases"][0]["observation"]["legacy_fallback"] = True
        with self.assertRaisesRegex(SchemaError, "extra=.*legacy_fallback"):
            validate_golden_suite(corrupted)


if __name__ == "__main__":
    unittest.main()

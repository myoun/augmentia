from __future__ import annotations

import unittest

from v3.promotion import (
    _arena_metrics,
    _coverage_metrics,
    _run_gate,
    _threshold_coverage,
    _validate_promotion_scale,
    _wilson_interval,
)


class PromotionGateV3Tests(unittest.TestCase):
    def test_arena_excludes_limits_and_requires_exact_seed_color_pairs(self) -> None:
        report = {
            "report_schema_id": "augmentia.arena.v3",
            "budget_kind": "simulations",
            "simulations": 8,
            "pairs": 2,
            "games": [
                _arena_game(0, 10, "white", "candidate_win", True, 8),
                _arena_game(0, 10, "black", "draw", True, 8),
                _arena_game(1, 11, "white", "limit", False, 8),
                _arena_game(1, 11, "black", "opponent_win", True, 8),
            ],
        }
        metrics = _arena_metrics(report)
        self.assertEqual(metrics["terminal_games"], 3)
        self.assertEqual(metrics["terminal_rate"], 0.75)
        self.assertEqual(metrics["score"], 0.5)
        self.assertTrue(metrics["simulation_budget_exact"])
        self.assertEqual(metrics["paired_seed_count"], 2)

    def test_inexact_simulation_budget_is_detected(self) -> None:
        report = {
            "report_schema_id": "augmentia.arena.v3",
            "budget_kind": "simulations",
            "simulations": 8,
            "pairs": 1,
            "games": [
                _arena_game(0, 10, "white", "draw", True, 7),
                _arena_game(0, 10, "black", "draw", True, 8),
            ],
        }
        self.assertFalse(_arena_metrics(report)["simulation_budget_exact"])

    def test_coverage_reports_explicit_missing_contract_regions(self) -> None:
        record = {
            "record_schema_id": "augmentia.training.decision.v3",
            "observation": {
                "phase": {"kind": "playing"},
                "pieces": [{"kind_code": 1}],
                "cards": [{"definition_id": "sacrifice"}],
                "runtime": [{"field_id": 7, "value": {"type": "bool", "value": True}}],
            },
            "legal_actions": {"actions": [{"kind": "move"}]},
        }
        metrics = _coverage_metrics([record])
        result = _threshold_coverage(
            metrics,
            {
                "required_phases": ["playing", "draft"],
                "required_action_kinds": ["move", "use_card"],
                "required_runtime_field_ids": [7, 8],
                "piece_kind_count_min": 1,
                "card_definition_count_min": 1,
            },
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_phases"], ["draft"])
        self.assertEqual(result["missing_action_kinds"], ["use_card"])
        self.assertEqual(result["missing_runtime_field_ids"], [8])

    def test_wilson_interval_and_gate_errors_are_fail_closed(self) -> None:
        lower, upper = _wilson_interval(50, 100)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        gate = _run_gate("broken", lambda: 1 / 0)
        self.assertFalse(gate["passed"])
        self.assertIn("ZeroDivisionError", gate["error"])

    def test_promotion_scale_cannot_be_weakened_to_smoke_budgets(self) -> None:
        with self.assertRaisesRegex(ValueError, "promotion threshold"):
            _validate_promotion_scale(
                {
                    "arena": {"pairs": 1, "simulations": 1, "move_time_ms": 1},
                    "thresholds": {
                        "tactical_positions_min": 1,
                        "tactical_top1_accuracy_min": 0.0,
                        "tactical_optimal_mass_min": 0.0,
                        "calibration_positions_min": 1,
                        "arena_terminal_games_min": 1,
                        "arena_terminal_rate_min": 0.0,
                        "champion_score_min": 0.0,
                        "champion_score_lcb_min": 0.0,
                        "v2_score_min": 0.0,
                        "v2_score_lcb_min": 0.0,
                        "value_rmse_max": 2.0,
                        "value_ece_max": 1.0,
                        "arena_color_score_gap_max": 1.0,
                        "arena_card_color_tv_max": 1.0,
                    },
                }
            )


def _arena_game(
    pair: int,
    seed: int,
    color: str,
    outcome: str,
    terminal: bool,
    simulations: int,
) -> dict:
    return {
        "pair_index": pair,
        "seed": seed,
        "candidate_color": color,
        "outcome": outcome,
        "terminal": terminal,
        "candidate": {"searches": 1, "simulations": simulations, "elapsed_ms": 10.0},
        "opponent": {"searches": 1, "simulations": 8, "elapsed_ms": 10.0},
        "candidate_cards": ["sacrifice"],
        "opponent_cards": ["parry"],
    }

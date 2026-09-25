from __future__ import annotations

import unittest

from v4.phase1_ab import verify_payload_seal
from v4.phase1_shadow_ablation import (
    DEFAULT_LAMBDAS,
    ShadowAblationConfig,
    ShadowAblationError,
    _cell_id,
    _metrics,
    _summary,
)


class ShadowAblationTests(unittest.TestCase):
    def test_config_is_bounded_and_fails_closed(self) -> None:
        config = ShadowAblationConfig(device="cpu", minimum_game_blocks=8)
        config.validate()
        self.assertEqual(config.lambdas, DEFAULT_LAMBDAS)
        with self.assertRaisesRegex(ShadowAblationError, "at least one"):
            ShadowAblationConfig(lambdas=()).validate()
        with self.assertRaisesRegex(ShadowAblationError, "unique"):
            ShadowAblationConfig(lambdas=(0.0, 0.0)).validate()
        with self.assertRaisesRegex(ShadowAblationError, "non-negative"):
            ShadowAblationConfig(lambdas=(-0.1,)).validate()
        with self.assertRaisesRegex(ShadowAblationError, "at least 8"):
            ShadowAblationConfig(minimum_game_blocks=7).validate()

    def test_cell_id_and_config_json_are_deterministic(self) -> None:
        self.assertEqual(_cell_id(0.05), "lambda-5.00000000e-02")
        config = ShadowAblationConfig(device="cpu")
        self.assertEqual(config.to_json(), config.to_json())
        self.assertEqual(config.trainer_config(0.05).__dict__["mixed_row_shadow_lambda"], 0.05)

    def test_summary_uses_zero_baseline_and_seals(self) -> None:
        metrics = {
            "conditional_kl": 0.7,
            "opened_support_soft_nll": 2.7,
            "opened_support_all_modal_top1": 0.1,
            "opened_support_fractional_modal_score": 0.05,
            "opened_support_tail_macro": 0.2,
            "opened_support_tail_micro": 0.1,
            "opened_support_all_modal_ece": 0.2,
            "opened_support_fractional_ece": 0.25,
        }
        plan = {"plan_sha256": "a" * 64, "source_corpus_sha256": "b" * 64, "split_sha256": "c" * 64, "initialization_sha256": "d" * 64}
        results = [
            {"cell_id": "lambda-0", "lambda": 0.0, "metrics": metrics},
            {"cell_id": "lambda-0.05", "lambda": 0.05, "metrics": {**metrics, "conditional_kl": 0.6}},
        ]
        summary = _summary(plan, results, evidence=False)
        self.assertEqual(summary["schema_id"], "augmentia.v4-phase1-shadow-ablation-summary.v1")
        self.assertFalse(summary["evidence_sufficient"])
        self.assertFalse(summary["promotable"])
        self.assertEqual(summary["rows"][0]["improvement_vs_lambda0"]["conditional_kl"], 0.0)
        self.assertAlmostEqual(summary["rows"][1]["improvement_vs_lambda0"]["conditional_kl"], 0.1)
        verify_payload_seal(summary, "summary_sha256")

    def test_summary_requires_zero_baseline(self) -> None:
        plan = {"plan_sha256": "a" * 64, "source_corpus_sha256": "b" * 64, "split_sha256": "c" * 64, "initialization_sha256": "d" * 64}
        with self.assertRaisesRegex(ShadowAblationError, "lambda=0"):
            _summary(plan, [{"cell_id": "x", "lambda": 0.1, "metrics": {}}], evidence=False)

    def test_metrics_requires_p0_primary_fields(self) -> None:
        with self.assertRaises(KeyError):
            _metrics({"conditional_kl": 1.0})


if __name__ == "__main__":
    unittest.main()

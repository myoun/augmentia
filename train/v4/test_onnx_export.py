from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from v3.tensorize import V3Vocabulary
from v3.schema import load_runtime_vocabulary

from v4.batch import FACTOR_FIELD_IDS, tensorize_factorized_cases
from v4.model import AugmentNetV4, V4ModelConfig
from v4.onnx_export import (
    V4_BUNDLE_SCHEMA_ID,
    assert_factorized_onnx_equivalent,
    export_factorized_policy_bundle_v4,
)


FIXTURE = Path(__file__).parents[1] / "v3" / "fixtures" / "observation_action_v3.json"


class _Identity:
    def to_json(self) -> dict[str, Any]:
        return {
            "ruleset_id": "augment_chess_20260818",
            "ruleset_manifest_sha256": "0" * 64,
        }


class FactorizedOnnxExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        suite = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.cases = suite["cases"][:2]
        cls.vocabulary = V3Vocabulary.from_cases(cls.cases, extra_tokens=load_runtime_vocabulary()["tokens"])

    def _factorized(self):
        visits = [
            [index + 1 for index in range(len(case["legal_actions"]["actions"]))]
            for case in self.cases
        ]
        return tensorize_factorized_cases(self.cases, self.vocabulary, visits)

    def _model(self) -> AugmentNetV4:
        torch.manual_seed(11)
        return AugmentNetV4(
            V4ModelConfig(
                text_vocabulary_size=self.vocabulary.size,
                channels=8,
                residual_blocks=1,
                model_dim=16,
                runtime_message_passes=1,
            )
        ).eval()

    def test_bundle_rejects_incomplete_runtime_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing required runtime tokens"):
                export_factorized_policy_bundle_v4(
                    self._model(), self._factorized().states, self._factorized(),
                    Path(directory) / "invalid.onnx", V3Vocabulary.from_cases(self.cases),
                    _Identity(),
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_bundle_exports_and_matches_eager_scoring(self) -> None:
        model = self._model()
        factorized = self._factorized()
        states = factorized.states
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v4-model.onnx"
            _, metadata_path = export_factorized_policy_bundle_v4(
                model,
                states,
                factorized,
                output,
                self.vocabulary,
                _Identity(),
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["metadata_schema_id"], V4_BUNDLE_SCHEMA_ID)
            self.assertEqual(
                metadata["factorized_policy_schema_id"],
                "augmentia.factorized-policy.v4",
            )
            self.assertEqual(
                metadata["factorized_scorer"]["inputs"][0]["name"], "state_embedding"
            )
            self.assertEqual(
                [item["name"] for item in metadata["factorized_scorer"]["outputs"]],
                ["prefix_logits"],
            )
            self.assertTrue((output.with_suffix(".state-encoder.onnx")).is_file())
            self.assertTrue((output.with_suffix(".factorized-scorer.onnx")).is_file())
            errors = assert_factorized_onnx_equivalent(model, states, factorized, metadata_path)
            self.assertLess(errors["value_max_abs_error"], 2e-4)
            self.assertLess(errors["prefix_max_abs_error"], 2e-4)
            self.assertTrue(errors["illegal_is_min"])

    def test_onnx_accepts_high_non_promotion_indexes_after_export(self) -> None:
        model = self._model()
        sample = self._factorized()
        with tempfile.TemporaryDirectory() as directory:
            _, metadata_path = export_factorized_policy_bundle_v4(
                model, sample.states, sample, Path(directory) / "v4-model.onnx",
                self.vocabulary, _Identity(),
            )
            # Change indexes after tracing so this also covers ONNX Gather,
            # rather than only validating the eager export sample.
            for field, index in (("destination", 63), ("target_square", 63), ("piece", 64), ("card", 65)):
                with self.subTest(field=field, index=index):
                    factorized = self._factorized()
                    for edges in (factorized.prefixes.prefix, factorized.prefixes.children):
                        edges.field[edges.mask] = FACTOR_FIELD_IDS[field]
                        edges.category[edges.mask] = 0
                        edges.index[edges.mask] = index
                    factorized.states.piece_slot[:, 0] = 64
                    factorized.states.card_slot[:, 0] = 65
                    errors = assert_factorized_onnx_equivalent(
                        model, factorized.states, factorized, metadata_path,
                    )
                    self.assertLess(errors["value_max_abs_error"], 2e-4)
                    self.assertLess(errors["prefix_max_abs_error"], 2e-4)
                    self.assertTrue(errors["illegal_is_min"])

    def test_bundle_fails_closed_on_schema_and_hash_mismatch(self) -> None:
        model = self._model()
        factorized = self._factorized()
        states = factorized.states
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v4-model.onnx"
            _, metadata_path = export_factorized_policy_bundle_v4(
                model,
                states,
                factorized,
                output,
                self.vocabulary,
                _Identity(),
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["metadata_schema_id"] = "stale"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "V4 factorized policy bundle"):
                assert_factorized_onnx_equivalent(model, states, factorized, metadata_path)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["metadata_schema_id"] = V4_BUNDLE_SCHEMA_ID
            metadata["factorized_scorer"]["sha256"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scorer graph hash"):
                assert_factorized_onnx_equivalent(model, states, factorized, metadata_path)


if __name__ == "__main__":
    unittest.main()

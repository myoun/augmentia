from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import torch

from v3.tensorize import V3Vocabulary

from v4.batch import FACTOR_FIELD_IDS, tensorize_factorized_cases
from v4.loss import conditional_prefix_cross_entropy
from v4.model import AugmentNetV4, V4ModelConfig


FIXTURE = Path(__file__).parents[1] / "v3" / "fixtures" / "observation_action_v3.json"


class FactorizedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        suite = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.case = suite["cases"][0]
        cls.vocabulary = V3Vocabulary.from_cases([cls.case])

    def _batch(self, visits: list[int]):
        case = {**self.case, "legal_actions": {**self.case["legal_actions"], "actions": self.case["legal_actions"]["actions"][:len(visits)]}}
        return tensorize_factorized_cases([case], self.vocabulary, [visits])

    def _model(self) -> AugmentNetV4:
        torch.manual_seed(7)
        return AugmentNetV4(
            V4ModelConfig(
                text_vocabulary_size=self.vocabulary.size,
                channels=8,
                residual_blocks=1,
                model_dim=16,
                runtime_message_passes=1,
            )
        )

    def test_forward_is_conditional_not_flat(self) -> None:
        batch = self._batch([4, 2, 1])
        model = self._model().eval()
        with torch.no_grad():
            output = model(batch)
            encoding = model.encode_state(batch.states)
            split_logits = model.score_prefixes(batch.prefixes, batch.states, encoding)
        self.assertEqual(output.prefix_logits.shape, batch.prefixes.legal_mask.shape)
        self.assertEqual(output.value.shape, (1,))
        self.assertTrue(torch.equal(output.prefix_logits, split_logits))
        self.assertTrue(torch.isfinite(output.prefix_logits[batch.prefixes.legal_mask]).all())
        self.assertTrue((output.prefix_logits[~batch.prefixes.legal_mask] < -1e20).all())

    def test_non_promotion_indexes_do_not_index_piece_kind_embedding(self) -> None:
        model = self._model().eval()
        for field, index in (("source_square", 63), ("destination", 63), ("target_square", 63), ("piece", 64), ("card", 65)):
            with self.subTest(field=field, index=index):
                batch = self._batch([4, 2, 1])
                # Exercise both prefix and child embeddings with valid typed
                # indexes outside the 59-row piece-kind vocabulary.
                for edges in (batch.prefixes.prefix, batch.prefixes.children):
                    edges.field[edges.mask] = FACTOR_FIELD_IDS[field]
                    edges.category[edges.mask] = 0
                    edges.index[edges.mask] = index
                batch.states.piece_slot[:, 0] = 64
                batch.states.card_slot[:, 0] = 65
                with torch.no_grad():
                    output = model(batch)
                self.assertTrue(torch.isfinite(output.prefix_logits[batch.prefixes.legal_mask]).all())
                self.assertTrue(torch.isfinite(output.value).all())

    def test_zero_logits_equal_uniform_complete_path_cross_entropy(self) -> None:
        batch = self._batch([3, 1])
        logits = torch.zeros_like(batch.prefixes.visit_counts, dtype=torch.float32)
        loss = conditional_prefix_cross_entropy(logits, batch.prefixes)
        expected = math.log(2.0)
        self.assertAlmostEqual(float(loss), expected, places=6)

    def test_tiny_factorized_scorer_overfits(self) -> None:
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            batch = self._batch([32, 0, 0])
            model = self._model()
            model.state_encoder.eval()
            for parameter in model.state_encoder.parameters():
                parameter.requires_grad_(False)
            with torch.no_grad():
                encoding = model.encode_state(batch.states)
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=0.03)
            with torch.no_grad():
                initial = float(
                    conditional_prefix_cross_entropy(
                        model.score_prefixes(batch.prefixes, batch.states, encoding),
                        batch.prefixes,
                    )
                )
            for _ in range(60):
                optimizer.zero_grad(set_to_none=True)
                logits = model.score_prefixes(batch.prefixes, batch.states, encoding)
                loss = conditional_prefix_cross_entropy(logits, batch.prefixes)
                loss.backward()
                optimizer.step()
            final = float(
                conditional_prefix_cross_entropy(
                    model.score_prefixes(batch.prefixes, batch.states, encoding),
                    batch.prefixes,
                ).detach()
            )
            self.assertLess(final, initial * 0.05)
            self.assertLess(final, 0.05)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()

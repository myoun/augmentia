from __future__ import annotations

import copy
from dataclasses import asdict, fields, replace
import unittest

import torch

from v3.model import AugmentNetV3
from v3.schema import load_golden_suite
from v3.tensorize import V3Vocabulary, tensorize_cases
from v4.batch import FACTOR_FIELD_IDS, tensorize_factorized_cases
from v4.loss import conditional_prefix_cross_entropy
from v4.model import AugmentNetV4, V4ModelConfig


class HybridEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.suite = load_golden_suite()
        cls.vocabulary = V3Vocabulary.from_golden_suite(cls.suite)
        cls.cases = [case for case in cls.suite["cases"] if case["legal_actions"]["actions"]]
        cls.rich = next(case for case in cls.cases if case["name"] == "runtime_rich")

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.old_threads)

    def _config(self, **changes):
        return V4ModelConfig(**dict(
            text_vocabulary_size=self.vocabulary.size, channels=8,
            residual_blocks=1, model_dim=16, runtime_message_passes=1,
            **changes,
        ))

    def _model(self):
        torch.manual_seed(83)
        return AugmentNetV4(self._config(encoder_kind="hybrid", hybrid_layers=2, hybrid_heads=4))

    def _batch(self, cases=None):
        cases = self.cases if cases is None else cases
        visits = [[i + 1 for i in range(len(case["legal_actions"]["actions"]))] for case in cases]
        return tensorize_factorized_cases(cases, self.vocabulary, visits)

    def test_baseline_remains_default_and_checkpoint_compatible(self):
        torch.manual_seed(13)
        default = AugmentNetV4(self._config())
        torch.manual_seed(13)
        explicit = AugmentNetV4(self._config(encoder_kind="baseline"))
        self.assertIs(type(default.state_encoder), AugmentNetV3)
        self.assertEqual(default.config.encoder_kind, "baseline")
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        explicit.load_state_dict(default.state_dict(), strict=True)
        for key, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(V4ModelConfig(**asdict(default.config)), default.config)
        # Old metadata has no encoder selection fields.
        old = asdict(default.config)
        for key in ("encoder_kind", "hybrid_layers", "hybrid_heads"):
            old.pop(key)
        self.assertEqual(V4ModelConfig(**old).encoder_kind, "baseline")

    def test_invalid_hybrid_config_and_injected_baseline_fail_closed(self):
        for changes in ({"encoder_kind": "unknown"}, {"encoder_kind": "hybrid", "hybrid_heads": 3},
                        {"encoder_kind": "hybrid", "hybrid_layers": 0},
                        {"encoder_kind": "hybrid", "hybrid_heads": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                AugmentNetV4(self._config(**changes))
        with self.assertRaises(ValueError):
            AugmentNetV4(self._config(encoder_kind="hybrid"), AugmentNetV3(self._config().v3_config()))
        with self.assertRaises(ValueError):
            AugmentNetV4(self._config(), self._model().state_encoder)

    def test_split_shapes_masks_and_determinism_across_public_phases(self):
        model, batch = self._model().eval(), self._batch()
        with torch.no_grad():
            encoded = model.encode_state(batch.states)
            output = model(batch)
            repeated = model(batch)
            split = model.score_prefixes(batch.prefixes, batch.states, encoded)
        self.assertEqual(encoded.state.shape, (len(self.cases), 16))
        self.assertEqual(encoded.board_tokens.shape, (len(self.cases), 64, 16))
        self.assertEqual(encoded.piece_entities.shape, (*batch.states.piece_mask.shape, 16))
        self.assertEqual(encoded.card_entities.shape, (*batch.states.card_mask.shape, 16))
        self.assertTrue(torch.equal(encoded.piece_entities[~batch.states.piece_mask], torch.zeros_like(encoded.piece_entities[~batch.states.piece_mask])))
        self.assertTrue(torch.equal(encoded.card_entities[~batch.states.card_mask], torch.zeros_like(encoded.card_entities[~batch.states.card_mask])))
        torch.testing.assert_close(output.prefix_logits, split, rtol=0, atol=0)
        torch.testing.assert_close(output.prefix_logits, repeated.prefix_logits, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output.prefix_logits[batch.prefixes.legal_mask]).all())
        self.assertTrue((output.prefix_logits[~batch.prefixes.legal_mask] == torch.finfo(torch.float32).min).all())
        self.assertTrue(torch.isfinite(output.value).all())
        self.assertTrue((output.value.abs() <= 1).all())
        self.assertEqual(output.forced_win_distance.shape, (len(self.cases),))

    def test_padding_content_and_batch_packing_do_not_change_encoding(self):
        model = self._model().eval()
        batch = tensorize_cases(self.cases, self.vocabulary)
        changed = copy.deepcopy(batch)
        # Valid but unrelated categorical IDs and numeric garbage under padding.
        for prefix in ("piece", "card"):
            mask = getattr(changed, prefix + "_mask")
            for item in fields(changed):
                if item.name.startswith(prefix + "_") and item.name not in (prefix + "_mask", prefix + "_slot"):
                    tensor = getattr(changed, item.name)
                    tensor[~mask] = 1
        with torch.no_grad():
            expected = model.encode_state(batch)
            actual = model.encode_state(changed)
            single = model.encode_state(tensorize_cases([self.rich], self.vocabulary))
        for item in fields(expected):
            torch.testing.assert_close(getattr(expected, item.name), getattr(actual, item.name), rtol=1e-5, atol=1e-6)
        rich_index = self.cases.index(self.rich)
        torch.testing.assert_close(expected.state[rich_index], single.state[0], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(expected.board_tokens[rich_index], single.board_tokens[0], rtol=1e-5, atol=2e-6)

    def test_added_padding_slots_do_not_change_valid_outputs(self):
        model = self._model().eval()
        batch = tensorize_cases([self.rich], self.vocabulary)
        tensors = {item.name: getattr(batch, item.name) for item in fields(batch)}
        for prefix, count in (("piece", 7), ("card", 5)):
            for name, tensor in list(tensors.items()):
                if name.startswith(prefix + "_"):
                    padding = torch.ones((tensor.shape[0], count, *tensor.shape[2:]), dtype=tensor.dtype)
                    if name == prefix + "_mask":
                        padding.zero_()
                    elif name == prefix + "_slot":
                        padding.fill_(9876)
                    tensors[name] = torch.cat((tensor, padding), dim=1)
        padded = replace(batch, **tensors)
        # entity_counts measures REAL records, not allocated padding width.
        torch.testing.assert_close(batch.entity_counts, padded.entity_counts, rtol=0, atol=0)
        with torch.no_grad():
            original, actual = model.encode_state(batch), model.encode_state(padded)
        torch.testing.assert_close(original.state, actual.state, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(original.board_tokens, actual.board_tokens, rtol=1e-5, atol=2e-6)
        for name, mask in (("piece_entities", batch.piece_mask), ("card_entities", batch.card_mask)):
            original_tokens, padded_tokens = getattr(original, name), getattr(actual, name)
            torch.testing.assert_close(original_tokens, padded_tokens[:, :mask.shape[1]], rtol=1e-5, atol=2e-6)
            self.assertEqual(int(torch.count_nonzero(padded_tokens[:, mask.shape[1]:])), 0)

    def test_attention_masks_keys_and_preserves_square_positions(self):
        from v4.hybrid import HybridAttentionBlock

        torch.manual_seed(23)
        block = HybridAttentionBlock(16, 4).eval()
        valid = torch.randn(2, 3, 16)
        padded = torch.cat((valid, torch.randn(2, 6, 16) * 1000), dim=1)
        with torch.no_grad():
            expected = block(valid, torch.ones(2, 3, dtype=torch.bool))
            actual = block(padded, torch.tensor([[True] * 3 + [False] * 6] * 2))
        torch.testing.assert_close(expected, actual[:, :3], rtol=1e-5, atol=2e-6)
        self.assertEqual(int(torch.count_nonzero(actual[:, 3:])), 0)
        encoder = self._model().state_encoder
        torch.testing.assert_close(encoder.hybrid_square_ids, torch.arange(1, 65), rtol=0, atol=0)
        square_tokens = encoder.square_embedding(encoder.hybrid_square_ids)
        self.assertEqual(torch.unique(square_tokens, dim=0).shape[0], 64)
        # With the CNN projection zeroed, different squares still have explicit
        # position information; tensor index 63 remains square 63, not a pool.
        with torch.no_grad():
            encoder.board_projection.weight.zero_()
            encoder.board_projection.bias.zero_()
            encoded = encoder.encode_state(tensorize_cases([self.rich], self.vocabulary))
        self.assertFalse(torch.allclose(encoded.board_tokens[:, 0], encoded.board_tokens[:, 63]))

    def test_entity_permutation_preserves_legal_scores(self):
        changed = copy.deepcopy(self.rich)
        changed["observation"]["pieces"].reverse()
        changed["observation"]["cards"].reverse()
        model = self._model().eval()
        with torch.no_grad():
            expected, actual = model(self._batch([self.rich])), model(self._batch([changed]))
        torch.testing.assert_close(expected.prefix_logits, actual.prefix_logits, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(expected.value, actual.value, rtol=2e-5, atol=2e-6)

    def test_public_branches_affect_encoding_but_actions_and_labels_do_not(self):
        model = self._model().eval()
        batch = tensorize_cases([self.rich], self.vocabulary)
        with torch.no_grad():
            expected = model.encode_state(batch).state
        for name in ("spatial", "global_features", "piece_numeric", "card_numeric", "runtime_numeric", "phase_numeric"):
            with self.subTest(branch=name):
                changed = copy.deepcopy(batch)
                getattr(changed, name).add_(0.25)
                with torch.no_grad():
                    actual = model.encode_state(changed).state
                self.assertGreater(float((actual - expected).abs().max()), 1e-6)
        changed = copy.deepcopy(batch)
        for item in fields(changed):
            if item.name.startswith("action_"):
                getattr(changed, item.name).zero_()
        with torch.no_grad():
            torch.testing.assert_close(expected, model.encode_state(changed).state, rtol=0, atol=0)

    def test_empty_entity_and_runtime_masks_are_finite(self):
        model = self._model().eval()
        batch = tensorize_cases([self.rich], self.vocabulary)
        for name in ("piece_mask", "card_mask", "runtime_mask", "phase_text_mask", "state_text_mask"):
            getattr(batch, name).zero_()
        with torch.no_grad():
            encoding = model.encode_state(batch)
        for item in fields(encoding):
            self.assertTrue(torch.isfinite(getattr(encoding, item.name)).all())
        self.assertEqual(int(torch.count_nonzero(encoding.piece_entities)), 0)
        self.assertEqual(int(torch.count_nonzero(encoding.card_entities)), 0)

    def test_policy_value_and_auxiliary_gradients_reach_hybrid_and_public_frontend(self):
        model, batch = self._model(), self._batch()
        self.assertGreater(int(torch.count_nonzero(batch.states.phase_numeric)), 0)
        output = model(batch)
        loss = conditional_prefix_cross_entropy(output.prefix_logits, batch.prefixes)
        loss = loss + (output.value - 0.4).square().mean() + (output.forced_win_distance - 2).square().mean()
        loss.backward()
        for name in ("state_encoder.hybrid_blocks.0.qkv.weight", "state_encoder.hybrid_blocks.1.feed_forward.0.weight",
                     "state_encoder.board_stem.0.weight", "state_encoder.runtime_numeric.0.weight",
                     "state_encoder.global_encoder.0.weight", "state_encoder.piece_numeric.0.weight",
                     "state_encoder.card_numeric.0.weight", "state_encoder.phase_numeric.0.weight",
                     "state_encoder.value_head.0.weight", "state_encoder.forced_win_distance_head.0.weight",
                     "prefix_scorer.0.weight"):
            with self.subTest(parameter=name):
                gradient = dict(model.named_parameters())[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)

    def test_optional_promotion_embedding_mask_is_preserved(self):
        model = self._model().eval()
        batch = self._batch([self.cases[0]])
        for edges in (batch.prefixes.prefix, batch.prefixes.children):
            edges.field[edges.mask] = FACTOR_FIELD_IDS["destination"]
            edges.category[edges.mask] = 0
            edges.index[edges.mask] = 63
        with torch.no_grad():
            output = model(batch)
        self.assertTrue(torch.isfinite(output.prefix_logits[batch.prefixes.legal_mask]).all())


if __name__ == "__main__":
    unittest.main()

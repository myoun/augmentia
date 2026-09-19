"""Actual CPU ONNX Runtime Hybrid parity, in temporary independent bundles only."""
from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import fields, replace
from pathlib import Path
import unittest

import numpy as np
import onnxruntime as ort
import torch

from v3.onnx_export import STATE_ENCODER_OUTPUT_NAMES, STATE_INPUT_NAMES
from v3.schema import load_golden_suite, load_runtime_vocabulary
from v3.tensorize import V3Vocabulary
from v4.batch import FACTOR_FIELD_IDS, V4Batch, tensorize_factorized_cases
from v4.model import AugmentNetV4, V4ModelConfig
from v4.onnx_export import (
    V4_FACTORIZED_SCORER_INPUT_NAMES,
    _scorer_trace_inputs,
    export_factorized_policy_bundle_v4,
)


class _Identity:
    def to_json(self):
        return {"ruleset_id": "augment_chess_20260818", "ruleset_manifest_sha256": "0" * 64}


def _padded_states(states):
    values = {item.name: getattr(states, item.name) for item in fields(states)}
    for prefix, count in (("piece", 5), ("card", 7)):
        for name, tensor in list(values.items()):
            if name.startswith(prefix + "_"):
                padding = torch.ones((tensor.shape[0], count, *tensor.shape[2:]), dtype=tensor.dtype)
                if name == prefix + "_mask":
                    padding.zero_()
                elif name == prefix + "_slot":
                    padding.fill_(9999)
                values[name] = torch.cat((tensor, padding), dim=1)
    return replace(states, **values)


class HybridOnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.suite = load_golden_suite()
        cls.cases = [case for case in cls.suite["cases"] if case["legal_actions"]["actions"]]
        cls.vocabulary = V3Vocabulary.from_cases(cls.cases, extra_tokens=load_runtime_vocabulary()["tokens"])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def _batch(self, cases):
        return tensorize_factorized_cases(cases, self.vocabulary, [
            [i + 1 for i in range(len(case["legal_actions"]["actions"]))] for case in cases
        ])

    def _model(self, kind):
        torch.manual_seed(29)
        return AugmentNetV4(V4ModelConfig(
            text_vocabulary_size=self.vocabulary.size, channels=8, residual_blocks=1,
            model_dim=16, runtime_message_passes=1, encoder_kind=kind,
            hybrid_layers=2, hybrid_heads=4,
        )).eval()

    def _sessions(self, path):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        return metadata, tuple(ort.InferenceSession(
            str(path.parent / metadata[key]["path"]), sess_options=options,
            providers=["CPUExecutionProvider"],
        ) for key in ("state_encoder", "factorized_scorer"))

    def _check_forward(self, model, batch, sessions, baseline_sessions):
        encoder, scorer = sessions
        feeds = {name: getattr(batch.states, name).numpy() for name in STATE_INPUT_NAMES}
        encoded = encoder.run(list(STATE_ENCODER_OUTPUT_NAMES), feeds)
        with torch.no_grad():
            eager_encoding = model.encode_state(batch.states)
            eager = model(batch)
        expected = (eager_encoding.state, eager_encoding.board_tokens,
                    eager_encoding.piece_entities, eager_encoding.card_entities, eager.value)
        for name, actual, tensor in zip(STATE_ENCODER_OUTPUT_NAMES, encoded, expected, strict=True):
            with self.subTest(output=name):
                self.assertEqual(actual.shape, tuple(tensor.shape))
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, tensor.numpy(), rtol=2e-4, atol=2e-4)
        trace_inputs = _scorer_trace_inputs(model, batch.states, batch)
        scorer_feeds = {name: value.numpy() for name, value in zip(
            V4_FACTORIZED_SCORER_INPUT_NAMES, trace_inputs, strict=True
        )}
        # End-to-end: feed actual exported encoder tensors, NOT eager cached tensors.
        scorer_feeds.update(dict(zip(STATE_ENCODER_OUTPUT_NAMES[:-1], encoded[:-1], strict=True)))
        (logits,) = scorer.run(["prefix_logits"], scorer_feeds)
        legal = batch.prefixes.legal_mask.numpy()
        np.testing.assert_allclose(logits[legal], eager.prefix_logits.numpy()[legal], rtol=2e-4, atol=2e-4)
        self.assertTrue(np.all(logits[~legal] == np.finfo(np.float32).min))
        self.assertEqual(logits.shape, tuple(batch.prefixes.legal_mask.shape))
        for output, mask in ((encoded[2], batch.states.piece_mask), (encoded[3], batch.states.card_mask)):
            self.assertTrue(np.all(output[~mask.numpy()] == 0))
        # Baseline and Hybrid graphs have identical concrete I/O shapes/dtypes,
        # but different learned numerical functions. Never require same values.
        baseline_encoded = baseline_sessions[0].run(list(STATE_ENCODER_OUTPUT_NAMES), feeds)
        for actual, baseline in zip(encoded, baseline_encoded, strict=True):
            self.assertEqual(actual.shape, baseline.shape)
            self.assertEqual(actual.dtype, baseline.dtype)
        baseline_feeds = {**scorer_feeds, **dict(zip(STATE_ENCODER_OUTPUT_NAMES[:-1], baseline_encoded[:-1], strict=True))}
        (baseline_logits,) = baseline_sessions[1].run(["prefix_logits"], baseline_feeds)
        self.assertEqual(logits.shape, baseline_logits.shape)
        self.assertEqual(logits.dtype, baseline_logits.dtype)
        return encoded

    def test_split_export_dynamic_padded_forward_and_baseline_io_contract(self):
        sample = self._batch(self.cases[:2])
        hybrid, baseline = self._model("hybrid"), self._model("baseline")
        with tempfile.TemporaryDirectory(prefix="augmentia-hybrid-") as directory:
            paths = {}
            for kind, model in (("baseline", baseline), ("hybrid", hybrid)):
                _, paths[kind] = export_factorized_policy_bundle_v4(
                    model, sample.states, sample, Path(directory) / (kind + ".onnx"),
                    self.vocabulary, _Identity(),
                )
            meta, sessions = self._sessions(paths["hybrid"])
            baseline_meta, baseline_sessions = self._sessions(paths["baseline"])
            self.assertEqual(meta["model_config"]["encoder_kind"], "hybrid")
            self.assertEqual(baseline_meta["model_config"]["encoder_kind"], "baseline")
            for graph in ("state_encoder", "factorized_scorer"):
                for direction in ("inputs", "outputs"):
                    self.assertEqual(meta[graph][direction], baseline_meta[graph][direction])
            # Batch/entity/runtime/prefix/child lengths all differ from tracing.
            rich = next(case for case in self.cases if case["name"] == "runtime_rich")
            variants = {"trace": sample, "all_phases": self._batch(self.cases),
                        "rich_single": self._batch([rich])}
            original = variants["rich_single"]
            variants["extra_padding"] = V4Batch(_padded_states(original.states), original.prefixes)
            empty = copy.deepcopy(original)
            empty.states.piece_mask.zero_()
            empty.states.card_mask.zero_()
            empty.states.runtime_mask.zero_()
            variants["empty_entity_masks"] = empty
            high = copy.deepcopy(original)
            for edges in (high.prefixes.prefix, high.prefixes.children):
                edges.field[edges.mask] = FACTOR_FIELD_IDS["destination"]
                edges.category[edges.mask] = 0
                edges.index[edges.mask] = 63
            variants["post_export_high_non_promotion"] = high
            results = {}
            for name, batch in variants.items():
                with self.subTest(variant=name):
                    results[name] = self._check_forward(hybrid, batch, sessions, baseline_sessions)
            for index in (0, 1, 4):
                np.testing.assert_allclose(results["rich_single"][index], results["extra_padding"][index], rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()

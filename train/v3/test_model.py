from __future__ import annotations

import copy
import unittest

import torch

from v3.model import AugmentNetV3, V3ModelConfig, compute_v3_loss
from v3.schema import SchemaError, load_golden_suite
from v3.tensorize import V3Vocabulary, tensorize_cases


class AugmentNetV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.suite = load_golden_suite()
        cls.vocabulary = V3Vocabulary.from_golden_suite(cls.suite)

    def test_dynamic_legal_actions_and_terminal_rows_are_masked(self) -> None:
        torch.manual_seed(7)
        batch = tensorize_cases(self.suite["cases"], self.vocabulary)
        model = self._small_model().eval()
        with torch.no_grad():
            output = model(batch)
        action_counts = [
            len(case["legal_actions"]["actions"]) for case in self.suite["cases"]
        ]
        self.assertEqual(
            output.policy_logits.shape, (len(action_counts), max(action_counts))
        )
        self.assertEqual(batch.action_mask.sum(dim=1).tolist(), action_counts)
        self.assertEqual(output.value.shape, (len(action_counts),))
        self.assertTrue(torch.isfinite(output.value).all())
        self.assertTrue(
            torch.all(
                output.policy_logits.masked_select(~batch.action_mask)
                == torch.finfo(output.policy_logits.dtype).min
            )
        )

    def test_default_parameter_and_vocabulary_baseline_is_fixed(self) -> None:
        model = AugmentNetV3(V3ModelConfig(text_vocabulary_size=self.vocabulary.size))
        self.assertEqual(self.vocabulary.size, 384)
        self.assertEqual(model.parameter_count(), 2_865_284)
        self.assertEqual(model.piece_kind_embedding.num_embeddings, 59)
        self.assertEqual(
            model.piece_kind_embedding(torch.arange(59)).shape,
            (59, model.config.model_dim),
        )
        self.assertEqual(
            V3Vocabulary.from_json(self.vocabulary.to_json()), self.vocabulary
        )

    def test_forward_is_deterministic_and_entity_order_invariant(self) -> None:
        case = copy.deepcopy(
            next(case for case in self.suite["cases"] if case["name"] == "runtime_rich")
        )
        reordered = copy.deepcopy(case)
        reordered["observation"]["pieces"].reverse()
        reordered["observation"]["cards"].reverse()
        torch.manual_seed(11)
        model = self._small_model().eval()
        original_batch = tensorize_cases([case], self.vocabulary)
        reordered_batch = tensorize_cases([reordered], self.vocabulary)
        with torch.no_grad():
            first = model(original_batch)
            second = model(original_batch)
            reordered_output = model(reordered_batch)
        torch.testing.assert_close(
            first.policy_logits, second.policy_logits, rtol=0, atol=0
        )
        torch.testing.assert_close(first.value, second.value, rtol=0, atol=0)
        torch.testing.assert_close(
            first.policy_logits, reordered_output.policy_logits, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            first.value, reordered_output.value, rtol=1e-5, atol=1e-6
        )

    def test_ordered_multi_square_targets_do_not_collapse(self) -> None:
        case = copy.deepcopy(
            next(case for case in self.suite["cases"] if case["name"] == "runtime_rich")
        )
        card_action = copy.deepcopy(
            next(
                action
                for action in case["legal_actions"]["actions"]
                if action["kind"] == "card"
            )
        )
        first = copy.deepcopy(card_action)
        second = copy.deepcopy(card_action)
        first["target"] = {"kind": "squares", "squares": [1, 9, 18]}
        second["target"] = {"kind": "squares", "squares": [18, 9, 1]}
        case["legal_actions"]["actions"] = [first, second]
        batch = tensorize_cases([case], self.vocabulary)
        self.assertTrue(
            torch.equal(
                batch.action_target_squares[0, 0], batch.action_target_squares[0, 1]
            )
        )
        self.assertFalse(
            torch.equal(
                batch.action_target_sequence[0, 0], batch.action_target_sequence[0, 1]
            )
        )

    def test_every_fixture_action_has_distinct_model_features(self) -> None:
        batch = tensorize_cases(self.suite["cases"], self.vocabulary)
        action_fields = [
            "action_kind",
            "action_piece_ref",
            "action_piece_kind",
            "action_from",
            "action_to",
            "action_promotion",
            "action_ability",
            "action_card_ref",
            "action_definition",
            "action_handler",
            "action_option",
            "action_target_kind",
            "action_target_squares",
            "action_target_sequence",
            "action_target_sequence_position",
            "action_target_sequence_mask",
            "action_numeric",
        ]
        for batch_index, case in enumerate(self.suite["cases"]):
            action_count = len(case["legal_actions"]["actions"])
            encoded = {
                b"".join(
                    getattr(batch, field)[batch_index, action_index]
                    .contiguous()
                    .numpy()
                    .tobytes()
                    for field in action_fields
                )
                for action_index in range(action_count)
            }
            self.assertEqual(len(encoded), action_count, case["name"])

    def test_unknown_text_is_rejected_instead_of_hashed(self) -> None:
        case = copy.deepcopy(
            next(
                case for case in self.suite["cases"] if case["name"] == "initial_draft"
            )
        )
        case["legal_actions"]["actions"][0]["definition_id"] = (
            "future-card-not-in-vocabulary"
        )
        with self.assertRaisesRegex(SchemaError, "unknown model vocabulary token"):
            tensorize_cases([case], self.vocabulary)

    def test_non_finite_structured_inputs_are_rejected(self) -> None:
        case = next(
            case for case in self.suite["cases"] if case["name"] == "standard_playing"
        )
        batch = tensorize_cases([case], self.vocabulary)
        batch.action_numeric[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "action_numeric contains NaN"):
            self._small_model()(batch)

    def test_value_and_policy_supervision_masks_are_independent(self) -> None:
        cases = [
            case for case in self.suite["cases"] if case["legal_actions"]["actions"]
        ][:3]
        batch = tensorize_cases(cases, self.vocabulary)
        model = self._small_model()
        output = model(batch)
        policy_targets = torch.zeros_like(output.policy_logits)
        policy_targets[:, 0] = 1.0
        policy_supervision = torch.tensor([True, True, False])
        policy_targets[~policy_supervision] = 0.0
        value_targets = torch.tensor([float("nan"), 1.0, -1.0])
        value_supervision = torch.tensor([False, True, True])
        losses = compute_v3_loss(
            output,
            batch.action_mask,
            policy_targets,
            policy_supervision,
            value_targets,
            value_supervision,
        )
        self.assertTrue(torch.isfinite(losses.total))
        self.assertEqual(losses.policy_samples, 2)
        self.assertEqual(losses.value_samples, 2)

    def test_tiny_fixture_can_be_intentionally_overfit(self) -> None:
        selected_names = {
            "initial_draft",
            "forced_move",
            "awaiting_choice",
            "promotion",
        }
        cases = [case for case in self.suite["cases"] if case["name"] in selected_names]
        batch = tensorize_cases(cases, self.vocabulary)
        torch.manual_seed(19)
        model = AugmentNetV3(
            V3ModelConfig(
                text_vocabulary_size=self.vocabulary.size,
                channels=16,
                residual_blocks=1,
                model_dim=32,
                runtime_message_passes=1,
            )
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=0.0)
        policy_targets = torch.zeros((len(cases), batch.action_mask.shape[1]))
        chosen = [
            index % len(case["legal_actions"]["actions"])
            for index, case in enumerate(cases)
        ]
        policy_targets[torch.arange(len(cases)), torch.tensor(chosen)] = 1.0
        policy_supervision = torch.ones(len(cases), dtype=torch.bool)
        value_targets = torch.tensor([-1.0, -0.25, 0.5, 1.0])
        value_supervision = torch.ones(len(cases), dtype=torch.bool)

        initial = None
        for _ in range(169):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = compute_v3_loss(
                output,
                batch.action_mask,
                policy_targets,
                policy_supervision,
                value_targets,
                value_supervision,
            )
            if initial is None:
                initial = float(losses.total.detach())
            losses.total.backward()
            optimizer.step()
        self.assertIsNotNone(initial)
        self.assertLess(float(losses.total.detach()), initial * 0.02)
        self.assertLess(float(losses.policy.detach()), 0.02)
        self.assertLess(float(losses.value.detach()), 0.01)

    def _small_model(self) -> AugmentNetV3:
        return AugmentNetV3(
            V3ModelConfig(
                text_vocabulary_size=self.vocabulary.size,
                channels=32,
                residual_blocks=2,
                model_dim=64,
                runtime_message_passes=2,
            )
        )


if __name__ == "__main__":
    unittest.main()

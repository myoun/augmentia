from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from v3.tensorize import V3Vocabulary
from v4.replay import FactorizedTrainingExampleV4
from v4.train import (
    POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT,
    V4Trainer,
    V4TrainingConfig,
)


FIXTURE = Path(__file__).parents[1] / "v3" / "fixtures" / "observation_action_v3.json"


class PauseResumeV4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        suite = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.case = suite["cases"][0]
        cls.vocabulary = V3Vocabulary.from_cases([cls.case])

    def _examples(self, count: int) -> list[FactorizedTrainingExampleV4]:
        action_count = len(self.case["legal_actions"]["actions"])
        visits = (8, *([0] * (action_count - 1)))
        return [
            FactorizedTrainingExampleV4(
                decision_id=f"decision-{index}",
                observation=self.case["observation"],
                legal_actions=self.case["legal_actions"],
                complete_visit_counts=visits,
                prefix_targets=(),
                opened_action_mask=(True, *([False] * (action_count - 1))),
                opened_masks=(),
                synthetic=True,
                value=1.0 if index % 2 == 0 else -1.0,
            )
            for index in range(count)
        ]

    def _config(self, batching_kind: str = "random") -> V4TrainingConfig:
        return V4TrainingConfig(
            learning_rate=1e-3,
            batch_size=1,
            epochs=1,
            value_weight=1.0,
            seed=1234,
            device="cpu",
            model_dim=16,
            channels=8,
            residual_blocks=1,
            runtime_message_passes=1,
            batching_kind=batching_kind,
        )

    def test_mid_epoch_pause_resume_matches_uninterrupted_training(self) -> None:
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            examples = self._examples(4)
            validation = self._examples(1)
            config = self._config()
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                continuous = V4Trainer(
                    examples,
                    root / "continuous",
                    config,
                    vocabulary=self.vocabulary,
                    validation_examples=validation,
                )
                continuous.fit()

                paused = V4Trainer(
                    examples,
                    root / "resumed",
                    config,
                    vocabulary=self.vocabulary,
                    validation_examples=validation,
                )
                original_step = paused.optimizer.step
                step_count = 0

                def step_and_pause(*args, **kwargs):
                    nonlocal step_count
                    result = original_step(*args, **kwargs)
                    step_count += 1
                    if step_count == 2:
                        paused.stop_requested = True
                    return result

                paused.optimizer.step = step_and_pause
                paused.fit()
                self.assertTrue(paused.paused)
                self.assertEqual(paused.completed_epochs, 0)
                self.assertIsNotNone(paused.epoch_progress)
                self.assertEqual(paused.epoch_progress["next_batch"], 2)
                self.assertTrue((root / "resumed" / "interrupt.pt").exists())

                resumed = V4Trainer.from_checkpoint(
                    root / "resumed" / "interrupt.pt",
                    examples,
                    config,
                    validation_examples=validation,
                    expected_vocabulary=self.vocabulary,
                )
                resumed.run_dir = root / "resumed"
                resumed.fit()

                self.assertFalse(resumed.paused)
                self.assertEqual(resumed.completed_epochs, 1)
                self.assertIsNone(resumed.epoch_progress)
                self.assertIsNone(resumed.pending_validation)
                self.assertEqual(len(resumed.history), 1)
                for name, expected in continuous.model.state_dict().items():
                    self.assertTrue(
                        torch.equal(expected, resumed.model.state_dict()[name]),
                        name,
                    )
                for section in ("train", "validation"):
                    for metric in (
                        "loss",
                        "policy_loss",
                        "policy_target_entropy",
                        "policy_kl",
                        "mixed_row_shadow_lambda",
                        "mixed_row_shadow_ce",
                        "mixed_row_shadow_weighted_loss",
                        "mixed_row_shadow_rows",
                        "value_loss",
                        "batches",
                        "examples",
                        "value_supervised_examples",
                    ):
                        self.assertEqual(
                            continuous.history[0][section][metric],
                            resumed.history[0][section][metric],
                        )
        finally:
            torch.set_num_threads(old_threads)

    def test_legacy_all_open_experiment_checkpoint_cannot_resume(self) -> None:
        opened = self._examples(1)
        all_open = [
            replace(
                example,
                opened_action_mask=(True,)
                * len(example.legal_actions["actions"]),
            )
            for example in opened
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trainer = V4Trainer(
                all_open,
                root,
                self._config(),
                vocabulary=self.vocabulary,
                validation_examples=all_open,
                policy_target_semantics_id=(
                    POLICY_TARGET_SEMANTICS_LEGACY_ALL_OPEN_EXPERIMENT
                ),
            )
            checkpoint, _sidecar = trainer.save_checkpoint(root / "experiment.pt")
            with self.assertRaisesRegex(ValueError, "opened-only"):
                V4Trainer.from_checkpoint(
                    checkpoint,
                    all_open,
                    self._config(),
                    validation_examples=all_open,
                    expected_vocabulary=self.vocabulary,
                )

    def test_bucketed_batches_group_similar_action_counts(self) -> None:
        actions = self.case["legal_actions"]["actions"]
        self.assertGreaterEqual(len(actions), 3)
        counts = [1, 3, 2, 1, 3, 2]
        examples = []
        for index, count in enumerate(counts):
            legal_actions = {**self.case["legal_actions"], "actions": actions[:count]}
            examples.append(
                FactorizedTrainingExampleV4(
                    decision_id=f"bucket-{index}",
                    observation=self.case["observation"],
                    legal_actions=legal_actions,
                    complete_visit_counts=(8, *([0] * (count - 1))),
                    prefix_targets=(),
                    opened_action_mask=(True, *([False] * (count - 1))),
                    opened_masks=(),
                    synthetic=True,
                    value=0.0,
                )
            )
        for batching_kind in ("bucketed", "cached_bucketed"):
            with self.subTest(batching_kind=batching_kind):
                config = V4TrainingConfig(
                    learning_rate=1e-3,
                    batch_size=2,
                    epochs=1,
                    value_weight=1.0,
                    seed=1234,
                    device="cpu",
                    model_dim=16,
                    channels=8,
                    residual_blocks=1,
                    runtime_message_passes=1,
                    batching_kind=batching_kind,
                    tensor_cache_max_batches=(2 if batching_kind == "cached_bucketed" else 128),
                )
                with tempfile.TemporaryDirectory() as temp:
                    trainer = V4Trainer(
                        examples,
                        Path(temp),
                        config,
                        vocabulary=self.vocabulary,
                    )
                    action_counts = []
                    for _batch_index, batch, _values in trainer._batches(
                        trainer.examples, shuffle=True, epoch=0
                    ):
                        counts_in_batch = batch.prefixes.complete_action_mask.sum(dim=1).tolist()
                        action_counts.append(counts_in_batch)
                    self.assertEqual(
                        sorted(action_counts), [[1, 1], [2, 2], [3, 3]]
                    )
                    if batching_kind == "cached_bucketed":
                        self.assertGreater(trainer.tensor_cache_build_seconds, 0.0)
                        self.assertGreater(trainer.tensor_cache_bytes, 0)
                        self.assertLessEqual(len(trainer._tensor_cache), 2)
                        self.assertLessEqual(trainer.tensor_cache_peak_bytes, trainer.tensor_cache_bytes)
                        self.assertGreater(trainer.tensor_cache_bypasses, 0)

    def test_bucketed_mid_epoch_pause_resume_matches_uninterrupted(self) -> None:
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            examples = self._examples(4)
            validation = self._examples(1)
            for batching_kind in ("bucketed", "cached_bucketed"):
                with self.subTest(batching_kind=batching_kind):
                    config = self._config(batching_kind)
                    with tempfile.TemporaryDirectory() as temp:
                        root = Path(temp)
                        continuous = V4Trainer(
                            examples,
                            root / "continuous",
                            config,
                            vocabulary=self.vocabulary,
                            validation_examples=validation,
                        )
                        continuous.fit()

                        paused = V4Trainer(
                            examples,
                            root / "resumed",
                            config,
                            vocabulary=self.vocabulary,
                            validation_examples=validation,
                        )
                        original_step = paused.optimizer.step
                        step_count = 0

                        def step_and_pause(*args, **kwargs):
                            nonlocal step_count
                            result = original_step(*args, **kwargs)
                            step_count += 1
                            if step_count == 2:
                                paused.stop_requested = True
                            return result

                        paused.optimizer.step = step_and_pause
                        paused.fit()
                        resumed = V4Trainer.from_checkpoint(
                            root / "resumed" / "interrupt.pt",
                            examples,
                            config,
                            validation_examples=validation,
                            expected_vocabulary=self.vocabulary,
                        )
                        resumed.run_dir = root / "resumed"
                        resumed.fit()
                        for name, expected in continuous.model.state_dict().items():
                            self.assertTrue(
                                torch.equal(expected, resumed.model.state_dict()[name]),
                                name,
                            )
        finally:
            torch.set_num_threads(old_threads)

    def test_pause_during_validation_resumes_without_retraining_epoch(self) -> None:
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            examples = self._examples(2)
            validation = self._examples(1)
            config = self._config()
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                trainer = V4Trainer(
                    examples,
                    root,
                    config,
                    vocabulary=self.vocabulary,
                    validation_examples=validation,
                )
                original_evaluate = trainer.evaluate

                def pause_validation(*args, **kwargs):
                    trainer.stop_requested = True
                    return None

                trainer.evaluate = pause_validation
                trainer.fit()
                self.assertTrue(trainer.paused)
                self.assertEqual(trainer.completed_epochs, 1)
                self.assertIsNone(trainer.epoch_progress)
                self.assertIsNotNone(trainer.pending_validation)

                resumed = V4Trainer.from_checkpoint(
                    root / "interrupt.pt",
                    examples,
                    config,
                    validation_examples=validation,
                    expected_vocabulary=self.vocabulary,
                )
                resumed.run_dir = root
                resumed.evaluate = original_evaluate.__get__(resumed, V4Trainer)
                resumed.fit()
                self.assertEqual(resumed.completed_epochs, 1)
                self.assertEqual(len(resumed.history), 1)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()

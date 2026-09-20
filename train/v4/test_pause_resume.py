from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from v3.tensorize import V3Vocabulary
from v4.replay import FactorizedTrainingExampleV4
from v4.train import V4Trainer, V4TrainingConfig


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
                value=1.0 if index % 2 == 0 else -1.0,
            )
            for index in range(count)
        ]

    def _config(self) -> V4TrainingConfig:
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

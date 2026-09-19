"""Supervised trainer for the V4 factorized conditional policy."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from v3.tensorize import V3Vocabulary

from .batch import tensorize_factorized_cases
from .loss import conditional_prefix_cross_entropy
from .model import AugmentNetV4, V4ModelConfig
from .replay import FactorizedTrainingExampleV4, load_game_v4

V4_TRAINER_SCHEMA_ID = "augmentia.v4-trainer-checkpoint.v1"


@dataclass(frozen=True)
class V4TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 1
    value_weight: float = 1.0
    seed: int = 2026091401
    device: str = "cpu"
    model_dim: int = 128
    channels: int = 128
    residual_blocks: int = 6
    runtime_message_passes: int = 2

    def validate(self) -> "V4TrainingConfig":
        if self.learning_rate <= 0.0 or self.batch_size == 0 or self.epochs == 0:
            raise ValueError("learning rate, batch size, and epochs must be positive")
        if self.value_weight < 0.0:
            raise ValueError("value weight must be non-negative")
        return self


class V4Trainer:
    """Train the conditional scorer and the reused value head from V4 games."""

    def __init__(
        self,
        examples: Sequence[FactorizedTrainingExampleV4],
        run_dir: str | Path,
        config: V4TrainingConfig | None = None,
        state_encoder_checkpoint: str | Path | None = None,
    ) -> None:
        self.config = (config or V4TrainingConfig()).validate()
        self.run_dir = Path(run_dir)
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("V4 training requires at least one decision example")
        cases = [
            {"observation": example.observation, "legal_actions": example.legal_actions}
            for example in self.examples
        ]
        self.vocabulary = V3Vocabulary.from_cases(cases)
        torch.manual_seed(self.config.seed)
        self.model = AugmentNetV4(
            V4ModelConfig(
                text_vocabulary_size=self.vocabulary.size,
                channels=self.config.channels,
                residual_blocks=self.config.residual_blocks,
                model_dim=self.config.model_dim,
                runtime_message_passes=self.config.runtime_message_passes,
            )
        )
        if state_encoder_checkpoint is not None:
            state = torch.load(state_encoder_checkpoint, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.state_encoder.load_state_dict(state, strict=False)
            if unexpected:
                raise ValueError(
                    f"state encoder checkpoint has unexpected keys: {sorted(unexpected)}"
                )
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def batch_count(self) -> int:
        size = self.config.batch_size
        return (len(self.examples) + size - 1) // size

    def _batches(self):
        generator = random.Random(self.config.seed)
        order = list(range(len(self.examples)))
        generator.shuffle(order)
        size = self.config.batch_size
        for start in range(0, len(order), size):
            selected = [self.examples[index] for index in order[start : start + size]]
            cases = [
                {"observation": item.observation, "legal_actions": item.legal_actions}
                for item in selected
            ]
            visits = [list(item.complete_visit_counts) for item in selected]
            batch = tensorize_factorized_cases(cases, self.vocabulary, visits)
            values = [item.value for item in selected]
            yield batch, values

    def train_epoch(self) -> dict[str, float]:
        """Run one epoch and return averaged policy/value losses."""
        self.model.train()
        policy_total = 0.0
        value_total = 0.0
        value_batches = 0
        batches = 0
        for batch, values in self._batches():
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch)
            policy_loss = conditional_prefix_cross_entropy(
                output.prefix_logits, batch.prefixes
            )
            targets = [
                item
                for item in values
                if item is not None
            ]
            if targets:
                value_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)
                mask = torch.tensor(
                    [item is not None for item in values], dtype=torch.bool, device=self.device
                )
                value_loss = torch.nn.functional.mse_loss(
                    output.value[mask], value_tensor
                )
                loss = policy_loss + self.config.value_weight * value_loss
                value_total += float(value_loss.detach())
                value_batches += 1
            else:
                loss = policy_loss
            loss.backward()
            self.optimizer.step()
            policy_total += float(policy_loss.detach())
            batches += 1
        return {
            "policy_loss": policy_total / max(1, batches),
            "value_loss": value_total / max(1, value_batches),
        }

    def fit(self) -> list[dict[str, float]]:
        history = []
        for _ in range(self.config.epochs):
            history.append(self.train_epoch())
        return history

    def save_checkpoint(self, path: str | Path) -> tuple[Path, Path]:
        """Write the V4 checkpoint plus its sidecar contract."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), target)
        sidecar = target.with_suffix(".v4-checkpoint.json")
        payload = {
            "schema_id": V4_TRAINER_SCHEMA_ID,
            "vocabulary": self.vocabulary.to_json(),
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.config),
            "example_count": len(self.examples),
        }
        sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target, sidecar

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        examples: Sequence[FactorizedTrainingExampleV4],
        config: V4TrainingConfig | None = None,
    ) -> "V4Trainer":
        """Rebuild a trainer whose weights start from a saved checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        sidecar = checkpoint_path.with_suffix(".v4-checkpoint.json")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("schema_id") != V4_TRAINER_SCHEMA_ID:
            raise ValueError("checkpoint sidecar is not a V4 trainer checkpoint")
        if config is None:
            config = V4TrainingConfig(
                **{
                    key: value
                    for key, value in payload["training_config"].items()
                    if key in V4TrainingConfig.__dataclass_fields__
                }
            )
        trainer = cls(examples, checkpoint_path.parent, config)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        trainer.model.load_state_dict(state, strict=True)
        trainer.model.to(trainer.device)
        return trainer


def train_from_games(
    game_paths: Sequence[str | Path],
    run_dir: str | Path,
    config: V4TrainingConfig | None = None,
) -> V4Trainer:
    examples: list[FactorizedTrainingExampleV4] = []
    for path in game_paths:
        examples.extend(load_game_v4(path))
    trainer = V4Trainer(examples, run_dir, config)
    history = trainer.fit()
    trainer.run_dir.mkdir(parents=True, exist_ok=True)
    (trainer.run_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return trainer

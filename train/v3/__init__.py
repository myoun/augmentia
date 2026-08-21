"""Versioned Python readers for Rust-authored Augmentia V3 records."""

from .model import AugmentNetV3, V3Loss, V3ModelConfig, V3ModelOutput, compute_v3_loss
from .schema import (
    ACTION_SCHEMA_ID,
    FIXTURE_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    SchemaError,
    load_golden_suite,
    validate_golden_suite,
)
from .tensorize import V3Batch, V3Vocabulary, tensorize_cases
from .replay import ReplayDatasetV3, TrainingBatchV3, TrainingExampleV3

__all__ = [
    "ACTION_SCHEMA_ID",
    "FIXTURE_SCHEMA_ID",
    "OBSERVATION_SCHEMA_ID",
    "AugmentNetV3",
    "SchemaError",
    "V3Batch",
    "V3Loss",
    "V3ModelConfig",
    "V3ModelOutput",
    "V3Vocabulary",
    "ReplayDatasetV3",
    "TrainingBatchV3",
    "TrainingExampleV3",
    "compute_v3_loss",
    "load_golden_suite",
    "tensorize_cases",
    "validate_golden_suite",
]

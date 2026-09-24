"""Augmentia V4 factorized policy package."""

from .replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID,
    GAME_RECORD_V4_SCHEMA_ID,
    GAME_RECORD_V5_SCHEMA_ID,
    LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
    LEGACY_GAME_RECORD_V4_SCHEMA_ID,
    FactorizedTrainingExampleV4,
    make_factorized_target_v4,
    make_legacy_factorized_target_v2,
    make_legacy_factorized_target_v4,
    parse_game_v4,
    validate_factorized_target_v4,
    validate_legacy_factorized_target_v4,
)

__all__ = [
    "FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID",
    "FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID",
    "GAME_RECORD_V4_SCHEMA_ID",
    "GAME_RECORD_V5_SCHEMA_ID",
    "LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID",
    "LEGACY_GAME_RECORD_V4_SCHEMA_ID",
    "FactorizedTrainingExampleV4",
    "make_factorized_target_v4",
    "make_legacy_factorized_target_v2",
    "make_legacy_factorized_target_v4",
    "parse_game_v4",
    "validate_factorized_target_v4",
    "validate_legacy_factorized_target_v4",
]

from .batch import (
    FACTOR_FIELD_IDS,
    FactorEdgeBatch,
    PrefixBatchV4,
    V4Batch,
    assert_visit_conservation,
    collate_factorized_batch,
    collate_v4_batch,
    encode_factor_edge,
    tensorize_factorized_cases,
)
from .loss import (
    conditional_prefix_cross_entropy,
    conditional_prefix_policy_metrics,
)
from .model import AugmentNetV4, V4ModelConfig, V4ModelOutput

__all__ += [
    "FACTOR_FIELD_IDS",
    "FactorEdgeBatch",
    "PrefixBatchV4",
    "V4Batch",
    "AugmentNetV4",
    "V4ModelConfig",
    "V4ModelOutput",
    "assert_visit_conservation",
    "collate_factorized_batch",
    "collate_v4_batch",
    "conditional_prefix_cross_entropy",
    "conditional_prefix_policy_metrics",
    "encode_factor_edge",
    "tensorize_factorized_cases",
]

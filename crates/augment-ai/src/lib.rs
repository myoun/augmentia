pub mod action_space;
pub mod action_v3;
pub mod bootstrap_v3;
pub mod data_v3;
pub mod data_v4;
pub mod encoder_v3;
pub mod evaluator;
pub mod factorized_v4;
pub mod mcts;
pub mod mcts_v3;
pub mod mcts_v4;
pub mod observation_v3;
pub mod quiescence;
pub mod runtime_v3;
pub mod runtime_v4;
#[cfg(test)]
mod runtime_v4_integration_tests;
pub mod self_play_v3;
pub mod self_play_v4;
pub mod tensor;

pub use action_space::{
    ACTION_SPACE_SIZE, ActionIndex, ActionMask, encode_action, generate_action_mask,
};
pub use action_v3::{
    ACTION_V3_SCHEMA_ID, ActionDescriptorV3, ActionTargetV3, ActionV3Error, LegalActionSetV3,
    PieceAbilityV3, decode_action_v3, encode_action_v3,
};
pub use bootstrap_v3::{
    ForcedWinConfigV3, ForcedWinErrorV3, ForcedWinSolutionV3, solve_forced_win_v3,
};
pub use data_v3::{
    DATASET_MANIFEST_V3_SCHEMA_ID, DECISION_RECORD_V3_SCHEMA_ID, DataRecordV3, DatasetManifestV3,
    DecisionRecordV3, GAME_RECORD_V3_SCHEMA_ID, GameRecordV3, ReplayDataV3Error, ShardSummaryV3,
    read_jsonl_zstd_shard, validate_manifest, verify_manifest_files, verify_manifest_replays,
    write_jsonl_zstd_shard,
};
pub use data_v4::{
    FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID, FactorizedMctsTargetV4, GAME_RECORD_V4_SCHEMA_ID,
    GameRecordV4, GameStepV4, V4DataError,
};
pub use encoder_v3::{
    BoardPieceV3, CardRecordV3, GlobalStateV3, NamedRuntimeValueV3, OBSERVATION_V3_GLOBAL_FEATURES,
    OBSERVATION_V3_REQUIRED_TEXT_TOKENS, OBSERVATION_V3_SPATIAL_PLANES,
    OBSERVATION_V3_SPATIAL_SIZE, ObservationV3, PhaseV3, RelativePlayerV3, RuntimeFeatureV3,
    RuntimeValueV3, encode_observation_v3, observation_text_tokens,
    observation_v3_runtime_text_tokens,
};
pub use evaluator::{Evaluator, HeuristicEvaluator, OnnxNeuralEvaluator};
pub use factorized_v4::{
    FACTORIZED_POLICY_V4_SCHEMA_ID, FactorizedPolicyV4Error, FactorizedPrefixVisitsV4,
    FactorizedTrieV4, aggregate_factorized_prefix_visits_v4, build_factorized_trie_v4,
    factorized_action_path_v4,
};
pub use mcts::{
    MctsConfig, MctsNode, MctsTree, RootActionDiagnostics, RootNoiseConfig, SearchDiagnostics,
};
pub use mcts_v3::{
    BatchEvaluatorV3, MctsV3Config, MctsV3Error, MctsV3Tree, RootActionDiagnosticsV3,
    RootNoiseConfigV3, SearchDiagnosticsV3, SearchResultV3,
};
pub use mcts_v4::{
    CHANCE_SAMPLING_V4_ID, DeterministicFactorizedEvaluatorV4, FactorizedEvaluationV4,
    FactorizedEvaluatorV4, MctsV4Config, MctsV4Error, MctsV4Tree, ROOT_SELECTION_V4_ID,
    RootActionVisitV4, SEARCH_SEMANTICS_V4_ID, SearchDiagnosticsV4, SearchResultV4,
};
pub use observation_v3::{
    ACTION_V2_SCHEMA_ID, CARD_RUNTIME_SOURCE_FIELDS, DERIVED_SOURCE_FIELDS,
    GAME_STATE_SOURCE_FIELDS, INTERNAL_STATE_EXCLUSIONS, OBSERVATION_V2_SCHEMA_ID,
    OBSERVATION_V3_SCHEMA_ID, PIECE_KIND_VOCABULARY, SELF_PLAY_V2_RECORD_SCHEMA_ID,
    V2_BASELINE_MODEL_ID,
};
pub use quiescence::QuiescenceSearch;
pub use runtime_v3::{
    CandidateEvaluationV3, EvaluationV3, MODEL_METADATA_V3_SCHEMA_ID, V3ExecutionProvider,
    V3OnnxRuntime, V3RuntimeError, V3RuntimeOptions,
};
pub use runtime_v4::{V4_BUNDLE_METADATA_SCHEMA_ID, V4OnnxRuntime, V4RuntimeError};
pub use self_play_v3::{
    SelfPlayAgentV3, SelfPlayGameConfigV3, SelfPlayV3Error, generate_self_play_game_v3,
};
pub use self_play_v4::{
    SelfPlayAgentV4, SelfPlayGameConfigV4, SelfPlayV4Error, generate_self_play_game_v4,
};
pub use tensor::{
    EncodedState, GLOBAL_FEATURES_SIZE, SPATIAL_CHANNELS, SPATIAL_SIZE, encode_state,
};
#[cfg(test)]
mod tests;

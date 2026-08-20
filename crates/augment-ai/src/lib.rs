pub mod action_v3;
pub mod action_space;
pub mod bootstrap_v3;
pub mod data_v3;
pub mod evaluator;
pub mod encoder_v3;
pub mod mcts;
pub mod observation_v3;
pub mod quiescence;
pub mod tensor;

pub use action_v3::{
    ACTION_V3_SCHEMA_ID, ActionDescriptorV3, ActionTargetV3, ActionV3Error, LegalActionSetV3,
    PieceAbilityV3, decode_action_v3, encode_action_v3,
};
pub use action_space::{
    ACTION_SPACE_SIZE, ActionIndex, ActionMask, encode_action, generate_action_mask,
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
pub use evaluator::{Evaluator, HeuristicEvaluator, OnnxNeuralEvaluator};
pub use encoder_v3::{
    BoardPieceV3, CardRecordV3, GlobalStateV3, NamedRuntimeValueV3, ObservationV3, PhaseV3,
    RelativePlayerV3, RuntimeFeatureV3, RuntimeValueV3, OBSERVATION_V3_GLOBAL_FEATURES,
    OBSERVATION_V3_SPATIAL_PLANES, OBSERVATION_V3_SPATIAL_SIZE, encode_observation_v3,
};
pub use mcts::{
    MctsConfig, MctsNode, MctsTree, RootActionDiagnostics, RootNoiseConfig, SearchDiagnostics,
};
pub use observation_v3::{
    ACTION_V2_SCHEMA_ID, CARD_RUNTIME_SOURCE_FIELDS, DERIVED_SOURCE_FIELDS,
    GAME_STATE_SOURCE_FIELDS, INTERNAL_STATE_EXCLUSIONS, OBSERVATION_V2_SCHEMA_ID,
    OBSERVATION_V3_SCHEMA_ID, PIECE_KIND_VOCABULARY, SELF_PLAY_V2_RECORD_SCHEMA_ID,
    V2_BASELINE_MODEL_ID,
};
pub use quiescence::QuiescenceSearch;
pub use tensor::{
    EncodedState, GLOBAL_FEATURES_SIZE, SPATIAL_CHANNELS, SPATIAL_SIZE, encode_state,
};
#[cfg(test)]
mod tests;

use crate::action_v3::{ACTION_V3_SCHEMA_ID, ActionDescriptorV3, ActionTargetV3, LegalActionSetV3};
use crate::encoder_v3::{
    OBSERVATION_V3_GLOBAL_FEATURES, OBSERVATION_V3_SPATIAL_PLANES, OBSERVATION_V3_SPATIAL_SIZE,
    ObservationV3, PhaseV3, StatusV3, encode_observation_v3,
};
use crate::observation_v3::{
    CARD_RUNTIME_SOURCE_FIELDS, OBSERVATION_V3_SCHEMA_ID, PIECE_KIND_VOCABULARY,
};
use augment_core::{DrawReason, Engine, GameStatus, Mode, Player, WinReason};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Component, Path};

pub const GAME_RECORD_V3_SCHEMA_ID: &str = "augmentia.replay.game.v3";
pub const DECISION_RECORD_V3_SCHEMA_ID: &str = "augmentia.training.decision.v3";
pub const DATASET_MANIFEST_V3_SCHEMA_ID: &str = "augmentia.dataset.manifest.v3";
pub const SPLIT_POLICY_V3_ALGORITHM: &str = "sha256_split_group_v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RulesetRefV3 {
    pub id: String,
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorProvenanceV3 {
    pub run_id: String,
    pub engine_commit: String,
    pub engine_api_version: u32,
    pub collector_id: String,
    pub dirty_worktree: bool,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbsolutePlayerV3 {
    White,
    Black,
}

impl AbsolutePlayerV3 {
    pub fn from_player(player: Player) -> Self {
        match player {
            Player::White => Self::White,
            Player::Black => Self::Black,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKindV3 {
    Random,
    RuleBased,
    Mcts,
    V2,
    Human,
    Imported,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifactV3 {
    pub model_id: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerAgentV3 {
    pub seat: AbsolutePlayerV3,
    pub agent_kind: AgentKindV3,
    pub agent_id: String,
    pub model: Option<ModelArtifactV3>,
    pub config_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PolicyTargetV3 {
    MctsVisits {
        visit_counts: Vec<u32>,
        completed_simulations: u32,
        root_value: f32,
        noise_seed_hex: Option<String>,
        sampling_temperature: f32,
    },
    BehaviorDistribution {
        probabilities: Vec<f32>,
        source: String,
    },
    None {
        reason: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameResultV3 {
    WhiteWin,
    BlackWin,
    Draw,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKindV3 {
    Decision,
    WallClock,
    Resource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorStageV3 {
    Initialization,
    Observation,
    Search,
    Engine,
    Serialization,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TerminationV3 {
    EngineTerminal {
        result: GameResultV3,
        reason: String,
        detail: Option<String>,
        final_state_fingerprint: String,
    },
    Limit {
        limit: LimitKindV3,
        reason: String,
        final_state_fingerprint: String,
    },
    Aborted {
        reason: String,
        final_state_fingerprint: String,
    },
    Error {
        stage: ErrorStageV3,
        code: String,
        detail: Option<String>,
        final_state_fingerprint: String,
    },
}

impl TerminationV3 {
    pub fn from_engine_status(
        status: &GameStatus,
        final_state_fingerprint: String,
    ) -> Option<Self> {
        match status {
            GameStatus::Ongoing => None,
            GameStatus::Won { winner, reason } => {
                let (reason, detail) = win_reason(reason);
                Some(Self::EngineTerminal {
                    result: match winner {
                        Player::White => GameResultV3::WhiteWin,
                        Player::Black => GameResultV3::BlackWin,
                    },
                    reason: reason.to_owned(),
                    detail,
                    final_state_fingerprint,
                })
            }
            GameStatus::Draw { reason } => {
                let (reason, detail) = draw_reason(reason);
                Some(Self::EngineTerminal {
                    result: GameResultV3::Draw,
                    reason: reason.to_owned(),
                    detail,
                    final_state_fingerprint,
                })
            }
        }
    }

    pub fn final_state_fingerprint(&self) -> &str {
        match self {
            Self::EngineTerminal {
                final_state_fingerprint,
                ..
            }
            | Self::Limit {
                final_state_fingerprint,
                ..
            }
            | Self::Aborted {
                final_state_fingerprint,
                ..
            }
            | Self::Error {
                final_state_fingerprint,
                ..
            } => final_state_fingerprint,
        }
    }

    pub fn value_for(&self, actor: AbsolutePlayerV3) -> Option<f32> {
        match self {
            Self::EngineTerminal { result, .. } => Some(match (result, actor) {
                (GameResultV3::Draw, _) => 0.0,
                (GameResultV3::WhiteWin, AbsolutePlayerV3::White)
                | (GameResultV3::BlackWin, AbsolutePlayerV3::Black) => 1.0,
                _ => -1.0,
            }),
            Self::Limit { .. } | Self::Aborted { .. } | Self::Error { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameStepV3 {
    pub decision_index: u32,
    pub engine_ply: u32,
    pub actor: AbsolutePlayerV3,
    pub state_fingerprint: String,
    pub observation: ObservationV3,
    pub legal_actions: LegalActionSetV3,
    pub policy_target: PolicyTargetV3,
    pub played_action_index: u32,
    pub next_state_fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameRecordV3 {
    pub record_schema_id: String,
    pub game_id: String,
    pub ruleset: RulesetRefV3,
    pub observation_schema_id: String,
    pub action_schema_id: String,
    pub generator: GeneratorProvenanceV3,
    pub mode_id: String,
    pub seed_hex: String,
    pub players: [PlayerAgentV3; 2],
    pub initial_state_fingerprint: String,
    pub steps: Vec<GameStepV3>,
    pub termination: TerminationV3,
    pub trajectory_sha256: String,
}

impl GameRecordV3 {
    pub fn value_target(&self, decision_index: usize) -> Option<f32> {
        self.steps
            .get(decision_index)
            .and_then(|step| self.termination.value_for(step.actor))
    }

    pub fn split_group_key(&self) -> String {
        format!(
            "game\0{}\0{}\0{}",
            self.ruleset.id, self.mode_id, self.seed_hex
        )
    }

    pub fn recompute_trajectory_sha256(&self) -> Result<String, ReplayDataV3Error> {
        #[derive(Serialize)]
        struct DigestStep<'a> {
            state_fingerprint: &'a str,
            played_action: &'a ActionDescriptorV3,
            next_state_fingerprint: &'a str,
        }

        #[derive(Serialize)]
        struct DigestInput<'a> {
            ruleset_id: &'a str,
            mode_id: &'a str,
            seed_hex: &'a str,
            initial_state_fingerprint: &'a str,
            steps: Vec<DigestStep<'a>>,
            termination: &'a TerminationV3,
        }

        let mut steps = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let action = step
                .legal_actions
                .actions
                .get(step.played_action_index as usize)
                .ok_or_else(|| {
                    ReplayDataV3Error::Validation(format!(
                        "game {} decision {} played_action_index is out of range",
                        self.game_id, step.decision_index
                    ))
                })?;
            steps.push(DigestStep {
                state_fingerprint: &step.state_fingerprint,
                played_action: action,
                next_state_fingerprint: &step.next_state_fingerprint,
            });
        }
        let bytes = serde_json::to_vec(&DigestInput {
            ruleset_id: &self.ruleset.id,
            mode_id: &self.mode_id,
            seed_hex: &self.seed_hex,
            initial_state_fingerprint: &self.initial_state_fingerprint,
            steps,
            termination: &self.termination,
        })?;
        Ok(sha256_bytes(&bytes))
    }

    pub fn refresh_trajectory_sha256(&mut self) -> Result<(), ReplayDataV3Error> {
        self.trajectory_sha256 = self.recompute_trajectory_sha256()?;
        Ok(())
    }

    pub fn recompute_training_content_sha256(&self) -> Result<String, ReplayDataV3Error> {
        #[derive(Serialize)]
        struct TrainingContent<'a> {
            ruleset: &'a RulesetRefV3,
            observation_schema_id: &'a str,
            action_schema_id: &'a str,
            mode_id: &'a str,
            seed_hex: &'a str,
            initial_state_fingerprint: &'a str,
            steps: &'a [GameStepV3],
            termination: &'a TerminationV3,
        }

        Ok(sha256_bytes(&serde_json::to_vec(&TrainingContent {
            ruleset: &self.ruleset,
            observation_schema_id: &self.observation_schema_id,
            action_schema_id: &self.action_schema_id,
            mode_id: &self.mode_id,
            seed_hex: &self.seed_hex,
            initial_state_fingerprint: &self.initial_state_fingerprint,
            steps: &self.steps,
            termination: &self.termination,
        })?))
    }

    pub fn validate(&self) -> Result<(), ReplayDataV3Error> {
        require_equal(
            &self.record_schema_id,
            GAME_RECORD_V3_SCHEMA_ID,
            "game.record_schema_id",
        )?;
        require_nonempty(&self.game_id, "game.game_id")?;
        validate_ruleset(&self.ruleset)?;
        require_equal(
            &self.observation_schema_id,
            OBSERVATION_V3_SCHEMA_ID,
            "game.observation_schema_id",
        )?;
        require_equal(
            &self.action_schema_id,
            ACTION_V3_SCHEMA_ID,
            "game.action_schema_id",
        )?;
        validate_generator(&self.generator)?;
        require_nonempty(&self.mode_id, "game.mode_id")?;
        validate_hex(&self.seed_hex, 16, "game.seed_hex")?;
        validate_players(&self.players)?;
        validate_hex(
            &self.initial_state_fingerprint,
            16,
            "game.initial_state_fingerprint",
        )?;

        let mut expected_fingerprint = self.initial_state_fingerprint.as_str();
        for (index, step) in self.steps.iter().enumerate() {
            if step.decision_index != index as u32 {
                return validation(format!(
                    "game {} decision_index {} must be {}",
                    self.game_id, step.decision_index, index
                ));
            }
            if step.state_fingerprint != expected_fingerprint {
                return validation(format!(
                    "game {} decision {} breaks the fingerprint chain",
                    self.game_id, index
                ));
            }
            validate_step(step, &self.ruleset.id, index)?;
            expected_fingerprint = &step.next_state_fingerprint;
        }
        if self.termination.final_state_fingerprint() != expected_fingerprint {
            return validation(format!(
                "game {} termination fingerprint does not match the final step",
                self.game_id
            ));
        }
        validate_termination(&self.termination)?;
        validate_hex(&self.trajectory_sha256, 64, "game.trajectory_sha256")?;
        if self.recompute_trajectory_sha256()? != self.trajectory_sha256 {
            return validation(format!(
                "game {} trajectory_sha256 does not match its semantic trajectory",
                self.game_id
            ));
        }
        Ok(())
    }

    pub fn validate_replay(&self, engine: &Engine) -> Result<(), ReplayDataV3Error> {
        self.validate()?;
        let seed = u64::from_str_radix(&self.seed_hex, 16).map_err(|error| {
            ReplayDataV3Error::Validation(format!("invalid game seed: {error}"))
        })?;
        let mode = match self.mode_id.as_str() {
            "standard" => Mode::Standard,
            "chaos" => Mode::Chaos,
            "grand" => Mode::Grand,
            other => Mode::Campaign(other.to_owned()),
        };
        let mut state = engine
            .new_game(mode, seed)
            .map_err(|error| ReplayDataV3Error::Replay(error.to_string()))?;
        if state.ruleset_id != self.ruleset.id {
            return replay(format!(
                "game {} ruleset {} does not match engine state {}",
                self.game_id, self.ruleset.id, state.ruleset_id
            ));
        }
        if fingerprint_hex(state.fingerprint()) != self.initial_state_fingerprint {
            return replay(format!(
                "game {} initial state fingerprint drifted",
                self.game_id
            ));
        }

        for step in &self.steps {
            if AbsolutePlayerV3::from_player(state.turn) != step.actor {
                return replay(format!(
                    "game {} decision {} actor does not match engine turn",
                    self.game_id, step.decision_index
                ));
            }
            if state.ply != step.engine_ply {
                return replay(format!(
                    "game {} decision {} engine ply drifted",
                    self.game_id, step.decision_index
                ));
            }
            if fingerprint_hex(state.fingerprint()) != step.state_fingerprint {
                return replay(format!(
                    "game {} decision {} state fingerprint drifted",
                    self.game_id, step.decision_index
                ));
            }
            if encode_observation_v3(&state) != step.observation {
                return replay(format!(
                    "game {} decision {} observation drifted",
                    self.game_id, step.decision_index
                ));
            }
            let legal = LegalActionSetV3::from_state(&state, engine)
                .map_err(|error| ReplayDataV3Error::Replay(error.to_string()))?;
            if legal != step.legal_actions {
                return replay(format!(
                    "game {} decision {} legal actions drifted",
                    self.game_id, step.decision_index
                ));
            }
            let transition = step
                .legal_actions
                .apply(step.played_action_index as usize, &state, engine)
                .map_err(|error| ReplayDataV3Error::Replay(error.to_string()))?;
            state = transition.state;
            if fingerprint_hex(state.fingerprint()) != step.next_state_fingerprint {
                return replay(format!(
                    "game {} decision {} next state fingerprint drifted",
                    self.game_id, step.decision_index
                ));
            }
        }
        validate_engine_termination(&self.termination, &state.status)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionSourceKindV3 {
    GameRecord,
    EngineSearch,
    ImportedGame,
    ManualScenario,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceDomainV3 {
    AugmentChess,
    StandardChess,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionProvenanceV3 {
    pub source_kind: DecisionSourceKindV3,
    pub source_domain: SourceDomainV3,
    pub source_id: String,
    pub split_group_id: String,
    pub generator_commit: Option<String>,
    pub license: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecisionPolicyTargetV3 {
    MctsVisits {
        visit_counts: Vec<u32>,
        completed_simulations: u32,
        root_value: f32,
    },
    BehaviorDistribution {
        probabilities: Vec<f32>,
        source: String,
    },
    OptimalActionSet {
        action_indices: Vec<u32>,
        proof_source: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyTrainingScopeV3 {
    AugmentPolicy,
    BoardPretraining,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionPolicySupervisionV3 {
    pub scope: PolicyTrainingScopeV3,
    pub target: DecisionPolicyTargetV3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecisionValueTargetV3 {
    TerminalOutcome { value: f32, source_game_id: String },
    Proven { value: f32, proof_source: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum AuxiliaryTargetValueV3 {
    Bool(bool),
    Integer(i64),
    Float(f32),
    Text(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuxiliaryTargetV3 {
    pub name: String,
    pub value: AuxiliaryTargetValueV3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionTargetsV3 {
    pub policy: Option<DecisionPolicySupervisionV3>,
    pub value: Option<DecisionValueTargetV3>,
    pub auxiliary: Vec<AuxiliaryTargetV3>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRecordV3 {
    pub record_schema_id: String,
    pub decision_id: String,
    pub provenance: DecisionProvenanceV3,
    pub ruleset: RulesetRefV3,
    pub observation_schema_id: String,
    pub action_schema_id: String,
    pub actor: AbsolutePlayerV3,
    pub state_fingerprint: String,
    pub observation: ObservationV3,
    pub legal_actions: LegalActionSetV3,
    pub targets: DecisionTargetsV3,
}

impl DecisionRecordV3 {
    pub fn split_group_key(&self) -> &str {
        &self.provenance.split_group_id
    }

    pub fn validate(&self) -> Result<(), ReplayDataV3Error> {
        require_equal(
            &self.record_schema_id,
            DECISION_RECORD_V3_SCHEMA_ID,
            "decision.record_schema_id",
        )?;
        require_nonempty(&self.decision_id, "decision.decision_id")?;
        require_nonempty(&self.provenance.source_id, "decision.provenance.source_id")?;
        require_nonempty(
            &self.provenance.split_group_id,
            "decision.provenance.split_group_id",
        )?;
        if let Some(commit) = &self.provenance.generator_commit {
            validate_hex(commit, 40, "decision.provenance.generator_commit")?;
        }
        validate_ruleset(&self.ruleset)?;
        require_equal(
            &self.observation_schema_id,
            OBSERVATION_V3_SCHEMA_ID,
            "decision.observation_schema_id",
        )?;
        require_equal(
            &self.action_schema_id,
            ACTION_V3_SCHEMA_ID,
            "decision.action_schema_id",
        )?;
        validate_hex(&self.state_fingerprint, 16, "decision.state_fingerprint")?;
        validate_observation(&self.observation, &self.ruleset.id, "decision.observation")?;
        validate_legal_actions(
            &self.legal_actions,
            &self.observation,
            "decision.legal_actions",
        )?;
        if self.legal_actions.actions.is_empty() {
            return validation(format!(
                "decision {} has no legal action to supervise",
                self.decision_id
            ));
        }
        validate_decision_targets(&self.targets, self.legal_actions.actions.len())?;
        if self.provenance.source_domain == SourceDomainV3::StandardChess {
            if self.targets.value.is_some() {
                return validation(format!(
                    "decision {} from standard chess cannot supervise V3 value",
                    self.decision_id
                ));
            }
            if self
                .targets
                .policy
                .as_ref()
                .is_some_and(|policy| policy.scope != PolicyTrainingScopeV3::BoardPretraining)
            {
                return validation(format!(
                    "decision {} from standard chess may only supervise board pretraining",
                    self.decision_id
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DataRecordV3 {
    Game(Box<GameRecordV3>),
    Decision(Box<DecisionRecordV3>),
}

impl DataRecordV3 {
    pub fn schema_id(&self) -> &'static str {
        match self {
            Self::Game(_) => GAME_RECORD_V3_SCHEMA_ID,
            Self::Decision(_) => DECISION_RECORD_V3_SCHEMA_ID,
        }
    }

    pub fn record_id(&self) -> &str {
        match self {
            Self::Game(record) => &record.game_id,
            Self::Decision(record) => &record.decision_id,
        }
    }

    pub fn split_group_key(&self) -> String {
        match self {
            Self::Game(record) => record.split_group_key(),
            Self::Decision(record) => record.split_group_key().to_owned(),
        }
    }

    pub fn validate(&self) -> Result<(), ReplayDataV3Error> {
        match self {
            Self::Game(record) => record.validate(),
            Self::Decision(record) => record.validate(),
        }
    }

    fn to_json_vec(&self) -> Result<Vec<u8>, ReplayDataV3Error> {
        Ok(match self {
            Self::Game(record) => serde_json::to_vec(record)?,
            Self::Decision(record) => serde_json::to_vec(record)?,
        })
    }

    fn from_json_slice(bytes: &[u8]) -> Result<Self, ReplayDataV3Error> {
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        let schema_id = value
            .get("record_schema_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                ReplayDataV3Error::Validation(
                    "record is missing string record_schema_id".to_owned(),
                )
            })?;
        let record = match schema_id {
            GAME_RECORD_V3_SCHEMA_ID => Self::Game(Box::new(serde_json::from_value(value)?)),
            DECISION_RECORD_V3_SCHEMA_ID => {
                Self::Decision(Box::new(serde_json::from_value(value)?))
            }
            other => {
                return validation(format!("unsupported record_schema_id {other}"));
            }
        };
        record.validate()?;
        Ok(record)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplitV3 {
    Train,
    Validation,
    Holdout,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitPolicyV3 {
    pub algorithm: String,
    pub salt: String,
    pub train_basis_points: u16,
    pub validation_basis_points: u16,
    pub holdout_basis_points: u16,
}

impl SplitPolicyV3 {
    pub fn assign(&self, split_group_key: &str) -> Result<DatasetSplitV3, ReplayDataV3Error> {
        validate_split_policy(self)?;
        let mut hasher = Sha256::new();
        hasher.update(self.salt.as_bytes());
        hasher.update([0]);
        hasher.update(split_group_key.as_bytes());
        let digest = hasher.finalize();
        let prefix = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
        let bucket = ((u64::from(prefix) * 10_000) >> 32) as u16;
        if bucket < self.train_basis_points {
            Ok(DatasetSplitV3::Train)
        } else if bucket < self.train_basis_points + self.validation_basis_points {
            Ok(DatasetSplitV3::Validation)
        } else {
            Ok(DatasetSplitV3::Holdout)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressionV3 {
    pub codec: String,
    pub level: i32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardManifestV3 {
    pub path: String,
    pub split: DatasetSplitV3,
    pub record_schema_id: String,
    pub sha256: String,
    pub compressed_bytes: u64,
    pub records: u64,
    pub games: u64,
    pub decisions: u64,
    pub game_steps: u64,
    pub policy_eligible_decisions: u64,
    pub value_eligible_decisions: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetTotalsV3 {
    pub shards: u64,
    pub records: u64,
    pub games: u64,
    pub decisions: u64,
    pub game_steps: u64,
    pub policy_eligible_decisions: u64,
    pub value_eligible_decisions: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetManifestV3 {
    pub manifest_schema_id: String,
    pub dataset_id: String,
    pub created_at_utc: String,
    pub observation_schema_id: String,
    pub action_schema_id: String,
    pub split_policy: SplitPolicyV3,
    pub compression: CompressionV3,
    pub shards: Vec<ShardManifestV3>,
    pub totals: DatasetTotalsV3,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardSummaryV3 {
    pub sha256: String,
    pub compressed_bytes: u64,
    pub record_schema_id: String,
    pub records: u64,
    pub games: u64,
    pub decisions: u64,
    pub game_steps: u64,
    pub policy_eligible_decisions: u64,
    pub value_eligible_decisions: u64,
}

impl ShardSummaryV3 {
    pub fn to_manifest_entry(&self, path: String, split: DatasetSplitV3) -> ShardManifestV3 {
        ShardManifestV3 {
            path,
            split,
            record_schema_id: self.record_schema_id.clone(),
            sha256: self.sha256.clone(),
            compressed_bytes: self.compressed_bytes,
            records: self.records,
            games: self.games,
            decisions: self.decisions,
            game_steps: self.game_steps,
            policy_eligible_decisions: self.policy_eligible_decisions,
            value_eligible_decisions: self.value_eligible_decisions,
        }
    }
}

pub fn write_jsonl_zstd_shard(
    path: impl AsRef<Path>,
    records: &[DataRecordV3],
    compression_level: i32,
) -> Result<ShardSummaryV3, ReplayDataV3Error> {
    if records.is_empty() {
        return validation("cannot write an empty V3 shard".to_owned());
    }
    let path = path.as_ref();
    validate_shard_path(path)?;
    if path.exists() {
        return validation(format!("refusing to overwrite shard {}", path.display()));
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ReplayDataV3Error::Validation("invalid shard filename".to_owned()))?;
    let partial_path = parent.join(format!("{file_name}.partial"));
    if partial_path.exists() {
        return validation(format!(
            "partial shard already exists: {}",
            partial_path.display()
        ));
    }

    let schema_id = records[0].schema_id();
    let result = (|| {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial_path)?;
        let buffered = BufWriter::new(file);
        let mut encoder = zstd::stream::write::Encoder::new(buffered, compression_level)?;
        let mut ids = BTreeSet::new();
        let mut training_contents = BTreeSet::new();
        let mut summary = ShardSummaryV3 {
            sha256: String::new(),
            compressed_bytes: 0,
            record_schema_id: schema_id.to_owned(),
            records: 0,
            games: 0,
            decisions: 0,
            game_steps: 0,
            policy_eligible_decisions: 0,
            value_eligible_decisions: 0,
        };

        for record in records {
            if record.schema_id() != schema_id {
                return validation("a shard must contain exactly one record schema".to_owned());
            }
            record.validate()?;
            if !ids.insert(record.record_id().to_owned()) {
                return validation(format!(
                    "duplicate record id {} in shard",
                    record.record_id()
                ));
            }
            match record {
                DataRecordV3::Game(game) => {
                    let content_sha256 = game.recompute_training_content_sha256()?;
                    if !training_contents.insert(content_sha256.clone()) {
                        return validation(format!(
                            "duplicate training content {content_sha256} in shard"
                        ));
                    }
                    summary.games += 1;
                    summary.game_steps += game.steps.len() as u64;
                    summary.policy_eligible_decisions += game
                        .steps
                        .iter()
                        .filter(|step| !matches!(step.policy_target, PolicyTargetV3::None { .. }))
                        .count() as u64;
                    if matches!(game.termination, TerminationV3::EngineTerminal { .. }) {
                        summary.value_eligible_decisions += game.steps.len() as u64;
                    }
                }
                DataRecordV3::Decision(decision) => {
                    summary.decisions += 1;
                    summary.policy_eligible_decisions +=
                        u64::from(decision.targets.policy.is_some());
                    summary.value_eligible_decisions += u64::from(decision.targets.value.is_some());
                }
            }
            let bytes = record.to_json_vec()?;
            encoder.write_all(&bytes)?;
            encoder.write_all(b"\n")?;
            summary.records += 1;
        }
        let mut buffered = encoder.finish()?;
        buffered.flush()?;
        let file = buffered.into_inner().map_err(|error| error.into_error())?;
        file.sync_all()?;
        Ok(summary)
    })();

    let mut summary = match result {
        Ok(summary) => summary,
        Err(error) => {
            let _ = std::fs::remove_file(&partial_path);
            return Err(error);
        }
    };
    std::fs::rename(&partial_path, path)?;
    summary.compressed_bytes = std::fs::metadata(path)?.len();
    summary.sha256 = sha256_file(path)?;
    Ok(summary)
}

pub fn read_jsonl_zstd_shard(
    path: impl AsRef<Path>,
) -> Result<Vec<DataRecordV3>, ReplayDataV3Error> {
    let path = path.as_ref();
    validate_shard_path(path)?;
    let file = File::open(path)?;
    let decoder = zstd::stream::read::Decoder::new(file)?;
    let mut reader = BufReader::new(decoder);
    let mut records = Vec::new();
    let mut ids = BTreeSet::new();
    let mut training_contents = BTreeSet::new();
    let mut schema_id: Option<&'static str> = None;
    let mut bytes = Vec::new();
    let mut line_number = 0usize;

    loop {
        bytes.clear();
        let read = reader.read_until(b'\n', &mut bytes)?;
        if read == 0 {
            break;
        }
        line_number += 1;
        if bytes.last() == Some(&b'\n') {
            bytes.pop();
        }
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
        if bytes.is_empty() {
            return validation(format!("empty JSON line at {line_number}"));
        }
        let record = DataRecordV3::from_json_slice(&bytes).map_err(|error| {
            ReplayDataV3Error::Validation(format!("line {line_number}: {error}"))
        })?;
        if let Some(expected) = schema_id {
            if expected != record.schema_id() {
                return validation(format!("line {line_number}: shard mixes record schemas"));
            }
        } else {
            schema_id = Some(record.schema_id());
        }
        if !ids.insert(record.record_id().to_owned()) {
            return validation(format!(
                "line {line_number}: duplicate record id {}",
                record.record_id()
            ));
        }
        if let DataRecordV3::Game(game) = &record {
            let content_sha256 = game.recompute_training_content_sha256()?;
            if !training_contents.insert(content_sha256.clone()) {
                return validation(format!(
                    "line {line_number}: duplicate training content {content_sha256}"
                ));
            }
        }
        records.push(record);
    }
    if records.is_empty() {
        return validation(format!("shard {} contains no records", path.display()));
    }
    Ok(records)
}

pub fn validate_manifest(manifest: &DatasetManifestV3) -> Result<(), ReplayDataV3Error> {
    require_equal(
        &manifest.manifest_schema_id,
        DATASET_MANIFEST_V3_SCHEMA_ID,
        "manifest.manifest_schema_id",
    )?;
    require_nonempty(&manifest.dataset_id, "manifest.dataset_id")?;
    if !manifest.created_at_utc.ends_with('Z') {
        return validation("manifest.created_at_utc must be UTC and end with Z".to_owned());
    }
    require_equal(
        &manifest.observation_schema_id,
        OBSERVATION_V3_SCHEMA_ID,
        "manifest.observation_schema_id",
    )?;
    require_equal(
        &manifest.action_schema_id,
        ACTION_V3_SCHEMA_ID,
        "manifest.action_schema_id",
    )?;
    validate_split_policy(&manifest.split_policy)?;
    require_equal(
        &manifest.compression.codec,
        "zstd",
        "manifest.compression.codec",
    )?;
    if !(-7..=22).contains(&manifest.compression.level) {
        return validation(
            "manifest.compression.level is outside zstd's supported range".to_owned(),
        );
    }
    if manifest.shards.is_empty() {
        return validation("manifest must contain at least one shard".to_owned());
    }

    let mut paths = BTreeSet::new();
    let mut totals = DatasetTotalsV3 {
        shards: manifest.shards.len() as u64,
        ..DatasetTotalsV3::default()
    };
    for shard in &manifest.shards {
        validate_relative_shard_path(&shard.path)?;
        if !paths.insert(shard.path.as_str()) {
            return validation(format!("duplicate shard path {}", shard.path));
        }
        if !matches!(
            shard.record_schema_id.as_str(),
            GAME_RECORD_V3_SCHEMA_ID | DECISION_RECORD_V3_SCHEMA_ID
        ) {
            return validation(format!(
                "shard {} has unsupported record schema {}",
                shard.path, shard.record_schema_id
            ));
        }
        validate_hex(&shard.sha256, 64, "manifest.shards[].sha256")?;
        if shard.compressed_bytes == 0 || shard.records == 0 {
            return validation(format!("shard {} is empty", shard.path));
        }
        if shard.records != shard.games + shard.decisions {
            return validation(format!("shard {} record counts do not add up", shard.path));
        }
        if shard.record_schema_id == GAME_RECORD_V3_SCHEMA_ID && shard.decisions != 0 {
            return validation(format!(
                "game shard {} contains decision counts",
                shard.path
            ));
        }
        if shard.record_schema_id == DECISION_RECORD_V3_SCHEMA_ID
            && (shard.games != 0 || shard.game_steps != 0)
        {
            return validation(format!(
                "decision shard {} contains game counts",
                shard.path
            ));
        }
        totals.records += shard.records;
        totals.games += shard.games;
        totals.decisions += shard.decisions;
        totals.game_steps += shard.game_steps;
        totals.policy_eligible_decisions += shard.policy_eligible_decisions;
        totals.value_eligible_decisions += shard.value_eligible_decisions;
    }
    if totals != manifest.totals {
        return validation("manifest totals do not equal the sum of shard metadata".to_owned());
    }
    Ok(())
}

pub fn verify_manifest_files(
    base_dir: impl AsRef<Path>,
    manifest: &DatasetManifestV3,
) -> Result<(), ReplayDataV3Error> {
    validate_manifest(manifest)?;
    let base_dir = base_dir.as_ref();
    let mut ids = BTreeSet::new();
    let mut training_contents = BTreeSet::new();
    for shard in &manifest.shards {
        let path = base_dir.join(Path::new(&shard.path));
        if std::fs::metadata(&path)?.len() != shard.compressed_bytes {
            return validation(format!("shard {} byte length mismatch", shard.path));
        }
        if sha256_file(&path)? != shard.sha256 {
            return validation(format!("shard {} SHA-256 mismatch", shard.path));
        }
        let records = read_jsonl_zstd_shard(&path)?;
        let summary = summarize_records(&records, &shard.sha256, shard.compressed_bytes)?;
        if summary.record_schema_id != shard.record_schema_id
            || summary.records != shard.records
            || summary.games != shard.games
            || summary.decisions != shard.decisions
            || summary.game_steps != shard.game_steps
            || summary.policy_eligible_decisions != shard.policy_eligible_decisions
            || summary.value_eligible_decisions != shard.value_eligible_decisions
        {
            return validation(format!("shard {} metadata mismatch", shard.path));
        }
        for record in records {
            if manifest.split_policy.assign(&record.split_group_key())? != shard.split {
                return validation(format!(
                    "record {} is stored in the wrong split",
                    record.record_id()
                ));
            }
            if !ids.insert(record.record_id().to_owned()) {
                return validation(format!(
                    "duplicate record id {} across shards",
                    record.record_id()
                ));
            }
            if let DataRecordV3::Game(game) = record
                && !training_contents.insert(game.recompute_training_content_sha256()?)
            {
                return validation(format!(
                    "duplicate training content from game {} across shards",
                    game.game_id
                ));
            }
        }
    }
    Ok(())
}

pub fn verify_manifest_replays(
    base_dir: impl AsRef<Path>,
    manifest: &DatasetManifestV3,
    engine: &Engine,
) -> Result<(), ReplayDataV3Error> {
    let base_dir = base_dir.as_ref();
    verify_manifest_files(base_dir, manifest)?;
    for shard in &manifest.shards {
        if shard.record_schema_id != GAME_RECORD_V3_SCHEMA_ID {
            continue;
        }
        for record in read_jsonl_zstd_shard(base_dir.join(Path::new(&shard.path)))? {
            if let DataRecordV3::Game(game) = record {
                game.validate_replay(engine)?;
            }
        }
    }
    Ok(())
}

fn summarize_records(
    records: &[DataRecordV3],
    sha256: &str,
    compressed_bytes: u64,
) -> Result<ShardSummaryV3, ReplayDataV3Error> {
    let schema_id = records
        .first()
        .ok_or_else(|| ReplayDataV3Error::Validation("empty shard".to_owned()))?
        .schema_id();
    let mut summary = ShardSummaryV3 {
        sha256: sha256.to_owned(),
        compressed_bytes,
        record_schema_id: schema_id.to_owned(),
        records: records.len() as u64,
        games: 0,
        decisions: 0,
        game_steps: 0,
        policy_eligible_decisions: 0,
        value_eligible_decisions: 0,
    };
    for record in records {
        if record.schema_id() != schema_id {
            return validation("shard mixes record schemas".to_owned());
        }
        match record {
            DataRecordV3::Game(game) => {
                summary.games += 1;
                summary.game_steps += game.steps.len() as u64;
                summary.policy_eligible_decisions += game
                    .steps
                    .iter()
                    .filter(|step| !matches!(step.policy_target, PolicyTargetV3::None { .. }))
                    .count() as u64;
                if matches!(game.termination, TerminationV3::EngineTerminal { .. }) {
                    summary.value_eligible_decisions += game.steps.len() as u64;
                }
            }
            DataRecordV3::Decision(decision) => {
                summary.decisions += 1;
                summary.policy_eligible_decisions += u64::from(decision.targets.policy.is_some());
                summary.value_eligible_decisions += u64::from(decision.targets.value.is_some());
            }
        }
    }
    Ok(summary)
}

fn validate_ruleset(ruleset: &RulesetRefV3) -> Result<(), ReplayDataV3Error> {
    require_nonempty(&ruleset.id, "ruleset.id")?;
    validate_hex(&ruleset.manifest_sha256, 64, "ruleset.manifest_sha256")
}

fn validate_generator(generator: &GeneratorProvenanceV3) -> Result<(), ReplayDataV3Error> {
    require_nonempty(&generator.run_id, "generator.run_id")?;
    validate_hex(&generator.engine_commit, 40, "generator.engine_commit")?;
    if generator.engine_api_version == 0 {
        return validation("generator.engine_api_version must be positive".to_owned());
    }
    if generator.dirty_worktree {
        return validation("reproducible V3 data cannot come from a dirty worktree".to_owned());
    }
    require_nonempty(&generator.collector_id, "generator.collector_id")
}

fn validate_players(players: &[PlayerAgentV3; 2]) -> Result<(), ReplayDataV3Error> {
    if players[0].seat != AbsolutePlayerV3::White || players[1].seat != AbsolutePlayerV3::Black {
        return validation("players must be ordered white, black".to_owned());
    }
    for (index, player) in players.iter().enumerate() {
        require_nonempty(&player.agent_id, &format!("players[{index}].agent_id"))?;
        if let Some(model) = &player.model {
            require_nonempty(&model.model_id, &format!("players[{index}].model.model_id"))?;
            validate_hex(&model.sha256, 64, &format!("players[{index}].model.sha256"))?;
        }
    }
    Ok(())
}

fn validate_step(
    step: &GameStepV3,
    ruleset_id: &str,
    index: usize,
) -> Result<(), ReplayDataV3Error> {
    validate_hex(
        &step.state_fingerprint,
        16,
        &format!("steps[{index}].state_fingerprint"),
    )?;
    validate_hex(
        &step.next_state_fingerprint,
        16,
        &format!("steps[{index}].next_state_fingerprint"),
    )?;
    validate_observation(
        &step.observation,
        ruleset_id,
        &format!("steps[{index}].observation"),
    )?;
    if step.observation.state.ply != step.engine_ply {
        return validation(format!(
            "steps[{index}].engine_ply does not match observation"
        ));
    }
    if !matches!(step.observation.state.status, StatusV3::Ongoing) {
        return validation(format!("steps[{index}] is not an actionable ongoing state"));
    }
    validate_legal_actions(
        &step.legal_actions,
        &step.observation,
        &format!("steps[{index}].legal_actions"),
    )?;
    if step.legal_actions.actions.is_empty() {
        return validation(format!("steps[{index}] has no legal actions"));
    }
    let played = step.played_action_index as usize;
    if played >= step.legal_actions.actions.len() {
        return validation(format!(
            "steps[{index}].played_action_index is out of range"
        ));
    }
    validate_policy_target(
        &step.policy_target,
        step.legal_actions.actions.len(),
        played,
        index,
    )
}

fn validate_observation(
    observation: &ObservationV3,
    ruleset_id: &str,
    path: &str,
) -> Result<(), ReplayDataV3Error> {
    require_equal(
        &observation.schema_id,
        OBSERVATION_V3_SCHEMA_ID,
        &format!("{path}.schema_id"),
    )?;
    require_equal(
        &observation.ruleset_id,
        ruleset_id,
        &format!("{path}.ruleset_id"),
    )?;
    if observation.spatial_shape != [OBSERVATION_V3_SPATIAL_PLANES, 8, 8]
        || observation.spatial.len() != OBSERVATION_V3_SPATIAL_SIZE
    {
        return validation(format!("{path}.spatial shape or length drifted"));
    }
    if observation.global_shape != [32]
        || observation.global.len() != OBSERVATION_V3_GLOBAL_FEATURES
    {
        return validation(format!("{path}.global shape or length drifted"));
    }
    if observation
        .spatial
        .iter()
        .chain(&observation.global)
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return validation(format!("{path} contains invalid dense values"));
    }
    let mut piece_slots = BTreeSet::new();
    for piece in &observation.pieces {
        if piece.slot >= observation.piece_slot_count || piece.square >= 64 {
            return validation(format!("{path}.pieces contains an invalid local reference"));
        }
        if !piece_slots.insert(piece.slot) {
            return validation(format!("{path}.pieces contains duplicate slots"));
        }
        if !valid_piece_kind(piece.kind_code) {
            return validation(format!("{path}.pieces contains an unknown piece kind"));
        }
    }
    let mut card_slots = BTreeSet::new();
    for card in &observation.cards {
        if card.slot >= observation.card_slot_count {
            return validation(format!("{path}.cards contains an invalid local reference"));
        }
        if !card_slots.insert(card.slot) {
            return validation(format!("{path}.cards contains duplicate slots"));
        }
        require_nonempty(
            &card.definition_id,
            &format!("{path}.cards[].definition_id"),
        )?;
    }
    if observation.runtime.len() != CARD_RUNTIME_SOURCE_FIELDS.len()
        || observation
            .runtime
            .iter()
            .enumerate()
            .any(|(index, field)| field.field_id != (index + 1) as u16)
    {
        return validation(format!("{path}.runtime field IDs drifted"));
    }
    Ok(())
}

fn validate_legal_actions(
    legal: &LegalActionSetV3,
    observation: &ObservationV3,
    path: &str,
) -> Result<(), ReplayDataV3Error> {
    require_equal(
        &legal.schema_id,
        ACTION_V3_SCHEMA_ID,
        &format!("{path}.schema_id"),
    )?;
    if legal.actions.iter().collect::<BTreeSet<_>>().len() != legal.actions.len() {
        return validation(format!("{path} contains duplicate descriptors"));
    }
    for (index, action) in legal.actions.iter().enumerate() {
        validate_action_descriptor(action, observation, &format!("{path}.actions[{index}]"))?;
    }
    Ok(())
}

fn validate_action_descriptor(
    action: &ActionDescriptorV3,
    observation: &ObservationV3,
    path: &str,
) -> Result<(), ReplayDataV3Error> {
    match action {
        ActionDescriptorV3::Move {
            piece,
            piece_kind,
            from,
            to,
            promotion,
        } => {
            validate_piece_ref(*piece, *piece_kind, *from, observation, path)?;
            validate_square(*to, path)?;
            if promotion.is_some_and(|kind| !valid_piece_kind(kind)) {
                return validation(format!("{path}.promotion is not a known piece kind"));
            }
        }
        ActionDescriptorV3::PieceAbility {
            piece,
            piece_kind,
            from,
            target,
            ..
        } => {
            validate_piece_ref(*piece, *piece_kind, *from, observation, path)?;
            validate_action_target(target, path)?;
        }
        ActionDescriptorV3::Card {
            card,
            definition_id,
            target,
        } => {
            let observed = observation.cards.iter().find(|item| item.slot == *card);
            if observed.is_none_or(|item| item.definition_id != *definition_id) {
                return validation(format!("{path} does not match an observation card slot"));
            }
            validate_action_target(target, path)?;
        }
        ActionDescriptorV3::DraftPick {
            offer_index,
            definition_id,
        } => match &observation.phase {
            PhaseV3::Draft { offers, .. }
                if offers
                    .get(*offer_index as usize)
                    .is_some_and(|offer| offer == definition_id) => {}
            _ => return validation(format!("{path} does not match the draft phase")),
        },
        ActionDescriptorV3::ResolveChoice {
            choice_slot,
            handler_id,
            option_index,
            option,
        } => match &observation.phase {
            PhaseV3::AwaitingChoice {
                choice_slot: expected_slot,
                handler_id: expected_handler,
                options,
                ..
            } if expected_slot == choice_slot
                && expected_handler == handler_id
                && options
                    .get(*option_index as usize)
                    .is_some_and(|expected| expected == option) => {}
            _ => return validation(format!("{path} does not match the pending choice")),
        },
    }
    Ok(())
}

fn validate_piece_ref(
    piece: u32,
    piece_kind: u8,
    square: u8,
    observation: &ObservationV3,
    path: &str,
) -> Result<(), ReplayDataV3Error> {
    validate_square(square, path)?;
    let matched = observation
        .pieces
        .iter()
        .find(|item| item.slot == piece)
        .is_some_and(|item| {
            item.kind_code == piece_kind
                && (item.square == square
                    || item.shadow_square.is_some_and(|shadow| shadow == square))
        });
    if !matched {
        return validation(format!("{path} does not match an observation piece slot"));
    }
    Ok(())
}

fn validate_action_target(target: &ActionTargetV3, path: &str) -> Result<(), ReplayDataV3Error> {
    match target {
        ActionTargetV3::None => Ok(()),
        ActionTargetV3::Square { square } | ActionTargetV3::PieceAt { square } => {
            validate_square(*square, path)
        }
        ActionTargetV3::Squares { squares } => {
            for square in squares {
                validate_square(*square, path)?;
            }
            Ok(())
        }
    }
}

fn validate_square(square: u8, path: &str) -> Result<(), ReplayDataV3Error> {
    if square >= 64 {
        validation(format!("{path} contains square {square} outside 0..63"))
    } else {
        Ok(())
    }
}

fn valid_piece_kind(code: u8) -> bool {
    PIECE_KIND_VOCABULARY
        .iter()
        .any(|(_, kind)| kind.stable_code() == code)
}

fn validate_policy_target(
    target: &PolicyTargetV3,
    action_count: usize,
    played: usize,
    step_index: usize,
) -> Result<(), ReplayDataV3Error> {
    match target {
        PolicyTargetV3::MctsVisits {
            visit_counts,
            completed_simulations,
            root_value,
            noise_seed_hex,
            sampling_temperature,
        } => {
            validate_visits(
                visit_counts,
                *completed_simulations,
                action_count,
                step_index,
            )?;
            if visit_counts[played] == 0 {
                return validation(format!(
                    "steps[{step_index}] played an unvisited MCTS action"
                ));
            }
            validate_unit_value(*root_value, &format!("steps[{step_index}].root_value"))?;
            if !sampling_temperature.is_finite() || *sampling_temperature < 0.0 {
                return validation(format!(
                    "steps[{step_index}].sampling_temperature is invalid"
                ));
            }
            if let Some(seed) = noise_seed_hex {
                validate_hex(seed, 16, &format!("steps[{step_index}].noise_seed_hex"))?;
            }
        }
        PolicyTargetV3::BehaviorDistribution {
            probabilities,
            source,
        } => {
            validate_distribution(probabilities, action_count, step_index)?;
            if probabilities[played] <= 0.0 {
                return validation(format!(
                    "steps[{step_index}] played an action with zero behavior probability"
                ));
            }
            require_nonempty(source, &format!("steps[{step_index}].policy_target.source"))?;
        }
        PolicyTargetV3::None { reason } => {
            require_nonempty(reason, &format!("steps[{step_index}].policy_target.reason"))?;
        }
    }
    Ok(())
}

fn validate_decision_targets(
    targets: &DecisionTargetsV3,
    action_count: usize,
) -> Result<(), ReplayDataV3Error> {
    if targets.policy.is_none() && targets.value.is_none() && targets.auxiliary.is_empty() {
        return validation("decision.targets must contain at least one target".to_owned());
    }
    if let Some(policy) = &targets.policy {
        match &policy.target {
            DecisionPolicyTargetV3::MctsVisits {
                visit_counts,
                completed_simulations,
                root_value,
            } => {
                validate_visits(visit_counts, *completed_simulations, action_count, 0)?;
                validate_unit_value(*root_value, "decision.targets.policy.root_value")?;
            }
            DecisionPolicyTargetV3::BehaviorDistribution {
                probabilities,
                source,
            } => {
                validate_distribution(probabilities, action_count, 0)?;
                require_nonempty(source, "decision.targets.policy.source")?;
            }
            DecisionPolicyTargetV3::OptimalActionSet {
                action_indices,
                proof_source,
            } => {
                if action_indices.is_empty() {
                    return validation("optimal action set must not be empty".to_owned());
                }
                let unique = action_indices.iter().copied().collect::<BTreeSet<_>>();
                if unique.len() != action_indices.len()
                    || action_indices
                        .iter()
                        .any(|index| *index as usize >= action_count)
                {
                    return validation("optimal action set contains invalid indices".to_owned());
                }
                require_nonempty(proof_source, "decision.targets.policy.proof_source")?;
            }
        }
    }
    if let Some(value) = &targets.value {
        match value {
            DecisionValueTargetV3::TerminalOutcome {
                value,
                source_game_id,
            } => {
                validate_unit_value(*value, "decision.targets.value")?;
                require_nonempty(source_game_id, "decision.targets.value.source_game_id")?;
            }
            DecisionValueTargetV3::Proven {
                value,
                proof_source,
            } => {
                validate_unit_value(*value, "decision.targets.value")?;
                require_nonempty(proof_source, "decision.targets.value.proof_source")?;
            }
        }
    }
    let mut names = BTreeSet::new();
    for target in &targets.auxiliary {
        require_nonempty(&target.name, "decision.targets.auxiliary[].name")?;
        if !names.insert(target.name.as_str()) {
            return validation(format!("duplicate auxiliary target {}", target.name));
        }
        if let AuxiliaryTargetValueV3::Float(value) = target.value
            && !value.is_finite()
        {
            return validation(format!("auxiliary target {} is not finite", target.name));
        }
    }
    Ok(())
}

fn validate_visits(
    visits: &[u32],
    completed_simulations: u32,
    action_count: usize,
    index: usize,
) -> Result<(), ReplayDataV3Error> {
    if visits.len() != action_count {
        return validation(format!("target {index} visit count length mismatch"));
    }
    if completed_simulations == 0
        || visits.iter().map(|count| u64::from(*count)).sum::<u64>()
            != u64::from(completed_simulations)
    {
        return validation(format!(
            "target {index} visit counts do not match simulations"
        ));
    }
    Ok(())
}

fn validate_distribution(
    probabilities: &[f32],
    action_count: usize,
    index: usize,
) -> Result<(), ReplayDataV3Error> {
    if probabilities.len() != action_count
        || probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || (probabilities.iter().sum::<f32>() - 1.0).abs() > 1e-4
    {
        return validation(format!("target {index} behavior distribution is invalid"));
    }
    Ok(())
}

fn validate_unit_value(value: f32, path: &str) -> Result<(), ReplayDataV3Error> {
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        validation(format!("{path} must be finite and inside [-1, 1]"))
    } else {
        Ok(())
    }
}

fn validate_termination(termination: &TerminationV3) -> Result<(), ReplayDataV3Error> {
    validate_hex(
        termination.final_state_fingerprint(),
        16,
        "termination.final_state_fingerprint",
    )?;
    match termination {
        TerminationV3::EngineTerminal { reason, .. }
        | TerminationV3::Limit { reason, .. }
        | TerminationV3::Aborted { reason, .. } => require_nonempty(reason, "termination.reason"),
        TerminationV3::Error { code, .. } => require_nonempty(code, "termination.code"),
    }
}

fn validate_engine_termination(
    termination: &TerminationV3,
    status: &GameStatus,
) -> Result<(), ReplayDataV3Error> {
    match (termination, status) {
        (
            TerminationV3::EngineTerminal {
                result,
                reason,
                detail,
                ..
            },
            GameStatus::Won {
                winner,
                reason: engine_reason,
            },
        ) => {
            let expected_result = match winner {
                Player::White => GameResultV3::WhiteWin,
                Player::Black => GameResultV3::BlackWin,
            };
            let (expected_reason, expected_detail) = win_reason(engine_reason);
            if *result != expected_result || reason != expected_reason || detail != &expected_detail
            {
                return replay("terminal win metadata does not match engine status".to_owned());
            }
            Ok(())
        }
        (
            TerminationV3::EngineTerminal {
                result,
                reason,
                detail,
                ..
            },
            GameStatus::Draw {
                reason: engine_reason,
            },
        ) => {
            let (expected_reason, expected_detail) = draw_reason(engine_reason);
            if *result != GameResultV3::Draw
                || reason != expected_reason
                || detail != &expected_detail
            {
                return replay("terminal draw metadata does not match engine status".to_owned());
            }
            Ok(())
        }
        (TerminationV3::EngineTerminal { .. }, GameStatus::Ongoing) => {
            replay("record claims a terminal result but engine is ongoing".to_owned())
        }
        (
            TerminationV3::Limit { .. }
            | TerminationV3::Aborted { .. }
            | TerminationV3::Error { .. },
            GameStatus::Ongoing,
        ) => Ok(()),
        _ => replay("non-terminal termination metadata hides an engine result".to_owned()),
    }
}

fn win_reason(reason: &WinReason) -> (&'static str, Option<String>) {
    match reason {
        WinReason::KingCaptured => ("king_captured", None),
        WinReason::NoLegalActions => ("no_legal_actions", None),
        WinReason::LowerStarsOnRepetition => ("lower_stars_on_repetition", None),
        WinReason::LowerStarsAfterDeathmatch => ("lower_stars_after_deathmatch", None),
        WinReason::CardEffect(detail) => ("card_effect", Some(detail.clone())),
    }
}

fn draw_reason(reason: &DrawReason) -> (&'static str, Option<String>) {
    match reason {
        DrawReason::EqualStarsOnRepetition => ("equal_stars_on_repetition", None),
        DrawReason::EqualStarsAfterDeathmatch => ("equal_stars_after_deathmatch", None),
        DrawReason::CardEffect(detail) => ("card_effect", Some(detail.clone())),
    }
}

fn validate_split_policy(policy: &SplitPolicyV3) -> Result<(), ReplayDataV3Error> {
    require_equal(
        &policy.algorithm,
        SPLIT_POLICY_V3_ALGORITHM,
        "split_policy.algorithm",
    )?;
    require_nonempty(&policy.salt, "split_policy.salt")?;
    if u32::from(policy.train_basis_points)
        + u32::from(policy.validation_basis_points)
        + u32::from(policy.holdout_basis_points)
        != 10_000
    {
        return validation("split policy basis points must total 10000".to_owned());
    }
    Ok(())
}

fn validate_shard_path(path: &Path) -> Result<(), ReplayDataV3Error> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if !name.ends_with(".jsonl.zst") {
        return validation(format!("shard {} must end with .jsonl.zst", path.display()));
    }
    Ok(())
}

fn validate_relative_shard_path(path: &str) -> Result<(), ReplayDataV3Error> {
    if path.contains('\\') {
        return validation(format!(
            "manifest shard path {path} must use portable forward slashes"
        ));
    }
    let candidate = Path::new(path);
    if candidate.is_absolute()
        || candidate.components().any(|part| {
            matches!(
                part,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return validation(format!(
            "manifest shard path {path} is not a safe relative path"
        ));
    }
    validate_shard_path(candidate)
}

fn sha256_file(path: &Path) -> Result<String, ReplayDataV3Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(bytes_to_hex(&hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    bytes_to_hex(&Sha256::digest(bytes))
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

pub fn fingerprint_hex(value: u64) -> String {
    format!("{value:016x}")
}

fn require_nonempty(value: &str, path: &str) -> Result<(), ReplayDataV3Error> {
    if value.trim().is_empty() {
        validation(format!("{path} must not be empty"))
    } else {
        Ok(())
    }
}

fn require_equal(actual: &str, expected: &str, path: &str) -> Result<(), ReplayDataV3Error> {
    if actual != expected {
        validation(format!("{path} must be {expected}, got {actual}"))
    } else {
        Ok(())
    }
}

fn validate_hex(value: &str, length: usize, path: &str) -> Result<(), ReplayDataV3Error> {
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        validation(format!(
            "{path} must be {length} lowercase hexadecimal characters"
        ))
    } else {
        Ok(())
    }
}

fn validation<T>(message: String) -> Result<T, ReplayDataV3Error> {
    Err(ReplayDataV3Error::Validation(message))
}

fn replay<T>(message: String) -> Result<T, ReplayDataV3Error> {
    Err(ReplayDataV3Error::Replay(message))
}

#[derive(Debug)]
pub enum ReplayDataV3Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Validation(String),
    Replay(String),
}

impl fmt::Display for ReplayDataV3Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Json(error) => write!(formatter, "JSON error: {error}"),
            Self::Validation(error) => write!(formatter, "validation error: {error}"),
            Self::Replay(error) => write!(formatter, "replay error: {error}"),
        }
    }
}

impl std::error::Error for ReplayDataV3Error {}

impl From<std::io::Error> for ReplayDataV3Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for ReplayDataV3Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LegalActionSetV3, encode_observation_v3};
    use augment_core::{BUNDLED_RULESET_ID, Engine};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn ruleset() -> RulesetRefV3 {
        RulesetRefV3 {
            id: BUNDLED_RULESET_ID.to_owned(),
            manifest_sha256: "a".repeat(64),
        }
    }

    fn agents() -> [PlayerAgentV3; 2] {
        [
            PlayerAgentV3 {
                seat: AbsolutePlayerV3::White,
                agent_kind: AgentKindV3::Random,
                agent_id: "random-v1".to_owned(),
                model: None,
                config_id: None,
            },
            PlayerAgentV3 {
                seat: AbsolutePlayerV3::Black,
                agent_kind: AgentKindV3::RuleBased,
                agent_id: "rules-v1".to_owned(),
                model: None,
                config_id: None,
            },
        ]
    }

    fn one_step_limit_game(seed: u64) -> GameRecordV3 {
        let engine = Engine::bundled_20260807().unwrap();
        let state = engine.new_game(Mode::Standard, seed).unwrap();
        let observation = encode_observation_v3(&state);
        let legal_actions = LegalActionSetV3::from_state(&state, &engine).unwrap();
        let transition = legal_actions.apply(0, &state, &engine).unwrap();
        let next_fingerprint = fingerprint_hex(transition.state.fingerprint());
        let action_count = legal_actions.actions.len();
        let mut probabilities = vec![0.0; action_count];
        probabilities[0] = 1.0;
        let step = GameStepV3 {
            decision_index: 0,
            engine_ply: state.ply,
            actor: AbsolutePlayerV3::from_player(state.turn),
            state_fingerprint: fingerprint_hex(state.fingerprint()),
            observation,
            legal_actions,
            policy_target: PolicyTargetV3::BehaviorDistribution {
                probabilities,
                source: "random-v1".to_owned(),
            },
            played_action_index: 0,
            next_state_fingerprint: next_fingerprint.clone(),
        };
        let mut game = GameRecordV3 {
            record_schema_id: GAME_RECORD_V3_SCHEMA_ID.to_owned(),
            game_id: format!("test-game-{seed}"),
            ruleset: ruleset(),
            observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
            action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
            generator: GeneratorProvenanceV3 {
                run_id: "test-run".to_owned(),
                engine_commit: "b".repeat(40),
                engine_api_version: 1,
                collector_id: "data-v3-test".to_owned(),
                dirty_worktree: false,
            },
            mode_id: "standard".to_owned(),
            seed_hex: fingerprint_hex(seed),
            players: agents(),
            initial_state_fingerprint: fingerprint_hex(state.fingerprint()),
            steps: vec![step],
            termination: TerminationV3::Limit {
                limit: LimitKindV3::Decision,
                reason: "test_limit".to_owned(),
                final_state_fingerprint: next_fingerprint,
            },
            trajectory_sha256: String::new(),
        };
        game.refresh_trajectory_sha256().unwrap();
        game
    }

    fn decision_from_game(game: &GameRecordV3) -> DecisionRecordV3 {
        let step = &game.steps[0];
        DecisionRecordV3 {
            record_schema_id: DECISION_RECORD_V3_SCHEMA_ID.to_owned(),
            decision_id: "decision-1".to_owned(),
            provenance: DecisionProvenanceV3 {
                source_kind: DecisionSourceKindV3::GameRecord,
                source_domain: SourceDomainV3::AugmentChess,
                source_id: game.game_id.clone(),
                split_group_id: game.split_group_key(),
                generator_commit: Some("b".repeat(40)),
                license: None,
            },
            ruleset: game.ruleset.clone(),
            observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
            action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
            actor: step.actor,
            state_fingerprint: step.state_fingerprint.clone(),
            observation: step.observation.clone(),
            legal_actions: step.legal_actions.clone(),
            targets: DecisionTargetsV3 {
                policy: Some(DecisionPolicySupervisionV3 {
                    scope: PolicyTrainingScopeV3::AugmentPolicy,
                    target: DecisionPolicyTargetV3::OptimalActionSet {
                        action_indices: vec![0],
                        proof_source: "test-solver".to_owned(),
                    },
                }),
                value: Some(DecisionValueTargetV3::Proven {
                    value: 1.0,
                    proof_source: "test-solver".to_owned(),
                }),
                auxiliary: vec![AuxiliaryTargetV3 {
                    name: "mate_distance".to_owned(),
                    value: AuxiliaryTargetValueV3::Integer(3),
                }],
            },
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("augmentia-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn game_record_replays_through_the_engine() {
        let game = one_step_limit_game(101);
        game.validate().unwrap();
        game.validate_replay(&Engine::bundled_20260807().unwrap())
            .unwrap();
        assert_eq!(game.value_target(0), None);
    }

    #[test]
    fn non_terminal_games_never_produce_value_targets() {
        let mut game = one_step_limit_game(102);
        for termination in [
            TerminationV3::Limit {
                limit: LimitKindV3::Decision,
                reason: "limit".to_owned(),
                final_state_fingerprint: game.steps[0].next_state_fingerprint.clone(),
            },
            TerminationV3::Aborted {
                reason: "cancelled".to_owned(),
                final_state_fingerprint: game.steps[0].next_state_fingerprint.clone(),
            },
            TerminationV3::Error {
                stage: ErrorStageV3::Search,
                code: "search_failed".to_owned(),
                detail: None,
                final_state_fingerprint: game.steps[0].next_state_fingerprint.clone(),
            },
        ] {
            game.termination = termination;
            game.refresh_trajectory_sha256().unwrap();
            game.validate().unwrap();
            assert_eq!(game.value_target(0), None);
        }
    }

    #[test]
    fn only_engine_terminal_results_produce_actor_relative_values() {
        let termination = TerminationV3::EngineTerminal {
            result: GameResultV3::WhiteWin,
            reason: "king_captured".to_owned(),
            detail: None,
            final_state_fingerprint: "0".repeat(16),
        };
        assert_eq!(termination.value_for(AbsolutePlayerV3::White), Some(1.0));
        assert_eq!(termination.value_for(AbsolutePlayerV3::Black), Some(-1.0));

        let draw = TerminationV3::EngineTerminal {
            result: GameResultV3::Draw,
            reason: "equal_stars_on_repetition".to_owned(),
            detail: None,
            final_state_fingerprint: "0".repeat(16),
        };
        assert_eq!(draw.value_for(AbsolutePlayerV3::White), Some(0.0));
        assert_eq!(draw.value_for(AbsolutePlayerV3::Black), Some(0.0));
    }

    #[test]
    fn policy_and_action_drift_are_rejected() {
        let mut game = one_step_limit_game(103);
        game.steps[0].played_action_index = u32::MAX;
        assert!(matches!(
            game.validate(),
            Err(ReplayDataV3Error::Validation(_))
        ));

        let mut game = one_step_limit_game(104);
        game.steps[0].policy_target = PolicyTargetV3::MctsVisits {
            visit_counts: vec![1],
            completed_simulations: 2,
            root_value: 0.0,
            noise_seed_hex: None,
            sampling_temperature: 1.0,
        };
        game.refresh_trajectory_sha256().unwrap();
        assert!(matches!(
            game.validate(),
            Err(ReplayDataV3Error::Validation(_))
        ));
    }

    #[test]
    fn standard_chess_decisions_are_restricted_to_board_pretraining() {
        let game = one_step_limit_game(105);
        let mut decision = decision_from_game(&game);
        decision.provenance.source_domain = SourceDomainV3::StandardChess;
        assert!(matches!(
            decision.validate(),
            Err(ReplayDataV3Error::Validation(_))
        ));
        decision.targets.policy.as_mut().unwrap().scope = PolicyTrainingScopeV3::BoardPretraining;
        decision.targets.value = None;
        decision.validate().unwrap();
    }

    #[test]
    fn shard_round_trip_and_manifest_checksums_are_verified() {
        let root = temp_dir("data-v3-shard");
        let split_policy = SplitPolicyV3 {
            algorithm: SPLIT_POLICY_V3_ALGORITHM.to_owned(),
            salt: "test-split-v1".to_owned(),
            train_basis_points: 9_000,
            validation_basis_points: 500,
            holdout_basis_points: 500,
        };
        let game = one_step_limit_game(106);
        let record = DataRecordV3::Game(Box::new(game.clone()));
        let split = split_policy.assign(&record.split_group_key()).unwrap();
        let relative = format!("shards/{split:?}/part-00000.jsonl.zst").to_lowercase();
        let path = root.join(Path::new(&relative));
        let summary = write_jsonl_zstd_shard(&path, std::slice::from_ref(&record), 3).unwrap();
        let loaded = read_jsonl_zstd_shard(&path).unwrap();
        assert_eq!(loaded, vec![record]);

        let entry = summary.to_manifest_entry(relative, split);
        let manifest = DatasetManifestV3 {
            manifest_schema_id: DATASET_MANIFEST_V3_SCHEMA_ID.to_owned(),
            dataset_id: "test-dataset".to_owned(),
            created_at_utc: "2026-08-21T00:00:00Z".to_owned(),
            observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
            action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
            split_policy,
            compression: CompressionV3 {
                codec: "zstd".to_owned(),
                level: 3,
            },
            shards: vec![entry],
            totals: DatasetTotalsV3 {
                shards: 1,
                records: 1,
                games: 1,
                decisions: 0,
                game_steps: 1,
                policy_eligible_decisions: 1,
                value_eligible_decisions: 0,
            },
        };
        validate_manifest(&manifest).unwrap();
        verify_manifest_files(&root, &manifest).unwrap();
        verify_manifest_replays(&root, &manifest, &Engine::bundled_20260807().unwrap()).unwrap();

        let mut corrupted = manifest;
        corrupted.shards[0].sha256 = "0".repeat(64);
        assert!(matches!(
            verify_manifest_files(&root, &corrupted),
            Err(ReplayDataV3Error::Validation(_))
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn decision_records_round_trip_in_their_own_shard() {
        let root = temp_dir("decision-v3-shard");
        let path = root.join("decision.jsonl.zst");
        let decision = decision_from_game(&one_step_limit_game(107));
        decision.validate().unwrap();
        let record = DataRecordV3::Decision(Box::new(decision));
        let summary = write_jsonl_zstd_shard(&path, std::slice::from_ref(&record), 3).unwrap();
        assert_eq!(summary.decisions, 1);
        assert_eq!(summary.games, 0);
        assert_eq!(read_jsonl_zstd_shard(&path).unwrap(), vec![record]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn duplicate_training_content_is_rejected_before_a_shard_is_published() {
        let root = temp_dir("duplicate-v3-shard");
        let path = root.join("duplicate.jsonl.zst");
        let first = one_step_limit_game(108);
        let mut duplicate = first.clone();
        duplicate.game_id = "different-id-same-trajectory".to_owned();
        let result = write_jsonl_zstd_shard(
            &path,
            &[
                DataRecordV3::Game(Box::new(first)),
                DataRecordV3::Game(Box::new(duplicate)),
            ],
            3,
        );
        assert!(matches!(result, Err(ReplayDataV3Error::Validation(_))));
        assert!(!path.exists());
        if root.exists() {
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn the_same_trajectory_with_different_policy_targets_is_preserved() {
        let root = temp_dir("multi-target-v3-shard");
        let path = root.join("multi-target.jsonl.zst");
        let first = one_step_limit_game(109);
        let mut second = first.clone();
        second.game_id = "same-trajectory-new-search".to_owned();
        let action_count = second.steps[0].legal_actions.actions.len();
        second.steps[0].policy_target = PolicyTargetV3::BehaviorDistribution {
            probabilities: vec![1.0 / action_count as f32; action_count],
            source: "different-search".to_owned(),
        };
        assert_eq!(first.trajectory_sha256, second.trajectory_sha256);
        assert_ne!(
            first.recompute_training_content_sha256().unwrap(),
            second.recompute_training_content_sha256().unwrap()
        );
        let summary = write_jsonl_zstd_shard(
            &path,
            &[
                DataRecordV3::Game(Box::new(first)),
                DataRecordV3::Game(Box::new(second)),
            ],
            3,
        )
        .unwrap();
        assert_eq!(summary.games, 2);
        std::fs::remove_dir_all(root).unwrap();
    }
}

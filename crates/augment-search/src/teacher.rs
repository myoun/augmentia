//! Teacher records: one JSONL line per search decision, with everything a learner needs.
//!
//! The record is the durable artifact of the search teacher. It carries the position it decided
//! (P2-1 provenance), the features a policy consumes, and the candidate set the teacher chose from.
//! A learner needs the whole candidate set to compare the teacher's choice against alternatives,
//! and a verifier needs it to measure top-k agreement.
//!
//! Records are self-contained: position and action features live in the line, so a verifier needs
//! neither the engine nor the original game to check a policy against them.

use augment_core::{Action, Engine, GameState, GameStatus, Mode, Player};
use serde::Serialize;

use crate::features::{ActionFeatures, PositionFeatures, action_features, position_features};
use crate::{SearchConfig, SearchReport, search};

/// One candidate the teacher chose among.
#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    /// Position in the engine's enumeration, so a record is reproducible.
    pub index: usize,
    /// Whether this was the teacher's choice.
    pub chosen: bool,
    /// Debug form of the action, for human reading only.
    pub action: String,
    pub features: ActionFeatures,
}

/// What the teacher decided about one position.
#[derive(Clone, Debug, Serialize)]
pub struct TeacherRecord {
    pub ruleset_id: String,
    pub ply: u32,
    pub actor: String,
    /// P2-1 provenance: the exact position, engine identity and budget behind this decision.
    pub provenance: crate::Provenance,
    pub position: PositionFeatures,
    pub candidates: Vec<Candidate>,
    /// The chosen action in debug form, matching the candidate marked `chosen`.
    pub chosen: Option<String>,
    pub score: f64,
    pub completed_depth: u8,
    /// Root coverage: how many candidates were enumerated and how many the search scored.
    pub root_actions: usize,
    pub scored_root_actions: usize,
    pub stop_reason: String,
    /// Facts kept as text rather than baked into the fixed feature vector.
    pub info: RecordInfo,
}

/// Supplemental facts a reader needs but the fixed feature vector does not carry.
#[derive(Clone, Debug, Serialize)]
pub struct RecordInfo {
    pub phase: String,
    pub cards: usize,
    pub actions_remaining: u16,
}

/// Build a record from a state and the report the teacher produced for it.
///
/// `actions` is the engine's root enumeration for this state. Passing it in keeps the record exact:
/// the candidates in the record are the candidates the teacher saw.
pub fn record_decision(
    state: &GameState,
    actions: &[Action],
    report: &SearchReport,
    actor: Player,
) -> TeacherRecord {
    let candidates = actions
        .iter()
        .enumerate()
        .map(|(index, action)| Candidate {
            index,
            chosen: report.selected_action.as_ref() == Some(action),
            action: format!("{action:?}"),
            features: action_features(state, action),
        })
        .collect();
    TeacherRecord {
        ruleset_id: state.ruleset_id.clone(),
        ply: state.ply,
        actor: format!("{actor:?}"),
        provenance: report.provenance.clone(),
        position: position_features(state, actor),
        candidates,
        chosen: report.action.clone(),
        score: report.score,
        completed_depth: report.completed_depth,
        root_actions: report.root_actions,
        scored_root_actions: report.scored_root_actions,
        stop_reason: report.stop_reason.clone(),
        info: RecordInfo {
            phase: format!("{:?}", state.phase),
            cards: state.cards.len(),
            actions_remaining: state.actions_remaining,
        },
    }
}

/// Play a self-play game and emit one teacher record per decision.
///
/// This is the collector the learning pipeline consumes: it walks a real game so every recorded
/// position is reachable, and records what the teacher decided at each one.
pub fn collect_game(
    engine: &Engine,
    seed: u64,
    config: SearchConfig,
    max_actions: u32,
) -> (Vec<TeacherRecord>, GameStatus) {
    let mut records = Vec::new();
    let mut state = match engine.new_game(Mode::Standard, seed) {
        Ok(state) => state,
        Err(_) => return (records, GameStatus::Ongoing),
    };
    let mut taken = 0;
    while state.status == GameStatus::Ongoing && taken < max_actions {
        let actions = match engine.legal_actions(&state) {
            Ok(actions) => actions,
            Err(_) => break,
        };
        let report = match search(engine, &state, config.clone()) {
            Ok(report) => report,
            Err(_) => break,
        };
        records.push(record_decision(&state, &actions, &report, state.turn));
        let Some(action) = report.selected_action else {
            break;
        };
        match engine.apply(&state, action) {
            Ok(transition) => state = transition.state,
            Err(_) => break,
        }
        taken += 1;
    }
    (records, state.status)
}

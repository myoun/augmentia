//! V4 replay contract for factorized MCTS targets.

use crate::data_v3::{
    AbsolutePlayerV3, GeneratorProvenanceV3, PlayerAgentV3, RulesetRefV3, TerminationV3,
};
use crate::observation_v3::OBSERVATION_V3_SCHEMA_ID;
use crate::{
    ActionDescriptorV3, FACTORIZED_POLICY_V4_SCHEMA_ID, FactorizedPrefixVisitsV4, LegalActionSetV3,
    ObservationV3, aggregate_factorized_prefix_visits_v4,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const GAME_RECORD_V4_SCHEMA_ID: &str = "augmentia.replay.game.v4";
pub const FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID: &str = "augmentia.factorized-mcts-visits.v2";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorizedMctsTargetV4 {
    pub schema_id: String,
    pub factorized_policy_schema_id: String,
    pub complete_visit_counts: Vec<u32>,
    pub completed_simulations: u32,
    pub root_value: f32,
    pub prefix_targets: Vec<FactorizedPrefixVisitsV4>,
    pub trie_sha256: String,
    pub search_semantics_id: String,
    pub root_selection_id: String,
    pub chance_sampling_id: String,
    pub chance_outcome_samples: u32,
}

impl FactorizedMctsTargetV4 {
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_visits(
        actions: &[ActionDescriptorV3],
        visits: Vec<u32>,
        completed_simulations: u32,
        root_value: f32,
        search_semantics_id: impl Into<String>,
        root_selection_id: impl Into<String>,
        chance_sampling_id: impl Into<String>,
        chance_outcome_samples: u32,
    ) -> Result<Self, V4DataError> {
        let prefix_targets = aggregate_factorized_prefix_visits_v4(actions, &visits)
            .map_err(|error| V4DataError(error.to_string()))?;
        let trie_sha256 = prefix_sha256(&prefix_targets)?;
        let target = Self {
            schema_id: FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID.to_owned(),
            factorized_policy_schema_id: FACTORIZED_POLICY_V4_SCHEMA_ID.to_owned(),
            complete_visit_counts: visits,
            completed_simulations,
            root_value,
            prefix_targets,
            trie_sha256,
            search_semantics_id: search_semantics_id.into(),
            root_selection_id: root_selection_id.into(),
            chance_sampling_id: chance_sampling_id.into(),
            chance_outcome_samples,
        };
        target.validate(actions)?;
        Ok(target)
    }

    pub fn validate(&self, actions: &[ActionDescriptorV3]) -> Result<(), V4DataError> {
        if self.schema_id != FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID
            || self.factorized_policy_schema_id != FACTORIZED_POLICY_V4_SCHEMA_ID
        {
            return invalid("factorized target schema mismatch");
        }
        if actions.is_empty() || self.complete_visit_counts.len() != actions.len() {
            return invalid("complete visits must align with legal actions");
        }
        if self.complete_visit_counts.iter().copied().sum::<u32>() != self.completed_simulations
            || self.completed_simulations == 0
        {
            return invalid("complete visits must sum to completed simulations");
        }
        if !self.root_value.is_finite() || !(-1.0001..=1.0001).contains(&self.root_value) {
            return invalid("root value must be finite and in [-1, 1]");
        }
        if self.search_semantics_id.is_empty()
            || self.root_selection_id.is_empty()
            || self.chance_sampling_id.is_empty()
            || !(1..=64).contains(&self.chance_outcome_samples)
        {
            return invalid("factorized target provenance and chance samples must be valid");
        }
        let expected = aggregate_factorized_prefix_visits_v4(actions, &self.complete_visit_counts)
            .map_err(|error| V4DataError(error.to_string()))?;
        if expected != self.prefix_targets {
            return invalid("prefix targets do not match complete visits and legal actions");
        }
        let expected_hash = prefix_sha256(&expected)?;
        if self.trie_sha256 != expected_hash {
            return invalid("factorized trie checksum mismatch");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameStepV4 {
    pub decision_index: u32,
    pub engine_ply: u32,
    pub actor: AbsolutePlayerV3,
    pub state_fingerprint: String,
    pub observation: ObservationV3,
    pub legal_actions: LegalActionSetV3,
    pub policy_target: FactorizedMctsTargetV4,
    pub played_action_index: u32,
    pub next_state_fingerprint: String,
}

impl GameStepV4 {
    pub fn validate(&self) -> Result<(), V4DataError> {
        let played = self.played_action_index as usize;
        if played >= self.legal_actions.len() {
            return invalid("played action index is out of range");
        }
        self.policy_target.validate(&self.legal_actions.actions)?;
        if self.policy_target.complete_visit_counts[played] == 0 {
            return invalid("played action must have a positive complete-action visit");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameRecordV4 {
    pub record_schema_id: String,
    pub game_id: String,
    pub ruleset: RulesetRefV3,
    pub observation_schema_id: String,
    pub action_schema_id: String,
    pub factorized_policy_schema_id: String,
    pub generator: GeneratorProvenanceV3,
    pub mode_id: String,
    pub seed_hex: String,
    pub players: [PlayerAgentV3; 2],
    pub initial_state_fingerprint: String,
    pub steps: Vec<GameStepV4>,
    pub termination: TerminationV3,
    pub trajectory_sha256: String,
}

impl GameRecordV4 {
    pub fn validate(&self) -> Result<(), V4DataError> {
        if self.record_schema_id != GAME_RECORD_V4_SCHEMA_ID
            || self.observation_schema_id != OBSERVATION_V3_SCHEMA_ID
            || self.action_schema_id != crate::ACTION_V3_SCHEMA_ID
            || self.factorized_policy_schema_id != FACTORIZED_POLICY_V4_SCHEMA_ID
        {
            return invalid("V4 game schema mismatch");
        }
        if self.game_id.is_empty() || self.steps.is_empty() {
            return invalid("V4 game must have an ID and steps");
        }
        for (index, step) in self.steps.iter().enumerate() {
            if step.decision_index as usize != index {
                return invalid("V4 decision indexes must be contiguous");
            }
            step.validate()?;
        }
        Ok(())
    }

    pub fn recompute_trajectory_sha256(&self) -> Result<String, V4DataError> {
        #[derive(Serialize)]
        struct DigestStep<'a> {
            state_fingerprint: &'a str,
            action: &'a ActionDescriptorV3,
            next_state_fingerprint: &'a str,
        }
        let steps = self
            .steps
            .iter()
            .map(|step| {
                let action = step
                    .legal_actions
                    .actions
                    .get(step.played_action_index as usize)
                    .ok_or_else(|| V4DataError("played action index is out of range".to_owned()))?;
                Ok(DigestStep {
                    state_fingerprint: &step.state_fingerprint,
                    action,
                    next_state_fingerprint: &step.next_state_fingerprint,
                })
            })
            .collect::<Result<Vec<_>, V4DataError>>()?;
        let bytes = serde_json::to_vec(&steps).map_err(|error| V4DataError(error.to_string()))?;
        Ok(hex_sha256(&bytes))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V4DataError(pub String);

impl fmt::Display for V4DataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "V4 data validation error: {}", self.0)
    }
}
impl std::error::Error for V4DataError {}

fn prefix_sha256(prefix: &[FactorizedPrefixVisitsV4]) -> Result<String, V4DataError> {
    let bytes = serde_json::to_vec(prefix).map_err(|error| V4DataError(error.to_string()))?;
    Ok(hex_sha256(&bytes))
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn invalid<T>(message: impl Into<String>) -> Result<T, V4DataError> {
    Err(V4DataError(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actions() -> Vec<ActionDescriptorV3> {
        vec![
            ActionDescriptorV3::Move {
                piece: 1,
                piece_kind: 2,
                from: 1,
                to: 16,
                promotion: None,
            },
            ActionDescriptorV3::Move {
                piece: 1,
                piece_kind: 2,
                from: 1,
                to: 18,
                promotion: None,
            },
        ]
    }

    #[test]
    fn target_round_trip_and_prefix_conservation() {
        let target = FactorizedMctsTargetV4::from_complete_visits(
            &actions(),
            vec![3, 7],
            10,
            0.25,
            "v4.factorized",
            "gumbel-sh.v1",
            "public-information-multisample.v1",
            8,
        )
        .unwrap();
        target.validate(&actions()).unwrap();
        let root = target
            .prefix_targets
            .iter()
            .find(|item| item.prefix.is_empty())
            .unwrap();
        assert_eq!(root.visit_counts, [10]);
        let json = serde_json::to_string(&target).unwrap();
        let decoded: FactorizedMctsTargetV4 = serde_json::from_str(&json).unwrap();
        assert_eq!(target, decoded);
    }

    #[test]
    fn corrupted_prefix_target_is_rejected() {
        let mut target = FactorizedMctsTargetV4::from_complete_visits(
            &actions(),
            vec![3, 7],
            10,
            0.25,
            "v4.factorized",
            "gumbel-sh.v1",
            "public-information-multisample.v1",
            8,
        )
        .unwrap();
        target.prefix_targets[0].visit_counts[0] += 1;
        assert!(target.validate(&actions()).is_err());
    }
}

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

pub const GAME_RECORD_V5_SCHEMA_ID: &str = "augmentia.replay.game.v5";
pub const FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID: &str = "augmentia.factorized-mcts-visits.v3";
/// Source-compatible name retained for callers that predate the v5 replay bump.
pub const GAME_RECORD_V4_SCHEMA_ID: &str = GAME_RECORD_V5_SCHEMA_ID;
/// Source-compatible name retained for callers that predate the v3 target bump.
pub const FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID: &str = FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorizedMctsTargetV4 {
    pub schema_id: String,
    pub factorized_policy_schema_id: String,
    pub complete_visit_counts: Vec<u32>,
    pub opened_action_mask: Vec<bool>,
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
        opened_action_mask: Vec<bool>,
        completed_simulations: u32,
        root_value: f32,
        search_semantics_id: impl Into<String>,
        root_selection_id: impl Into<String>,
        chance_sampling_id: impl Into<String>,
        chance_outcome_samples: u32,
    ) -> Result<Self, V4DataError> {
        validate_opened_action_mask(actions, &visits, &opened_action_mask)?;
        let prefix_targets =
            aggregate_factorized_prefix_visits_v4(actions, &visits, &opened_action_mask)
                .map_err(|error| V4DataError(error.to_string()))?;
        let trie_sha256 = trie_sha256(&opened_action_mask, &prefix_targets)?;
        let target = Self {
            schema_id: FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID.to_owned(),
            factorized_policy_schema_id: FACTORIZED_POLICY_V4_SCHEMA_ID.to_owned(),
            complete_visit_counts: visits,
            opened_action_mask,
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
        if self.schema_id != FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID
            || self.factorized_policy_schema_id != FACTORIZED_POLICY_V4_SCHEMA_ID
        {
            return invalid("factorized target schema mismatch");
        }
        if actions.is_empty() || self.complete_visit_counts.len() != actions.len() {
            return invalid("complete visits must align with legal actions");
        }
        validate_opened_action_mask(
            actions,
            &self.complete_visit_counts,
            &self.opened_action_mask,
        )?;
        let visit_total = self
            .complete_visit_counts
            .iter()
            .try_fold(0u32, |total, visits| total.checked_add(*visits))
            .ok_or_else(|| V4DataError("complete visit total overflowed".to_owned()))?;
        if visit_total != self.completed_simulations || self.completed_simulations == 0 {
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
        let expected = aggregate_factorized_prefix_visits_v4(
            actions,
            &self.complete_visit_counts,
            &self.opened_action_mask,
        )
        .map_err(|error| V4DataError(error.to_string()))?;
        if expected != self.prefix_targets {
            return invalid(
                "prefix targets do not match complete visits, opened masks, and legal actions",
            );
        }
        let expected_hash = trie_sha256(&self.opened_action_mask, &expected)?;
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
    pub factorized_target_schema_id: String,
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
        if self.record_schema_id != GAME_RECORD_V5_SCHEMA_ID
            || self.observation_schema_id != OBSERVATION_V3_SCHEMA_ID
            || self.action_schema_id != crate::ACTION_V3_SCHEMA_ID
            || self.factorized_policy_schema_id != FACTORIZED_POLICY_V4_SCHEMA_ID
            || self.factorized_target_schema_id != FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID
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
        if self.trajectory_sha256 != self.recompute_trajectory_sha256()? {
            return invalid("V4 trajectory checksum mismatch");
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

fn trie_sha256(
    opened_action_mask: &[bool],
    prefix: &[FactorizedPrefixVisitsV4],
) -> Result<String, V4DataError> {
    #[derive(Serialize)]
    struct PrefixHashEnvelopeV4<'a> {
        opened_action_mask: &'a [bool],
        prefix_targets: &'a [FactorizedPrefixVisitsV4],
    }
    let bytes = serde_json::to_vec(&PrefixHashEnvelopeV4 {
        opened_action_mask,
        prefix_targets: prefix,
    })
    .map_err(|error| V4DataError(error.to_string()))?;
    Ok(hex_sha256(&bytes))
}

fn validate_opened_action_mask(
    actions: &[ActionDescriptorV3],
    visits: &[u32],
    opened_action_mask: &[bool],
) -> Result<(), V4DataError> {
    if actions.len() != opened_action_mask.len() {
        return invalid("opened action mask must align with legal actions");
    }
    if !opened_action_mask.iter().any(|opened| *opened) {
        return invalid("opened action mask must contain at least one action");
    }
    if visits
        .iter()
        .zip(opened_action_mask)
        .any(|(visits, opened)| *visits > 0 && !opened)
    {
        return invalid("every action with positive visits must be opened");
    }
    Ok(())
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
    use crate::ActionTargetV3;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SharedTargetFixtureV3 {
        actions: Vec<ActionDescriptorV3>,
        target: FactorizedMctsTargetV4,
    }

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

    fn wide_actions(count: u32) -> Vec<ActionDescriptorV3> {
        (0..count)
            .map(|card| ActionDescriptorV3::Card {
                card,
                definition_id: format!("fixture-{card}"),
                target: ActionTargetV3::None,
            })
            .collect()
    }

    #[test]
    fn shared_python_rust_v3_target_fixture_round_trips() {
        let fixture: SharedTargetFixtureV3 = serde_json::from_str(include_str!(
            "../../../train/v4/fixtures/factorized_mcts_target_v3.json"
        ))
        .unwrap();
        fixture.target.validate(&fixture.actions).unwrap();
        assert_eq!(
            fixture.target.trie_sha256,
            "e2c30b93c35768e1364720b2aaefa5cfee33bdc59c3fbb0f7debfa83e9a39605"
        );
        assert_eq!(fixture.target.opened_action_mask, [true, true, false]);
        assert_eq!(fixture.target.complete_visit_counts, [3, 0, 0]);

        let root = fixture
            .target
            .prefix_targets
            .iter()
            .find(|row| row.prefix.is_empty())
            .unwrap();
        assert!(root.opened_masks.iter().any(|opened| *opened));
        assert!(root.opened_masks.iter().any(|opened| !*opened));
        assert!(root.visit_counts.iter().any(|visits| *visits > 0));

        let rebuilt = FactorizedMctsTargetV4::from_complete_visits(
            &fixture.actions,
            fixture.target.complete_visit_counts.clone(),
            fixture.target.opened_action_mask.clone(),
            fixture.target.completed_simulations,
            fixture.target.root_value,
            fixture.target.search_semantics_id.clone(),
            fixture.target.root_selection_id.clone(),
            fixture.target.chance_sampling_id.clone(),
            fixture.target.chance_outcome_samples,
        )
        .unwrap();
        assert_eq!(rebuilt, fixture.target);

        let canonical = serde_json::to_vec(&fixture.target).unwrap();
        let decoded: FactorizedMctsTargetV4 = serde_json::from_slice(&canonical).unwrap();
        decoded.validate(&fixture.actions).unwrap();
        assert_eq!(canonical, serde_json::to_vec(&decoded).unwrap());
    }

    #[test]
    fn target_round_trip_and_prefix_conservation() {
        let target = FactorizedMctsTargetV4::from_complete_visits(
            &actions(),
            vec![3, 7],
            vec![true, true],
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
            vec![true, true],
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

    #[test]
    fn wide_mixed_prefix_excludes_closed_siblings_but_keeps_opened_zero_visit() {
        let actions = wide_actions(32);
        let mut visits = vec![0u32; actions.len()];
        for visit in visits.iter_mut().take(15) {
            *visit = 2;
        }
        let mut opened_action_mask = vec![false; actions.len()];
        for opened in opened_action_mask.iter_mut().take(16) {
            *opened = true;
        }
        let target = FactorizedMctsTargetV4::from_complete_visits(
            &actions,
            visits,
            opened_action_mask,
            30,
            0.125,
            "v4.factorized-opened.v3",
            "gumbel-sh.v1",
            "public-information-multisample.v1",
            8,
        )
        .unwrap();
        target.validate(&actions).unwrap();

        let mixed = target
            .prefix_targets
            .iter()
            .find(|row| row.prefix.as_slice() == ["family:card"])
            .unwrap();
        assert_eq!(mixed.children.len(), 32);
        assert_eq!(mixed.visit_counts.len(), 32);
        assert_eq!(mixed.opened_masks.len(), mixed.children.len());
        assert_eq!(
            mixed.opened_masks.iter().filter(|opened| **opened).count(),
            16
        );
        let opened_zero = mixed
            .children
            .iter()
            .position(|child| child == "card:15")
            .unwrap();
        let closed_zero = mixed
            .children
            .iter()
            .position(|child| child == "card:16")
            .unwrap();
        assert!(mixed.opened_masks[opened_zero]);
        assert_eq!(mixed.visit_counts[opened_zero], 0);
        assert!(!mixed.opened_masks[closed_zero]);
        assert_eq!(mixed.visit_counts[closed_zero], 0);
        let denominator = mixed
            .visit_counts
            .iter()
            .zip(&mixed.opened_masks)
            .filter(|(_, opened)| **opened)
            .map(|(visits, _)| visits)
            .sum::<u32>();
        assert_eq!(denominator, target.completed_simulations);
    }

    #[test]
    fn opened_contract_is_strict_canonical_and_fail_closed() {
        let actions = wide_actions(32);
        let mut visits = vec![0u32; actions.len()];
        visits[..16].fill(1);
        let mut opened = vec![false; actions.len()];
        opened[..15].fill(true);
        opened[15] = true;
        let target = FactorizedMctsTargetV4::from_complete_visits(
            &actions,
            visits,
            opened,
            16,
            0.0,
            "v4.factorized-opened.v3",
            "gumbel-sh.v1",
            "public-information-multisample.v1",
            8,
        )
        .unwrap();

        let canonical = serde_json::to_vec(&target).unwrap();
        let decoded: FactorizedMctsTargetV4 = serde_json::from_slice(&canonical).unwrap();
        decoded.validate(&actions).unwrap();
        assert_eq!(canonical, serde_json::to_vec(&decoded).unwrap());

        let mut missing_root = serde_json::to_value(&target).unwrap();
        missing_root
            .as_object_mut()
            .unwrap()
            .remove("opened_action_mask");
        assert!(serde_json::from_value::<FactorizedMctsTargetV4>(missing_root).is_err());

        let mut missing_prefix = serde_json::to_value(&target).unwrap();
        missing_prefix["prefix_targets"][0]
            .as_object_mut()
            .unwrap()
            .remove("opened_masks");
        assert!(serde_json::from_value::<FactorizedMctsTargetV4>(missing_prefix).is_err());

        let mut unknown_prefix = serde_json::to_value(&target).unwrap();
        unknown_prefix["prefix_targets"][0]["unexpected"] = serde_json::Value::Bool(true);
        assert!(serde_json::from_value::<FactorizedMctsTargetV4>(unknown_prefix).is_err());

        let mut positive_closed = target.clone();
        positive_closed.opened_action_mask[0] = false;
        assert!(positive_closed.validate(&actions).is_err());
    }
}

//! Canonical V4 factorized-policy paths derived from complete ActionDescriptorV3 values.
//!
//! The engine still owns complete action legality. This module only supplies the stable
//! search/training trie used to aggregate complete-action MCTS visits into conditional targets.

use crate::{ActionDescriptorV3, ActionTargetV3, PieceAbilityV3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

pub const FACTORIZED_POLICY_V4_SCHEMA_ID: &str = "augmentia.factorized-policy.v4.1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FactorizedPolicyV4Error {
    DuplicatePath,
    PrefixCollision,
    InvalidTargetSquare(u8),
    EmptySquaresTarget,
    VisitsLengthMismatch,
    OpenedMasksLengthMismatch,
    PositiveVisitClosed,
    NoOpenedActions,
    EmptyActions,
}

impl fmt::Display for FactorizedPolicyV4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicatePath => write!(
                formatter,
                "complete actions must have unique canonical paths"
            ),
            Self::PrefixCollision => {
                write!(formatter, "a complete action is a prefix of another action")
            }
            Self::InvalidTargetSquare(square) => {
                write!(formatter, "target square {square} is outside 0..64")
            }
            Self::EmptySquaresTarget => {
                write!(formatter, "squares target must contain at least one square")
            }
            Self::VisitsLengthMismatch => write!(
                formatter,
                "actions and visits must be non-empty and aligned"
            ),
            Self::OpenedMasksLengthMismatch => write!(
                formatter,
                "actions and opened masks must be non-empty and aligned"
            ),
            Self::PositiveVisitClosed => {
                write!(formatter, "an action with positive visits must be opened")
            }
            Self::NoOpenedActions => write!(formatter, "at least one action must be opened"),
            Self::EmptyActions => {
                write!(formatter, "factorized policy requires at least one action")
            }
        }
    }
}

impl std::error::Error for FactorizedPolicyV4Error {}

#[derive(Clone, Debug, Default)]
pub struct FactorizedTrieV4 {
    children: BTreeMap<String, FactorizedTrieV4>,
    action_index: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorizedPrefixVisitsV4 {
    pub prefix: Vec<String>,
    pub children: Vec<String>,
    pub visit_counts: Vec<u32>,
    /// One entry per direct child in canonical order.
    pub opened_masks: Vec<bool>,
}

pub fn factorized_action_path_v4(
    action: &ActionDescriptorV3,
) -> Result<Vec<String>, FactorizedPolicyV4Error> {
    let mut path = Vec::new();
    match action {
        ActionDescriptorV3::Move {
            piece,
            from,
            to,
            promotion,
            ..
        } => {
            path.push("family:move".to_owned());
            path.push(format!("piece:{piece}"));
            path.push(format!("source_square:{from}"));
            path.push(format!("destination:{to}"));
            path.push(match promotion {
                Some(value) => format!("promotion:{value}"),
                None => "promotion:none".to_owned(),
            });
        }
        ActionDescriptorV3::PieceAbility {
            piece,
            ability,
            target,
            ..
        } => {
            path.push("family:piece_ability".to_owned());
            path.push(format!("piece:{piece}"));
            path.push(format!("ability:{}", ability_name(*ability)));
            path.extend(target_path(target)?);
        }
        ActionDescriptorV3::Card { card, target, .. } => {
            path.push("family:card".to_owned());
            path.push(format!("card:{card}"));
            path.extend(target_path(target)?);
        }
        ActionDescriptorV3::DraftPick { offer_index, .. } => {
            path.push("family:draft_pick".to_owned());
            path.push(format!("offer:{offer_index}"));
        }
        ActionDescriptorV3::ResolveChoice {
            choice_slot,
            option_index,
            ..
        } => {
            path.push("family:resolve_choice".to_owned());
            path.push(format!("choice:{choice_slot}"));
            path.push(format!("option:{option_index}"));
        }
    }
    path.push("complete:action".to_owned());
    Ok(path)
}

pub fn build_factorized_trie_v4(
    actions: &[ActionDescriptorV3],
) -> Result<FactorizedTrieV4, FactorizedPolicyV4Error> {
    if actions.is_empty() {
        return Err(FactorizedPolicyV4Error::EmptyActions);
    }
    let mut root = FactorizedTrieV4::default();
    for (action_index, action) in actions.iter().enumerate() {
        let mut node = &mut root;
        for token in factorized_action_path_v4(action)? {
            if node.action_index.is_some() {
                return Err(FactorizedPolicyV4Error::PrefixCollision);
            }
            node = node.children.entry(token).or_default();
        }
        if node.action_index.is_some() || !node.children.is_empty() {
            return Err(FactorizedPolicyV4Error::DuplicatePath);
        }
        node.action_index = Some(action_index);
    }
    Ok(root)
}

pub fn aggregate_factorized_prefix_visits_v4(
    actions: &[ActionDescriptorV3],
    visits: &[u32],
    opened_masks: &[bool],
) -> Result<Vec<FactorizedPrefixVisitsV4>, FactorizedPolicyV4Error> {
    if actions.is_empty() {
        return Err(FactorizedPolicyV4Error::EmptyActions);
    }
    if actions.len() != visits.len() {
        return Err(FactorizedPolicyV4Error::VisitsLengthMismatch);
    }
    if actions.len() != opened_masks.len() {
        return Err(FactorizedPolicyV4Error::OpenedMasksLengthMismatch);
    }
    if !opened_masks.iter().any(|opened| *opened) {
        return Err(FactorizedPolicyV4Error::NoOpenedActions);
    }
    if visits
        .iter()
        .zip(opened_masks)
        .any(|(visit, opened)| *visit > 0 && !opened)
    {
        return Err(FactorizedPolicyV4Error::PositiveVisitClosed);
    }
    let trie = build_factorized_trie_v4(actions)?;
    let mut output = Vec::new();
    aggregate_node(&trie, &mut Vec::new(), visits, opened_masks, &mut output)?;
    Ok(output)
}

fn aggregate_node(
    node: &FactorizedTrieV4,
    prefix: &mut Vec<String>,
    visits: &[u32],
    opened_masks: &[bool],
    output: &mut Vec<FactorizedPrefixVisitsV4>,
) -> Result<(u32, bool), FactorizedPolicyV4Error> {
    if node.children.is_empty() {
        let action_index = node
            .action_index
            .ok_or(FactorizedPolicyV4Error::PrefixCollision)?;
        let visit = *visits
            .get(action_index)
            .ok_or(FactorizedPolicyV4Error::VisitsLengthMismatch)?;
        let opened = *opened_masks
            .get(action_index)
            .ok_or(FactorizedPolicyV4Error::OpenedMasksLengthMismatch)?;
        if visit > 0 && !opened {
            return Err(FactorizedPolicyV4Error::PositiveVisitClosed);
        }
        return Ok((visit, opened));
    }
    let mut children = Vec::with_capacity(node.children.len());
    let mut counts = Vec::with_capacity(node.children.len());
    let mut child_opened = Vec::with_capacity(node.children.len());
    let mut total = 0u32;
    let mut any_opened = false;
    for (token, child) in &node.children {
        prefix.push(token.clone());
        let (count, opened) = aggregate_node(child, prefix, visits, opened_masks, output)?;
        prefix.pop();
        total = total
            .checked_add(count)
            .ok_or(FactorizedPolicyV4Error::VisitsLengthMismatch)?;
        any_opened |= opened;
        children.push(token.clone());
        counts.push(count);
        child_opened.push(opened);
    }
    output.push(FactorizedPrefixVisitsV4 {
        prefix: prefix.clone(),
        children,
        visit_counts: counts,
        opened_masks: child_opened,
    });
    Ok((total, any_opened))
}

fn target_path(target: &ActionTargetV3) -> Result<Vec<String>, FactorizedPolicyV4Error> {
    match target {
        ActionTargetV3::None => Ok(vec!["target_kind:none".to_owned()]),
        ActionTargetV3::Square { square } => one_square_target("square", *square),
        ActionTargetV3::PieceAt { square } => one_square_target("piece_at", *square),
        ActionTargetV3::Squares { squares } => {
            if squares.is_empty() {
                return Err(FactorizedPolicyV4Error::EmptySquaresTarget);
            }
            let mut path = vec!["target_kind:squares".to_owned()];
            for (position, square) in squares.iter().copied().enumerate() {
                validate_square(square)?;
                path.push(format!("target_{position}:{square}"));
            }
            Ok(path)
        }
    }
}

fn one_square_target(kind: &str, square: u8) -> Result<Vec<String>, FactorizedPolicyV4Error> {
    validate_square(square)?;
    Ok(vec![
        format!("target_kind:{kind}"),
        format!("target_square:{square}"),
    ])
}

fn validate_square(square: u8) -> Result<(), FactorizedPolicyV4Error> {
    if square >= 64 {
        Err(FactorizedPolicyV4Error::InvalidTargetSquare(square))
    } else {
        Ok(())
    }
}

fn ability_name(ability: PieceAbilityV3) -> &'static str {
    match ability {
        PieceAbilityV3::WizardLightning => "wizard_lightning",
        PieceAbilityV3::WizardShield => "wizard_shield",
        PieceAbilityV3::WizardMeteor => "wizard_meteor",
        PieceAbilityV3::WizardTimeStop => "wizard_time_stop",
        PieceAbilityV3::ColossusSector => "colossus_sector",
        PieceAbilityV3::MerchantBuy => "merchant_buy",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct GoldenFixture {
        cases: Vec<GoldenCase>,
    }

    #[derive(Deserialize)]
    struct GoldenCase {
        action: ActionDescriptorV3,
        path: Vec<String>,
    }

    #[test]
    fn shared_python_rust_golden_paths_match() {
        let fixture: GoldenFixture = serde_json::from_str(include_str!(
            "../../../train/v3/fixtures/factorized_policy_v4.json"
        ))
        .unwrap();
        for case in fixture.cases {
            assert_eq!(factorized_action_path_v4(&case.action).unwrap(), case.path);
        }
    }

    #[test]
    fn aggregate_visits_preserves_total_at_root() {
        let actions = vec![
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
        ];
        let targets =
            aggregate_factorized_prefix_visits_v4(&actions, &[3, 7], &[true, true]).unwrap();
        let root = targets
            .iter()
            .find(|target| target.prefix.is_empty())
            .unwrap();
        assert_eq!(root.children, ["family:move"]);
        assert_eq!(root.visit_counts, [10]);
    }

    #[test]
    fn target_sequences_have_distinct_canonical_paths() {
        let first = ActionDescriptorV3::Card {
            card: 2,
            definition_id: "example".to_owned(),
            target: ActionTargetV3::Squares { squares: vec![1] },
        };
        let second = ActionDescriptorV3::Card {
            card: 2,
            definition_id: "example".to_owned(),
            target: ActionTargetV3::Squares {
                squares: vec![1, 9],
            },
        };
        assert_ne!(
            factorized_action_path_v4(&first).unwrap(),
            factorized_action_path_v4(&second).unwrap()
        );
        build_factorized_trie_v4(&[first, second]).unwrap();
    }

    #[test]
    fn move_source_square_disambiguates_quantum_shadow_paths() {
        let physical = ActionDescriptorV3::Move {
            piece: 7,
            piece_kind: 4,
            from: 8,
            to: 24,
            promotion: None,
        };
        let shadow = ActionDescriptorV3::Move {
            piece: 7,
            piece_kind: 4,
            from: 16,
            to: 24,
            promotion: None,
        };
        assert_ne!(
            factorized_action_path_v4(&physical).unwrap(),
            factorized_action_path_v4(&shadow).unwrap()
        );
        build_factorized_trie_v4(&[physical, shadow]).unwrap();
    }

    #[test]
    fn duplicate_complete_actions_are_rejected() {
        let action = ActionDescriptorV3::DraftPick {
            offer_index: 0,
            definition_id: "siren".to_owned(),
        };
        assert_eq!(
            build_factorized_trie_v4(&[action.clone(), action]).unwrap_err(),
            FactorizedPolicyV4Error::DuplicatePath,
        );
    }
}

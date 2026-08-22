use serde::Deserialize;
use std::collections::BTreeSet;
use std::fmt;

use crate::{SUPPORTED_ENGINE_API_VERSION, SUPPORTED_SCHEMA_VERSION};

const BUNDLED_20260818: &str =
    include_str!("../../../rulesets/augment_chess_20260818/manifest.json");

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct RulesetManifest {
    pub id: String,
    pub snapshot_date: String,
    pub schema_version: u32,
    pub engine_api_version: u32,
    pub source: SourceLinks,
    pub artifacts: Artifacts,
    pub coverage: Coverage,
    pub base_rules: BaseRules,
    pub opening_rule: OpeningRuleConfig,
    pub modes: Vec<ModeConfig>,
}

impl RulesetManifest {
    pub fn bundled_20260807() -> Result<Self, RulesetError> {
        Self::bundled_20260818()
    }

    pub fn bundled_20260818() -> Result<Self, RulesetError> {
        Self::from_json(BUNDLED_20260818)
    }

    pub fn from_json(json: &str) -> Result<Self, RulesetError> {
        let manifest: Self = serde_json::from_str(json)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), RulesetError> {
        let expected_suffix = self.snapshot_date.replace('-', "");
        if !self.id.starts_with("augment_chess_") || !self.id.ends_with(&expected_suffix) {
            return Err(RulesetError::InvalidManifest(format!(
                "ruleset id {:?} must end with snapshot date {expected_suffix}",
                self.id
            )));
        }
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(RulesetError::InvalidManifest(format!(
                "unsupported schema_version {}; engine supports {}",
                self.schema_version, SUPPORTED_SCHEMA_VERSION
            )));
        }
        if self.engine_api_version != SUPPORTED_ENGINE_API_VERSION {
            return Err(RulesetError::InvalidManifest(format!(
                "unsupported engine_api_version {}; engine supports {}",
                self.engine_api_version, SUPPORTED_ENGINE_API_VERSION
            )));
        }
        if self.coverage.implemented_card_count > self.coverage.target_card_count {
            return Err(RulesetError::InvalidManifest(
                "implemented_card_count cannot exceed target_card_count".to_owned(),
            ));
        }
        if self.coverage.behavioral_slice_card_count < self.coverage.implemented_card_count
            || self.coverage.behavioral_slice_card_count > self.coverage.target_card_count
            || self.coverage.site_parity_verified_card_count > self.coverage.implemented_card_count
        {
            return Err(RulesetError::InvalidManifest(
                "card coverage counters are inconsistent".to_owned(),
            ));
        }
        if self.base_rules.repetition_count < 2 {
            return Err(RulesetError::InvalidManifest(
                "repetition_count must be at least two".to_owned(),
            ));
        }
        if self.base_rules.deathmatch_starts_at_move_count == 0
            || self.base_rules.deathmatch_quiet_move_count == 0
        {
            return Err(RulesetError::InvalidManifest(
                "deathmatch thresholds must be positive".to_owned(),
            ));
        }
        if self.opening_rule.chance_denominator == 0
            || self.opening_rule.chance_numerator > self.opening_rule.chance_denominator
        {
            return Err(RulesetError::InvalidManifest(
                "opening RULE chance must be a valid fraction".to_owned(),
            ));
        }
        if self.base_rules.repetition_tiebreak != "lower_half_star_total" {
            return Err(RulesetError::InvalidManifest(format!(
                "unsupported repetition_tiebreak {:?}",
                self.base_rules.repetition_tiebreak
            )));
        }

        let mut mode_ids = BTreeSet::new();
        for mode in &self.modes {
            if !mode_ids.insert(mode.id.as_str()) {
                return Err(RulesetError::InvalidManifest(format!(
                    "duplicate mode id {:?}",
                    mode.id
                )));
            }
        }
        for required in ["standard", "chaos", "grand"] {
            if !mode_ids.contains(required) {
                return Err(RulesetError::InvalidManifest(format!(
                    "missing required mode {required:?}"
                )));
            }
        }
        Ok(())
    }

    pub fn mode(&self, id: &str) -> Option<&ModeConfig> {
        self.modes.iter().find(|mode| mode.id == id)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct SourceLinks {
    pub site: String,
    pub rules: String,
    pub updates: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct Artifacts {
    pub source_manifest: String,
    pub catalog: String,
    pub implemented_cards: String,
    pub notation: String,
    pub notation_raw: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
pub struct OpeningRuleConfig {
    pub enabled: bool,
    pub chance_numerator: u64,
    pub chance_denominator: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct Coverage {
    pub status: String,
    pub target_card_count: u32,
    pub implemented_card_count: u32,
    pub behavioral_slice_card_count: u32,
    pub site_parity_verified_card_count: u32,
    pub notes: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct BaseRules {
    pub king_capture_wins: bool,
    pub immobility_loses: bool,
    pub repetition_count: u8,
    pub repetition_tiebreak: String,
    pub deathmatch_starts_at_move_count: u32,
    pub deathmatch_quiet_move_count: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct ModeConfig {
    pub id: String,
    pub draft: DraftConfig,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DraftConfig {
    Rounds {
        checkpoints: Vec<String>,
        bundle_count: u16,
        cards_per_bundle: u16,
        picks_per_round: u16,
    },
    SharedPool {
        pool_size: u16,
        opening_cards: u16,
        middle_cards: u16,
        piece_cards: u16,
        end_cards: u16,
        picks_per_player: u16,
        first_player: String,
        has_midgame_drafts: bool,
    },
}

#[derive(Debug)]
pub enum RulesetError {
    InvalidJson(serde_json::Error),
    InvalidManifest(String),
}

impl fmt::Display for RulesetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(error) => write!(f, "invalid ruleset JSON: {error}"),
            Self::InvalidManifest(message) => write!(f, "invalid ruleset manifest: {message}"),
        }
    }
}

impl std::error::Error for RulesetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidJson(error) => Some(error),
            Self::InvalidManifest(_) => None,
        }
    }
}

impl From<serde_json::Error> for RulesetError {
    fn from(value: serde_json::Error) -> Self {
        Self::InvalidJson(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BUNDLED_RULESET_ID;

    #[test]
    fn bundled_manifest_is_valid_and_explicitly_incomplete() {
        let manifest = RulesetManifest::bundled_20260807().unwrap();
        assert_eq!(manifest.id, BUNDLED_RULESET_ID);
        assert_eq!(manifest.snapshot_date, "2026-08-18");
        assert_eq!(manifest.coverage.status, "partial");
        assert_eq!(manifest.coverage.target_card_count, 208);
        assert_eq!(manifest.coverage.implemented_card_count, 208);
        assert_eq!(manifest.coverage.behavioral_slice_card_count, 208);
        assert_eq!(manifest.coverage.site_parity_verified_card_count, 0);
        assert_eq!(manifest.base_rules.deathmatch_starts_at_move_count, 45);
        assert_eq!(manifest.base_rules.deathmatch_quiet_move_count, 10);
        assert_eq!(manifest.opening_rule.chance_numerator, 2);
        assert_eq!(manifest.opening_rule.chance_denominator, 5);
        assert_eq!(
            manifest.artifacts.source_manifest,
            "sources/source-manifest.json"
        );
        assert_eq!(
            manifest.artifacts.implemented_cards,
            "cards/implemented.json"
        );
        assert_eq!(manifest.artifacts.catalog, "cards/catalog.json");
        assert_eq!(manifest.artifacts.notation, "notation.json");
    }

    #[test]
    fn all_public_modes_are_present() {
        let manifest = RulesetManifest::bundled_20260807().unwrap();
        for mode in ["standard", "chaos", "grand"] {
            assert!(manifest.mode(mode).is_some());
        }
    }
}

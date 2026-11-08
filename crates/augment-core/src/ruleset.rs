use serde::Deserialize;
use std::collections::BTreeSet;
use std::fmt;

use crate::{SUPPORTED_ENGINE_API_VERSION, SUPPORTED_SCHEMA_VERSION};

const BUNDLED_20260818: &str =
    include_str!("../../../rulesets/augment_chess_20260818/manifest.json");
const BUNDLED_20260915: &str =
    include_str!("../../../rulesets/augment_chess_20260915/manifest.json");
const BUNDLED_20261017: &str =
    include_str!("../../../rulesets/augment_chess_20261017/manifest.json");
const BUNDLED_20261102: &str =
    include_str!("../../../rulesets/augment_chess_20261102/manifest.json");

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
    /// Present once the snapshot is released. Absent while it is still in progress,
    /// so earlier snapshots keep parsing unchanged.
    #[serde(default)]
    pub release: Option<Release>,
}

/// Identifies the exact code state of a released snapshot.
///
/// A manifest cannot record the hash of the commit that contains it, because writing
/// that hash changes it. So `engine_commit` names the commit that carries the engine
/// and profile being released, while the tag points at a later commit whose only
/// change is this attestation. `scripts/verify-release-attestation.mjs` checks that
/// the two commits differ by nothing else.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct Release {
    /// Release revision within this snapshot, such as `r1`.
    pub revision: String,
    /// Annotated tag that marks the release.
    pub ruleset_tag: String,
    /// Commit holding the released engine state: 40 lowercase hex characters.
    #[serde(default)]
    pub engine_commit: Option<String>,
}

impl Release {
    /// Whether the engine commit is a full, well-formed hash.
    pub fn engine_commit_is_recorded(&self) -> bool {
        self.engine_commit.as_ref().is_some_and(|commit| {
            commit.len() == 40
                && commit
                    .bytes()
                    .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        })
    }

    /// The tag must be named for this snapshot and revision, so a tag cannot be
    /// retargeted to a manifest that does not describe it.
    pub fn tag_matches(&self, ruleset_id: &str) -> bool {
        self.ruleset_tag == format!("ruleset/{ruleset_id}-{}", self.revision)
    }
}

impl RulesetManifest {
    pub fn bundled_latest() -> Result<Self, RulesetError> {
        Self::from_json(BUNDLED_20261102)
    }

    /// The previous snapshot, kept for the transition tests that pin historical
    /// behavior while the current profile catches up.
    pub fn bundled_20261017() -> Result<Self, RulesetError> {
        Self::from_json(BUNDLED_20261017)
    }

    pub fn bundled_20260807() -> Result<Self, RulesetError> {
        Self::bundled_20260818()
    }

    pub fn bundled_20260818() -> Result<Self, RulesetError> {
        Self::from_json(BUNDLED_20260818)
    }

    /// Announced September rules. The legacy constructors remain frozen metadata loaders.
    pub fn bundled_20260915() -> Result<Self, RulesetError> {
        Self::from_json(BUNDLED_20260915)
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
        if self.coverage.publicly_active_card_count > self.coverage.target_card_count {
            return Err(RulesetError::InvalidManifest(
                "publicly_active_card_count cannot exceed target_card_count".to_owned(),
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
        // `status` is a claim about the publicly active catalog, so it must agree with
        // the counters rather than being trusted on its own.
        let expected_status = if self.coverage.covers_all_public_cards() {
            "publicly_active"
        } else {
            "partial"
        };
        if self.coverage.status != expected_status {
            return Err(RulesetError::InvalidManifest(format!(
                "coverage status must be {expected_status:?} for {} of {} active cards",
                self.coverage.implemented_card_count, self.coverage.publicly_active_card_count
            )));
        }
        if let Some(release) = &self.release {
            if release.revision.is_empty() {
                return Err(RulesetError::InvalidManifest(
                    "release revision must not be empty".to_owned(),
                ));
            }
            if !release.tag_matches(&self.id) {
                return Err(RulesetError::InvalidManifest(format!(
                    "release tag {:?} must be \"ruleset/{}-{}\"",
                    release.ruleset_tag, self.id, release.revision
                )));
            }
            // A recorded hash must be complete; a placeholder is not an attestation.
            if let Some(commit) = &release.engine_commit
                && !release.engine_commit_is_recorded()
            {
                return Err(RulesetError::InvalidManifest(format!(
                    "engine_commit must be 40 lowercase hex characters, got {commit:?}"
                )));
            }
        }
        if self.base_rules.repetition_count < 2 {
            return Err(RulesetError::InvalidManifest(
                "repetition_count must be at least two".to_owned(),
            ));
        }
        if self.base_rules.deathmatch_enabled
            && (self.base_rules.deathmatch_starts_at_move_count == 0
                || self.base_rules.deathmatch_quiet_move_count == 0)
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
        if !matches!(
            self.base_rules.repetition_tiebreak.as_str(),
            "lower_half_star_total" | "draw"
        ) {
            return Err(RulesetError::InvalidManifest(format!(
                "unsupported repetition_tiebreak {:?}",
                self.base_rules.repetition_tiebreak
            )));
        }
        if self.base_rules.fifty_move_completed_turns == Some(0) {
            return Err(RulesetError::InvalidManifest(
                "fifty-move threshold must be positive".into(),
            ));
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

/// How far this snapshot's engine reaches, stated as separate claims.
///
/// Two questions get answered separately because they are different evidence:
/// whether every publicly active card and rule has an executable implementation,
/// and whether that implementation was confirmed to match the live site's behavior.
/// A snapshot can complete the first without the second, and saying so is the point.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct Coverage {
    /// Coverage of the publicly active catalog, which is what the release gate asks
    /// for: `publicly_active` or `partial`.
    pub status: String,
    /// Cards the catalog lists, including ones upstream disabled.
    pub target_card_count: u32,
    /// Cards the catalog enables, which is the denominator for `status`.
    pub publicly_active_card_count: u32,
    pub implemented_card_count: u32,
    pub behavioral_slice_card_count: u32,
    /// Cards confirmed against the live site. Zero means behavior parity is not
    /// claimed, whatever `status` says.
    pub site_parity_verified_card_count: u32,
    pub notes: String,
}

impl Coverage {
    /// Every publicly active card is executable. This is the release-gate condition.
    pub fn covers_all_public_cards(&self) -> bool {
        self.implemented_card_count >= self.publicly_active_card_count
    }

    /// Behavior parity with the live site was verified for every implemented card.
    pub fn site_parity_is_complete(&self) -> bool {
        self.site_parity_verified_card_count >= self.implemented_card_count
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct BaseRules {
    pub king_capture_wins: bool,
    pub immobility_loses: bool,
    pub repetition_count: u8,
    pub repetition_tiebreak: String,
    #[serde(default = "legacy_deathmatch_enabled")]
    pub deathmatch_enabled: bool,
    #[serde(default)]
    pub fifty_move_completed_turns: Option<u32>,
    pub deathmatch_starts_at_move_count: u32,
    pub deathmatch_quiet_move_count: u32,
}

fn legacy_deathmatch_enabled() -> bool {
    true
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

    /// The current snapshot keeps the star tiebreak and the extension rule enabled.
    ///
    /// This test previously asserted the opposite, that the latest rules removed both. The source
    /// enables the extension by default and resolves repetition and the star limit by star count,
    /// with the fewer-star side winning, so the old assertions recorded a rule that was missing.
    #[test]
    fn latest_rules_keep_the_star_tiebreak_and_extension() {
        let latest = RulesetManifest::bundled_latest().unwrap();
        assert_eq!(latest.id, crate::LATEST_RULESET_ID);
        assert_eq!(
            latest.base_rules.repetition_tiebreak,
            "lower_half_star_total"
        );
        assert!(latest.base_rules.deathmatch_enabled);
        assert_eq!(latest.base_rules.deathmatch_starts_at_move_count, 45);
        assert_eq!(latest.base_rules.deathmatch_quiet_move_count, 10);
        assert_eq!(latest.base_rules.fifty_move_completed_turns, Some(100));
        assert_eq!(latest.opening_rule.chance_numerator, 1);
        assert_eq!(latest.opening_rule.chance_denominator, 5);
        // The older snapshot keeps the same policy, so history still parses.
        let old = RulesetManifest::bundled_20260915().unwrap();
        assert!(old.base_rules.deathmatch_enabled);
        assert_eq!(old.base_rules.fifty_move_completed_turns, None);
        assert_eq!(old.base_rules.repetition_tiebreak, "lower_half_star_total");
    }

    /// The release gate asks for card coverage, not behavior parity. The current
    /// snapshot meets the first and does not claim the second, and those two facts
    /// must stay separately visible.
    #[test]
    fn latest_covers_every_public_card_without_claiming_site_parity() {
        let latest = RulesetManifest::bundled_latest().unwrap();
        assert!(
            latest.coverage.covers_all_public_cards(),
            "every publicly active card must be executable"
        );
        assert_eq!(latest.coverage.status, "publicly_active");
        assert!(
            !latest.coverage.site_parity_is_complete(),
            "behavior parity must not be implied by card coverage"
        );
        // The disabled cards are catalogued but not public, so the two counts differ.
        assert!(latest.coverage.target_card_count > latest.coverage.publicly_active_card_count);
        assert_eq!(
            latest.coverage.implemented_card_count,
            latest.coverage.publicly_active_card_count
        );
    }

    /// The release block names the tag for this snapshot and, once attested, a full
    /// engine commit. `scripts/verify-release-attestation.mjs` proves the hash is real.
    #[test]
    fn the_latest_snapshot_declares_a_release_for_itself() {
        let latest = RulesetManifest::bundled_latest().unwrap();
        let release = latest
            .release
            .as_ref()
            .expect("the latest snapshot must declare a release");
        assert!(
            release.tag_matches(&latest.id),
            "tag must name this snapshot"
        );
        assert_eq!(
            release.ruleset_tag,
            format!("ruleset/{}-{}", latest.id, release.revision)
        );
        if let Some(commit) = &release.engine_commit {
            assert!(
                release.engine_commit_is_recorded(),
                "a recorded engine commit must be a full hash, got {commit:?}"
            );
        }
    }

    /// Earlier snapshots predate the release block and must keep loading.
    #[test]
    fn snapshots_without_a_release_block_still_parse() {
        for manifest in [
            RulesetManifest::bundled_20260807().unwrap(),
            RulesetManifest::bundled_20260818().unwrap(),
            RulesetManifest::bundled_20260915().unwrap(),
            RulesetManifest::bundled_20261017().unwrap(),
        ] {
            assert!(
                manifest.release.is_none(),
                "{} predates releases",
                manifest.id
            );
        }
    }

    #[test]
    fn bundled_manifest_is_valid_and_explicitly_incomplete() {
        let manifest = RulesetManifest::bundled_20260807().unwrap();
        assert_eq!(manifest.id, BUNDLED_RULESET_ID);
        assert_eq!(manifest.snapshot_date, "2026-08-18");
        // Every publicly active card is executable, which is the card-coverage claim.
        assert_eq!(manifest.coverage.status, "publicly_active");
        assert!(manifest.coverage.covers_all_public_cards());
        assert_eq!(manifest.coverage.target_card_count, 208);
        assert_eq!(manifest.coverage.publicly_active_card_count, 208);
        assert_eq!(manifest.coverage.implemented_card_count, 208);
        assert_eq!(manifest.coverage.behavioral_slice_card_count, 208);
        // ...but behavior parity with the live site is still not claimed.
        assert!(!manifest.coverage.site_parity_is_complete());
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
    fn september_manifest_reports_partial_coverage_without_replacing_legacy() {
        let old = RulesetManifest::bundled_20260818().unwrap();
        assert_eq!(old, RulesetManifest::bundled_20260807().unwrap());
        assert_eq!(old.id, "augment_chess_20260818");
        let latest = RulesetManifest::bundled_20260915().unwrap();
        assert_eq!(latest.id, "augment_chess_20260915");
        assert_eq!(latest.snapshot_date, "2026-09-15");
        assert_eq!(latest.coverage.target_card_count, 240);
        assert_eq!(latest.coverage.publicly_active_card_count, 240);
        assert!(latest.coverage.covers_all_public_cards());
        // Card coverage is complete; site behavior parity remains unverified.
        assert_eq!(latest.coverage.status, "publicly_active");
        assert!(!latest.coverage.site_parity_is_complete());
        assert_eq!(latest.coverage.site_parity_verified_card_count, 0);
        let implemented: serde_json::Value = serde_json::from_str(include_str!(
            "../../../rulesets/augment_chess_20260915/cards/implemented.json"
        ))
        .unwrap();
        assert_eq!(
            latest.coverage.implemented_card_count as usize,
            implemented["cards"].as_array().unwrap().len()
        );
        assert_eq!(
            implemented["scope"]["implemented_cards"].as_u64(),
            Some(latest.coverage.implemented_card_count.into())
        );
        assert!(implemented["cards"].as_array().unwrap().iter().all(|card| {
            card["activation"].is_string()
                && card["compatibility"].is_array()
                && card["engine_primitives"].is_array()
        }));
        assert_eq!(
            latest.coverage.behavioral_slice_card_count,
            latest.coverage.implemented_card_count
        );
    }

    #[test]
    fn all_public_modes_are_present() {
        let manifest = RulesetManifest::bundled_20260807().unwrap();
        for mode in ["standard", "chaos", "grand"] {
            assert!(manifest.mode(mode).is_some());
        }
    }
}

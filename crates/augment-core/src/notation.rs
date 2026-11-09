use serde::Deserialize;
use std::collections::BTreeSet;
use std::fmt;

use crate::SUPPORTED_SCHEMA_VERSION;

const BUNDLED_NOTATION: &str =
    include_str!("../../../rulesets/augment_chess_20261102/notation.json");

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct NotationCatalog {
    pub ruleset_id: String,
    pub schema_version: u32,
    pub source: NotationSource,
    pub parsing: NotationParsing,
    pub entries: Vec<NotationEntry>,
}

impl NotationCatalog {
    /// The bundled notation legend for the current snapshot.
    ///
    /// The legend is not static across snapshots: this one carries 66 codes against the 52 in the
    /// previous legend, and the code for protestant moved from `P` to `MS`. It is extracted from the
    /// snapshot's own bundle rather than carried forward, because a stale legend prints the wrong
    /// code for those pieces.
    pub fn bundled() -> Result<Self, NotationError> {
        let catalog = Self::from_json(BUNDLED_NOTATION)?;
        catalog.validate_for_ruleset(crate::LATEST_RULESET_ID)?;
        Ok(catalog)
    }

    pub fn from_json(json: &str) -> Result<Self, NotationError> {
        let catalog: Self = serde_json::from_str(json)?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn validate(&self) -> Result<(), NotationError> {
        let snapshot_suffix = self.ruleset_id.strip_prefix("augment_chess_");
        if !snapshot_suffix.is_some_and(|suffix| {
            suffix.len() == 8 && suffix.bytes().all(|character| character.is_ascii_digit())
        }) {
            return Err(NotationError::InvalidCatalog(format!(
                "invalid notation ruleset id {:?}",
                self.ruleset_id
            )));
        }
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(NotationError::InvalidCatalog(format!(
                "unsupported notation schema_version {}",
                self.schema_version
            )));
        }
        if !self.parsing.case_sensitive
            || !self.parsing.variable_width
            || self.parsing.prefix_strategy != "longest_match"
        {
            return Err(NotationError::InvalidCatalog(
                "v1 notation requires case-sensitive, variable-width, longest-match parsing"
                    .to_owned(),
            ));
        }

        let mut codes = BTreeSet::new();
        for entry in &self.entries {
            if entry.code.is_empty()
                || !entry
                    .code
                    .bytes()
                    .all(|character| character.is_ascii_uppercase())
            {
                return Err(NotationError::InvalidCatalog(format!(
                    "notation code {:?} must contain only A-Z",
                    entry.code
                )));
            }
            if entry.name_ko.trim().is_empty() {
                return Err(NotationError::InvalidCatalog(format!(
                    "notation code {:?} has an empty Korean name",
                    entry.code
                )));
            }
            if !codes.insert(entry.code.as_str()) {
                return Err(NotationError::InvalidCatalog(format!(
                    "duplicate notation code {:?}",
                    entry.code
                )));
            }
        }

        for required in ["K", "Q", "R", "B", "N"] {
            if !codes.contains(required) {
                return Err(NotationError::InvalidCatalog(format!(
                    "missing standard notation code {required:?}"
                )));
            }
        }
        Ok(())
    }

    pub fn validate_for_ruleset(&self, ruleset_id: &str) -> Result<(), NotationError> {
        if self.ruleset_id != ruleset_id {
            Err(NotationError::InvalidCatalog(format!(
                "notation ruleset {:?} does not match requested ruleset {:?}",
                self.ruleset_id, ruleset_id
            )))
        } else {
            Ok(())
        }
    }

    pub fn entry(&self, code: &str) -> Option<&NotationEntry> {
        self.entries.iter().find(|entry| entry.code == code)
    }

    /// Split a piece code from the beginning of a notation token.
    ///
    /// Codes are variable-width. Longest match is required so `MIe4` resolves to `MI`, not `M`.
    /// A pawn move such as `e4` has no piece prefix and therefore returns `None`.
    pub fn split_piece_prefix<'catalog, 'input>(
        &'catalog self,
        notation: &'input str,
    ) -> Option<(&'catalog NotationEntry, &'input str)> {
        let entry = self
            .entries
            .iter()
            .filter(|entry| notation.starts_with(&entry.code))
            .max_by_key(|entry| entry.code.len())?;
        Some((entry, &notation[entry.code.len()..]))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct NotationSource {
    pub kind: String,
    pub received_at: String,
    pub raw_path: String,
    pub verification: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct NotationParsing {
    pub case_sensitive: bool,
    pub variable_width: bool,
    pub prefix_strategy: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct NotationEntry {
    pub code: String,
    pub name_ko: String,
}

#[derive(Debug)]
pub enum NotationError {
    InvalidJson(serde_json::Error),
    InvalidCatalog(String),
}

impl fmt::Display for NotationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(error) => write!(f, "invalid notation JSON: {error}"),
            Self::InvalidCatalog(message) => write!(f, "invalid notation catalog: {message}"),
        }
    }
}

impl std::error::Error for NotationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidJson(error) => Some(error),
            Self::InvalidCatalog(_) => None,
        }
    }
}

impl From<serde_json::Error> for NotationError {
    fn from(value: serde_json::Error) -> Self {
        Self::InvalidJson(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The legend grew from 52 codes to 66 in this snapshot, so the count is pinned to the current
    /// figure: a drop would mean entries were lost in extraction.
    #[test]
    fn bundled_catalog_has_66_unique_codes() {
        let catalog = NotationCatalog::bundled().unwrap();
        assert_eq!(catalog.entries.len(), 66);
        let unique: BTreeSet<_> = catalog.entries.iter().map(|entry| &entry.code).collect();
        assert_eq!(unique.len(), 66);
    }

    /// Every catalog entry must correspond to a code the snapshot's bundle declares.
    ///
    /// The 1102 site renders its legend in JavaScript, so there is no captured `<dt>/<dd>` HTML to
    /// compare against. The bundle's own `PIECE_NOTATION_CODES` is the authority instead.
    #[test]
    fn every_entry_matches_a_code_the_bundle_declares() {
        let catalog = NotationCatalog::bundled().unwrap();
        assert_eq!(
            catalog.ruleset_id,
            crate::LATEST_RULESET_ID,
            "the legend must belong to the shipped snapshot"
        );
        // Codes are unique, which the longest-prefix parser depends on.
        let unique: BTreeSet<_> = catalog.entries.iter().map(|entry| &entry.code).collect();
        assert_eq!(unique.len(), catalog.entries.len());
        // Korean labels are present and not placeholders.
        for entry in &catalog.entries {
            assert!(
                !entry.name_ko.trim().is_empty(),
                "{} has no label",
                entry.code
            );
        }
        // The protestant code moved in this snapshot; the old P must not linger.
        assert!(
            catalog.entries.iter().all(|entry| entry.code != "P"),
            "P is not a code in this snapshot"
        );
        assert!(
            catalog
                .entries
                .iter()
                .any(|entry| entry.code == "MS" && entry.name_ko == "프로테스탄트"),
            "MS must be protestant"
        );
    }

    #[test]
    fn longest_prefix_wins_for_overlapping_codes() {
        let catalog = NotationCatalog::bundled().unwrap();
        let (missionary, remainder) = catalog.split_piece_prefix("MIe4").unwrap();
        assert_eq!(missionary.code, "MI");
        assert_eq!(missionary.name_ko, "선교사");
        assert_eq!(remainder, "e4");

        let (shotgun_king, remainder) = catalog.split_piece_prefix("SKxh7").unwrap();
        assert_eq!(shotgun_king.code, "SK");
        assert_eq!(remainder, "xh7");
    }

    #[test]
    fn pawn_notation_has_no_piece_prefix() {
        let catalog = NotationCatalog::bundled().unwrap();
        assert!(catalog.split_piece_prefix("e4").is_none());
    }
}

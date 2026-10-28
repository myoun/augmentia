//! Shared card definitions and ruleset-aware lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CardActivation {
    Passive,
    Active,
    Rule,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CardPhase {
    Opening,
    Middle,
    End,
    Piece,
    Rule,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImplementedCardDefinition {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub effect: &'static str,
    pub activation: CardActivation,
    pub phase: CardPhase,
    pub half_stars: u16,
    pub ends_turn: bool,
    pub turn_exclusive: bool,
}

/// Select metadata from the requested ruleset. Current definitions own every field.
pub fn ruleset_card(ruleset_id: &str, id: &str) -> Option<ImplementedCardDefinition> {
    if ruleset_id == crate::LATEST_RULESET_ID {
        crate::latest::card_definition(id)
    } else {
        crate::compat::card_definitions::implemented_card(id).copied()
    }
}

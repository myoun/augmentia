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

/// Card metadata for a ruleset. Only the current snapshot ships, so the id is not consulted.
pub fn ruleset_card(_ruleset_id: &str, id: &str) -> Option<ImplementedCardDefinition> {
    crate::latest::card_definition(id)
}

pub const KING_OF_THE_HILL: &str = "king-of-the-hill";
pub const GENEVA_CONVENTION: &str = "geneva-convention";
pub const PARRY: &str = "parry";
pub const EMPTY_LUNCHBOX: &str = "empty-lunchbox";
pub const PLATFORM: &str = "platform";

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
}

pub const IMPLEMENTED_CARDS: [ImplementedCardDefinition; 5] = [
    ImplementedCardDefinition {
        id: KING_OF_THE_HILL,
        name_ko: "언덕의 왕",
        effect: "kingOfTheHill",
        activation: CardActivation::Passive,
        phase: CardPhase::Opening,
        half_stars: 5,
    },
    ImplementedCardDefinition {
        id: GENEVA_CONVENTION,
        name_ko: "제네바 협약",
        effect: "genevaConvention",
        activation: CardActivation::Passive,
        phase: CardPhase::Opening,
        half_stars: 6,
    },
    ImplementedCardDefinition {
        id: PARRY,
        name_ko: "패링",
        effect: "parry",
        activation: CardActivation::Active,
        phase: CardPhase::Middle,
        half_stars: 8,
    },
    ImplementedCardDefinition {
        id: EMPTY_LUNCHBOX,
        name_ko: "빈 찬합",
        effect: "emptyLunchbox",
        activation: CardActivation::Active,
        phase: CardPhase::End,
        half_stars: 7,
    },
    ImplementedCardDefinition {
        id: PLATFORM,
        name_ko: "발판",
        effect: "platformRule",
        activation: CardActivation::Rule,
        phase: CardPhase::Rule,
        half_stars: 0,
    },
];

pub fn implemented_card(id: &str) -> Option<&'static ImplementedCardDefinition> {
    IMPLEMENTED_CARDS.iter().find(|card| card.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUNDLED_IMPLEMENTED_CARDS_JSON: &str =
        include_str!("../../../rulesets/augment_chess_20260807/cards/implemented.json");

    #[test]
    fn implemented_ids_and_effects_are_unique() {
        let ids: std::collections::BTreeSet<_> =
            IMPLEMENTED_CARDS.iter().map(|card| card.id).collect();
        let effects: std::collections::BTreeSet<_> =
            IMPLEMENTED_CARDS.iter().map(|card| card.effect).collect();
        assert_eq!(ids.len(), IMPLEMENTED_CARDS.len());
        assert_eq!(effects.len(), IMPLEMENTED_CARDS.len());
    }

    #[test]
    fn compiled_cards_match_the_snapshot_catalog() {
        let document: serde_json::Value =
            serde_json::from_str(BUNDLED_IMPLEMENTED_CARDS_JSON).unwrap();
        let cards = document["cards"].as_array().unwrap();

        assert_eq!(cards.len(), IMPLEMENTED_CARDS.len());
        for (snapshot, compiled) in cards.iter().zip(IMPLEMENTED_CARDS) {
            assert_eq!(snapshot["id"].as_str(), Some(compiled.id));
            assert_eq!(snapshot["name_ko"].as_str(), Some(compiled.name_ko));
            assert_eq!(snapshot["effect"].as_str(), Some(compiled.effect));
            assert_eq!(
                snapshot["half_stars"].as_u64(),
                Some(compiled.half_stars.into())
            );
            assert_eq!(
                snapshot["phase"].as_str(),
                Some(match compiled.phase {
                    CardPhase::Opening => "OPENING",
                    CardPhase::Middle => "MIDDLE",
                    CardPhase::End => "END",
                    CardPhase::Piece => "PIECE",
                    CardPhase::Rule => "RULE",
                })
            );
        }
    }
}

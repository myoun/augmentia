pub const KING_OF_THE_HILL: &str = "king-of-the-hill";
pub const GENEVA_CONVENTION: &str = "geneva-convention";
pub const PARRY: &str = "parry";
pub const EMPTY_LUNCHBOX: &str = "empty-lunchbox";
pub const PLATFORM: &str = "platform";
pub const SPRINT: &str = "sprint";
pub const EARLY_PROMOTION: &str = "early-promotion";
pub const INITIATIVE: &str = "initiative";
pub const REVELATION: &str = "revelation";
pub const DISARM: &str = "disarm";
pub const DICE: &str = "dice";
pub const FREEZE: &str = "freeze";
pub const REPOSITION: &str = "reposition";
pub const ASSASSIN: &str = "assassin";
pub const GRASSHOPPER: &str = "grasshopper";
pub const JUDGMENT: &str = "judgment";
pub const CLEANUP_SACRIFICE: &str = "cleanup-sacrifice";

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

pub const IMPLEMENTED_CARDS: [ImplementedCardDefinition; 17] = [
    ImplementedCardDefinition {
        id: KING_OF_THE_HILL,
        name_ko: "언덕의 왕",
        effect: "kingOfTheHill",
        activation: CardActivation::Passive,
        phase: CardPhase::Opening,
        half_stars: 5,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: GENEVA_CONVENTION,
        name_ko: "제네바 협약",
        effect: "genevaConvention",
        activation: CardActivation::Passive,
        phase: CardPhase::Opening,
        half_stars: 6,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: PARRY,
        name_ko: "패링",
        effect: "parry",
        activation: CardActivation::Active,
        phase: CardPhase::Middle,
        half_stars: 8,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: EMPTY_LUNCHBOX,
        name_ko: "빈 찬합",
        effect: "emptyLunchbox",
        activation: CardActivation::Active,
        phase: CardPhase::End,
        half_stars: 7,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: PLATFORM,
        name_ko: "발판",
        effect: "platformRule",
        activation: CardActivation::Rule,
        phase: CardPhase::Rule,
        half_stars: 0,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: SPRINT,
        name_ko: "질주",
        effect: "pawnSprint",
        activation: CardActivation::Passive,
        phase: CardPhase::Opening,
        half_stars: 6,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: EARLY_PROMOTION,
        name_ko: "조기 진급",
        effect: "earlyPromotion",
        activation: CardActivation::Passive,
        phase: CardPhase::Opening,
        half_stars: 7,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: INITIATIVE,
        name_ko: "선공권",
        effect: "initiative",
        activation: CardActivation::Passive,
        phase: CardPhase::Opening,
        half_stars: 4,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: REVELATION,
        name_ko: "계시",
        effect: "revelation",
        activation: CardActivation::Rule,
        phase: CardPhase::Rule,
        half_stars: 3,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: DISARM,
        name_ko: "무장해제",
        effect: "disarm",
        activation: CardActivation::Active,
        phase: CardPhase::Middle,
        half_stars: 8,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: DICE,
        name_ko: "주사위",
        effect: "dice",
        activation: CardActivation::Active,
        phase: CardPhase::Middle,
        half_stars: 6,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: FREEZE,
        name_ko: "빙결",
        effect: "freeze",
        activation: CardActivation::Active,
        phase: CardPhase::Middle,
        half_stars: 6,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: REPOSITION,
        name_ko: "재배치",
        effect: "reposition",
        activation: CardActivation::Active,
        phase: CardPhase::Middle,
        half_stars: 5,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: ASSASSIN,
        name_ko: "암살자",
        effect: "assassin",
        activation: CardActivation::Active,
        phase: CardPhase::Piece,
        half_stars: 7,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: GRASSHOPPER,
        name_ko: "그래스호퍼",
        effect: "grasshopper",
        activation: CardActivation::Active,
        phase: CardPhase::Piece,
        half_stars: 6,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: JUDGMENT,
        name_ko: "레드카드",
        effect: "judgment",
        activation: CardActivation::Active,
        phase: CardPhase::End,
        half_stars: 4,
        ends_turn: false,
        turn_exclusive: false,
    },
    ImplementedCardDefinition {
        id: CLEANUP_SACRIFICE,
        name_ko: "강제 교환",
        effect: "cleanupSacrifice",
        activation: CardActivation::Active,
        phase: CardPhase::End,
        half_stars: 3,
        ends_turn: false,
        turn_exclusive: false,
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
    const BUNDLED_CARD_CATALOG_JSON: &str =
        include_str!("../../../rulesets/augment_chess_20260807/cards/catalog.json");

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
            assert_eq!(snapshot["ends_turn"].as_bool(), Some(compiled.ends_turn));
            assert_eq!(
                snapshot["turn_exclusive"].as_bool(),
                Some(compiled.turn_exclusive)
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

    #[test]
    fn full_snapshot_catalog_contains_all_visible_cards_and_compiled_subset() {
        let document: serde_json::Value = serde_json::from_str(BUNDLED_CARD_CATALOG_JSON).unwrap();
        let implemented_document: serde_json::Value =
            serde_json::from_str(BUNDLED_IMPLEMENTED_CARDS_JSON).unwrap();
        let cards = document["cards"].as_array().unwrap();
        assert_eq!(document["raw_card_definitions"].as_u64(), Some(193));
        assert_eq!(document["visible_card_count"].as_u64(), Some(192));
        assert_eq!(
            document["source_bundle_sha256"],
            implemented_document["source_bundle_sha256"]
        );
        assert_eq!(cards.len(), 192);
        let phase_counts = cards.iter().fold(
            std::collections::BTreeMap::<&str, usize>::new(),
            |mut counts, card| {
                *counts.entry(card["phase"].as_str().unwrap()).or_default() += 1;
                counts
            },
        );
        assert_eq!(phase_counts["OPENING"], 30);
        assert_eq!(phase_counts["MIDDLE"], 71);
        assert_eq!(phase_counts["PIECE"], 22);
        assert_eq!(phase_counts["END"], 44);
        assert_eq!(phase_counts["RULE"], 25);

        for compiled in IMPLEMENTED_CARDS {
            let catalog = cards
                .iter()
                .find(|card| card["id"].as_str() == Some(compiled.id))
                .unwrap_or_else(|| panic!("{} is missing from the full catalog", compiled.id));
            assert_eq!(catalog["name_ko"].as_str(), Some(compiled.name_ko));
            assert_eq!(catalog["effect"].as_str(), Some(compiled.effect));
            if catalog["half_stars"].is_null() {
                assert_eq!(compiled.half_stars, 0);
            } else {
                assert_eq!(
                    catalog["half_stars"].as_u64(),
                    Some(compiled.half_stars.into())
                );
            }
        }
    }
}

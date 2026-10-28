//! Canonical October metadata and a fail-closed inherited-handler profile.
use std::collections::BTreeSet;
use std::sync::OnceLock;

use crate::{
    CardActivation, CardPhase, ImplementedCardDefinition, LATEST_RULESET_ID, implemented_card,
};
use serde::Deserialize;

pub const MAN: &str = "man";
pub const IQP: &str = "iqp";
pub const HOPE: &str = "hope";
pub const ASSEMBLY: &str = "assembly";
pub const ARCHBISHOP: &str = "archbishop";
pub const CHANCELLOR: &str = "chancellor";
pub const SEAL: &str = "seal";
pub const GREEK_GIFT: &str = "greek-gift";
pub const ASCENSION: &str = "ascension";
pub const TABOO: &str = "taboo";
pub const GRUDGE: &str = "grudge";
pub const PAWN_CHAIN: &str = "pawn-chain";
pub const PENTA_CHECK: &str = "penta-check";

/// Native minor family from the captured v19 catalog, expressed in core names.
pub(crate) fn latest_minor_kind(kind: crate::PieceKind) -> bool {
    use crate::PieceKind::*;
    matches!(
        kind,
        Knight
            | Bishop
            | Camel
            | Clockwork
            | Parrot
            | Wizard
            | Recruiter
            | Trickster
            | Scarecrow
            | Monster
            | Alfil
            | Checker
            | Guard
            | Lobster
            | Missionary
            | Alibaba
            | CheckerKing
            | Log
            | Knightmaster
            | Protestant
            | Grasshopper
            | BabyBear
            | Campfire
            | Paladin
            | Cannon
            | SiegeRam
            | Dragon
            | Slime
            | Unicorn
            | Cardinal
            | Thief
            | Idol
            | Brutus
    )
}

/// Captured v19 major family (existing core piece names only).
pub(crate) fn latest_major_kind(kind: crate::PieceKind) -> bool {
    use crate::PieceKind::*;
    matches!(
        kind,
        Queen
            | Rook
            | Amazon
            | Man
            | Colossus
            | BigRook
            | Herald
            | Jester
            | Hook
            | PrimeMinister
            | Assassin
            | Windmill
            | Bear
            | Octopus
            | Hedgehog
            | Princess
            | BigBishop
            | MagicGirl
            | Berserker
            | Siren
            | Reaper
            | Undead
            | Archbishop
            | Chancellor
    )
}

#[derive(Debug, Deserialize)]
pub struct LatestCard {
    pub id: String,
    pub name_ko: String,
    pub text_ko: String,
    pub effect: String,
    pub phase: String,
    pub half_stars: u16,
    pub enabled: bool,
    pub rarity: Option<String>,
    pub activation: String,
    pub turn_policy: String,
    pub draft_weight_units: u64,
}

pub fn latest_catalog() -> &'static [LatestCard] {
    #[derive(Deserialize)]
    struct Catalog {
        cards: Vec<LatestCard>,
    }
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    &CATALOG
        .get_or_init(|| {
            serde_json::from_str(include_str!(
                "../../../rulesets/augment_chess_20261017/cards/catalog.json"
            ))
            .expect("bundled canonical catalog")
        })
        .cards
}

impl LatestCard {
    pub fn supported(&self) -> bool {
        #[derive(Deserialize)]
        struct Profile {
            cards: Vec<Id>,
        }
        #[derive(Deserialize)]
        struct Id {
            id: String,
        }
        static IDS: OnceLock<BTreeSet<String>> = OnceLock::new();
        self.enabled
            && IDS
                .get_or_init(|| {
                    let profile: Profile = serde_json::from_str(include_str!(
                        "../../../rulesets/augment_chess_20261017/cards/implemented.json"
                    ))
                    .expect("bundled inherited-handler profile");
                    profile.cards.into_iter().map(|card| card.id).collect()
                })
                .contains(&self.id)
    }
}

/// Old snapshots retain their historical definitions. October never falls back
/// to an old handler for a new, disabled, or changed-contract card.
pub fn ruleset_card(ruleset_id: &str, id: &str) -> Option<ImplementedCardDefinition> {
    if ruleset_id != LATEST_RULESET_ID {
        return implemented_card(id).copied();
    }
    let card = latest_catalog()
        .iter()
        .find(|card| card.id == id && card.supported())?;
    Some(ImplementedCardDefinition {
        id: &card.id,
        name_ko: &card.name_ko,
        effect: &card.effect,
        half_stars: card.half_stars,
        phase: match card.phase.as_str() {
            "OPENING" => CardPhase::Opening,
            "MIDDLE" => CardPhase::Middle,
            "END" => CardPhase::End,
            "PIECE" => CardPhase::Piece,
            "RULE" => CardPhase::Rule,
            _ => return None,
        },
        activation: if card.phase == "RULE" {
            CardActivation::Rule
        } else {
            match card.activation.as_str() {
                "PASSIVE" => CardActivation::Passive,
                "ACTIVE" => CardActivation::Active,
                _ => return None,
            }
        },
        ends_turn: card.turn_policy == "END_TURN",
        turn_exclusive: implemented_card(id).is_some_and(|old| old.turn_exclusive),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Action, Board, DrawReason, Engine, GameState, GameStatus, Mode, MoveAction, Piece,
        PieceKind, Player,
    };

    fn state() -> GameState {
        let mut state = GameState::standard(LATEST_RULESET_ID, Mode::Standard, 1);
        state.completed_drafts = [true; 3];
        state
    }
    fn movement(from: &str, to: &str) -> Action {
        Action::Move(MoveAction::new(
            from.parse().unwrap(),
            to.parse().unwrap(),
            None,
        ))
    }

    #[test]
    fn latest_repetition_draws_regardless_of_stars_and_has_priority_over_fifty_moves() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state();
        state.half_star_totals = [2, 10];
        state.finalize_initial_setup();
        for round in 0..2 {
            for (i, (from, to)) in [("g1", "f3"), ("g8", "f6"), ("f3", "g1"), ("f6", "g8")]
                .into_iter()
                .enumerate()
            {
                if round == 1 && i == 3 {
                    state.automatic_draw.as_mut().unwrap().quiet_turns = 99;
                }
                state = engine.apply(&state, movement(from, to)).unwrap().state;
            }
        }
        assert_eq!(
            state.status,
            GameStatus::Draw {
                reason: DrawReason::ThreefoldRepetition
            }
        );
    }

    #[test]
    fn fifty_moves_count_completed_turns_and_preserve_progress_across_extra_actions() {
        let engine = Engine::bundled_latest().unwrap();
        let mut quiet = state();
        quiet.actions_remaining = 2;
        quiet.automatic_draw.as_mut().unwrap().quiet_turns = 99;
        quiet = engine.apply(&quiet, movement("g1", "f3")).unwrap().state;
        assert_eq!(quiet.status, GameStatus::Ongoing);
        assert_eq!(quiet.automatic_draw.as_ref().unwrap().quiet_turns, 99);
        quiet = engine.apply(&quiet, movement("b1", "c3")).unwrap().state;
        assert_eq!(
            quiet.status,
            GameStatus::Draw {
                reason: DrawReason::FiftyMoveRule
            }
        );

        let mut progress = state();
        progress.actions_remaining = 2;
        progress.automatic_draw.as_mut().unwrap().quiet_turns = 99;
        progress = engine.apply(&progress, movement("e2", "e4")).unwrap().state;
        assert_eq!(progress.automatic_draw.as_ref().unwrap().quiet_turns, 0);
        assert!(progress.automatic_draw.as_ref().unwrap().progress_this_turn);
        progress = engine.apply(&progress, movement("g1", "f3")).unwrap().state;
        assert_eq!(progress.automatic_draw.as_ref().unwrap().quiet_turns, 0);
        assert!(!progress.automatic_draw.as_ref().unwrap().progress_this_turn);
        progress = engine.apply(&progress, movement("g8", "f6")).unwrap().state;
        assert_eq!(progress.automatic_draw.as_ref().unwrap().quiet_turns, 1);
    }

    #[test]
    fn promotion_and_enemy_capture_reset_the_quiet_counter() {
        let engine = Engine::bundled_latest().unwrap();
        let mut board = Board::empty();
        for (square, owner, kind) in [
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b7", Player::White, PieceKind::Pawn),
            ("g1", Player::White, PieceKind::Rook),
            ("g7", Player::Black, PieceKind::Knight),
        ] {
            board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let mut state =
            GameState::from_board(LATEST_RULESET_ID, Mode::Standard, board, Player::White, 1);
        state.completed_drafts = [true; 3];
        state.automatic_draw.as_mut().unwrap().quiet_turns = 99;
        let promoted = engine
            .apply(
                &state,
                Action::Move(MoveAction::new(
                    "b7".parse().unwrap(),
                    "b8".parse().unwrap(),
                    Some(PieceKind::Queen),
                )),
            )
            .unwrap()
            .state;
        assert_eq!(promoted.automatic_draw.as_ref().unwrap().quiet_turns, 0);
        let captured = engine.apply(&state, movement("g1", "g7")).unwrap().state;
        assert_eq!(captured.automatic_draw.as_ref().unwrap().quiet_turns, 0);
    }

    #[test]
    fn latest_never_starts_deathmatch_and_rejects_injected_unsupported_cards() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state();
        state.turns_taken = [45; 2];
        state = engine.apply(&state, movement("g1", "f3")).unwrap().state;
        assert!(state.deathmatch.is_none());
        assert!(
            engine
                .grant_card(&state, 1, "merchant-guild", Player::White)
                .is_err()
        );
        state.applied_rules.push("acceleration".into());
        assert!(engine.legal_actions(&state).is_err());
    }

    #[test]
    fn quiet_draw_memory_affects_replay_but_not_repetition_identity() {
        let state = state();
        let mut changed = state.clone();
        changed.automatic_draw.as_mut().unwrap().quiet_turns = 99;
        assert_eq!(state.position_key(), changed.position_key());
        assert_ne!(state.fingerprint(), changed.fingerprint());
    }

    #[test]
    fn final_draft_excludes_common_cards_and_only_offers_supported_contracts() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state();
        state.completed_drafts = [true, true, false];
        state.turn = Player::Black;
        state.turns_taken = [20, 19];
        let next = engine.apply(&state, movement("g8", "f6")).unwrap().state;
        let crate::GamePhase::Draft(draft) = &next.phase else {
            panic!("end draft missing");
        };
        assert_eq!(draft.phase, crate::DraftPhase::End);
        assert_eq!(draft.offers.len(), 3);
        for id in &draft.offers {
            let card = latest_catalog().iter().find(|card| card.id == *id).unwrap();
            assert!(card.supported());
            assert_ne!(card.rarity.as_deref(), Some("common"));
        }
    }

    #[test]
    fn latest_seeded_drafts_and_play_are_deterministic_and_supported() {
        let engine = Engine::bundled_latest().unwrap();
        for seed in 0..12 {
            let mut state = engine.new_game(Mode::Standard, seed).unwrap();
            for step in 0..64 {
                if state.status != GameStatus::Ongoing {
                    break;
                }
                let actions = engine.legal_actions(&state).unwrap();
                assert!(!actions.is_empty());
                let action = actions[step % actions.len()].clone();
                let next = engine.apply(&state, action.clone()).unwrap();
                assert_eq!(next, engine.apply(&state, action).unwrap());
                state = next.state;
                assert!(
                    state
                        .cards
                        .iter()
                        .all(|card| engine.card_definition(&card.definition_id).is_some())
                );
            }
        }
    }
    #[test]
    fn canonical_profile_is_explicit_and_fail_closed() {
        let catalog = latest_catalog();
        assert_eq!(catalog.len(), 275);
        assert_eq!(catalog.iter().filter(|c| c.enabled).count(), 271);
        assert_eq!(catalog.iter().filter(|c| c.supported()).count(), 128);
        for card in catalog {
            assert_eq!(
                ruleset_card(LATEST_RULESET_ID, &card.id).is_some(),
                card.supported()
            );
        }
        assert!(ruleset_card(LATEST_RULESET_ID, "qxe1").is_none());
        assert!(ruleset_card("augment_chess_20260915", "qxe1").is_some());
        assert!(ruleset_card(LATEST_RULESET_ID, "merchant-guild").is_none());
    }
}

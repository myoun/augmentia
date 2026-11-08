//! Current card metadata, composed only from its own catalog and execution profile.
use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::{CardActivation, CardPhase, ImplementedCardDefinition};
use serde::Deserialize;

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
            | DonQuixote
            | Medium
            | Beacon
            | TransmissionTower
            | Barrier
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
            | Grappler
            | RevolvingDoor
            | Killer
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
                "../../../rulesets/augment_chess_20261102/cards/catalog.json"
            ))
            .expect("bundled canonical catalog")
        })
        .cards
}

/// Identifies the catalog and card profile this engine executes.
///
/// A search record that does not name its ruleset cannot be replayed or compared, so the identity
/// travels with every decision. The catalog hash is the source's own fingerprint; the profile hash
/// covers the supported-card set, which is what decides which cards can appear.
pub fn latest_identity() -> (&'static str, &'static str, u64) {
    #[derive(Deserialize)]
    struct Catalog {
        source_catalog_hash: String,
    }
    static IDENTITY: OnceLock<(String, u64)> = OnceLock::new();
    let (catalog_hash, profile_hash) = IDENTITY.get_or_init(|| {
        let catalog: Catalog = serde_json::from_str(include_str!(
            "../../../rulesets/augment_chess_20261102/cards/catalog.json"
        ))
        .expect("bundled canonical catalog");
        // Hash the supported ids in sorted order so the value changes only when the set does.
        // FNV-1a, matching the state fingerprint's hasher so the value is stable across runs.
        let mut ids: Vec<&str> = execution_profile().keys().map(String::as_str).collect();
        ids.sort_unstable();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for id in ids {
            for byte in id.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        }
        (catalog.source_catalog_hash, hash)
    });
    (
        crate::LATEST_RULESET_ID,
        catalog_hash.as_str(),
        *profile_hash,
    )
}

#[derive(Deserialize)]
struct ExecutionPolicy {
    id: String,
    // Required: a missing policy must never silently inherit another version's value.
    turn_exclusive: bool,
}

#[derive(Deserialize)]
struct ExecutionProfile {
    cards: Vec<ExecutionPolicy>,
}

fn execution_profile() -> &'static BTreeMap<String, bool> {
    static PROFILE: OnceLock<BTreeMap<String, bool>> = OnceLock::new();
    PROFILE.get_or_init(|| {
        let profile: ExecutionProfile = serde_json::from_str(include_str!(
            "../../../rulesets/augment_chess_20261102/cards/implemented.json"
        ))
        .expect("bundled executable card profile with explicit turn policies");
        let mut policies = BTreeMap::new();
        for card in profile.cards {
            assert!(
                policies.insert(card.id, card.turn_exclusive).is_none(),
                "duplicate card policy"
            );
        }
        policies
    })
}

/// Cards the canonical catalog still declares unported. They must never execute.
#[cfg(test)]
fn pending_ids() -> Vec<String> {
    #[derive(Deserialize)]
    struct Pending {
        id: String,
    }
    #[derive(Deserialize)]
    struct Profile {
        pending: Vec<Pending>,
    }
    serde_json::from_str::<Profile>(include_str!(
        "../../../rulesets/augment_chess_20261102/cards/implemented.json"
    ))
    .expect("bundled executable card profile")
    .pending
    .into_iter()
    .map(|entry| entry.id)
    .collect()
}

impl LatestCard {
    pub fn supported(&self) -> bool {
        self.enabled && execution_profile().contains_key(&self.id)
    }
}

/// Construct a supported current card from its catalog and execution profile.
pub(crate) fn card_definition(id: &str) -> Option<ImplementedCardDefinition> {
    let turn_exclusive = *execution_profile().get(id)?;
    let card = latest_catalog()
        .iter()
        .find(|card| card.id == id && card.supported())?;
    let phase = match card.phase.as_str() {
        "OPENING" => CardPhase::Opening,
        "MIDDLE" => CardPhase::Middle,
        "END" => CardPhase::End,
        "PIECE" => CardPhase::Piece,
        "RULE" => CardPhase::Rule,
        _ => return None,
    };
    let activation = if card.phase == "RULE" {
        CardActivation::Rule
    } else {
        match card.activation.as_str() {
            "PASSIVE" => CardActivation::Passive,
            "ACTIVE" => CardActivation::Active,
            _ => return None,
        }
    };
    Some(ImplementedCardDefinition {
        id: &card.id,
        name_ko: &card.name_ko,
        effect: &card.effect,
        half_stars: card.half_stars,
        phase,
        activation,
        // Only a card that spends a charge can end a turn, and a passive never has one.
        //
        // The catalog cannot be trusted for this field alone: its canonical source
        // (`factoryDefinitions`) records `turn_policy: PRESERVE_TURN` for eighteen passive cards,
        // while the bundle's own v2 builder rejects that pair as `Inconsistent v2 card activation`.
        // The source disagrees with itself, so the executable rule is derived from the activation
        // instead of copied. `passive_cards_never_end_a_turn` pins the normalisation.
        ends_turn: activation == CardActivation::Active && card.turn_policy == "END_TURN",
        turn_exclusive,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Action, Board, DrawReason, Engine, GameState, GameStatus, LATEST_RULESET_ID, Mode,
        MoveAction, Piece, PieceKind, Player, WinReason, ruleset_card,
    };

    #[test]
    fn execution_profile_requires_an_explicit_turn_policy_for_every_card() {
        let mut profile: serde_json::Value = serde_json::from_str(include_str!(
            "../../../rulesets/augment_chess_20261102/cards/implemented.json"
        ))
        .unwrap();
        profile["cards"][0]
            .as_object_mut()
            .unwrap()
            .remove("turn_exclusive");
        let error = serde_json::from_value::<ExecutionProfile>(profile)
            .err()
            .unwrap();
        assert!(error.to_string().contains("turn_exclusive"));
    }

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

    /// Threefold repetition resolves by star count, and the fewer-star side wins.
    ///
    /// The source's `starWinner` returns the side with fewer stars, with equal totals drawing. This
    /// test previously asserted a plain draw regardless of stars, which was the disabled-tiebreak
    /// behaviour rather than the rule.
    #[test]
    fn latest_repetition_resolves_by_star_count() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state();
        // White holds fewer stars, so White must win the star tiebreak.
        state.half_star_totals = [2, 10];
        state.finalize_initial_setup();
        // Three repetitions need the cycle played twice more after the first pass.
        for _ in 0..2 {
            for (from, to) in [("g1", "f3"), ("g8", "f6"), ("f3", "g1"), ("f6", "g8")] {
                state = engine.apply(&state, movement(from, to)).unwrap().state;
            }
        }
        assert_eq!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::LowerStarsOnRepetition,
            }
        );
    }

    /// Equal star totals on repetition draw, since there is no fewer-star side.
    #[test]
    fn latest_repetition_draws_when_stars_are_equal() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state();
        state.half_star_totals = [4, 4];
        state.finalize_initial_setup();
        for _ in 0..2 {
            for (from, to) in [("g1", "f3"), ("g8", "f6"), ("f3", "g1"), ("f6", "g8")] {
                state = engine.apply(&state, movement(from, to)).unwrap().state;
            }
        }
        assert_eq!(
            state.status,
            GameStatus::Draw {
                reason: DrawReason::EqualStarsOnRepetition
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

    /// The extension rule is enabled in this snapshot, so reaching the star limit starts it.
    ///
    /// It used to be disabled here, which this test asserted. The source enables it by default
    /// (`localRuleOptions` defaults `deathmatchEnabled` to true) and starts it at 45 shared turns,
    /// so the old assertion recorded a bug rather than a rule.
    #[test]
    fn latest_starts_the_extension_at_the_star_limit() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state();
        // The extension starts once both sides have completed the star limit, since the shared turn
        // count is the smaller of the two.
        state.turns_taken = [45; 2];
        state = engine.apply(&state, movement("g1", "f3")).unwrap().state;
        let deathmatch = state
            .deathmatch
            .as_ref()
            .expect("the extension must start at the star limit");
        assert!(deathmatch.started_at_shared_turn >= 45);
        // The interval defaults to the manifest value.
        assert_eq!(deathmatch.interval_full_turns, 10);
        // Upstream-disabled cards stay rejected no matter how far porting gets.
        for disabled in ["qxe1", "democracy", "underground-bunker"] {
            assert!(
                engine
                    .grant_card(&state, 1, disabled, Player::White)
                    .is_err(),
                "{disabled} must stay rejected"
            );
        }
        state.applied_rules.push("garden".into());
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
        assert_eq!(catalog.len(), 281);
        assert_eq!(catalog.iter().filter(|c| c.enabled).count(), 272);
        // The supported count grows as cards are ported, so it is checked against
        // the execution profile itself instead of a pinned snapshot.
        let profile = crate::latest::execution_profile().len();
        assert_eq!(catalog.iter().filter(|c| c.supported()).count(), profile);
        for card in catalog {
            assert_eq!(
                ruleset_card(LATEST_RULESET_ID, &card.id).is_some(),
                card.supported()
            );
        }
        // Upstream-disabled cards stay rejected regardless of port progress.
        for disabled in [
            "qxe1",
            "democracy",
            "underground-bunker",
            "beacon",
            "blue-jeans",
            "chess-n-pow-30",
            "martyrdom",
            "monochrome-chess",
            "royal-shield",
        ] {
            assert!(
                ruleset_card(LATEST_RULESET_ID, disabled).is_none(),
                "{disabled} must stay rejected"
            );
        }
        assert!(ruleset_card("augment_chess_20260915", "qxe1").is_some());
        // A card still queued in `pending` must not be executable.
        for pending in crate::latest::pending_ids() {
            assert!(
                ruleset_card(LATEST_RULESET_ID, &pending).is_none(),
                "{pending} is pending but executable"
            );
        }
    }

    /// The catalog's canonical source records `turn_policy: PRESERVE_TURN` for eighteen passive
    /// cards, while the bundle's v2 builder rejects that pair. The field is inert for a passive, so
    /// the engine derives `ends_turn` from the activation instead of copying the policy.
    ///
    /// This pins both sides: no passive may end a turn, and the catalog really does still contain
    /// those eighteen entries, so the normalisation stays necessary rather than becoming dead code
    /// if the source is ever corrected.
    #[test]
    fn passive_cards_never_end_a_turn() {
        let mut passive_with_policy = 0;
        for card in latest_catalog() {
            let Some(definition) = card_definition(&card.id) else {
                continue;
            };
            if definition.activation == CardActivation::Passive
                || definition.activation == CardActivation::Rule
            {
                assert!(
                    !definition.ends_turn,
                    "{} is {:?} and must not end a turn",
                    card.id, definition.activation
                );
            }
            if card.activation == "PASSIVE" && card.turn_policy != "NOT_APPLICABLE" {
                passive_with_policy += 1;
            }
        }
        assert_eq!(
            passive_with_policy, 18,
            "the catalog's passive-with-policy set changed; re-check whether the normalisation is still needed"
        );
    }

    /// The extension ends the game after a full interval without irreversible progress, and the
    /// fewer-star side wins, matching the source's `deathmatch_inactivity` result.
    #[test]
    fn extension_ends_after_a_quiet_interval_and_the_fewer_star_side_wins() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state();
        state.turns_taken = [45; 2];
        state.half_star_totals = [3, 9];
        // Backdating the turn counter skips setup; re-run it so the opening has resolved.
        state.finalize_initial_setup();
        // Find a legal opening move rather than hard-coding one, since setup resolution and the
        // backdated turn counter can leave different pieces available.
        let first_move = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .find(|action| matches!(action, Action::Move(_)))
            .expect("the position must have a legal move");
        state = engine.apply(&state, first_move).unwrap().state;
        let starting = state.deathmatch.expect("the extension must be running");
        assert_eq!(starting.interval_full_turns, 10);

        // Set the gauge one tick short of the interval, then let a quiet turn finish it. Setting the
        // gauge directly isolates the ending rule from the several quiet turns needed to reach it,
        // which this test is not about.
        state.deathmatch.as_mut().unwrap().quiet_full_turns = starting.interval_full_turns - 1;
        // Pawn moves and captures reset the gauge by rule, so pick a quiet non-pawn move.
        let quiet = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .find(|action| match action {
                Action::Move(movement) => {
                    !matches!(
                        state.board.get(movement.from).map(|piece| piece.kind),
                        Some(PieceKind::Pawn | PieceKind::BerolinaPawn)
                    ) && state.board.get(movement.to).is_none()
                }
                _ => false,
            })
            .expect("a quiet move must exist");
        state = engine.apply(&state, quiet).unwrap().state;

        assert_eq!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::LowerStarsAfterDeathmatch,
            },
            "the fewer-star side must win when the extension ends: {:?}",
            state.deathmatch
        );
    }
}

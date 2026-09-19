use crate::{
    AMAZON, APPRENTICE_KNIGHTS, ASSASSIN, Action, ActionTarget, BERSERKER, BIG_BISHOP, BIG_ROOK,
    BRUTUS, CALLING_CARD, CHECKER, CLEANUP, CLEANUP_SACRIFICE, CONSTITUTIONAL_MONARCHY, COOL_GUY,
    CardActivation, CardInstance, CardPhase, CardRuntimeState, ChoiceContinuation, ChoiceRegistry,
    DEMOCRACY, DICE, DISARM, DRAGON, DUTCH, DeathmatchState, DelayedHazard, DelayedHazardKind,
    DiceLockStatus, DraftPhase, DraftState, DrawReason, EAGLE, EARLY_PROMOTION, EASTERN_POLICY,
    ELEPHANT_ESCAPE, EMPTY_LUNCHBOX, EffectCommand, EffectRegistry, EmptyLunchboxStatus,
    EnPassantState, Event, FAST_GROWTH, FIANCHETTO, FIELD_PROMOTION, FREEZE,
    ForcedMoveContinuation, ForcedMoveSource, FrozenStatus, GENEVA_CONVENTION, GRASSHOPPER, GUARD,
    GamePhase, GameState, GameStatus, HERALD, HOLDOUT, HOOK, HORDE, HYPOCRISY, IDOL,
    IMPLEMENTED_CARDS, INITIATIVE, IRON_MONARCH, InitiativeStatus, JESTER, JUDGMENT, JudgmentExile,
    KING_OF_THE_HILL, KNIGHTMASTER, KNIGHTMATE, LAST_STAND, LOCAL_CONSCRIPTION, LOG, LONDON_SYSTEM,
    LOYALIST, LargePieceStatus, LogStatus, MAGIC_GIRL, MARTYRDOM, MERCHANT_GUILD, MISSIONARY, Mode,
    MoveAction, MoveReplayState, ORDINATION, OTHERWORLD, PARRY, PAWN_CONVERSION, PEGASUS, PLATFORM,
    ParrotMemoryStatus, PendingChoice, PendingOtherworld, Piece, PieceAbility, PieceKind,
    PlatformRuleState, Player, QUEEN_CAVALRY, QUEENS_GAMBIT, QXE1, REAPER, REFORMATION, REPOSITION,
    REVELATION, ReplayCell, RepositionStatus, RulesetManifest, SIEGE_RAM, SIREN, SLIME, SPRINT,
    STANDARD_BEARER, SUMMON_COLOSSUS, Square, THIEF, TRICKSTER, Transition, UNDEAD, WINDMILL,
    WIZARD, WinReason, WindmillMode, implemented_card,
};
use std::collections::BTreeSet;
use std::fmt;

mod end;
mod middle;
mod rule;
mod september;

use crate::{
    BACKWARD_KNIGHT, CONVERSION, CORNER_KICK, ENCOURAGEMENT, EXHAUSTION, FILE_SURGE, GHOST,
    HORSE_RIDING, IMPERIAL_STUDIES, INJURY, LEAP, MAD_HORSE, MONGOLIAN_GAMBIT, OVERWHELM,
    QUEEN_AFTERIMAGE, RADICAL_CHARGE, RELIGIOUS_VICTORY, RETREAT, ROOK_LIFT,
};

pub struct Engine {
    ruleset: RulesetManifest,
    effects: EffectRegistry,
    choices: ChoiceRegistry,
}

/// Search-only random stream carried by one determinization.
///
/// The base seed must be derived from public search information. It must never contain the
/// actual `GameState` RNG state. Each transition that really consumes randomness advances the
/// event index exactly once; deterministic transitions leave the context untouched.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchChanceContext {
    base_seed: u64,
    event_index: u64,
}

impl SearchChanceContext {
    pub const DERIVATION_VERSION: &'static str = "augmentia.search-chance.splitmix64.v1";

    pub const fn new(base_seed: u64) -> Self {
        Self {
            base_seed,
            event_index: 0,
        }
    }

    pub const fn base_seed(&self) -> u64 {
        self.base_seed
    }

    pub const fn event_index(&self) -> u64 {
        self.event_index
    }

    fn next_event_seed(&mut self) -> u64 {
        let index = self.event_index;
        self.event_index = self.event_index.wrapping_add(1);
        mix_search_seed(
            self.base_seed ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x4348_414e_4345_5f31,
        )
    }
}

#[derive(Clone, Debug)]
pub struct SearchTransition {
    pub transition: Transition,
    pub sampled_chance: bool,
    pub chance_event_index: Option<u64>,
}

const MAX_EFFECT_EVENTS_PER_RESOLUTION: usize = 4_096;
const MAX_AUTOMATIC_TURN_ENDS_PER_TRANSITION: usize = 256;
const DRAFT_BALANCE_MAX_HALF_STAR_GAP: u64 = 2;
const DRAFT_BALANCE_EXTRA_ATTEMPTS: usize = 2;
const TRICKSTER_MOVEMENT_KINDS: &[PieceKind] = &[
    PieceKind::Queen,
    PieceKind::Rook,
    PieceKind::Bishop,
    PieceKind::Missionary,
    PieceKind::Knight,
    PieceKind::Pawn,
    PieceKind::Protestant,
    PieceKind::Herald,
    PieceKind::Cannon,
    PieceKind::Fanatic,
    PieceKind::PrimeMinister,
    PieceKind::Alibaba,
    PieceKind::Amazon,
    PieceKind::Cardinal,
    PieceKind::Unicorn,
    PieceKind::Jester,
    PieceKind::Camel,
    PieceKind::Hook,
    PieceKind::Grasshopper,
    PieceKind::Dragon,
    PieceKind::Man,
    PieceKind::Assassin,
    PieceKind::Reaper,
    PieceKind::Knightmaster,
    PieceKind::StandardBearer,
    PieceKind::Guard,
    PieceKind::Recruiter,
    PieceKind::Squire,
    PieceKind::Checker,
    PieceKind::CheckerKing,
    PieceKind::Wizard,
    PieceKind::Alfil,
    PieceKind::Windmill,
    PieceKind::Idol,
    PieceKind::Lobster,
    PieceKind::BabyBear,
    PieceKind::Bear,
    PieceKind::SiegeRam,
    PieceKind::MagicGirl,
    PieceKind::Berserker,
    PieceKind::Slime,
    PieceKind::Siren,
    PieceKind::Undead,
];
const KNIGHT_OFFSETS: &[(i8, i8)] = &[
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DraftOfferAverage {
    total_half_stars: u32,
    card_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DraftAverageGap {
    numerator: u64,
    denominator: u64,
}

impl Engine {
    pub fn new(ruleset: RulesetManifest) -> Self {
        Self {
            ruleset,
            effects: EffectRegistry::default(),
            choices: ChoiceRegistry::default(),
        }
    }

    pub fn bundled_20260807() -> Result<Self, crate::RulesetError> {
        Self::bundled_20260818()
    }

    pub fn bundled_20260818() -> Result<Self, crate::RulesetError> {
        RulesetManifest::bundled_20260818().map(Self::new)
    }

    pub fn bundled_20260915() -> Result<Self, crate::RulesetError> {
        RulesetManifest::bundled_20260915().map(Self::new)
    }

    pub fn ruleset(&self) -> &RulesetManifest {
        &self.ruleset
    }

    pub fn effects_mut(&mut self) -> &mut EffectRegistry {
        &mut self.effects
    }

    pub fn choices_mut(&mut self) -> &mut ChoiceRegistry {
        &mut self.choices
    }

    pub fn new_game(&self, mode: Mode, seed: u64) -> Result<GameState, EngineError> {
        self.new_game_transition(mode, seed)
            .map(|transition| transition.state)
    }

    /// Create a game together with its deterministic opening chance events.
    ///
    /// Callers that persist replay logs should prefer this method over `new_game` so the opening
    /// RULE check and initial draft offer are retained.
    pub fn new_game_transition(&self, mode: Mode, seed: u64) -> Result<Transition, EngineError> {
        if self.ruleset.mode(mode.id()).is_none() {
            return Err(EngineError::UnknownMode(mode.id().to_owned()));
        }
        if mode != Mode::Standard {
            return Err(EngineError::UnsupportedMode(mode.id().to_owned()));
        }

        let mut state = GameState::standard(self.ruleset.id.clone(), mode, seed);
        state.reset_repetition_history();
        let mut events = Vec::new();
        self.resolve_opening_rule(&mut state, &mut events)?;
        self.start_draft(&mut state, DraftPhase::Opening, Player::White, None, None);
        let GamePhase::Draft(draft) = &state.phase else {
            unreachable!("start_draft always installs a draft")
        };
        events.push(Event::DraftStarted {
            phase: DraftPhase::Opening,
            player: Player::White,
            offers: draft.offers.clone(),
        });
        Ok(Transition { state, events })
    }

    fn resolve_opening_rule(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        if !self.ruleset.opening_rule.enabled {
            return Ok(());
        }
        let (roll, triggered) = state.rng_mut().web_probability(
            self.ruleset.opening_rule.chance_numerator,
            self.ruleset.opening_rule.chance_denominator,
        );
        events.push(Event::RandomSample {
            purpose: "opening-rule-check".to_owned(),
            roll_53: roll,
        });

        let definition_id = if triggered {
            let mut executable_rules: Vec<_> = IMPLEMENTED_CARDS
                .iter()
                .filter(|card| card.activation == CardActivation::Rule)
                .filter(|card| {
                    self.ruleset.id == "augment_chess_20260915"
                        || card.id != crate::CAPTURE_THE_FLAG
                })
                .collect();
            executable_rules.sort_by_key(|card| card.name_ko);
            let (selection_roll, index) = state.rng_mut().web_choice_index(executable_rules.len());
            events.push(Event::RandomSample {
                purpose: "opening-rule-selection".to_owned(),
                roll_53: selection_roll,
            });
            index.map(|index| executable_rules[index].id.to_owned())
        } else {
            None
        };
        events.push(Event::OpeningRuleChecked {
            triggered,
            definition_id: definition_id.clone(),
        });
        if let Some(definition_id) = definition_id {
            self.install_rule(state, &definition_id, events)?;
        }
        Ok(())
    }

    /// Add a player-owned card as a draft/acquisition transition.
    ///
    /// Passive cards are applied immediately, matching the web bundle. Active cards receive one
    /// charge and appear in `legal_actions` without consuming the player's board move.
    pub fn grant_card(
        &self,
        state: &GameState,
        instance_id: u32,
        definition_id: &str,
        owner: Player,
    ) -> Result<Transition, EngineError> {
        self.ensure_ruleset(state)?;
        let definition = implemented_card(definition_id)
            .ok_or_else(|| EngineError::UnknownCard(definition_id.to_owned()))?;
        if definition.activation == CardActivation::Rule {
            return Err(EngineError::CardIsRule(definition_id.to_owned()));
        }
        if state
            .cards
            .iter()
            .any(|card| card.instance_id == instance_id)
        {
            return Err(EngineError::DuplicateCardInstance(instance_id));
        }

        let mut next = state.clone();
        next.add_card(CardInstance {
            instance_id,
            definition_id: definition_id.to_owned(),
            owner,
            half_stars: definition.half_stars,
            charges: u16::from(definition.activation == CardActivation::Active),
            available_after_turn: None,
        });
        let mut events = vec![Event::CardGranted {
            owner,
            instance_id,
            definition_id: definition_id.to_owned(),
        }];
        self.apply_card_definition(&mut next, definition_id, owner, &mut events);
        self.resolve_effect_queue_from(&mut next, &mut events, 0)?;
        Ok(Transition {
            state: next,
            events,
        })
    }

    fn apply_opening_card(
        &self,
        state: &mut GameState,
        actor: Player,
        definition_id: &str,
        target: &ActionTarget,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let none = || {
            (target == &ActionTarget::None)
                .then_some(())
                .ok_or_else(|| {
                    EngineError::Invariant(format!("{definition_id} must not have a target"))
                })
        };
        match definition_id {
            APPRENTICE_KNIGHTS => {
                none()?;
                for origin in opening_squares(actor, &["b2", "g2"], &["b7", "g7"]) {
                    let found = state.board.iter().find(|(_, piece)| {
                        piece.owner == actor
                            && piece.kind == PieceKind::Pawn
                            && state.card_runtime.piece_origins.get(&piece.id) == Some(&origin)
                    });
                    if let Some((square, mut pawn)) = found {
                        pawn.kind = PieceKind::Squire;
                        state.board.set(square, Some(pawn));
                        self.mark_transformed_piece(state, pawn);
                    }
                }
            }
            CHECKER => {
                none()?;
                for origin in opening_squares(actor.opponent(), &["a2", "h2"], &["a7", "h7"]) {
                    let found = state.board.iter().find(|(_, piece)| {
                        piece.owner == actor.opponent()
                            && piece.kind == PieceKind::Pawn
                            && state.card_runtime.piece_origins.get(&piece.id) == Some(&origin)
                    });
                    if let Some((square, mut pawn)) = found {
                        pawn.kind = PieceKind::Checker;
                        state.board.set(square, Some(pawn));
                        self.mark_transformed_piece(state, pawn);
                    }
                }
            }
            ELEPHANT_ESCAPE => {
                none()?;
                for square in opening_squares(actor, &["a3", "h3"], &["a6", "h6"]) {
                    let piece =
                        Piece::new(actor, PieceKind::Alfil).with_id(state.allocate_piece_id());
                    state.board.set(square, Some(piece));
                    self.register_spawned_piece(state, square, piece, true);
                }
            }
            GUARD => {
                none()?;
                let king_origin = opening_square(actor, "e1", "e8");
                let pawn_origin = opening_square(actor, "e2", "e7");
                let Some((square, mut pawn)) = state.board.iter().find(|(_, piece)| {
                    piece.owner == actor
                        && matches!(piece.kind, PieceKind::Pawn | PieceKind::Fanatic)
                        && state.card_runtime.piece_origins.get(&piece.id) == Some(&pawn_origin)
                }) else {
                    return Err(EngineError::Invariant(format!(
                        "guard has no pawn originating in front of {king_origin}"
                    )));
                };
                pawn.kind = PieceKind::Guard;
                state.board.set(square, Some(pawn));
                self.mark_transformed_piece(state, pawn);
            }
            LAST_STAND => {
                none()?;
                let pawns = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Pawn)
                    .collect::<Vec<_>>();
                for (square, mut pawn) in pawns {
                    self.clear_queens_gambit_protection(state, pawn.id);
                    pawn.kind = PieceKind::Fanatic;
                    state.board.set(square, Some(pawn));
                }
            }
            QUEEN_CAVALRY => {
                none()?;
                let mut candidates = state
                    .board
                    .iter()
                    .filter(|(square, piece)| {
                        piece.owner == actor && piece.kind == PieceKind::Pawn && square.file() == 3
                    })
                    .collect::<Vec<_>>();
                candidates.sort_by_key(|(square, _)| match actor {
                    Player::White => square.rank(),
                    Player::Black => 7 - square.rank(),
                });
                let Some((square, mut pawn)) = candidates.first().copied() else {
                    return Err(EngineError::Invariant(
                        "queen cavalry has no pawn on the d-file".to_owned(),
                    ));
                };
                pawn.kind = PieceKind::Knight;
                state.card_runtime.shielded_piece_ids.remove(&pawn.id);
                state.board.set(square, Some(pawn));
                self.mark_transformed_piece(state, pawn);
            }
            DUTCH => {
                none()?;
                let pieces = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor
                            && matches!(
                                piece.kind,
                                PieceKind::Rook | PieceKind::Bishop | PieceKind::Knight
                            )
                    })
                    .collect::<Vec<_>>();
                for (square, mut piece) in pieces {
                    self.clear_queens_gambit_protection(state, piece.id);
                    piece.kind = PieceKind::Windmill;
                    state.board.set(square, Some(piece));
                    state
                        .card_runtime
                        .windmill_modes
                        .insert(piece.id, WindmillMode::Bishop);
                    self.mark_transformed_piece(state, piece);
                }
            }
            HORDE => {
                none()?;
                self.apply_horde_layout(state, actor, events)?;
            }
            KNIGHTMATE => {
                none()?;
                let pieces = state.board.iter().collect::<Vec<_>>();
                let mut found_king = false;
                for (square, mut piece) in pieces {
                    if piece.owner != actor {
                        continue;
                    }
                    if self.is_royal_piece(state, piece) {
                        piece.kind = PieceKind::RoyalKnight;
                        state.card_runtime.royal_piece_ids.insert(piece.id);
                        found_king = true;
                    } else if piece.kind == PieceKind::Knight {
                        piece.kind = PieceKind::Man;
                    } else {
                        continue;
                    }
                    state.board.set(square, Some(piece));
                    self.mark_transformed_piece(state, piece);
                }
                if !found_king {
                    return Err(EngineError::Invariant("knightmate has no king".to_owned()));
                }
            }
            LONDON_SYSTEM => {
                none()?;
                self.apply_london_layout(state, actor, events);
            }
            SUMMON_COLOSSUS => {
                none()?;
                self.apply_summon_colossus(state, actor, events)?;
            }
            BIG_ROOK => {
                none()?;
                self.apply_big_rook_layout(state, actor, events);
            }
            BIG_BISHOP => {
                none()?;
                self.apply_big_bishop_layout(state, actor, events);
            }
            CALLING_CARD => {
                none()?;
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor.opponent()
                            && piece.kind != PieceKind::Pawn
                            && !self.is_royal_piece(state, *piece)
                            && !matches!(piece.kind, PieceKind::Guard | PieceKind::Jester)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "calling-card".to_owned(),
                    roll_53: roll,
                });
                let (_, piece) = candidates[index.ok_or_else(|| {
                    EngineError::Invariant("calling card has no target".to_owned())
                })?];
                state.card_runtime.calling_cards.insert(piece.id, actor);
            }
            DEMOCRACY => {
                none()?;
                state.card_runtime.democracy[actor.index()] = true;
            }
            FAST_GROWTH => {
                none()?;
                state.card_runtime.fast_growth[actor.index()] = true;
            }
            FIANCHETTO => {
                none()?;
                state.card_runtime.fianchetto[actor.index()] = true;
            }
            FIELD_PROMOTION => {
                none()?;
                state.card_runtime.field_promotion[actor.index()] = true;
            }
            HOLDOUT => {
                let (_, pawn) = piece_target(state, target)?;
                state
                    .card_runtime
                    .holdout_promotions
                    .insert(pawn.id, state.shared_turn_count().saturating_add(14));
            }
            IRON_MONARCH => {
                none()?;
                state.card_runtime.iron_monarch[actor.index()] = true;
            }
            LOYALIST => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.loyalist_piece_ids.insert(piece.id);
            }
            MARTYRDOM => {
                none()?;
                let bishops = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Bishop)
                    .collect::<Vec<_>>();
                for (square, bishop) in bishops {
                    self.remove_piece_for_card(
                        state,
                        square,
                        bishop,
                        actor.opponent(),
                        definition_id,
                        events,
                    );
                }
                for (_, pawn) in state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Pawn)
                {
                    state.card_runtime.shielded_piece_ids.insert(pawn.id);
                }
            }
            MERCHANT_GUILD => {
                none()?;
                self.apply_merchant_guild(state, actor, events)?;
            }
            OTHERWORLD => {
                none()?;
                let pawns = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Pawn)
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(pawns.len());
                events.push(Event::RandomSample {
                    purpose: "otherworld-pawn".to_owned(),
                    roll_53: roll,
                });
                let (square, pawn) = pawns[index
                    .ok_or_else(|| EngineError::Invariant("otherworld has no pawn".to_owned()))?];
                let origin = state
                    .card_runtime
                    .piece_origins
                    .get(&pawn.id)
                    .copied()
                    .unwrap_or(square);
                state.board.set(square, None);
                self.clear_removed_piece_statuses(state, pawn.id);
                state
                    .card_runtime
                    .pending_otherworld
                    .push(PendingOtherworld {
                        piece_id: pawn.id,
                        owner: actor,
                        square,
                        origin,
                        due_ply: state.ply.saturating_add(28),
                    });
            }
            PAWN_CONVERSION => {
                none()?;
                state.card_runtime.pawn_conversion[actor.index()] = true;
            }
            QUEENS_GAMBIT => {
                let (queen_square, queen) = piece_target(state, target)?;
                self.remove_piece_for_card(
                    state,
                    queen_square,
                    queen,
                    actor.opponent(),
                    definition_id,
                    events,
                );
                let candidate_files = (0_u8..8)
                    .filter(|file| ![2, 3, 4].contains(file))
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidate_files.len());
                events.push(Event::RandomSample {
                    purpose: "queens-gambit-file".to_owned(),
                    roll_53: roll,
                });
                let random_file = candidate_files[index.unwrap_or(0)];
                for (square, pawn) in state.board.iter().filter(|(square, piece)| {
                    piece.owner == actor
                        && piece.kind == PieceKind::Pawn
                        && (square.file() == 3 || square.file() == random_file)
                }) {
                    let _ = square;
                    state.card_runtime.protected_piece_ids.insert(pawn.id);
                    state.card_runtime.queens_gambit_protection.insert(pawn.id);
                }
            }
            QXE1 => {
                none()?;
                self.apply_qxe1(state, actor, events)?;
            }
            THIEF => {
                none()?;
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor
                            && piece.kind == PieceKind::Queen
                            && !self.is_royal_piece(state, *piece)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "thief-target".to_owned(),
                    roll_53: roll,
                });
                let (_, mut piece) = candidates[index.ok_or_else(|| {
                    EngineError::Invariant("thief has no nonroyal queen".to_owned())
                })?];
                let square = state
                    .board
                    .find_by_id(piece.id)
                    .expect("selected thief queen remains on board")
                    .0;
                self.clear_piece_kind_runtime(state, piece.id);
                piece.kind = PieceKind::Thief;
                state.board.set(square, Some(piece));
                state.card_runtime.submerged_piece_ids.insert(piece.id);
                self.mark_transformed_piece(state, piece);
            }
            crate::FALSE_START => {
                none()?;
                self.apply_false_start(state, actor);
            }
            crate::LOCUST_SWARM => {
                none()?;
                for (_, piece) in state.board.iter() {
                    if piece.owner == actor
                        && piece.kind != PieceKind::Pawn
                        && !state.card_runtime.moved_piece_ids.contains(&piece.id)
                    {
                        state.card_runtime.locust_swarm_piece_ids.insert(piece.id);
                    }
                }
            }
            _ => unreachable!("opening card dispatcher received {definition_id}"),
        }
        Ok(())
    }

    fn apply_false_start(&self, state: &mut GameState, actor: Player) {
        let direction = if actor == Player::White { 1 } else { -1 };
        let pieces = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == actor)
            .collect::<Vec<_>>();
        let mut shifts = [0_i8; 8];
        for file in 0..8_u8 {
            let in_file = pieces
                .iter()
                .filter(|(square, _)| square.file() == file)
                .collect::<Vec<_>>();
            for step in 1..=2_i8 {
                let clear = in_file.iter().all(|(square, _)| {
                    square.offset(0, direction * step).is_some_and(|target| {
                        state
                            .board
                            .get(target)
                            .is_none_or(|occupant| occupant.owner == actor)
                    })
                });
                if !clear {
                    break;
                }
                shifts[usize::from(file)] = step;
            }
        }

        let moving = pieces
            .into_iter()
            .filter(|(square, _)| shifts[usize::from(square.file())] > 0)
            .collect::<Vec<_>>();
        for (square, _) in &moving {
            state.board.set(*square, None);
        }
        for (square, piece) in moving {
            let shift = shifts[usize::from(square.file())];
            let target = square
                .offset(0, direction * shift)
                .expect("false-start shift was checked");
            state.board.set(target, Some(piece));
        }
        state.en_passant = None;
    }

    fn mark_transformed_piece(&self, state: &mut GameState, piece: Piece) {
        self.clear_queens_gambit_protection(state, piece.id);
        state.card_runtime.moved_piece_ids.insert(piece.id);
        self.mark_fresh_no_capture(state, piece);
    }

    fn choose_trickster_kind(&self, state: &mut GameState, events: &mut Vec<Event>) -> PieceKind {
        let (roll, index) = state
            .rng_mut()
            .web_choice_index(TRICKSTER_MOVEMENT_KINDS.len());
        events.push(Event::RandomSample {
            purpose: "trickster-movement".to_owned(),
            roll_53: roll,
        });
        TRICKSTER_MOVEMENT_KINDS[index.unwrap_or(0)]
    }

    fn piece_ability_kind(&self, state: &GameState, piece: Piece) -> PieceKind {
        if piece.kind == PieceKind::Trickster {
            state
                .card_runtime
                .trickster_move_kinds
                .get(&piece.id)
                .copied()
                .unwrap_or(PieceKind::Queen)
        } else {
            piece.kind
        }
    }

    fn piece_has_ability(&self, state: &GameState, piece: Piece, kind: PieceKind) -> bool {
        self.piece_ability_kind(state, piece) == kind
    }

    fn is_ranged_piece(&self, state: &GameState, piece: Piece) -> bool {
        matches!(
            self.piece_ability_kind(state, piece),
            PieceKind::Rook
                | PieceKind::Bishop
                | PieceKind::Queen
                | PieceKind::Bear
                | PieceKind::Amazon
                | PieceKind::Cardinal
                | PieceKind::Cannon
                | PieceKind::Herald
                | PieceKind::Hook
                | PieceKind::Protestant
                | PieceKind::Windmill
                | PieceKind::BigRook
                | PieceKind::Jester
                | PieceKind::Idol
        )
    }

    fn register_spawned_piece(
        &self,
        state: &mut GameState,
        square: Square,
        piece: Piece,
        moved: bool,
    ) {
        state.card_runtime.piece_origins.insert(piece.id, square);
        if moved {
            state.card_runtime.moved_piece_ids.insert(piece.id);
        }
        self.mark_fresh_no_capture(state, piece);
    }

    fn delete_piece_for_card(
        &self,
        state: &mut GameState,
        square: Square,
        piece: Piece,
        actor: Player,
        cause: &str,
        events: &mut Vec<Event>,
    ) {
        state.board.set(square, None);
        self.clear_removed_piece_statuses(state, piece.id);
        events.push(Event::PieceRemoved {
            actor,
            piece,
            at: square,
            cause: cause.to_owned(),
        });
    }

    fn remove_piece_for_card(
        &self,
        state: &mut GameState,
        square: Square,
        piece: Piece,
        remover: Player,
        cause: &str,
        events: &mut Vec<Event>,
    ) {
        state.board.set(square, None);
        self.record_new_capture_reactions(state, piece);
        self.clear_removed_piece_statuses(state, piece.id);
        events.push(Event::PieceRemoved {
            actor: remover,
            piece,
            at: square,
            cause: cause.to_owned(),
        });
        self.grant_wizard_mana_for_loss(state, piece.owner);
        self.resolve_reaper_deaths(state, &[(square, piece)], remover, events);
    }

    fn record_new_capture_reactions(&self, state: &mut GameState, captured: Piece) {
        state.card_runtime.magic_girl_surge[captured.owner.index()] = true;
        if captured.kind == PieceKind::Pawn && state.card_runtime.resolve[captured.owner.index()] {
            state.card_runtime.resolve_ready[captured.owner.index()] = true;
        }
        if state.card_runtime.recurrence_piece_ids.remove(&captured.id) {
            if !state
                .card_runtime
                .pending_recurrences
                .iter()
                .any(|pending| pending.piece.id == captured.id)
            {
                state
                    .card_runtime
                    .pending_recurrences
                    .push(crate::PendingRecurrence { piece: captured });
            }
            state
                .card_runtime
                .undead_resurrections
                .retain(|resurrection| resurrection.piece.id != captured.id);
            return;
        }
        let undead_ability = captured.kind == PieceKind::Undead
            || (captured.kind == PieceKind::Trickster
                && state
                    .card_runtime
                    .trickster_move_kinds
                    .get(&captured.id)
                    .is_some_and(|kind| *kind == PieceKind::Undead));
        if undead_ability {
            state
                .card_runtime
                .undead_resurrections
                .push(crate::UndeadResurrection {
                    piece: captured,
                    due_ply: state.ply.saturating_add(6),
                });
        }
    }

    fn replace_layout(
        &self,
        state: &mut GameState,
        actor: Player,
        layout: &[(&str, PieceKind, bool)],
        cause: &str,
        events: &mut Vec<Event>,
    ) {
        let targets: BTreeSet<_> = layout
            .iter()
            .map(|(name, _, _)| name.parse::<Square>().expect("static layout square"))
            .collect();
        let enemies = targets
            .iter()
            .filter_map(|square| {
                state
                    .board
                    .get(*square)
                    .filter(|piece| piece.owner != actor)
                    .map(|piece| (*square, piece))
            })
            .collect::<Vec<_>>();
        for (square, piece) in enemies {
            self.remove_piece_for_card(state, square, piece, actor, cause, events);
        }
        let allied = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == actor)
            .collect::<Vec<_>>();
        for (square, piece) in allied {
            state.board.set(square, None);
            self.clear_removed_piece_statuses(state, piece.id);
        }
        for &(name, kind, moved) in layout {
            let square = name.parse::<Square>().expect("static layout square");
            let piece = Piece::new(actor, kind).with_id(state.allocate_piece_id());
            state.board.set(square, Some(piece));
            self.register_spawned_piece(state, square, piece, moved);
        }
    }

    fn apply_london_layout(&self, state: &mut GameState, actor: Player, events: &mut Vec<Event>) {
        const WHITE: &[(&str, PieceKind, bool)] = &[
            ("a1", PieceKind::Rook, false),
            ("d1", PieceKind::Queen, false),
            ("e1", PieceKind::King, false),
            ("h1", PieceKind::Rook, false),
            ("f4", PieceKind::Bishop, true),
            ("d3", PieceKind::Bishop, true),
            ("f3", PieceKind::Knight, true),
            ("d2", PieceKind::Knight, true),
            ("a2", PieceKind::Pawn, false),
            ("b2", PieceKind::Pawn, false),
            ("c3", PieceKind::Pawn, true),
            ("d4", PieceKind::Pawn, true),
            ("e3", PieceKind::Pawn, true),
            ("f2", PieceKind::Pawn, false),
            ("g2", PieceKind::Pawn, false),
            ("h2", PieceKind::Pawn, false),
        ];
        const BLACK: &[(&str, PieceKind, bool)] = &[
            ("a8", PieceKind::Rook, false),
            ("d8", PieceKind::Queen, false),
            ("e8", PieceKind::King, false),
            ("h8", PieceKind::Rook, false),
            ("f5", PieceKind::Bishop, true),
            ("d6", PieceKind::Bishop, true),
            ("f6", PieceKind::Knight, true),
            ("d7", PieceKind::Knight, true),
            ("a7", PieceKind::Pawn, false),
            ("b7", PieceKind::Pawn, false),
            ("c6", PieceKind::Pawn, true),
            ("d5", PieceKind::Pawn, true),
            ("e6", PieceKind::Pawn, true),
            ("f7", PieceKind::Pawn, false),
            ("g7", PieceKind::Pawn, false),
            ("h7", PieceKind::Pawn, false),
        ];
        self.replace_layout(
            state,
            actor,
            if actor == Player::White { WHITE } else { BLACK },
            LONDON_SYSTEM,
            events,
        );
    }

    fn apply_horde_layout(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let Some((king_square, king)) = state
            .board
            .iter()
            .find(|(_, piece)| piece.owner == actor && self.is_royal_piece(state, *piece))
        else {
            return Err(EngineError::Invariant("horde has no king".to_owned()));
        };
        let home = opening_square(actor, "e1", "e8");
        if state
            .board
            .get(home)
            .is_some_and(|piece| piece.owner != actor)
        {
            return Err(EngineError::Invariant(
                "horde king home is occupied by the enemy".to_owned(),
            ));
        }
        let allied = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == actor && piece.id != king.id)
            .collect::<Vec<_>>();
        for (square, piece) in allied {
            state.board.set(square, None);
            self.clear_removed_piece_statuses(state, piece.id);
        }
        state.board.set(king_square, None);
        state.board.set(home, Some(king));
        state.card_runtime.moved_piece_ids.insert(king.id);
        state
            .card_runtime
            .piece_origins
            .entry(king.id)
            .or_insert(home);

        let ranks: &[u8] = if actor == Player::White {
            &[0, 1, 2, 3]
        } else {
            &[7, 6, 5, 4]
        };
        let extra_rank = if actor == Player::White { 4 } else { 3 };
        let mut squares = Vec::new();
        for &rank in ranks {
            for file in 0..8 {
                squares.push(Square::new(file, rank).expect("horde square"));
            }
        }
        for file in [1, 2, 6, 5] {
            squares.push(Square::new(file, extra_rank).expect("horde extra square"));
        }
        for square in squares {
            if square == home || state.board.get(square).is_some() {
                continue;
            }
            let pawn = Piece::new(actor, PieceKind::Pawn).with_id(state.allocate_piece_id());
            state.board.set(square, Some(pawn));
            self.register_spawned_piece(state, square, pawn, false);
        }
        let _ = events;
        Ok(())
    }

    fn apply_summon_colossus(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let anchor = opening_square(actor, "d3", "d7");
        let pawns = web_ordered_pieces(state)
            .into_iter()
            .filter(|(square, piece)| {
                piece.owner == actor
                    && piece.kind == PieceKind::Pawn
                    && !matches!(square.file(), 0 | 7)
            })
            .collect::<Vec<_>>();
        if pawns.len() != 6 {
            return Err(EngineError::Invariant(
                "summon colossus requires the six non-rook-file pawns".to_owned(),
            ));
        }
        let sacrifice_ids: BTreeSet<_> = pawns.iter().map(|(_, piece)| piece.id).collect();
        for cell in large_piece_cells(anchor) {
            if state
                .board
                .get(cell)
                .is_some_and(|piece| !sacrifice_ids.contains(&piece.id))
            {
                return Err(EngineError::Invariant(
                    "colossus footprint is blocked".to_owned(),
                ));
            }
        }
        for (square, pawn) in pawns {
            self.remove_piece_for_card(
                state,
                square,
                pawn,
                actor.opponent(),
                SUMMON_COLOSSUS,
                events,
            );
        }
        let colossus = Piece::new(actor, PieceKind::Colossus).with_id(state.allocate_piece_id());
        state.board.set(anchor, Some(colossus));
        state.card_runtime.large_pieces.insert(
            colossus.id,
            LargePieceStatus {
                anchor,
                hp: 3,
                max_hp: 3,
            },
        );
        self.register_spawned_piece(state, anchor, colossus, true);
        Ok(())
    }

    fn apply_big_rook_layout(&self, state: &mut GameState, actor: Player, events: &mut Vec<Event>) {
        let anchor = opening_square(actor, "g2", "g8");
        let placements: &[(&str, PieceKind)] = if actor == Player::White {
            &[
                ("f3", PieceKind::Knight),
                ("g3", PieceKind::Pawn),
                ("h3", PieceKind::Pawn),
            ]
        } else {
            &[
                ("f6", PieceKind::Knight),
                ("g6", PieceKind::Pawn),
                ("h6", PieceKind::Pawn),
            ]
        };
        let mut targets = large_piece_cells(anchor);
        targets.extend(
            placements
                .iter()
                .map(|(name, _)| name.parse::<Square>().expect("static ROOK square")),
        );
        targets.sort();
        targets.dedup();
        for square in targets {
            if let Some(piece) = state.board.get(square) {
                let remover = if piece.owner == actor {
                    actor.opponent()
                } else {
                    actor
                };
                self.remove_piece_for_card(state, square, piece, remover, BIG_ROOK, events);
            }
        }
        for &(name, kind) in placements {
            let square = name.parse::<Square>().expect("static ROOK square");
            let piece = Piece::new(actor, kind).with_id(state.allocate_piece_id());
            state.board.set(square, Some(piece));
            self.register_spawned_piece(state, square, piece, true);
        }
        let rook = Piece::new(actor, PieceKind::BigRook).with_id(state.allocate_piece_id());
        state.board.set(anchor, Some(rook));
        state.card_runtime.large_pieces.insert(
            rook.id,
            LargePieceStatus {
                anchor,
                hp: 2,
                max_hp: 2,
            },
        );
        self.register_spawned_piece(state, anchor, rook, false);
    }

    fn apply_big_bishop_layout(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        let anchor = opening_square(actor, "f2", "f8");
        let placements: &[(&str, PieceKind)] = if actor == Player::White {
            &[
                ("f3", PieceKind::Pawn),
                ("g3", PieceKind::Pawn),
                ("h3", PieceKind::Knight),
            ]
        } else {
            &[
                ("f6", PieceKind::Pawn),
                ("g6", PieceKind::Pawn),
                ("h6", PieceKind::Knight),
            ]
        };
        let mut targets = large_piece_cells(anchor);
        targets.extend(
            placements
                .iter()
                .map(|(name, _)| name.parse::<Square>().expect("static BISHOP square")),
        );
        targets.sort();
        targets.dedup();
        for square in targets {
            if let Some(piece) = state.board.get(square) {
                let remover = if piece.owner == actor {
                    actor.opponent()
                } else {
                    actor
                };
                self.remove_piece_for_card(state, square, piece, remover, BIG_BISHOP, events);
            }
        }
        for &(name, kind) in placements {
            let square = name.parse::<Square>().expect("static BISHOP square");
            let piece = Piece::new(actor, kind).with_id(state.allocate_piece_id());
            state.board.set(square, Some(piece));
            self.register_spawned_piece(state, square, piece, true);
        }
        let bishop = Piece::new(actor, PieceKind::BigBishop).with_id(state.allocate_piece_id());
        state.board.set(anchor, Some(bishop));
        state.card_runtime.large_pieces.insert(
            bishop.id,
            LargePieceStatus {
                anchor,
                hp: 2,
                max_hp: 2,
            },
        );
        self.register_spawned_piece(state, anchor, bishop, false);
    }

    fn apply_merchant_guild(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let Some((from, mut king)) = state
            .board
            .iter()
            .find(|(_, piece)| piece.owner == actor && self.is_royal_piece(state, *piece))
        else {
            return Err(EngineError::Invariant(
                "merchant guild has no king".to_owned(),
            ));
        };
        let direction = if actor == Player::White { 1 } else { -1 };
        let to = from.offset(0, direction * 2).ok_or_else(|| {
            EngineError::Invariant("merchant destination is off board".to_owned())
        })?;
        if let Some(blocker) = state.board.get(to) {
            if blocker.owner != actor {
                return Err(EngineError::Invariant(
                    "merchant destination has an enemy".to_owned(),
                ));
            }
            self.remove_piece_for_card(
                state,
                to,
                blocker,
                actor.opponent(),
                MERCHANT_GUILD,
                events,
            );
        }
        state.board.set(from, None);
        self.clear_queens_gambit_protection(state, king.id);
        king.kind = PieceKind::Merchant;
        state.board.set(to, Some(king));
        state.card_runtime.merchant_gold.insert(king.id, 0);
        state.card_runtime.royal_piece_ids.insert(king.id);
        state.card_runtime.moved_piece_ids.insert(king.id);
        Ok(())
    }

    fn apply_qxe1(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let kings = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::King)
            .collect::<Vec<_>>();
        let queens = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Queen)
            .collect::<Vec<_>>();
        let mut plans = Vec::new();
        for &(king_square, king) in &kings {
            for &(queen_square, queen) in &queens {
                if clear_queen_line(&state.board, queen_square, king_square) {
                    plans.push((
                        queen_square
                            .file()
                            .abs_diff(king_square.file())
                            .max(queen_square.rank().abs_diff(king_square.rank())),
                        king_square,
                        king,
                        queen_square,
                        queen,
                    ));
                }
            }
        }
        plans.sort_by_key(|plan| (plan.0, plan.1, plan.3));
        let Some((_, throne, king, queen_square, queen)) = plans.first().copied() else {
            return Err(EngineError::Invariant(
                "qxe1 has no usurpation line".to_owned(),
            ));
        };
        state.board.set(queen_square, None);
        self.remove_piece_for_card(state, throne, king, actor, QXE1, events);
        state.board.set(throne, Some(queen));
        state.card_runtime.royal_piece_ids.insert(queen.id);
        state.card_runtime.moved_piece_ids.insert(queen.id);
        Ok(())
    }

    fn is_royal_piece(&self, state: &GameState, piece: Piece) -> bool {
        piece.kind == PieceKind::King
            || matches!(
                piece.kind,
                PieceKind::RoyalKnight | PieceKind::Merchant | PieceKind::DarkWizard
            )
            || state.card_runtime.royal_piece_ids.contains(&piece.id)
    }

    fn opening_card_none_legal(
        &self,
        state: &GameState,
        definition_id: &str,
        actor: Player,
    ) -> bool {
        let has = |kind| {
            state
                .board
                .iter()
                .any(|(_, piece)| piece.owner == actor && piece.kind == kind)
        };
        match definition_id {
            APPRENTICE_KNIGHTS => opening_squares(actor, &["b2", "g2"], &["b7", "g7"])
                .into_iter()
                .any(|origin| {
                    state.board.iter().any(|(_, piece)| {
                        piece.owner == actor
                            && piece.kind == PieceKind::Pawn
                            && state.card_runtime.piece_origins.get(&piece.id) == Some(&origin)
                    })
                }),
            CHECKER => opening_squares(actor.opponent(), &["a2", "h2"], &["a7", "h7"])
                .into_iter()
                .any(|origin| {
                    state.board.iter().any(|(_, piece)| {
                        piece.owner == actor.opponent()
                            && piece.kind == PieceKind::Pawn
                            && state.card_runtime.piece_origins.get(&piece.id) == Some(&origin)
                    })
                }),
            ELEPHANT_ESCAPE => opening_squares(actor, &["a3", "h3"], &["a6", "h6"])
                .into_iter()
                .all(|square| self.board_piece_at(state, square).is_none()),
            GUARD => {
                let origin = opening_square(actor, "e2", "e7");
                state.board.iter().any(|(_, piece)| {
                    piece.owner == actor
                        && matches!(piece.kind, PieceKind::Pawn | PieceKind::Fanatic)
                        && state.card_runtime.piece_origins.get(&piece.id) == Some(&origin)
                })
            }
            LAST_STAND | OTHERWORLD => has(PieceKind::Pawn),
            QUEEN_CAVALRY => state.board.iter().any(|(square, piece)| {
                piece.owner == actor && piece.kind == PieceKind::Pawn && square.file() == 3
            }),
            CALLING_CARD => state.board.iter().any(|(_, piece)| {
                piece.owner == actor.opponent()
                    && piece.kind != PieceKind::Pawn
                    && !self.is_royal_piece(state, piece)
                    && !matches!(piece.kind, PieceKind::Guard | PieceKind::Jester)
            }),
            DEMOCRACY => has(PieceKind::Pawn) && !state.card_runtime.democracy[actor.index()],
            FIANCHETTO => has(PieceKind::Bishop),
            FIELD_PROMOTION => has(PieceKind::Pawn),
            HORDE => {
                state
                    .board
                    .iter()
                    .any(|(_, piece)| piece.owner == actor && self.is_royal_piece(state, piece))
                    && self
                        .board_piece_at(state, opening_square(actor, "e1", "e8"))
                        .is_none_or(|piece| piece.owner == actor)
            }
            KNIGHTMATE => state
                .board
                .iter()
                .any(|(_, piece)| piece.owner == actor && self.is_royal_piece(state, piece)),
            MERCHANT_GUILD => state
                .board
                .iter()
                .find(|(_, piece)| piece.owner == actor && self.is_royal_piece(state, *piece))
                .and_then(|(square, _)| {
                    square.offset(0, if actor == Player::White { 2 } else { -2 })
                })
                .is_some_and(|destination| {
                    self.board_piece_at(state, destination).is_none_or(|piece| {
                        piece.owner == actor
                            && !state.card_runtime.large_pieces.contains_key(&piece.id)
                    })
                }),
            QXE1 => state.board.iter().any(|(king_square, king)| {
                king.owner == actor
                    && king.kind == PieceKind::King
                    && state.board.iter().any(|(queen_square, queen)| {
                        queen.owner == actor
                            && queen.kind == PieceKind::Queen
                            && clear_queen_line(&state.board, queen_square, king_square)
                    })
            }),
            MARTYRDOM => has(PieceKind::Bishop) && has(PieceKind::Pawn),
            SUMMON_COLOSSUS => {
                let pawns = state
                    .board
                    .iter()
                    .filter(|(square, piece)| {
                        piece.owner == actor
                            && piece.kind == PieceKind::Pawn
                            && !matches!(square.file(), 0 | 7)
                    })
                    .map(|(_, piece)| piece.id)
                    .collect::<BTreeSet<_>>();
                pawns.len() == 6
                    && large_piece_cells(opening_square(actor, "d3", "d7"))
                        .into_iter()
                        .all(|square| {
                            self.board_piece_at(state, square)
                                .is_none_or(|piece| pawns.contains(&piece.id))
                        })
            }
            THIEF => state.board.iter().any(|(_, piece)| {
                piece.owner == actor
                    && piece.kind == PieceKind::Queen
                    && !self.is_royal_piece(state, piece)
            }),
            crate::FALSE_START => state.board.iter().any(|(_, piece)| piece.owner == actor),
            crate::LOCUST_SWARM => state.board.iter().any(|(_, piece)| {
                piece.owner == actor
                    && piece.kind != PieceKind::Pawn
                    && !state.card_runtime.moved_piece_ids.contains(&piece.id)
            }),
            BIG_BISHOP | BIG_ROOK | DUTCH | FAST_GROWTH | IRON_MONARCH | LONDON_SYSTEM
            | PAWN_CONVERSION => true,
            _ => false,
        }
    }

    /// Install a global RULE card. RULE cards are not owned and do not contribute stars.
    pub fn apply_rule(
        &self,
        state: &GameState,
        definition_id: &str,
    ) -> Result<Transition, EngineError> {
        self.ensure_ruleset(state)?;
        let mut next = state.clone();
        let mut events = Vec::new();
        self.install_rule(&mut next, definition_id, &mut events)?;
        Ok(Transition {
            state: next,
            events,
        })
    }

    fn install_rule(
        &self,
        state: &mut GameState,
        definition_id: &str,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let definition = implemented_card(definition_id)
            .ok_or_else(|| EngineError::UnknownCard(definition_id.to_owned()))?;
        if definition.activation != CardActivation::Rule {
            return Err(EngineError::CardIsNotRule(definition_id.to_owned()));
        }
        if state
            .applied_rules
            .iter()
            .any(|applied| applied == definition_id)
        {
            return Err(EngineError::RuleAlreadyApplied(definition_id.to_owned()));
        }

        state.applied_rules.push(definition_id.to_owned());
        state.applied_rules.sort();
        events.push(Event::RuleApplied {
            definition_id: definition_id.to_owned(),
        });
        match definition_id {
            crate::CAPTURE_THE_FLAG => self.install_capture_the_flag(state, events),
            PLATFORM => {
                let current_turn = state.shared_turn_count();
                state.card_runtime.platform = Some(PlatformRuleState {
                    cell: None,
                    spawned_at: current_turn,
                    next_at: current_turn.saturating_add(5),
                    triggered_piece_ids: BTreeSet::new(),
                    nonce: 0,
                });
                self.spawn_platform(state, current_turn, events);
            }
            REVELATION => {
                state.card_runtime.revelation = true;
                if let Some(deathmatch) = &mut state.deathmatch {
                    deathmatch.interval_full_turns = 5;
                    deathmatch.quiet_full_turns = deathmatch.quiet_full_turns.min(5);
                }
            }
            _ => {
                if !self.install_rule_ext(state, definition_id, events)? {
                    return Err(EngineError::Invariant(format!(
                        "implemented RULE {definition_id:?} has no installer"
                    )));
                }
            }
        }
        Ok(())
    }

    fn draft_candidates_for_categories(
        &self,
        state: &GameState,
        categories: &[CardPhase],
        excluded: &BTreeSet<String>,
    ) -> Vec<String> {
        let owned: BTreeSet<_> = state
            .cards
            .iter()
            .map(|card| card.definition_id.as_str())
            .collect();

        IMPLEMENTED_CARDS
            .iter()
            .filter(|card| card.activation != CardActivation::Rule)
            .filter(|card| !owned.contains(card.id))
            .filter(|card| !excluded.contains(card.id))
            .filter(|card| categories.contains(&card.phase))
            .map(|card| card.id.to_owned())
            .collect()
    }

    fn draw_weighted_draft_cards(
        &self,
        state: &mut GameState,
        phase: DraftPhase,
        categories: &[CardPhase],
        count: usize,
        excluded: &mut BTreeSet<String>,
    ) -> Vec<String> {
        let mut candidates = self.draft_candidates_for_categories(state, categories, excluded);
        let mut picked = Vec::new();
        while picked.len() < count && !candidates.is_empty() {
            let weights: Vec<_> = candidates
                .iter()
                .map(|candidate| draft_card_weight_units(candidate, phase))
                .collect();
            let total_weight: u64 = weights.iter().sum();
            let roll = state.rng_mut().next_unit_53();
            let scaled_roll = u128::from(roll) * u128::from(total_weight);
            let mut cumulative_weight = 0_u64;
            let index = weights
                .iter()
                .position(|weight| {
                    cumulative_weight = cumulative_weight.saturating_add(*weight);
                    scaled_roll <= u128::from(cumulative_weight) * (1_u128 << 53)
                })
                .unwrap_or(weights.len() - 1);
            let card = candidates.remove(index);
            excluded.insert(card.clone());
            picked.push(card);
        }
        picked
    }

    fn draw_draft_offers_once(
        &self,
        state: &mut GameState,
        _player: Player,
        phase: DraftPhase,
    ) -> Vec<String> {
        let mut excluded = BTreeSet::new();
        match phase {
            DraftPhase::Opening => self.draw_weighted_draft_cards(
                state,
                phase,
                &[CardPhase::Opening, CardPhase::Middle, CardPhase::Piece],
                3,
                &mut excluded,
            ),
            DraftPhase::Middle => {
                let mut offers = self.draw_weighted_draft_cards(
                    state,
                    phase,
                    &[CardPhase::Middle],
                    2,
                    &mut excluded,
                );
                offers.extend(self.draw_weighted_draft_cards(
                    state,
                    phase,
                    &[CardPhase::Piece],
                    1,
                    &mut excluded,
                ));
                if offers.len() < 3 {
                    offers.extend(self.draw_weighted_draft_cards(
                        state,
                        phase,
                        &[CardPhase::Middle, CardPhase::Piece],
                        3 - offers.len(),
                        &mut excluded,
                    ));
                }
                offers
            }
            DraftPhase::End => self.draw_weighted_draft_cards(
                state,
                phase,
                &[CardPhase::Middle, CardPhase::End],
                3,
                &mut excluded,
            ),
        }
    }

    fn draw_draft_offers(
        &self,
        state: &mut GameState,
        player: Player,
        phase: DraftPhase,
        balance_target: Option<DraftOfferAverage>,
    ) -> Vec<String> {
        let mut best = self.draw_draft_offers_once(state, player, phase);
        let Some(target) = balance_target else {
            return best;
        };
        if best.len() < 2 {
            return best;
        }
        let Some(mut best_gap) =
            draft_offer_average(&best).map(|average| draft_average_gap(average, target))
        else {
            return best;
        };
        if draft_gap_within_balance_limit(best_gap) {
            return best;
        }

        for _ in 0..DRAFT_BALANCE_EXTRA_ATTEMPTS {
            let candidate = self.draw_draft_offers_once(state, player, phase);
            let Some(candidate_gap) =
                draft_offer_average(&candidate).map(|average| draft_average_gap(average, target))
            else {
                continue;
            };
            if draft_gap_is_smaller(candidate_gap, best_gap) {
                best = candidate;
                best_gap = candidate_gap;
                if draft_gap_within_balance_limit(best_gap) {
                    break;
                }
            }
        }
        best
    }

    fn start_draft(
        &self,
        state: &mut GameState,
        phase: DraftPhase,
        player: Player,
        resume_turn: Option<Player>,
        balance_target: Option<DraftOfferAverage>,
    ) {
        self.return_judgment_exiles_for_draft(state, phase);
        let resume_turn = resume_turn.unwrap_or(Player::White);
        let offers = self.draw_draft_offers(state, player, phase, balance_target);
        state.turn = player;
        state.phase = GamePhase::Draft(DraftState {
            phase,
            offers,
            resume_turn,
        });
    }

    fn return_judgment_exiles_for_draft(&self, state: &mut GameState, phase: DraftPhase) {
        let pending = std::mem::take(&mut state.card_runtime.judgment_exiles);
        for exile in pending {
            if exile.return_at_draft != phase {
                state.card_runtime.judgment_exiles.push(exile);
                continue;
            }
            let destination = state
                .board
                .get(exile.square)
                .is_none()
                .then_some(exile.square)
                .or_else(|| {
                    let ranks: Vec<u8> = if exile.piece.owner == Player::White {
                        (0..8).collect()
                    } else {
                        (0..8).rev().collect()
                    };
                    ranks.into_iter().find_map(|rank| {
                        (0..8).find_map(|file| {
                            let square = Square::new(file, rank)?;
                            state.board.get(square).is_none().then_some(square)
                        })
                    })
                });
            if let Some(square) = destination {
                state.board.set(square, Some(exile.piece));
            } else {
                state.card_runtime.judgment_exiles.push(exile);
            }
        }
    }

    #[allow(dead_code)] // Preserved campaign handler; public black-magic now summons an ally.
    fn missionary_spawn_candidates(&self, state: &GameState, origin: Square) -> Vec<Square> {
        let open = |square: &Square| {
            self.board_piece_at(state, *square).is_none()
                && !state.card_runtime.black_holes.contains(square)
                && !state.card_runtime.barricades.contains(square)
                && state.card_runtime.crown_pos != Some(*square)
                && !state
                    .card_runtime
                    .pending_scarecrows
                    .iter()
                    .any(|pending| pending.square == *square)
                && !(state.card_runtime.collapsed_files_ranks
                    && (matches!(square.file(), 0 | 7) || matches!(square.rank(), 0 | 7)))
        };
        // Client row order runs from rank 8 to rank 1. Prefer diagonals, then
        // orthogonals, then the first nonempty Chebyshev ring.
        for offsets in [
            &[(-1, 1), (1, 1), (-1, -1), (1, -1)][..],
            &[(0, 1), (0, -1), (-1, 0), (1, 0)][..],
        ] {
            let candidates = offsets
                .iter()
                .filter_map(|&(df, dr)| origin.offset(df, dr))
                .filter(open)
                .collect::<Vec<_>>();
            if !candidates.is_empty() {
                return candidates;
            }
        }
        for distance in 2_i8..8 {
            let mut candidates = Vec::new();
            for row_delta in -distance..=distance {
                for file_delta in -distance..=distance {
                    if row_delta.abs().max(file_delta.abs()) == distance
                        && let Some(square) = origin.offset(file_delta, -row_delta)
                        && open(&square)
                    {
                        candidates.push(square);
                    }
                }
            }
            if !candidates.is_empty() {
                return candidates;
            }
        }
        Vec::new()
    }

    pub(super) fn black_magic_spawn_candidates(
        &self,
        state: &GameState,
        actor: Player,
    ) -> Vec<Square> {
        let Some((royal_square, _)) = web_ordered_pieces(state)
            .into_iter()
            .find(|(_, piece)| piece.owner == actor && self.is_royal_piece(state, *piece))
        else {
            return Vec::new();
        };
        let rank = royal_square.rank() as i8;
        let file = royal_square.file() as i8;
        (-1_i8..=1)
            .flat_map(move |dr| (-1_i8..=1).map(move |df| (dr, df)))
            .filter(|(dr, df)| (*dr, *df) != (0, 0))
            .map(|(dr, df)| (rank + dr, file + df))
            .filter(|(dr, df)| (0..8).contains(dr) && (0..8).contains(df))
            .map(|(dr, df)| Square::new(df as u8, dr as u8).unwrap())
            .filter(|square| state.board.get(*square).is_none())
            .collect()
    }

    #[allow(dead_code)]
    pub(super) fn resolve_black_tower_legacy(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let actor = state.turn;
        let pieces = web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, piece)| piece.owner == actor)
            .collect::<Vec<_>>();
        let royal = pieces
            .iter()
            .copied()
            .find(|(_, piece)| self.is_royal_piece(state, *piece))
            .ok_or_else(|| EngineError::Invariant("black magic has no king".to_owned()))?;
        let mut removed = 0_usize;
        for (square, mut piece) in pieces {
            if piece.id == royal.1.id {
                self.clear_piece_kind_runtime(state, piece.id);
                piece.kind = PieceKind::DarkWizard;
                state.board.set(square, Some(piece));
                state.card_runtime.moved_piece_ids.insert(piece.id);
                state.card_runtime.piece_origins.insert(piece.id, square);
                continue;
            }
            self.remove_piece_for_card(
                state,
                square,
                piece,
                actor.opponent(),
                crate::BLACK_MAGIC,
                events,
            );
            removed += 1;
        }
        let spawn_count = removed.div_ceil(2);
        let ranks: Vec<u8> = if actor == Player::White {
            (0..8).collect()
        } else {
            (0..8).rev().collect()
        };
        let open = ranks
            .into_iter()
            .flat_map(|rank| (0..8).filter_map(move |file| Square::new(file, rank)))
            .filter(|square| state.board.get(*square).is_none())
            .take(spawn_count)
            .collect::<Vec<_>>();
        for square in open {
            let monster = Piece::new(actor, PieceKind::Monster).with_id(state.allocate_piece_id());
            state.board.set(square, Some(monster));
            self.mark_fresh_no_capture(state, monster);
        }
        Ok(())
    }

    fn apply_card_definition(
        &self,
        state: &mut GameState,
        definition_id: &str,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        match definition_id {
            crate::HIGHLANDER => self.resolve_highlander_victory(state),
            BRUTUS => {
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == owner
                            && piece.kind == PieceKind::Rook
                            && !self.is_royal_piece(state, *piece)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "brutus-target".to_owned(),
                    roll_53: roll,
                });
                if let Some(index) = index {
                    let (square, mut piece) = candidates[index];
                    self.clear_piece_kind_runtime(state, piece.id);
                    piece.kind = PieceKind::Brutus;
                    state.board.set(square, Some(piece));
                    self.mark_transformed_piece(state, piece);
                }
            }
            crate::RECURRENCE => {
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == owner
                            && piece.kind != PieceKind::Pawn
                            && !state.card_runtime.recurrence_piece_ids.contains(&piece.id)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "recurrence-target".to_owned(),
                    roll_53: roll,
                });
                if let Some(index) = index {
                    state
                        .card_runtime
                        .recurrence_piece_ids
                        .insert(candidates[index].1.id);
                }
            }
            crate::MAJESTY => state.card_runtime.majesty[owner.index()] = true,
            KING_OF_THE_HILL => state.card_runtime.hill_king[owner.index()] = true,
            crate::KILLER_KING => state.card_runtime.killer_king[owner.index()] = true,
            crate::PROFICIENCY => state.card_runtime.proficiency[owner.index()] = true,
            crate::DISASSEMBLY => {}
            crate::MUTATION => state.card_runtime.mutation[owner.index()] = true,
            crate::LONG_EN_PASSANT => state.card_runtime.long_en_passant[owner.index()] = true,
            crate::VANGUARD => state.card_runtime.vanguard[owner.index()] = true,
            crate::INFILTRATION => state.card_runtime.infiltration[owner.index()] = true,
            crate::RESOLVE => state.card_runtime.resolve[owner.index()] = true,
            GENEVA_CONVENTION => state.card_runtime.geneva_convention[owner.index()] = true,
            SPRINT => state.card_runtime.pawn_sprint[owner.index()] = true,
            EARLY_PROMOTION => state.card_runtime.early_promotion[owner.index()] = true,
            INITIATIVE => {
                let target = owner.opponent();
                state.card_runtime.initiative[target.index()] = Some(InitiativeStatus {
                    by: owner,
                    start_turn: state.turns_taken[target.index()],
                    limit: 10,
                });
            }
            BACKWARD_KNIGHT => state.card_runtime.backward_knight[owner.index()] = true,
            CORNER_KICK => state.card_runtime.corner_kick[owner.index()] = true,
            FILE_SURGE => state.card_runtime.file_surge[owner.index()] = true,
            HORSE_RIDING => {
                let knights = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == owner && piece.kind == PieceKind::Knight)
                    .collect::<Vec<_>>();
                for (square, piece) in knights {
                    self.delete_piece_for_card(state, square, piece, owner, HORSE_RIDING, events);
                }
                state.card_runtime.horse_riding[owner.index()] = true;
            }
            INJURY => state.card_runtime.knight_injury[owner.opponent().index()] = true,
            LEAP => state.card_runtime.pawn_leap[owner.index()] = true,
            RADICAL_CHARGE => state.card_runtime.radical_charge[owner.index()] = true,
            RETREAT => state.card_runtime.retreat[owner.index()] = true,
            ROOK_LIFT => state.card_runtime.rook_lift[owner.index()] = true,
            ENCOURAGEMENT => state.card_runtime.encouragement[owner.index()] = true,
            EXHAUSTION => state.card_runtime.exhaustion[owner.opponent().index()] = true,
            GHOST => {
                for (_, piece) in state.board.iter() {
                    if piece.owner == owner && piece.kind == PieceKind::Pawn {
                        state.card_runtime.ghost_piece_ids.insert(piece.id);
                    }
                }
            }
            OVERWHELM => state.card_runtime.overwhelm[owner.index()] = true,
            IMPERIAL_STUDIES => state.card_runtime.imperial_studies[owner.index()] = true,
            MAD_HORSE => state.card_runtime.mad_horse[owner.index()] = true,
            QUEEN_AFTERIMAGE => state.card_runtime.queen_afterimage[owner.index()] = true,
            RELIGIOUS_VICTORY => {
                state.card_runtime.religious_victory[owner.index()] = true;
                self.resolve_religious_victory(state);
            }
            CONVERSION => {
                let knights = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == owner && piece.kind == PieceKind::Knight)
                    .collect::<Vec<_>>();
                for (square, mut piece) in knights {
                    self.clear_queens_gambit_protection(state, piece.id);
                    piece.kind = PieceKind::Bishop;
                    state.board.set(square, Some(piece));
                    state.card_runtime.moved_piece_ids.insert(piece.id);
                }
            }
            MONGOLIAN_GAMBIT => {
                let pieces = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == owner
                            && !self.is_royal_piece(state, *piece)
                            && piece.kind != PieceKind::Merchant
                    })
                    .collect::<Vec<_>>();
                for (square, mut piece) in pieces {
                    self.clear_piece_kind_runtime(state, piece.id);
                    self.clear_queens_gambit_protection(state, piece.id);
                    piece.kind = PieceKind::Knight;
                    state.board.set(square, Some(piece));
                    state.card_runtime.moved_piece_ids.insert(piece.id);
                }
            }
            _ => {}
        }
    }

    fn activate_cards_for_turn(
        &self,
        state: &mut GameState,
        player: Player,
        events: &mut Vec<Event>,
    ) {
        let ready: Vec<_> = state
            .cards
            .iter()
            .enumerate()
            .filter(|(_, card)| card.owner == player)
            .filter(|(_, card)| {
                card.available_after_turn
                    .is_some_and(|turn| state.turns_taken[player.index()] >= turn)
            })
            .map(|(index, card)| (index, card.instance_id, card.definition_id.clone()))
            .collect();

        for (index, instance_id, definition_id) in ready {
            state.cards[index].available_after_turn = None;
            if implemented_card(&definition_id)
                .is_some_and(|definition| definition.activation == CardActivation::Passive)
            {
                self.apply_card_definition(state, &definition_id, player, events);
            }
            events.push(Event::CardActivated {
                owner: player,
                instance_id,
                definition_id,
            });
        }
    }

    fn apply_draft_pick(
        &self,
        state: &GameState,
        action: Action,
        offer_index: u16,
    ) -> Result<Transition, EngineError> {
        let mut next = state.clone();
        let actor = next.turn;
        let draft = match &next.phase {
            GamePhase::Draft(draft) => draft.clone(),
            _ => {
                return Err(EngineError::Invariant(
                    "draft action outside draft".to_owned(),
                ));
            }
        };
        let definition_id = draft
            .offers
            .get(usize::from(offer_index))
            .cloned()
            .ok_or_else(|| EngineError::Invariant("legal draft index is missing".to_owned()))?;
        let definition = implemented_card(&definition_id)
            .ok_or_else(|| EngineError::UnknownCard(definition_id.clone()))?;
        let instance_id = next.allocate_card_instance_id();
        let delayed = draft_card_uses_next_turn_delay(draft.phase, definition.phase);
        next.add_card(CardInstance {
            instance_id,
            definition_id: definition_id.clone(),
            owner: actor,
            half_stars: definition.half_stars,
            charges: u16::from(definition.activation == CardActivation::Active),
            available_after_turn: delayed
                .then(|| next.turns_taken[actor.index()].saturating_add(1)),
        });
        if draft.phase == DraftPhase::Opening
            && next.ply == 0
            && definition.phase == CardPhase::Opening
            && definition.activation == CardActivation::Active
        {
            next.card_runtime.opening_auto_card_ids.insert(instance_id);
        }
        let mut events = vec![
            Event::ActionApplied { actor, action },
            Event::DraftPicked {
                phase: draft.phase,
                player: actor,
                offer_index,
                definition_id: definition_id.clone(),
            },
            Event::CardGranted {
                owner: actor,
                instance_id,
                definition_id: definition_id.clone(),
            },
        ];
        if !delayed && definition.activation == CardActivation::Passive {
            self.apply_card_definition(&mut next, &definition_id, actor, &mut events);
        }

        if actor == Player::White {
            self.start_draft(
                &mut next,
                draft.phase,
                Player::Black,
                Some(draft.resume_turn),
                draft_offer_average(&draft.offers),
            );
            let GamePhase::Draft(next_draft) = &next.phase else {
                unreachable!("start_draft always installs a draft")
            };
            if next_draft.offers.is_empty() {
                events.push(Event::DraftSkipped {
                    phase: draft.phase,
                    player: Player::Black,
                });
                self.complete_draft_phase(&mut next, draft.phase, draft.resume_turn, &mut events)?;
            } else {
                events.push(Event::DraftStarted {
                    phase: next_draft.phase,
                    player: Player::Black,
                    offers: next_draft.offers.clone(),
                });
            }
        } else {
            self.complete_draft_phase(&mut next, draft.phase, draft.resume_turn, &mut events)?;
        }

        Ok(Transition {
            state: next,
            events,
        })
    }

    fn complete_draft_phase(
        &self,
        state: &mut GameState,
        phase: DraftPhase,
        resume_turn: Player,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        state.completed_drafts[phase.index()] = true;
        state.phase = GamePhase::Playing;
        state.turn = resume_turn;
        state.actions_remaining = 1;
        state.cards_used_this_turn[resume_turn.index()] = 0;
        state.take_turn_end_request();
        self.activate_cards_for_turn(state, resume_turn, events);
        events.push(Event::DraftCompleted { phase, resume_turn });
        events.push(Event::TurnChanged {
            player: resume_turn,
        });
        events.push(Event::TurnStarted {
            player: resume_turn,
            actions_remaining: state.actions_remaining,
        });
        self.resolve_effect_queue_from(state, events, 0)?;
        self.resolve_missing_king(state, events);
        if state.status != GameStatus::Ongoing {
            self.push_game_ended(state, events);
        } else if matches!(state.phase, GamePhase::Playing) {
            if state.take_turn_end_request() {
                self.finish_turn(state, resume_turn, events)?;
            } else {
                self.resolve_no_action_loss(state, resume_turn, events);
                if state.status == GameStatus::Ongoing {
                    state.record_current_position();
                }
            }
        }
        Ok(())
    }

    pub fn legal_actions(&self, state: &GameState) -> Result<Vec<Action>, EngineError> {
        self.ensure_ruleset(state)?;
        if state.status != GameStatus::Ongoing {
            return Ok(Vec::new());
        }
        match &state.phase {
            GamePhase::Playing => {
                let mut actions = if state.actions_remaining > 0 {
                    self.generate_move_actions(state)
                } else {
                    Vec::new()
                };
                if state.actions_remaining > 0 {
                    self.generate_piece_ability_actions(state, &mut actions);
                }
                self.generate_card_actions(state, &mut actions);
                actions.sort();
                actions.dedup();
                Ok(actions)
            }
            GamePhase::ForcedMove { piece_id, .. } => {
                Ok(self.generate_piece_move_actions(state, *piece_id))
            }
            GamePhase::Draft(draft) => Ok((0..draft.offers.len())
                .map(|index| Action::DraftPick {
                    offer_index: index as u16,
                })
                .collect()),
            GamePhase::AwaitingChoice(choice) => {
                if choice.owner != state.turn {
                    return Err(EngineError::Invariant(
                        "pending choice owner does not match turn".to_owned(),
                    ));
                }
                Ok((0..choice.options.len())
                    .map(|option_index| Action::ResolveChoice {
                        choice_id: choice.id,
                        option_index: option_index as u16,
                    })
                    .collect())
            }
        }
    }

    /// Apply an action transactionally. The input state is never mutated on success or failure.
    pub fn apply(&self, state: &GameState, action: Action) -> Result<Transition, EngineError> {
        self.ensure_ruleset(state)?;
        if state.status != GameStatus::Ongoing {
            return Err(EngineError::GameAlreadyEnded);
        }

        let legal_actions = self.legal_actions(state)?;
        if legal_actions.binary_search(&action).is_err()
            && !self.is_dynamic_card_action_legal(state, &action)
        {
            return Err(EngineError::IllegalAction(format!("{action:?}")));
        }

        let mut transition = match action.clone() {
            Action::Move(movement) => self.apply_move(state, action, movement),
            Action::UsePieceAbility {
                piece_id,
                ability,
                target,
            } => self.apply_piece_ability(state, action, piece_id, ability, target),
            Action::UseCard {
                instance_id,
                target,
            } => self.apply_active_card(state, action, instance_id, target),
            Action::DraftPick { offer_index } => self.apply_draft_pick(state, action, offer_index),
            Action::ResolveChoice {
                choice_id,
                option_index,
            } => self.apply_choice(state, action, choice_id, option_index),
        }?;
        self.refresh_submerged_pieces(&mut transition.state);
        self.resolve_empty_forced_move(&mut transition.state, &mut transition.events)?;
        self.advance_capture_flags(&mut transition.state, state.turn, false);
        self.refresh_submerged_pieces(&mut transition.state);
        if transition.state.status != GameStatus::Ongoing {
            self.push_game_ended(&transition.state, &mut transition.events);
        }
        Ok(transition)
    }

    /// Apply an action for search without revealing the actual future RNG stream.
    ///
    /// A deterministic transition is returned byte-for-byte as the normal engine would produce
    /// it. If the normal transition would consume RNG, the discarded probe is repeated from the
    /// same public state with a seed supplied by `SearchChanceContext`. The probe is never exposed
    /// to the caller, so the actual future outcome cannot enter tree statistics.
    pub fn apply_for_search(
        &self,
        state: &GameState,
        action: Action,
        chance: &mut SearchChanceContext,
    ) -> Result<SearchTransition, EngineError> {
        let probed = self.apply(state, action.clone())?;
        if probed.state.rng_state() == state.rng_state() {
            return Ok(SearchTransition {
                transition: probed,
                sampled_chance: false,
                chance_event_index: None,
            });
        }

        let event_index = chance.event_index();
        let seed = chance.next_event_seed();
        let mut sampled_state = state.clone();
        *sampled_state.rng_mut() = crate::DeterministicRng::new(seed);
        let transition = self.apply(&sampled_state, action)?;
        Ok(SearchTransition {
            transition,
            sampled_chance: true,
            chance_event_index: Some(event_index),
        })
    }

    /// Forced-move effects can become empty after later commands in the same
    /// transition remove or block the selected piece. Such a phase is not an
    /// actionable state and must honor its continuation before it is exposed
    /// to callers (and, in particular, before MCTS tries to evaluate it).
    fn resolve_empty_forced_move(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        for _ in 0..MAX_AUTOMATIC_TURN_ENDS_PER_TRANSITION {
            let (piece_id, continuation) = match &state.phase {
                GamePhase::ForcedMove {
                    piece_id,
                    continuation,
                    ..
                } => (*piece_id, *continuation),
                _ => return Ok(()),
            };
            if !self.generate_piece_move_actions(state, piece_id).is_empty() {
                return Ok(());
            }

            let actor = state.turn;
            state.phase = GamePhase::Playing;
            match continuation {
                ForcedMoveContinuation::FinishTurn => self.finish_turn(state, actor, events)?,
                ForcedMoveContinuation::ResumeTurn => {
                    self.resolve_no_action_loss(state, actor, events);
                    return Ok(());
                }
            }
        }
        Err(EngineError::AutomaticTurnEndLimit(
            MAX_AUTOMATIC_TURN_ENDS_PER_TRANSITION,
        ))
    }

    fn is_dynamic_card_action_legal(&self, state: &GameState, action: &Action) -> bool {
        let Action::UseCard {
            instance_id,
            target: ActionTarget::Squares(squares),
        } = action
        else {
            return false;
        };
        let Some(card) = state.cards.iter().find(|card| {
            card.instance_id == *instance_id
                && card.owner == state.turn
                && card.charges > 0
                && card
                    .available_after_turn
                    .is_none_or(|turn| state.turns_taken[state.turn.index()] >= turn)
        }) else {
            return false;
        };
        let unique = squares.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != squares.len() {
            return false;
        }
        match card.definition_id.as_str() {
            CLEANUP => {
                !squares.is_empty()
                    && squares.len() <= 3
                    && squares.iter().all(|square| {
                        state.board.get(*square).is_some_and(|piece| {
                            piece.owner == state.turn
                                && !state.card_runtime.large_pieces.contains_key(&piece.id)
                                && !matches!(
                                    piece.kind,
                                    PieceKind::Monster | PieceKind::Scarecrow | PieceKind::Football
                                )
                        })
                    })
            }
            HYPOCRISY => {
                squares.len() == 4
                    && squares.iter().all(|square| {
                        state.board.get(*square).is_none()
                            && !state.card_runtime.black_holes.contains(square)
                            && !(state.card_runtime.collapsed_files_ranks
                                && (matches!(square.file(), 0 | 7)
                                    || matches!(square.rank(), 0 | 7)))
                    })
            }
            _ => false,
        }
    }

    fn apply_piece_ability(
        &self,
        state: &GameState,
        action: Action,
        piece_id: u32,
        ability: PieceAbility,
        target: ActionTarget,
    ) -> Result<Transition, EngineError> {
        let mut next = state.clone();
        let actor = next.turn;
        let (_, source_piece) = next.board.find_by_id(piece_id).ok_or_else(|| {
            EngineError::Invariant("legal ability has no source piece".to_owned())
        })?;
        if source_piece.owner != actor {
            return Err(EngineError::Invariant(
                "ability source is not owned by the actor".to_owned(),
            ));
        }
        if ability == PieceAbility::ColossusSector {
            let ActionTarget::Squares(squares) = target else {
                return Err(EngineError::Invariant(
                    "colossus sector requires four squares".to_owned(),
                ));
            };
            let mut events = vec![Event::ActionApplied { actor, action }];
            let mut seen = BTreeSet::new();
            for square in squares {
                let Some((anchor, target_piece)) = self.board_piece_entry_at(&next, square) else {
                    continue;
                };
                if target_piece.owner == actor
                    || target_piece.kind == PieceKind::Guard
                    || !seen.insert(target_piece.id)
                {
                    continue;
                }
                if next
                    .card_runtime
                    .shielded_piece_ids
                    .contains(&target_piece.id)
                    || next
                        .card_runtime
                        .protected_piece_ids
                        .contains(&target_piece.id)
                {
                    continue;
                }
                if let Some(status) = next.card_runtime.large_pieces.get_mut(&target_piece.id) {
                    status.hp = status.hp.saturating_sub(1);
                    if status.hp > 0 {
                        continue;
                    }
                }
                self.remove_piece_for_card(
                    &mut next,
                    anchor,
                    target_piece,
                    actor,
                    "colossus-sector",
                    &mut events,
                );
                self.resolve_calling_card_capture(
                    &mut next,
                    target_piece,
                    Some(source_piece),
                    &mut events,
                );
            }
            next.actions_remaining = next.actions_remaining.saturating_sub(1);
            next.ply = next.ply.saturating_add(1);
            next.quiet_ply = 0;
            next.mark_deathmatch_progress();
            self.resolve_after_counted_move(&mut next, &mut events);
            self.resolve_missing_king(&mut next, &mut events);
            if next.status == GameStatus::Ongoing {
                self.finish_turn(&mut next, actor, &mut events)?;
            } else {
                self.push_game_ended(&next, &mut events);
            }
            return Ok(Transition {
                state: next,
                events,
            });
        }
        if ability == PieceAbility::MerchantBuy {
            let (square, mut purchased) = piece_target(&next, &target)?;
            let cost = merchant_cost(purchased.kind);
            let gold = next.card_runtime.merchant_gold.entry(piece_id).or_default();
            *gold = gold.saturating_sub(cost);
            let royal_purchase = self.is_royal_piece(&next, purchased)
                && !next.card_runtime.democracy[purchased.owner.index()];
            if royal_purchase {
                next.status = GameStatus::Won {
                    winner: actor,
                    reason: WinReason::CardEffect("merchant-purchase".to_owned()),
                };
            } else {
                purchased.owner = actor;
                next.board.set(square, Some(purchased));
                next.card_runtime.moved_piece_ids.insert(purchased.id);
            }
            let mut events = vec![Event::ActionApplied { actor, action }];
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("merchant-buy:{piece_id}:{}:{cost}", purchased.id),
            });
            next.actions_remaining = next.actions_remaining.saturating_sub(1);
            if next.status == GameStatus::Ongoing {
                self.finish_turn(&mut next, actor, &mut events)?;
            } else {
                self.push_game_ended(&next, &mut events);
            }
            return Ok(Transition {
                state: next,
                events,
            });
        }
        let wizard = source_piece;
        if !self.piece_has_ability(&next, wizard, PieceKind::Wizard) {
            return Err(EngineError::Invariant(
                "wizard ability source is not a wizard".to_owned(),
            ));
        }
        let cost = match ability {
            PieceAbility::WizardLightning => 1,
            PieceAbility::WizardShield => 2,
            PieceAbility::WizardMeteor => 3,
            PieceAbility::WizardTimeStop => 5,
            PieceAbility::ColossusSector | PieceAbility::MerchantBuy => unreachable!(),
        };
        let mana = next.card_runtime.wizard_mana.entry(piece_id).or_default();
        if *mana < cost {
            return Err(EngineError::Invariant(
                "legal wizard ability has insufficient mana".to_owned(),
            ));
        }
        *mana -= cost;
        let mut events = vec![Event::ActionApplied { actor, action }];
        let mut ends_turn = true;
        match ability {
            PieceAbility::WizardLightning => {
                let square = square_target(&target)?;
                next.card_runtime.delayed_hazards.push(DelayedHazard {
                    kind: DelayedHazardKind::Lightning,
                    owner: actor,
                    trigger_after: actor.opponent(),
                    squares: vec![square],
                });
            }
            PieceAbility::WizardShield => {
                let (_, piece) = piece_target(&next, &target)?;
                next.card_runtime.shielded_piece_ids.insert(piece.id);
            }
            PieceAbility::WizardMeteor => {
                let square = square_target(&target)?;
                let base_file = square.file().min(6);
                let base_rank = square.rank().min(6);
                let mut squares = Vec::with_capacity(4);
                for file_delta in 0..=1 {
                    for rank_delta in 0..=1 {
                        squares.push(
                            Square::new(base_file + file_delta, base_rank + rank_delta)
                                .expect("clamped meteor square"),
                        );
                    }
                }
                next.card_runtime.delayed_hazards.push(DelayedHazard {
                    kind: DelayedHazardKind::Meteor,
                    owner: actor,
                    trigger_after: actor.opponent(),
                    squares,
                });
            }
            PieceAbility::WizardTimeStop => {
                next.card_runtime.skip_turns[actor.opponent().index()] = true;
                ends_turn = false;
            }
            PieceAbility::ColossusSector | PieceAbility::MerchantBuy => unreachable!(),
        }
        events.push(Event::EffectApplied {
            source_card_instance: 0,
            command: format!("wizard:{piece_id}:{ability:?}"),
        });
        if ends_turn {
            next.actions_remaining = next.actions_remaining.saturating_sub(1);
            self.finish_turn(&mut next, actor, &mut events)?;
        } else {
            self.resolve_no_action_loss(&mut next, actor, &mut events);
        }
        Ok(Transition {
            state: next,
            events,
        })
    }

    fn apply_active_card(
        &self,
        state: &GameState,
        action: Action,
        instance_id: u32,
        target: ActionTarget,
    ) -> Result<Transition, EngineError> {
        let mut next = state.clone();
        let actor = next.turn;
        let card_index = next
            .cards
            .iter()
            .position(|card| card.instance_id == instance_id)
            .ok_or_else(|| EngineError::Invariant("legal card action has no card".to_owned()))?;
        let definition_id = next.cards[card_index].definition_id.clone();
        let definition = *implemented_card(&definition_id).ok_or_else(|| {
            EngineError::Invariant(format!("legal card {definition_id:?} is not implemented"))
        })?;
        next.cards[card_index].charges = next.cards[card_index].charges.saturating_sub(1);
        next.cards_used_this_turn[actor.index()] =
            next.cards_used_this_turn[actor.index()].saturating_add(1);
        let mut events = vec![
            Event::ActionApplied { actor, action },
            Event::CardUsed {
                owner: actor,
                instance_id,
                definition_id: definition_id.clone(),
            },
        ];
        match definition_id.as_str() {
            APPRENTICE_KNIGHTS
            | BIG_BISHOP
            | BIG_ROOK
            | CALLING_CARD
            | CHECKER
            | DEMOCRACY
            | DUTCH
            | ELEPHANT_ESCAPE
            | FAST_GROWTH
            | FIANCHETTO
            | GUARD
            | HOLDOUT
            | HORDE
            | IRON_MONARCH
            | KNIGHTMATE
            | LAST_STAND
            | LONDON_SYSTEM
            | LOYALIST
            | MARTYRDOM
            | MERCHANT_GUILD
            | OTHERWORLD
            | PAWN_CONVERSION
            | QUEEN_CAVALRY
            | QUEENS_GAMBIT
            | QXE1
            | SUMMON_COLOSSUS
            | THIEF
            | crate::FALSE_START
            | crate::LOCUST_SWARM => {
                self.apply_opening_card(&mut next, actor, &definition_id, &target, &mut events)?;
            }
            PARRY => {
                let (_, target_piece) = piece_target(&next, &target)?;
                next.card_runtime.parry_piece_ids.insert(target_piece.id);
            }
            EMPTY_LUNCHBOX => {
                let (_, target_piece) = piece_target(&next, &target)?;
                next.card_runtime.empty_lunchboxes.insert(
                    target_piece.id,
                    EmptyLunchboxStatus {
                        by: actor,
                        deadline_turn: next.turns_taken[target_piece.owner.index()]
                            .saturating_add(3),
                    },
                );
            }
            DISARM => {
                let (_, target_piece) = piece_target(&next, &target)?;
                next.card_runtime.disarmed.insert(target_piece.id, 1);
            }
            DICE => {
                if target != ActionTarget::None {
                    return Err(EngineError::Invariant(
                        "dice must not have a target".to_owned(),
                    ));
                }
                let kinds = [
                    PieceKind::Pawn,
                    PieceKind::Knight,
                    PieceKind::Bishop,
                    PieceKind::Rook,
                    PieceKind::Queen,
                    PieceKind::King,
                ];
                let (roll, index) = next.rng_mut().web_choice_index(kinds.len());
                events.push(Event::RandomSample {
                    purpose: "dice".to_owned(),
                    roll_53: roll,
                });
                let kind = kinds[index.expect("six-sided die has a result")];
                next.card_runtime.dice_locks[actor.opponent().index()] = Some(DiceLockStatus {
                    kind,
                    remaining_turns: 2,
                });
            }
            FREEZE => {
                if target != ActionTarget::None {
                    return Err(EngineError::Invariant(
                        "freeze must not have a target".to_owned(),
                    ));
                }
                self.apply_freeze(&mut next, actor, &mut events);
            }
            REPOSITION => {
                let (_, target_piece) = piece_target(&next, &target)?;
                next.card_runtime.reposition = Some(RepositionStatus {
                    piece_id: target_piece.id,
                    used: false,
                });
            }
            crate::METAL => {
                let (_, target_piece) = piece_target(&next, &target)?;
                next.card_runtime
                    .metalized_piece_ids
                    .insert(target_piece.id);
                next.card_runtime.metal_cooldowns.remove(&target_piece.id);
            }
            ASSASSIN | GRASSHOPPER => {
                let (target_square, mut target_piece) = piece_target(&next, &target)?;
                self.clear_queens_gambit_protection(&mut next, target_piece.id);
                target_piece.kind = if definition_id == ASSASSIN {
                    PieceKind::Assassin
                } else {
                    PieceKind::Grasshopper
                };
                next.board.set(target_square, Some(target_piece));
                next.card_runtime.moved_piece_ids.insert(target_piece.id);
                self.mark_fresh_no_capture(&mut next, target_piece);
            }
            CONSTITUTIONAL_MONARCHY
            | DRAGON
            | EASTERN_POLICY
            | HERALD
            | KNIGHTMASTER
            | PEGASUS
            | IDOL
            | JESTER
            | LOCAL_CONSCRIPTION
            | REAPER
            | WIZARD
            | LOG
            | MISSIONARY
            | STANDARD_BEARER
            | BERSERKER
            | MAGIC_GIRL
            | SIEGE_RAM
            | SIREN
            | SLIME
            | TRICKSTER
            | UNDEAD
            | crate::CLOCKWORK
            | crate::PARROT
            | crate::PALADIN
            | crate::PRINCESS
            | crate::OCTOPUS
            | crate::HEDGEHOG
            | crate::CAMPFIRE => {
                let (target_square, mut target_piece) = piece_target(&next, &target)?;
                if definition_id == MISSIONARY {
                    let candidates = self.missionary_spawn_candidates(&next, target_square);
                    let (roll, index) = next.rng_mut().web_choice_index(candidates.len());
                    let square = candidates[index.ok_or_else(|| {
                        EngineError::Invariant(
                            "legal missionary card has no spawn square".to_owned(),
                        )
                    })?];
                    events.push(Event::RandomSample {
                        purpose: "missionary-second-spawn".to_owned(),
                        roll_53: roll,
                    });
                    let summoned =
                        Piece::new(actor, PieceKind::Missionary).with_id(next.allocate_piece_id());
                    next.board.set(square, Some(summoned));
                    self.register_spawned_piece(&mut next, square, summoned, true);
                    self.clear_queens_gambit_protection(&mut next, target_piece.id);
                }
                self.clear_queens_gambit_protection(&mut next, target_piece.id);
                target_piece.kind = match definition_id.as_str() {
                    CONSTITUTIONAL_MONARCHY => PieceKind::PrimeMinister,
                    DRAGON => PieceKind::Dragon,
                    EASTERN_POLICY => PieceKind::Cannon,
                    HERALD => PieceKind::Herald,
                    KNIGHTMASTER => PieceKind::Knightmaster,
                    PEGASUS => PieceKind::Unicorn,
                    IDOL => PieceKind::Idol,
                    JESTER => PieceKind::Jester,
                    LOCAL_CONSCRIPTION => PieceKind::Recruiter,
                    REAPER => PieceKind::Reaper,
                    WIZARD => PieceKind::Wizard,
                    LOG => PieceKind::Log,
                    MISSIONARY => PieceKind::Missionary,
                    STANDARD_BEARER => PieceKind::StandardBearer,
                    BERSERKER => PieceKind::Berserker,
                    MAGIC_GIRL => PieceKind::MagicGirl,
                    SIEGE_RAM => PieceKind::SiegeRam,
                    SIREN => PieceKind::Siren,
                    SLIME => PieceKind::Slime,
                    TRICKSTER => PieceKind::Trickster,
                    UNDEAD => PieceKind::Undead,
                    crate::CLOCKWORK => PieceKind::Clockwork,
                    crate::PARROT => PieceKind::Parrot,
                    crate::PALADIN => PieceKind::Paladin,
                    crate::PRINCESS => PieceKind::Princess,
                    crate::OCTOPUS => PieceKind::Octopus,
                    crate::HEDGEHOG => PieceKind::Hedgehog,
                    crate::CAMPFIRE => PieceKind::Campfire,
                    _ => unreachable!(),
                };
                next.board.set(target_square, Some(target_piece));
                next.card_runtime.moved_piece_ids.insert(target_piece.id);
                self.mark_fresh_no_capture(&mut next, target_piece);
                match target_piece.kind {
                    PieceKind::Hedgehog => {
                        next.card_runtime
                            .bear_retaliations
                            .insert(target_piece.id, 2);
                    }
                    PieceKind::Herald if next.shared_turn_count() >= 2 => {
                        next.card_runtime.herald_jump_locked_until.insert(
                            target_piece.id,
                            next.turns_taken[actor.index()].saturating_add(1),
                        );
                    }
                    PieceKind::Wizard => {
                        next.card_runtime.wizard_mana.insert(target_piece.id, 0);
                    }
                    PieceKind::Reaper => {
                        next.card_runtime.reaper_souls.insert(target_piece.id, 0);
                    }
                    PieceKind::Trickster => {
                        let kind = self.choose_trickster_kind(&mut next, &mut events);
                        next.card_runtime
                            .trickster_move_kinds
                            .insert(target_piece.id, kind);
                    }
                    _ => {}
                }
            }
            ORDINATION => {
                let (target_square, mut target_piece) = piece_target(&next, &target)?;
                let sacrifice = web_ordered_pieces(&next).into_iter().find(|(_, piece)| {
                    piece.owner == actor
                        && piece.kind == PieceKind::Bishop
                        && piece.id != target_piece.id
                });
                let (sacrifice_square, sacrificed) = sacrifice.ok_or_else(|| {
                    EngineError::Invariant("ordination requires a second bishop".to_owned())
                })?;
                self.remove_piece_for_card(
                    &mut next,
                    sacrifice_square,
                    sacrificed,
                    actor.opponent(),
                    ORDINATION,
                    &mut events,
                );
                self.clear_queens_gambit_protection(&mut next, target_piece.id);
                target_piece.kind = PieceKind::Cardinal;
                next.board.set(target_square, Some(target_piece));
                next.card_runtime.moved_piece_ids.insert(target_piece.id);
                self.mark_fresh_no_capture(&mut next, target_piece);
            }
            AMAZON | HOOK => {
                let [queen_square, sacrifice_square] = two_square_target(&target)?;
                let mut queen = next.board.get(queen_square).ok_or_else(|| {
                    EngineError::Invariant("piece card has no selected queen".to_owned())
                })?;
                let sacrificed = next.board.get(sacrifice_square).ok_or_else(|| {
                    EngineError::Invariant("piece card has no sacrifice".to_owned())
                })?;
                next.board.set(sacrifice_square, None);
                self.clear_removed_piece_statuses(&mut next, sacrificed.id);
                events.push(Event::PieceRemoved {
                    actor: actor.opponent(),
                    piece: sacrificed,
                    at: sacrifice_square,
                    cause: definition_id.clone(),
                });
                self.resolve_reaper_deaths(
                    &mut next,
                    &[(sacrifice_square, sacrificed)],
                    actor.opponent(),
                    &mut events,
                );
                self.clear_queens_gambit_protection(&mut next, queen.id);
                queen.kind = if definition_id == AMAZON {
                    PieceKind::Amazon
                } else {
                    PieceKind::Hook
                };
                next.board.set(queen_square, Some(queen));
                next.card_runtime.moved_piece_ids.insert(queen.id);
                self.mark_fresh_no_capture(&mut next, queen);
            }
            EAGLE => {
                if target != ActionTarget::None {
                    return Err(EngineError::Invariant(
                        "alibaba must not have a target".to_owned(),
                    ));
                }
                let knights = web_ordered_pieces(&next)
                    .into_iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Knight)
                    .collect::<Vec<_>>();
                for &(square, mut knight) in &knights {
                    self.clear_queens_gambit_protection(&mut next, knight.id);
                    knight.kind = PieceKind::Alibaba;
                    next.board.set(square, Some(knight));
                    next.card_runtime.moved_piece_ids.insert(knight.id);
                    self.mark_fresh_no_capture(&mut next, knight);
                }
                for (origin, _) in knights {
                    let candidates = self.nearest_alibaba_candidates(&next, origin, actor);
                    if candidates.is_empty() {
                        break;
                    }
                    let (roll, index) = next.rng_mut().web_choice_index(candidates.len());
                    events.push(Event::RandomSample {
                        purpose: "alibaba-placement".to_owned(),
                        roll_53: roll,
                    });
                    let square = candidates[index.expect("non-empty Alibaba candidates")];
                    let piece =
                        Piece::new(actor, PieceKind::Alibaba).with_id(next.allocate_piece_id());
                    next.board.set(square, Some(piece));
                    next.card_runtime.moved_piece_ids.insert(piece.id);
                    self.mark_fresh_no_capture(&mut next, piece);
                }
            }
            REFORMATION => {
                if target != ActionTarget::None {
                    return Err(EngineError::Invariant(
                        "reformation must not have a target".to_owned(),
                    ));
                }
                let bishops = web_ordered_pieces(&next)
                    .into_iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Bishop)
                    .collect::<Vec<_>>();
                for (square, mut bishop) in bishops {
                    self.clear_queens_gambit_protection(&mut next, bishop.id);
                    bishop.kind = PieceKind::Protestant;
                    next.board.set(square, Some(bishop));
                    next.card_runtime.moved_piece_ids.insert(bishop.id);
                    self.mark_fresh_no_capture(&mut next, bishop);
                }
            }
            WINDMILL => {
                let [bishop_square, rook_square] = two_square_target(&target)?;
                for square in [bishop_square, rook_square] {
                    let mut piece = next.board.get(square).ok_or_else(|| {
                        EngineError::Invariant("windmill has a missing selected piece".to_owned())
                    })?;
                    self.clear_queens_gambit_protection(&mut next, piece.id);
                    piece.kind = PieceKind::Windmill;
                    next.board.set(square, Some(piece));
                    next.card_runtime.moved_piece_ids.insert(piece.id);
                    next.card_runtime
                        .windmill_modes
                        .insert(piece.id, WindmillMode::Bishop);
                    self.mark_fresh_no_capture(&mut next, piece);
                }
            }
            JUDGMENT => {
                let (target_square, target_piece) = piece_target(&next, &target)?;
                next.board.set(target_square, None);
                let return_at_draft = if !next.completed_drafts[DraftPhase::Middle.index()] {
                    Some(DraftPhase::Middle)
                } else if !next.completed_drafts[DraftPhase::End.index()] {
                    Some(DraftPhase::End)
                } else {
                    None
                };
                if let Some(return_at_draft) = return_at_draft {
                    next.card_runtime.judgment_exiles.push(JudgmentExile {
                        piece: target_piece,
                        square: target_square,
                        return_at_draft,
                    });
                } else {
                    self.clear_removed_piece_statuses(&mut next, target_piece.id);
                }
                let remover = if target_piece.owner == actor {
                    actor.opponent()
                } else {
                    actor
                };
                events.push(Event::PieceRemoved {
                    actor: remover,
                    piece: target_piece,
                    at: target_square,
                    cause: JUDGMENT.to_owned(),
                });
                if return_at_draft.is_none() {
                    self.grant_wizard_mana_for_loss(&mut next, target_piece.owner);
                    self.resolve_reaper_deaths(
                        &mut next,
                        &[(target_square, target_piece)],
                        remover,
                        &mut events,
                    );
                }
            }
            CLEANUP_SACRIFICE => {
                let (selected_square, selected_piece) = piece_target(&next, &target)?;
                let candidates = web_ordered_pieces(&next)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor.opponent()
                            && piece.kind == selected_piece.kind
                            && !self.is_royal_piece(&next, *piece)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = next.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "cleanup-sacrifice".to_owned(),
                    roll_53: roll,
                });
                let (victim_square, victim_piece) = candidates[index.ok_or_else(|| {
                    EngineError::Invariant(
                        "legal cleanup sacrifice has no matching enemy".to_owned(),
                    )
                })?];
                next.board.set(victim_square, None);
                self.clear_removed_piece_statuses(&mut next, victim_piece.id);
                events.push(Event::PieceRemoved {
                    actor,
                    piece: victim_piece,
                    at: victim_square,
                    cause: CLEANUP_SACRIFICE.to_owned(),
                });
                self.grant_wizard_mana_for_loss(&mut next, victim_piece.owner);
                self.resolve_reaper_deaths(
                    &mut next,
                    &[(victim_square, victim_piece)],
                    actor,
                    &mut events,
                );
                next.board.set(selected_square, None);
                self.clear_removed_piece_statuses(&mut next, selected_piece.id);
                events.push(Event::PieceRemoved {
                    actor: actor.opponent(),
                    piece: selected_piece,
                    at: selected_square,
                    cause: CLEANUP_SACRIFICE.to_owned(),
                });
                self.grant_wizard_mana_for_loss(&mut next, selected_piece.owner);
                self.resolve_reaper_deaths(
                    &mut next,
                    &[(selected_square, selected_piece)],
                    actor.opponent(),
                    &mut events,
                );
                self.break_initiative_by_capture(&mut next, victim_piece.owner, actor);
            }
            _ => {
                if !self.apply_middle_active_card(
                    &mut next,
                    actor,
                    &definition_id,
                    &target,
                    &mut events,
                )? && !self.apply_end_active_card(
                    &mut next,
                    actor,
                    &definition_id,
                    &target,
                    &mut events,
                )? {
                    return Err(EngineError::Invariant(format!(
                        "unsupported active card {definition_id:?} was legal"
                    )));
                }
            }
        }
        next.quiet_ply = 0;
        next.mark_deathmatch_progress();
        self.resolve_effect_queue_from(&mut next, &mut events, 0)?;
        self.resolve_missing_king(&mut next, &mut events);
        self.resolve_herald_victory(&mut next);
        if next.status != GameStatus::Ongoing {
            self.push_game_ended(&next, &mut events);
        } else if matches!(next.phase, GamePhase::Playing) {
            if definition.ends_turn || next.take_turn_end_request() {
                self.finish_turn(&mut next, actor, &mut events)?;
            } else {
                self.resolve_no_action_loss(&mut next, actor, &mut events);
            }
        }
        Ok(Transition {
            state: next,
            events,
        })
    }

    fn apply_choice(
        &self,
        state: &GameState,
        action: Action,
        choice_id: u32,
        option_index: u16,
    ) -> Result<Transition, EngineError> {
        let mut next = state.clone();
        let actor = next.turn;
        let choice = match &next.phase {
            GamePhase::AwaitingChoice(choice) => choice.clone(),
            _ => {
                return Err(EngineError::Invariant(
                    "choice action outside pending choice".to_owned(),
                ));
            }
        };
        if choice.id != choice_id || usize::from(option_index) >= choice.options.len() {
            return Err(EngineError::Invariant(
                "legal choice action does not match pending choice".to_owned(),
            ));
        }
        if let Some(piece_id) = choice
            .handler_id
            .strip_prefix(crate::FIELD_PROMOTION_CHOICE_PREFIX)
            .and_then(|value| value.parse::<u32>().ok())
        {
            next.phase = GamePhase::Playing;
            let kind = match option_index {
                0 => PieceKind::Queen,
                1 => PieceKind::Rook,
                2 => PieceKind::Bishop,
                3 => PieceKind::Knight,
                _ => unreachable!("field promotion exposes four choices"),
            };
            if let Some((square, mut pawn)) = next.board.find_by_id(piece_id) {
                pawn.kind = kind;
                next.board.set(square, Some(pawn));
                next.card_runtime.field_promoted_piece_ids.insert(piece_id);
            }
            let mut events = vec![
                Event::ActionApplied { actor, action },
                Event::ChoiceResolved {
                    choice_id,
                    owner: actor,
                    option_index,
                },
            ];
            self.finish_turn(&mut next, actor, &mut events)?;
            return Ok(Transition {
                state: next,
                events,
            });
        }
        if choice.handler_id == middle::TROLLEY_CHOICE_HANDLER {
            next.phase = GamePhase::Playing;
            let mut events = vec![
                Event::ActionApplied { actor, action },
                Event::ChoiceResolved {
                    choice_id,
                    owner: actor,
                    option_index,
                },
            ];
            self.resolve_middle_trolley_choice(&mut next, usize::from(option_index), &mut events)?;
            if next.status != GameStatus::Ongoing {
                self.push_game_ended(&next, &mut events);
            } else {
                self.resolve_no_action_loss(&mut next, actor, &mut events);
            }
            return Ok(Transition {
                state: next,
                events,
            });
        }
        let handler = self.choices.get(&choice.handler_id).ok_or_else(|| {
            EngineError::Invariant(format!(
                "pending choice handler {:?} is not registered",
                choice.handler_id
            ))
        })?;
        let commands = handler.resolve(&choice, option_index, &next);
        next.phase = GamePhase::Playing;

        let mut events = vec![
            Event::ActionApplied { actor, action },
            Event::ChoiceResolved {
                choice_id,
                owner: actor,
                option_index,
            },
        ];
        self.apply_effect_commands(
            &mut next,
            &mut events,
            choice.source_card_instance,
            commands,
        )?;
        self.resolve_effect_queue_from(&mut next, &mut events, 0)?;
        self.resolve_missing_king(&mut next, &mut events);

        if next.status != GameStatus::Ongoing {
            self.push_game_ended(&next, &mut events);
        } else if matches!(next.phase, GamePhase::Playing) {
            if choice.continuation == ChoiceContinuation::FinishTurn
                || next.take_turn_end_request()
                || next.actions_remaining == 0
            {
                self.finish_turn(&mut next, actor, &mut events)?;
            } else {
                self.resolve_no_action_loss(&mut next, actor, &mut events);
            }
        }

        Ok(Transition {
            state: next,
            events,
        })
    }

    fn choose_opening_auto_action(
        &self,
        state: &mut GameState,
        instance_id: u32,
        definition_id: &str,
        events: &mut Vec<Event>,
    ) -> Option<Action> {
        let mut actions = Vec::new();
        self.generate_card_actions_with_opening_auto(state, &mut actions, true);
        let candidates = actions
            .into_iter()
            .filter(|action| {
                matches!(action, Action::UseCard { instance_id: candidate, .. } if *candidate == instance_id)
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }
        if matches!(
            candidates.as_slice(),
            [Action::UseCard {
                target: ActionTarget::None,
                ..
            }]
        ) {
            return candidates.into_iter().next();
        }
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        events.push(Event::RandomSample {
            purpose: format!("opening-auto-target:{definition_id}"),
            roll_53: roll,
        });
        index.map(|index| candidates[index].clone())
    }

    fn apply_opening_auto_action(
        &self,
        state: &GameState,
        action: Action,
    ) -> Result<Transition, EngineError> {
        let Action::UseCard {
            instance_id,
            target,
        } = action.clone()
        else {
            return Err(EngineError::Invariant(
                "opening auto action must use a card".to_owned(),
            ));
        };
        let actions_remaining = state.actions_remaining;
        let mut ready = state.clone();
        ready
            .card_runtime
            .opening_auto_card_ids
            .remove(&instance_id);
        ready.actions_remaining = ready.actions_remaining.max(1);
        let mut transition = self.apply_active_card(&ready, action, instance_id, target)?;
        transition.state.actions_remaining = actions_remaining;
        Ok(transition)
    }

    fn opening_move_rollback_state(
        &self,
        after_move: &GameState,
        before_move: &GameState,
    ) -> GameState {
        let pending = after_move.card_runtime.opening_auto_card_ids.clone();
        let mut rollback = after_move.clone();
        rollback.board = before_move.board.clone();
        rollback.en_passant = before_move.en_passant;
        rollback.card_runtime = before_move.card_runtime.clone();
        rollback.card_runtime.opening_auto_card_ids = pending;
        rollback
    }

    fn resolve_opening_cards_after_first_move(
        &self,
        state: &mut GameState,
        before_move: &GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) -> Result<bool, EngineError> {
        if state.turns_taken[actor.index()] != 0 {
            return Ok(false);
        }
        let pending = state
            .cards
            .iter()
            .filter(|card| {
                card.owner == actor
                    && card.charges > 0
                    && state
                        .card_runtime
                        .opening_auto_card_ids
                        .contains(&card.instance_id)
            })
            .map(|card| (card.instance_id, card.definition_id.clone()))
            .collect::<Vec<_>>();
        let mut resolved_count = 0_usize;
        let mut move_rolled_back = false;

        for (instance_id, definition_id) in pending {
            let action =
                self.choose_opening_auto_action(state, instance_id, &definition_id, events);
            if let Some(action) = action {
                let transition = self.apply_opening_auto_action(state, action)?;
                *state = transition.state;
                events.extend(transition.events);
                resolved_count += 1;
                continue;
            }

            if resolved_count == 0 && !move_rolled_back {
                let mut rollback = self.opening_move_rollback_state(state, before_move);
                if let Some(action) = self.choose_opening_auto_action(
                    &mut rollback,
                    instance_id,
                    &definition_id,
                    events,
                ) {
                    let transition = self.apply_opening_auto_action(&rollback, action)?;
                    *state = transition.state;
                    events.push(Event::EffectApplied {
                        source_card_instance: instance_id,
                        command: format!("opening-auto-move-rollback:{definition_id}"),
                    });
                    events.extend(transition.events);
                    resolved_count += 1;
                    move_rolled_back = true;
                    continue;
                }
            }

            state
                .card_runtime
                .opening_auto_card_ids
                .remove(&instance_id);
            events.push(Event::EffectApplied {
                source_card_instance: instance_id,
                command: format!("opening-auto-failed:{definition_id}"),
            });
        }
        Ok(move_rolled_back)
    }

    fn apply_move(
        &self,
        state: &GameState,
        action: Action,
        movement: MoveAction,
    ) -> Result<Transition, EngineError> {
        let mut next = state.clone();
        let actor = next.turn;
        let replay_before_board = next.board.clone();
        let replay_before_en_passant = next.en_passant;
        let mut replay_before_runtime = next.card_runtime.clone();
        replay_before_runtime.move_replays = [None, None];
        let mut events = vec![Event::ActionApplied {
            actor,
            action: action.clone(),
        }];
        let forced_continuation = match &next.phase {
            GamePhase::ForcedMove { continuation, .. } => Some(*continuation),
            _ => None,
        };
        if forced_continuation.is_none() {
            if next.actions_remaining == 0 {
                return Err(EngineError::Invariant(
                    "legal move has no remaining action budget".to_owned(),
                ));
            }
            next.actions_remaining -= 1;
        }
        // A new forced move emitted while resolving this move must be distinguishable from the
        // forced move that is currently being fulfilled.
        next.phase = GamePhase::Playing;
        if movement.from != movement.to {
            let memory_piece = next.move_source_piece(movement.from);
            let memory_source = memory_piece
                .map(|piece| self.piece_ability_kind(&next, piece))
                .unwrap_or(PieceKind::Pawn);
            let memory_effective = memory_piece
                .map(|piece| piece.kind)
                .unwrap_or(PieceKind::Pawn);
            // A Parrot relocating keeps the previous memory (live JS 4168-4177).
            if memory_source != PieceKind::Parrot {
                next.card_runtime.parrot_memory[actor.index()] = Some(ParrotMemoryStatus {
                    source: memory_source,
                    effective: memory_effective,
                });
            }
        }
        let mut moving_piece = next
            .move_source_piece(movement.from)
            .ok_or_else(|| EngineError::Invariant("legal move has no source piece".to_owned()))?;
        let moved_ability_kind = self.piece_ability_kind(&next, moving_piece);
        let moved_as_kind = moving_piece.kind;
        if moved_ability_kind == PieceKind::SiegeRam {
            let file_delta = movement.to.file() as i8 - movement.from.file() as i8;
            let rank_delta = movement.to.rank() as i8 - movement.from.rank() as i8;
            if file_delta.abs() + rank_delta.abs() == 2 {
                let middle = movement
                    .from
                    .offset(file_delta.signum(), rank_delta.signum())
                    .expect("siege ram midpoint");
                if let Some(victim) = next.board.get(middle) {
                    let remover = if victim.owner == actor {
                        actor.opponent()
                    } else {
                        actor
                    };
                    self.remove_piece_for_card(
                        &mut next,
                        middle,
                        victim,
                        remover,
                        SIEGE_RAM,
                        &mut events,
                    );
                    if victim.owner != actor {
                        moving_piece.capture_count = moving_piece.capture_count.saturating_add(1);
                    }
                }
            }
        }
        let castling_rook = self.standard_castling_rook_plan(&next, moving_piece, movement);
        let idol_source = if moved_ability_kind == PieceKind::Idol {
            None
        } else {
            self.available_adjacent_idol(&next, movement.from, moving_piece.owner)
        };
        let destination_entry = self.board_piece_entry_at(&next, movement.to);
        let mut destination_piece = destination_entry.map(|(_, piece)| piece);
        let mut en_passant_capture = (moved_ability_kind == PieceKind::Pawn
            && destination_piece.is_none()
            && movement.from.file() != movement.to.file())
        .then_some(next.en_passant)
        .flatten()
        .filter(|en_passant| {
            en_passant.target == movement.to
                && en_passant.pawn_owner != actor
                && en_passant.captured.rank() == movement.from.rank()
                && (en_passant.captured.file() == movement.to.file()
                    || next.card_runtime.long_en_passant[actor.index()])
        })
        .and_then(|en_passant| {
            next.board
                .get(en_passant.captured)
                .filter(|piece| {
                    piece.kind == PieceKind::Pawn && piece.owner == en_passant.pawn_owner
                })
                .map(|piece| (en_passant.captured, piece))
        });
        if en_passant_capture.is_none()
            && moved_ability_kind == PieceKind::Pawn
            && destination_piece.is_none()
            && next.card_runtime.en_passant_frenzy[actor.index()]
            && movement.from.file() != movement.to.file()
            && let Some(adjacent) = Square::new(movement.to.file(), movement.from.rank())
            && let Some(target) = next.board.get(adjacent)
            && self.can_capture(&next, moving_piece, target)
        {
            en_passant_capture = Some((adjacent, target));
        }
        let mut capture_square = en_passant_capture
            .map(|(square, _)| square)
            .or_else(|| destination_entry.map(|(square, _)| square))
            .unwrap_or(movement.to);
        let mut target_piece = destination_piece.or(en_passant_capture.map(|(_, piece)| piece));
        if let Some(target) = target_piece
            && target.owner != actor
            && (next.card_runtime.quantum_positions.contains_key(&target.id)
                || next
                    .card_runtime
                    .quantum_positions
                    .values()
                    .any(|pos| *pos == capture_square))
        {
            let (roll, choice) = next.rng_mut().web_choice_index(2);
            events.push(Event::RandomSample {
                purpose: "quantum-collapse".to_owned(),
                roll_53: roll,
            });
            let shadow_pos = next.card_runtime.quantum_positions.remove(&target.id);
            if choice == Some(1) {
                if let Some(shadow_pos) = shadow_pos
                    && next
                        .board
                        .get(capture_square)
                        .is_some_and(|p| p.id == target.id)
                {
                    next.board.set(capture_square, None);
                    next.board.set(shadow_pos, Some(target));
                }
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("quantum-miss:{}:{}", target.id, capture_square),
                });
                destination_piece = None;
                target_piece = None;
                capture_square = movement.to;
            }
        }

        let irreversible = moved_ability_kind == PieceKind::Pawn || target_piece.is_some();
        let mut moving_survived = true;

        let substitution_swap = next.card_runtime.substitution[actor.index()]
            && destination_piece
                .is_some_and(|target| target.owner != actor && target.kind == moving_piece.kind);
        let switcheroo_swap = next.card_runtime.switcheroo[actor.index()]
            && self.is_royal_piece(&next, moving_piece)
            && destination_piece
                .is_some_and(|target| target.owner == actor && target.kind == PieceKind::Pawn);
        let relay_swap = next.card_runtime.relay[actor.index()]
            && !self.piece_has_ability(&next, moving_piece, PieceKind::Slime)
            && (movement.from.file() == movement.to.file()
                || movement.from.rank() == movement.to.rank())
            && destination_piece.is_some_and(|target| {
                target.owner == actor && !self.piece_has_ability(&next, target, PieceKind::Slime)
            });
        if substitution_swap || switcheroo_swap || relay_swap {
            let swapped = destination_piece.expect("swap destination was checked");
            next.board.set(movement.from, Some(swapped));
            next.board.set(movement.to, Some(moving_piece));
            next.card_runtime.substitution[actor.index()] = false;
            next.card_runtime.switcheroo[actor.index()] = false;
            next.card_runtime.relay[actor.index()] = false;
            next.card_runtime.moved_piece_ids.insert(moving_piece.id);
            next.card_runtime.moved_piece_ids.insert(swapped.id);
            next.en_passant = None;
            next.ply = next.ply.saturating_add(1);
            next.quiet_ply = next.quiet_ply.saturating_add(1);
            self.note_middle_piece_moved(&mut next, moving_piece);
            self.resolve_after_counted_move(&mut next, &mut events);
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!(
                    "{}:{}:{}",
                    if substitution_swap {
                        "substitution"
                    } else if relay_swap {
                        "relay"
                    } else {
                        "switcheroo"
                    },
                    movement.from,
                    movement.to
                ),
            });
            self.record_completed_piece_move(
                &mut next,
                actor,
                moving_piece,
                &replay_before_board,
                replay_before_en_passant,
                &replay_before_runtime,
                &mut events,
            );
            self.finish_turn(&mut next, actor, &mut events)?;
            return Ok(Transition {
                state: next,
                events,
            });
        }

        if let Some(target) = target_piece
            && target.owner != actor
            && next.card_runtime.evasion_piece_ids.contains(&target.id)
        {
            let candidates = self.middle_evasion_destinations(&next, capture_square, movement.to);
            let (roll, selected) = next.rng_mut().web_choice_index(candidates.len());
            events.push(Event::RandomSample {
                purpose: "evasion-capture".to_owned(),
                roll_53: roll,
            });
            if let Some(selected) = selected {
                let destination = candidates[selected];
                next.board.set(capture_square, None);
                next.board.set(destination, Some(target));
                next.card_runtime.evasion_piece_ids.remove(&target.id);
                next.card_runtime.moved_piece_ids.insert(target.id);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("evasion:{}:{}->{}", target.id, capture_square, destination),
                });
                destination_piece = None;
                target_piece = None;
                capture_square = movement.to;
            }
        }

        if let Some(counter) = target_piece
            && counter.owner != actor
            && next.card_runtime.mistake_card[actor.index()]
        {
            let (roll, reversed) = next.rng_mut().web_probability(1, 2);
            events.push(Event::RandomSample {
                purpose: "mistake-card".to_owned(),
                roll_53: roll,
            });
            if reversed {
                next.board.set(movement.from, None);
                next.board.set(capture_square, None);
                let mut counter = counter;
                counter.capture_count = counter.capture_count.saturating_add(1);
                next.board.set(movement.from, Some(counter));
                self.record_new_capture_reactions(&mut next, moving_piece);
                self.clear_removed_piece_statuses(&mut next, moving_piece.id);
                events.push(Event::PieceCaptured {
                    actor: counter.owner,
                    piece: moving_piece,
                    at: movement.from,
                });
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("mistake-reverse:{}:{}", counter.id, moving_piece.id),
                });
                next.ply = next.ply.saturating_add(1);
                next.quiet_ply = 0;
                next.mark_deathmatch_progress();
                self.resolve_after_counted_move(&mut next, &mut events);
                self.record_completed_piece_move(
                    &mut next,
                    actor,
                    moving_piece,
                    &replay_before_board,
                    replay_before_en_passant,
                    &replay_before_runtime,
                    &mut events,
                );
                self.resolve_missing_king(&mut next, &mut events);
                if next.status == GameStatus::Ongoing {
                    self.finish_turn(&mut next, actor, &mut events)?;
                } else {
                    self.push_game_ended(&next, &mut events);
                }
                return Ok(Transition {
                    state: next,
                    events,
                });
            }
        }
        if let Some(bear) = target_piece
            && bear.owner != actor
            && next
                .card_runtime
                .bear_retaliations
                .get(&bear.id)
                .is_some_and(|remaining| *remaining > 0)
        {
            if let Some(remaining) = next.card_runtime.bear_retaliations.get_mut(&bear.id) {
                *remaining = remaining.saturating_sub(1);
            }
            next.board.set(movement.from, None);
            self.record_new_capture_reactions(&mut next, moving_piece);
            self.clear_removed_piece_statuses(&mut next, moving_piece.id);
            events.push(Event::PieceCaptured {
                actor: bear.owner,
                piece: moving_piece,
                at: movement.from,
            });
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("bear-retaliation:{}:{}", bear.id, moving_piece.id),
            });
            next.ply = next.ply.saturating_add(1);
            next.quiet_ply = 0;
            next.mark_deathmatch_progress();
            self.resolve_after_counted_move(&mut next, &mut events);
            self.record_completed_piece_move(
                &mut next,
                actor,
                moving_piece,
                &replay_before_board,
                replay_before_en_passant,
                &replay_before_runtime,
                &mut events,
            );
            self.resolve_missing_king(&mut next, &mut events);
            if next.status == GameStatus::Ongoing {
                self.finish_turn(&mut next, actor, &mut events)?;
            } else {
                self.push_game_ended(&next, &mut events);
            }
            return Ok(Transition {
                state: next,
                events,
            });
        }
        let middle_capture = target_piece.map(|target| self.middle_capture_context(&next, target));

        if matches!(moving_piece.kind, PieceKind::BigRook | PieceKind::Colossus) {
            let targets = self.large_landing_targets(&next, moving_piece, movement.to);
            for (square, target) in targets {
                let remover = if target.owner == actor {
                    actor.opponent()
                } else {
                    actor
                };
                self.remove_piece_for_card(
                    &mut next,
                    square,
                    target,
                    remover,
                    "large-piece-landing",
                    &mut events,
                );
            }
            next.board.set(movement.from, None);
            next.board.set(movement.to, Some(moving_piece));
            if let Some(status) = next.card_runtime.large_pieces.get_mut(&moving_piece.id) {
                status.anchor = movement.to;
            }
            next.card_runtime.moved_piece_ids.insert(moving_piece.id);
            next.en_passant = None;
            next.ply = next.ply.saturating_add(1);
            next.quiet_ply = if target_piece.is_some() {
                0
            } else {
                next.quiet_ply.saturating_add(1)
            };
            next.mark_deathmatch_progress();
            self.note_middle_piece_moved(&mut next, moving_piece);
            self.resolve_after_counted_move(&mut next, &mut events);
            self.record_completed_piece_move(
                &mut next,
                actor,
                moving_piece,
                &replay_before_board,
                replay_before_en_passant,
                &replay_before_runtime,
                &mut events,
            );
            self.resolve_missing_king(&mut next, &mut events);
            if next.status == GameStatus::Ongoing {
                self.finish_turn(&mut next, actor, &mut events)?;
            } else {
                self.push_game_ended(&next, &mut events);
            }
            return Ok(Transition {
                state: next,
                events,
            });
        }

        if moving_piece.kind == PieceKind::Log {
            let file_delta = movement.to.file() as i8 - movement.from.file() as i8;
            let rank_delta = movement.to.rank() as i8 - movement.from.rank() as i8;
            next.card_runtime.log_statuses.insert(
                moving_piece.id,
                LogStatus {
                    file_delta,
                    rank_delta,
                    roll_after_turn: next.turns_taken[actor.index()].saturating_add(1),
                },
            );
            next.card_runtime.moved_piece_ids.insert(moving_piece.id);
            next.ply = next.ply.saturating_add(1);
            next.quiet_ply = next.quiet_ply.saturating_add(1);
            self.note_middle_piece_moved(&mut next, moving_piece);
            self.resolve_after_counted_move(&mut next, &mut events);
            self.record_completed_piece_move(
                &mut next,
                actor,
                moving_piece,
                &replay_before_board,
                replay_before_en_passant,
                &replay_before_runtime,
                &mut events,
            );
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!(
                    "log-direction:{}:{file_delta}:{rank_delta}",
                    moving_piece.id
                ),
            });
            self.finish_turn(&mut next, actor, &mut events)?;
            return Ok(Transition {
                state: next,
                events,
            });
        }

        if moved_ability_kind == PieceKind::Dragon
            && destination_piece.is_some_and(|piece| piece.owner == actor)
        {
            let swapped = destination_piece.expect("checked friendly destination");
            next.board.set(movement.from, Some(swapped));
            next.board.set(movement.to, Some(moving_piece));
            next.card_runtime.moved_piece_ids.insert(moving_piece.id);
            next.card_runtime.moved_piece_ids.insert(swapped.id);
            next.en_passant = None;
            next.ply = next.ply.saturating_add(1);
            next.quiet_ply = next.quiet_ply.saturating_add(1);
            self.note_middle_piece_moved(&mut next, moving_piece);
            self.resolve_after_counted_move(&mut next, &mut events);
            self.record_completed_piece_move(
                &mut next,
                actor,
                moving_piece,
                &replay_before_board,
                replay_before_en_passant,
                &replay_before_runtime,
                &mut events,
            );
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("dragon-swap:{}:{}", movement.from, movement.to),
            });
            self.apply_idol_encore(&mut next, idol_source, moving_piece.id);
            if next.actions_remaining == 0 {
                self.finish_turn(&mut next, actor, &mut events)?;
            } else {
                self.resolve_no_action_loss(&mut next, actor, &mut events);
            }
            return Ok(Transition {
                state: next,
                events,
            });
        }

        if let Some(target) = target_piece
            && target.owner != actor
            && moved_ability_kind != PieceKind::Missionary
            && (next.card_runtime.shielded_piece_ids.remove(&target.id)
                || next.card_runtime.protected_piece_ids.contains(&target.id))
        {
            next.ply = next.ply.saturating_add(1);
            next.quiet_ply = next.quiet_ply.saturating_add(1);
            self.resolve_after_counted_move(&mut next, &mut events);
            self.record_completed_piece_move(
                &mut next,
                actor,
                moving_piece,
                &replay_before_board,
                replay_before_en_passant,
                &replay_before_runtime,
                &mut events,
            );
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("shield-broken:{}", target.id),
            });
            self.finish_turn(&mut next, actor, &mut events)?;
            return Ok(Transition {
                state: next,
                events,
            });
        }

        if let Some(target) = target_piece
            && let Some(status) = next.card_runtime.large_pieces.get_mut(&target.id)
        {
            status.hp = status.hp.saturating_sub(1);
            let defeated = status.hp == 0;
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("hp-hit:{}:{}/{}", target.id, status.hp, status.max_hp),
            });
            if defeated {
                next.board.set(capture_square, None);
                next.card_runtime.large_pieces.remove(&target.id);
                self.record_new_capture_reactions(&mut next, target);
                self.clear_removed_piece_statuses(&mut next, target.id);
                events.push(Event::PieceCaptured {
                    actor,
                    piece: target,
                    at: capture_square,
                });
                self.resolve_calling_card_capture(
                    &mut next,
                    target,
                    Some(moving_piece),
                    &mut events,
                );
            }
            next.ply = next.ply.saturating_add(1);
            next.quiet_ply = if defeated {
                0
            } else {
                next.quiet_ply.saturating_add(1)
            };
            next.en_passant = None;
            self.resolve_after_counted_move(&mut next, &mut events);
            self.record_completed_piece_move(
                &mut next,
                actor,
                moving_piece,
                &replay_before_board,
                replay_before_en_passant,
                &replay_before_runtime,
                &mut events,
            );
            self.resolve_missing_king(&mut next, &mut events);
            if next.status == GameStatus::Ongoing {
                self.finish_turn(&mut next, actor, &mut events)?;
            } else {
                self.push_game_ended(&next, &mut events);
            }
            return Ok(Transition {
                state: next,
                events,
            });
        }

        if moved_ability_kind == PieceKind::Missionary
            && let Some(mut converted) = target_piece
            && converted.owner != actor
        {
            let former_owner = converted.owner;
            converted.owner = actor;
            next.board.set(movement.to, Some(converted));
            next.card_runtime.moved_piece_ids.insert(moving_piece.id);
            next.card_runtime.moved_piece_ids.insert(converted.id);
            next.en_passant = None;
            next.ply = next.ply.saturating_add(1);
            next.quiet_ply = 0;
            next.mark_deathmatch_progress();
            self.note_middle_piece_moved(&mut next, moving_piece);
            self.resolve_after_counted_move(&mut next, &mut events);
            self.record_completed_piece_move(
                &mut next,
                actor,
                moving_piece,
                &replay_before_board,
                replay_before_en_passant,
                &replay_before_runtime,
                &mut events,
            );
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("missionary-convert:{}:{}", converted.id, movement.to),
            });
            if self.is_royal_piece(&next, converted) {
                next.status = GameStatus::Won {
                    winner: actor,
                    reason: WinReason::CardEffect("missionary-conversion".to_owned()),
                };
                self.push_game_ended(&next, &mut events);
            } else {
                self.break_initiative_by_capture(&mut next, former_owner, actor);
                self.apply_idol_encore(&mut next, idol_source, moving_piece.id);
                if next.actions_remaining == 0 {
                    self.finish_turn(&mut next, actor, &mut events)?;
                } else {
                    self.resolve_no_action_loss(&mut next, actor, &mut events);
                }
            }
            return Ok(Transition {
                state: next,
                events,
            });
        }

        let checker_jump = if matches!(
            moved_ability_kind,
            PieceKind::Checker | PieceKind::CheckerKing
        ) {
            let mut checker = moving_piece;
            checker.kind = moved_ability_kind;
            self.checker_jump_capture(&next, movement.from, movement.to, checker)
        } else {
            None
        };
        if let Some((mid, captured)) = checker_jump {
            let captured_square = next
                .board
                .find_by_id(captured.id)
                .map(|(square, _)| square)
                .unwrap_or(mid);
            next.board.set(movement.from, None);
            let mut checker = moving_piece;
            let mut jumped_attack = false;
            let mut defeated = false;
            if next.card_runtime.shielded_piece_ids.remove(&captured.id) {
                jumped_attack = true;
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("shield-broken:{}", captured.id),
                });
            } else if !next.card_runtime.protected_piece_ids.contains(&captured.id) {
                jumped_attack = true;
                if let Some(status) = next.card_runtime.large_pieces.get_mut(&captured.id) {
                    status.hp = status.hp.saturating_sub(1);
                    defeated = status.hp == 0;
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("hp-hit:{}:{}/{}", captured.id, status.hp, status.max_hp),
                    });
                } else {
                    defeated = true;
                }
                if defeated {
                    next.board.set(captured_square, None);
                    self.record_new_capture_reactions(&mut next, captured);
                    self.clear_removed_piece_statuses(&mut next, captured.id);
                    checker.capture_count = checker.capture_count.saturating_add(1);
                    events.push(Event::PieceCaptured {
                        actor,
                        piece: captured,
                        at: captured_square,
                    });
                    self.resolve_calling_card_capture(
                        &mut next,
                        captured,
                        Some(checker),
                        &mut events,
                    );
                }
            }
            if checker.kind == PieceKind::Checker
                && ((checker.owner == Player::White && movement.to.rank() == 7)
                    || (checker.owner == Player::Black && movement.to.rank() == 0))
            {
                checker.kind = PieceKind::CheckerKing;
            }
            next.board.set(movement.to, Some(checker));
            next.card_runtime.moved_piece_ids.insert(checker.id);
            next.en_passant = None;
            next.ply = next.ply.saturating_add(1);
            next.quiet_ply = if defeated {
                0
            } else {
                next.quiet_ply.saturating_add(1)
            };
            next.mark_deathmatch_progress();
            self.note_middle_piece_moved(&mut next, moving_piece);
            self.resolve_after_counted_move(&mut next, &mut events);
            self.record_completed_piece_move(
                &mut next,
                actor,
                moving_piece,
                &replay_before_board,
                replay_before_en_passant,
                &replay_before_runtime,
                &mut events,
            );
            self.resolve_missing_king(&mut next, &mut events);
            if next.status != GameStatus::Ongoing {
                self.push_game_ended(&next, &mut events);
            } else if jumped_attack {
                let mut followups = Vec::new();
                self.generate_checker_moves(&next, movement.to, checker, &mut followups);
                let has_capture = followups.iter().any(|action| {
                    matches!(action, Action::Move(movement) if movement.from.file().abs_diff(movement.to.file()) == 2)
                });
                if has_capture {
                    next.phase = GamePhase::ForcedMove {
                        piece_id: checker.id,
                        source: ForcedMoveSource::Effect(CHECKER.to_owned()),
                        continuation: ForcedMoveContinuation::FinishTurn,
                    };
                    events.push(Event::ForcedMoveStarted {
                        player: actor,
                        piece_id: checker.id,
                        source: CHECKER.to_owned(),
                    });
                } else {
                    self.finish_turn(&mut next, actor, &mut events)?;
                }
            } else {
                self.finish_turn(&mut next, actor, &mut events)?;
            }
            return Ok(Transition {
                state: next,
                events,
            });
        }

        next.board.set(movement.from, None);
        if let Some((real_square, _)) = next.board.find_by_id(moving_piece.id) {
            next.board.set(real_square, None);
        }
        next.card_runtime.quantum_positions.remove(&moving_piece.id);
        let parry_attempt = target_piece.filter(|target| {
            target.owner != actor && next.card_runtime.parry_piece_ids.remove(&target.id)
        });
        let parry_triggered = if parry_attempt.is_some() {
            let (roll, triggered) = next.rng_mut().web_probability(2, 5);
            events.push(Event::RandomSample {
                purpose: "parry".to_owned(),
                roll_53: roll,
            });
            triggered
        } else {
            false
        };

        if let Some(captured) = target_piece {
            next.board.set(capture_square, None);
            if parry_triggered {
                self.record_new_capture_reactions(&mut next, moving_piece);
                self.clear_removed_piece_statuses(&mut next, moving_piece.id);
                let mut retaliator = captured;
                retaliator.capture_count = retaliator.capture_count.saturating_add(1);
                next.board.set(movement.from, Some(retaliator));
                self.break_initiative_by_capture(&mut next, moving_piece.owner, captured.owner);
                moving_survived = false;
                events.push(Event::PieceCaptured {
                    actor: captured.owner,
                    piece: moving_piece,
                    at: movement.to,
                });
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("parry:{}:{}->{}", captured.id, movement.to, movement.from),
                });
                self.grant_wizard_mana_for_loss(&mut next, moving_piece.owner);
                self.resolve_reaper_deaths(
                    &mut next,
                    &[(movement.to, moving_piece)],
                    captured.owner,
                    &mut events,
                );
            } else {
                self.record_new_capture_reactions(&mut next, captured);
                self.clear_removed_piece_statuses(&mut next, captured.id);
                let mut placed_piece = moving_piece;
                if let Some(promotion) = movement.promotion {
                    placed_piece.kind = promotion;
                }
                placed_piece.capture_count = placed_piece.capture_count.saturating_add(1);
                if moved_ability_kind == PieceKind::Squire {
                    placed_piece.kind = PieceKind::Knight;
                    next.card_runtime
                        .trickster_move_kinds
                        .remove(&moving_piece.id);
                }
                next.board.set(movement.to, Some(placed_piece));
                events.push(Event::PieceCaptured {
                    actor,
                    piece: captured,
                    at: capture_square,
                });
                self.break_initiative_by_capture(&mut next, captured.owner, actor);
                self.grant_wizard_mana_for_loss(&mut next, captured.owner);
                self.resolve_reaper_deaths(
                    &mut next,
                    &[(capture_square, captured)],
                    actor,
                    &mut events,
                );
                self.resolve_calling_card_capture(
                    &mut next,
                    captured,
                    Some(placed_piece),
                    &mut events,
                );
                if self.is_royal_piece(&next, moving_piece)
                    && matches!(captured.kind, PieceKind::Pawn | PieceKind::Fanatic)
                    && next.card_runtime.iron_monarch[actor.index()]
                {
                    next.actions_remaining = next.actions_remaining.saturating_add(1);
                }
                // Disassembly (live disassembleMovedQueen, guard order preserved): moved queen converts
                // to a bishop on the destination; a fresh rook spawns on the origin only when that square
                // is empty and unblocked (collapsed, black hole, crown ground, installation, quantum).
                if moving_survived
                    && moved_as_kind == PieceKind::Queen
                    && placed_piece.kind == PieceKind::Queen
                    && next.card_runtime.disassembly[actor.index()]
                    && !self.is_royal_piece(&next, placed_piece)
                    && movement.from != movement.to
                {
                    let mut bishop = placed_piece;
                    bishop.kind = PieceKind::Bishop;
                    next.board.set(movement.to, Some(bishop));
                    if next.board.get(movement.from).is_none() {
                        let mut rook = placed_piece;
                        rook.kind = PieceKind::Rook;
                        rook.id = next.allocate_piece_id();
                        next.card_runtime.moved_piece_ids.insert(rook.id);
                        next.board.set(movement.from, Some(rook));
                    }
                }
            }
        } else {
            let mut placed_piece = moving_piece;
            if let Some(promotion) = movement.promotion {
                placed_piece.kind = promotion;
            }
            next.board.set(movement.to, Some(placed_piece));
            // Disassembly also applies to quiet queen moves (live disassembleMovedQueen).
            if moving_survived
                && next.card_runtime.disassembly[actor.index()]
                && placed_piece.kind == PieceKind::Queen
                && next.board.get(movement.from).is_none()
            {
                let mut bishop = placed_piece;
                bishop.kind = PieceKind::Bishop;
                next.board.set(movement.to, Some(bishop));
                let mut rook = placed_piece;
                rook.kind = PieceKind::Rook;
                rook.id = next.allocate_piece_id();
                next.card_runtime.moved_piece_ids.insert(rook.id);
                next.board.set(movement.from, Some(rook));
            }
        }
        if moving_survived && movement.promotion.is_some() {
            self.clear_promotion_inherited_traits(&mut next, moving_piece.id);
        }
        // Outpost protection ends when the protected piece itself moves.
        next.card_runtime
            .outpost_protected_piece_ids
            .remove(&moving_piece.id);
        if next.card_runtime.infiltration[actor.index()] {
            let opponent_home_rank = match actor {
                Player::White => 7,
                Player::Black => 0,
            };
            if movement.to.rank() == opponent_home_rank {
                let has_neighbor = [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]
                .iter()
                .any(|(df, dr)| {
                    movement
                        .to
                        .offset(*df, *dr)
                        .and_then(|sq| next.board.get(sq))
                        .is_some()
                });
                if !has_neighbor {
                    next.card_runtime
                        .submerged_piece_ids
                        .insert(moving_piece.id);
                }
            }
        }
        if moving_survived && moved_ability_kind == PieceKind::Thief {
            next.card_runtime
                .submerged_piece_ids
                .insert(moving_piece.id);
        }
        if let Some((rook_from, rook_to, rook)) = castling_rook {
            next.board.set(rook_from, None);
            next.board.set(rook_to, Some(rook));
            next.card_runtime.moved_piece_ids.insert(rook.id);
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!(
                    "castle:{}:{}:{}:{}",
                    movement.from, movement.to, rook_from, rook_to
                ),
            });
        }
        if moving_survived
            && matches!(
                moved_ability_kind,
                PieceKind::Pawn | PieceKind::StandardBearer | PieceKind::King | PieceKind::Rook
            )
        {
            next.card_runtime.moved_piece_ids.insert(moving_piece.id);
        }
        if moving_survived
            && moved_ability_kind == PieceKind::Checker
            && ((actor == Player::White && movement.to.rank() == 7)
                || (actor == Player::Black && movement.to.rank() == 0))
            && let Some(mut checker) = next.board.get(movement.to)
        {
            checker.kind = PieceKind::CheckerKing;
            next.board.set(movement.to, Some(checker));
        }
        if moving_survived && moved_ability_kind == PieceKind::Recruiter {
            let pawn = Piece::new(actor, PieceKind::Pawn).with_id(next.allocate_piece_id());
            next.board.set(movement.from, Some(pawn));
        }
        if moving_survived && moved_ability_kind == PieceKind::Slime {
            let slime = Piece::new(actor, PieceKind::Slime).with_id(next.allocate_piece_id());
            next.board.set(movement.from, Some(slime));
            self.register_spawned_piece(&mut next, movement.from, slime, true);
        }
        if moving_survived && moved_ability_kind == PieceKind::Herald {
            next.card_runtime
                .herald_jump_locked_until
                .remove(&moving_piece.id);
        }
        if moving_survived && moving_piece.kind == PieceKind::Windmill {
            let mode = next
                .card_runtime
                .windmill_modes
                .entry(moving_piece.id)
                .or_insert(WindmillMode::Bishop);
            *mode = match *mode {
                WindmillMode::Bishop => WindmillMode::Rook,
                WindmillMode::Rook => WindmillMode::Bishop,
            };
        }
        if moving_survived {
            moving_survived = self.resolve_middle_after_move(
                &mut next,
                moving_piece,
                movement,
                target_piece.map(|target| (capture_square, target)),
                middle_capture,
                &mut events,
            );
        }
        if moving_survived {
            self.apply_idol_encore(&mut next, idol_source, moving_piece.id);
            if moving_piece.kind == PieceKind::Pawn
                && next.card_runtime.resolve[actor.index()]
                && next.card_runtime.resolve_ready[actor.index()]
            {
                next.card_runtime.resolve_ready[actor.index()] = false;
                next.actions_remaining = next.actions_remaining.saturating_add(1);
            }
        }
        if moving_survived {
            self.resolve_end_after_move(
                &mut next,
                moving_piece,
                movement,
                target_piece.map(|target| (capture_square, target)),
                &mut events,
            );
            self.resolve_rule_after_move(
                &mut next,
                moving_piece,
                movement,
                target_piece.map(|target| (capture_square, target)),
                &mut events,
            );
        }
        if moving_survived
            && moving_piece.kind == PieceKind::Pawn
            && target_piece.is_some()
            && next.card_runtime.field_promotion[actor.index()]
            && !next
                .card_runtime
                .field_promoted_piece_ids
                .contains(&moving_piece.id)
            && let Some((_, pawn)) = next.board.find_by_id(moving_piece.id)
            && pawn.kind == PieceKind::Pawn
            && pawn.capture_count >= 2
        {
            let choice_id = next.allocate_choice_id();
            next.phase = GamePhase::AwaitingChoice(PendingChoice {
                id: choice_id,
                owner: actor,
                source_card_instance: 0,
                handler_id: format!(
                    "{}{}",
                    crate::FIELD_PROMOTION_CHOICE_PREFIX,
                    moving_piece.id
                ),
                options: vec![
                    "queen".to_owned(),
                    "rook".to_owned(),
                    "bishop".to_owned(),
                    "knight".to_owned(),
                ],
                continuation: ChoiceContinuation::FinishTurn,
            });
        }

        let previous_en_passant = next.en_passant;
        next.en_passant = if moving_survived
            && moved_ability_kind == PieceKind::Pawn
            && movement.from.file() == movement.to.file()
            && movement.from.rank().abs_diff(movement.to.rank()) == 2
        {
            Some(EnPassantState {
                target: Square::new(
                    movement.from.file(),
                    (movement.from.rank() + movement.to.rank()) / 2,
                )
                .expect("midpoint of a legal pawn double-step is on board"),
                captured: movement.to,
                pawn_owner: actor,
            })
        } else {
            previous_en_passant.filter(|en_passant| en_passant.pawn_owner == actor)
        };

        next.ply = next.ply.saturating_add(1);
        next.quiet_ply = if irreversible {
            0
        } else {
            next.quiet_ply.saturating_add(1)
        };
        if irreversible {
            next.mark_deathmatch_progress();
        }

        self.resolve_after_counted_move(&mut next, &mut events);
        self.record_completed_piece_move(
            &mut next,
            actor,
            moving_piece,
            &replay_before_board,
            replay_before_en_passant,
            &replay_before_runtime,
            &mut events,
        );

        self.resolve_effect_queue_from(&mut next, &mut events, 0)?;

        self.resolve_missing_king(&mut next, &mut events);
        self.resolve_herald_victory(&mut next);
        if next.status != GameStatus::Ongoing {
            self.push_game_ended(&next, &mut events);
            return Ok(Transition {
                state: next,
                events,
            });
        }
        if !matches!(next.phase, GamePhase::Playing) {
            return Ok(Transition {
                state: next,
                events,
            });
        }
        let opening_move_rolled_back =
            self.resolve_opening_cards_after_first_move(&mut next, state, actor, &mut events)?;
        self.resolve_missing_king(&mut next, &mut events);
        self.resolve_herald_victory(&mut next);
        if next.status != GameStatus::Ongoing {
            self.push_game_ended(&next, &mut events);
            return Ok(Transition {
                state: next,
                events,
            });
        }
        if opening_move_rolled_back {
            next.actions_remaining = 0;
            self.finish_turn(&mut next, actor, &mut events)?;
            return Ok(Transition {
                state: next,
                events,
            });
        }
        if next.take_turn_end_request() {
            self.finish_turn(&mut next, actor, &mut events)?;
            return Ok(Transition {
                state: next,
                events,
            });
        }
        if next.status == GameStatus::Ongoing
            && moving_survived
            && self.try_start_platform_extra_move(
                &mut next,
                movement.to,
                moving_piece.id,
                &mut events,
            )
        {
            return Ok(Transition {
                state: next,
                events,
            });
        }
        if next.status == GameStatus::Ongoing
            && moving_survived
            && self.try_start_reposition_extra_move(&mut next, moving_piece.id, &mut events)
        {
            return Ok(Transition {
                state: next,
                events,
            });
        }
        if forced_continuation == Some(ForcedMoveContinuation::FinishTurn)
            || next.actions_remaining == 0
        {
            self.finish_turn(&mut next, actor, &mut events)?;
        } else {
            self.resolve_no_action_loss(&mut next, actor, &mut events);
        }
        Ok(Transition {
            state: next,
            events,
        })
    }

    fn record_move_replay(
        &self,
        state: &mut GameState,
        actor: Player,
        before: &crate::Board,
        en_passant_before: Option<EnPassantState>,
        runtime_before: &CardRuntimeState,
    ) {
        if state.ruleset_id != "augment_chess_20260818" {
            state.card_runtime.move_replays[actor.index()] = None;
            return;
        }
        if runtime_before.free_move_no_capture[actor.index()] {
            state.card_runtime.move_replays[actor.index()] = None;
            return;
        }
        let cells = (0_u8..64)
            .filter_map(Square::from_index)
            .filter_map(|square| {
                let old = before.get(square);
                let new = state.board.get(square);
                (old != new).then_some(ReplayCell {
                    square,
                    before: old,
                    after: new,
                })
            })
            .collect::<Vec<_>>();
        let mut runtime_after = state.card_runtime.clone();
        runtime_after.move_replays = [None, None];
        if cells.is_empty() && *runtime_before == runtime_after {
            state.card_runtime.move_replays[actor.index()] = None;
        } else {
            state.card_runtime.move_replays[actor.index()] = Some(MoveReplayState {
                cells,
                en_passant_before,
                en_passant_after: state.en_passant,
                runtime_before: Box::new(runtime_before.clone()),
                runtime_after: Box::new(runtime_after),
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_completed_piece_move(
        &self,
        state: &mut GameState,
        actor: Player,
        moving_piece: Piece,
        before: &crate::Board,
        en_passant_before: Option<EnPassantState>,
        runtime_before: &CardRuntimeState,
        events: &mut Vec<Event>,
    ) {
        let moving_survives = state.board.find_by_id(moving_piece.id).is_some();
        let relocated = |piece_id| {
            before
                .find_by_id(piece_id)
                .zip(state.board.find_by_id(piece_id))
                .is_some_and(|((before_square, _), (after_square, _))| {
                    before_square != after_square
                })
        };
        let spent_locusts = runtime_before
            .locust_swarm_piece_ids
            .iter()
            .copied()
            .filter(|piece_id| relocated(*piece_id))
            .collect::<Vec<_>>();
        for piece_id in spent_locusts {
            state.card_runtime.locust_swarm_piece_ids.remove(&piece_id);
        }
        let moved_metals = state
            .card_runtime
            .metalized_piece_ids
            .iter()
            .copied()
            .filter(|piece_id| relocated(*piece_id))
            .collect::<Vec<_>>();
        for piece_id in moved_metals {
            state.card_runtime.metal_cooldowns.insert(piece_id, 4);
        }
        if moving_survives {
            let captures = before
                .iter()
                .filter(|(_, piece)| piece.owner != actor)
                .filter(|(_, piece)| {
                    state
                        .board
                        .find_by_id(piece.id)
                        .is_none_or(|(_, current)| current.owner == actor)
                })
                .count() as u8;
            if captures > 0 {
                state
                    .card_runtime
                    .manner_captured_last_piece_ids
                    .insert(moving_piece.id);
                if state.card_runtime.saturation_rule
                    || state
                        .card_runtime
                        .potion_saturation_piece_ids
                        .contains(&moving_piece.id)
                {
                    let count = state
                        .card_runtime
                        .saturation_capture_counts
                        .entry(moving_piece.id)
                        .or_default();
                    *count = count.saturating_add(captures).min(3);
                }
            } else {
                state
                    .card_runtime
                    .manner_captured_last_piece_ids
                    .remove(&moving_piece.id);
            }
        }
        // After-move abilities must observe the sampled trickster type before it rerolls.
        self.resolve_herald_victory(state);
        if moving_piece.kind == PieceKind::Trickster
            && state
                .board
                .find_by_id(moving_piece.id)
                .is_some_and(|(_, current)| current.kind == PieceKind::Trickster)
        {
            let kind = self.choose_trickster_kind(state, events);
            state
                .card_runtime
                .trickster_move_kinds
                .insert(moving_piece.id, kind);
        }
        self.advance_capture_flags(state, actor, false);
        self.record_move_replay(state, actor, before, en_passant_before, runtime_before);
    }

    fn apply_freeze(&self, state: &mut GameState, actor: Player, events: &mut Vec<Event>) {
        let mut candidates = web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, piece)| {
                piece.owner == actor.opponent()
                    && !self.is_royal_piece(state, *piece)
                    && !state.card_runtime.frozen.contains_key(&piece.id)
            })
            .collect::<Vec<_>>();
        for index in (1..candidates.len()).rev() {
            let (roll, selected) = state.rng_mut().web_choice_index(index + 1);
            events.push(Event::RandomSample {
                purpose: format!("freeze-shuffle:{index}"),
                roll_53: roll,
            });
            candidates.swap(index, selected.expect("non-empty shuffle prefix"));
        }
        for (_, piece) in candidates.into_iter().take(3) {
            state.card_runtime.frozen.insert(
                piece.id,
                FrozenStatus {
                    source: actor,
                    remaining_turns: 3,
                },
            );
        }
    }

    fn mark_fresh_no_capture(&self, state: &mut GameState, piece: Piece) {
        state.card_runtime.fresh_no_capture_until.insert(
            piece.id,
            state.turns_taken[piece.owner.index()].saturating_add(1),
        );
    }

    fn break_initiative_by_capture(
        &self,
        state: &mut GameState,
        target_owner: Player,
        attacker: Player,
    ) {
        if state.card_runtime.initiative[target_owner.index()]
            .is_some_and(|status| status.by == attacker)
        {
            state.card_runtime.initiative[target_owner.index()] = None;
        }
    }

    fn ensure_ruleset(&self, state: &GameState) -> Result<(), EngineError> {
        if state.ruleset_id != self.ruleset.id {
            Err(EngineError::RulesetMismatch {
                engine: self.ruleset.id.clone(),
                state: state.ruleset_id.clone(),
            })
        } else {
            Ok(())
        }
    }

    fn apply_effect_commands(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
        source_card_instance: u32,
        commands: Vec<EffectCommand>,
    ) -> Result<(), EngineError> {
        for command in commands {
            let emitted = self.reduce_effect(state, source_card_instance, &command)?;
            events.push(Event::EffectApplied {
                source_card_instance,
                command: format!("{command:?}"),
            });
            events.extend(emitted);
        }
        Ok(())
    }

    fn resolve_effect_queue_from(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
        start_index: usize,
    ) -> Result<(), EngineError> {
        let mut cursor = start_index;
        let mut processed = 0_usize;
        while cursor < events.len() {
            if processed >= MAX_EFFECT_EVENTS_PER_RESOLUTION {
                return Err(EngineError::EffectResolutionLimit(
                    MAX_EFFECT_EVENTS_PER_RESOLUTION,
                ));
            }
            processed += 1;
            let event = events[cursor].clone();
            cursor += 1;

            let mut cards = state.cards.clone();
            cards.sort_by_key(|card| (card.owner.index(), card.instance_id));
            for card in &cards {
                let Some(handler) = self.effects.get(&card.definition_id) else {
                    continue;
                };
                let commands = handler.react(card, &event, state);
                self.apply_effect_commands(state, events, card.instance_id, commands)?;
            }
        }
        Ok(())
    }

    fn emit_and_resolve(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
        event: Event,
    ) -> Result<(), EngineError> {
        let start_index = events.len();
        events.push(event);
        self.resolve_effect_queue_from(state, events, start_index)
    }

    fn reduce_effect(
        &self,
        state: &mut GameState,
        source_card_instance: u32,
        command: &EffectCommand,
    ) -> Result<Vec<Event>, EngineError> {
        let mut emitted = Vec::new();
        match command {
            EffectCommand::AdjustHalfStars { player, delta } => {
                let current = i32::from(state.half_star_totals[player.index()]);
                let adjusted = current + i32::from(*delta);
                if !(0..=i32::from(u16::MAX)).contains(&adjusted) {
                    return Err(EngineError::InvalidEffect(
                        "half-star adjustment is out of range".to_owned(),
                    ));
                }
                state.half_star_totals[player.index()] = adjusted as u16;
            }
            EffectCommand::SpawnPiece {
                square,
                piece,
                replace,
            } => {
                if !replace && state.board.get(*square).is_some() {
                    return Err(EngineError::InvalidEffect(format!(
                        "cannot spawn onto occupied square {square}"
                    )));
                }
                if let Some(replaced) = state.board.get(*square) {
                    self.clear_removed_piece_statuses(state, replaced.id);
                }
                let mut spawned = *piece;
                if spawned.id == 0 {
                    spawned.id = state.allocate_piece_id();
                }
                state.board.set(*square, Some(spawned));
                state.mark_deathmatch_progress();
            }
            EffectCommand::RemovePiece { square } => {
                if let Some(removed) = state.board.get(*square) {
                    self.clear_removed_piece_statuses(state, removed.id);
                }
                state.board.set(*square, None);
                state.mark_deathmatch_progress();
            }
            EffectCommand::GrantActions { player, count } => {
                if *player != state.turn || *count == 0 {
                    return Err(EngineError::InvalidEffect(
                        "extra actions must be granted to the current player with positive count"
                            .to_owned(),
                    ));
                }
                state.actions_remaining = state.actions_remaining.saturating_add(*count);
            }
            EffectCommand::ForceMove {
                player,
                piece_id,
                source,
                finish_turn,
            } => {
                if *player != state.turn {
                    return Err(EngineError::InvalidEffect(
                        "forced move owner must be the current player".to_owned(),
                    ));
                }
                let Some((_, piece)) = state.board.find_by_id(*piece_id) else {
                    return Err(EngineError::InvalidEffect(format!(
                        "forced move piece {piece_id} does not exist"
                    )));
                };
                if piece.owner != *player {
                    return Err(EngineError::InvalidEffect(format!(
                        "forced move piece {piece_id} has the wrong owner"
                    )));
                }
                state.phase = GamePhase::ForcedMove {
                    piece_id: *piece_id,
                    source: ForcedMoveSource::Effect(source.clone()),
                    continuation: if *finish_turn {
                        ForcedMoveContinuation::FinishTurn
                    } else {
                        ForcedMoveContinuation::ResumeTurn
                    },
                };
                emitted.push(Event::ForcedMoveStarted {
                    player: *player,
                    piece_id: *piece_id,
                    source: source.clone(),
                });
            }
            EffectCommand::RequestChoice {
                owner,
                handler_id,
                options,
                continuation,
            } => {
                if *owner != state.turn || options.is_empty() {
                    return Err(EngineError::InvalidEffect(
                        "choice must belong to the current player and contain options".to_owned(),
                    ));
                }
                if self.choices.get(handler_id).is_none() {
                    return Err(EngineError::InvalidEffect(format!(
                        "choice handler {handler_id:?} is not registered"
                    )));
                }
                let choice_id = state.allocate_choice_id();
                state.phase = GamePhase::AwaitingChoice(PendingChoice {
                    id: choice_id,
                    owner: *owner,
                    source_card_instance,
                    handler_id: handler_id.clone(),
                    options: options.clone(),
                    continuation: *continuation,
                });
                emitted.push(Event::ChoiceRequested {
                    choice_id,
                    owner: *owner,
                    handler_id: handler_id.clone(),
                    options: options.clone(),
                });
            }
            EffectCommand::EndTurn { player } => {
                if *player != state.turn || matches!(state.phase, GamePhase::AwaitingChoice(_)) {
                    return Err(EngineError::InvalidEffect(
                        "turn-end request is invalid in the current phase".to_owned(),
                    ));
                }
                state.request_turn_end();
            }
            EffectCommand::Win { winner, reason } => {
                state.status = GameStatus::Won {
                    winner: *winner,
                    reason: WinReason::CardEffect(reason.clone()),
                };
            }
            EffectCommand::Draw { reason } => {
                state.status = GameStatus::Draw {
                    reason: DrawReason::CardEffect(reason.clone()),
                };
            }
        }
        Ok(emitted)
    }

    fn grant_wizard_mana_for_loss(&self, state: &mut GameState, owner: Player) {
        let wizard_ids = state
            .board
            .iter()
            .filter(|(_, piece)| {
                piece.owner == owner && self.piece_has_ability(state, *piece, PieceKind::Wizard)
            })
            .map(|(_, piece)| piece.id)
            .collect::<Vec<_>>();
        for wizard_id in wizard_ids {
            let mana = state.card_runtime.wizard_mana.entry(wizard_id).or_default();
            *mana = mana.saturating_add(1).min(5);
        }
    }

    fn resolve_reaper_deaths(
        &self,
        state: &mut GameState,
        deaths: &[(Square, Piece)],
        captured_by: Player,
        events: &mut Vec<Event>,
    ) {
        if state.status != GameStatus::Ongoing {
            return;
        }
        let dead_ids = deaths
            .iter()
            .map(|(_, piece)| piece.id)
            .collect::<BTreeSet<_>>();
        for &(death_square, victim) in deaths {
            if captured_by != victim.owner.opponent() {
                continue;
            }
            let witnesses = state
                .board
                .iter()
                .filter(|(square, piece)| {
                    self.piece_has_ability(state, *piece, PieceKind::Reaper)
                        && !dead_ids.contains(&piece.id)
                        && square.file().abs_diff(death_square.file()) <= 1
                        && square.rank().abs_diff(death_square.rank()) <= 1
                })
                .collect::<Vec<_>>();
            for (reaper_square, reaper) in witnesses {
                let souls = state
                    .card_runtime
                    .reaper_souls
                    .entry(reaper.id)
                    .or_default();
                *souls = souls.saturating_add(1).min(4);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("reaper-soul:{}:{}/4", reaper.id, *souls),
                });
                if *souls < 4 || state.status != GameStatus::Ongoing {
                    continue;
                }
                let Some((royal_square, royal)) = state.board.iter().find(|(_, piece)| {
                    piece.owner == reaper.owner.opponent() && self.is_royal_piece(state, *piece)
                }) else {
                    state.status = GameStatus::Won {
                        winner: reaper.owner,
                        reason: WinReason::CardEffect("reaper-four-souls".to_owned()),
                    };
                    continue;
                };
                state.board.set(royal_square, None);
                self.clear_removed_piece_statuses(state, royal.id);
                if state
                    .board
                    .get(reaper_square)
                    .is_some_and(|piece| piece.id == reaper.id)
                {
                    state.board.set(reaper_square, None);
                    state.board.set(royal_square, Some(reaper));
                }
                events.push(Event::PieceRemoved {
                    actor: reaper.owner,
                    piece: royal,
                    at: royal_square,
                    cause: "reaper-execution".to_owned(),
                });
                state.status = GameStatus::Won {
                    winner: reaper.owner,
                    reason: WinReason::CardEffect("reaper-four-souls".to_owned()),
                };
            }
        }
    }

    fn available_adjacent_idol(
        &self,
        state: &GameState,
        from: Square,
        owner: Player,
    ) -> Option<u32> {
        for (file_delta, rank_delta) in [
            (-1, 1),
            (1, 1),
            (-1, -1),
            (1, -1),
            (0, 1),
            (0, -1),
            (-1, 0),
            (1, 0),
        ] {
            let Some(square) = from.offset(file_delta, rank_delta) else {
                continue;
            };
            let Some(piece) = state.board.get(square) else {
                continue;
            };
            if piece.owner == owner
                && self.piece_has_ability(state, piece, PieceKind::Idol)
                && state.card_runtime.idol_used_turn.get(&piece.id).copied()
                    != Some(state.turns_taken[owner.index()])
            {
                return Some(piece.id);
            }
        }
        None
    }

    fn apply_idol_encore(&self, state: &mut GameState, idol_id: Option<u32>, moved_piece_id: u32) {
        let Some(idol_id) = idol_id else {
            return;
        };
        state
            .card_runtime
            .idol_rest_piece_ids
            .insert(moved_piece_id);
        if self.generate_move_actions(state).is_empty() {
            state
                .card_runtime
                .idol_rest_piece_ids
                .remove(&moved_piece_id);
            return;
        }
        state
            .card_runtime
            .idol_used_turn
            .insert(idol_id, state.turns_taken[state.turn.index()]);
        state.actions_remaining = state.actions_remaining.saturating_add(1);
    }

    fn resolve_herald_victory(&self, state: &mut GameState) {
        if state.status != GameStatus::Ongoing {
            return;
        }
        let heralds = state
            .board
            .iter()
            .filter(|(_, piece)| self.piece_has_ability(state, *piece, PieceKind::Herald))
            .collect::<Vec<_>>();
        for (herald_square, herald) in heralds {
            let reached_royal = state.board.iter().any(|(square, piece)| {
                piece.owner == herald.owner.opponent()
                    && self.is_royal_piece(state, piece)
                    && square.file().abs_diff(herald_square.file()) <= 1
                    && square.rank().abs_diff(herald_square.rank()) <= 1
            });
            if reached_royal {
                state.status = GameStatus::Won {
                    winner: herald.owner,
                    reason: WinReason::CardEffect("herald-arrival".to_owned()),
                };
                break;
            }
        }
    }

    fn clear_promotion_inherited_traits(&self, state: &mut GameState, piece_id: u32) {
        state.card_runtime.shielded_piece_ids.remove(&piece_id);
        state.card_runtime.explosive_piece_ids.remove(&piece_id);
        state.card_runtime.chimera_piece_ids.remove(&piece_id);
        state.card_runtime.witch_trials.remove(&piece_id);
        state.card_runtime.severed.remove(&piece_id);
        if state
            .card_runtime
            .potion_basic_training_piece_ids
            .remove(&piece_id)
        {
            state
                .card_runtime
                .basic_training_piece_ids
                .remove(&piece_id);
        }
        if state.card_runtime.potion_manner_piece_ids.remove(&piece_id) {
            state
                .card_runtime
                .manner_captured_last_piece_ids
                .remove(&piece_id);
        }
        if state
            .card_runtime
            .potion_saturation_piece_ids
            .remove(&piece_id)
        {
            state
                .card_runtime
                .saturation_capture_counts
                .remove(&piece_id);
        }
        state.card_runtime.charge_pawns.remove(&piece_id);
        state.card_runtime.trojan_piece_ids.remove(&piece_id);
        state
            .card_runtime
            .feudal_contracts
            .retain(|contract| contract.pawn_id != piece_id);
        state
            .card_runtime
            .vip_pending
            .retain(|pending| pending.piece_id != piece_id);
    }

    fn clear_removed_piece_statuses(&self, state: &mut GameState, piece_id: u32) {
        if let Some(flags) = &mut state.card_runtime.capture_the_flag {
            for occupation in &mut flags.occupations {
                if occupation.is_some_and(|entry| entry.piece_id == piece_id) {
                    *occupation = None;
                }
            }
        }
        state.card_runtime.moved_piece_ids.remove(&piece_id);
        state.card_runtime.locust_swarm_piece_ids.remove(&piece_id);
        state.card_runtime.metalized_piece_ids.remove(&piece_id);
        state.card_runtime.metal_cooldowns.remove(&piece_id);
        state.card_runtime.recurrence_piece_ids.remove(&piece_id);
        state.card_runtime.fresh_no_capture_until.remove(&piece_id);
        state.card_runtime.disarmed.remove(&piece_id);
        state.card_runtime.frozen.remove(&piece_id);
        state.card_runtime.poison_stun_turns.remove(&piece_id);
        if state
            .card_runtime
            .reposition
            .is_some_and(|status| status.piece_id == piece_id)
        {
            state.card_runtime.reposition = None;
        }
        state.card_runtime.parry_piece_ids.remove(&piece_id);
        state.card_runtime.empty_lunchboxes.remove(&piece_id);
        state.card_runtime.stake_statuses.remove(&piece_id);
        state
            .card_runtime
            .herald_jump_locked_until
            .remove(&piece_id);
        state.card_runtime.windmill_modes.remove(&piece_id);
        state.card_runtime.log_statuses.remove(&piece_id);
        state.card_runtime.wizard_mana.remove(&piece_id);
        state.card_runtime.trickster_move_kinds.remove(&piece_id);
        state
            .card_runtime
            .siren_exposure
            .retain(|(siren_id, target_id), _| *siren_id != piece_id && *target_id != piece_id);
        state.card_runtime.reaper_souls.remove(&piece_id);
        state.card_runtime.shielded_piece_ids.remove(&piece_id);
        state.card_runtime.protected_piece_ids.remove(&piece_id);
        state.card_runtime.idol_used_turn.remove(&piece_id);
        state.card_runtime.idol_rest_piece_ids.remove(&piece_id);
        state.card_runtime.loyalist_piece_ids.remove(&piece_id);
        state.card_runtime.royal_piece_ids.remove(&piece_id);
        state.card_runtime.holdout_promotions.remove(&piece_id);
        state.card_runtime.merchant_gold.remove(&piece_id);
        state.card_runtime.large_pieces.remove(&piece_id);
        state.card_runtime.piece_origins.remove(&piece_id);
        state
            .card_runtime
            .basic_training_piece_ids
            .remove(&piece_id);
        state
            .card_runtime
            .promotion_rush_piece_ids
            .remove(&piece_id);
        state.card_runtime.evasion_piece_ids.remove(&piece_id);
        state.card_runtime.ghost_piece_ids.remove(&piece_id);
        state.card_runtime.inertia_piece_ids.remove(&piece_id);
        state.card_runtime.ice_sheet.remove(&piece_id);
        state.card_runtime.severed.remove(&piece_id);
        state.card_runtime.chameleon_piece_ids.remove(&piece_id);
        state.card_runtime.chimera_piece_ids.remove(&piece_id);
        state.card_runtime.imperial_moves.remove(&piece_id);
        state.card_runtime.poisoned_piece_ids.remove(&piece_id);
        state.card_runtime.sacrifice_protection.remove(&piece_id);
        state.card_runtime.last_resistance.remove(&piece_id);
        state.card_runtime.coronation_protection.remove(&piece_id);
        state
            .card_runtime
            .queens_gambit_protection
            .remove(&piece_id);
        state
            .card_runtime
            .potion_basic_training_piece_ids
            .remove(&piece_id);
        state.card_runtime.potion_manner_piece_ids.remove(&piece_id);
        state
            .card_runtime
            .manner_captured_last_piece_ids
            .remove(&piece_id);
        state
            .card_runtime
            .potion_saturation_piece_ids
            .remove(&piece_id);
        state
            .card_runtime
            .saturation_capture_counts
            .remove(&piece_id);
        state.card_runtime.explosive_piece_ids.remove(&piece_id);
        state.card_runtime.trojan_piece_ids.remove(&piece_id);
        state.card_runtime.quantum_positions.remove(&piece_id);
        state.card_runtime.stealth_piece_ids.remove(&piece_id);
        state.card_runtime.submerged_piece_ids.remove(&piece_id);
        state.card_runtime.nullification_piece_ids.remove(&piece_id);
        state.card_runtime.witch_trials.remove(&piece_id);
        state.card_runtime.baby_bear_grow_turns.remove(&piece_id);
        state.card_runtime.bear_retaliations.remove(&piece_id);
        state.card_runtime.trickster_move_kinds.remove(&piece_id);
        state
            .card_runtime
            .feudal_contracts
            .retain(|contract| contract.pawn_id != piece_id && contract.guardian_id != piece_id);
        state
            .card_runtime
            .twin_bonds
            .retain(|bond| bond.first_id != piece_id && bond.second_id != piece_id);
    }

    /// Drops state that belongs to a piece's concrete type while retaining generic
    /// card statuses that remain attached to the physical piece after a transform.
    fn clear_queens_gambit_protection(&self, state: &mut GameState, piece_id: u32) {
        if state
            .card_runtime
            .queens_gambit_protection
            .remove(&piece_id)
        {
            state.card_runtime.protected_piece_ids.remove(&piece_id);
        }
    }

    fn clear_piece_kind_runtime(&self, state: &mut GameState, piece_id: u32) {
        self.clear_queens_gambit_protection(state, piece_id);
        state
            .card_runtime
            .herald_jump_locked_until
            .remove(&piece_id);
        state.card_runtime.windmill_modes.remove(&piece_id);
        state.card_runtime.log_statuses.remove(&piece_id);
        state.card_runtime.wizard_mana.remove(&piece_id);
        state.card_runtime.reaper_souls.remove(&piece_id);
        state.card_runtime.merchant_gold.remove(&piece_id);
        state.card_runtime.large_pieces.remove(&piece_id);
        state.card_runtime.baby_bear_grow_turns.remove(&piece_id);
        state.card_runtime.bear_retaliations.remove(&piece_id);
    }

    fn resolve_calling_card_capture(
        &self,
        state: &mut GameState,
        captured: Piece,
        attacker: Option<Piece>,
        events: &mut Vec<Event>,
    ) {
        let Some(by) = state.card_runtime.calling_cards.remove(&captured.id) else {
            return;
        };
        let remaining = web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, piece)| piece.owner == captured.owner && piece.id != captured.id)
            .collect::<Vec<_>>();
        let nonroyal = remaining
            .iter()
            .copied()
            .filter(|(_, piece)| {
                attacker.is_none_or(|attacker| piece.id != attacker.id)
                    && !self.is_royal_piece(state, *piece)
                    && !matches!(piece.kind, PieceKind::Guard | PieceKind::Jester)
            })
            .collect::<Vec<_>>();
        let candidates = if nonroyal.is_empty()
            && !remaining.is_empty()
            && remaining
                .iter()
                .all(|(_, piece)| self.is_royal_piece(state, *piece))
        {
            remaining
                .into_iter()
                .filter(|(_, piece)| attacker.is_none_or(|attacker| piece.id != attacker.id))
                .collect::<Vec<_>>()
        } else {
            nonroyal
        };
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        events.push(Event::RandomSample {
            purpose: "calling-card-followup".to_owned(),
            roll_53: roll,
        });
        let Some(index) = index else {
            return;
        };
        let (square, piece) = candidates[index];
        state.card_runtime.calling_cards.remove(&piece.id);
        self.remove_piece_for_card(state, square, piece, by, CALLING_CARD, events);
    }

    fn resolve_after_counted_move(&self, state: &mut GameState, events: &mut Vec<Event>) {
        let mut pending = std::mem::take(&mut state.card_runtime.pending_otherworld);
        let mut remaining = Vec::new();
        for entry in pending.drain(..) {
            if state.ply < entry.due_ply {
                remaining.push(entry);
                continue;
            }
            let captured_blocker = if let Some(blocker) = self.board_piece_at(state, entry.square) {
                let blocker_square = state
                    .board
                    .find_by_id(blocker.id)
                    .map(|(square, _)| square)
                    .unwrap_or(entry.square);
                self.remove_piece_for_card(
                    state,
                    blocker_square,
                    blocker,
                    entry.owner,
                    OTHERWORLD,
                    events,
                );
                true
            } else {
                false
            };
            let mut wizard = Piece::new(entry.owner, PieceKind::Wizard).with_id(entry.piece_id);
            wizard.capture_count = u16::from(captured_blocker);
            state.board.set(entry.square, Some(wizard));
            state.card_runtime.wizard_mana.insert(wizard.id, 0);
            state
                .card_runtime
                .piece_origins
                .insert(wizard.id, entry.origin);
            state.card_runtime.moved_piece_ids.insert(wizard.id);
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("otherworld-return:{}:{}", wizard.id, entry.square),
            });
        }
        state.card_runtime.pending_otherworld = remaining;
        let pending_lobsters = std::mem::take(&mut state.card_runtime.pending_lobsters);
        for pending in pending_lobsters {
            if state.ply < pending.due_ply {
                state.card_runtime.pending_lobsters.push(pending);
            } else if self.board_piece_at(state, pending.square).is_none()
                && !state.card_runtime.barricades.contains(&pending.square)
                && !state.card_runtime.black_holes.contains(&pending.square)
                && state.card_runtime.crown_pos != Some(pending.square)
            {
                let lobster = Piece::new(pending.owner, PieceKind::Lobster)
                    .with_id(state.allocate_piece_id());
                state.board.set(pending.square, Some(lobster));
                self.register_spawned_piece(state, pending.square, lobster, true);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("lobster-spawn:{}:{}", lobster.id, pending.square),
                });
            }
        }
        self.resolve_undead_resurrections(state, events);
        self.tick_siren_exposure(state, events);
        self.resolve_gomoku_victory(state);
    }

    fn resolve_undead_resurrections(&self, state: &mut GameState, events: &mut Vec<Event>) {
        let pending = std::mem::take(&mut state.card_runtime.undead_resurrections);
        for mut resurrection in pending {
            if state.ply < resurrection.due_ply {
                state.card_runtime.undead_resurrections.push(resurrection);
                continue;
            }
            let rank = if resurrection.piece.owner == Player::White {
                0
            } else {
                7
            };
            let candidates = (0..8)
                .filter_map(|file| Square::new(file, rank))
                .filter(|square| state.board.get(*square).is_none())
                .collect::<Vec<_>>();
            let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
            events.push(Event::RandomSample {
                purpose: "undead-resurrection".to_owned(),
                roll_53: roll,
            });
            if let Some(index) = index {
                let square = candidates[index];
                let piece = resurrection.piece;
                state.board.set(square, Some(piece));
                state.card_runtime.moved_piece_ids.insert(piece.id);
                if piece.kind == PieceKind::Trickster {
                    state
                        .card_runtime
                        .trickster_move_kinds
                        .insert(piece.id, PieceKind::Undead);
                }
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("undead-return:{}:{}", piece.id, square),
                });
            } else {
                resurrection.due_ply = state.ply.saturating_add(1);
                state.card_runtime.undead_resurrections.push(resurrection);
            }
        }
    }

    fn tick_siren_exposure(&self, state: &mut GameState, events: &mut Vec<Event>) {
        let sirens = state
            .board
            .iter()
            .filter(|(_, piece)| {
                piece.kind == PieceKind::Siren
                    || (piece.kind == PieceKind::Trickster
                        && state
                            .card_runtime
                            .trickster_move_kinds
                            .get(&piece.id)
                            .is_some_and(|kind| *kind == PieceKind::Siren))
            })
            .collect::<Vec<_>>();
        let active_sirens = sirens
            .iter()
            .map(|(_, piece)| piece.id)
            .collect::<BTreeSet<_>>();
        state
            .card_runtime
            .siren_exposure
            .retain(|(siren_id, _), _| active_sirens.contains(siren_id));
        for (square, siren) in sirens {
            let nearby = state
                .board
                .iter()
                .filter(|(target_square, target)| {
                    target.owner != siren.owner
                        && !matches!(
                            target.kind,
                            PieceKind::Monster | PieceKind::Scarecrow | PieceKind::Crown
                        )
                        && squares_are_adjacent(square, *target_square)
                })
                .map(|(_, target)| target.id)
                .collect::<BTreeSet<_>>();
            state
                .card_runtime
                .siren_exposure
                .retain(|(siren_id, target_id), _| {
                    *siren_id != siren.id || nearby.contains(target_id)
                });
            for target_id in nearby {
                let count = state
                    .card_runtime
                    .siren_exposure
                    .entry((siren.id, target_id))
                    .or_default();
                *count = count.saturating_add(1);
                if *count < 4 {
                    continue;
                }
                if let Some((target_square, mut target)) = state.board.find_by_id(target_id) {
                    target.owner = siren.owner;
                    state.board.set(target_square, Some(target));
                    state
                        .card_runtime
                        .siren_exposure
                        .remove(&(siren.id, target_id));
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("siren-convert:{}:{}", siren.id, target.id),
                    });
                }
            }
        }
    }

    fn resolve_gomoku_victory(&self, state: &mut GameState) {
        self.resolve_highlander_victory(state);
        if state.status != GameStatus::Ongoing {
            return;
        }
        for player in [Player::White, Player::Black] {
            if !state.card_runtime.gomoku[player.index()] {
                continue;
            }
            for file in 0..8 {
                let mut consecutive = 0_u8;
                for rank in 0..8 {
                    let occupied = Square::new(file, rank)
                        .and_then(|square| state.board.get(square))
                        .is_some_and(|piece| piece.owner == player);
                    consecutive = if occupied {
                        consecutive.saturating_add(1)
                    } else {
                        0
                    };
                    if consecutive >= 5 {
                        state.status = GameStatus::Won {
                            winner: player,
                            reason: WinReason::CardEffect("gomoku".to_owned()),
                        };
                        return;
                    }
                }
            }
        }
    }

    fn resolve_holdout_promotions(&self, state: &mut GameState, player: Player) {
        let due = state
            .card_runtime
            .holdout_promotions
            .iter()
            .filter_map(|(piece_id, ready)| {
                (state.shared_turn_count() >= *ready)
                    .then_some(*piece_id)
                    .filter(|piece_id| {
                        state.board.find_by_id(*piece_id).is_some_and(|(_, piece)| {
                            piece.owner == player && piece.kind == PieceKind::Pawn
                        })
                    })
            })
            .collect::<Vec<_>>();
        for piece_id in due {
            if let Some((square, mut pawn)) = state.board.find_by_id(piece_id) {
                pawn.kind = PieceKind::Queen;
                state.board.set(square, Some(pawn));
                state.card_runtime.moved_piece_ids.insert(piece_id);
            }
            state.card_runtime.holdout_promotions.remove(&piece_id);
        }
    }

    fn grant_merchant_gold(&self, state: &mut GameState, player: Player) {
        let merchants = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == player && piece.kind == PieceKind::Merchant)
            .map(|(_, piece)| piece.id)
            .collect::<Vec<_>>();
        for piece_id in merchants {
            let gold = state
                .card_runtime
                .merchant_gold
                .entry(piece_id)
                .or_default();
            *gold = gold.saturating_add(1);
        }
    }

    fn resolve_pending_recurrences(&self, state: &mut GameState, events: &mut Vec<Event>) {
        let pending = std::mem::take(&mut state.card_runtime.pending_recurrences);
        for recurrence in pending {
            if state.board.find_by_id(recurrence.piece.id).is_some() {
                continue;
            }
            let home_rank = if recurrence.piece.owner == Player::White {
                0
            } else {
                7
            };
            let candidates = (0..8_u8)
                .filter_map(|file| Square::new(file, home_rank))
                .filter(|square| state.board.get(*square).is_none())
                .collect::<Vec<_>>();
            let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
            events.push(Event::RandomSample {
                purpose: format!("recurrence-respawn:{}", recurrence.piece.id),
                roll_53: roll,
            });
            let Some(index) = index else {
                state.card_runtime.pending_recurrences.push(recurrence);
                continue;
            };
            let square = candidates[index];
            state.board.set(square, Some(recurrence.piece));
            state
                .card_runtime
                .piece_origins
                .insert(recurrence.piece.id, square);
            state
                .card_runtime
                .moved_piece_ids
                .insert(recurrence.piece.id);
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("recurrence:{}:{square}", recurrence.piece.id),
            });
        }
    }

    fn resolve_missing_king(&self, state: &mut GameState, events: &mut Vec<Event>) {
        self.resolve_pending_recurrences(state, events);
        if state.status != GameStatus::Ongoing || !self.ruleset.base_rules.king_capture_wins {
            return;
        }
        let defeated = |state: &GameState, player: Player| {
            if state.card_runtime.democracy[player.index()] {
                !state
                    .board
                    .iter()
                    .any(|(_, piece)| piece.owner == player && piece.kind == PieceKind::Pawn)
            } else {
                !state.board.iter().any(|(_, piece)| {
                    piece.owner == player
                        && (piece.kind == PieceKind::King
                            || matches!(
                                piece.kind,
                                PieceKind::RoyalKnight
                                    | PieceKind::Merchant
                                    | PieceKind::DarkWizard
                            )
                            || state.card_runtime.royal_piece_ids.contains(&piece.id))
                }) && !state
                    .card_runtime
                    .pending_recurrences
                    .iter()
                    .any(|pending| {
                        pending.piece.owner == player
                            && matches!(
                                pending.piece.kind,
                                PieceKind::King
                                    | PieceKind::RoyalKnight
                                    | PieceKind::Merchant
                                    | PieceKind::DarkWizard
                            )
                    })
            }
        };
        if defeated(state, Player::White) {
            state.status = GameStatus::Won {
                winner: Player::Black,
                reason: WinReason::KingCaptured,
            };
        } else if defeated(state, Player::Black) {
            state.status = GameStatus::Won {
                winner: Player::White,
                reason: WinReason::KingCaptured,
            };
        }
    }

    fn resolve_lower_stars_after_deathmatch(&self, state: &mut GameState) {
        let [white, black] = state.half_star_totals;
        state.status = match white.cmp(&black) {
            std::cmp::Ordering::Less => GameStatus::Won {
                winner: Player::White,
                reason: WinReason::LowerStarsAfterDeathmatch,
            },
            std::cmp::Ordering::Greater => GameStatus::Won {
                winner: Player::Black,
                reason: WinReason::LowerStarsAfterDeathmatch,
            },
            std::cmp::Ordering::Equal => GameStatus::Draw {
                reason: DrawReason::EqualStarsAfterDeathmatch,
            },
        };
    }

    fn tick_deathmatch_after_turn(&self, state: &mut GameState, actor: Player) -> bool {
        if actor != Player::Black {
            return false;
        }
        let Some(deathmatch) = &mut state.deathmatch else {
            return false;
        };
        if deathmatch.progress_this_round {
            deathmatch.quiet_full_turns = 0;
            deathmatch.progress_this_round = false;
            return false;
        }
        deathmatch.quiet_full_turns = deathmatch
            .quiet_full_turns
            .saturating_add(1)
            .min(deathmatch.interval_full_turns);
        if deathmatch.quiet_full_turns < deathmatch.interval_full_turns {
            return false;
        }
        self.resolve_lower_stars_after_deathmatch(state);
        true
    }

    fn start_deathmatch_if_due(&self, state: &mut GameState, events: &mut Vec<Event>) {
        if state.deathmatch.is_some()
            || state.shared_turn_count() < self.ruleset.base_rules.deathmatch_starts_at_move_count
        {
            return;
        }
        let shared_turn = state.shared_turn_count();
        let interval_full_turns = if state.card_runtime.revelation {
            5
        } else {
            self.ruleset.base_rules.deathmatch_quiet_move_count.max(1)
        };
        state.deathmatch = Some(DeathmatchState {
            started_at_shared_turn: shared_turn,
            quiet_full_turns: 0,
            interval_full_turns,
            progress_this_round: false,
        });
        events.push(Event::DeathmatchStarted {
            shared_turn,
            interval_full_turns,
        });
    }

    fn start_due_milestone_draft(
        &self,
        state: &mut GameState,
        resume_turn: Player,
        events: &mut Vec<Event>,
    ) -> bool {
        if !state.completed_drafts[DraftPhase::Opening.index()] {
            return false;
        }
        let shared_turn = state.shared_turn_count();
        let phase = if !state.completed_drafts[DraftPhase::Middle.index()] && shared_turn >= 10 {
            Some(DraftPhase::Middle)
        } else if state.completed_drafts[DraftPhase::Middle.index()]
            && !state.completed_drafts[DraftPhase::End.index()]
            && shared_turn >= 20
        {
            Some(DraftPhase::End)
        } else {
            None
        };
        let Some(phase) = phase else {
            return false;
        };

        self.start_draft(state, phase, Player::White, Some(resume_turn), None);
        let GamePhase::Draft(draft) = &state.phase else {
            unreachable!("start_draft always installs a draft")
        };
        if draft.offers.is_empty() {
            events.push(Event::DraftSkipped {
                phase,
                player: Player::White,
            });
            self.start_draft(state, phase, Player::Black, Some(resume_turn), None);
            let GamePhase::Draft(black_draft) = &state.phase else {
                unreachable!("start_draft always installs a draft")
            };
            if black_draft.offers.is_empty() {
                events.push(Event::DraftSkipped {
                    phase,
                    player: Player::Black,
                });
                state.completed_drafts[phase.index()] = true;
                state.phase = GamePhase::Playing;
                state.turn = resume_turn;
                events.push(Event::DraftCompleted { phase, resume_turn });
                return false;
            }
            events.push(Event::DraftStarted {
                phase,
                player: Player::Black,
                offers: black_draft.offers.clone(),
            });
            return true;
        }
        events.push(Event::DraftStarted {
            phase,
            player: Player::White,
            offers: draft.offers.clone(),
        });
        true
    }

    fn finish_turn(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        self.finish_turn_chain(state, actor, events, 0)
    }

    fn arrest_unsubmerged_thieves(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        let thieves = state
            .board
            .iter()
            .filter(|(_, piece)| {
                piece.owner == actor
                    && piece.kind == PieceKind::Thief
                    && !state.card_runtime.submerged_piece_ids.contains(&piece.id)
            })
            .collect::<Vec<_>>();
        for (square, thief) in thieves {
            state.board.set(square, None);
            self.clear_removed_piece_statuses(state, thief.id);
            events.push(Event::PieceRemoved {
                actor,
                piece: thief,
                at: square,
                cause: "thief-arrest".to_owned(),
            });
        }
    }

    fn resolve_delayed_hazards(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        let hazards = std::mem::take(&mut state.card_runtime.delayed_hazards);
        for hazard in hazards {
            if hazard.trigger_after != actor {
                state.card_runtime.delayed_hazards.push(hazard);
                continue;
            }
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("wizard-hazard:{:?}", hazard.kind),
            });
            for square in hazard.squares {
                let Some(piece) = state.board.get(square) else {
                    continue;
                };
                if state.card_runtime.frozen.contains_key(&piece.id) {
                    continue;
                }
                if state.card_runtime.shielded_piece_ids.remove(&piece.id) {
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("shield-broken:{}", piece.id),
                    });
                    continue;
                }
                state.board.set(square, None);
                self.record_new_capture_reactions(state, piece);
                self.clear_removed_piece_statuses(state, piece.id);
                events.push(Event::PieceRemoved {
                    actor: hazard.owner,
                    piece,
                    at: square,
                    cause: match hazard.kind {
                        DelayedHazardKind::Lightning => "wizard-lightning",
                        DelayedHazardKind::Meteor => "wizard-meteor",
                    }
                    .to_owned(),
                });
                self.grant_wizard_mana_for_loss(state, piece.owner);
                self.resolve_reaper_deaths(state, &[(square, piece)], hazard.owner, events);
            }
        }
    }

    fn advance_logs_after_turn(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        let log_ids = state
            .card_runtime
            .log_statuses
            .iter()
            .filter(|(piece_id, status)| {
                status.roll_after_turn <= state.turns_taken[actor.index()]
                    && state
                        .board
                        .find_by_id(**piece_id)
                        .is_some_and(|(_, piece)| {
                            piece.owner == actor && piece.kind == PieceKind::Log
                        })
            })
            .map(|(piece_id, _)| *piece_id)
            .collect::<Vec<_>>();
        for piece_id in log_ids {
            let Some((from, log_piece)) = state.board.find_by_id(piece_id) else {
                state.card_runtime.log_statuses.remove(&piece_id);
                continue;
            };
            let Some(status) = state.card_runtime.log_statuses.get(&piece_id).copied() else {
                continue;
            };
            let Some(to) = from.offset(status.file_delta, status.rank_delta) else {
                state.card_runtime.log_statuses.remove(&piece_id);
                continue;
            };
            match state.board.get(to) {
                None => {
                    state.board.set(from, None);
                    state.board.set(to, Some(log_piece));
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("log-roll:{piece_id}:{from}->{to}"),
                    });
                }
                Some(target)
                    if self.can_capture(state, log_piece, target)
                        && !state.card_runtime.shielded_piece_ids.contains(&target.id) =>
                {
                    state.board.set(from, None);
                    state.board.set(to, Some(log_piece));
                    self.record_new_capture_reactions(state, target);
                    self.clear_removed_piece_statuses(state, target.id);
                    events.push(Event::PieceCaptured {
                        actor,
                        piece: target,
                        at: to,
                    });
                    self.grant_wizard_mana_for_loss(state, target.owner);
                    self.resolve_reaper_deaths(state, &[(to, target)], actor, events);
                    state.mark_deathmatch_progress();
                }
                Some(target) => {
                    if state.card_runtime.shielded_piece_ids.remove(&target.id)
                        && target.owner != actor
                    {
                        events.push(Event::EffectApplied {
                            source_card_instance: 0,
                            command: format!("shield-broken:{}", target.id),
                        });
                    }
                    state.card_runtime.log_statuses.remove(&piece_id);
                }
            }
        }
    }

    fn finish_turn_chain(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
        automatic_turn_ends: usize,
    ) -> Result<(), EngineError> {
        if automatic_turn_ends >= MAX_AUTOMATIC_TURN_ENDS_PER_TRANSITION {
            return Err(EngineError::AutomaticTurnEndLimit(
                MAX_AUTOMATIC_TURN_ENDS_PER_TRANSITION,
            ));
        }
        state.take_turn_end_request();
        if state.status != GameStatus::Ongoing {
            self.push_game_ended(state, events);
            return Ok(());
        }

        self.emit_and_resolve(state, events, Event::TurnEnding { player: actor })?;
        // Live refreshes the move result before delayed hazards resolve (JS 79897
        // before 79971): an enemy that stepped adjacent is revealed even if a
        // hazard removes it before the board is inspected again.
        self.refresh_submerged_pieces(state);
        self.arrest_unsubmerged_thieves(state, actor, events);
        self.resolve_delayed_hazards(state, actor, events);
        self.advance_logs_after_turn(state, actor, events);
        self.resolve_missing_king(state, events);
        self.resolve_herald_victory(state);
        if state.status != GameStatus::Ongoing {
            self.push_game_ended(state, events);
            return Ok(());
        }
        if !matches!(state.phase, GamePhase::Playing) {
            return Ok(());
        }
        // An EndTurn command raised while TurnEnding is already being handled is satisfied by
        // this same transition and must not leak into the opponent's turn.
        state.take_turn_end_request();

        state.turns_taken[actor.index()] = state.turns_taken[actor.index()].saturating_add(1);
        self.advance_capture_flags(state, actor, true);
        if state.status != GameStatus::Ongoing {
            self.push_game_ended(state, events);
            return Ok(());
        }
        state.card_runtime.idol_rest_piece_ids.retain(|piece_id| {
            state
                .board
                .find_by_id(*piece_id)
                .is_some_and(|(_, piece)| piece.owner != actor)
        });
        self.resolve_empty_lunchboxes_after_turn(state, actor, events);
        if state.status != GameStatus::Ongoing {
            self.push_game_ended(state, events);
            return Ok(());
        }
        self.tick_platform_after_turn(state, events);
        self.tick_card_statuses_after_turn(state, actor);
        self.tick_middle_statuses_after_turn(state, actor, events);
        self.tick_end_statuses_after_turn(state, actor, events);
        self.complete_octopus_turn(state, actor);
        state.card_runtime.free_move_no_capture[actor.index()] = false;
        state.card_runtime.relay[actor.index()] = false;
        state.card_runtime.magic_girl_surge[actor.index()] = false;
        self.resolve_due_premoves_after_turn(state, actor, events);
        if self.tick_deathmatch_after_turn(state, actor) {
            self.push_game_ended(state, events);
            return Ok(());
        }
        self.start_deathmatch_if_due(state, events);
        events.push(Event::TurnEnded { player: actor });

        let defender = actor.opponent();
        let next_player = if state.card_runtime.skip_turns[defender.index()] {
            state.card_runtime.skip_turns[defender.index()] = false;
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("time-stop-skip:{defender}"),
            });
            actor
        } else {
            defender
        };
        state.turn = next_player;
        state.phase = GamePhase::Playing;
        let black_turns = state.turns_taken[Player::Black.index()];
        state.actions_remaining = if state.card_runtime.acceleration
            && (black_turns >= 3 || (next_player == Player::Black && black_turns >= 2))
        {
            2
        } else {
            1
        };
        state.cards_used_this_turn[actor.index()] = 0;
        state.cards_used_this_turn[defender.index()] = 0;
        let next_turn_event_start = events.len();
        self.resolve_holdout_promotions(state, next_player);
        self.grant_merchant_gold(state, next_player);
        events.push(Event::TurnChanged {
            player: next_player,
        });
        self.activate_cards_for_turn(state, next_player, events);
        events.push(Event::TurnStarted {
            player: next_player,
            actions_remaining: state.actions_remaining,
        });
        self.resolve_pending_portals_for_turn(state, next_player, events);
        self.resolve_middle_turn_start(state, next_player, events);
        self.resolve_effect_queue_from(state, events, next_turn_event_start)?;
        self.resolve_missing_king(state, events);
        if state.status != GameStatus::Ongoing {
            self.push_game_ended(state, events);
            return Ok(());
        }
        if !matches!(state.phase, GamePhase::Playing) {
            return Ok(());
        }
        if state.take_turn_end_request() {
            return self.finish_turn_chain(state, next_player, events, automatic_turn_ends + 1);
        }
        if self.start_due_milestone_draft(state, next_player, events) {
            return Ok(());
        }
        self.resolve_no_action_loss(state, next_player, events);

        if state.status == GameStatus::Ongoing {
            let repetition_count = state.record_current_position();
            if repetition_count >= self.ruleset.base_rules.repetition_count {
                let [white, black] = state.half_star_totals;
                state.status = match white.cmp(&black) {
                    std::cmp::Ordering::Less => GameStatus::Won {
                        winner: Player::White,
                        reason: WinReason::LowerStarsOnRepetition,
                    },
                    std::cmp::Ordering::Greater => GameStatus::Won {
                        winner: Player::Black,
                        reason: WinReason::LowerStarsOnRepetition,
                    },
                    std::cmp::Ordering::Equal => GameStatus::Draw {
                        reason: DrawReason::EqualStarsOnRepetition,
                    },
                };
            }
        }
        if state.status != GameStatus::Ongoing {
            self.push_game_ended(state, events);
        }
        Ok(())
    }

    fn resolve_pending_portals_for_turn(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let pending = std::mem::take(&mut state.card_runtime.pending_portals);
        let mut selected = None;
        for portal in pending {
            if portal.owner == owner && state.turns_taken[owner.index()] >= portal.due_turn {
                selected = Some(portal);
            } else {
                state.card_runtime.pending_portals.push(portal);
            }
        }
        if let Some(portal) = selected
            && portal.squares.iter().all(|square| {
                !state.card_runtime.black_holes.contains(square)
                    && state.card_runtime.crown_pos != Some(*square)
                    && !(state.card_runtime.collapsed_files_ranks
                        && (matches!(square.file(), 0 | 7) || matches!(square.rank(), 0 | 7)))
            })
        {
            state.card_runtime.portals = Some(portal.squares);
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("portal-install:{}:{}", portal.squares[0], portal.squares[1]),
            });
        }
    }

    fn tick_card_statuses_after_turn(&self, state: &mut GameState, actor: Player) {
        let disarmed_ids = state
            .card_runtime
            .disarmed
            .keys()
            .copied()
            .filter(|piece_id| {
                state
                    .board
                    .find_by_id(*piece_id)
                    .is_some_and(|(_, piece)| piece.owner == actor)
            })
            .collect::<Vec<_>>();
        for piece_id in disarmed_ids {
            if let Some(remaining) = state.card_runtime.disarmed.get_mut(&piece_id) {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    state.card_runtime.disarmed.remove(&piece_id);
                }
            }
        }

        let frozen_ids = state
            .card_runtime
            .frozen
            .keys()
            .copied()
            .filter(|piece_id| {
                state
                    .board
                    .find_by_id(*piece_id)
                    .is_some_and(|(_, piece)| piece.owner == actor)
            })
            .collect::<Vec<_>>();
        for piece_id in frozen_ids {
            if let Some(status) = state.card_runtime.frozen.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                if status.remaining_turns == 0 {
                    state.card_runtime.frozen.remove(&piece_id);
                }
            }
        }

        let poison_stunned_ids = state
            .card_runtime
            .poison_stun_turns
            .keys()
            .copied()
            .filter(|piece_id| {
                state
                    .board
                    .find_by_id(*piece_id)
                    .is_some_and(|(_, piece)| piece.owner == actor)
            })
            .collect::<Vec<_>>();
        for piece_id in poison_stunned_ids {
            if let Some(remaining) = state.card_runtime.poison_stun_turns.get_mut(&piece_id) {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    state.card_runtime.poison_stun_turns.remove(&piece_id);
                }
            }
        }

        if let Some(lock) = &mut state.card_runtime.dice_locks[actor.index()] {
            lock.remaining_turns = lock.remaining_turns.saturating_sub(1);
            if lock.remaining_turns == 0 {
                state.card_runtime.dice_locks[actor.index()] = None;
            }
        }
        if state.card_runtime.initiative[actor.index()].is_some_and(|status| {
            state.turns_taken[actor.index()].saturating_sub(status.start_turn) >= status.limit
        }) {
            state.card_runtime.initiative[actor.index()] = None;
        }
        state
            .card_runtime
            .fresh_no_capture_until
            .retain(|piece_id, deadline| {
                state
                    .board
                    .find_by_id(*piece_id)
                    .is_some_and(|(_, piece)| state.turns_taken[piece.owner.index()] < *deadline)
            });
        if state.card_runtime.reposition.is_some_and(|status| {
            state
                .board
                .find_by_id(status.piece_id)
                .is_none_or(|(_, piece)| piece.owner == actor)
        }) {
            state.card_runtime.reposition = None;
        }
        let metal_ids = state
            .card_runtime
            .metal_cooldowns
            .keys()
            .copied()
            .filter(|piece_id| {
                state
                    .board
                    .find_by_id(*piece_id)
                    .is_some_and(|(_, piece)| piece.owner == actor)
            })
            .collect::<Vec<_>>();
        for piece_id in metal_ids {
            if let Some(remaining) = state.card_runtime.metal_cooldowns.get_mut(&piece_id) {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    state.card_runtime.metal_cooldowns.remove(&piece_id);
                }
            }
        }
    }

    fn resolve_no_action_loss(
        &self,
        state: &mut GameState,
        blocked_player: Player,
        events: &mut Vec<Event>,
    ) {
        if state.status != GameStatus::Ongoing
            || !self.ruleset.base_rules.immobility_loses
            || !matches!(state.phase, GamePhase::Playing)
        {
            return;
        }

        let has_move = state.actions_remaining > 0 && !self.generate_move_actions(state).is_empty();
        let mut card_actions = Vec::new();
        self.generate_card_actions(state, &mut card_actions);
        if !has_move && card_actions.is_empty() {
            state.status = GameStatus::Won {
                winner: blocked_player.opponent(),
                reason: WinReason::NoLegalActions,
            };
            self.push_game_ended(state, events);
        }
    }

    fn push_game_ended(&self, state: &GameState, events: &mut Vec<Event>) {
        if !events
            .iter()
            .any(|event| matches!(event, Event::GameEnded { .. }))
        {
            events.push(Event::GameEnded {
                status: state.status.clone(),
            });
        }
    }

    fn resolve_empty_lunchboxes_after_turn(
        &self,
        state: &mut GameState,
        moving_color: Player,
        events: &mut Vec<Event>,
    ) {
        let due_ids: Vec<_> = state
            .card_runtime
            .empty_lunchboxes
            .keys()
            .copied()
            .filter(|piece_id| {
                state
                    .board
                    .find_by_id(*piece_id)
                    .is_some_and(|(_, piece)| piece.owner == moving_color)
            })
            .collect();
        let royal_square = state
            .board
            .iter()
            .find(|(_, piece)| piece.owner == moving_color && self.is_royal_piece(state, *piece))
            .map(|(square, _)| square);

        for piece_id in due_ids {
            let Some(status) = state.card_runtime.empty_lunchboxes.get(&piece_id).copied() else {
                continue;
            };
            let Some((square, piece)) = state.board.find_by_id(piece_id) else {
                state.card_runtime.empty_lunchboxes.remove(&piece_id);
                continue;
            };
            if royal_square.is_some_and(|royal| squares_are_adjacent(square, royal)) {
                state.card_runtime.empty_lunchboxes.remove(&piece_id);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("empty-lunchbox-survived:{piece_id}:{square}"),
                });
                continue;
            }
            if state.turns_taken[moving_color.index()] < status.deadline_turn {
                continue;
            }
            state.board.set(square, None);
            self.clear_removed_piece_statuses(state, piece_id);
            state.mark_deathmatch_progress();
            events.push(Event::PieceRemoved {
                actor: status.by,
                piece,
                at: square,
                cause: EMPTY_LUNCHBOX.to_owned(),
            });
        }
    }

    fn spawn_platform(&self, state: &mut GameState, shared_turn: u32, events: &mut Vec<Event>) {
        if state.card_runtime.platform.is_none() {
            return;
        }
        // The web board uses row 0 as rank 8, so candidate order is a8..h8, a7..h7, ... a1..h1.
        let mut candidates = Vec::new();
        for rank in (0..8).rev() {
            for file in 0..8 {
                let square = Square::new(file, rank).expect("8x8 platform square");
                if state.board.get(square).is_none() {
                    candidates.push(square);
                }
            }
        }
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        let cell = index.map(|index| candidates[index]);
        let platform = state
            .card_runtime
            .platform
            .as_mut()
            .expect("platform existence was checked");
        platform.cell = cell;
        platform.spawned_at = shared_turn;
        platform.next_at = shared_turn.saturating_add(5);
        platform.triggered_piece_ids.clear();
        platform.nonce = platform.nonce.saturating_add(1);
        events.push(Event::RandomSample {
            purpose: "platform-cell".to_owned(),
            roll_53: roll,
        });
        events.push(Event::PlatformSpawned {
            square: cell,
            shared_turn,
            nonce: platform.nonce,
        });
    }

    fn tick_platform_after_turn(&self, state: &mut GameState, events: &mut Vec<Event>) {
        let shared_turn = state.shared_turn_count();
        if state
            .card_runtime
            .platform
            .as_ref()
            .is_some_and(|platform| shared_turn >= platform.next_at)
        {
            self.spawn_platform(state, shared_turn, events);
        }
    }

    fn try_start_platform_extra_move(
        &self,
        state: &mut GameState,
        landing: Square,
        piece_id: u32,
        events: &mut Vec<Event>,
    ) -> bool {
        let marked_triggered = {
            let Some(platform) = state.card_runtime.platform.as_mut() else {
                return false;
            };
            platform.cell == Some(landing)
                && platform.triggered_piece_ids.insert(piece_id)
                && state.board.find_by_id(piece_id).is_some()
        };
        if !marked_triggered {
            return false;
        }
        // Bug-compatible ordering: the piece ID remains triggered even when it has no legal move.
        if self.generate_piece_move_actions(state, piece_id).is_empty() {
            return false;
        }
        state.phase = GamePhase::ForcedMove {
            piece_id,
            source: ForcedMoveSource::Platform,
            continuation: ForcedMoveContinuation::FinishTurn,
        };
        events.push(Event::ForcedMoveStarted {
            player: state.turn,
            piece_id,
            source: PLATFORM.to_owned(),
        });
        true
    }

    fn try_start_reposition_extra_move(
        &self,
        state: &mut GameState,
        piece_id: u32,
        events: &mut Vec<Event>,
    ) -> bool {
        let should_start = state
            .card_runtime
            .reposition
            .is_some_and(|status| status.piece_id == piece_id && !status.used);
        if !should_start {
            return false;
        }
        if let Some(status) = &mut state.card_runtime.reposition {
            status.used = true;
        }
        if self.generate_piece_move_actions(state, piece_id).is_empty() {
            state.card_runtime.reposition = None;
            return false;
        }
        state.phase = GamePhase::ForcedMove {
            piece_id,
            source: ForcedMoveSource::Effect(REPOSITION.to_owned()),
            continuation: ForcedMoveContinuation::FinishTurn,
        };
        events.push(Event::ForcedMoveStarted {
            player: state.turn,
            piece_id,
            source: REPOSITION.to_owned(),
        });
        true
    }

    fn generate_card_actions(&self, state: &GameState, actions: &mut Vec<Action>) {
        self.generate_card_actions_with_opening_auto(state, actions, false);
    }

    fn generate_card_actions_with_opening_auto(
        &self,
        state: &GameState,
        actions: &mut Vec<Action>,
        include_opening_auto: bool,
    ) {
        for card in &state.cards {
            let definition = implemented_card(&card.definition_id);
            if card.owner != state.turn
                || card.charges == 0
                || (!include_opening_auto
                    && state
                        .card_runtime
                        .opening_auto_card_ids
                        .contains(&card.instance_id))
                || card
                    .available_after_turn
                    .is_some_and(|available| state.turns_taken[card.owner.index()] < available)
                || definition.is_some_and(|definition| {
                    definition.turn_exclusive && state.cards_used_this_turn[state.turn.index()] > 0
                })
            {
                continue;
            }
            match card.definition_id.as_str() {
                APPRENTICE_KNIGHTS
                | BIG_BISHOP
                | BIG_ROOK
                | CALLING_CARD
                | CHECKER
                | DEMOCRACY
                | DUTCH
                | ELEPHANT_ESCAPE
                | FAST_GROWTH
                | FIANCHETTO
                | GUARD
                | HORDE
                | IRON_MONARCH
                | KNIGHTMATE
                | LAST_STAND
                | LONDON_SYSTEM
                | MARTYRDOM
                | MERCHANT_GUILD
                | OTHERWORLD
                | PAWN_CONVERSION
                | QUEEN_CAVALRY
                | QXE1
                | SUMMON_COLOSSUS
                | THIEF
                | crate::FALSE_START
                | crate::LOCUST_SWARM => {
                    if self.opening_card_none_legal(state, &card.definition_id, state.turn) {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::None,
                        });
                    }
                }
                HOLDOUT => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn && piece.kind == PieceKind::Pawn {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                LOYALIST => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn
                            && !self.is_royal_piece(state, piece)
                            && !matches!(
                                piece.kind,
                                PieceKind::Merchant | PieceKind::BigRook | PieceKind::Colossus
                            )
                            && !state.card_runtime.loyalist_piece_ids.contains(&piece.id)
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                QUEENS_GAMBIT => {
                    let protected_pawns = state.board.iter().any(|(square, piece)| {
                        piece.owner == state.turn
                            && piece.kind == PieceKind::Pawn
                            && matches!(square.file(), 2 | 3)
                            && !state.card_runtime.shielded_piece_ids.contains(&piece.id)
                    });
                    if protected_pawns {
                        for (square, piece) in state.board.iter() {
                            if piece.owner == state.turn
                                && piece.kind == PieceKind::Queen
                                && !self.is_royal_piece(state, piece)
                            {
                                actions.push(Action::UseCard {
                                    instance_id: card.instance_id,
                                    target: ActionTarget::PieceAt(square),
                                });
                            }
                        }
                    }
                }
                PARRY => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn
                            && piece.kind != PieceKind::Monster
                            && !state.card_runtime.parry_piece_ids.contains(&piece.id)
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                EMPTY_LUNCHBOX => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn.opponent()
                            && !self.is_royal_piece(state, piece)
                            && !state.card_runtime.empty_lunchboxes.contains_key(&piece.id)
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                DISARM => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn.opponent()
                            && !self.is_royal_piece(state, piece)
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                DICE => actions.push(Action::UseCard {
                    instance_id: card.instance_id,
                    target: ActionTarget::None,
                }),
                FREEZE => {
                    let eligible = state
                        .board
                        .iter()
                        .filter(|(_, piece)| {
                            piece.owner == state.turn.opponent()
                                && !self.is_royal_piece(state, *piece)
                        })
                        .collect::<Vec<_>>();
                    if eligible.len() > 4
                        && eligible
                            .iter()
                            .any(|(_, piece)| !state.card_runtime.frozen.contains_key(&piece.id))
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::None,
                        });
                    }
                }
                REPOSITION => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn
                            && !self.is_royal_piece(state, piece)
                            && piece.kind != PieceKind::Monster
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                crate::METAL => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn
                            && !self.is_royal_piece(state, piece)
                            && self.is_ranged_piece(state, piece)
                            && !state.card_runtime.metalized_piece_ids.contains(&piece.id)
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                crate::CLOCKWORK => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn
                            && !self.is_royal_piece(state, piece)
                            && matches!(
                                piece.kind,
                                PieceKind::Knight | PieceKind::Bishop | PieceKind::Camel
                            )
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                crate::PALADIN => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn && piece.kind == PieceKind::Knight {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                crate::HEDGEHOG => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn && piece.kind == PieceKind::Queen {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                ASSASSIN => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn && piece.kind == PieceKind::Knight {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                GRASSHOPPER => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn
                            && matches!(piece.kind, PieceKind::Knight | PieceKind::Bishop)
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                CONSTITUTIONAL_MONARCHY
                | IDOL
                | JESTER
                | LOCAL_CONSCRIPTION
                | REAPER
                | WIZARD
                | SIREN
                | UNDEAD => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn && piece.kind == PieceKind::Queen {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                DRAGON
                | HERALD
                | BERSERKER
                | MAGIC_GIRL
                | SIEGE_RAM
                | SLIME
                | TRICKSTER
                | crate::PARROT
                | crate::PRINCESS
                | crate::OCTOPUS
                | crate::CAMPFIRE => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn && piece.kind == PieceKind::Rook {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                EASTERN_POLICY => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn
                            && matches!(piece.kind, PieceKind::Knight | PieceKind::Bishop)
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                KNIGHTMASTER | PEGASUS => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn && piece.kind == PieceKind::Knight {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                LOG | STANDARD_BEARER => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn && piece.kind == PieceKind::Pawn {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                MISSIONARY | ORDINATION => {
                    let bishop_count = state
                        .board
                        .iter()
                        .filter(|(_, piece)| {
                            piece.owner == state.turn && piece.kind == PieceKind::Bishop
                        })
                        .count();
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn
                            && piece.kind == PieceKind::Bishop
                            && (card.definition_id != ORDINATION || bishop_count >= 2)
                            && (card.definition_id != MISSIONARY
                                || !self.missionary_spawn_candidates(state, square).is_empty())
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                AMAZON | HOOK => {
                    let sacrifice_kind = if card.definition_id == AMAZON {
                        PieceKind::Knight
                    } else {
                        PieceKind::Rook
                    };
                    let queens = state
                        .board
                        .iter()
                        .filter(|(_, piece)| {
                            piece.owner == state.turn && piece.kind == PieceKind::Queen
                        })
                        .collect::<Vec<_>>();
                    let sacrifices = state
                        .board
                        .iter()
                        .filter(|(_, piece)| {
                            piece.owner == state.turn && piece.kind == sacrifice_kind
                        })
                        .collect::<Vec<_>>();
                    for (queen_square, _) in &queens {
                        for (sacrifice_square, _) in &sacrifices {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::Squares(vec![
                                    *queen_square,
                                    *sacrifice_square,
                                ]),
                            });
                        }
                    }
                }
                EAGLE => {
                    if state.board.iter().any(|(_, piece)| {
                        piece.owner == state.turn && piece.kind == PieceKind::Knight
                    }) {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::None,
                        });
                    }
                }
                REFORMATION => {
                    if state.board.iter().any(|(_, piece)| {
                        piece.owner == state.turn && piece.kind == PieceKind::Bishop
                    }) {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::None,
                        });
                    }
                }
                WINDMILL => {
                    let bishops = state
                        .board
                        .iter()
                        .filter(|(_, piece)| {
                            piece.owner == state.turn && piece.kind == PieceKind::Bishop
                        })
                        .collect::<Vec<_>>();
                    let rooks = state
                        .board
                        .iter()
                        .filter(|(_, piece)| {
                            piece.owner == state.turn && piece.kind == PieceKind::Rook
                        })
                        .collect::<Vec<_>>();
                    for (bishop_square, _) in &bishops {
                        for (rook_square, _) in &rooks {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::Squares(vec![*bishop_square, *rook_square]),
                            });
                        }
                    }
                }
                JUDGMENT => {
                    let maximum = state
                        .board
                        .iter()
                        .filter(|(_, piece)| {
                            piece.capture_count >= 2 && !self.is_royal_piece(state, *piece)
                        })
                        .map(|(_, piece)| piece.capture_count)
                        .max()
                        .unwrap_or(0);
                    if maximum >= 2 {
                        for (square, piece) in state.board.iter() {
                            if piece.capture_count == maximum && !self.is_royal_piece(state, piece)
                            {
                                actions.push(Action::UseCard {
                                    instance_id: card.instance_id,
                                    target: ActionTarget::PieceAt(square),
                                });
                            }
                        }
                    }
                }
                CLEANUP_SACRIFICE => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner != state.turn || self.is_royal_piece(state, piece) {
                            continue;
                        }
                        let has_match = state.board.iter().any(|(_, candidate)| {
                            candidate.owner == state.turn.opponent()
                                && candidate.kind == piece.kind
                                && !self.is_royal_piece(state, candidate)
                        });
                        if has_match {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                _ => {
                    self.generate_middle_card_actions(state, card, actions);
                    self.generate_end_card_actions(state, card, card.owner, actions);
                }
            }
        }
    }

    fn generate_move_actions(&self, state: &GameState) -> Vec<Action> {
        let mut forced_checker_captures = Vec::new();
        for (from, piece) in state.board.iter().filter(|(_, piece)| {
            piece.owner == state.turn
                && matches!(piece.kind, PieceKind::Checker | PieceKind::CheckerKing)
        }) {
            let mut candidates = Vec::new();
            self.generate_checker_moves(state, from, piece, &mut candidates);
            forced_checker_captures.extend(candidates.into_iter().filter(|action| {
                matches!(action, Action::Move(movement) if movement.from.file().abs_diff(movement.to.file()) == 2)
            }));
        }
        if !forced_checker_captures.is_empty() {
            forced_checker_captures.sort();
            forced_checker_captures.dedup();
            return forced_checker_captures;
        }
        let mut actions = Vec::new();
        for (from, piece) in state.board.iter() {
            if piece.owner != state.turn {
                continue;
            }
            self.generate_moves_for_piece(state, from, piece, &mut actions);
            if let Some(shadow_square) = state.card_runtime.quantum_positions.get(&piece.id) {
                self.generate_moves_for_piece(state, *shadow_square, piece, &mut actions);
            }
        }
        let scarecrow_captures = actions
            .iter()
            .filter(|action| {
                matches!(action, Action::Move(movement) if state.board.get(movement.to).is_some_and(|target| {
                    target.owner != state.turn && target.kind == PieceKind::Scarecrow
                }))
            })
            .cloned()
            .collect::<Vec<_>>();
        if !scarecrow_captures.is_empty() {
            return scarecrow_captures;
        }
        actions.sort();
        actions.dedup();
        actions
    }

    fn generate_piece_move_actions(&self, state: &GameState, piece_id: u32) -> Vec<Action> {
        let mut actions = Vec::new();
        if let Some((from, piece)) = state.board.find_by_id(piece_id)
            && piece.owner == state.turn
        {
            self.generate_moves_for_piece(state, from, piece, &mut actions);
            if let Some(shadow_square) = state.card_runtime.quantum_positions.get(&piece_id) {
                self.generate_moves_for_piece(state, *shadow_square, piece, &mut actions);
            }
        }
        actions.sort();
        actions.dedup();
        actions
    }

    fn generate_moves_for_piece(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        all_actions: &mut Vec<Action>,
    ) {
        if state.card_runtime.zugzwang[piece.owner.index()] && !self.is_royal_piece(state, piece) {
            return;
        }
        if state.card_runtime.frozen.contains_key(&piece.id)
            || state
                .card_runtime
                .metal_cooldowns
                .get(&piece.id)
                .is_some_and(|remaining| *remaining > 0)
            || state.card_runtime.poison_stun_turns.contains_key(&piece.id)
            || state.card_runtime.idol_rest_piece_ids.contains(&piece.id)
            || state.card_runtime.dice_locks[piece.owner.index()]
                .is_some_and(|lock| lock.remaining_turns > 0 && lock.kind == piece.kind)
        {
            return;
        }
        if self.middle_piece_is_exhausted(state, piece) {
            return;
        }
        // Live getLegalMoves gates Clockwork before all movement providers and
        // before materializing quantum shadows. Only physical board allies count.
        if self.piece_has_ability(state, piece, PieceKind::Clockwork)
            && !(-1..=1).any(|df| {
                (-1..=1).any(|dr| {
                    (df != 0 || dr != 0)
                        && from
                            .offset(df, dr)
                            .and_then(|square| state.board.get(square))
                            .is_some_and(|neighbor| {
                                neighbor.owner == piece.owner && neighbor.id != piece.id
                            })
                })
            })
        {
            return;
        }
        // A trickster remains a trickster for identity, targeting, death and reroll semantics,
        // but every movement/ability query for the turn is dispatched as its sampled type.
        // The 0818 pool excludes Trickster and Log, so this recursive dispatch cannot loop and
        // also reuses the complete movement implementation for all 43 eligible types.
        if piece.kind == PieceKind::Trickster {
            let mut ability_piece = piece;
            ability_piece.kind = self.piece_ability_kind(state, piece);
            self.generate_moves_for_piece(state, from, ability_piece, all_actions);
            return;
        }
        let mut generated = Vec::new();
        let actions = &mut generated;
        if state.card_runtime.socialism[piece.owner.index()] > 0
            && !self.piece_has_ability(state, piece, PieceKind::Slime)
            && !self.is_royal_piece(state, piece)
        {
            self.generate_socialism_moves(state, from, piece, actions);
            self.filter_middle_moves(state, from, piece, actions);
            self.filter_metal_moves(state, piece, from, actions);
            all_actions.extend(generated);
            return;
        }
        match piece.kind {
            PieceKind::Pawn | PieceKind::Squire => {
                self.generate_pawn_moves(state, from, piece, actions)
            }
            PieceKind::Knight => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-2, -1),
                    (-2, 1),
                    (-1, -2),
                    (-1, 2),
                    (1, -2),
                    (1, 2),
                    (2, -1),
                    (2, 1),
                ],
                actions,
            ),
            PieceKind::Bishop => self.generate_slider_moves(
                state,
                from,
                piece,
                // Reversal: an original board bishop moves as a rook for the armed turn.
                if state.card_runtime.reversal[piece.owner.index()]
                    && state
                        .board
                        .get(from)
                        .is_some_and(|origin| origin.kind == PieceKind::Bishop)
                {
                    &[(-1, 0), (1, 0), (0, -1), (0, 1)]
                } else {
                    &[(-1, -1), (-1, 1), (1, -1), (1, 1)]
                },
                actions,
            ),
            PieceKind::Rook => self.generate_slider_moves(
                state,
                from,
                piece,
                // Reversal: an original board rook moves as a bishop for the armed turn.
                if state.card_runtime.reversal[piece.owner.index()]
                    && state
                        .board
                        .get(from)
                        .is_some_and(|origin| origin.kind == PieceKind::Rook)
                {
                    &[(-1, -1), (-1, 1), (1, -1), (1, 1)]
                } else {
                    &[(-1, 0), (1, 0), (0, -1), (0, 1)]
                },
                actions,
            ),
            PieceKind::Queen => self.generate_slider_moves(
                state,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ],
                actions,
            ),
            PieceKind::King => {
                self.generate_leaper_moves(
                    state,
                    from,
                    piece,
                    &[
                        (-1, -1),
                        (-1, 0),
                        (-1, 1),
                        (0, -1),
                        (0, 1),
                        (1, -1),
                        (1, 0),
                        (1, 1),
                    ],
                    actions,
                );
                self.generate_castling_moves(state, from, piece, actions);
                if state.card_runtime.killer_king[piece.owner.index()] {
                    self.generate_killer_king_rook_captures(state, from, piece, actions);
                }
                if state.card_runtime.hill_king[piece.owner.index()] && is_center_two_by_two(from) {
                    self.generate_slider_moves(
                        state,
                        from,
                        piece,
                        &[
                            (-1, -1),
                            (-1, 1),
                            (1, -1),
                            (1, 1),
                            (-1, 0),
                            (1, 0),
                            (0, -1),
                            (0, 1),
                        ],
                        actions,
                    );
                    self.generate_knight_moves(state, from, piece, actions);
                }
            }
            PieceKind::Assassin => {
                self.generate_leaper_moves(
                    state,
                    from,
                    piece,
                    &[
                        (-2, -1),
                        (-2, 1),
                        (-1, -2),
                        (-1, 2),
                        (1, -2),
                        (1, 2),
                        (2, -1),
                        (2, 1),
                    ],
                    actions,
                );
                self.generate_assassin_royal_captures(state, from, piece, actions);
            }
            PieceKind::Grasshopper => {
                self.generate_grasshopper_moves(state, from, piece, actions);
            }
            PieceKind::Amazon => {
                self.generate_slider_moves(
                    state,
                    from,
                    piece,
                    &[
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                    ],
                    actions,
                );
                self.generate_knight_moves(state, from, piece, actions);
            }
            PieceKind::PrimeMinister => {
                self.generate_prime_minister_moves(state, from, piece, actions);
            }
            PieceKind::Dragon => {
                self.generate_knight_moves(state, from, piece, actions);
                for (target, target_piece) in state.board.iter() {
                    if target != from
                        && target_piece.owner == piece.owner
                        && !self.piece_has_ability(state, target_piece, PieceKind::Slime)
                    {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                    }
                }
            }
            PieceKind::Cannon => self.generate_cannon_moves(state, from, piece, actions),
            PieceKind::Herald => self.generate_herald_moves(state, from, piece, actions),
            PieceKind::Knightmaster => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
                actions,
            ),
            PieceKind::Unicorn => self.generate_unicorn_moves(state, from, piece, actions),
            PieceKind::Hook => self.generate_hook_moves(state, from, piece, actions),
            PieceKind::Idol => self.generate_slider_moves(
                state,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ],
                actions,
            ),
            PieceKind::Recruiter | PieceKind::Wizard => {
                self.generate_empty_leaper_moves(
                    state,
                    from,
                    &[
                        (-1, -1),
                        (-1, 0),
                        (-1, 1),
                        (0, -1),
                        (0, 1),
                        (1, -1),
                        (1, 0),
                        (1, 1),
                    ],
                    actions,
                );
            }
            PieceKind::Jester => self.generate_slider_moves(
                state,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ],
                actions,
            ),
            PieceKind::Reaper => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ],
                actions,
            ),
            PieceKind::Alibaba => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-2, -2),
                    (-2, 0),
                    (-2, 2),
                    (0, -2),
                    (0, 2),
                    (2, -2),
                    (2, 0),
                    (2, 2),
                ],
                actions,
            ),
            PieceKind::Log => {
                for &(file_delta, rank_delta) in &[
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ] {
                    if let Some(target) = from.offset(file_delta, rank_delta) {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                    }
                }
            }
            PieceKind::Missionary => self.generate_missionary_moves(state, from, piece, actions),
            PieceKind::Cardinal => self.generate_cardinal_moves(state, from, piece, actions),
            PieceKind::Protestant => {
                self.generate_protestant_moves(state, from, piece, actions);
            }
            PieceKind::StandardBearer => {
                self.generate_pawn_moves(state, from, piece, actions);
            }
            PieceKind::Windmill => match state
                .card_runtime
                .windmill_modes
                .get(&piece.id)
                .copied()
                .unwrap_or(WindmillMode::Bishop)
            {
                WindmillMode::Bishop => self.generate_slider_moves(
                    state,
                    from,
                    piece,
                    &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
                    actions,
                ),
                WindmillMode::Rook => self.generate_slider_moves(
                    state,
                    from,
                    piece,
                    &[(-1, 0), (1, 0), (0, -1), (0, 1)],
                    actions,
                ),
            },
            PieceKind::Alfil => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[(-2, -2), (-2, 2), (2, -2), (2, 2)],
                actions,
            ),
            PieceKind::Guard => self.generate_empty_leaper_moves(
                state,
                from,
                &[
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ],
                actions,
            ),
            PieceKind::Fanatic => self.generate_fanatic_moves(state, from, piece, actions),
            PieceKind::Checker | PieceKind::CheckerKing => {
                self.generate_checker_moves(state, from, piece, actions)
            }
            PieceKind::RoyalKnight => self.generate_knight_moves(state, from, piece, actions),
            PieceKind::Man => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ],
                actions,
            ),
            PieceKind::BigRook => self.generate_big_rook_moves(state, from, piece, actions),
            PieceKind::BigBishop => self.generate_big_bishop_moves(state, from, piece, actions),
            PieceKind::Brutus => self.generate_brutus_moves(state, from, piece, actions),
            PieceKind::Thief => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-3, 0),
                    (-2, 0),
                    (-1, 0),
                    (1, 0),
                    (2, 0),
                    (3, 0),
                    (0, -3),
                    (0, -2),
                    (0, -1),
                    (0, 1),
                    (0, 2),
                    (0, 3),
                ],
                actions,
            ),
            PieceKind::Colossus => self.generate_large_step_moves(state, from, piece, actions),
            PieceKind::Merchant => {}
            PieceKind::DarkWizard => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ],
                actions,
            ),
            PieceKind::Camel => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-3, -1),
                    (-3, 1),
                    (-1, -3),
                    (-1, 3),
                    (1, -3),
                    (1, 3),
                    (3, -1),
                    (3, 1),
                ],
                actions,
            ),
            PieceKind::Ferz => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
                actions,
            ),
            PieceKind::Bat => {
                for &(file_delta, rank_delta) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let mut cursor = from;
                    for _ in 0..2 {
                        let Some(target) = cursor.offset(file_delta, rank_delta) else {
                            break;
                        };
                        match self.board_piece_at(state, target) {
                            None => actions.push(Action::Move(MoveAction::new(from, target, None))),
                            Some(target_piece) if self.can_capture(state, piece, target_piece) => {
                                actions.push(Action::Move(MoveAction::new(from, target, None)));
                                break;
                            }
                            Some(_) => break,
                        }
                        cursor = target;
                    }
                }
            }
            PieceKind::Lobster => {
                let direction = if piece.owner == Player::White { 1 } else { -1 };
                self.generate_leaper_moves(
                    state,
                    from,
                    piece,
                    &[(-1, direction), (0, direction), (1, direction)],
                    actions,
                );
            }
            PieceKind::Bear => self.generate_slider_moves(
                state,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ],
                actions,
            ),
            PieceKind::Vip => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ],
                actions,
            ),
            PieceKind::SiegeRam => self.generate_siege_ram_moves(state, from, piece, actions),
            PieceKind::MagicGirl => {
                if state.card_runtime.magic_girl_surge[piece.owner.index()] {
                    self.generate_amazon_moves(state, from, piece, actions);
                } else {
                    self.generate_king_step_moves(state, from, piece, actions);
                }
            }
            PieceKind::Berserker => {
                let allied = state
                    .board
                    .iter()
                    .filter(|(_, candidate)| candidate.owner == piece.owner)
                    .count();
                if allied <= 5 {
                    self.generate_amazon_moves(state, from, piece, actions);
                } else {
                    self.generate_king_step_moves(state, from, piece, actions);
                    if allied <= 9 {
                        self.generate_slider_moves(
                            state,
                            from,
                            piece,
                            &[(-1, 0), (1, 0), (0, -1), (0, 1)],
                            actions,
                        );
                    }
                }
            }
            PieceKind::Slime => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[(-3, 0), (3, 0), (0, -3), (0, 3)],
                actions,
            ),
            PieceKind::Clockwork => self.generate_slider_moves(
                state,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ],
                actions,
            ),
            // Parrot base movement mirrors the owner's remembered last move
            // (live parrotBaseMoves, JS 4205-4320).
            PieceKind::Parrot => self.generate_parrot_memory_moves(state, from, piece, actions),
            // Paladin (live L82172): extended knight jumps.
            PieceKind::Paladin => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-2, -1),
                    (-2, 1),
                    (-1, -2),
                    (-1, 2),
                    (1, -2),
                    (1, 2),
                    (2, -1),
                    (2, 1),
                ],
                actions,
            ),
            // Princess (live L1275-1277): full queen rays only while no own queen
            // exists on the board; otherwise one-step diagonal jumps.
            PieceKind::Princess
                if !state.board.iter().any(|(_, existing)| {
                    existing.owner == piece.owner && existing.kind == PieceKind::Queen
                }) =>
            {
                self.generate_slider_moves(
                    state,
                    from,
                    piece,
                    &[
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                    ],
                    actions,
                )
            }
            PieceKind::Princess => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
                actions,
            ),
            PieceKind::Octopus => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ],
                actions,
            ),
            PieceKind::Hedgehog => self.generate_king_step_moves(state, from, piece, actions),
            PieceKind::Campfire => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[(-1, 0), (1, 0), (0, -1), (0, 1)],
                actions,
            ),
            PieceKind::Siren | PieceKind::Undead => {
                self.generate_king_step_moves(state, from, piece, actions)
            }
            PieceKind::Trickster => unreachable!("trickster dispatches through its ability kind"),
            PieceKind::Monster
            | PieceKind::Scarecrow
            | PieceKind::BabyBear
            | PieceKind::Crown
            | PieceKind::Football => {}
        }
        self.generate_loyalist_moves(state, from, piece, actions);
        self.augment_middle_moves(state, from, piece, actions);
        if state
            .card_runtime
            .locust_swarm_piece_ids
            .contains(&piece.id)
        {
            self.generate_locust_swarm_moves(state, from, piece, actions);
        }
        self.filter_middle_moves(state, from, piece, actions);
        if state.card_runtime.relay[piece.owner.index()]
            && !self.piece_has_ability(state, piece, PieceKind::Slime)
        {
            for (square, ally) in state.board.iter() {
                if ally.owner == piece.owner
                    && ally.id != piece.id
                    && !self.piece_has_ability(state, ally, PieceKind::Slime)
                    && (square.file() == from.file() || square.rank() == from.rank())
                    && !state.card_runtime.large_pieces.contains_key(&ally.id)
                    && !state.card_runtime.large_pieces.contains_key(&piece.id)
                    && !self.paladin_radiance_blocks_move(
                        state,
                        piece,
                        MoveAction::new(from, square, None),
                        false,
                    )
                {
                    generated.push(Action::Move(MoveAction::new(from, square, None)));
                }
            }
        }
        // Symmetry: alongside normal moves, a mirror move across the vertical center
        // line (file 7 - file, same rank) ignores intermediate blockers. Own-piece
        // destinations are rejected; enemy destinations become captures
        // (live symmetryMoves JS:82366 with normal move generation, 101339-101341).
        if state.card_runtime.symmetry[piece.owner.index()] {
            let mirror_file = 7 - from.file();
            if let Some(mirror) = Square::new(mirror_file, from.rank())
                && mirror != from
                && !state.card_runtime.large_pieces.contains_key(&piece.id)
                && !self.paladin_radiance_blocks_move(
                    state,
                    piece,
                    MoveAction::new(from, mirror, None),
                    false,
                )
            {
                match state.board.get(mirror) {
                    None => generated.push(Action::Move(MoveAction::new(from, mirror, None))),
                    Some(target_piece) if target_piece.owner != piece.owner => {
                        if self.can_capture(state, piece, target_piece) {
                            generated.push(Action::Move(MoveAction::new(from, mirror, None)));
                        }
                    }
                    Some(_) => {}
                }
            }
        }
        self.filter_metal_moves(state, piece, from, &mut generated);
        all_actions.extend(generated);
    }

    fn filter_metal_moves(
        &self,
        state: &GameState,
        piece: Piece,
        from: Square,
        actions: &mut Vec<Action>,
    ) {
        if !state.card_runtime.metalized_piece_ids.contains(&piece.id) {
            return;
        }
        actions.retain(|action| {
            matches!(action, Action::Move(movement)
                if movement.from == from
                    && from.file().abs_diff(movement.to.file()) <= 1
                    && from.rank().abs_diff(movement.to.rank()) <= 1)
        });
    }

    fn generate_king_step_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        self.generate_leaper_moves(
            state,
            from,
            piece,
            &[
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ],
            actions,
        );
    }

    fn generate_amazon_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        self.generate_slider_moves(
            state,
            from,
            piece,
            &[
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ],
            actions,
        );
        self.generate_knight_moves(state, from, piece, actions);
    }

    fn generate_siege_ram_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for (file_delta, rank_delta) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let mut cursor = from;
            for _ in 0..2 {
                let Some(target) = cursor.offset(file_delta, rank_delta) else {
                    break;
                };
                let blocked = state.board.get(target).is_some_and(|victim| {
                    self.is_royal_piece(state, victim) && victim.owner == piece.owner
                        || (victim.owner != piece.owner && !self.can_capture(state, piece, victim))
                });
                if blocked {
                    break;
                }
                actions.push(Action::Move(MoveAction::new(from, target, None)));
                cursor = target;
            }
        }
    }

    fn generate_pawn_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        if piece.kind == PieceKind::Pawn {
            let adjacent_knightmasters = state
                .board
                .iter()
                .filter(|(square, candidate)| {
                    candidate.owner == piece.owner
                        && candidate.kind == PieceKind::Knightmaster
                        && square.file().abs_diff(from.file()) <= 1
                        && square.rank().abs_diff(from.rank()) <= 1
                })
                .map(|(_, candidate)| candidate)
                .collect::<Vec<_>>();
            if !adjacent_knightmasters.is_empty() {
                let capture_ready = adjacent_knightmasters
                    .iter()
                    .any(|master| self.capture_status_allows(state, *master));
                for &(file_delta, rank_delta) in KNIGHT_OFFSETS {
                    let Some(target) = from.offset(file_delta, rank_delta) else {
                        continue;
                    };
                    match state.board.get(target) {
                        None => actions.push(Action::Move(MoveAction::new(from, target, None))),
                        Some(target_piece)
                            if capture_ready
                                && self.can_capture_ignoring_fresh(state, piece, target_piece)
                                && self.capture_status_allows_ignoring_fresh(state, piece) =>
                        {
                            actions.push(Action::Move(MoveAction::new(from, target, None)));
                        }
                        Some(_) => {}
                    }
                }
                return;
            }
        }
        let (direction, double_step_ranks, promotion_rank) = match piece.owner {
            Player::White => (
                1,
                [1, 0],
                if state.card_runtime.fast_growth[piece.owner.index()] {
                    4
                } else if state.card_runtime.early_promotion[piece.owner.index()] {
                    5
                } else {
                    7
                },
            ),
            Player::Black => (
                -1,
                [6, 7],
                if state.card_runtime.fast_growth[piece.owner.index()] {
                    3
                } else if state.card_runtime.early_promotion[piece.owner.index()] {
                    2
                } else {
                    0
                },
            ),
        };

        if piece.kind == PieceKind::Pawn && state.card_runtime.pawn_conversion[piece.owner.index()]
        {
            for file_delta in [-1, 1] {
                if let Some(one_step) = from.offset(file_delta, direction)
                    && self.board_piece_at(state, one_step).is_none()
                {
                    self.push_card_pawn_move(state, piece, from, one_step, promotion_rank, actions);
                    if !state.card_runtime.moved_piece_ids.contains(&piece.id)
                        && double_step_ranks.contains(&from.rank())
                        && let Some(two_steps) = from.offset(file_delta * 2, direction * 2)
                        && self.board_piece_at(state, two_steps).is_none()
                    {
                        actions.push(Action::Move(MoveAction::new(from, two_steps, None)));
                        if state.card_runtime.pawn_sprint[piece.owner.index()]
                            && let Some(three_steps) = from.offset(file_delta * 3, direction * 3)
                            && self.board_piece_at(state, three_steps).is_none()
                        {
                            self.push_card_pawn_move(
                                state,
                                piece,
                                from,
                                three_steps,
                                promotion_rank,
                                actions,
                            );
                        }
                    }
                }
            }
            if let Some(target) = from.offset(0, direction)
                && self
                    .board_piece_at(state, target)
                    .is_some_and(|target_piece| self.can_capture(state, piece, target_piece))
            {
                self.push_card_pawn_move(state, piece, from, target, promotion_rank, actions);
            }
            return;
        }

        // A concealed enemy piece (stealth, non-royal) is invisible to this mover: the pawn may
        // step onto or through its square (live addPawnDirectionalMoves.movementTarget).
        let forward_open = |square: Square| {
            self.board_piece_at(state, square).is_none_or(|target| {
                target.owner != piece.owner
                    && state.card_runtime.stealth_piece_ids.contains(&target.id)
                    && !self.is_royal_piece(state, target)
            })
        };
        // Proficiency (live addInternalPawnMoves): a pawn on the promotion-adjacent rank may
        // capture straight ahead, which standard pawn rules forbid.
        if state.card_runtime.proficiency[piece.owner.index()]
            && from.rank()
                == match piece.owner {
                    Player::White => 6,
                    Player::Black => 1,
                }
            && let Some(promotion_capture) = from.offset(0, direction)
            && let Some(target_piece) = self.board_piece_at(state, promotion_capture)
            && self.can_capture(state, piece, target_piece)
        {
            self.push_card_pawn_move(
                state,
                piece,
                from,
                promotion_capture,
                promotion_rank,
                actions,
            );
        }
        if let Some(one_step) = from.offset(0, direction)
            && forward_open(one_step)
        {
            self.push_card_pawn_move(state, piece, from, one_step, promotion_rank, actions);
            if !state.card_runtime.moved_piece_ids.contains(&piece.id)
                && double_step_ranks.contains(&from.rank())
                && let Some(two_steps) = from.offset(0, direction * 2)
                && forward_open(two_steps)
            {
                actions.push(Action::Move(MoveAction::new(from, two_steps, None)));
                if state.card_runtime.pawn_sprint[piece.owner.index()]
                    && let Some(three_steps) = from.offset(0, direction * 3)
                    && forward_open(three_steps)
                {
                    self.push_card_pawn_move(
                        state,
                        piece,
                        from,
                        three_steps,
                        promotion_rank,
                        actions,
                    );
                }
            }
        }

        for file_delta in [-1, 1] {
            let Some(target) = from.offset(file_delta, direction) else {
                continue;
            };
            let regular_capture = self
                .board_piece_at(state, target)
                .is_some_and(|target_piece| {
                    if piece.kind == PieceKind::StandardBearer {
                        self.can_capture_ignoring_fresh(state, piece, target_piece)
                            && self.capture_status_allows_ignoring_fresh(state, piece)
                    } else {
                        self.can_capture(state, piece, target_piece)
                    }
                });
            let en_passant_capture = state.en_passant.is_some_and(|en_passant| {
                en_passant.target == target
                    && en_passant.pawn_owner != piece.owner
                    && en_passant.captured.rank() == from.rank()
                    && en_passant.captured.file() == target.file()
                    && state
                        .board
                        .get(en_passant.captured)
                        .is_some_and(|captured| {
                            captured.kind == PieceKind::Pawn
                                && captured.owner == en_passant.pawn_owner
                                && if piece.kind == PieceKind::StandardBearer {
                                    self.can_capture_ignoring_fresh(state, piece, captured)
                                        && self.capture_status_allows_ignoring_fresh(state, piece)
                                } else {
                                    self.can_capture(state, piece, captured)
                                }
                        })
            });
            if regular_capture || en_passant_capture {
                self.push_card_pawn_move(state, piece, from, target, promotion_rank, actions);
            }
        }

        if piece.kind == PieceKind::Pawn && state.card_runtime.vanguard[piece.owner.index()] {
            let own_pawns = state
                .board
                .iter()
                .filter(|(_, candidate)| {
                    candidate.owner == piece.owner && candidate.kind == PieceKind::Pawn
                })
                .collect::<Vec<_>>();
            let unique_on_rank = own_pawns
                .iter()
                .filter(|(square, _)| square.rank() == from.rank())
                .count()
                == 1;
            let foremost = own_pawns.iter().all(|(square, _)| match piece.owner {
                Player::White => square.rank() <= from.rank(),
                Player::Black => square.rank() >= from.rank(),
            });
            if unique_on_rank && foremost {
                for file_delta in [-1, 1] {
                    if let Some(target) = from.offset(file_delta, direction)
                        && state.board.get(target).is_none()
                    {
                        self.push_card_pawn_move(
                            state,
                            piece,
                            from,
                            target,
                            promotion_rank,
                            actions,
                        );
                    }
                }
            }
        }

        let same_rank_bearers = state
            .board
            .iter()
            .filter(|(square, candidate)| {
                square.rank() == from.rank()
                    && candidate.owner == piece.owner
                    && candidate.kind == PieceKind::StandardBearer
            })
            .map(|(_, candidate)| candidate)
            .collect::<Vec<_>>();
        if piece.kind == PieceKind::StandardBearer || !same_rank_bearers.is_empty() {
            let capture_ready = same_rank_bearers
                .iter()
                .any(|bearer| self.capture_status_allows(state, *bearer));
            for file_delta in [-1, 1] {
                let Some(target) = from.offset(file_delta, 0) else {
                    continue;
                };
                match state.board.get(target) {
                    None => actions.push(Action::Move(MoveAction::new(from, target, None))),
                    Some(target_piece)
                        if capture_ready
                            && self.can_capture_ignoring_fresh(state, piece, target_piece)
                            && self.capture_status_allows_ignoring_fresh(state, piece) =>
                    {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                    }
                    Some(_) => {}
                }
            }
        }
    }

    fn push_card_pawn_move(
        &self,
        state: &GameState,
        piece: Piece,
        from: Square,
        to: Square,
        promotion_rank: u8,
        actions: &mut Vec<Action>,
    ) {
        if piece.kind == PieceKind::Squire || to.rank() != promotion_rank {
            actions.push(Action::Move(MoveAction::new(from, to, None)));
            return;
        }
        // Mutation (live JS:4136 passive, 95714+ adds "monster" to promotion choices).
        let mut promotions: Vec<PieceKind> = if state.card_runtime.fast_growth[piece.owner.index()]
        {
            vec![PieceKind::Knight, PieceKind::Bishop]
        } else {
            vec![
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
            ]
        };
        if state.card_runtime.mutation[piece.owner.index()] {
            promotions.push(PieceKind::Monster);
        }
        let promotions = &promotions[..];
        for &promotion in promotions {
            actions.push(Action::Move(MoveAction::new(from, to, Some(promotion))));
        }
    }

    fn generate_fanatic_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let direction = if piece.owner == Player::White { 1 } else { -1 };
        for distance in 1..=2 {
            let Some(target) = from.offset(0, direction * distance) else {
                break;
            };
            match self.board_piece_at(state, target) {
                None => actions.push(Action::Move(MoveAction::new(from, target, None))),
                Some(target_piece) if self.can_capture(state, piece, target_piece) => {
                    actions.push(Action::Move(MoveAction::new(from, target, None)));
                    break;
                }
                Some(_) => break,
            }
        }
    }

    fn checker_jump_capture(
        &self,
        state: &GameState,
        from: Square,
        to: Square,
        piece: Piece,
    ) -> Option<(Square, Piece)> {
        if !matches!(piece.kind, PieceKind::Checker | PieceKind::CheckerKing)
            || from.file().abs_diff(to.file()) != 2
            || from.rank().abs_diff(to.rank()) != 2
        {
            return None;
        }

        let rank_delta = i16::from(to.rank()) - i16::from(from.rank());
        let moves_in_allowed_direction = piece.kind == PieceKind::CheckerKing
            || (piece.owner == Player::White && rank_delta == 2)
            || (piece.owner == Player::Black && rank_delta == -2);
        if !moves_in_allowed_direction || self.board_piece_at(state, to).is_some() {
            return None;
        }

        let mid = Square::new((from.file() + to.file()) / 2, (from.rank() + to.rank()) / 2)?;
        self.board_piece_at(state, mid)
            .filter(|target| self.can_capture(state, piece, *target))
            .map(|captured| (mid, captured))
    }

    fn generate_checker_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let directions: &[(i8, i8)] = if piece.kind == PieceKind::CheckerKing {
            &[(-1, -1), (-1, 1), (1, -1), (1, 1)]
        } else if piece.owner == Player::White {
            &[(1, -1), (1, 1)]
        } else {
            &[(-1, -1), (-1, 1)]
        };
        let mut captures = Vec::new();
        for &(rank_delta, file_delta) in directions {
            let Some(landing) = from.offset(file_delta * 2, rank_delta * 2) else {
                continue;
            };
            if self
                .checker_jump_capture(state, from, landing, piece)
                .is_some()
            {
                captures.push(Action::Move(MoveAction::new(from, landing, None)));
            }
        }
        if !captures.is_empty() {
            actions.extend(captures);
            return;
        }
        for &(rank_delta, file_delta) in directions {
            if let Some(target) = from.offset(file_delta, rank_delta)
                && self.board_piece_at(state, target).is_none()
            {
                actions.push(Action::Move(MoveAction::new(from, target, None)));
            }
        }
    }

    fn generate_loyalist_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        if !state.card_runtime.loyalist_piece_ids.contains(&piece.id)
            || self.piece_has_ability(state, piece, PieceKind::Slime)
        {
            return;
        }
        let anchors = state
            .board
            .iter()
            .filter(|(_, candidate)| {
                candidate.owner == piece.owner
                    && (self.is_royal_piece(state, *candidate)
                        || candidate.kind == PieceKind::Merchant)
            })
            .map(|(square, _)| square)
            .collect::<Vec<_>>();
        for anchor in anchors {
            for rank_delta in -1..=1 {
                for file_delta in -1..=1 {
                    if (file_delta != 0 || rank_delta != 0)
                        && let Some(target) = anchor.offset(file_delta, rank_delta)
                        && self.board_piece_at(state, target).is_none()
                    {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                    }
                }
            }
        }
    }

    fn generate_large_step_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
            if let Some(anchor) = from.offset(file_delta, rank_delta)
                && self.large_landing_is_legal(state, piece, anchor, usize::MAX)
            {
                actions.push(Action::Move(MoveAction::new(from, anchor, None)));
            }
        }
    }

    fn generate_big_rook_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let mut cursor = from;
            while let Some(anchor) = cursor.offset(file_delta, rank_delta) {
                if !self.large_landing_is_legal(state, piece, anchor, 2) {
                    break;
                }
                actions.push(Action::Move(MoveAction::new(from, anchor, None)));
                if !self.large_landing_targets(state, piece, anchor).is_empty() {
                    break;
                }
                cursor = anchor;
            }
        }
    }

    fn generate_big_bishop_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in &[(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            let mut cursor = from;
            while let Some(anchor) = cursor.offset(file_delta, rank_delta) {
                if !self.large_landing_is_legal(state, piece, anchor, 2) {
                    break;
                }
                actions.push(Action::Move(MoveAction::new(from, anchor, None)));
                if !self.large_landing_targets(state, piece, anchor).is_empty() {
                    break;
                }
                cursor = anchor;
            }
        }
    }

    fn generate_brutus_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        self.generate_slider_moves(
            state,
            from,
            piece,
            &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            actions,
        );
        for &(file_delta, rank_delta) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let mut cursor = from;
            while let Some(target) = cursor.offset(file_delta, rank_delta) {
                match self.board_piece_at(state, target) {
                    None => cursor = target,
                    Some(target_piece) => {
                        if target_piece.owner == piece.owner
                            && self.is_royal_piece(state, target_piece)
                        {
                            actions.push(Action::Move(MoveAction::new(from, target, None)));
                        }
                        break;
                    }
                }
            }
        }
    }

    fn large_landing_is_legal(
        &self,
        state: &GameState,
        piece: Piece,
        anchor: Square,
        max_captures: usize,
    ) -> bool {
        let cells = large_piece_cells(anchor);
        if cells.len() != 4 {
            return false;
        }
        let targets = self.large_landing_targets(state, piece, anchor);
        targets.len() <= max_captures
            && targets.iter().all(|(_, target)| {
                if piece.kind == PieceKind::BigRook && target.owner == piece.owner {
                    !self.is_royal_piece(state, *target)
                } else {
                    target.owner != piece.owner && self.can_capture(state, piece, *target)
                }
            })
    }

    fn large_landing_targets(
        &self,
        state: &GameState,
        piece: Piece,
        anchor: Square,
    ) -> Vec<(Square, Piece)> {
        let mut seen = BTreeSet::new();
        large_piece_cells(anchor)
            .into_iter()
            .filter_map(|cell| {
                self.board_piece_entry_at(state, cell)
                    .filter(|(_, target)| target.id != piece.id && seen.insert(target.id))
            })
            .collect()
    }

    fn board_piece_at(&self, state: &GameState, square: Square) -> Option<Piece> {
        self.board_piece_entry_at(state, square)
            .map(|(_, piece)| piece)
    }

    fn board_piece_entry_at(&self, state: &GameState, square: Square) -> Option<(Square, Piece)> {
        state
            .board
            .get(square)
            .map(|piece| (square, piece))
            .or_else(|| {
                state
                    .card_runtime
                    .large_pieces
                    .iter()
                    .find_map(|(piece_id, status)| {
                        large_piece_cells(status.anchor)
                            .contains(&square)
                            .then(|| state.board.find_by_id(*piece_id))
                            .flatten()
                    })
            })
            .or_else(|| {
                state
                    .card_runtime
                    .quantum_positions
                    .iter()
                    .find_map(|(piece_id, position)| {
                        (*position == square)
                            .then(|| state.board.find_by_id(*piece_id))
                            .flatten()
                    })
            })
    }

    fn fianchetto_destination_blocked(
        &self,
        state: &GameState,
        piece: Piece,
        from: Square,
        to: Square,
    ) -> bool {
        let diagonal = |square: Square| {
            if square.file() == square.rank() {
                1
            } else if square.file() + square.rank() == 7 {
                2
            } else {
                0
            }
        };
        let target_diagonal = diagonal(to);
        piece.kind == PieceKind::Pawn
            && target_diagonal != 0
            && diagonal(from) != target_diagonal
            && state.card_runtime.fianchetto[piece.owner.opponent().index()]
            && state.board.iter().any(|(square, candidate)| {
                candidate.owner == piece.owner.opponent()
                    && candidate.kind == PieceKind::Bishop
                    && diagonal(square) == target_diagonal
            })
    }

    /// Killer King: the king may capture along rook lines, but rook-style quiet moves stay illegal.
    fn generate_killer_king_rook_captures(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let mut cursor = from;
            while let Some(target) = cursor.offset(file_delta, rank_delta) {
                if let Some(target_piece) = self.board_piece_at(state, target) {
                    if self.can_capture(state, piece, target_piece)
                        && !self.majesty_destination_blocked(state, piece, target)
                    {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                    }
                    break;
                }
                cursor = target;
            }
        }
    }

    fn generate_leaper_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        offsets: &[(i8, i8)],
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in offsets {
            let Some(target) = from.offset(file_delta, rank_delta) else {
                continue;
            };
            match self.board_piece_at(state, target) {
                None if !self.fianchetto_destination_blocked(state, piece, from, target)
                    && !self.majesty_destination_blocked(state, piece, target) =>
                {
                    actions.push(Action::Move(MoveAction::new(from, target, None)))
                }
                Some(target_piece) if self.can_capture(state, piece, target_piece) => {
                    if !self.fianchetto_destination_blocked(state, piece, from, target)
                        && !self.majesty_destination_blocked(state, piece, target)
                    {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                    }
                }
                None => {}
                Some(_) => {}
            }
        }
    }

    fn generate_slider_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        directions: &[(i8, i8)],
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in directions {
            let mut cursor = from;
            while let Some(target) = cursor.offset(file_delta, rank_delta) {
                match self.board_piece_at(state, target) {
                    None => {
                        if !self.fianchetto_destination_blocked(state, piece, from, target)
                            && !self.majesty_destination_blocked(state, piece, target)
                        {
                            actions.push(Action::Move(MoveAction::new(from, target, None)));
                        }
                    }
                    Some(target_piece)
                        if target_piece.owner != piece.owner
                            && state
                                .card_runtime
                                .stealth_piece_ids
                                .contains(&target_piece.id)
                            && !self.is_royal_piece(state, target_piece) =>
                    {
                        if self.can_capture(state, piece, target_piece)
                            && !self.fianchetto_destination_blocked(state, piece, from, target)
                        {
                            actions.push(Action::Move(MoveAction::new(from, target, None)));
                        }
                        cursor = target;
                        continue;
                    }
                    Some(target_piece) if self.can_capture(state, piece, target_piece) => {
                        if !self.fianchetto_destination_blocked(state, piece, from, target)
                            && !self.majesty_destination_blocked(state, piece, target)
                        {
                            actions.push(Action::Move(MoveAction::new(from, target, None)));
                        }
                        break;
                    }
                    Some(target_piece)
                        if piece.kind == PieceKind::Rook
                            && state
                                .board
                                .get(from)
                                .is_some_and(|origin| origin.kind == PieceKind::Rook)
                            && state.card_runtime.overtake[piece.owner.index()]
                            && target_piece.owner == piece.owner
                            && !matches!(
                                target_piece.kind,
                                PieceKind::Monster | PieceKind::Football
                            )
                            && !state.card_runtime.black_holes.contains(&target) =>
                    {
                        cursor = target;
                        continue;
                    }
                    Some(target_piece)
                        if target_piece.owner == piece.owner
                            && state
                                .card_runtime
                                .ghost_piece_ids
                                .contains(&target_piece.id) =>
                    {
                        cursor = target;
                        continue;
                    }
                    Some(_) => break,
                }
                cursor = target;
            }
        }
    }

    fn generate_assassin_royal_captures(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in &[
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ] {
            let mut cursor = from;
            while let Some(target) = cursor.offset(file_delta, rank_delta) {
                if let Some(target_piece) = state.board.get(target) {
                    if self.is_royal_piece(state, target_piece)
                        && self.can_capture(state, piece, target_piece)
                    {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                    }
                    break;
                }
                cursor = target;
            }
        }
    }

    fn generate_locust_swarm_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for (file_delta, rank_delta) in [
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ] {
            let Some(hurdle) = from.offset(file_delta, rank_delta) else {
                continue;
            };
            if state.board.get(hurdle).is_none() {
                continue;
            }
            let Some(landing) = hurdle.offset(file_delta, rank_delta) else {
                continue;
            };
            match state.board.get(landing) {
                None => actions.push(Action::Move(MoveAction::new(from, landing, None))),
                Some(target) if self.can_capture(state, piece, target) => {
                    actions.push(Action::Move(MoveAction::new(from, landing, None)));
                }
                Some(_) => {}
            }
        }
    }

    fn generate_grasshopper_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in &[
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ] {
            let mut cursor = from;
            let hurdle = loop {
                let Some(target) = cursor.offset(file_delta, rank_delta) else {
                    break None;
                };
                if state.board.get(target).is_some() {
                    break Some(target);
                }
                cursor = target;
            };
            let Some(landing) = hurdle.and_then(|square| square.offset(file_delta, rank_delta))
            else {
                continue;
            };
            match state.board.get(landing) {
                None => actions.push(Action::Move(MoveAction::new(from, landing, None))),
                Some(target) if self.can_capture(state, piece, target) => {
                    actions.push(Action::Move(MoveAction::new(from, landing, None)));
                }
                Some(_) => {}
            }
        }
    }

    fn generate_knight_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        self.generate_leaper_moves(state, from, piece, KNIGHT_OFFSETS, actions);
    }

    fn generate_empty_leaper_moves(
        &self,
        state: &GameState,
        from: Square,
        offsets: &[(i8, i8)],
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in offsets {
            if let Some(target) = from.offset(file_delta, rank_delta)
                && self.board_piece_at(state, target).is_none()
            {
                actions.push(Action::Move(MoveAction::new(from, target, None)));
            }
        }
    }

    fn generate_prime_minister_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        const KING_OFFSETS: &[(i8, i8)] = &[
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];
        for &(first_file, first_rank) in KING_OFFSETS {
            let Some(mid) = from.offset(first_file, first_rank) else {
                continue;
            };
            if state.board.get(mid).is_some() {
                continue;
            }
            for &(second_file, second_rank) in KING_OFFSETS {
                let Some(target) = mid.offset(second_file, second_rank) else {
                    continue;
                };
                if target == from {
                    continue;
                }
                match state.board.get(target) {
                    None => actions.push(Action::Move(MoveAction::new(from, target, None))),
                    Some(target_piece) if self.can_capture(state, piece, target_piece) => {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                    }
                    Some(_) => {}
                }
            }
        }
    }

    fn generate_cannon_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let mut cursor = from;
            let mut jumped = false;
            while let Some(target) = cursor.offset(file_delta, rank_delta) {
                match (jumped, state.board.get(target)) {
                    (false, None) => {}
                    (false, Some(hurdle)) if hurdle.kind != PieceKind::Cannon => jumped = true,
                    (false, Some(_)) => break,
                    (true, None) => {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                    }
                    (true, Some(target_piece)) => {
                        if target_piece.kind != PieceKind::Cannon
                            && self.can_capture(state, piece, target_piece)
                        {
                            actions.push(Action::Move(MoveAction::new(from, target, None)));
                        }
                        break;
                    }
                }
                cursor = target;
            }
        }
    }

    fn generate_herald_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let jump_locked = state
            .card_runtime
            .herald_jump_locked_until
            .get(&piece.id)
            .is_some_and(|deadline| state.turns_taken[piece.owner.index()] < *deadline);
        for &(file_delta, rank_delta) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let mut cursor = from;
            for _ in 0..3 {
                let Some(target) = cursor.offset(file_delta, rank_delta) else {
                    break;
                };
                if state.board.get(target).is_none() {
                    actions.push(Action::Move(MoveAction::new(from, target, None)));
                } else if jump_locked {
                    break;
                }
                cursor = target;
            }
        }
    }

    fn generate_unicorn_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for index in 0..64 {
            let target = Square::from_index(index).expect("board index");
            if target == from {
                continue;
            }
            match state.board.get(target) {
                None => actions.push(Action::Move(MoveAction::new(from, target, None))),
                Some(target_piece)
                    if KNIGHT_OFFSETS.iter().any(|(file_delta, rank_delta)| {
                        from.offset(*file_delta, *rank_delta) == Some(target)
                    }) && self.can_capture(state, piece, target_piece) =>
                {
                    actions.push(Action::Move(MoveAction::new(from, target, None)));
                }
                Some(_) => {}
            }
        }
    }

    fn generate_missionary_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in &[(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            let Some(target) = from.offset(file_delta, rank_delta) else {
                continue;
            };
            if state
                .board
                .get(target)
                .is_none_or(|target_piece| target_piece.owner != piece.owner)
            {
                actions.push(Action::Move(MoveAction::new(from, target, None)));
            }
        }
    }

    fn generate_hook_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        const ORTHOGONALS: &[(i8, i8)] = &[(-1, 0), (1, 0), (0, -1), (0, 1)];
        for &(file_delta, rank_delta) in ORTHOGONALS {
            let mut cursor = from;
            while let Some(target) = cursor.offset(file_delta, rank_delta) {
                match state.board.get(target) {
                    None => {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                        for &(turn_file, turn_rank) in ORTHOGONALS {
                            if turn_file * file_delta + turn_rank * rank_delta != 0 {
                                continue;
                            }
                            let mut bent_cursor = target;
                            while let Some(bent_target) = bent_cursor.offset(turn_file, turn_rank) {
                                match state.board.get(bent_target) {
                                    None => actions.push(Action::Move(MoveAction::new(
                                        from,
                                        bent_target,
                                        None,
                                    ))),
                                    Some(target_piece)
                                        if self.can_capture(state, piece, target_piece) =>
                                    {
                                        actions.push(Action::Move(MoveAction::new(
                                            from,
                                            bent_target,
                                            None,
                                        )));
                                        break;
                                    }
                                    Some(_) => break,
                                }
                                bent_cursor = bent_target;
                            }
                        }
                    }
                    Some(target_piece) if self.can_capture(state, piece, target_piece) => {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                        break;
                    }
                    Some(_) => break,
                }
                cursor = target;
            }
        }
    }

    fn generate_cardinal_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for &(start_file, start_rank) in &[(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            let (mut file_delta, mut rank_delta) = (start_file, start_rank);
            let mut cursor = from;
            let mut seen = BTreeSet::new();
            for _ in 0..256 {
                let mut file = cursor.file() as i8 + file_delta;
                let mut rank = cursor.rank() as i8 + rank_delta;
                if !(0..8).contains(&file) {
                    file_delta *= -1;
                    file = cursor.file() as i8 + file_delta;
                }
                if !(0..8).contains(&rank) {
                    rank_delta *= -1;
                    rank = cursor.rank() as i8 + rank_delta;
                }
                let Some(target) = Square::new(file as u8, rank as u8) else {
                    break;
                };
                if !seen.insert((target, file_delta, rank_delta)) {
                    break;
                }
                match state.board.get(target) {
                    None => actions.push(Action::Move(MoveAction::new(from, target, None))),
                    Some(target_piece) if self.can_capture(state, piece, target_piece) => {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                        break;
                    }
                    Some(_) => break,
                }
                cursor = target;
            }
        }
    }

    fn generate_protestant_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in &[(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            for distance in 1..=3 {
                let Some(target) = from.offset(file_delta * distance, rank_delta * distance) else {
                    continue;
                };
                match state.board.get(target) {
                    None => actions.push(Action::Move(MoveAction::new(from, target, None))),
                    Some(target_piece) if self.can_capture(state, piece, target_piece) => {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                    }
                    Some(_) => {}
                }
            }
        }
    }

    fn generate_piece_ability_actions(&self, state: &GameState, actions: &mut Vec<Action>) {
        for (_, piece) in state.board.iter() {
            if piece.owner != state.turn {
                continue;
            }
            if piece.kind == PieceKind::Colossus {
                let Some(status) = state.card_runtime.large_pieces.get(&piece.id) else {
                    continue;
                };
                for sector in colossus_attack_sectors(status.anchor, piece.owner) {
                    if sector.iter().any(|square| {
                        self.board_piece_at(state, *square).is_some_and(|target| {
                            target.owner != piece.owner && target.kind != PieceKind::Guard
                        })
                    }) {
                        actions.push(Action::UsePieceAbility {
                            piece_id: piece.id,
                            ability: PieceAbility::ColossusSector,
                            target: ActionTarget::Squares(sector),
                        });
                    }
                }
                continue;
            }
            if piece.kind == PieceKind::Merchant {
                let gold = state
                    .card_runtime
                    .merchant_gold
                    .get(&piece.id)
                    .copied()
                    .unwrap_or(0);
                for (square, target) in state.board.iter() {
                    if target.owner != piece.owner
                        && target.kind != PieceKind::Merchant
                        && !state.card_runtime.frozen.contains_key(&target.id)
                        && merchant_cost(target.kind) <= gold
                    {
                        actions.push(Action::UsePieceAbility {
                            piece_id: piece.id,
                            ability: PieceAbility::MerchantBuy,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
                continue;
            }
            if !self.piece_has_ability(state, piece, PieceKind::Wizard) {
                continue;
            }
            let mana = state
                .card_runtime
                .wizard_mana
                .get(&piece.id)
                .copied()
                .unwrap_or(0);
            if mana >= 1 {
                for index in 0..64 {
                    actions.push(Action::UsePieceAbility {
                        piece_id: piece.id,
                        ability: PieceAbility::WizardLightning,
                        target: ActionTarget::Square(
                            Square::from_index(index).expect("board index"),
                        ),
                    });
                }
            }
            if mana >= 2 {
                for (square, target) in state.board.iter() {
                    if target.owner == piece.owner {
                        actions.push(Action::UsePieceAbility {
                            piece_id: piece.id,
                            ability: PieceAbility::WizardShield,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            if mana >= 3 {
                for index in 0..64 {
                    actions.push(Action::UsePieceAbility {
                        piece_id: piece.id,
                        ability: PieceAbility::WizardMeteor,
                        target: ActionTarget::Square(
                            Square::from_index(index).expect("board index"),
                        ),
                    });
                }
            }
            if mana >= 5 {
                actions.push(Action::UsePieceAbility {
                    piece_id: piece.id,
                    ability: PieceAbility::WizardTimeStop,
                    target: ActionTarget::None,
                });
            }
        }
    }

    fn nearest_alibaba_candidates(
        &self,
        state: &GameState,
        origin: Square,
        owner: Player,
    ) -> Vec<Square> {
        for radius in 1..=7 {
            let mut candidates = Vec::new();
            for rank in (0..8).rev() {
                for file in 0..8 {
                    let square = Square::new(file, rank).expect("board index");
                    if square
                        .file()
                        .abs_diff(origin.file())
                        .max(square.rank().abs_diff(origin.rank()))
                        == radius
                        && state.board.get(square).is_none()
                    {
                        candidates.push(square);
                    }
                }
            }
            if candidates.is_empty() {
                continue;
            }
            let safe = candidates
                .iter()
                .copied()
                .filter(|square| !self.is_square_attacked_by(state, *square, owner.opponent()))
                .collect::<Vec<_>>();
            return if safe.is_empty() { candidates } else { safe };
        }
        Vec::new()
    }

    fn is_square_attacked_by(&self, state: &GameState, square: Square, attacker: Player) -> bool {
        let mut probe = state.clone();
        let king_ids = probe
            .board
            .iter()
            .filter(|(_, piece)| piece.kind == PieceKind::King)
            .map(|(_, piece)| piece.id)
            .collect::<Vec<_>>();
        probe.card_runtime.moved_piece_ids.extend(king_ids);
        probe.board.set(
            square,
            Some(Piece::new(attacker.opponent(), PieceKind::Alibaba).with_id(u32::MAX)),
        );
        for (from, piece) in probe.board.iter() {
            if piece.owner != attacker {
                continue;
            }
            let mut actions = Vec::new();
            self.generate_moves_for_piece(&probe, from, piece, &mut actions);
            if actions
                .into_iter()
                .any(|action| matches!(action, Action::Move(movement) if movement.to == square))
            {
                return true;
            }
        }
        false
    }

    fn generate_castling_moves(
        &self,
        state: &GameState,
        from: Square,
        king: Piece,
        actions: &mut Vec<Action>,
    ) {
        let home_rank = if king.owner == Player::White { 0 } else { 7 };
        let home = Square::new(4, home_rank).expect("standard king home is on board");
        if from != home
            || state.card_runtime.moved_piece_ids.contains(&king.id)
            || self.is_square_attacked_by(state, from, king.owner.opponent())
        {
            return;
        }

        const KING_SIDE_CLEAR: &[u8] = &[5, 6];
        const QUEEN_SIDE_CLEAR: &[u8] = &[1, 2, 3];
        const KING_SIDE_SAFE: &[u8] = &[5, 6];
        const QUEEN_SIDE_SAFE: &[u8] = &[3, 2];
        let sides = [
            (7, 6, KING_SIDE_CLEAR, KING_SIDE_SAFE),
            (0, 2, QUEEN_SIDE_CLEAR, QUEEN_SIDE_SAFE),
        ];
        for (rook_file, king_file, clear_files, safe_files) in sides {
            let rook_square = Square::new(rook_file, home_rank).expect("castling rook square");
            let Some(rook) = state.board.get(rook_square) else {
                continue;
            };
            if rook.owner != king.owner
                || rook.kind != PieceKind::Rook
                || state.card_runtime.moved_piece_ids.contains(&rook.id)
                || clear_files.iter().any(|file| {
                    self.board_piece_at(
                        state,
                        Square::new(*file, home_rank).expect("castling path square"),
                    )
                    .is_some()
                })
                || safe_files.iter().any(|file| {
                    self.is_square_attacked_by(
                        state,
                        Square::new(*file, home_rank).expect("castling safe square"),
                        king.owner.opponent(),
                    )
                })
            {
                continue;
            }
            actions.push(Action::Move(MoveAction::new(
                from,
                Square::new(king_file, home_rank).expect("castling king destination"),
                None,
            )));
        }
    }

    fn standard_castling_rook_plan(
        &self,
        state: &GameState,
        king: Piece,
        movement: MoveAction,
    ) -> Option<(Square, Square, Piece)> {
        let home_rank = if king.owner == Player::White { 0 } else { 7 };
        if king.kind != PieceKind::King
            || movement.from != Square::new(4, home_rank)?
            || movement.to.rank() != home_rank
            || !matches!(movement.to.file(), 2 | 6)
        {
            return None;
        }
        let (rook_file, rook_to_file) = if movement.to.file() > movement.from.file() {
            (7, 5)
        } else {
            (0, 3)
        };
        let rook_from = Square::new(rook_file, movement.from.rank())?;
        let rook_to = Square::new(rook_to_file, movement.from.rank())?;
        let rook = state.board.get(rook_from)?;
        (rook.owner == king.owner && rook.kind == PieceKind::Rook)
            .then_some((rook_from, rook_to, rook))
    }

    fn can_capture(&self, state: &GameState, attacker: Piece, target: Piece) -> bool {
        if state
            .card_runtime
            .outpost_protected_piece_ids
            .contains(&target.id)
        {
            return false;
        }
        self.can_capture_ignoring_fresh(state, attacker, target)
            && self.capture_status_allows(state, attacker)
    }

    fn capture_status_allows(&self, state: &GameState, attacker: Piece) -> bool {
        self.capture_status_allows_ignoring_fresh(state, attacker)
            && state
                .card_runtime
                .fresh_no_capture_until
                .get(&attacker.id)
                .is_none_or(|deadline| state.turns_taken[attacker.owner.index()] >= *deadline)
    }

    fn capture_status_allows_ignoring_fresh(&self, state: &GameState, attacker: Piece) -> bool {
        let manner_active = state.applied_rules.iter().any(|rule| rule == COOL_GUY)
            || state
                .card_runtime
                .potion_manner_piece_ids
                .contains(&attacker.id);
        let manner_locked = manner_active
            && state
                .card_runtime
                .manner_captured_last_piece_ids
                .contains(&attacker.id);
        let saturation_active = state.card_runtime.saturation_rule
            || state
                .card_runtime
                .potion_saturation_piece_ids
                .contains(&attacker.id);
        let saturation_locked = saturation_active
            && state
                .card_runtime
                .saturation_capture_counts
                .get(&attacker.id)
                .copied()
                .unwrap_or(0)
                >= 3;
        state.ply >= state.card_runtime.chaos_no_capture_until_ply
            && !state.card_runtime.free_move_no_capture[attacker.owner.index()]
            && !state.card_runtime.disarmed.contains_key(&attacker.id)
            && !manner_locked
            && !saturation_locked
            && state
                .card_runtime
                .reposition
                .is_none_or(|status| status.piece_id != attacker.id)
            && !state.card_runtime.initiative[attacker.owner.index()].is_some_and(|status| {
                state.turns_taken[attacker.owner.index()].saturating_sub(status.start_turn)
                    < status.limit
            })
    }

    fn campfire_protects(&self, state: &GameState, target: Piece) -> bool {
        if self.is_royal_piece(state, target) || target.kind == PieceKind::Scarecrow {
            return false;
        }
        let Some((square, _)) = state.board.find_by_id(target.id) else {
            return false;
        };
        [(-1, 0), (1, 0), (0, -1), (0, 1)].iter().any(|&(df, dr)| {
            square
                .offset(df, dr)
                .and_then(|cell| state.board.get(cell))
                .is_some_and(|fire| {
                    fire.owner == target.owner
                        && fire.id != target.id
                        && self.piece_has_ability(state, fire, PieceKind::Campfire)
                })
        })
    }

    fn can_capture_ignoring_fresh(
        &self,
        state: &GameState,
        attacker: Piece,
        target: Piece,
    ) -> bool {
        if target.kind == PieceKind::Rook
            && state.card_runtime.frontline_response[target.owner.index()]
            && let (Some((from, _)), Some((to, _))) = (
                state.board.find_by_id(attacker.id),
                state.board.find_by_id(target.id),
            )
            && (from.file() == to.file() || from.rank() == to.rank())
        {
            return false;
        }
        (target.kind == PieceKind::Monster || attacker.owner != target.owner)
            && state.card_runtime.armistice_remaining_ply == 0
            && !state.card_runtime.frozen.contains_key(&target.id)
            && !self.middle_submerged_protects(state, target)
            && !(state.card_runtime.overwhelm[target.owner.index()]
                && self.is_royal_piece(state, attacker)
                && (target.kind == PieceKind::Queen || self.is_royal_piece(state, target)))
            && !self.middle_encouragement_protects(state, target)
            && !self.campfire_protects(state, target)
            && !(state
                .card_runtime
                .nullification_piece_ids
                .contains(&target.id)
                && attacker.kind == target.kind)
            && target.kind != PieceKind::Guard
            && !matches!(
                attacker.kind,
                PieceKind::Idol
                    | PieceKind::Recruiter
                    | PieceKind::Wizard
                    | PieceKind::Herald
                    | PieceKind::Paladin
                    | PieceKind::Campfire
            )
            && (attacker.kind != PieceKind::Jester || self.is_royal_piece(state, target))
            && (target.kind != PieceKind::Jester || self.is_royal_piece(state, attacker))
            && !(attacker.kind == PieceKind::Queen
                && target.kind == PieceKind::Pawn
                && state.card_runtime.geneva_convention[target.owner.index()])
    }
}

fn mix_search_seed(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn piece_target(state: &GameState, target: &ActionTarget) -> Result<(Square, Piece), EngineError> {
    let ActionTarget::PieceAt(square) = target else {
        return Err(EngineError::Invariant(
            "implemented active card requires a piece target".to_owned(),
        ));
    };
    let piece = state
        .board
        .get(*square)
        .ok_or_else(|| EngineError::Invariant("legal card target has no piece".to_owned()))?;
    Ok((*square, piece))
}

fn square_target(target: &ActionTarget) -> Result<Square, EngineError> {
    match target {
        ActionTarget::Square(square) | ActionTarget::PieceAt(square) => Ok(*square),
        _ => Err(EngineError::Invariant(
            "action requires exactly one square".to_owned(),
        )),
    }
}

fn opening_square(player: Player, white: &str, black: &str) -> Square {
    if player == Player::White {
        white
    } else {
        black
    }
    .parse()
    .expect("static opening square")
}

fn opening_squares(player: Player, white: &[&str], black: &[&str]) -> Vec<Square> {
    if player == Player::White {
        white
    } else {
        black
    }
    .iter()
    .map(|name| name.parse().expect("static opening square"))
    .collect()
}

fn large_piece_cells(anchor: Square) -> Vec<Square> {
    [(0, 0), (1, 0), (0, -1), (1, -1)]
        .into_iter()
        .filter_map(|(file_delta, rank_delta)| anchor.offset(file_delta, rank_delta))
        .collect()
}

fn clear_queen_line(board: &crate::Board, from: Square, to: Square) -> bool {
    let file_delta = to.file() as i8 - from.file() as i8;
    let rank_delta = to.rank() as i8 - from.rank() as i8;
    if file_delta != 0 && rank_delta != 0 && file_delta.abs() != rank_delta.abs() {
        return false;
    }
    if file_delta == 0 && rank_delta == 0 {
        return false;
    }
    let file_step = file_delta.signum();
    let rank_step = rank_delta.signum();
    let mut cursor = from;
    while let Some(next) = cursor.offset(file_step, rank_step) {
        if next == to {
            return true;
        }
        if board.get(next).is_some() {
            return false;
        }
        cursor = next;
    }
    false
}

fn colossus_attack_sectors(anchor: Square, player: Player) -> [Vec<Square>; 2] {
    let ranks = if player == Player::White {
        [
            anchor.rank().saturating_add(2),
            anchor.rank().saturating_add(3),
        ]
    } else {
        [
            anchor.rank().saturating_sub(3),
            anchor.rank().saturating_sub(4),
        ]
    };
    let sector = |files: [u8; 2]| {
        ranks
            .into_iter()
            .flat_map(|rank| {
                files
                    .into_iter()
                    .filter_map(move |file| Square::new(file, rank))
            })
            .collect::<Vec<_>>()
    };
    [sector([6, 7]), sector([0, 1])]
}

fn merchant_cost(kind: PieceKind) -> u16 {
    match kind {
        PieceKind::Pawn | PieceKind::Fanatic => 2,
        PieceKind::Knight
        | PieceKind::Bishop
        | PieceKind::Protestant
        | PieceKind::Man
        | PieceKind::Dragon
        | PieceKind::Alfil => 3,
        PieceKind::Assassin | PieceKind::Windmill => 4,
        PieceKind::Rook
        | PieceKind::Herald
        | PieceKind::Cannon
        | PieceKind::Cardinal
        | PieceKind::RoyalKnight => 5,
        PieceKind::BigRook
        | PieceKind::BigBishop
        | PieceKind::Brutus
        | PieceKind::Wizard
        | PieceKind::Queen
        | PieceKind::PrimeMinister
        | PieceKind::Jester
        | PieceKind::Recruiter
        | PieceKind::Reaper
        | PieceKind::Idol => 9,
        PieceKind::Amazon => 13,
        PieceKind::Colossus => 15,
        PieceKind::King => 20,
        PieceKind::Merchant => 100,
        PieceKind::Hook => 14,
        PieceKind::Guard => 1,
        PieceKind::Log
        | PieceKind::Squire
        | PieceKind::Checker
        | PieceKind::CheckerKing
        | PieceKind::StandardBearer => 2,
        PieceKind::Grasshopper
        | PieceKind::Thief
        | PieceKind::Knightmaster
        | PieceKind::Unicorn
        | PieceKind::Missionary => 4,
        PieceKind::Alibaba => 3,
        PieceKind::DarkWizard => 6,
        PieceKind::Camel | PieceKind::Bat => 3,
        PieceKind::Ferz | PieceKind::Lobster => 2,
        PieceKind::BabyBear => 1,
        PieceKind::Bear => 9,
        PieceKind::SiegeRam | PieceKind::Slime | PieceKind::Clockwork => 5,
        PieceKind::Parrot | PieceKind::Campfire => 5,
        PieceKind::Paladin => 4,
        PieceKind::Princess => 6,
        PieceKind::Octopus => 5,
        PieceKind::MagicGirl => 6,
        PieceKind::Berserker => 7,
        PieceKind::Siren | PieceKind::Trickster | PieceKind::Undead | PieceKind::Hedgehog => 9,
        PieceKind::Monster | PieceKind::Scarecrow | PieceKind::Crown | PieceKind::Football => 0,
        PieceKind::Vip => 4,
    }
}

fn two_square_target(target: &ActionTarget) -> Result<[Square; 2], EngineError> {
    let ActionTarget::Squares(squares) = target else {
        return Err(EngineError::Invariant(
            "action requires a pair of squares".to_owned(),
        ));
    };
    let [first, second] = squares.as_slice() else {
        return Err(EngineError::Invariant(
            "action requires exactly two squares".to_owned(),
        ));
    };
    Ok([*first, *second])
}

/// Web `forEachSquare` order is a8..h8, a7..h7, ... a1..h1.
fn web_ordered_pieces(state: &GameState) -> Vec<(Square, Piece)> {
    let mut pieces = Vec::new();
    for rank in (0..8).rev() {
        for file in 0..8 {
            let square = Square::new(file, rank).expect("8x8 web-order square");
            if let Some(piece) = state.board.get(square) {
                pieces.push((square, piece));
            }
        }
    }
    pieces
}

fn is_center_two_by_two(square: Square) -> bool {
    matches!(square.file(), 3 | 4) && matches!(square.rank(), 3 | 4)
}

fn squares_are_adjacent(left: Square, right: Square) -> bool {
    let file_delta = left.file().abs_diff(right.file());
    let rank_delta = left.rank().abs_diff(right.rank());
    file_delta <= 1 && rank_delta <= 1 && (file_delta != 0 || rank_delta != 0)
}

/// Integer form of the web bundle's draft weights. Base weights use eighths; OPENING cards in
/// the first draft receive the bundle's 1.3 multiplier. The common denominator is irrelevant to
/// weighted selection, so the returned units are exact (`base_eighths * 10` or `* 13`).
fn draft_card_weight_units(definition_id: &str, draft_phase: DraftPhase) -> u64 {
    let Some(card) = implemented_card(definition_id) else {
        return 0;
    };
    let score = card.half_stars;
    let base_eighths = if score <= 5 {
        9
    } else {
        u64::from(14_u16.saturating_sub(score.clamp(6, 10)))
    };
    let phase_multiplier = if draft_phase == DraftPhase::Opening && card.phase == CardPhase::Opening
    {
        13
    } else {
        10
    };
    base_eighths * phase_multiplier
}

fn draft_offer_average(offers: &[String]) -> Option<DraftOfferAverage> {
    let scores: Vec<_> = offers
        .iter()
        .filter_map(|definition_id| implemented_card(definition_id))
        .map(|card| u32::from(card.half_stars))
        .collect();
    (!scores.is_empty()).then(|| DraftOfferAverage {
        total_half_stars: scores.iter().sum(),
        card_count: scores.len() as u32,
    })
}

fn draft_average_gap(left: DraftOfferAverage, right: DraftOfferAverage) -> DraftAverageGap {
    let left_scaled = u64::from(left.total_half_stars) * u64::from(right.card_count);
    let right_scaled = u64::from(right.total_half_stars) * u64::from(left.card_count);
    DraftAverageGap {
        numerator: left_scaled.abs_diff(right_scaled),
        denominator: u64::from(left.card_count) * u64::from(right.card_count),
    }
}

fn draft_gap_is_smaller(candidate: DraftAverageGap, current: DraftAverageGap) -> bool {
    u128::from(candidate.numerator) * u128::from(current.denominator)
        < u128::from(current.numerator) * u128::from(candidate.denominator)
}

fn draft_gap_within_balance_limit(gap: DraftAverageGap) -> bool {
    gap.numerator <= DRAFT_BALANCE_MAX_HALF_STAR_GAP * gap.denominator
}

fn draft_card_uses_next_turn_delay(draft_phase: DraftPhase, card_phase: CardPhase) -> bool {
    matches!(draft_phase, DraftPhase::Middle | DraftPhase::End)
        && matches!(card_phase, CardPhase::Middle | CardPhase::End)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineError {
    RulesetMismatch { engine: String, state: String },
    UnknownMode(String),
    UnsupportedMode(String),
    UnknownCard(String),
    CardIsRule(String),
    CardIsNotRule(String),
    DuplicateCardInstance(u32),
    RuleAlreadyApplied(String),
    GameAlreadyEnded,
    UnsupportedPhase(&'static str),
    IllegalAction(String),
    InvalidEffect(String),
    EffectResolutionLimit(usize),
    AutomaticTurnEndLimit(usize),
    Invariant(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RulesetMismatch { engine, state } => {
                write!(f, "ruleset mismatch: engine={engine:?}, state={state:?}")
            }
            Self::UnknownMode(mode) => write!(f, "unknown mode {mode:?}"),
            Self::UnsupportedMode(mode) => write!(f, "mode {mode:?} is not implemented yet"),
            Self::UnknownCard(card) => write!(f, "unknown or unimplemented card {card:?}"),
            Self::CardIsRule(card) => write!(f, "card {card:?} is a global RULE"),
            Self::CardIsNotRule(card) => write!(f, "card {card:?} is not a global RULE"),
            Self::DuplicateCardInstance(instance_id) => {
                write!(f, "card instance {instance_id} already exists")
            }
            Self::RuleAlreadyApplied(card) => write!(f, "RULE {card:?} is already applied"),
            Self::GameAlreadyEnded => f.write_str("the game has already ended"),
            Self::UnsupportedPhase(phase) => write!(f, "phase {phase:?} is not implemented yet"),
            Self::IllegalAction(action) => write!(f, "illegal action {action}"),
            Self::InvalidEffect(message) => write!(f, "invalid effect command: {message}"),
            Self::EffectResolutionLimit(limit) => {
                write!(f, "effect resolution exceeded {limit} events")
            }
            Self::AutomaticTurnEndLimit(limit) => {
                write!(f, "automatic turn ending exceeded {limit} turns")
            }
            Self::Invariant(message) => write!(f, "engine invariant failed: {message}"),
        }
    }
}

impl std::error::Error for EngineError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BUNDLED_RULESET_ID, Board, CHESS_344200, CardInstance, ChoiceHandler, EffectHandler,
    };
    use std::str::FromStr;

    fn movement(from: &str, to: &str) -> Action {
        Action::Move(MoveAction::new(
            Square::from_str(from).unwrap(),
            Square::from_str(to).unwrap(),
            None,
        ))
    }

    fn use_card(instance_id: u32, target: &str) -> Action {
        Action::UseCard {
            instance_id,
            target: ActionTarget::PieceAt(target.parse().unwrap()),
        }
    }

    fn use_card_without_target(instance_id: u32) -> Action {
        Action::UseCard {
            instance_id,
            target: ActionTarget::None,
        }
    }

    fn minimal_state(pieces: &[(&str, Player, PieceKind)], turn: Player, seed: u64) -> GameState {
        let mut board = Board::empty();
        for (square, owner, kind) in pieces {
            board.set(square.parse().unwrap(), Some(Piece::new(*owner, *kind)));
        }
        GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, turn, seed)
    }

    fn september_state(pieces: &[(&str, Player, PieceKind)], turn: Player, seed: u64) -> GameState {
        let mut board = Board::empty();
        for (square, owner, kind) in pieces {
            board.set(square.parse().unwrap(), Some(Piece::new(*owner, *kind)));
        }
        GameState::from_board("augment_chess_20260915", Mode::Standard, board, turn, seed)
    }

    fn grant_test_card(state: &mut GameState, definition_id: &str, instance_id: u32) {
        let definition = implemented_card(definition_id).unwrap();
        state.add_card(CardInstance {
            instance_id,
            definition_id: definition_id.to_owned(),
            owner: state.turn,
            half_stars: definition.half_stars,
            charges: 1,
            available_after_turn: None,
        });
    }

    fn has_move(engine: &Engine, state: &GameState, from: &str, to: &str) -> bool {
        engine
            .legal_actions(state)
            .unwrap()
            .contains(&movement(from, to))
    }

    fn standard_play_state(seed: u64) -> GameState {
        GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, seed)
    }

    fn first_draft_pick(engine: &Engine, state: &GameState) -> Transition {
        engine
            .apply(state, Action::DraftPick { offer_index: 0 })
            .unwrap()
    }

    fn grant_test_opening_auto_card(state: &mut GameState, definition_id: &str, instance_id: u32) {
        grant_test_card(state, definition_id, instance_id);
        state.card_runtime.opening_auto_card_ids.insert(instance_id);
    }

    #[test]
    fn standard_game_runs_white_then_black_opening_draft_before_play() {
        let engine = Engine::bundled_20260807().unwrap();
        // This test covers the ordinary opening-draft flow, so select a seed
        // where the optional global opening rule did not rewrite the board.
        let state = (0..100)
            .map(|seed| engine.new_game(Mode::Standard, seed).unwrap())
            .find(|state| state.applied_rules.is_empty())
            .expect("a 40% RULE check must miss within the first 100 seeds");
        let GamePhase::Draft(draft) = &state.phase else {
            panic!("standard game must start in the opening draft")
        };
        assert_eq!(draft.phase, DraftPhase::Opening);
        assert_eq!(state.turn, Player::White);
        assert_eq!(draft.offers.len(), 3);
        assert_eq!(
            draft.offers.iter().cloned().collect::<BTreeSet<_>>().len(),
            3
        );
        assert!(draft.offers.iter().all(|id| {
            implemented_card(id).is_some_and(|card| {
                matches!(
                    card.phase,
                    CardPhase::Opening | CardPhase::Middle | CardPhase::Piece
                )
            })
        }));
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .iter()
                .all(|action| matches!(action, Action::DraftPick { .. }))
        );

        let after_white = first_draft_pick(&engine, &state).state;
        assert_eq!(after_white.turn, Player::Black);
        let GamePhase::Draft(black_draft) = &after_white.phase else {
            panic!("black must receive the second opening draft")
        };
        assert_eq!(after_white.cards.len(), 1);
        assert!(
            !black_draft
                .offers
                .contains(&after_white.cards[0].definition_id)
        );

        let after_black = first_draft_pick(&engine, &after_white).state;
        assert_eq!(after_black.turn, Player::White);
        assert_eq!(after_black.phase, GamePhase::Playing);
        assert!(after_black.completed_drafts[DraftPhase::Opening.index()]);
        assert_eq!(after_black.cards.len(), 2);
        assert!(
            engine
                .legal_actions(&after_black)
                .unwrap()
                .iter()
                .filter(|action| matches!(action, Action::Move(_)))
                .count()
                >= 20
        );
    }

    #[test]
    fn initial_draft_opening_card_is_hidden_then_used_after_the_first_move() {
        let engine = Engine::bundled_20260807().unwrap();
        // Adding implemented cards shifts the seeded draft pools, so the seed search
        // validates the full hidden-then-auto-used invariant instead of pinning a seed.
        let (state, opening_id) = (0..1_000)
            .find_map(|seed| {
                let state = engine.new_game(Mode::Standard, seed).ok()?;
                let GamePhase::Draft(draft) = &state.phase else {
                    return None;
                };
                draft.offers.iter().enumerate().find_map(|(index, id)| {
                    let (state, opening_id) = (state.clone(), id.clone());
                    let picked = engine
                        .apply(&state, Action::DraftPick { offer_index: index as u16 })
                        .ok()?
                        .state;
                    let picked = first_draft_pick(&engine, &picked).state;
                    let instance_id = picked
                        .cards
                        .iter()
                        .find(|card| card.owner == Player::White && card.definition_id == opening_id)?
                        .instance_id;
                    if !picked
                        .card_runtime
                        .opening_auto_card_ids
                        .contains(&instance_id)
                    {
                        return None;
                    }
                    if engine.legal_actions(&picked).unwrap().iter().any(|action| {
                        matches!(action, Action::UseCard { instance_id: candidate, .. } if *candidate == instance_id)
                    }) {
                        return None;
                    }
                    let first_move = engine
                        .legal_actions(&picked)
                        .unwrap()
                        .into_iter()
                        .find(|action| matches!(action, Action::Move(_)))?;
                    let after = engine.apply(&picked, first_move).ok()?.state;
                    let consumed = after
                        .cards
                        .iter()
                        .find(|card| card.instance_id == instance_id)
                        .is_some_and(|card| card.charges == 0);
                    consumed.then_some((picked, opening_id))
                })
            })
            .expect("a bounded seed search must find an auto-used active OPENING offer");
        let card = state
            .cards
            .iter()
            .find(|card| card.owner == Player::White && card.definition_id == opening_id)
            .unwrap();
        let instance_id = card.instance_id;
        assert!(
            state
                .card_runtime
                .opening_auto_card_ids
                .contains(&instance_id)
        );
        assert!(engine.legal_actions(&state).unwrap().iter().all(|action| {
            !matches!(action, Action::UseCard { instance_id: candidate, .. } if *candidate == instance_id)
        }));

        let first_move = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .find(|action| matches!(action, Action::Move(_)))
            .expect("standard opening must have a legal move");
        let transition = engine.apply(&state, first_move).unwrap();
        let card = transition
            .state
            .cards
            .iter()
            .find(|card| card.instance_id == instance_id)
            .unwrap();
        assert_eq!(card.charges, 0);
        assert!(
            !transition
                .state
                .card_runtime
                .opening_auto_card_ids
                .contains(&instance_id)
        );
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::CardUsed {
                owner: Player::White,
                instance_id: used,
                definition_id,
            } if *used == instance_id && definition_id == &opening_id
        )));
        assert_eq!(transition.state.turn, Player::Black);
    }

    #[test]
    fn targeted_opening_card_chooses_a_deterministic_target_after_the_first_move() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Pawn),
                ("b2", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            30,
        );
        grant_test_opening_auto_card(&mut state, HOLDOUT, 90);
        assert!(engine.legal_actions(&state).unwrap().iter().all(|action| {
            !matches!(
                action,
                Action::UseCard {
                    instance_id: 90,
                    ..
                }
            )
        }));

        let left = engine.apply(&state, movement("h1", "g1")).unwrap();
        let right = engine.apply(&state, movement("h1", "g1")).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.state.card_runtime.holdout_promotions.len(), 1);
        assert!(left.events.iter().any(|event| matches!(
            event,
            Event::RandomSample { purpose, .. } if purpose == "opening-auto-target:holdout"
        )));
        assert!(left.events.iter().any(|event| matches!(
            event,
            Event::CardUsed {
                instance_id: 90,
                ..
            }
        )));
    }

    #[test]
    fn blocked_opening_card_rolls_back_the_first_move_before_auto_use() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(31);
        grant_test_opening_auto_card(&mut state, ELEPHANT_ESCAPE, 91);

        let transition = engine.apply(&state, movement("a2", "a3")).unwrap();
        assert_eq!(transition.state.ply, 1, "the attempted move still counts");
        assert_eq!(
            transition
                .state
                .board
                .get("a2".parse().unwrap())
                .unwrap()
                .kind,
            PieceKind::Pawn
        );
        assert_eq!(
            transition
                .state
                .board
                .get("a3".parse().unwrap())
                .unwrap()
                .kind,
            PieceKind::Alfil
        );
        assert_eq!(
            transition
                .state
                .board
                .get("h3".parse().unwrap())
                .unwrap()
                .kind,
            PieceKind::Alfil
        );
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::EffectApplied {
                source_card_instance: 91,
                command,
            } if command == "opening-auto-move-rollback:elephant-escape"
        )));
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::CardUsed {
                instance_id: 91,
                ..
            }
        )));
        assert_eq!(transition.state.turn, Player::Black);
    }

    #[test]
    fn opening_rule_check_is_deterministic_and_precedes_the_first_draft() {
        let engine = Engine::bundled_20260807().unwrap();
        let hit = (0..100)
            .find_map(|seed| {
                let transition = engine.new_game_transition(Mode::Standard, seed).unwrap();
                transition
                    .events
                    .iter()
                    .any(|event| {
                        matches!(
                            event,
                            Event::OpeningRuleChecked {
                                triggered: true,
                                definition_id: Some(id),
                            } if id == PLATFORM
                        )
                    })
                    .then_some((seed, transition))
            })
            .expect("a 40% RULE check must hit within the first 100 seeds");
        let miss = (0..100)
            .find_map(|seed| {
                let transition = engine.new_game_transition(Mode::Standard, seed).unwrap();
                transition
                    .events
                    .iter()
                    .any(|event| {
                        matches!(
                            event,
                            Event::OpeningRuleChecked {
                                triggered: false,
                                definition_id: None,
                            }
                        )
                    })
                    .then_some(transition)
            })
            .expect("a 40% RULE check must miss within the first 100 seeds");

        assert_eq!(engine.new_game(Mode::Standard, hit.0).unwrap(), hit.1.state);
        assert_eq!(hit.1.state.applied_rules, [PLATFORM]);
        assert!(hit.1.state.card_runtime.platform.is_some());
        assert!(miss.state.applied_rules.is_empty());
        let rule_index = hit
            .1
            .events
            .iter()
            .position(|event| matches!(event, Event::RuleApplied { .. }))
            .unwrap();
        let draft_index = hit
            .1
            .events
            .iter()
            .position(|event| matches!(event, Event::DraftStarted { .. }))
            .unwrap();
        assert!(rule_index < draft_index);
    }

    #[test]
    fn chess_344200_randomizes_all_pieces_with_kings_on_home_ranks() {
        let engine = Engine::bundled_20260818().unwrap();
        let transition = engine
            .new_game_transition(Mode::Standard, 202_608_280_008)
            .unwrap();
        let state = &transition.state;

        assert!(state.applied_rules.iter().any(|rule| rule == CHESS_344200));
        let piece_ids = state
            .board
            .iter()
            .map(|(_, piece)| piece.id)
            .collect::<Vec<_>>();
        let unique_ids = piece_ids.iter().copied().collect::<BTreeSet<_>>();

        assert_eq!(piece_ids.len(), 32);
        assert_eq!(unique_ids.len(), piece_ids.len());
        assert!(state.board.iter().any(|(square, _)| square.index() < 32));
        for player in [Player::White, Player::Black] {
            let (square, _) = state
                .board
                .iter()
                .find(|(_, piece)| piece.owner == player && piece.kind == PieceKind::King)
                .unwrap();
            assert_eq!(square.rank(), if player == Player::White { 0 } else { 7 });
        }
        assert!(state.board.iter().all(|(square, piece)| {
            state.card_runtime.piece_origins.get(&piece.id) == Some(&square)
        }));
        assert_eq!(
            transition
                .events
                .iter()
                .filter(|event| matches!(
                    event,
                    Event::RandomSample { purpose, .. }
                        if purpose.starts_with("chess-344200-")
                ))
                .count(),
            32
        );
        assert_eq!(
            engine.new_game(Mode::Standard, 202_608_280_008).unwrap(),
            *state
        );
    }

    #[test]
    fn first_draft_uses_opening_middle_and_piece_but_not_end_or_rule_cards() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = standard_play_state(5);
        let candidates: BTreeSet<_> = engine
            .draft_candidates_for_categories(
                &state,
                &[CardPhase::Opening, CardPhase::Middle, CardPhase::Piece],
                &BTreeSet::new(),
            )
            .into_iter()
            .collect();

        assert!(candidates.contains(KING_OF_THE_HILL));
        assert!(candidates.contains(GENEVA_CONVENTION));
        assert!(candidates.contains(PARRY));
        assert!(candidates.contains(SPRINT));
        assert!(candidates.contains(EARLY_PROMOTION));
        assert!(candidates.contains(INITIATIVE));
        assert!(candidates.contains(DISARM));
        assert!(candidates.contains(DICE));
        assert!(candidates.contains(FREEZE));
        assert!(candidates.contains(REPOSITION));
        assert!(candidates.contains(ASSASSIN));
        assert!(candidates.contains(GRASSHOPPER));
        assert!(!candidates.contains(EMPTY_LUNCHBOX));
        assert!(!candidates.contains(JUDGMENT));
        assert!(!candidates.contains(CLEANUP_SACRIFICE));
        assert!(!candidates.contains(PLATFORM));
        assert!(!candidates.contains(REVELATION));
    }

    #[test]
    fn first_draft_applies_the_web_opening_weight_boost() {
        assert_eq!(
            draft_card_weight_units(KING_OF_THE_HILL, DraftPhase::Opening),
            117
        );
        assert_eq!(
            draft_card_weight_units(GENEVA_CONVENTION, DraftPhase::Opening),
            104
        );
        assert_eq!(draft_card_weight_units(PARRY, DraftPhase::Opening), 60);
        assert_eq!(
            draft_card_weight_units(KING_OF_THE_HILL, DraftPhase::Middle),
            90
        );
    }

    #[test]
    fn later_draft_card_delay_depends_on_card_category() {
        assert!(!draft_card_uses_next_turn_delay(
            DraftPhase::Opening,
            CardPhase::Middle
        ));
        assert!(draft_card_uses_next_turn_delay(
            DraftPhase::Middle,
            CardPhase::Middle
        ));
        assert!(!draft_card_uses_next_turn_delay(
            DraftPhase::Middle,
            CardPhase::Piece
        ));
        assert!(draft_card_uses_next_turn_delay(
            DraftPhase::End,
            CardPhase::Middle
        ));
        assert!(draft_card_uses_next_turn_delay(
            DraftPhase::End,
            CardPhase::End
        ));
    }

    #[test]
    fn draft_balance_gap_uses_exact_half_star_averages() {
        let white = DraftOfferAverage {
            total_half_stars: 24,
            card_count: 3,
        };
        let within_limit = draft_average_gap(
            DraftOfferAverage {
                total_half_stars: 30,
                card_count: 3,
            },
            white,
        );
        let outside_limit = draft_average_gap(
            DraftOfferAverage {
                total_half_stars: 31,
                card_count: 3,
            },
            white,
        );

        assert!(draft_gap_within_balance_limit(within_limit));
        assert!(!draft_gap_within_balance_limit(outside_limit));
        assert!(draft_gap_is_smaller(within_limit, outside_limit));
    }

    #[test]
    fn chaos_and_grand_are_explicitly_out_of_scope() {
        let engine = Engine::bundled_20260807().unwrap();
        assert_eq!(
            engine.new_game(Mode::Chaos, 1).unwrap_err(),
            EngineError::UnsupportedMode("chaos".to_owned())
        );
        assert_eq!(
            engine.new_game(Mode::Grand, 1).unwrap_err(),
            EngineError::UnsupportedMode("grand".to_owned())
        );
    }

    #[test]
    fn middle_draft_starts_after_both_players_complete_ten_turns() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(11);
        state.completed_drafts[DraftPhase::Opening.index()] = true;
        state.turn = Player::Black;
        state.turns_taken = [10, 9];

        let action = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .find(|action| matches!(action, Action::Move(_)))
            .expect("the transformed test position must still have a legal move");
        let transition = engine.apply(&state, action).unwrap();
        let GamePhase::Draft(draft) = &transition.state.phase else {
            panic!("the 10-turn checkpoint must start a draft")
        };
        assert_eq!(draft.phase, DraftPhase::Middle);
        assert_eq!(transition.state.turn, Player::White);
        assert_eq!(draft.resume_turn, Player::White);
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::DraftStarted {
                phase: DraftPhase::Middle,
                player: Player::White,
                ..
            }
        )));
    }

    #[test]
    fn end_draft_starts_after_both_players_complete_twenty_turns() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(12);
        state.completed_drafts[DraftPhase::Opening.index()] = true;
        state.completed_drafts[DraftPhase::Middle.index()] = true;
        state.turn = Player::Black;
        state.turns_taken = [20, 19];

        let action = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .find(|action| matches!(action, Action::Move(_)))
            .expect("the exhausted draft-pool position must have a legal move");
        let transition = engine.apply(&state, action).unwrap();
        let GamePhase::Draft(draft) = &transition.state.phase else {
            panic!("the 20-turn checkpoint must start a draft")
        };
        assert_eq!(draft.phase, DraftPhase::End);
        assert_eq!(draft.resume_turn, Player::White);
    }

    #[test]
    fn middle_draft_card_is_locked_for_its_owners_first_turn() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(13);
        state.completed_drafts[DraftPhase::Opening.index()] = true;
        state.turn = Player::Black;
        state.turns_taken = [10, 9];
        state = engine.apply(&state, movement("g8", "f6")).unwrap().state;
        // New passive middle cards changed seeded pools; pin a known active offer for determinism.
        if let GamePhase::Draft(draft) = &mut state.phase {
            draft.offers[0] = "parry".to_owned();
        }
        let white_offer_index = match &state.phase {
            GamePhase::Draft(draft) => draft
                .offers
                .iter()
                .position(|id| {
                    implemented_card(id).is_some_and(|card| {
                        card.phase == CardPhase::Middle && card.activation == CardActivation::Active
                    })
                })
                .expect("middle draft must offer an active MIDDLE card"),
            _ => panic!("middle draft must be active"),
        };
        let white_definition_id = match &state.phase {
            GamePhase::Draft(draft) => draft.offers[white_offer_index].clone(),
            _ => unreachable!(),
        };
        state = engine
            .apply(
                &state,
                Action::DraftPick {
                    offer_index: white_offer_index as u16,
                },
            )
            .unwrap()
            .state;
        state = first_draft_pick(&engine, &state).state;

        let white_card = state
            .cards
            .iter()
            .find(|card| card.owner == Player::White && card.definition_id == white_definition_id)
            .unwrap();
        assert_eq!(white_card.available_after_turn, Some(11));
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .iter()
                .any(|action| matches!(action, Action::UseCard { .. }))
        );

        state = engine.apply(&state, movement("g1", "f3")).unwrap().state;
        state = engine.apply(&state, movement("f6", "g8")).unwrap().state;
        let white_card = state
            .cards
            .iter()
            .find(|card| card.owner == Player::White && card.definition_id == white_definition_id)
            .unwrap();
        assert_eq!(white_card.available_after_turn, None);
        assert!(engine
            .legal_actions(&state)
            .unwrap()
            .iter()
            .any(|action| matches!(action, Action::UseCard { instance_id, .. } if *instance_id == white_card.instance_id)));
    }

    #[test]
    fn milestone_draft_skips_both_players_when_the_pool_is_exhausted() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(15);
        state.completed_drafts[DraftPhase::Opening.index()] = true;
        state.turn = Player::Black;
        state.turns_taken = [10, 9];
        for (offset, definition) in IMPLEMENTED_CARDS
            .iter()
            .filter(|card| matches!(card.phase, CardPhase::Middle | CardPhase::Piece))
            .enumerate()
        {
            state = engine
                .grant_card(&state, 77 + offset as u32, definition.id, Player::White)
                .unwrap()
                .state;
        }

        let action = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .find(|action| matches!(action, Action::Move(_)))
            .expect("the exhausted middle-pool position must have a legal move");
        let transition = engine.apply(&state, action).unwrap();
        assert_eq!(transition.state.phase, GamePhase::Playing);
        assert_eq!(transition.state.turn, Player::White);
        assert!(transition.state.completed_drafts[DraftPhase::Middle.index()]);
        assert_eq!(
            transition
                .events
                .iter()
                .filter(|event| matches!(
                    event,
                    Event::DraftSkipped {
                        phase: DraftPhase::Middle,
                        ..
                    }
                ))
                .count(),
            2
        );
    }

    #[test]
    fn deathmatch_starts_at_45_and_resolves_after_ten_quiet_full_turns() {
        let mut engine = Engine::bundled_20260807().unwrap();
        engine.ruleset.base_rules.repetition_count = u8::MAX;
        let mut state = standard_play_state(17);
        state.completed_drafts = [true; 3];
        state.turn = Player::Black;
        state.turns_taken = [45, 44];
        state.half_star_totals = [2, 4];

        let started = engine.apply(&state, movement("g8", "f6")).unwrap();
        state = started.state;
        assert!(started.events.iter().any(|event| matches!(
            event,
            Event::DeathmatchStarted {
                shared_turn: 45,
                interval_full_turns: 10
            }
        )));
        assert_eq!(state.deathmatch.unwrap().quiet_full_turns, 0);

        for round in 0..10 {
            let white = if round % 2 == 0 {
                ("g1", "f3")
            } else {
                ("f3", "g1")
            };
            let black = if round % 2 == 0 {
                ("f6", "g8")
            } else {
                ("g8", "f6")
            };
            state = engine
                .apply(&state, movement(white.0, white.1))
                .unwrap()
                .state;
            state = engine
                .apply(&state, movement(black.0, black.1))
                .unwrap()
                .state;
        }
        assert_eq!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::LowerStarsAfterDeathmatch,
            }
        );
    }

    #[test]
    fn pawn_move_resets_the_deathmatch_quiet_window() {
        let mut engine = Engine::bundled_20260807().unwrap();
        engine.ruleset.base_rules.repetition_count = u8::MAX;
        let mut state = standard_play_state(19);
        state.completed_drafts = [true; 3];
        state.turn = Player::Black;
        state.turns_taken = [45, 44];
        state = engine.apply(&state, movement("g8", "f6")).unwrap().state;

        for round in 0..9 {
            let white = if round % 2 == 0 {
                ("g1", "f3")
            } else {
                ("f3", "g1")
            };
            let black = if round % 2 == 0 {
                ("f6", "g8")
            } else {
                ("g8", "f6")
            };
            state = engine
                .apply(&state, movement(white.0, white.1))
                .unwrap()
                .state;
            state = engine
                .apply(&state, movement(black.0, black.1))
                .unwrap()
                .state;
        }
        assert_eq!(state.deathmatch.unwrap().quiet_full_turns, 9);

        state = engine.apply(&state, movement("e2", "e3")).unwrap().state;
        state = engine.apply(&state, movement("g8", "f6")).unwrap().state;
        assert_eq!(state.status, GameStatus::Ongoing);
        assert_eq!(state.deathmatch.unwrap().quiet_full_turns, 0);
    }

    #[test]
    fn en_passant_is_generated_applied_and_logged_as_a_capture() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(23);
        for (from, to) in [("e2", "e4"), ("a7", "a6"), ("e4", "e5"), ("d7", "d5")] {
            state = engine.apply(&state, movement(from, to)).unwrap().state;
        }
        assert_eq!(state.en_passant.unwrap().target, "d6".parse().unwrap());
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("e5", "d6"))
        );

        let transition = engine.apply(&state, movement("e5", "d6")).unwrap();
        assert!(transition.state.board.get("d5".parse().unwrap()).is_none());
        assert_eq!(
            transition
                .state
                .board
                .get("d6".parse().unwrap())
                .unwrap()
                .owner,
            Player::White
        );
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::PieceCaptured { at, piece, .. }
                if *at == "d5".parse::<Square>().unwrap() && piece.kind == PieceKind::Pawn
        )));
    }

    #[test]
    fn standard_castling_moves_the_king_and_rook_for_both_colors() {
        let engine = Engine::bundled_20260807().unwrap();
        let white = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h1", Player::White, PieceKind::Rook),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            24,
        );
        assert!(has_move(&engine, &white, "e1", "g1"));
        assert!(has_move(&engine, &white, "e1", "c1"));
        let king_id = white.board.get("e1".parse().unwrap()).unwrap().id;
        let rook_id = white.board.get("h1".parse().unwrap()).unwrap().id;
        let transition = engine.apply(&white, movement("e1", "g1")).unwrap();
        assert_eq!(
            transition
                .state
                .board
                .get("g1".parse().unwrap())
                .unwrap()
                .id,
            king_id
        );
        assert_eq!(
            transition
                .state
                .board
                .get("f1".parse().unwrap())
                .unwrap()
                .id,
            rook_id
        );
        assert!(transition.state.board.get("e1".parse().unwrap()).is_none());
        assert!(transition.state.board.get("h1".parse().unwrap()).is_none());
        assert!(
            transition
                .state
                .card_runtime
                .moved_piece_ids
                .contains(&king_id)
        );
        assert!(
            transition
                .state
                .card_runtime
                .moved_piece_ids
                .contains(&rook_id)
        );
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::EffectApplied { command, .. } if command == "castle:e1:g1:h1:f1"
        )));

        let black = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("e8", Player::Black, PieceKind::King),
                ("a8", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::Rook),
            ],
            Player::Black,
            25,
        );
        assert!(has_move(&engine, &black, "e8", "g8"));
        assert!(has_move(&engine, &black, "e8", "c8"));
        let black = engine.apply(&black, movement("e8", "c8")).unwrap().state;
        assert_eq!(
            black.board.get("c8".parse().unwrap()).unwrap().kind,
            PieceKind::King
        );
        assert_eq!(
            black.board.get("d8".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
    }

    #[test]
    fn castling_requires_unmoved_clear_and_safe_king_and_rook() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut moved = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h1", Player::White, PieceKind::Rook),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            26,
        );
        let king = moved.board.get("e1".parse().unwrap()).unwrap();
        moved.card_runtime.moved_piece_ids.insert(king.id);
        assert!(!has_move(&engine, &moved, "e1", "g1"));
        assert!(!has_move(&engine, &moved, "e1", "c1"));

        let blocked = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("h1", Player::White, PieceKind::Rook),
                ("f1", Player::White, PieceKind::Bishop),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            27,
        );
        assert!(!has_move(&engine, &blocked, "e1", "g1"));

        let through_check = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("h1", Player::White, PieceKind::Rook),
                ("a8", Player::Black, PieceKind::King),
                ("f8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            28,
        );
        assert!(!has_move(&engine, &through_check, "e1", "g1"));

        let in_check = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("h1", Player::White, PieceKind::Rook),
                ("a8", Player::Black, PieceKind::King),
                ("e8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            29,
        );
        assert!(!has_move(&engine, &in_check, "e1", "g1"));
    }

    #[test]
    fn en_passant_expires_after_the_opponents_next_move() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(29);
        for (from, to) in [
            ("e2", "e4"),
            ("a7", "a6"),
            ("e4", "e5"),
            ("d7", "d5"),
            ("g1", "f3"),
        ] {
            state = engine.apply(&state, movement(from, to)).unwrap().state;
        }
        assert!(state.en_passant.is_none());
        state = engine.apply(&state, movement("a6", "a5")).unwrap().state;
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("e5", "d6"))
        );
    }

    #[test]
    fn web_pawns_may_double_step_from_either_home_rank() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::Pawn),
                ("h1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            31,
        );
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("a1", "a3"))
        );
    }

    #[test]
    fn standard_position_has_twenty_ordered_actions() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = standard_play_state(1);
        let actions = engine.legal_actions(&state).unwrap();
        assert_eq!(actions.len(), 20);
        assert!(actions.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn e2_e4_is_a_deterministic_pure_transition() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = standard_play_state(99);
        let before = state.fingerprint();
        let left = engine.apply(&state, movement("e2", "e4")).unwrap();
        let right = engine.apply(&state, movement("e2", "e4")).unwrap();

        assert_eq!(state.fingerprint(), before, "input state was mutated");
        assert_eq!(left, right);
        assert_eq!(left.state.turn, Player::Black);
        assert_eq!(left.state.ply, 1);
        assert_eq!(left.state.board.get("e2".parse().unwrap()), None);
        let moved = left.state.board.get("e4".parse().unwrap()).unwrap();
        assert_eq!(moved.owner, Player::White);
        assert_eq!(moved.kind, PieceKind::Pawn);
        assert_ne!(moved.id, 0);
    }

    #[test]
    fn illegal_action_does_not_mutate_input() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = engine.new_game(Mode::Standard, 1).unwrap();
        let before = state.clone();
        assert!(engine.apply(&state, movement("e2", "e5")).is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn capturing_the_last_king_wins() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut board = Board::empty();
        board.set(
            "a1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook)),
        );
        board.set(
            "h1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King)),
        );
        board.set(
            "a8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King)),
        );
        let state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 1);
        let result = engine.apply(&state, movement("a1", "a8")).unwrap();
        assert_eq!(
            result.state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::KingCaptured
            }
        );
    }

    #[test]
    fn third_repetition_with_equal_stars_is_a_draw() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(1);
        let cycle = [("g1", "f3"), ("g8", "f6"), ("f3", "g1"), ("f6", "g8")];
        for _ in 0..2 {
            for (from, to) in cycle {
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
    fn lower_star_total_wins_on_third_repetition() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(1);
        state.half_star_totals = [2, 4];
        state.record_current_position();
        let cycle = [("g1", "f3"), ("g8", "f6"), ("f3", "g1"), ("f6", "g8")];
        for _ in 0..2 {
            for (from, to) in cycle {
                state = engine.apply(&state, movement(from, to)).unwrap().state;
            }
        }
        assert_eq!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::LowerStarsOnRepetition
            }
        );
    }

    #[test]
    fn pawn_promotion_exposes_four_deterministically_ordered_actions() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut board = Board::empty();
        board.set(
            "h1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King)),
        );
        board.set(
            "a7".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn)),
        );
        board.set(
            "h8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King)),
        );
        let state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 1);
        let promotions: Vec<_> = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .filter(|action| matches!(action, Action::Move(movement) if movement.from == "a7".parse().unwrap()))
            .collect();
        assert_eq!(promotions.len(), 4);
        assert!(promotions.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn engine_rejects_a_state_from_another_snapshot() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = GameState::standard("augment_chess_20990101", Mode::Standard, 1);
        assert!(matches!(
            engine.legal_actions(&state),
            Err(EngineError::RulesetMismatch { .. })
        ));
    }

    #[test]
    fn king_of_the_hill_is_passive_and_adds_queen_rays_only_from_the_center() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("d4", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1,
        );
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("d4", "d8"))
        );

        let state = engine
            .grant_card(&state, 10, KING_OF_THE_HILL, Player::White)
            .unwrap()
            .state;
        assert!(state.card_runtime.hill_king[Player::White.index()]);
        assert_eq!(state.half_star_totals, [5, 0]);
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("d4", "d8"))
        );

        let state = engine.apply(&state, movement("d4", "d8")).unwrap().state;
        let state = engine.apply(&state, movement("h8", "g8")).unwrap().state;
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("d8", "d1"))
        );
    }

    #[test]
    fn geneva_convention_blocks_only_enemy_queen_capturing_the_owners_pawn() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a4", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
                ("a8", Player::Black, PieceKind::Queen),
            ],
            Player::Black,
            1,
        );
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("a8", "a4"))
        );
        let state = engine
            .grant_card(&state, 11, GENEVA_CONVENTION, Player::White)
            .unwrap()
            .state;
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("a8", "a4"))
        );
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("a8", "a5"))
        );
    }

    #[test]
    fn sprint_adds_a_three_square_unmoved_pawn_move_without_en_passant() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("a2", Player::White, PieceKind::Pawn),
                ("h1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            41,
        );
        let state = engine
            .grant_card(&state, 101, SPRINT, Player::White)
            .unwrap()
            .state;
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("a2", "a5"))
        );

        let state = engine.apply(&state, movement("a2", "a5")).unwrap().state;
        assert!(state.en_passant.is_none());
    }

    #[test]
    fn early_promotion_advances_the_white_promotion_rank_by_two() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("e5", Player::White, PieceKind::Pawn),
                ("h1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            43,
        );
        let state = engine
            .grant_card(&state, 102, EARLY_PROMOTION, Player::White)
            .unwrap()
            .state;
        let promotions = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .filter(|action| {
                matches!(action, Action::Move(movement)
                    if movement.from == "e5".parse().unwrap()
                        && movement.to == "e6".parse().unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(promotions.len(), 4);
        assert!(promotions.iter().all(
            |action| matches!(action, Action::Move(movement) if movement.promotion.is_some())
        ));
    }

    #[test]
    fn initiative_blocks_opponent_captures_until_its_owner_captures_them() {
        let engine = Engine::bundled_20260807().unwrap();
        let blocked = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
                ("a8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            47,
        );
        let blocked = engine
            .grant_card(&blocked, 103, INITIATIVE, Player::White)
            .unwrap()
            .state;
        let blocked = engine.apply(&blocked, movement("h1", "g1")).unwrap().state;
        assert!(
            !engine
                .legal_actions(&blocked)
                .unwrap()
                .contains(&movement("a8", "a1"))
        );
        assert!(
            engine
                .legal_actions(&blocked)
                .unwrap()
                .contains(&movement("a8", "a7"))
        );

        let clearing = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
                ("a2", Player::Black, PieceKind::Pawn),
            ],
            Player::White,
            47,
        );
        let clearing = engine
            .grant_card(&clearing, 104, INITIATIVE, Player::White)
            .unwrap()
            .state;
        let clearing = engine.apply(&clearing, movement("a1", "a2")).unwrap().state;
        assert!(clearing.card_runtime.initiative[Player::Black.index()].is_none());
    }

    #[test]
    fn revelation_shortens_both_current_and_future_deathmatches() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(53);
        state.deathmatch = Some(DeathmatchState {
            started_at_shared_turn: 45,
            quiet_full_turns: 8,
            interval_full_turns: 10,
            progress_this_round: false,
        });
        let state = engine.apply_rule(&state, REVELATION).unwrap().state;
        assert!(state.card_runtime.revelation);
        assert_eq!(state.deathmatch.unwrap().interval_full_turns, 5);
        assert_eq!(state.deathmatch.unwrap().quiet_full_turns, 5);

        let mut future = standard_play_state(53);
        future.completed_drafts = [true; 3];
        future.turn = Player::Black;
        future.turns_taken = [45, 44];
        future = engine.apply_rule(&future, REVELATION).unwrap().state;
        let transition = engine.apply(&future, movement("g8", "f6")).unwrap();
        assert_eq!(transition.state.deathmatch.unwrap().interval_full_turns, 5);
    }

    #[test]
    fn disarm_blocks_capture_through_the_targets_next_completed_turn() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
                ("a8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            59,
        );
        let state = engine
            .grant_card(&state, 105, DISARM, Player::White)
            .unwrap()
            .state;
        let state = engine.apply(&state, use_card(105, "a8")).unwrap().state;
        let target_id = state.board.get("a8".parse().unwrap()).unwrap().id;
        let state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
        let black_actions = engine.legal_actions(&state).unwrap();
        assert!(!black_actions.contains(&movement("a8", "a1")));
        assert!(black_actions.contains(&movement("a8", "a7")));

        let state = engine.apply(&state, movement("a8", "a7")).unwrap().state;
        assert!(!state.card_runtime.disarmed.contains_key(&target_id));
    }

    #[test]
    fn dice_uses_one_deterministic_roll_and_locks_a_type_for_two_turns() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = engine
            .grant_card(&standard_play_state(61), 106, DICE, Player::White)
            .unwrap()
            .state;
        let transition = engine.apply(&state, use_card_without_target(106)).unwrap();
        assert_eq!(
            transition
                .events
                .iter()
                .filter(|event| matches!(event, Event::RandomSample { purpose, .. } if purpose == "dice"))
                .count(),
            1
        );
        let mut state = transition.state;
        let locked = state.card_runtime.dice_locks[Player::Black.index()].unwrap();
        assert_eq!(locked.remaining_turns, 2);
        state = engine.apply(&state, movement("e2", "e3")).unwrap().state;
        assert!(engine.legal_actions(&state).unwrap().iter().all(|action| {
            let Action::Move(movement) = action else {
                return true;
            };
            state
                .board
                .get(movement.from)
                .is_none_or(|piece| piece.kind != locked.kind)
        }));

        engine.tick_card_statuses_after_turn(&mut state, Player::Black);
        assert_eq!(
            state.card_runtime.dice_locks[Player::Black.index()]
                .unwrap()
                .remaining_turns,
            1
        );
        engine.tick_card_statuses_after_turn(&mut state, Player::Black);
        assert!(state.card_runtime.dice_locks[Player::Black.index()].is_none());
    }

    #[test]
    fn freeze_preserves_last_warmth_and_freezes_three_for_three_turns() {
        let engine = Engine::bundled_20260807().unwrap();
        let pieces = [
            ("h1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a7", Player::Black, PieceKind::Pawn),
            ("b7", Player::Black, PieceKind::Pawn),
            ("c7", Player::Black, PieceKind::Pawn),
            ("d7", Player::Black, PieceKind::Pawn),
            ("e7", Player::Black, PieceKind::Pawn),
        ];
        let state = engine
            .grant_card(
                &minimal_state(&pieces, Player::White, 67),
                107,
                FREEZE,
                Player::White,
            )
            .unwrap()
            .state;
        let transition = engine.apply(&state, use_card_without_target(107)).unwrap();
        assert_eq!(transition.state.card_runtime.frozen.len(), 3);
        assert_eq!(
            transition
                .events
                .iter()
                .filter(|event| matches!(event, Event::RandomSample { purpose, .. } if purpose.starts_with("freeze-shuffle:")))
                .count(),
            4
        );
        let frozen_ids = transition
            .state
            .card_runtime
            .frozen
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut state = engine
            .apply(&transition.state, movement("h1", "g1"))
            .unwrap()
            .state;
        assert!(engine.legal_actions(&state).unwrap().iter().all(|action| {
            let Action::Move(movement) = action else {
                return true;
            };
            state
                .board
                .get(movement.from)
                .is_none_or(|piece| !frozen_ids.contains(&piece.id))
        }));
        for expected in [2, 1] {
            engine.tick_card_statuses_after_turn(&mut state, Player::Black);
            assert!(
                state
                    .card_runtime
                    .frozen
                    .values()
                    .all(|status| status.remaining_turns == expected)
            );
        }
        engine.tick_card_statuses_after_turn(&mut state, Player::Black);
        assert!(state.card_runtime.frozen.is_empty());

        let last_warmth = minimal_state(&pieces[..6], Player::White, 67);
        let last_warmth = engine
            .grant_card(&last_warmth, 108, FREEZE, Player::White)
            .unwrap()
            .state;
        assert!(
            !engine
                .legal_actions(&last_warmth)
                .unwrap()
                .contains(&use_card_without_target(108))
        );
    }

    #[test]
    fn reposition_forces_a_second_non_capture_with_the_same_piece() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
                ("a3", Player::Black, PieceKind::Pawn),
            ],
            Player::White,
            71,
        );
        let state = engine
            .grant_card(&state, 109, REPOSITION, Player::White)
            .unwrap()
            .state;
        let state = engine.apply(&state, use_card(109, "a1")).unwrap().state;
        let rook_id = state.board.get("a1".parse().unwrap()).unwrap().id;
        let actions = engine.legal_actions(&state).unwrap();
        assert!(actions.contains(&movement("a1", "a2")));
        assert!(!actions.contains(&movement("a1", "a3")));

        let state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert_eq!(
            state.phase,
            GamePhase::ForcedMove {
                piece_id: rook_id,
                source: ForcedMoveSource::Effect(REPOSITION.to_owned()),
                continuation: ForcedMoveContinuation::FinishTurn,
            }
        );
        let actions = engine.legal_actions(&state).unwrap();
        assert!(actions.iter().all(
            |action| matches!(action, Action::Move(movement) if movement.from == "a2".parse().unwrap())
        ));
        assert!(!actions.contains(&movement("a2", "a3")));

        let state = engine.apply(&state, movement("a2", "b2")).unwrap().state;
        assert_eq!(state.turn, Player::Black);
        assert!(state.card_runtime.reposition.is_none());
    }

    #[test]
    fn assassin_keeps_knight_moves_and_can_ray_capture_only_a_royal() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("b1", Player::White, PieceKind::Knight),
                ("b8", Player::Black, PieceKind::King),
                ("h8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            73,
        );
        state.turns_taken = [2, 2];
        state = engine
            .grant_card(&state, 110, ASSASSIN, Player::White)
            .unwrap()
            .state;
        state = engine.apply(&state, use_card(110, "b1")).unwrap().state;
        assert_eq!(
            state.board.get("b1".parse().unwrap()).unwrap().kind,
            PieceKind::Assassin
        );
        let immediate = engine.legal_actions(&state).unwrap();
        assert!(immediate.contains(&movement("b1", "c3")));
        assert!(!immediate.contains(&movement("b1", "b8")));

        state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
        state = engine.apply(&state, movement("h8", "g8")).unwrap().state;
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("b1", "b8"))
        );
    }

    #[test]
    fn grasshopper_lands_exactly_one_square_beyond_the_first_hurdle() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Knight),
                ("d5", Player::White, PieceKind::Pawn),
                ("e4", Player::White, PieceKind::Pawn),
                ("f4", Player::Black, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            79,
        );
        let state = engine
            .grant_card(&state, 111, GRASSHOPPER, Player::White)
            .unwrap()
            .state;
        let state = engine.apply(&state, use_card(111, "d4")).unwrap().state;
        assert_eq!(
            state.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Grasshopper
        );
        let actions = engine.legal_actions(&state).unwrap();
        assert!(actions.contains(&movement("d4", "d6")));
        assert!(!actions.contains(&movement("d4", "f4")));
        assert!(!actions.contains(&movement("d4", "d5")));
        // A transformation cannot capture even on the opening turns. The lock expires
        // after the owner completes one turn; quiet grasshopper moves stay available.
        let mut ready = state.clone();
        ready.turns_taken[Player::White.index()] += 1;
        let actions = engine.legal_actions(&ready).unwrap();
        assert!(actions.contains(&movement("d4", "f4")));
    }

    #[test]
    fn judgment_targets_every_piece_tied_for_the_highest_positive_capture_count() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
                ("a8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            83,
        );
        for square in ["a1", "a8", "h8"] {
            let square = Square::from_str(square).unwrap();
            let mut piece = state.board.get(square).unwrap();
            piece.capture_count = 3;
            state.board.set(square, Some(piece));
        }
        state = engine
            .grant_card(&state, 112, JUDGMENT, Player::White)
            .unwrap()
            .state;
        let actions = engine.legal_actions(&state).unwrap();
        assert!(actions.contains(&use_card(112, "a1")));
        assert!(actions.contains(&use_card(112, "a8")));
        assert!(!actions.contains(&use_card(112, "h8")));

        let state = engine.apply(&state, use_card(112, "a8")).unwrap().state;
        assert!(state.board.get("a8".parse().unwrap()).is_none());
        assert_eq!(state.status, GameStatus::Ongoing);
    }

    #[test]
    fn cleanup_sacrifice_is_seeded_and_removes_enemy_before_own_piece() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h7", Player::Black, PieceKind::King),
                ("a8", Player::Black, PieceKind::Rook),
                ("b8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            89,
        );
        let state = engine
            .grant_card(&state, 113, CLEANUP_SACRIFICE, Player::White)
            .unwrap()
            .state;
        let left = engine.apply(&state, use_card(113, "a1")).unwrap();
        let right = engine.apply(&state, use_card(113, "a1")).unwrap();
        assert_eq!(left, right);
        assert!(left.state.board.get("a1".parse().unwrap()).is_none());
        assert_eq!(
            left.state
                .board
                .iter()
                .filter(|(_, piece)| {
                    piece.owner == Player::Black && piece.kind == PieceKind::Rook
                })
                .count(),
            1
        );
        let removals = left
            .events
            .iter()
            .filter_map(|event| match event {
                Event::PieceRemoved { piece, cause, .. } if cause == CLEANUP_SACRIFICE => {
                    Some(*piece)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(removals.len(), 2);
        assert_eq!(removals[0].owner, Player::Black);
        assert_eq!(removals[1].owner, Player::White);
    }

    #[test]
    fn parry_success_removes_the_attacker_and_restores_defender_at_attack_origin() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
                ("a8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            3,
        );
        let state = engine
            .grant_card(&state, 12, PARRY, Player::White)
            .unwrap()
            .state;
        let state = engine.apply(&state, use_card(12, "a1")).unwrap().state;
        let parrying_id = state.board.get("a1".parse().unwrap()).unwrap().id;
        assert_eq!(
            state.turn,
            Player::White,
            "active card must not consume the move"
        );
        let state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
        let transition = engine.apply(&state, movement("a8", "a1")).unwrap();
        let state = transition.state;

        assert_eq!(state.board.get("a1".parse().unwrap()), None);
        let restored = state.board.get("a8".parse().unwrap()).unwrap();
        assert_eq!(restored.id, parrying_id);
        assert_eq!(restored.owner, Player::White);
        assert!(!state.card_runtime.parry_piece_ids.contains(&parrying_id));
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::RandomSample {
                purpose,
                roll_53: 1_021_869_836_427_313
            } if purpose == "parry"
        )));
        assert_eq!(state.turn, Player::White);
        assert_eq!(state.fingerprint(), 0xe18e_15c9_9501_f591);
    }

    #[test]
    fn parry_failure_still_consumes_the_status_and_capture_continues() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
                ("a8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            1,
        );
        let state = engine
            .grant_card(&state, 13, PARRY, Player::White)
            .unwrap()
            .state;
        let state = engine.apply(&state, use_card(13, "a1")).unwrap().state;
        let parrying_id = state.board.get("a1".parse().unwrap()).unwrap().id;
        let state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
        let transition = engine.apply(&state, movement("a8", "a1")).unwrap();
        let state = transition.state;

        let attacker = state.board.get("a1".parse().unwrap()).unwrap();
        assert_eq!(attacker.owner, Player::Black);
        assert_eq!(state.board.get("a8".parse().unwrap()), None);
        assert!(!state.card_runtime.parry_piece_ids.contains(&parrying_id));
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::RandomSample {
                purpose,
                roll_53: 5_103_132_997_656_651
            } if purpose == "parry"
        )));
    }

    #[test]
    fn empty_lunchbox_removes_target_after_its_third_completed_turn() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("a8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            5,
        );
        let state = engine
            .grant_card(&state, 14, EMPTY_LUNCHBOX, Player::White)
            .unwrap()
            .state;
        let mut state = engine.apply(&state, use_card(14, "a8")).unwrap().state;
        for (white_move, black_move) in [
            (("h1", "g1"), ("a8", "a7")),
            (("g1", "f1"), ("a7", "a6")),
            (("f1", "e1"), ("a6", "a5")),
        ] {
            state = engine
                .apply(&state, movement(white_move.0, white_move.1))
                .unwrap()
                .state;
            state = engine
                .apply(&state, movement(black_move.0, black_move.1))
                .unwrap()
                .state;
        }
        assert_eq!(state.board.get("a5".parse().unwrap()), None);
        assert!(state.card_runtime.empty_lunchboxes.is_empty());
        assert_eq!(state.turns_taken[Player::Black.index()], 3);
        assert_eq!(state.fingerprint(), 0x54cf_4f5f_325e_f346);
    }

    #[test]
    fn empty_lunchbox_clears_early_as_soon_as_target_ends_adjacent_to_king() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("f8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            5,
        );
        let state = engine
            .grant_card(&state, 15, EMPTY_LUNCHBOX, Player::White)
            .unwrap()
            .state;
        let state = engine.apply(&state, use_card(15, "f8")).unwrap().state;
        let state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
        let state = engine.apply(&state, movement("f8", "g8")).unwrap().state;
        assert!(state.board.get("g8".parse().unwrap()).is_some());
        assert!(state.card_runtime.empty_lunchboxes.is_empty());
        assert_eq!(state.turns_taken[Player::Black.index()], 1);
    }

    #[test]
    fn platform_rule_spawns_immediately_using_web_candidate_order() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            3,
        );
        let transition = engine.apply_rule(&state, PLATFORM).unwrap();
        let platform = transition.state.card_runtime.platform.as_ref().unwrap();
        assert_eq!(platform.cell, Some("a7".parse().unwrap()));
        assert_eq!(platform.spawned_at, 0);
        assert_eq!(platform.next_at, 5);
        assert_eq!(platform.nonce, 1);
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::PlatformSpawned {
                square: Some(square),
                shared_turn: 0,
                nonce: 1
            } if *square == "a7".parse().unwrap()
        )));
    }

    #[test]
    fn platform_empty_candidate_bug_still_consumes_one_random_sample() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut board = Board::empty();
        for rank in 0..8 {
            for file in 0..8 {
                let square = Square::new(file, rank).unwrap();
                let piece = if square == "a1".parse().unwrap() {
                    Piece::new(Player::White, PieceKind::King)
                } else if square == "h8".parse().unwrap() {
                    Piece::new(Player::Black, PieceKind::King)
                } else {
                    Piece::new(Player::White, PieceKind::Pawn)
                };
                board.set(square, Some(piece));
            }
        }
        let state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 3);
        let rng_before = state.rng_state();
        let transition = engine.apply_rule(&state, PLATFORM).unwrap();
        let platform = transition.state.card_runtime.platform.as_ref().unwrap();
        assert_eq!(platform.cell, None);
        assert_ne!(transition.state.rng_state(), rng_before);
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::RandomSample {
                purpose,
                roll_53: 1_021_869_836_427_313
            } if purpose == "platform-cell"
        )));
    }

    #[test]
    fn platform_forces_the_same_piece_to_move_again_before_turn_ends() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            8,
        );
        let rook_id = state.board.get("a1".parse().unwrap()).unwrap().id;
        state.applied_rules.push(PLATFORM.to_owned());
        state.card_runtime.platform = Some(PlatformRuleState {
            cell: Some("a2".parse().unwrap()),
            spawned_at: 0,
            next_at: 5,
            triggered_piece_ids: BTreeSet::new(),
            nonce: 1,
        });
        let state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert_eq!(state.turn, Player::White);
        assert_eq!(state.turns_taken, [0, 0]);
        assert_eq!(
            state.phase,
            GamePhase::ForcedMove {
                piece_id: rook_id,
                source: ForcedMoveSource::Platform,
                continuation: ForcedMoveContinuation::FinishTurn,
            }
        );
        assert!(engine.legal_actions(&state).unwrap().iter().all(
            |action| matches!(action, Action::Move(movement) if movement.from == "a2".parse().unwrap())
        ));

        let state = engine.apply(&state, movement("a2", "a3")).unwrap().state;
        assert_eq!(state.phase, GamePhase::Playing);
        assert_eq!(state.turn, Player::Black);
        assert_eq!(state.turns_taken, [1, 0]);
        assert_eq!(state.ply, 2);
        assert_eq!(state.fingerprint(), 0x93a5_5b6b_07fe_343d);
    }

    #[test]
    fn platform_marks_piece_triggered_even_when_no_extra_move_exists() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Pawn),
                ("a4", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            8,
        );
        let pawn_id = state.board.get("a2".parse().unwrap()).unwrap().id;
        state.applied_rules.push(PLATFORM.to_owned());
        state.card_runtime.platform = Some(PlatformRuleState {
            cell: Some("a3".parse().unwrap()),
            spawned_at: 0,
            next_at: 5,
            triggered_piece_ids: BTreeSet::new(),
            nonce: 1,
        });
        let state = engine.apply(&state, movement("a2", "a3")).unwrap().state;
        assert_eq!(state.phase, GamePhase::Playing);
        assert_eq!(state.turn, Player::Black);
        assert!(
            state
                .card_runtime
                .platform
                .unwrap()
                .triggered_piece_ids
                .contains(&pawn_id)
        );
    }

    struct CaptureTax;

    impl EffectHandler for CaptureTax {
        fn definition_id(&self) -> &str {
            "test.capture_tax"
        }

        fn react(
            &self,
            card: &CardInstance,
            event: &Event,
            _state: &GameState,
        ) -> Vec<EffectCommand> {
            if matches!(event, Event::PieceCaptured { actor, .. } if *actor == card.owner) {
                vec![EffectCommand::AdjustHalfStars {
                    player: card.owner,
                    delta: 1,
                }]
            } else {
                Vec::new()
            }
        }
    }

    #[test]
    fn card_handlers_reduce_commands_in_the_engine() {
        let mut engine = Engine::bundled_20260807().unwrap();
        engine.effects_mut().register(CaptureTax).unwrap();
        let mut board = Board::empty();
        board.set(
            "a1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook)),
        );
        board.set(
            "h1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King)),
        );
        board.set(
            "a2".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Pawn)),
        );
        board.set(
            "h8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King)),
        );
        let mut state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 1);
        state.add_card(CardInstance {
            instance_id: 1,
            definition_id: "test.capture_tax".to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 0,
            available_after_turn: None,
        });
        let result = engine.apply(&state, movement("a1", "a2")).unwrap();
        assert_eq!(result.state.half_star_totals, [1, 0]);
        assert!(
            result
                .events
                .iter()
                .any(|event| matches!(event, Event::EffectApplied { .. }))
        );
    }

    struct ExtraMoveAfterE2E4;

    impl EffectHandler for ExtraMoveAfterE2E4 {
        fn definition_id(&self) -> &str {
            "test.extra_move"
        }

        fn react(
            &self,
            card: &CardInstance,
            event: &Event,
            _state: &GameState,
        ) -> Vec<EffectCommand> {
            if matches!(
                event,
                Event::ActionApplied {
                    actor,
                    action: Action::Move(movement),
                } if *actor == card.owner
                    && movement.from == "e2".parse().unwrap()
                    && movement.to == "e4".parse().unwrap()
            ) {
                vec![EffectCommand::GrantActions {
                    player: card.owner,
                    count: 1,
                }]
            } else {
                Vec::new()
            }
        }
    }

    #[test]
    fn action_budget_supports_multiple_moves_without_special_case_turn_code() {
        let mut engine = Engine::bundled_20260807().unwrap();
        engine.effects_mut().register(ExtraMoveAfterE2E4).unwrap();
        let mut state = standard_play_state(37);
        state.add_card(CardInstance {
            instance_id: 1,
            definition_id: "test.extra_move".to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 0,
            available_after_turn: None,
        });

        let first = engine.apply(&state, movement("e2", "e4")).unwrap();
        assert_eq!(first.state.turn, Player::White);
        assert_eq!(first.state.actions_remaining, 1);
        assert_eq!(first.state.turns_taken, [0, 0]);

        let second = engine.apply(&first.state, movement("g1", "f3")).unwrap();
        assert_eq!(second.state.turn, Player::Black);
        assert_eq!(second.state.actions_remaining, 1);
        assert_eq!(second.state.turns_taken, [1, 0]);
        assert!(second.events.iter().any(|event| matches!(
            event,
            Event::TurnEnding {
                player: Player::White
            }
        )));
        assert!(second.events.iter().any(|event| matches!(
            event,
            Event::TurnStarted {
                player: Player::Black,
                actions_remaining: 1
            }
        )));
    }

    struct ChooseStarAfterE2E4;

    impl EffectHandler for ChooseStarAfterE2E4 {
        fn definition_id(&self) -> &str {
            "test.choose_star"
        }

        fn react(
            &self,
            card: &CardInstance,
            event: &Event,
            _state: &GameState,
        ) -> Vec<EffectCommand> {
            if matches!(
                event,
                Event::ActionApplied {
                    actor,
                    action: Action::Move(movement),
                } if *actor == card.owner
                    && movement.from == "e2".parse().unwrap()
                    && movement.to == "e4".parse().unwrap()
            ) {
                vec![EffectCommand::RequestChoice {
                    owner: card.owner,
                    handler_id: "test.star_choice".to_owned(),
                    options: vec!["half-star".to_owned(), "one-star".to_owned()],
                    continuation: ChoiceContinuation::FinishTurn,
                }]
            } else {
                Vec::new()
            }
        }
    }

    struct StarChoice;

    impl ChoiceHandler for StarChoice {
        fn handler_id(&self) -> &str {
            "test.star_choice"
        }

        fn resolve(
            &self,
            choice: &PendingChoice,
            option_index: u16,
            _state: &GameState,
        ) -> Vec<EffectCommand> {
            vec![EffectCommand::AdjustHalfStars {
                player: choice.owner,
                delta: (option_index + 1) as i16,
            }]
        }
    }

    #[test]
    fn pending_choice_exposes_only_choice_actions_and_resumes_deterministically() {
        let mut engine = Engine::bundled_20260807().unwrap();
        engine.effects_mut().register(ChooseStarAfterE2E4).unwrap();
        engine.choices_mut().register(StarChoice).unwrap();
        let mut state = standard_play_state(41);
        state.add_card(CardInstance {
            instance_id: 7,
            definition_id: "test.choose_star".to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 0,
            available_after_turn: None,
        });

        let waiting = engine.apply(&state, movement("e2", "e4")).unwrap();
        let GamePhase::AwaitingChoice(choice) = &waiting.state.phase else {
            panic!("the move must suspend on a choice")
        };
        assert_eq!(waiting.state.turn, Player::White);
        assert_eq!(choice.options, ["half-star", "one-star"]);
        let choice_id = choice.id;
        assert_eq!(
            engine.legal_actions(&waiting.state).unwrap(),
            vec![
                Action::ResolveChoice {
                    choice_id,
                    option_index: 0,
                },
                Action::ResolveChoice {
                    choice_id,
                    option_index: 1,
                },
            ]
        );

        let resolved = engine
            .apply(
                &waiting.state,
                Action::ResolveChoice {
                    choice_id,
                    option_index: 1,
                },
            )
            .unwrap();
        assert_eq!(resolved.state.half_star_totals, [2, 0]);
        assert_eq!(resolved.state.phase, GamePhase::Playing);
        assert_eq!(resolved.state.turn, Player::Black);
        assert!(resolved.events.iter().any(|event| matches!(
            event,
            Event::ChoiceResolved {
                choice_id: resolved_id,
                owner: Player::White,
                option_index: 1
            } if *resolved_id == choice_id
        )));
    }

    struct ForceRookAfterLanding;

    impl EffectHandler for ForceRookAfterLanding {
        fn definition_id(&self) -> &str {
            "test.force_rook"
        }

        fn react(
            &self,
            card: &CardInstance,
            event: &Event,
            state: &GameState,
        ) -> Vec<EffectCommand> {
            if matches!(
                event,
                Event::ActionApplied {
                    actor,
                    action: Action::Move(movement),
                } if *actor == card.owner
                    && movement.from == "a1".parse().unwrap()
                    && movement.to == "a2".parse().unwrap()
            ) {
                let piece_id = state.board.get("a2".parse().unwrap()).unwrap().id;
                vec![
                    EffectCommand::GrantActions {
                        player: card.owner,
                        count: 1,
                    },
                    EffectCommand::ForceMove {
                        player: card.owner,
                        piece_id,
                        source: card.definition_id.clone(),
                        finish_turn: false,
                    },
                ]
            } else {
                Vec::new()
            }
        }
    }

    struct ForceBlockedRookAfterLanding;

    impl EffectHandler for ForceBlockedRookAfterLanding {
        fn definition_id(&self) -> &str {
            "test.force_blocked_rook"
        }

        fn react(
            &self,
            card: &CardInstance,
            event: &Event,
            state: &GameState,
        ) -> Vec<EffectCommand> {
            if matches!(
                event,
                Event::ActionApplied {
                    actor,
                    action: Action::Move(movement),
                } if *actor == card.owner
                    && movement.from == "a1".parse().unwrap()
                    && movement.to == "a2".parse().unwrap()
            ) {
                let piece_id = state.board.get("a2".parse().unwrap()).unwrap().id;
                vec![
                    EffectCommand::SpawnPiece {
                        square: "a1".parse().unwrap(),
                        piece: Piece::new(card.owner, PieceKind::Pawn),
                        replace: false,
                    },
                    EffectCommand::GrantActions {
                        player: card.owner,
                        count: 1,
                    },
                    EffectCommand::ForceMove {
                        player: card.owner,
                        piece_id,
                        source: card.definition_id.clone(),
                        finish_turn: false,
                    },
                ]
            } else {
                Vec::new()
            }
        }
    }

    #[test]
    fn forced_move_can_resume_the_remaining_normal_action_budget() {
        let mut engine = Engine::bundled_20260807().unwrap();
        engine
            .effects_mut()
            .register(ForceRookAfterLanding)
            .unwrap();
        let mut state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            43,
        );
        let rook_id = state.board.get("a1".parse().unwrap()).unwrap().id;
        state.add_card(CardInstance {
            instance_id: 9,
            definition_id: "test.force_rook".to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 0,
            available_after_turn: None,
        });

        state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert_eq!(
            state.phase,
            GamePhase::ForcedMove {
                piece_id: rook_id,
                source: ForcedMoveSource::Effect("test.force_rook".to_owned()),
                continuation: ForcedMoveContinuation::ResumeTurn,
            }
        );
        assert_eq!(state.actions_remaining, 1);

        state = engine.apply(&state, movement("a2", "a3")).unwrap().state;
        assert_eq!(state.phase, GamePhase::Playing);
        assert_eq!(state.turn, Player::White);
        assert_eq!(state.actions_remaining, 1);

        state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
        assert_eq!(state.turn, Player::Black);
        assert_eq!(state.turns_taken, [1, 0]);
    }

    #[test]
    fn empty_forced_move_resumes_before_the_transition_is_exposed() {
        let mut engine = Engine::bundled_20260807().unwrap();
        engine
            .effects_mut()
            .register(ForceBlockedRookAfterLanding)
            .unwrap();
        let mut state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("a3", Player::White, PieceKind::Pawn),
                ("b2", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            44,
        );
        state.add_card(CardInstance {
            instance_id: 10,
            definition_id: "test.force_blocked_rook".to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 0,
            available_after_turn: None,
        });

        let next = engine.apply(&state, movement("a1", "a2")).unwrap().state;

        assert_eq!(next.phase, GamePhase::Playing);
        assert_eq!(next.turn, Player::White);
        assert_eq!(next.actions_remaining, 1);
        assert!(!engine.legal_actions(&next).unwrap().is_empty());
    }

    #[test]
    fn a_playable_card_prevents_no_action_loss_until_it_is_used() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut board = Board::empty();
        for rank in 0..8 {
            for file in 0..8 {
                let square = Square::new(file, rank).unwrap();
                let piece = if square == "a8".parse().unwrap() {
                    Piece::new(Player::White, PieceKind::King)
                } else if square == "h1".parse().unwrap() {
                    Piece::new(Player::Black, PieceKind::King)
                } else {
                    Piece::new(Player::White, PieceKind::Pawn)
                };
                board.set(square, Some(piece));
            }
        }
        let state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::Black, 47);
        let state = engine
            .grant_card(&state, 11, PARRY, Player::White)
            .unwrap()
            .state;
        let state = engine.apply(&state, movement("h1", "g1")).unwrap().state;

        assert_eq!(state.status, GameStatus::Ongoing);
        assert_eq!(state.turn, Player::White);
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .iter()
                .all(|action| matches!(action, Action::UseCard { .. }))
        );

        let transition = engine.apply(&state, use_card(11, "a8")).unwrap();
        assert_eq!(
            transition.state.status,
            GameStatus::Won {
                winner: Player::Black,
                reason: WinReason::NoLegalActions,
            }
        );
    }

    #[test]
    fn all_single_target_part_one_cards_transform_the_expected_piece() {
        let engine = Engine::bundled_20260807().unwrap();
        let cases = [
            (
                CONSTITUTIONAL_MONARCHY,
                PieceKind::Queen,
                PieceKind::PrimeMinister,
            ),
            (DRAGON, PieceKind::Rook, PieceKind::Dragon),
            (EASTERN_POLICY, PieceKind::Knight, PieceKind::Cannon),
            (HERALD, PieceKind::Rook, PieceKind::Herald),
            (KNIGHTMASTER, PieceKind::Knight, PieceKind::Knightmaster),
            (PEGASUS, PieceKind::Knight, PieceKind::Unicorn),
            (IDOL, PieceKind::Queen, PieceKind::Idol),
            (JESTER, PieceKind::Queen, PieceKind::Jester),
            (LOCAL_CONSCRIPTION, PieceKind::Queen, PieceKind::Recruiter),
            (REAPER, PieceKind::Queen, PieceKind::Reaper),
            (WIZARD, PieceKind::Queen, PieceKind::Wizard),
            (LOG, PieceKind::Pawn, PieceKind::Log),
            (MISSIONARY, PieceKind::Bishop, PieceKind::Missionary),
            (ORDINATION, PieceKind::Bishop, PieceKind::Cardinal),
            (STANDARD_BEARER, PieceKind::Pawn, PieceKind::StandardBearer),
        ];
        for (index, (card, source, expected)) in cases.into_iter().enumerate() {
            let mut pieces = vec![
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, source),
                ("h8", Player::Black, PieceKind::King),
            ];
            if card == ORDINATION {
                pieces.push(("c3", Player::White, PieceKind::Bishop));
            }
            let mut state = minimal_state(&pieces, Player::White, index as u64 + 1);
            grant_test_card(&mut state, card, 100 + index as u32);
            let transition = engine
                .apply(&state, use_card(100 + index as u32, "d4"))
                .unwrap();
            assert_eq!(
                transition
                    .state
                    .board
                    .get("d4".parse().unwrap())
                    .unwrap()
                    .kind,
                expected,
                "{card}"
            );
        }
    }

    #[test]
    fn composite_piece_cards_apply_sacrifices_and_double_transform() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut amazon_state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Queen),
                ("b2", Player::White, PieceKind::Knight),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            9,
        );
        grant_test_card(&mut amazon_state, AMAZON, 1);
        amazon_state = engine
            .apply(
                &amazon_state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Squares(vec![
                        "d4".parse().unwrap(),
                        "b2".parse().unwrap(),
                    ]),
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            amazon_state.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Amazon
        );
        assert!(amazon_state.board.get("b2".parse().unwrap()).is_none());

        let mut hook_state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Queen),
                ("b2", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            10,
        );
        grant_test_card(&mut hook_state, HOOK, 2);
        hook_state = engine
            .apply(
                &hook_state,
                Action::UseCard {
                    instance_id: 2,
                    target: ActionTarget::Squares(vec![
                        "d4".parse().unwrap(),
                        "b2".parse().unwrap(),
                    ]),
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            hook_state.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Hook
        );
        assert!(hook_state.board.get("b2".parse().unwrap()).is_none());

        let mut windmill_state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("c3", Player::White, PieceKind::Bishop),
                ("f3", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            11,
        );
        grant_test_card(&mut windmill_state, WINDMILL, 3);
        windmill_state = engine
            .apply(
                &windmill_state,
                Action::UseCard {
                    instance_id: 3,
                    target: ActionTarget::Squares(vec![
                        "c3".parse().unwrap(),
                        "f3".parse().unwrap(),
                    ]),
                },
            )
            .unwrap()
            .state;
        for square in ["c3", "f3"] {
            let piece = windmill_state.board.get(square.parse().unwrap()).unwrap();
            assert_eq!(piece.kind, PieceKind::Windmill);
            assert_eq!(
                windmill_state.card_runtime.windmill_modes.get(&piece.id),
                Some(&WindmillMode::Bishop)
            );
        }
    }

    #[test]
    fn special_piece_move_generators_cover_rays_leaps_bends_and_bounces() {
        let engine = Engine::bundled_20260807().unwrap();
        let cases = [
            (PieceKind::Amazon, "d4", "d8"),
            (PieceKind::Amazon, "d4", "f5"),
            (PieceKind::PrimeMinister, "d4", "f5"),
            (PieceKind::Alibaba, "d4", "f6"),
            (PieceKind::Protestant, "d4", "g7"),
            (PieceKind::Hook, "d4", "f5"),
            (PieceKind::Cardinal, "a2", "b1"),
        ];
        for (kind, from, to) in cases {
            let state = minimal_state(
                &[
                    ("a1", Player::White, PieceKind::King),
                    (from, Player::White, kind),
                    ("h8", Player::Black, PieceKind::King),
                ],
                Player::White,
                3,
            );
            assert!(has_move(&engine, &state, from, to), "{kind:?} {from}-{to}");
        }

        let cannon = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Cannon),
                ("a2", Player::White, PieceKind::Pawn),
                ("a5", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        assert!(has_move(&engine, &cannon, "a1", "a3"));
        assert!(has_move(&engine, &cannon, "a1", "a5"));
        assert!(!has_move(&engine, &cannon, "a1", "a2"));

        let unicorn = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Unicorn),
                ("h7", Player::Black, PieceKind::Rook),
                ("f5", Player::Black, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            5,
        );
        assert!(has_move(&engine, &unicorn, "d4", "a7"));
        assert!(has_move(&engine, &unicorn, "d4", "f5"));
        assert!(!has_move(&engine, &unicorn, "d4", "h7"));
    }

    #[test]
    fn dragon_recruiter_missionary_and_windmill_apply_move_side_effects() {
        let engine = Engine::bundled_20260807().unwrap();
        let dragon = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Dragon),
                ("g7", Player::White, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1,
        );
        let dragon_id = dragon.board.get("d4".parse().unwrap()).unwrap().id;
        let bishop_id = dragon.board.get("g7".parse().unwrap()).unwrap().id;
        let dragon = engine.apply(&dragon, movement("d4", "g7")).unwrap().state;
        assert_eq!(
            dragon.board.get("g7".parse().unwrap()).unwrap().id,
            dragon_id
        );
        assert_eq!(
            dragon.board.get("d4".parse().unwrap()).unwrap().id,
            bishop_id
        );

        let recruiter = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Recruiter),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            2,
        );
        let recruiter = engine
            .apply(&recruiter, movement("d4", "e5"))
            .unwrap()
            .state;
        assert_eq!(
            recruiter.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );

        let missionary = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Missionary),
                ("e5", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            3,
        );
        let missionary = engine
            .apply(&missionary, movement("d4", "e5"))
            .unwrap()
            .state;
        assert_eq!(
            missionary.board.get("e5".parse().unwrap()).unwrap().owner,
            Player::White
        );
        assert_eq!(
            missionary.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Missionary
        );

        let mut windmill = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Windmill),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        let windmill_id = windmill.board.get("d4".parse().unwrap()).unwrap().id;
        windmill
            .card_runtime
            .windmill_modes
            .insert(windmill_id, WindmillMode::Bishop);
        windmill = engine.apply(&windmill, movement("d4", "e5")).unwrap().state;
        assert_eq!(
            windmill.card_runtime.windmill_modes.get(&windmill_id),
            Some(&WindmillMode::Rook)
        );
    }

    #[test]
    fn knightmaster_and_standard_bearer_modify_friendly_pawn_moves() {
        let engine = Engine::bundled_20260807().unwrap();
        let knightmaster = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Knightmaster),
                ("e4", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1,
        );
        assert!(has_move(&engine, &knightmaster, "e4", "f6"));
        assert!(!has_move(&engine, &knightmaster, "e4", "e5"));

        let bearer = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("b3", Player::White, PieceKind::Pawn),
                ("f3", Player::White, PieceKind::StandardBearer),
                ("c3", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            2,
        );
        assert!(has_move(&engine, &bearer, "b3", "c3"));
    }

    #[test]
    fn idol_grants_one_extra_action_and_blocks_the_same_piece_repeat() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Idol),
                ("e4", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            8,
        );
        let rook_id = state.board.get("e4".parse().unwrap()).unwrap().id;
        let state = engine.apply(&state, movement("e4", "e5")).unwrap().state;
        assert_eq!(state.turn, Player::White);
        assert_eq!(state.actions_remaining, 1);
        assert!(state.card_runtime.idol_rest_piece_ids.contains(&rook_id));
        assert!(
            engine.legal_actions(&state).unwrap().iter().all(
                |action| !matches!(action, Action::Move(m) if m.from == "e5".parse().unwrap())
            )
        );
    }

    #[test]
    fn mass_piece_cards_transform_and_spawn_deterministically() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut eagle = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("b2", Player::White, PieceKind::Knight),
                ("f2", Player::White, PieceKind::Knight),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            17,
        );
        grant_test_card(&mut eagle, EAGLE, 1);
        eagle = engine
            .apply(&eagle, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(
            eagle
                .board
                .iter()
                .filter(|(_, piece)| {
                    piece.owner == Player::White && piece.kind == PieceKind::Alibaba
                })
                .count(),
            4
        );

        let mut reformation = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("c2", Player::White, PieceKind::Bishop),
                ("f2", Player::White, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            18,
        );
        grant_test_card(&mut reformation, REFORMATION, 2);
        reformation = engine
            .apply(&reformation, use_card_without_target(2))
            .unwrap()
            .state;
        assert_eq!(
            reformation
                .board
                .iter()
                .filter(|(_, piece)| piece.kind == PieceKind::Protestant)
                .count(),
            2
        );
    }

    #[test]
    fn log_waits_until_the_end_of_its_owners_next_turn_before_rolling() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Log),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            22,
        );
        let log_id = state.board.get("d4".parse().unwrap()).unwrap().id;
        let state = engine.apply(&state, movement("d4", "e4")).unwrap().state;
        assert_eq!(state.turn, Player::Black);
        assert_eq!(state.board.get("d4".parse().unwrap()).unwrap().id, log_id);
        assert!(state.board.get("e4".parse().unwrap()).is_none());

        let state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        assert_eq!(state.turn, Player::White);
        assert_eq!(state.board.get("d4".parse().unwrap()).unwrap().id, log_id);

        let state = engine.apply(&state, movement("a1", "b1")).unwrap().state;
        assert_eq!(state.board.get("e4".parse().unwrap()).unwrap().id, log_id);
        assert!(state.board.get("d4".parse().unwrap()).is_none());
    }

    #[test]
    fn wizard_mana_spells_shields_hazards_and_time_stop_are_stateful() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut lightning = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Wizard),
                ("e5", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            30,
        );
        let wizard_id = lightning.board.get("d4".parse().unwrap()).unwrap().id;
        lightning.card_runtime.wizard_mana.insert(wizard_id, 1);
        lightning = engine
            .apply(
                &lightning,
                Action::UsePieceAbility {
                    piece_id: wizard_id,
                    ability: PieceAbility::WizardLightning,
                    target: ActionTarget::Square("e5".parse().unwrap()),
                },
            )
            .unwrap()
            .state;
        assert!(lightning.board.get("e5".parse().unwrap()).is_some());
        lightning = engine
            .apply(&lightning, movement("h8", "h7"))
            .unwrap()
            .state;
        assert!(lightning.board.get("e5".parse().unwrap()).is_none());

        let mut shield = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Wizard),
                ("c4", Player::White, PieceKind::Pawn),
                ("c5", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            31,
        );
        let wizard_id = shield.board.get("d4".parse().unwrap()).unwrap().id;
        let pawn_id = shield.board.get("c4".parse().unwrap()).unwrap().id;
        shield.card_runtime.wizard_mana.insert(wizard_id, 2);
        shield = engine
            .apply(
                &shield,
                Action::UsePieceAbility {
                    piece_id: wizard_id,
                    ability: PieceAbility::WizardShield,
                    target: ActionTarget::PieceAt("c4".parse().unwrap()),
                },
            )
            .unwrap()
            .state;
        shield = engine.apply(&shield, movement("c5", "c4")).unwrap().state;
        assert_eq!(shield.board.get("c4".parse().unwrap()).unwrap().id, pawn_id);
        assert_eq!(
            shield.board.get("c5".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        assert!(!shield.card_runtime.shielded_piece_ids.contains(&pawn_id));

        let mut stopped = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Wizard),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            32,
        );
        let wizard_id = stopped.board.get("d4".parse().unwrap()).unwrap().id;
        stopped.card_runtime.wizard_mana.insert(wizard_id, 5);
        stopped = engine
            .apply(
                &stopped,
                Action::UsePieceAbility {
                    piece_id: wizard_id,
                    ability: PieceAbility::WizardTimeStop,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        assert_eq!(stopped.turn, Player::White);
        assert_eq!(stopped.actions_remaining, 1);
        stopped = engine.apply(&stopped, movement("d4", "e4")).unwrap().state;
        assert_eq!(stopped.turn, Player::White);
        assert_eq!(stopped.turns_taken, [1, 0]);
    }

    #[test]
    fn capturing_a_friendly_wizard_piece_grants_mana() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Wizard),
                ("c4", Player::White, PieceKind::Pawn),
                ("c5", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::Black,
            33,
        );
        let wizard_id = state.board.get("d4".parse().unwrap()).unwrap().id;
        state.card_runtime.wizard_mana.insert(wizard_id, 0);
        state = engine.apply(&state, movement("c5", "c4")).unwrap().state;
        assert_eq!(state.card_runtime.wizard_mana.get(&wizard_id), Some(&1));
    }

    #[test]
    fn fourth_reaper_soul_executes_the_enemy_king() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Reaper),
                ("e5", Player::Black, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            40,
        );
        let reaper_id = state.board.get("d4".parse().unwrap()).unwrap().id;
        state.card_runtime.reaper_souls.insert(reaper_id, 3);
        let transition = engine.apply(&state, movement("d4", "e5")).unwrap();
        assert_eq!(
            transition.state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect("reaper-four-souls".to_owned()),
            }
        );
        assert_eq!(
            transition
                .state
                .board
                .get("h8".parse().unwrap())
                .unwrap()
                .id,
            reaper_id
        );
    }

    #[test]
    fn herald_transform_next_to_enemy_king_wins_immediately() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Rook),
                ("e5", Player::Black, PieceKind::King),
            ],
            Player::White,
            41,
        );
        grant_test_card(&mut state, HERALD, 1);
        let transition = engine.apply(&state, use_card(1, "d4")).unwrap();
        assert_eq!(
            transition.state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect("herald-arrival".to_owned()),
            }
        );
    }

    #[test]
    fn jester_can_only_capture_a_royal_and_only_a_royal_can_capture_it() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Jester),
                ("e4", Player::Black, PieceKind::Rook),
                ("d8", Player::Black, PieceKind::King),
            ],
            Player::White,
            42,
        );
        assert!(!has_move(&engine, &state, "d4", "e4"));
        assert!(has_move(&engine, &state, "d4", "d8"));

        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Jester),
                ("d5", Player::Black, PieceKind::Rook),
                ("e5", Player::Black, PieceKind::King),
            ],
            Player::Black,
            43,
        );
        assert!(!has_move(&engine, &state, "d5", "d4"));
        assert!(has_move(&engine, &state, "e5", "d4"));
    }

    #[test]
    fn opening_pawn_transform_cards_follow_original_files() {
        let engine = Engine::bundled_20260807().unwrap();
        for (card, expected) in [
            (APPRENTICE_KNIGHTS, PieceKind::Squire),
            (CHECKER, PieceKind::Checker),
        ] {
            let mut state = standard_play_state(101);
            grant_test_card(&mut state, card, 1);
            let state = engine
                .apply(&state, use_card_without_target(1))
                .unwrap()
                .state;
            assert_eq!(
                state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        let owner = if card == CHECKER {
                            Player::Black
                        } else {
                            Player::White
                        };
                        piece.owner == owner && piece.kind == expected
                    })
                    .count(),
                2
            );
        }

        let mut state = standard_play_state(102);
        grant_test_card(&mut state, GUARD, 1);
        let state = engine
            .apply(&state, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(
            state.board.get("e2".parse().unwrap()).unwrap().kind,
            PieceKind::Guard
        );

        let mut state = standard_play_state(103);
        grant_test_card(&mut state, QUEEN_CAVALRY, 1);
        let state = engine
            .apply(&state, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(
            state.board.get("d2".parse().unwrap()).unwrap().kind,
            PieceKind::Knight
        );
    }

    #[test]
    fn every_part_two_opening_card_executes_from_the_standard_opening() {
        let engine = Engine::bundled_20260807().unwrap();
        let cards = [
            APPRENTICE_KNIGHTS,
            BIG_ROOK,
            CALLING_CARD,
            CHECKER,
            DEMOCRACY,
            DUTCH,
            ELEPHANT_ESCAPE,
            FAST_GROWTH,
            FIANCHETTO,
            GUARD,
            HOLDOUT,
            HORDE,
            IRON_MONARCH,
            KNIGHTMATE,
            LAST_STAND,
            LONDON_SYSTEM,
            LOYALIST,
            MARTYRDOM,
            MERCHANT_GUILD,
            OTHERWORLD,
            PAWN_CONVERSION,
            QUEEN_CAVALRY,
            QUEENS_GAMBIT,
            QXE1,
            SUMMON_COLOSSUS,
        ];
        for (index, card) in cards.into_iter().enumerate() {
            let instance_id = 500 + index as u32;
            let mut state = standard_play_state(200 + index as u64);
            grant_test_card(&mut state, card, instance_id);
            let action = engine
                .legal_actions(&state)
                .unwrap()
                .into_iter()
                .find(|action| {
                    matches!(action, Action::UseCard { instance_id: id, .. } if *id == instance_id)
                })
                .unwrap_or_else(|| panic!("{card} has no legal opening action"));
            let transition = engine
                .apply(&state, action)
                .unwrap_or_else(|error| panic!("{card} failed: {error}"));
            assert_eq!(
                transition
                    .state
                    .cards
                    .iter()
                    .find(|owned| owned.instance_id == instance_id)
                    .unwrap()
                    .charges,
                0,
                "{card} did not consume its charge"
            );
        }
    }

    #[test]
    fn squire_fanatic_alfil_and_checker_moves_match_the_bundle() {
        let engine = Engine::bundled_20260807().unwrap();
        let squire = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Squire),
                ("e5", Player::Black, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            104,
        );
        let squire = engine.apply(&squire, movement("d4", "e5")).unwrap().state;
        assert_eq!(
            squire.board.get("e5".parse().unwrap()).unwrap().kind,
            PieceKind::Knight
        );

        let fanatic = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Fanatic),
                ("d6", Player::Black, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            105,
        );
        assert!(has_move(&engine, &fanatic, "d4", "d5"));
        assert!(has_move(&engine, &fanatic, "d4", "d6"));

        let alfil = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Alfil),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            106,
        );
        assert!(has_move(&engine, &alfil, "d4", "f6"));
        assert!(!has_move(&engine, &alfil, "d4", "e5"));

        let checker = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("c3", Player::White, PieceKind::Checker),
                ("d4", Player::Black, PieceKind::Pawn),
                ("f6", Player::Black, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            107,
        );
        let first = engine.apply(&checker, movement("c3", "e5")).unwrap().state;
        assert!(matches!(first.phase, GamePhase::ForcedMove { .. }));
        assert!(has_move(&engine, &first, "e5", "g7"));
    }

    #[test]
    fn checker_jump_lands_and_breaks_a_consumable_shield_without_capturing() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("c3", Player::White, PieceKind::Checker),
                ("d4", Player::Black, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            108,
        );
        let shielded = state.board.get("d4".parse().unwrap()).unwrap();
        state.card_runtime.shielded_piece_ids.insert(shielded.id);

        let state = engine.apply(&state, movement("c3", "e5")).unwrap().state;

        assert_eq!(
            state.board.get("e5".parse().unwrap()).unwrap().kind,
            PieceKind::Checker
        );
        assert_eq!(state.board.get("d4".parse().unwrap()), Some(shielded));
        assert!(!state.card_runtime.shielded_piece_ids.contains(&shielded.id));
    }

    #[test]
    fn promotion_rush_checker_two_square_diagonal_is_not_a_capture_jump() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Checker),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            109,
        );
        let checker = state.board.get("a2".parse().unwrap()).unwrap();
        state
            .card_runtime
            .promotion_rush_piece_ids
            .insert(checker.id);

        assert!(has_move(&engine, &state, "a2", "c4"));
        let state = engine.apply(&state, movement("a2", "c4")).unwrap().state;

        assert_eq!(state.board.get("c4".parse().unwrap()), Some(checker));
        assert_eq!(state.board.get("b3".parse().unwrap()), None);
    }

    #[test]
    fn mass_opening_cards_build_expected_armies() {
        let engine = Engine::bundled_20260807().unwrap();

        let mut dutch = standard_play_state(108);
        grant_test_card(&mut dutch, DUTCH, 1);
        let dutch = engine
            .apply(&dutch, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(
            dutch
                .board
                .iter()
                .filter(
                    |(_, piece)| piece.owner == Player::White && piece.kind == PieceKind::Windmill
                )
                .count(),
            6
        );

        let mut horde = standard_play_state(109);
        grant_test_card(&mut horde, HORDE, 1);
        let horde = engine
            .apply(&horde, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(
            horde
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == Player::White && piece.kind == PieceKind::Pawn)
                .count(),
            35
        );

        let mut london = standard_play_state(110);
        grant_test_card(&mut london, LONDON_SYSTEM, 1);
        let london = engine
            .apply(&london, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(
            london.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        assert_eq!(
            london.board.get("f4".parse().unwrap()).unwrap().kind,
            PieceKind::Bishop
        );
        assert_eq!(
            london
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == Player::White)
                .count(),
            16
        );
    }

    #[test]
    fn big_rook_and_colossus_have_large_piece_state() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut rook = standard_play_state(111);
        grant_test_card(&mut rook, BIG_ROOK, 1);
        let rook = engine
            .apply(&rook, use_card_without_target(1))
            .unwrap()
            .state;
        let piece = rook.board.get("g2".parse().unwrap()).unwrap();
        assert_eq!(piece.kind, PieceKind::BigRook);
        assert_eq!(rook.card_runtime.large_pieces[&piece.id].hp, 2);

        let mut colossus = standard_play_state(112);
        grant_test_card(&mut colossus, SUMMON_COLOSSUS, 1);
        let colossus = engine
            .apply(&colossus, use_card_without_target(1))
            .unwrap()
            .state;
        let piece = colossus.board.get("d3".parse().unwrap()).unwrap();
        assert_eq!(piece.kind, PieceKind::Colossus);
        assert_eq!(colossus.card_runtime.large_pieces[&piece.id].hp, 3);
        assert_eq!(
            colossus
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == Player::White && piece.kind == PieceKind::Pawn)
                .count(),
            2
        );
    }

    #[test]
    fn colossus_sector_skips_protected_targets_without_breaking_their_shield() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(113);
        grant_test_card(&mut state, SUMMON_COLOSSUS, 1);
        let mut state = engine
            .apply(&state, use_card_without_target(1))
            .unwrap()
            .state;
        let colossus = state.board.get("d3".parse().unwrap()).unwrap();
        let target_square = "g5".parse().unwrap();
        let target = Piece::new(Player::Black, PieceKind::Pawn).with_id(state.allocate_piece_id());
        state.board.set(target_square, Some(target));
        state.card_runtime.shielded_piece_ids.insert(target.id);
        let action = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .find(|action| {
                matches!(
                    action,
                    Action::UsePieceAbility {
                        piece_id,
                        ability: PieceAbility::ColossusSector,
                        ..
                    } if *piece_id == colossus.id
                )
            })
            .unwrap();

        let state = engine.apply(&state, action).unwrap().state;

        assert_eq!(state.board.get(target_square), Some(target));
        assert!(state.card_runtime.shielded_piece_ids.contains(&target.id));
    }

    #[test]
    fn democracy_knightmate_qxe1_and_merchant_replace_royalty() {
        let engine = Engine::bundled_20260807().unwrap();

        let mut knightmate = standard_play_state(113);
        grant_test_card(&mut knightmate, KNIGHTMATE, 1);
        let knightmate = engine
            .apply(&knightmate, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(
            knightmate.board.get("e1".parse().unwrap()).unwrap().kind,
            PieceKind::RoyalKnight
        );
        assert_eq!(
            knightmate
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == Player::White && piece.kind == PieceKind::Man)
                .count(),
            2
        );

        let mut qxe1 = standard_play_state(114);
        grant_test_card(&mut qxe1, QXE1, 1);
        let qxe1 = engine
            .apply(&qxe1, use_card_without_target(1))
            .unwrap()
            .state;
        let heir = qxe1.board.get("e1".parse().unwrap()).unwrap();
        assert_eq!(heir.kind, PieceKind::Queen);
        assert!(qxe1.card_runtime.royal_piece_ids.contains(&heir.id));

        let mut royal_augments = qxe1.clone();
        royal_augments.turn = Player::White;
        royal_augments.actions_remaining = 1;
        royal_augments.card_runtime.horse_riding[Player::White.index()] = true;
        royal_augments.card_runtime.iron_monarch[Player::White.index()] = true;
        royal_augments.board.set(
            "e2".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Pawn)),
        );
        assert!(has_move(&engine, &royal_augments, "e1", "f3"));
        royal_augments = engine
            .apply(&royal_augments, movement("e1", "e2"))
            .unwrap()
            .state;
        assert_eq!(royal_augments.turn, Player::White);
        assert_eq!(royal_augments.actions_remaining, 1);

        let mut royal_exclusion = qxe1.clone();
        grant_test_card(&mut royal_exclusion, DISARM, 2);
        assert!(
            !engine
                .legal_actions(&royal_exclusion)
                .unwrap()
                .contains(&use_card(2, "e1"))
        );

        let mut merchant = standard_play_state(115);
        grant_test_card(&mut merchant, MERCHANT_GUILD, 1);
        let merchant = engine
            .apply(&merchant, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(
            merchant.board.get("e3".parse().unwrap()).unwrap().kind,
            PieceKind::Merchant
        );

        let mut democracy = minimal_state(
            &[
                ("a2", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            116,
        );
        democracy.card_runtime.democracy[Player::White.index()] = true;
        engine.resolve_missing_king(&mut democracy, &mut Vec::new());
        assert_eq!(democracy.status, GameStatus::Ongoing);
        democracy.board.set("a2".parse().unwrap(), None);
        engine.resolve_missing_king(&mut democracy, &mut Vec::new());
        assert!(matches!(
            democracy.status,
            GameStatus::Won {
                winner: Player::Black,
                ..
            }
        ));
    }

    #[test]
    fn pawn_conversion_fast_growth_fianchetto_and_iron_monarch_modify_moves() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut conversion = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Pawn),
                ("d5", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            117,
        );
        conversion.card_runtime.pawn_conversion[Player::White.index()] = true;
        assert!(has_move(&engine, &conversion, "d4", "d5"));
        assert!(has_move(&engine, &conversion, "d4", "e5"));

        let mut growth = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("e4", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            118,
        );
        growth.card_runtime.fast_growth[Player::White.index()] = true;
        let promotions = engine
            .legal_actions(&growth)
            .unwrap()
            .into_iter()
            .filter(|action| matches!(action, Action::Move(m) if m.from == "e4".parse().unwrap() && m.to == "e5".parse().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(promotions.len(), 2);

        let mut fianchetto = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("b1", Player::White, PieceKind::Bishop),
                ("a8", Player::Black, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            119,
        );
        fianchetto.card_runtime.fianchetto[Player::Black.index()] = true;
        assert!(has_move(&engine, &fianchetto, "b1", "e4"));

        let mut iron = minimal_state(
            &[
                ("d4", Player::White, PieceKind::King),
                ("e5", Player::Black, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            120,
        );
        iron.card_runtime.iron_monarch[Player::White.index()] = true;
        let iron = engine.apply(&iron, movement("d4", "e5")).unwrap().state;
        assert_eq!(iron.turn, Player::White);
        assert_eq!(iron.actions_remaining, 1);

        for (kind, extra) in [
            (PieceKind::Fanatic, true),
            (PieceKind::Rook, false),
            (PieceKind::Bishop, false),
            (PieceKind::Knight, false),
            (PieceKind::Queen, false),
        ] {
            let mut iron = minimal_state(
                &[
                    ("d4", Player::White, PieceKind::King),
                    ("e5", Player::Black, kind),
                    ("h8", Player::Black, PieceKind::King),
                ],
                Player::White,
                120,
            );
            iron.card_runtime.iron_monarch[Player::White.index()] = true;
            let next = engine.apply(&iron, movement("d4", "e5")).unwrap().state;
            assert_eq!(next.turn == Player::White, extra, "{kind:?}");
        }

        let mut blocked_pawn = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("d3", Player::White, PieceKind::Pawn),
                ("a1", Player::Black, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            119,
        );
        blocked_pawn.card_runtime.fianchetto[Player::Black.index()] = true;
        assert!(!has_move(&engine, &blocked_pawn, "d3", "d4"));
    }

    #[test]
    fn delayed_opening_cards_resolve_on_bundle_timers() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut holdout = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            121,
        );
        let pawn = holdout.board.get("a2".parse().unwrap()).unwrap();
        holdout.card_runtime.holdout_promotions.insert(pawn.id, 14);
        holdout.turns_taken = [14, 14];
        engine.resolve_holdout_promotions(&mut holdout, Player::White);
        assert_eq!(
            holdout.board.get("a2".parse().unwrap()).unwrap().kind,
            PieceKind::Queen
        );

        let mut otherworld = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("a2", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            122,
        );
        otherworld
            .card_runtime
            .pending_otherworld
            .push(PendingOtherworld {
                piece_id: 77,
                owner: Player::White,
                square: "a2".parse().unwrap(),
                origin: "b2".parse().unwrap(),
                due_ply: 28,
            });
        otherworld.ply = 28;
        let mut events = Vec::new();
        engine.resolve_after_counted_move(&mut otherworld, &mut events);
        let wizard = otherworld.board.get("a2".parse().unwrap()).unwrap();
        assert_eq!(wizard.kind, PieceKind::Wizard);
        assert_eq!(wizard.id, 77);
        assert_eq!(wizard.capture_count, 1);
    }

    #[test]
    fn shields_loyalist_and_calling_card_apply_followup_effects() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut gambit = standard_play_state(123);
        grant_test_card(&mut gambit, QUEENS_GAMBIT, 1);
        let gambit = engine.apply(&gambit, use_card(1, "d1")).unwrap().state;
        assert!(gambit.board.get("d1".parse().unwrap()).is_none());
        let d_pawn = gambit.board.get("d2".parse().unwrap()).unwrap();
        assert!(gambit.card_runtime.protected_piece_ids.contains(&d_pawn.id));
        assert_eq!(gambit.card_runtime.protected_piece_ids.len(), 2);

        let mut martyrdom = standard_play_state(124);
        grant_test_card(&mut martyrdom, MARTYRDOM, 1);
        let martyrdom = engine
            .apply(&martyrdom, use_card_without_target(1))
            .unwrap()
            .state;
        assert!(!martyrdom.board.contains(Player::White, PieceKind::Bishop));
        assert_eq!(martyrdom.card_runtime.shielded_piece_ids.len(), 8);

        let mut loyalist = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a4", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            125,
        );
        let rook = loyalist.board.get("a4".parse().unwrap()).unwrap();
        loyalist.card_runtime.loyalist_piece_ids.insert(rook.id);
        assert!(has_move(&engine, &loyalist, "a4", "d1"));

        let mut calling = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("a8", Player::Black, PieceKind::Rook),
                ("b8", Player::Black, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            126,
        );
        let rook = calling.board.get("a8".parse().unwrap()).unwrap();
        calling
            .card_runtime
            .calling_cards
            .insert(rook.id, Player::White);
        calling.board.set("a8".parse().unwrap(), None);
        let mut events = Vec::new();
        engine.resolve_calling_card_capture(&mut calling, rook, None, &mut events);
        assert!(calling.board.get("b8".parse().unwrap()).is_none());
    }

    #[test]
    fn every_active_middle_card_exposes_and_executes_a_legal_action() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut pieces = vec![
            ("e1", Player::White, PieceKind::King),
            ("d1", Player::White, PieceKind::Queen),
            ("a1", Player::White, PieceKind::Rook),
            ("h1", Player::White, PieceKind::Rook),
            ("c3", Player::White, PieceKind::Bishop),
            ("f3", Player::White, PieceKind::Bishop),
            ("b3", Player::White, PieceKind::Knight),
            ("g3", Player::White, PieceKind::Knight),
            ("e8", Player::Black, PieceKind::King),
            ("d8", Player::Black, PieceKind::Queen),
            ("a8", Player::Black, PieceKind::Rook),
            ("h8", Player::Black, PieceKind::Rook),
            ("c6", Player::Black, PieceKind::Bishop),
            ("f6", Player::Black, PieceKind::Bishop),
            ("b6", Player::Black, PieceKind::Knight),
            ("g6", Player::Black, PieceKind::Knight),
        ];
        for square in ["a4", "b4", "c4", "d4"] {
            pieces.push((square, Player::White, PieceKind::Pawn));
        }
        // An advanced White piece gives half-locked target cards (outpost) a legal target.
        pieces.push(("d5", Player::White, PieceKind::Pawn));
        for square in ["e5", "f5", "g5", "h5"] {
            pieces.push((square, Player::Black, PieceKind::Pawn));
        }
        let base = minimal_state(&pieces, Player::White, 20_260_807);

        for definition in IMPLEMENTED_CARDS.iter().filter(|definition| {
            definition.phase == CardPhase::Middle && definition.activation == CardActivation::Active
        }) {
            let mut state = base.clone();
            grant_test_card(&mut state, definition.id, 9_000);
            let action = engine
                .legal_actions(&state)
                .unwrap_or_else(|error| {
                    panic!("{} action generation failed: {error}", definition.id)
                })
                .into_iter()
                .find(|action| {
                    matches!(
                        action,
                        Action::UseCard {
                            instance_id: 9_000,
                            ..
                        }
                    )
                })
                .unwrap_or_else(|| panic!("{} exposed no card action", definition.id));
            engine
                .apply(&state, action)
                .unwrap_or_else(|error| panic!("{} failed to execute: {error}", definition.id));
        }
    }

    #[test]
    fn middle_move_providers_and_capture_filters_change_legal_moves() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut leap = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Pawn),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1,
        );
        engine.apply_card_definition(&mut leap, LEAP, Player::White, &mut Vec::new());
        assert!(has_move(&engine, &leap, "a2", "b4"));

        let pawn = leap.board.get("a2".parse().unwrap()).unwrap();
        leap.board.set(
            "a1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook)),
        );
        leap.card_runtime.ghost_piece_ids.insert(pawn.id);
        assert!(has_move(&engine, &leap, "a1", "a3"));

        let mut armistice = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("a8", Player::Black, PieceKind::Queen),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            2,
        );
        assert!(has_move(&engine, &armistice, "a1", "a8"));
        armistice.card_runtime.armistice_remaining_ply = 4;
        assert!(!has_move(&engine, &armistice, "a1", "a8"));
    }

    #[test]
    fn poisoned_pawn_stuns_its_capturer_for_two_future_turns() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("a7", Player::Black, PieceKind::Pawn),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            3,
        );
        let pawn = state.board.get("a7".parse().unwrap()).unwrap();
        let rook = state.board.get("a1".parse().unwrap()).unwrap();
        state.card_runtime.poisoned_piece_ids.insert(pawn.id);
        let state = engine.apply(&state, movement("a1", "a7")).unwrap().state;
        assert_eq!(
            state
                .card_runtime
                .frozen
                .get(&rook.id)
                .map(|status| status.remaining_turns),
            Some(2)
        );
    }

    #[test]
    fn black_magic_keeps_the_dark_wizard_as_the_royal_piece() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(4);
        let rook = state.board.get("a1".parse().unwrap()).unwrap();
        state.card_runtime.large_pieces.insert(
            rook.id,
            LargePieceStatus {
                anchor: "a1".parse().unwrap(),
                hp: 2,
                max_hp: 2,
            },
        );
        let mut events = Vec::new();
        engine
            .resolve_black_tower_legacy(&mut state, &mut events)
            .unwrap();
        assert!(state.board.contains(Player::White, PieceKind::DarkWizard));
        assert!(!state.board.contains(Player::White, PieceKind::King));
        assert!(!state.card_runtime.large_pieces.contains_key(&rook.id));
        assert_eq!(
            state
                .board
                .iter()
                .filter(
                    |(_, piece)| piece.owner == Player::White && piece.kind == PieceKind::Monster
                )
                .count(),
            8
        );
    }

    #[test]
    fn insight_reveals_enemy_stealth_pieces() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("e5", Player::Black, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            922,
        );
        let bishop = state.board.get("e5".parse().unwrap()).unwrap();
        state.card_runtime.stealth_piece_ids.insert(bishop.id);
        grant_test_card(&mut state, crate::INSIGHT, 1);
        let revealed = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        assert!(!revealed.card_runtime.stealth_piece_ids.contains(&bishop.id));
    }

    // Live source: disassembleMovedQueen (main-6qeAcxbX.js L6028591) — with disassembly active,
    // a moved non-royal queen becomes a bishop on its destination and a rook is created on the
    // origin square if it is empty and unblocked. state.disassembly[movingColor] resets after
    // the move batch (js "if (state.disassembly) state.disassembly[movingColor] = false").
    // Live source: septemberOvertakeCanCross (main-6qeAcxbX.js) — while overtake is active for
    // the mover, a rook treats friendly pieces (except wall/football/monster/blackHole/coffin)
    // as transparent in rayMoves, landing after them normally.
    #[test]
    fn overtake_lets_the_rook_cross_friendly_pieces_for_this_turn() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut board = crate::Board::empty();
        for (square, owner, kind) in [
            ("a1", Player::White, PieceKind::Rook),
            ("a2", Player::White, PieceKind::Pawn),
            ("e1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a4", Player::Black, PieceKind::Pawn),
        ] {
            board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let without = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::White,
            927,
        );
        let actions = engine.legal_actions(&without).unwrap();
        assert!(!actions.contains(&movement("a1", "a3")));
        let with = engine
            .grant_card(&without, 1, "overtake", Player::White)
            .unwrap()
            .state;
        // The effect arms only after the card is actually used (active card).
        assert!(
            !engine
                .legal_actions(&with)
                .unwrap()
                .contains(&movement("a1", "a3"))
        );
        let activated = engine
            .apply(&with, use_card_without_target(1))
            .unwrap()
            .state;
        let actions = engine.legal_actions(&activated).unwrap();
        assert!(actions.contains(&movement("a1", "a3")));
        assert!(actions.contains(&movement("a1", "a4")));
        // After the mover's turn completes, the turn-scoped effect is cleared again.
        let after_move = engine
            .apply(&activated, movement("a1", "a3"))
            .unwrap()
            .state;
        assert_eq!(
            after_move.board.get("a3".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        assert_eq!(
            after_move.board.get("a2".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        let next_own = engine
            .apply(&after_move, movement("h8", "h7"))
            .unwrap()
            .state;
        assert_eq!(next_own.turn, Player::White);
        // a3→a1 requires crossing the friendly pawn on a2, so it is blocked again
        // once the turn-scoped overtake effect has been cleared.
        let own_again = engine.legal_actions(&next_own).unwrap();
        assert!(!own_again.contains(&movement("a3", "a1")));
    }

    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L1114-1119: "reversal", "END", 4, "사용한 턴에 아군 룩과 비숍의 행마가 뒤바뀝니다."
    // L86549-86553: case "reversal" sets state.reversal[state.turn] = true.
    // L82195/L82204: bishop uses rookDirections; rook uses bishopDirections while armed.
    // L80215: if (state.reversal) state.reversal[movingColor] = false;
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L1049-1055: outpost, MIDDLE, 3.5, own-outpost-piece (active default).
    // L1197-1199: White target cells must reach the opponent half of the board.
    // L86120: rejects royal identity, already protected, and rule-object targets.
    // L86127: item2.outpostProtected = true; L95452: protection is recognized.
    // L1244-1246: delete outpostProtected only when the piece's occupied cells change.
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L4131-4133: symmetry is MIDDLE 3.5, active by default, for the current turn.
    // L4156-4157: destination is { row, col: columns - 1 - col }.
    // L4186-4188: card use sets state2[card2.id][color2] = true.
    // L101339-101341: friendly landing is rejected; capturable enemy landing is allowed.
    // L82366-82367: mirror moves are added to normal moves. L80208 clears the flag.
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L3973/L3982: active END 4.0, own-extinction; delete the selected type on both sides.
    // L709-710: minor targets are exactly knight, bishop, camel in the current catalog.
    // L8858: requires a friendly nonroyal eligible target.
    // L8859-8862: victims are ALL entries whose item.type equals selected.type, then remove.
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L1121-1127: miracle, MIDDLE 4.5, active by default, no target in the current catalog.
    // L7431/73698: using miracle ends the user's turn (END_TURN).
    // L91144-91152: targets are enemy pieces a friendly bishop could capture right now
    //   (getLegalMoves of that bishop), skipping encouraged pieces.
    // L91157-91177: every such enemy piece is converted to the mover's color; ALL friendly
    //   bishops (no selection in this catalog) are then removed as sacrifices.
    // L73698: miracle is in the endsTurn card list, so the user's turn ends.
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L4131 card$k(..., passive2 = false default); L4136 mutation is END 2.5 with passive
    //   TRUE, so it arms on acquisition without a UseCard action.
    // L4159-4160: canMutationPromote requires a pawn whose color has mutation armed and
    //   which reaches the last rank (white row 0 / black row rows-1).
    // L95714/L95725: promotionChoicesFor appends "monster" to the standard four choices.
    // L79065: a monster promotion result is marked as an allied monster (아군 취급).
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L3981: long-en-passant is MIDDLE 2.0, passive (card$l ... , true), so it arms on grant.
    // L101313-101321: addInternalPawnMoves merges longEnPassantCandidates when armed.
    // L4035-4048: a candidate needs an active right whose landing is one rank ahead of the
    //   capturer, a different file, and the victim pawn on the right's captured square.
    // L2839-2864: the right exists only after a standard double step (or sprint triple).
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L4135: clockwork, PIECE 4, own-minor. L4198-4202: friendly nonroyal
    // bishop/knight/camel -> card2.id, moved=true, freshNoCaptureUntil set.
    // L4148-4153: clockworkHasNeighbor requires a different friendly piece in
    // the surrounding eight squares. L82179-82180: movement uses queen rays.
    #[test]
    fn clockwork_neighbor_gate_precedes_extra_move_providers() {
        let engine = Engine::bundled_20260818().unwrap();
        let base = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Clockwork),
                ("d7", Player::White, PieceKind::Rook),
            ],
            Player::White,
            935,
        );
        let origin: Square = "d4".parse().unwrap();
        // The live getLegalMoves neighbor gate precedes all movement suppliers.
        for provider in 0..3 {
            let mut state = base.clone();
            match provider {
                0 => state.card_runtime.socialism[Player::White.index()] = 1,
                1 => state.card_runtime.symmetry[Player::White.index()] = true,
                _ => state.card_runtime.relay[Player::White.index()] = true,
            }
            assert!(
                !engine.legal_actions(&state).unwrap().iter().any(|action| {
                    matches!(action, Action::Move(movement) if movement.from == origin)
                }),
                "provider {provider} bypassed Clockwork's neighbor gate"
            );
            let neighbor_id = state.allocate_piece_id();
            state.board.set(
                "c4".parse().unwrap(),
                Some(Piece::new(Player::White, PieceKind::Pawn).with_id(neighbor_id)),
            );
            assert!(
                engine.legal_actions(&state).unwrap().iter().any(|action| {
                    matches!(action, Action::Move(movement) if movement.from == origin)
                }),
                "provider {provider} remained blocked with a real ally neighbor"
            );
        }
        let mut shadow_only = base.clone();
        let rook = shadow_only.board.get("d7".parse().unwrap()).unwrap();
        shadow_only
            .card_runtime
            .quantum_positions
            .insert(rook.id, "c4".parse().unwrap());
        assert!(
            !engine
                .legal_actions(&shadow_only)
                .unwrap()
                .iter()
                .any(|action| {
                    matches!(action, Action::Move(movement) if movement.from == origin)
                }),
            "a quantum shadow is not a raw-board neighbor"
        );
    }

    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L4137 catalog own-rook; L4196-4200 own nonroyal rook -> parrot, moved=true.
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L52916 initializes parrotMovement to {white:null,black:null}; L4206 gives no base moves.
    // L59312-59315 setLastMove records every relocation, even before card acquisition.
    // L4168-4182 records own ability type; a Parrot retains previous memory.
    // L4196-4200 transforms the rook without resetting memory.
    // L4212 permits enemy captures unless quiet; L4231 Clockwork=queen rays,
    // L4240 Princess=diagonal step, L4283 Campfire=quiet orthogonal,
    // L4225-4230 Pawn=forward/diagonal capture with double-step only if unmoved.
    // L78887 records before Trickster reroll at L78977-78979; L4170 uses sampled ability.
    #[test]
    fn parrot_remembers_own_preacquisition_moves_and_copies_only_base_movement() {
        let engine = Engine::bundled_20260818().unwrap();
        let base = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Rook),
                ("e7", Player::Black, PieceKind::Pawn),
            ],
            Player::White,
            945,
        );
        let unremembered = engine
            .grant_card(&base, 1, "parrot", Player::White)
            .unwrap()
            .state;
        let unremembered = engine
            .apply(&unremembered, use_card(1, "d4"))
            .unwrap()
            .state;
        assert!(
            !engine
                .legal_actions(&unremembered)
                .unwrap()
                .iter()
                .any(|action| {
                    matches!(action, Action::Move(mv) if mv.from == "d4".parse::<Square>().unwrap())
                })
        );

        // Record a rook move BEFORE owning the card. Preserve one action for the Parrot.
        let mut before_card = base.clone();
        before_card.actions_remaining = 2;
        let remembered = engine
            .apply(&before_card, movement("d4", "d5"))
            .unwrap()
            .state;
        assert_eq!(remembered.turn, Player::White);
        let granted = engine
            .grant_card(&remembered, 2, "parrot", Player::White)
            .unwrap()
            .state;
        let transformed = engine.apply(&granted, use_card(2, "d5")).unwrap().state;
        let actions = engine.legal_actions(&transformed).unwrap();
        assert!(actions.contains(&movement("d5", "e5")));
        assert!(!actions.contains(&movement("d5", "e6")));
        let moved = engine
            .apply(&transformed, movement("d5", "e5"))
            .unwrap()
            .state;
        assert_eq!(moved.turn, Player::Black);
        let next = engine.apply(&moved, movement("h8", "h7")).unwrap().state;
        assert_eq!(next.turn, Player::White);
        // Parrot's own move kept the rook memory; enemy movement did not overwrite it.
        let next_moves = engine.legal_actions(&next).unwrap();
        assert!(next_moves.contains(&movement("e5", "e7")));
        assert!(!next_moves.contains(&movement("e5", "f6")));
        let capture = engine.apply(&next, movement("e5", "e7")).unwrap().state;
        assert_eq!(
            capture.board.get("e7".parse().unwrap()).unwrap().kind,
            PieceKind::Parrot
        );
        assert_eq!(
            capture.board.get("e7".parse().unwrap()).unwrap().owner,
            Player::White
        );

        // Each case records an actual source move, then checks the Parrot's base mapping.
        for (index, kind, destination, parrot_square, enemy_square, allowed, forbidden) in [
            (
                0,
                PieceKind::Clockwork,
                "b4",
                "d4",
                "d6",
                vec!["e5", "d6"],
                vec!["f5"],
            ),
            (
                1,
                PieceKind::Princess,
                "c4",
                "d4",
                "e5",
                vec!["e5", "c3"],
                vec!["d5", "f6"],
            ),
            (
                2,
                PieceKind::Campfire,
                "b4",
                "d4",
                "e4",
                vec!["d5", "c4"],
                vec!["e4", "e5", "d6"],
            ),
            (
                3,
                PieceKind::Pawn,
                "b4",
                "d2",
                "e3",
                vec!["d3", "e3"],
                vec!["d4", "c3", "d1"],
            ),
            (
                4,
                PieceKind::Trickster,
                "c5",
                "d4",
                "f5",
                vec!["f5", "e6"],
                vec!["d5", "e5"],
            ),
        ] {
            let mut state = minimal_state(
                &[
                    ("a1", Player::White, PieceKind::King),
                    ("h8", Player::Black, PieceKind::King),
                    ("b3", Player::White, kind),
                    ("a3", Player::White, PieceKind::Pawn),
                    (parrot_square, Player::White, PieceKind::Rook),
                    (enemy_square, Player::Black, PieceKind::Pawn),
                ],
                Player::White,
                946 + index,
            );
            state.actions_remaining = 2;
            if kind == PieceKind::Trickster {
                let id = state.board.get("b3".parse().unwrap()).unwrap().id;
                state
                    .card_runtime
                    .trickster_move_kinds
                    .insert(id, PieceKind::Knight);
            }
            let recorded = engine
                .apply(&state, movement("b3", destination))
                .unwrap()
                .state;
            assert_eq!(recorded.turn, Player::White);
            let granted = engine
                .grant_card(&recorded, 3, "parrot", Player::White)
                .unwrap()
                .state;
            let mut copied = engine
                .apply(&granted, use_card(3, parrot_square))
                .unwrap()
                .state;
            // Isolate base movement from source aura/neighbors and fresh-transform capture lock.
            copied.board.set(destination.parse().unwrap(), None);
            copied.board.set("a3".parse().unwrap(), None);
            let parrot_id = copied.board.get(parrot_square.parse().unwrap()).unwrap().id;
            copied
                .card_runtime
                .fresh_no_capture_until
                .remove(&parrot_id);
            let legal = engine.legal_actions(&copied).unwrap();
            for target in allowed {
                assert!(
                    legal.contains(&movement(parrot_square, target)),
                    "{kind:?}: missing {target}"
                );
            }
            for target in forbidden {
                assert!(
                    !legal.contains(&movement(parrot_square, target)),
                    "{kind:?}: unexpected {target}"
                );
            }
        }
    }

    #[test]
    fn september_six_parrot_transformation_and_basic_behavior() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Rook),
                ("g7", Player::Black, PieceKind::Rook),
                ("b2", Player::White, PieceKind::Pawn),
            ],
            Player::White,
            936,
        );
        let granted = engine
            .grant_card(&state, 1, "parrot", Player::White)
            .unwrap()
            .state;
        assert_eq!(
            granted.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        let targets = engine.legal_actions(&granted).unwrap();
        assert!(targets.contains(&use_card(1, "d4")));
        for square in ["a1", "g7", "b2"] {
            assert!(!targets.contains(&use_card(1, square)));
        }
        let changed = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        let result = changed.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(result.kind, PieceKind::Parrot);
        assert_eq!(result.owner, Player::White);
        assert_eq!(changed.turn, Player::White);
    }

    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L3851 catalog own-knight; L3903-3914 transforms own nonroyal knight.
    // L82170-82171: knight leaps. L3932: captures prohibited.
    #[test]
    fn september_six_paladin_transformation_and_basic_behavior() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Knight),
                ("g7", Player::Black, PieceKind::Knight),
                ("b2", Player::White, PieceKind::Pawn),
            ],
            Player::White,
            937,
        );
        let granted = engine
            .grant_card(&state, 1, "paladin", Player::White)
            .unwrap()
            .state;
        assert_eq!(
            granted.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Knight
        );
        let targets = engine.legal_actions(&granted).unwrap();
        assert!(targets.contains(&use_card(1, "d4")));
        for square in ["a1", "g7", "b2"] {
            assert!(!targets.contains(&use_card(1, square)));
        }
        let changed = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        let result = changed.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(result.kind, PieceKind::Paladin);
        assert_eq!(result.owner, Player::White);
        assert_eq!(changed.turn, Player::White);
        let moves = engine.legal_actions(&changed).unwrap();
        assert!(moves.contains(&movement("d4", "f5")));
        assert!(!moves.contains(&movement("d4", "d5")));
    }

    #[test]
    fn paladin_radiance_blocks_dark_square_routes_and_captures() {
        let engine = Engine::bundled_20260818().unwrap();

        let capture = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Paladin),
                ("f5", Player::Black, PieceKind::Pawn),
            ],
            Player::White,
            947,
        );
        let capture_moves = engine.legal_actions(&capture).unwrap();
        assert!(capture_moves.contains(&movement("d4", "b5")));
        assert!(!capture_moves.contains(&movement("d4", "f5")));

        // Rust e4 is a light square. Its eight-neighbor radiance includes d4/d5.
        // A dark-square bishop may reach c3, but may not land on or cross radiance.
        let dark = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("e4", Player::White, PieceKind::Paladin),
                ("h8", Player::Black, PieceKind::King),
                ("a1", Player::Black, PieceKind::Bishop),
                ("c1", Player::Black, PieceKind::Knight),
            ],
            Player::Black,
            948,
        );
        let dark_moves = engine.legal_actions(&dark).unwrap();
        assert!(dark_moves.contains(&movement("a1", "c3")));
        assert!(!dark_moves.contains(&movement("a1", "d4")));
        assert!(!dark_moves.contains(&movement("a1", "g7")));
        assert!(dark_moves.contains(&movement("c1", "a2")));
        assert!(!dark_moves.contains(&movement("c1", "d3")));

        // A piece that starts on a light square is exempt from enemy radiance.
        let light = minimal_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("e4", Player::White, PieceKind::Paladin),
                ("h8", Player::Black, PieceKind::King),
                ("a2", Player::Black, PieceKind::Bishop),
            ],
            Player::Black,
            949,
        );
        assert!(
            engine
                .legal_actions(&light)
                .unwrap()
                .contains(&movement("a2", "g8"))
        );

        // Trickster uses its sampled Paladin ability as a radiance source.
        let mut trickster = dark.clone();
        let paladin = trickster.board.get("e4".parse().unwrap()).unwrap();
        let mut sampled = paladin;
        sampled.kind = PieceKind::Trickster;
        trickster.board.set("e4".parse().unwrap(), Some(sampled));
        trickster
            .card_runtime
            .trickster_move_kinds
            .insert(sampled.id, PieceKind::Paladin);
        assert!(
            !engine
                .legal_actions(&trickster)
                .unwrap()
                .contains(&movement("a1", "d4"))
        );
    }

    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L1085-1090 catalog; L86533-86534 transforms rook.
    // L1172/L82667-82668: one-step diagonals with own queen; queen rays without one.
    #[test]
    fn september_six_princess_transformation_and_basic_behavior() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Rook),
                ("g7", Player::Black, PieceKind::Rook),
                ("b2", Player::White, PieceKind::Pawn),
                ("h2", Player::White, PieceKind::Queen),
            ],
            Player::White,
            938,
        );
        let granted = engine
            .grant_card(&state, 1, "princess", Player::White)
            .unwrap()
            .state;
        assert_eq!(
            granted.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        let targets = engine.legal_actions(&granted).unwrap();
        assert!(targets.contains(&use_card(1, "d4")));
        for square in ["a1", "g7", "b2"] {
            assert!(!targets.contains(&use_card(1, square)));
        }
        let changed = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        let result = changed.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(result.kind, PieceKind::Princess);
        assert_eq!(result.owner, Player::White);
        assert_eq!(changed.turn, Player::White);
        let moves = engine.legal_actions(&changed).unwrap();
        assert!(moves.contains(&movement("d4", "e5")));
        assert!(!moves.contains(&movement("d4", "d6")));
        assert!(!moves.contains(&movement("d4", "f6")));
        let mut no_queen = changed.clone();
        no_queen.board.set("h2".parse().unwrap(), None);
        let empowered = engine.legal_actions(&no_queen).unwrap();
        assert!(empowered.contains(&movement("d4", "d6")));
        assert!(empowered.contains(&movement("d4", "f6")));
    }

    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L3852 catalog own-rook; L3903-3914 transforms own nonroyal rook.
    // L82173-82174: jumpMoves with eight unit queenDirections.
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L3903-3916: transformation does not set submerged.
    // L3881-3890: completeThreeTurn grants Octopus submerge without an enemy in eight neighbors.
    // L79945-79953: remaining extra actions return before completeThreeTurn.
    // L85283-85296: an adjacent enemy clears submerged; L84953 blocks capture while submerged.
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L79901-79911: free-move endMove handling occurs after refreshSubmergedPieces (L79897).
    // L85283-85296: any adjacent enemy deletes submerged, rather than suspending protection.
    // Regression: a due-premove sequence must observe its intermediate board positions.
    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L79897 refreshSubmergedPieces precedes autoAdvanceLogs (L79930) and
    // applyDelayedHazards (L79971). L85283-85296 permanently clears adjacent submerge.
    // This tests reveal ordering only, not the separate Octopus grant-vs-hazard order.
    #[test]
    fn submerged_octopus_stays_revealed_when_the_approaching_enemy_dies_to_a_hazard() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Octopus),
                ("d8", Player::Black, PieceKind::Rook),
            ],
            Player::Black,
            944,
        );
        let octopus_id = state.board.get("d4".parse().unwrap()).unwrap().id;
        let rook_id = state.board.get("d8".parse().unwrap()).unwrap().id;
        state.card_runtime.submerged_piece_ids.insert(octopus_id);
        state
            .card_runtime
            .delayed_hazards
            .push(crate::DelayedHazard {
                kind: crate::DelayedHazardKind::Lightning,
                owner: Player::White,
                trigger_after: Player::Black,
                squares: vec!["d5".parse().unwrap()],
            });
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&movement("d8", "d5"))
        );
        let transition = engine.apply(&state, movement("d8", "d5")).unwrap();
        // Removal at d5 proves the rook reached the adjacent square before the hazard.
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::PieceRemoved { piece, at, cause, .. }
                if piece.id == rook_id && *at == "d5".parse::<Square>().unwrap()
                    && cause == "wizard-lightning"
        )));
        let after = transition.state;
        assert_eq!(after.turn, Player::White);
        assert!(after.board.get("d8".parse().unwrap()).is_none());
        assert!(after.board.get("d5".parse().unwrap()).is_none());
        assert_eq!(
            after.board.get("d4".parse().unwrap()).unwrap().id,
            octopus_id
        );
        assert!(after.card_runtime.delayed_hazards.is_empty());
        assert!(!after.card_runtime.submerged_piece_ids.contains(&octopus_id));
    }

    #[test]
    fn due_premoves_reveal_submerged_octopus_even_when_the_enemy_moves_away() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Octopus),
                ("d8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            943,
        );
        let octopus_id = state.board.get("d4".parse().unwrap()).unwrap().id;
        let rook_id = state.board.get("d8".parse().unwrap()).unwrap().id;
        state.card_runtime.submerged_piece_ids.insert(octopus_id);
        state.turns_taken[Player::White.index()] = 1;
        state
            .card_runtime
            .pending_premoves
            .push(crate::PendingPremove {
                owner: Player::Black,
                trigger_player: Player::White,
                trigger_turn: 1,
                moves: vec![
                    MoveAction::new("d8".parse().unwrap(), "d5".parse().unwrap(), None),
                    MoveAction::new("d5".parse().unwrap(), "d8".parse().unwrap(), None),
                ],
            });
        let mut events = Vec::new();
        engine.resolve_due_premoves_after_turn(&mut state, Player::White, &mut events);
        // Both noncapturing moves ran: d5 was adjacent to d4, d8 is not.
        for (from, to) in [("d8", "d5"), ("d5", "d8")] {
            let expected = format!("premove:{}:{}->{}", rook_id, from, to);
            assert!(events.iter().any(|event| matches!(
                event, Event::EffectApplied { command, .. } if command == &expected
            )));
        }
        assert_eq!(state.turn, Player::White);
        assert!(state.card_runtime.pending_premoves.is_empty());
        assert_eq!(state.board.get("d8".parse().unwrap()).unwrap().id, rook_id);
        assert!(state.board.get("d5".parse().unwrap()).is_none());
        assert_eq!(
            state.board.get("d4".parse().unwrap()).unwrap().id,
            octopus_id
        );
        assert!(!state.card_runtime.submerged_piece_ids.contains(&octopus_id));
        engine.refresh_submerged_pieces(&mut state);
        assert!(!state.card_runtime.submerged_piece_ids.contains(&octopus_id));
    }

    #[test]
    fn octopus_submerges_only_after_its_completed_turn_and_reveals_next_to_an_enemy() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Rook),
                ("d8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            942,
        );
        let granted = engine
            .grant_card(&state, 1, "octopus", Player::White)
            .unwrap()
            .state;
        let mut transformed = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        let octopus_id = transformed.board.get("d4".parse().unwrap()).unwrap().id;
        assert_eq!(
            transformed.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Octopus
        );
        assert!(
            !transformed
                .card_runtime
                .submerged_piece_ids
                .contains(&octopus_id)
        );

        // Two actions allow a genuine within-turn boundary to be tested.
        transformed.actions_remaining = 2;
        let between = engine
            .apply(&transformed, movement("a1", "b1"))
            .unwrap()
            .state;
        assert_eq!(between.turn, Player::White);
        assert_eq!(between.actions_remaining, 1);
        assert!(
            !between
                .card_runtime
                .submerged_piece_ids
                .contains(&octopus_id)
        );

        // d8 is four ranks from d4: no enemy neighbor, but an unobstructed capture ray.
        let completed = engine.apply(&between, movement("b1", "a1")).unwrap().state;
        assert_eq!(completed.turn, Player::Black);
        assert!(
            completed
                .card_runtime
                .submerged_piece_ids
                .contains(&octopus_id)
        );
        assert!(
            !engine
                .legal_actions(&completed)
                .unwrap()
                .contains(&movement("d8", "d4"))
        );

        // An enemy can approach and leave during extra actions. Revelation must persist
        // after it leaves; the set must not merely suppress protection while adjacent.
        let mut extra_actions = completed.clone();
        extra_actions.actions_remaining = 3;
        let near = engine
            .apply(&extra_actions, movement("d8", "d5"))
            .unwrap()
            .state;
        assert_eq!(near.turn, Player::Black);
        assert_eq!(near.actions_remaining, 2);
        assert!(!near.card_runtime.submerged_piece_ids.contains(&octopus_id));
        let away = engine.apply(&near, movement("d5", "d8")).unwrap().state;
        assert_eq!(away.turn, Player::Black);
        assert_eq!(away.actions_remaining, 1);
        assert!(!away.card_runtime.submerged_piece_ids.contains(&octopus_id));
        assert!(
            engine
                .legal_actions(&away)
                .unwrap()
                .contains(&movement("d8", "d4"))
        );

        // Approach, not capture: the enemy rook stops on d5, adjacent to the Octopus.
        let approached = engine
            .apply(&completed, movement("d8", "d5"))
            .unwrap()
            .state;
        assert_eq!(approached.turn, Player::White);
        assert!(
            !approached
                .card_runtime
                .submerged_piece_ids
                .contains(&octopus_id)
        );
        let exposed = engine
            .apply(&approached, movement("a1", "b1"))
            .unwrap()
            .state;
        assert_eq!(exposed.turn, Player::Black);
        assert!(
            !exposed
                .card_runtime
                .submerged_piece_ids
                .contains(&octopus_id)
        );
        assert!(
            engine
                .legal_actions(&exposed)
                .unwrap()
                .contains(&movement("d5", "d4"))
        );
    }

    #[test]
    fn september_six_octopus_transformation_and_basic_behavior() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Rook),
                ("g7", Player::Black, PieceKind::Rook),
                ("b2", Player::White, PieceKind::Pawn),
            ],
            Player::White,
            939,
        );
        let granted = engine
            .grant_card(&state, 1, "octopus", Player::White)
            .unwrap()
            .state;
        assert_eq!(
            granted.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        let targets = engine.legal_actions(&granted).unwrap();
        assert!(targets.contains(&use_card(1, "d4")));
        for square in ["a1", "g7", "b2"] {
            assert!(!targets.contains(&use_card(1, square)));
        }
        let changed = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        let result = changed.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(result.kind, PieceKind::Octopus);
        assert_eq!(result.owner, Player::White);
        assert_eq!(changed.turn, Player::White);
        let moves = engine.legal_actions(&changed).unwrap();
        assert!(moves.contains(&movement("d4", "d5")));
        assert!(moves.contains(&movement("d4", "e5")));
        assert!(!moves.contains(&movement("d4", "d6")));
    }

    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L1058-1064 catalog own-queen; L86537-86538 transforms queen.
    // L86114 initializes bearRetaliationsRemaining; L1164-1165 gives two charges.
    // L82310-82311: king-step movement.
    #[test]
    fn september_six_hedgehog_transformation_and_basic_behavior() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Queen),
                ("g7", Player::Black, PieceKind::Queen),
                ("b2", Player::White, PieceKind::Pawn),
            ],
            Player::White,
            940,
        );
        let granted = engine
            .grant_card(&state, 1, "hedgehog", Player::White)
            .unwrap()
            .state;
        assert_eq!(
            granted.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Queen
        );
        let targets = engine.legal_actions(&granted).unwrap();
        assert!(targets.contains(&use_card(1, "d4")));
        for square in ["a1", "g7", "b2"] {
            assert!(!targets.contains(&use_card(1, square)));
        }
        let changed = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        let result = changed.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(result.kind, PieceKind::Hedgehog);
        assert_eq!(result.owner, Player::White);
        assert_eq!(changed.turn, Player::White);
        assert_eq!(
            changed.card_runtime.bear_retaliations.get(&result.id),
            Some(&2)
        );
        let moves = engine.legal_actions(&changed).unwrap();
        assert!(moves.contains(&movement("d4", "e5")));
        assert!(!moves.contains(&movement("d4", "d6")));
    }

    #[test]
    fn hedgehog_retaliates_twice_before_it_can_be_captured() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Queen),
                ("d8", Player::Black, PieceKind::Rook),
                ("h4", Player::Black, PieceKind::Rook),
                ("d1", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            950,
        );
        let granted = engine
            .grant_card(&state, 1, "hedgehog", Player::White)
            .unwrap()
            .state;
        let changed = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        let hedgehog_id = changed.board.get("d4".parse().unwrap()).unwrap().id;

        let black = engine.apply(&changed, movement("a1", "b1")).unwrap().state;
        let first = engine.apply(&black, movement("d8", "d4")).unwrap().state;
        assert!(first.board.get("d8".parse().unwrap()).is_none());
        assert_eq!(
            first.board.get("d4".parse().unwrap()).unwrap().id,
            hedgehog_id
        );
        assert_eq!(
            first.card_runtime.bear_retaliations.get(&hedgehog_id),
            Some(&1)
        );

        let black = engine.apply(&first, movement("b1", "a1")).unwrap().state;
        let second = engine.apply(&black, movement("h4", "d4")).unwrap().state;
        assert!(second.board.get("h4".parse().unwrap()).is_none());
        assert_eq!(
            second.board.get("d4".parse().unwrap()).unwrap().id,
            hedgehog_id
        );
        assert_eq!(
            second.card_runtime.bear_retaliations.get(&hedgehog_id),
            Some(&0)
        );

        let black = engine.apply(&second, movement("a1", "b1")).unwrap().state;
        let third = engine.apply(&black, movement("d1", "d4")).unwrap().state;
        let survivor = third.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(survivor.owner, Player::Black);
        assert_eq!(survivor.kind, PieceKind::Rook);
    }

    // Live source: artifacts/ruleset-0915-baseline/main-6qeAcxbX.js:
    // L1040-1046 catalog own-rook; L86531-86532 transforms rook.
    // L1168/L82316-82317: one orthogonal step, no captures.
    // L1257-1266/L95455: adjacent orthogonal friendly nonroyal gets protection.
    #[test]
    fn september_six_campfire_transformation_and_basic_behavior() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Rook),
                ("g7", Player::Black, PieceKind::Rook),
                ("b2", Player::White, PieceKind::Pawn),
                ("d5", Player::White, PieceKind::Pawn),
                ("d8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            941,
        );
        let granted = engine
            .grant_card(&state, 1, "campfire", Player::White)
            .unwrap()
            .state;
        assert_eq!(
            granted.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        let targets = engine.legal_actions(&granted).unwrap();
        assert!(targets.contains(&use_card(1, "d4")));
        for square in ["a1", "g7", "b2"] {
            assert!(!targets.contains(&use_card(1, square)));
        }
        let changed = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        let result = changed.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(result.kind, PieceKind::Campfire);
        assert_eq!(result.owner, Player::White);
        assert_eq!(changed.turn, Player::White);
        let before = engine.apply(&granted, movement("a1", "b1")).unwrap().state;
        assert_eq!(before.turn, Player::Black);
        assert!(
            engine
                .legal_actions(&before)
                .unwrap()
                .contains(&movement("d8", "d5"))
        );
        let moves = engine.legal_actions(&changed).unwrap();
        assert!(moves.contains(&movement("d4", "e4")));
        assert!(!moves.contains(&movement("d4", "e5")));
        let opponent = engine.apply(&changed, movement("a1", "b1")).unwrap().state;
        assert_eq!(opponent.turn, Player::Black);
        assert!(
            !engine
                .legal_actions(&opponent)
                .unwrap()
                .contains(&movement("d8", "d5"))
        );
    }

    #[test]
    fn clockwork_transforms_an_allied_minor_and_requires_a_friendly_neighbor_to_move() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Knight),
                ("h2", Player::White, PieceKind::Rook),
                ("f7", Player::Black, PieceKind::Bishop),
            ],
            Player::White,
            935,
        );
        let granted = engine
            .grant_card(&state, 1, "clockwork", Player::White)
            .unwrap()
            .state;
        let targets = engine.legal_actions(&granted).unwrap();
        assert!(targets.contains(&use_card(1, "d4")));
        for square in ["a1", "h2", "f7"] {
            assert!(!targets.contains(&use_card(1, square)));
        }
        let transformed = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        let clockwork = transformed.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(clockwork.kind, PieceKind::Clockwork);
        assert_eq!(clockwork.owner, Player::White);
        assert_eq!(transformed.turn, Player::White);
        assert!(!engine.legal_actions(&transformed).unwrap().iter().any(|action| {
            matches!(action, Action::Move(movement) if movement.from == "d4".parse::<Square>().unwrap())
        }));

        // Add a friendly neighbor on c4; the unobstructed d-file is a queen ray.
        let mut supported = transformed.clone();
        supported.board.set(
            "c4".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn)),
        );
        assert!(
            engine
                .legal_actions(&supported)
                .unwrap()
                .contains(&movement("d4", "d6"))
        );
    }

    #[test]
    fn long_en_passant_lets_a_distant_pawn_capture_the_just_doublestepped_pawn() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut board = crate::Board::empty();
        for (square, owner, kind) in [
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b5", Player::White, PieceKind::Pawn),
            ("e7", Player::Black, PieceKind::Pawn),
        ] {
            board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        // Black double-steps e7 -> e5; White then owns the en-passant right, so the
        // long-en-passant card must be armed for the CAPTURER (White), matching
        // state.longEnPassant[color] semantics in addInternalPawnMoves.
        let state = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::Black,
            934,
        );
        let stepped = engine.apply(&state, movement("e7", "e5")).unwrap().state;
        assert_eq!(stepped.turn, Player::White);
        let ep = stepped.en_passant.unwrap();
        assert_eq!(ep.target, "e6".parse().unwrap());
        // Without the card, the distant pawn on b5 has no en-passant reach at all.
        assert!(
            !engine
                .legal_actions(&stepped)
                .unwrap()
                .contains(&movement("b5", "e6"))
        );

        let granted = engine
            .grant_card(&stepped, 1, "long-en-passant", Player::White)
            .unwrap()
            .state;
        let actions = engine.legal_actions(&granted).unwrap();
        assert!(actions.contains(&movement("b5", "e6")));

        let transition = engine.apply(&granted, movement("b5", "e6")).unwrap();
        let captured = transition.state;
        assert_eq!(
            captured.board.get("e6".parse().unwrap()).unwrap().owner,
            Player::White
        );
        assert_eq!(
            captured.board.get("e6".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        assert!(captured.board.get("e5".parse().unwrap()).is_none());
        assert!(transition.events.iter().any(|event| matches!(
            event,
            Event::PieceCaptured { at, piece, .. }
                if at == &"e5".parse::<Square>().unwrap() && piece.kind == PieceKind::Pawn
        )));
    }

    #[test]
    fn mutation_lets_a_last_rank_pawn_promote_into_a_friendly_monster() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("a7", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            933,
        );
        let monster_promotion = |from: &str, to: &str| {
            Action::Move(MoveAction::new(
                from.parse().unwrap(),
                to.parse().unwrap(),
                Some(PieceKind::Monster),
            ))
        };
        // Without the card the pawn only exposes the standard four promotion kinds.
        let base_actions = engine.legal_actions(&state).unwrap();
        assert!(base_actions.contains(&Action::Move(MoveAction::new(
            "a7".parse().unwrap(),
            "a8".parse().unwrap(),
            Some(PieceKind::Queen)
        ))));
        assert!(!base_actions.contains(&monster_promotion("a7", "a8")));

        // Passive card: granting it is enough for the effect to arm.
        let granted = engine
            .grant_card(&state, 1, "mutation", Player::White)
            .unwrap()
            .state;
        let actions = engine.legal_actions(&granted).unwrap();
        assert!(actions.contains(&monster_promotion("a7", "a8")));
        assert!(actions.contains(&Action::Move(MoveAction::new(
            "a7".parse().unwrap(),
            "a8".parse().unwrap(),
            Some(PieceKind::Queen)
        ))));

        let promoted = engine
            .apply(&granted, monster_promotion("a7", "a8"))
            .unwrap()
            .state;
        let monster = promoted.board.get("a8".parse().unwrap()).unwrap();
        assert_eq!(monster.owner, Player::White);
        assert_eq!(monster.kind, PieceKind::Monster);
        assert_eq!(promoted.turn, Player::Black);
    }

    #[test]
    fn miracle_converts_every_bishop_capturable_enemy_and_sacrifices_all_friendly_bishops() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("b2", Player::White, PieceKind::Bishop),
                ("d4", Player::White, PieceKind::Bishop),
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("f6", Player::Black, PieceKind::Rook),
                ("d6", Player::Black, PieceKind::Knight),
                ("c5", Player::Black, PieceKind::Pawn),
                ("g5", Player::Black, PieceKind::Pawn),
            ],
            Player::White,
            932,
        );
        let granted = engine
            .grant_card(&state, 1, "miracle", Player::White)
            .unwrap()
            .state;
        let actions = engine.legal_actions(&granted).unwrap();
        assert!(actions.contains(&use_card_without_target(1)));
        let converted = engine
            .apply(&granted, use_card_without_target(1))
            .unwrap()
            .state;
        // Sacrificed bishops are gone from the board.
        assert!(converted.board.get("b2".parse().unwrap()).is_none());
        assert!(converted.board.get("d4".parse().unwrap()).is_none());
        // The rook on f6 lies on b2's diagonal and is captured, so it is converted.
        let converted_rook = converted.board.get("f6".parse().unwrap()).unwrap();
        assert_eq!(converted_rook.owner, Player::White);
        assert_eq!(converted_rook.kind, PieceKind::Rook);
        // d6 and g5 are unreachable for either bishop and stay with Black.
        assert_eq!(
            converted.board.get("d6".parse().unwrap()).unwrap().owner,
            Player::Black
        );
        assert_eq!(
            converted.board.get("g5".parse().unwrap()).unwrap().owner,
            Player::Black
        );
        // The pawn on c5 is capturable from b2 and converts.
        let converted_pawn = converted.board.get("c5".parse().unwrap()).unwrap();
        assert_eq!(converted_pawn.owner, Player::White);
        assert_eq!(converted_pawn.kind, PieceKind::Pawn);
        // The user's turn ends (END_TURN).
        assert_eq!(converted.turn, Player::Black);
    }

    #[test]
    fn extinction_targets_an_allied_minor_and_removes_its_type_from_both_sides() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Bishop),
                ("f4", Player::White, PieceKind::Bishop),
                ("c6", Player::Black, PieceKind::Bishop),
                ("f7", Player::Black, PieceKind::Bishop),
                ("b1", Player::White, PieceKind::Knight),
                ("b8", Player::Black, PieceKind::Knight),
                ("g2", Player::White, PieceKind::Camel),
                ("d1", Player::White, PieceKind::Rook),
                ("a2", Player::White, PieceKind::Pawn),
            ],
            Player::White,
            931,
        );
        let granted = engine
            .grant_card(&state, 1, "extinction", Player::White)
            .unwrap()
            .state;
        assert_eq!(granted.board.iter().count(), 11);
        let targets = engine.legal_actions(&granted).unwrap();
        for square in ["c3", "f4", "b1", "g2"] {
            assert!(
                targets.contains(&use_card(1, square)),
                "eligible minor at {square}"
            );
        }
        for square in ["a1", "d1", "a2", "c6", "b8"] {
            assert!(
                !targets.contains(&use_card(1, square)),
                "ineligible target at {square}"
            );
        }
        let removed = engine.apply(&granted, use_card(1, "c3")).unwrap().state;
        for square in ["c3", "f4", "c6", "f7"] {
            assert!(
                removed.board.get(square.parse().unwrap()).is_none(),
                "bishop at {square}"
            );
        }
        for square in ["a1", "h8", "b1", "b8", "g2", "d1", "a2"] {
            assert_eq!(
                removed.board.get(square.parse().unwrap()),
                granted.board.get(square.parse().unwrap())
            );
        }
        assert_eq!(removed.board.iter().count(), 7);
        assert_eq!(removed.status, GameStatus::Ongoing);
    }

    #[test]
    fn symmetry_adds_a_mirror_capture_for_one_turn_ignoring_intermediate_blockers() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("b3", Player::White, PieceKind::Knight),
                ("d3", Player::White, PieceKind::Pawn),
                ("b5", Player::White, PieceKind::Knight),
                ("g5", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
                ("g3", Player::Black, PieceKind::Pawn),
            ],
            Player::White,
            930,
        );
        let granted = engine
            .grant_card(&state, 1, "symmetry", Player::White)
            .unwrap()
            .state;
        assert!(
            !engine
                .legal_actions(&granted)
                .unwrap()
                .contains(&movement("b3", "g3"))
        );
        let activated = engine
            .apply(&granted, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(activated.turn, Player::White);
        let actions = engine.legal_actions(&activated).unwrap();
        assert!(actions.contains(&movement("b3", "g3")));
        assert!(actions.contains(&movement("b3", "c5")));
        assert!(!actions.contains(&movement("b5", "g5")));
        let moved = engine
            .apply(&activated, movement("b3", "g3"))
            .unwrap()
            .state;
        assert_eq!(moved.turn, Player::Black);
        assert!(moved.board.get("b3".parse().unwrap()).is_none());
        let mirrored = moved.board.get("g3".parse().unwrap()).unwrap();
        assert_eq!(mirrored.owner, Player::White);
        assert_eq!(mirrored.kind, PieceKind::Knight);
        assert_eq!(
            moved.board.get("d3".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        let next_own = engine.apply(&moved, movement("h8", "h7")).unwrap().state;
        assert_eq!(next_own.turn, Player::White);
        let restored = engine.legal_actions(&next_own).unwrap();
        assert!(!restored.contains(&movement("g3", "b3")));
        assert!(restored.contains(&movement("g3", "e4")));
    }

    #[test]
    fn outpost_protects_an_advanced_nonroyal_until_that_piece_moves() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a5", Player::White, PieceKind::King),
                ("d5", Player::White, PieceKind::Rook),
                ("b4", Player::White, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
                ("d8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            929,
        );
        let granted = engine
            .grant_card(&state, 1, "outpost", Player::White)
            .unwrap()
            .state;
        let targets = engine.legal_actions(&granted).unwrap();
        assert!(targets.contains(&use_card(1, "d5")));
        assert!(!targets.contains(&use_card(1, "a5")));
        assert!(!targets.contains(&use_card(1, "b4")));
        assert!(!targets.contains(&use_card(1, "d8")));
        // Merely owning the active card must not protect the rook.
        let unused = engine.apply(&granted, movement("a5", "a4")).unwrap().state;
        assert!(
            engine
                .legal_actions(&unused)
                .unwrap()
                .contains(&movement("d8", "d5"))
        );

        let protected = engine.apply(&granted, use_card(1, "d5")).unwrap().state;
        assert_eq!(protected.turn, Player::White);
        let opponent = engine
            .apply(&protected, movement("a5", "a4"))
            .unwrap()
            .state;
        assert_eq!(opponent.turn, Player::Black);
        assert!(
            !engine
                .legal_actions(&opponent)
                .unwrap()
                .contains(&movement("d8", "d5"))
        );
        let next_own = engine.apply(&opponent, movement("h8", "h7")).unwrap().state;
        assert_eq!(next_own.turn, Player::White);
        let moved = engine.apply(&next_own, movement("d5", "d6")).unwrap().state;
        assert_eq!(moved.turn, Player::Black);
        assert!(
            engine
                .legal_actions(&moved)
                .unwrap()
                .contains(&movement("d8", "d6"))
        );
    }

    #[test]
    fn reversal_swaps_rook_and_bishop_moves_only_after_use_and_for_one_turn() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Rook),
                ("b3", Player::White, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            928,
        );
        let granted = engine
            .grant_card(&state, 1, "reversal", Player::White)
            .unwrap()
            .state;
        let before_use = engine.legal_actions(&granted).unwrap();
        assert!(before_use.contains(&movement("d4", "d5")));
        assert!(!before_use.contains(&movement("d4", "e5")));
        assert!(before_use.contains(&movement("b3", "c4")));
        assert!(!before_use.contains(&movement("b3", "b4")));

        let activated = engine
            .apply(&granted, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(activated.turn, Player::White);
        let reversed = engine.legal_actions(&activated).unwrap();
        assert!(reversed.contains(&movement("d4", "e5")));
        assert!(!reversed.contains(&movement("d4", "d5")));
        assert!(reversed.contains(&movement("b3", "b4")));
        assert!(!reversed.contains(&movement("b3", "c4")));

        let moved = engine
            .apply(&activated, movement("d4", "e5"))
            .unwrap()
            .state;
        assert_eq!(
            moved.board.get("e5".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        assert_eq!(moved.turn, Player::Black);
        let next_own = engine.apply(&moved, movement("h8", "h7")).unwrap().state;
        assert_eq!(next_own.turn, Player::White);
        let restored = engine.legal_actions(&next_own).unwrap();
        assert!(restored.contains(&movement("e5", "e6")));
        assert!(!restored.contains(&movement("e5", "f6")));
        assert!(restored.contains(&movement("b3", "c4")));
        assert!(!restored.contains(&movement("b3", "b4")));
    }

    #[test]
    fn disassembly_splits_the_moved_queen_into_bishop_and_rook() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut board = crate::Board::empty();
        for (square, owner, kind) in [
            ("a1", Player::White, PieceKind::King),
            ("d4", Player::White, PieceKind::Queen),
            ("h8", Player::Black, PieceKind::King),
        ] {
            board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let without = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::White,
            926,
        );
        let moved = engine.apply(&without, movement("d4", "d5")).unwrap().state;
        assert_eq!(
            moved.board.get("d5".parse().unwrap()).unwrap().kind,
            PieceKind::Queen
        );
        let with = engine
            .grant_card(&without, 1, "disassembly", Player::White)
            .unwrap()
            .state;
        // Active card: the effect arms only when the card is used.
        let moved_without_use = engine.apply(&with, movement("d4", "d5")).unwrap().state;
        assert_eq!(
            moved_without_use
                .board
                .get("d5".parse().unwrap())
                .unwrap()
                .kind,
            PieceKind::Queen
        );
        let activated = engine
            .apply(&with, use_card_without_target(1))
            .unwrap()
            .state;
        let split = engine
            .apply(&activated, movement("d4", "d5"))
            .unwrap()
            .state;
        let bishop = split.board.get("d5".parse().unwrap()).unwrap();
        assert_eq!(bishop.kind, PieceKind::Bishop);
        assert_eq!(bishop.owner, Player::White);
        let rook = split.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(rook.kind, PieceKind::Rook);
        assert_eq!(rook.owner, Player::White);
    }

    // Live source: addInternalPawnMoves (main-6qeAcxbX.js ~L11828896) — proficiency lets a pawn
    // on the rank before promotion capture one step forward: state.proficiency?.[color] &&
    // row === (white ? 1 : boardRowCount()-2) && target && canCaptureTarget(...).
    #[test]
    fn proficiency_lets_the_pawn_capture_forward_from_the_seventh_rank() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut board = crate::Board::empty();
        for (square, owner, kind) in [
            ("a1", Player::White, PieceKind::King),
            ("e7", Player::White, PieceKind::Pawn),
            ("e8", Player::Black, PieceKind::Rook),
            ("h8", Player::Black, PieceKind::King),
        ] {
            board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let without = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::White,
            925,
        );
        let actions = engine.legal_actions(&without).unwrap();
        assert!(!actions.contains(&movement("e7", "e8")));
        let with = engine
            .grant_card(&without, 1, "proficiency", Player::White)
            .unwrap()
            .state;
        let actions = engine.legal_actions(&with).unwrap();
        // Reaching the last rank promotes, so the forward capture is a promotion capture.
        let promotion_capture = Action::Move(MoveAction::new(
            "e7".parse().unwrap(),
            "e8".parse().unwrap(),
            Some(PieceKind::Queen),
        ));
        assert!(actions.contains(&promotion_capture));
        let captured = engine.apply(&with, promotion_capture).unwrap().state;
        assert!(captured.board.get("e8".parse().unwrap()).unwrap().kind == PieceKind::Queen);
        assert_eq!(
            captured.board.get("e8".parse().unwrap()).unwrap().owner,
            Player::White
        );
    }

    #[test]
    fn killer_king_lets_the_king_capture_like_a_rook() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut board = crate::Board::empty();
        for (square, owner, kind) in [
            ("a1", Player::White, PieceKind::King),
            ("e1", Player::White, PieceKind::King),
            ("h1", Player::Black, PieceKind::King),
        ] {
            board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let without = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::White,
            924,
        );
        let actions = engine.legal_actions(&without).unwrap();
        assert!(!actions.contains(&movement("e1", "h1")));

        let with = engine
            .grant_card(&without, 1, "killer-king", Player::White)
            .unwrap()
            .state;
        let actions = engine.legal_actions(&with).unwrap();
        assert!(actions.contains(&movement("e1", "h1")));
        assert!(!actions.contains(&movement("e1", "e4")));
        let victory = engine.apply(&with, movement("e1", "h1")).unwrap().state;
        assert!(matches!(
            victory.status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));
    }

    #[test]
    fn pawn_steps_onto_or_through_a_concealed_enemy_square() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("e2", Player::White, PieceKind::Pawn),
                ("e3", Player::Black, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            923,
        );
        let hidden_pawn = state.board.get("e3".parse().unwrap()).unwrap();
        state.card_runtime.stealth_piece_ids.insert(hidden_pawn.id);
        let actions = engine.legal_actions(&state).unwrap();
        assert!(actions.contains(&movement("e2", "e3")));
        assert!(actions.contains(&movement("e2", "e4")));
        let stepped = engine.apply(&state, movement("e2", "e3")).unwrap().state;
        let occupant = stepped.board.get("e3".parse().unwrap()).unwrap();
        assert_eq!(occupant.owner, Player::White);
        assert!(stepped.board.find_by_id(hidden_pawn.id).is_none());
    }

    #[test]
    fn stealth_concealment_survives_own_turn_end_and_clears_after_opponents_turn() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("e4", Player::Black, PieceKind::Rook),
                ("e5", Player::White, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            921,
        );
        grant_test_card(&mut state, crate::STEALTH, 1);
        let hidden = engine.apply(&state, use_card(1, "e5")).unwrap().state;
        assert!(
            hidden
                .card_runtime
                .stealth_piece_ids
                .contains(&hidden.board.get("e5".parse().unwrap()).unwrap().id)
        );
        // The owner's own turn end keeps the concealment.
        let own = engine.apply(&hidden, movement("a1", "a2")).unwrap().state;
        assert!(
            own.card_runtime
                .stealth_piece_ids
                .contains(&own.board.get("e5".parse().unwrap()).unwrap().id)
        );
        // The opponent's completed turn removes it: hidden only for that opponent turn.
        let after_enemy = engine.apply(&own, movement("h8", "h7")).unwrap().state;
        assert!(after_enemy.card_runtime.stealth_piece_ids.is_empty());
    }

    #[test]
    fn stealth_bishop_is_absent_from_enemy_move_generation_and_installations() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("e4", Player::Black, PieceKind::Rook),
                ("e5", Player::White, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::Black,
            920,
        );
        let bishop = state.board.get("e5".parse().unwrap()).unwrap();
        state.card_runtime.stealth_piece_ids.insert(bishop.id);
        let enemy_actions = engine.legal_actions(&state).unwrap();
        // The hidden bishop must not block the ray, but landing on it is legal.
        assert!(enemy_actions.contains(&movement("e4", "e5")));
        assert!(engine.can_capture(
            &state,
            state.board.get("e4".parse().unwrap()).unwrap(),
            bishop
        ));
        assert!(enemy_actions.iter().any(|action| matches!(action,
            Action::Move(m) if m.to == "e6".parse().unwrap())));
        assert!(enemy_actions.iter().any(|action| matches!(action,
            Action::Move(m) if m.to == "e8".parse().unwrap())));
        grant_test_card(&mut state, crate::LOBSTER, 1);
        let reserved = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Square("e5".parse().unwrap()),
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            reserved
                .card_runtime
                .pending_lobsters
                .iter()
                .find(|p| p.square == "e5".parse().unwrap())
                .map(|p| p.owner),
            Some(Player::Black)
        );
        let landed = engine.apply(&state, movement("e4", "e5")).unwrap().state;
        assert!(landed.board.find_by_id(bishop.id).is_none());
        let rook = landed.board.get("e5".parse().unwrap()).unwrap();
        assert_eq!(rook.owner, Player::Black);
        assert_eq!(rook.kind, PieceKind::Rook);
    }

    #[test]
    fn portal_gun_installs_chosen_pair_at_next_own_turn_start() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            919,
        );
        grant_test_card(&mut state, crate::PORTAL_GUN, 1);
        let squares = ["c3".parse().unwrap(), "f6".parse().unwrap()];
        let action = Action::UseCard {
            instance_id: 1,
            target: ActionTarget::Squares(squares.to_vec()),
        };
        let pending = engine.apply(&state, action).unwrap().state;
        assert!(pending.card_runtime.portals.is_none());
        assert_eq!(pending.card_runtime.pending_portals.len(), 1);
        let mut changed = pending.clone();
        changed.card_runtime.pending_portals[0].due_turn += 1;
        assert_ne!(changed.fingerprint(), pending.fingerprint());
        assert_ne!(changed.position_key(), pending.position_key());
        let once = engine.apply(&pending, movement("a1", "a2")).unwrap().state;
        assert!(once.card_runtime.portals.is_none());
        let twice = engine.apply(&once, movement("h8", "h7")).unwrap().state;
        assert_eq!(twice.card_runtime.portals, Some(squares));
        assert!(twice.card_runtime.pending_portals.is_empty());
        let rollback = engine.opening_move_rollback_state(&twice, &once);
        assert_eq!(
            rollback.card_runtime.pending_portals,
            once.card_runtime.pending_portals
        );
        let mut due = once.clone();
        due.turn = Player::White;
        due.actions_remaining = 2;
        let mut after = engine.apply(&due, movement("a2", "a3")).unwrap().state;
        engine.resolve_pending_portals_for_turn(&mut after, Player::White, &mut Vec::new());
        // Runtime delta restoration must cover both pending and installed portal state.
        let replay = after.card_runtime.move_replays[Player::White.index()]
            .as_mut()
            .unwrap();
        replay.runtime_after.pending_portals.clear();
        replay.runtime_after.portals = Some(squares);
        grant_test_card(&mut after, crate::REPLAY, 2);
        let restored = engine
            .apply(
                &after,
                Action::UseCard {
                    instance_id: 2,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            restored.card_runtime.pending_portals,
            due.card_runtime.pending_portals
        );
        assert_eq!(restored.card_runtime.portals, due.card_runtime.portals);
    }

    #[test]
    fn lobster_pending_is_fingerprinted_restored_and_canceled_when_blocked() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            918,
        );
        state
            .card_runtime
            .pending_lobsters
            .push(crate::PendingLobster {
                owner: Player::White,
                square: "d4".parse().unwrap(),
                due_ply: 1,
            });
        let mut changed = state.clone();
        changed.card_runtime.pending_lobsters[0].due_ply += 1;
        assert_ne!(changed.fingerprint(), state.fingerprint());
        assert_ne!(changed.position_key(), state.position_key());
        state.actions_remaining = 2;
        let mut moved = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert!(moved.card_runtime.pending_lobsters.is_empty());
        assert_eq!(
            engine
                .opening_move_rollback_state(&moved, &state)
                .card_runtime
                .pending_lobsters,
            state.card_runtime.pending_lobsters
        );
        moved.turn = Player::White;
        grant_test_card(&mut moved, crate::REPLAY, 1);
        let restored = engine
            .apply(
                &moved,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            restored.card_runtime.pending_lobsters,
            state.card_runtime.pending_lobsters
        );
        assert_eq!(restored.board, state.board);
        state.card_runtime.barricades.insert("d4".parse().unwrap());
        let blocked = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert!(blocked.board.get("d4".parse().unwrap()).is_none());
        assert!(blocked.card_runtime.pending_lobsters.is_empty());
    }

    #[test]
    fn lobster_spawns_after_two_counted_moves_not_at_card_use() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            917,
        );
        grant_test_card(&mut state, crate::LOBSTER, 1);
        let pending = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Square("d4".parse().unwrap()),
                },
            )
            .unwrap()
            .state;
        assert!(pending.board.get("d4".parse().unwrap()).is_none());
        let once = engine.apply(&pending, movement("a1", "a2")).unwrap().state;
        assert!(once.board.get("d4".parse().unwrap()).is_none());
        let twice = engine.apply(&once, movement("h8", "h7")).unwrap().state;
        let lobster = twice.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(lobster.kind, PieceKind::Lobster);
        assert_eq!(lobster.owner, Player::White);
        assert!(
            twice
                .card_runtime
                .fresh_no_capture_until
                .contains_key(&lobster.id)
        );
    }

    #[test]
    fn queens_gambit_protection_is_removed_by_non_promotion_card_transforms() {
        let engine = Engine::bundled_20260818().unwrap();
        for (card, kind) in [
            (crate::ASSASSIN, PieceKind::Knight),
            (crate::PEGASUS, PieceKind::Knight),
            (crate::REAPER, PieceKind::Queen),
            (crate::SLIME, PieceKind::Rook),
            (crate::NECROMANCY, PieceKind::Pawn),
        ] {
            let mut state = minimal_state(
                &[
                    ("a1", Player::White, PieceKind::King),
                    ("d4", Player::White, kind),
                    ("h8", Player::Black, PieceKind::King),
                ],
                Player::White,
                916,
            );
            let piece = state.board.get("d4".parse().unwrap()).unwrap();
            state.card_runtime.protected_piece_ids.insert(piece.id);
            state.card_runtime.queens_gambit_protection.insert(piece.id);
            grant_test_card(&mut state, card, 1);
            let result = engine.apply(&state, use_card(1, "d4")).unwrap();
            assert!(
                !result
                    .state
                    .card_runtime
                    .queens_gambit_protection
                    .contains(&piece.id),
                "{card}"
            );
            assert!(
                !result
                    .state
                    .card_runtime
                    .protected_piece_ids
                    .contains(&piece.id),
                "{card}"
            );
        }
    }

    #[test]
    fn zugzwang_ends_users_turn_and_forces_opponents_royal_move() {
        let engine = Engine::bundled_20260818().unwrap();
        assert!(crate::implemented_card(crate::ZUGZWANG).unwrap().ends_turn);
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d6", Player::Black, PieceKind::Rook),
            ],
            Player::White,
            915,
        );
        grant_test_card(&mut state, crate::ZUGZWANG, 1);
        let result = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .unwrap();
        assert_eq!(result.state.turn, Player::Black);
        assert_eq!(result.state.turns_taken, [1, 0]);
        assert!(result.state.card_runtime.zugzwang[Player::Black.index()]);
        let actions = engine.legal_actions(&result.state).unwrap();
        assert!(actions.iter().any(|a| matches!(a, Action::Move(_))));
        assert!(actions.iter().all(|a| match a {
            Action::Move(m) => m.from == "h8".parse().unwrap(),
            _ => true,
        }));
        let moved = engine.apply(&result.state, movement("h8", "h7")).unwrap();
        assert!(!moved.state.card_runtime.zugzwang[Player::Black.index()]);
        assert_eq!(moved.state.turn, Player::White);
    }

    #[test]
    fn macho_chess_king_wins_only_at_the_opposite_edge() {
        let engine = Engine::bundled_20260818().unwrap();
        for (owner, from, to, enemy) in [
            (Player::White, "d7", "d8", "h1"),
            (Player::Black, "d2", "d1", "h8"),
        ] {
            let mut state = minimal_state(
                &[
                    (from, owner, PieceKind::King),
                    (enemy, owner.opponent(), PieceKind::King),
                ],
                owner,
                913,
            );
            state.card_runtime.macho_chess = true;
            let result = engine.apply(&state, movement(from, to)).unwrap();
            assert_eq!(
                result.state.status,
                GameStatus::Won {
                    winner: owner,
                    reason: WinReason::CardEffect(crate::MACHO_CHESS.to_owned()),
                }
            );
            assert!(
                result
                    .events
                    .iter()
                    .any(|event| matches!(event, Event::GameEnded { .. }))
            );
        }
        let mut state = minimal_state(
            &[
                ("d2", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            914,
        );
        state.card_runtime.macho_chess = true;
        assert_eq!(
            engine
                .apply(&state, movement("d2", "d1"))
                .unwrap()
                .state
                .status,
            GameStatus::Ongoing
        );
    }

    #[test]
    fn missionary_creates_two_missionaries_from_one_bishop() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            912,
        );
        grant_test_card(&mut state, MISSIONARY, 1);
        let action = use_card(1, "d4");
        assert!(engine.legal_actions(&state).unwrap().contains(&action));
        let first = engine.apply(&state, action.clone()).unwrap();
        let second = engine.apply(&state, action).unwrap();
        assert_eq!(first, second);
        let missionaries = first
            .state
            .board
            .iter()
            .filter(|(_, piece)| {
                piece.owner == Player::White && piece.kind == PieceKind::Missionary
            })
            .collect::<Vec<_>>();
        assert_eq!(missionaries.len(), 2);
        assert_ne!(missionaries[0].1.id, missionaries[1].1.id);
        assert!(
            missionaries
                .iter()
                .any(|(square, _)| *square == "d4".parse().unwrap())
        );
        assert!(missionaries.iter().all(|(_, piece)| {
            first
                .state
                .card_runtime
                .fresh_no_capture_until
                .contains_key(&piece.id)
        }));
    }

    #[test]
    fn slime_cannot_teleport_via_loyalist_or_twin_bonds() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Slime),
                ("f4", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            911,
        );
        let slime = state.board.get("d4".parse().unwrap()).unwrap();
        let rook = state.board.get("f4".parse().unwrap()).unwrap();
        state.card_runtime.loyalist_piece_ids.insert(slime.id);
        assert!(!has_move(&engine, &state, "d4", "a2"));
        assert!(has_move(&engine, &state, "d4", "d7"));
        state.card_runtime.twin_bonds.push(crate::TwinBond {
            first_id: slime.id,
            second_id: rook.id,
        });
        let moved_slime = engine.apply(&state, movement("d4", "d7")).unwrap().state;
        assert_eq!(
            moved_slime.board.get("d7".parse().unwrap()).unwrap().id,
            slime.id
        );
        assert_eq!(
            moved_slime.board.get("f4".parse().unwrap()).unwrap().id,
            rook.id
        );
        let moved_rook = engine.apply(&state, movement("f4", "f5")).unwrap().state;
        assert_eq!(
            moved_rook.board.get("d4".parse().unwrap()).unwrap().id,
            slime.id
        );
        assert_eq!(
            moved_rook.board.get("f5".parse().unwrap()).unwrap().id,
            rook.id
        );
    }

    #[test]
    fn slime_does_not_gain_card_supplied_movement() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Slime),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            910,
        );
        let slime = state.board.get("d4".parse().unwrap()).unwrap();
        for provider in ["training", "promotion", "imperial", "socialism"] {
            let mut changed = state.clone();
            match provider {
                "training" => {
                    changed
                        .card_runtime
                        .basic_training_piece_ids
                        .insert(slime.id);
                }
                "promotion" => {
                    changed
                        .card_runtime
                        .promotion_rush_piece_ids
                        .insert(slime.id);
                }
                "imperial" => {
                    changed
                        .card_runtime
                        .imperial_moves
                        .insert(slime.id, [PieceKind::Queen].into_iter().collect());
                }
                "socialism" => {
                    changed.card_runtime.socialism[Player::White.index()] = 1;
                }
                _ => unreachable!(),
            }
            assert!(has_move(&engine, &changed, "d4", "d7"), "{provider}");
            assert!(!has_move(&engine, &changed, "d4", "d5"), "{provider}");
        }
    }

    #[test]
    fn slime_cannot_relay_as_source_or_target() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Slime),
                ("d6", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            909,
        );
        state.card_runtime.relay[Player::White.index()] = true;
        assert!(has_move(&engine, &state, "d4", "d7"));
        for (from, to) in [("d4", "d6"), ("d6", "d4")] {
            assert!(!has_move(&engine, &state, from, to));
            assert!(matches!(
                engine.apply(&state, movement(from, to)),
                Err(EngineError::IllegalAction(_))
            ));
        }
    }

    #[test]
    fn slime_retains_normal_leaps_but_cannot_be_dragon_swapped() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Slime),
                ("b3", Player::White, PieceKind::Dragon),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            908,
        );
        assert!(has_move(&engine, &state, "d4", "d7"));
        assert!(!has_move(&engine, &state, "b3", "d4"));
        assert!(matches!(
            engine.apply(&state, movement("b3", "d4")),
            Err(EngineError::IllegalAction(_))
        ));
    }

    #[test]
    fn horse_riding_deletes_own_knights_with_events_but_without_capture_reactions() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Knight),
                ("e4", Player::White, PieceKind::Knight),
                ("c4", Player::White, PieceKind::Wizard),
                ("d5", Player::Black, PieceKind::Reaper),
                ("g7", Player::Black, PieceKind::Knight),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            907,
        );
        let knights: Vec<_> = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == Player::White && piece.kind == PieceKind::Knight)
            .collect();
        let wizard = state.board.get("c4".parse().unwrap()).unwrap();
        let reaper = state.board.get("d5".parse().unwrap()).unwrap();
        state.card_runtime.wizard_mana.insert(wizard.id, 2);
        state.card_runtime.reaper_souls.insert(reaper.id, 3);
        for (_, piece) in &knights {
            state.card_runtime.shielded_piece_ids.insert(piece.id);
        }
        let transition = engine
            .grant_card(&state, 911, HORSE_RIDING, Player::White)
            .unwrap();
        let after = &transition.state;
        assert!(after.card_runtime.horse_riding[Player::White.index()]);
        assert!(after.board.contains(Player::Black, PieceKind::Knight));
        assert!(after.board.contains(Player::White, PieceKind::King));
        for (square, piece) in knights {
            assert!(after.board.get(square).is_none());
            assert!(!after.card_runtime.shielded_piece_ids.contains(&piece.id));
            assert_eq!(
                transition
                    .events
                    .iter()
                    .filter(|event| matches!(event,
                        Event::PieceRemoved { actor: Player::White, piece: removed, at, cause }
                            if removed.id == piece.id && *at == square && cause == HORSE_RIDING
                    ))
                    .count(),
                1
            );
        }
        assert!(
            !transition
                .events
                .iter()
                .any(|event| matches!(event, Event::PieceCaptured { .. }))
        );
        assert_eq!(after.card_runtime.magic_girl_surge, [false, false]);
        assert_eq!(after.card_runtime.wizard_mana[&wizard.id], 2);
        assert_eq!(after.card_runtime.reaper_souls[&reaper.id], 3);
        assert!(after.card_runtime.undead_resurrections.is_empty());
        assert_eq!(after.status, GameStatus::Ongoing);
    }

    #[test]
    fn black_magic_legal_action_requires_an_adjacent_open_square() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut blocked = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Pawn),
                ("b1", Player::White, PieceKind::Pawn),
                ("b2", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            905,
        );
        grant_test_card(&mut blocked, crate::BLACK_MAGIC, 1);
        let actions = engine.legal_actions(&blocked).unwrap();
        assert!(
            !actions
                .iter()
                .any(|action| matches!(action, Action::UseCard { instance_id: 1, .. }))
        );

        let mut open = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("c3", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            906,
        );
        grant_test_card(&mut open, crate::BLACK_MAGIC, 1);
        let actions = engine.legal_actions(&open).unwrap();
        assert!(actions.iter().any(|action| matches!(
            action,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None
            }
        )));
        let state = engine
            .apply(
                &open,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            state.board.get("b1".parse().unwrap()).unwrap().kind,
            PieceKind::Monster
        );
        assert_eq!(
            state.board.get("b1".parse().unwrap()).unwrap().owner,
            Player::White
        );
    }

    #[test]
    fn black_magic_summons_one_adjacent_allied_monster_0915() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = standard_play_state(4);
        state.board.set("d2".parse().unwrap(), None);
        let rook = state.board.get("a1".parse().unwrap()).unwrap();
        state.card_runtime.large_pieces.insert(
            rook.id,
            LargePieceStatus {
                anchor: "a1".parse().unwrap(),
                hp: 2,
                max_hp: 2,
            },
        );
        grant_test_card(&mut state, crate::BLACK_MAGIC, 1);
        let mut state = engine
            .apply(&state, use_card_without_target(1))
            .unwrap()
            .state;
        assert_eq!(state.status, GameStatus::Ongoing);
        // 0915: no sacrifice, no dark wizard; the king survives untouched and the
        // large rook runtime is preserved.
        assert!(state.board.contains(Player::White, PieceKind::King));
        assert!(!state.board.contains(Player::White, PieceKind::DarkWizard));
        assert!(state.card_runtime.large_pieces.contains_key(&rook.id));
        // Exactly one allied monster spawns on an open square adjacent to the
        // royal. From the standard opening the first candidate is d2 beside e1.
        let monsters: Vec<_> = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == Player::White && piece.kind == PieceKind::Monster)
            .collect();
        assert_eq!(monsters.len(), 1);
        assert_eq!(monsters[0].0.to_string(), "d2");
        // Monster cannot be targeted by friendly buff cards like Parry
        grant_test_card(&mut state, crate::PARRY, 2);
        let parry_actions = engine.legal_actions(&state).unwrap();
        for action in parry_actions {
            if let Action::UseCard {
                instance_id: 2,
                target: ActionTarget::PieceAt(sq),
            } = action
            {
                assert_ne!(state.board.get(sq).unwrap().kind, PieceKind::Monster);
            }
        }
    }

    #[test]
    fn quantum_mechanics_supports_shadow_moves_and_collapses_on_attack() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("d4", Player::Black, PieceKind::Pawn),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        let rook = state.board.get("a1".parse().unwrap()).unwrap();
        state
            .card_runtime
            .quantum_positions
            .insert(rook.id, "d1".parse().unwrap());
        // Rook can move from real square a1 -> a4, AND from shadow square d1 -> d3
        assert!(has_move(&engine, &state, "a1", "a4"));
        assert!(has_move(&engine, &state, "d1", "d3"));

        // Moving from shadow collapses the piece to d3
        let moved = engine.apply(&state, movement("d1", "d3")).unwrap().state;
        assert!(moved.board.get("d3".parse().unwrap()).is_some());
        assert!(moved.board.get("a1".parse().unwrap()).is_none());
        assert!(!moved.card_runtime.quantum_positions.contains_key(&rook.id));
    }

    #[test]
    fn premove_supports_multi_step_queue() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Pawn),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        grant_test_card(&mut state, crate::PREMOVE, 1);
        let state = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Squares(vec![
                        "a2".parse().unwrap(),
                        "a3".parse().unwrap(),
                    ]),
                },
            )
            .unwrap()
            .state;
        assert_eq!(state.card_runtime.pending_premoves.len(), 1);
        assert_eq!(state.card_runtime.pending_premoves[0].moves.len(), 1);
    }

    #[test]
    fn part_four_end_cards_execute_and_modify_state() {
        let engine = Engine::bundled_20260807().unwrap();

        // Baby Bear
        let mut bb_state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("d1", Player::White, PieceKind::Queen),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        grant_test_card(&mut bb_state, crate::BABY_BEAR, 1);
        let bb_state = engine
            .apply(
                &bb_state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::PieceAt("d1".parse().unwrap()),
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            bb_state.board.get("d1".parse().unwrap()).unwrap().kind,
            PieceKind::BabyBear
        );

        // Bribe
        let mut bribe_state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("b1", Player::White, PieceKind::Knight),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        grant_test_card(&mut bribe_state, crate::BRIBE, 2);
        let bribe_state = engine
            .apply(
                &bribe_state,
                Action::UseCard {
                    instance_id: 2,
                    target: ActionTarget::PieceAt("b1".parse().unwrap()),
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            bribe_state.board.get("b1".parse().unwrap()).unwrap().kind,
            PieceKind::Amazon
        );

        // Racing King
        let mut rk_state = minimal_state(
            &[
                ("e7", Player::White, PieceKind::King),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        rk_state.card_runtime.racing_king[Player::White.index()] = true;
        let rk_next = engine.apply(&rk_state, movement("e7", "d8")).unwrap().state;
        assert!(matches!(
            rk_next.status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));

        // Lobster spawn
        let mut lob_state = standard_play_state(4);
        grant_test_card(&mut lob_state, crate::LOBSTER, 3);
        let lob_state = engine
            .apply(
                &lob_state,
                Action::UseCard {
                    instance_id: 3,
                    target: ActionTarget::Square("d4".parse().unwrap()),
                },
            )
            .unwrap()
            .state;
        assert!(lob_state.board.get("d4".parse().unwrap()).is_none());
        let lob_state = engine
            .apply(&lob_state, movement("e2", "e4"))
            .unwrap()
            .state;
        assert!(lob_state.board.get("d4".parse().unwrap()).is_none());
        let lob_state = engine
            .apply(&lob_state, movement("e7", "e5"))
            .unwrap()
            .state;
        assert_eq!(
            lob_state.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Lobster
        );

        // Conscription
        let mut consc_state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        grant_test_card(&mut consc_state, crate::CONSCRIPTION, 4);
        let consc_state = engine
            .apply(
                &consc_state,
                Action::UseCard {
                    instance_id: 4,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        assert!(consc_state.board.get("c2".parse().unwrap()).is_some());
        assert!(consc_state.board.get("d2".parse().unwrap()).is_some());
    }

    #[test]
    fn part_five_and_six_rule_and_meta_cards_execute() {
        let engine = Engine::bundled_20260807().unwrap();

        // Black hole rule
        let bh_state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("d1", Player::White, PieceKind::Rook),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        let bh_state = engine
            .apply_rule(&bh_state, crate::BLACK_HOLE)
            .unwrap()
            .state;
        assert!(
            bh_state
                .card_runtime
                .black_holes
                .contains(&"d4".parse().unwrap())
        );
        let moved = engine.apply(&bh_state, movement("d1", "d4")).unwrap().state;
        assert!(moved.board.get("d4".parse().unwrap()).is_none());

        // Transcendence rule
        let tr_state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("e4", Player::White, PieceKind::Pawn),
                ("d5", Player::Black, PieceKind::Pawn),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        let tr_state = engine
            .apply_rule(&tr_state, crate::TRANSCENDENCE)
            .unwrap()
            .state;
        let tr_next = engine.apply(&tr_state, movement("e4", "d5")).unwrap().state;
        assert_eq!(
            tr_next.board.get("d5".parse().unwrap()).unwrap().kind,
            PieceKind::Knight
        );

        // Insight meta card
        let mut ins_state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            4,
        );
        let k_id = ins_state.board.get("e1".parse().unwrap()).unwrap().id;
        let opp_k_id = ins_state.board.get("e8".parse().unwrap()).unwrap().id;
        ins_state.card_runtime.disarmed.insert(k_id, 2);
        ins_state.card_runtime.shielded_piece_ids.insert(opp_k_id);
        grant_test_card(&mut ins_state, crate::INSIGHT, 1);
        let ins_next = engine
            .apply(
                &ins_state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        assert!(!ins_next.card_runtime.disarmed.contains_key(&k_id));
        assert!(!ins_next.card_runtime.shielded_piece_ids.contains(&opp_k_id));
    }

    #[test]
    fn field_promotion_prompts_after_the_second_capture() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Pawn),
                ("e5", Player::Black, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            200,
        );
        let mut pawn = state.board.get("d4".parse().unwrap()).unwrap();
        pawn.capture_count = 1;
        state.board.set("d4".parse().unwrap(), Some(pawn));
        state.card_runtime.field_promotion[Player::White.index()] = true;

        state = engine.apply(&state, movement("d4", "e5")).unwrap().state;
        let GamePhase::AwaitingChoice(choice) = &state.phase else {
            panic!("field promotion must expose a choice");
        };
        let choice_id = choice.id;
        assert_eq!(choice.options, ["queen", "rook", "bishop", "knight"]);
        state = engine
            .apply(
                &state,
                Action::ResolveChoice {
                    choice_id,
                    option_index: 0,
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            state.board.get("e5".parse().unwrap()).unwrap().kind,
            PieceKind::Queen
        );
        assert!(
            state
                .card_runtime
                .field_promoted_piece_ids
                .contains(&pawn.id)
        );
    }

    #[test]
    fn cleanup_and_hypocrisy_accept_exact_dynamic_square_sets() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut cleanup = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Pawn),
                ("b2", Player::White, PieceKind::Rook),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            201,
        );
        grant_test_card(&mut cleanup, CLEANUP, 1);
        cleanup = engine
            .apply(
                &cleanup,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Squares(vec![
                        "a2".parse().unwrap(),
                        "b2".parse().unwrap(),
                    ]),
                },
            )
            .unwrap()
            .state;
        assert!(cleanup.board.get("a2".parse().unwrap()).is_none());
        assert!(cleanup.board.get("b2".parse().unwrap()).is_none());

        let mut hypocrisy = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            202,
        );
        grant_test_card(&mut hypocrisy, HYPOCRISY, 2);
        let squares = ["a2", "b2", "c2", "d2"]
            .map(|name| name.parse().unwrap())
            .to_vec();
        hypocrisy = engine
            .apply(
                &hypocrisy,
                Action::UseCard {
                    instance_id: 2,
                    target: ActionTarget::Squares(squares.clone()),
                },
            )
            .unwrap()
            .state;
        assert!(squares.into_iter().all(|square| {
            hypocrisy
                .board
                .get(square)
                .is_some_and(|piece| piece.owner == Player::Black && piece.kind == PieceKind::Pawn)
        }));
    }

    #[test]
    fn new_piece_movement_and_capture_reactions_are_stateful() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut siege = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::SiegeRam),
                ("d5", Player::White, PieceKind::Pawn),
                ("d6", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            203,
        );
        assert!(has_move(&engine, &siege, "d4", "d6"));
        siege = engine.apply(&siege, movement("d4", "d6")).unwrap().state;
        assert!(siege.board.get("d5".parse().unwrap()).is_none());
        assert_eq!(
            siege.board.get("d6".parse().unwrap()).unwrap().kind,
            PieceKind::SiegeRam
        );

        let mut slime = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Slime),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            204,
        );
        slime = engine.apply(&slime, movement("d4", "d7")).unwrap().state;
        assert_eq!(
            slime.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Slime
        );
        assert_eq!(
            slime.board.get("d7".parse().unwrap()).unwrap().kind,
            PieceKind::Slime
        );

        let mut reactions = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::MagicGirl),
                ("e4", Player::White, PieceKind::Siren),
                ("f4", Player::Black, PieceKind::Pawn),
                ("g4", Player::White, PieceKind::Undead),
                ("g8", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::Black,
            205,
        );
        reactions = engine
            .apply(&reactions, movement("g8", "g4"))
            .unwrap()
            .state;
        assert!(reactions.card_runtime.magic_girl_surge[Player::White.index()]);
        assert_eq!(reactions.card_runtime.undead_resurrections.len(), 1);
        assert!(has_move(&engine, &reactions, "d4", "d8"));
        let mut events = Vec::new();
        for _ in 0..4 {
            engine.tick_siren_exposure(&mut reactions, &mut events);
        }
        assert_eq!(
            reactions.board.get("f4".parse().unwrap()).unwrap().owner,
            Player::White
        );
        reactions.ply = reactions.card_runtime.undead_resurrections[0].due_ply;
        engine.resolve_undead_resurrections(&mut reactions, &mut events);
        assert!(reactions.board.contains(Player::White, PieceKind::Undead));
    }

    #[test]
    fn relay_frontline_gomoku_and_replay_follow_0818_rules() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut relay = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Rook),
                ("d2", Player::White, PieceKind::Bishop),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            206,
        );
        relay.card_runtime.relay[Player::White.index()] = true;
        relay = engine.apply(&relay, movement("a2", "d2")).unwrap().state;
        assert_eq!(
            relay.board.get("a2".parse().unwrap()).unwrap().kind,
            PieceKind::Bishop
        );
        assert_eq!(
            relay.board.get("d2".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );

        let mut frontline = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a4", Player::White, PieceKind::Rook),
                ("h4", Player::Black, PieceKind::Rook),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::Black,
            207,
        );
        frontline.card_runtime.frontline_response[Player::White.index()] = true;
        assert!(!has_move(&engine, &frontline, "h4", "a4"));

        let mut gomoku = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Pawn),
                ("a3", Player::White, PieceKind::Pawn),
                ("a4", Player::White, PieceKind::Pawn),
                ("a5", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            208,
        );
        grant_test_card(&mut gomoku, crate::GOMOKU, 3);
        gomoku = engine
            .apply(&gomoku, use_card_without_target(3))
            .unwrap()
            .state;
        assert_eq!(
            gomoku.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect("gomoku".to_owned()),
            }
        );

        let mut replay = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            209,
        );
        grant_test_card(&mut replay, crate::REPLAY, 4);
        replay = engine.apply(&replay, movement("a1", "a2")).unwrap().state;
        replay = engine.apply(&replay, movement("h8", "h7")).unwrap().state;
        replay = engine
            .apply(&replay, use_card_without_target(4))
            .unwrap()
            .state;
        assert_eq!(
            replay.board.get("a1".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        assert!(replay.board.get("a2".parse().unwrap()).is_none());
    }

    #[test]
    fn chess_n_pow_30_builds_equal_value_independent_armies_and_blocks_early_capture() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("e8", Player::Black, PieceKind::King),
            ],
            Player::White,
            210,
        );
        let state = engine
            .apply_rule(&state, crate::CHESS_N_POW_30)
            .unwrap()
            .state;
        assert_eq!(state.board.iter().count(), 32);
        let lineup = |player| {
            state
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == player && piece.kind != PieceKind::King)
                .map(|(_, piece)| piece.kind)
                .collect::<Vec<_>>()
        };
        let white = lineup(Player::White);
        let black = lineup(Player::Black);
        assert_eq!(white.len(), 15);
        assert_eq!(black.len(), 15);
        let value = |lineup: &[PieceKind]| {
            lineup
                .iter()
                .map(|kind| {
                    rule::CHESS_N_POW_30_KINDS
                        .iter()
                        .find(|(candidate, _, _)| candidate == kind)
                        .map(|(_, value, _)| u16::from(*value))
                        .unwrap()
                })
                .sum::<u16>()
        };
        assert_eq!(value(&white), value(&black));
        assert!((39..=70).contains(&value(&white)));
        assert_eq!(state.card_runtime.chaos_no_capture_until_ply, state.ply + 4);
    }

    #[test]
    fn suspicious_potion_exposes_the_exact_28_effect_union() {
        let mut effects = BTreeSet::new();
        for (royal, kind, ranged) in [
            (true, PieceKind::King, false),
            (false, PieceKind::Pawn, false),
            (false, PieceKind::Knight, false),
            (false, PieceKind::Queen, true),
            (false, PieceKind::Man, false),
        ] {
            effects.extend(middle::suspicious_potion_effects(royal, kind, ranged));
        }
        assert_eq!(effects.len(), 28);
        assert!(effects.contains("last-resistance"));
        assert!(effects.contains("queens-gambit-protection"));
        assert!(effects.contains("trojan-horse"));
        assert!(effects.contains("severance"));
    }

    #[test]
    fn trickster_uses_all_43_non_log_non_trickster_movement_types() {
        assert_eq!(TRICKSTER_MOVEMENT_KINDS.len(), 43);
        assert_eq!(
            TRICKSTER_MOVEMENT_KINDS
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            43
        );
        assert!(!TRICKSTER_MOVEMENT_KINDS.contains(&PieceKind::Log));
        assert!(!TRICKSTER_MOVEMENT_KINDS.contains(&PieceKind::Trickster));

        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Trickster),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            211,
        );
        let trickster = state.board.get("d4".parse().unwrap()).unwrap();
        state
            .card_runtime
            .trickster_move_kinds
            .insert(trickster.id, PieceKind::Wizard);
        state.card_runtime.wizard_mana.insert(trickster.id, 1);
        assert!(engine.legal_actions(&state).unwrap().iter().any(|action| {
            matches!(
                action,
                Action::UsePieceAbility {
                    piece_id,
                    ability: PieceAbility::WizardLightning,
                    ..
                } if *piece_id == trickster.id
            )
        }));
    }

    #[test]
    fn replay_restores_captured_piece_runtime_after_the_opponents_move() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("a2", Player::Black, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            212,
        );
        let pawn = state.board.get("a2".parse().unwrap()).unwrap();
        state.card_runtime.basic_training_piece_ids.insert(pawn.id);
        grant_test_card(&mut state, crate::REPLAY, 5);
        state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        state = engine
            .apply(&state, use_card_without_target(5))
            .unwrap()
            .state;
        assert_eq!(state.board.get("a2".parse().unwrap()), Some(pawn));
        assert!(
            state
                .card_runtime
                .basic_training_piece_ids
                .contains(&pawn.id)
        );
    }

    #[test]
    fn replay_is_not_legal_after_the_moved_large_piece_loses_hp() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("e1", Player::White, PieceKind::King),
                ("a1", Player::White, PieceKind::BigRook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            213,
        );
        let rook = state.board.get("a1".parse().unwrap()).unwrap();
        state.card_runtime.large_pieces.insert(
            rook.id,
            LargePieceStatus {
                anchor: "a1".parse().unwrap(),
                hp: 2,
                max_hp: 2,
            },
        );
        grant_test_card(&mut state, crate::REPLAY, 5);

        let rook_move = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .find(|action| {
                matches!(
                    action,
                    Action::Move(movement) if movement.from == "a1".parse().unwrap()
                )
            })
            .expect("Big Rook must have a legal move");
        state = engine.apply(&state, rook_move).unwrap().state;
        state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        state
            .card_runtime
            .large_pieces
            .get_mut(&rook.id)
            .unwrap()
            .hp = 1;

        let replay = use_card_without_target(5);
        assert!(!engine.legal_actions(&state).unwrap().contains(&replay));
        assert!(matches!(
            engine.apply(&state, replay),
            Err(EngineError::IllegalAction(_))
        ));
    }

    #[test]
    fn september_deletion_and_undead_timing_regression() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("b2", Player::White, PieceKind::Undead),
                ("c2", Player::White, PieceKind::Wizard),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            901,
        );
        let square = "b2".parse().unwrap();
        let undead = state.board.get(square).unwrap();
        let wizard = state.board.get("c2".parse().unwrap()).unwrap();
        let mut events = Vec::new();
        engine.delete_piece_for_card(
            &mut state,
            square,
            undead,
            Player::White,
            CLEANUP,
            &mut events,
        );
        assert!(state.board.get(square).is_none());
        assert!(state.card_runtime.undead_resurrections.is_empty());
        assert_eq!(state.card_runtime.magic_girl_surge, [false, false]);
        assert_eq!(
            state
                .card_runtime
                .wizard_mana
                .get(&wizard.id)
                .copied()
                .unwrap_or(0),
            0
        );
        state.ply = 10;
        engine.record_new_capture_reactions(&mut state, undead);
        assert_eq!(state.card_runtime.undead_resurrections[0].due_ply, 16);
    }

    #[test]
    fn september_queens_gambit_transform_protection_regression() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("d2", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            902,
        );
        let pawn = state.board.get("d2".parse().unwrap()).unwrap();
        state.card_runtime.protected_piece_ids.insert(pawn.id);
        state.card_runtime.queens_gambit_protection.insert(pawn.id);
        engine.clear_promotion_inherited_traits(&mut state, pawn.id);
        assert!(state.card_runtime.protected_piece_ids.contains(&pawn.id));
        engine.mark_transformed_piece(&mut state, pawn);
        assert!(!state.card_runtime.protected_piece_ids.contains(&pawn.id));
        assert!(
            !state
                .card_runtime
                .queens_gambit_protection
                .contains(&pawn.id)
        );
    }

    #[test]
    fn search_chance_is_reproducible_and_independent_of_actual_rng() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = engine.new_game(Mode::Standard, 701).unwrap();
        let actual_rng_before = state.rng_state();
        assert!(matches!(state.phase, GamePhase::Draft(_)));
        let mut different_actual_rng = state.clone();
        different_actual_rng.rng_mut().next_u64();
        assert_ne!(state.rng_state(), different_actual_rng.rng_state());

        let action = Action::DraftPick { offer_index: 0 };
        let mut left_context = SearchChanceContext::new(0xabc1);
        let mut right_context = SearchChanceContext::new(0xabc1);
        let left = engine
            .apply_for_search(&state, action.clone(), &mut left_context)
            .unwrap();
        let right = engine
            .apply_for_search(&different_actual_rng, action, &mut right_context)
            .unwrap();

        assert!(left.sampled_chance);
        assert_eq!(left.chance_event_index, Some(0));
        assert_eq!(left.transition.state, right.transition.state);
        assert_eq!(left_context.event_index(), 1);
        assert_eq!(right_context.event_index(), 1);
        assert_eq!(state.rng_state(), actual_rng_before);
    }

    #[test]
    fn deterministic_search_transition_is_identical_and_does_not_advance_context() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 17);
        state.phase = GamePhase::Playing;
        let action = engine.legal_actions(&state).unwrap().remove(0);
        let expected = engine.apply(&state, action.clone()).unwrap();
        let mut context = SearchChanceContext::new(99);
        let actual = engine
            .apply_for_search(&state, action, &mut context)
            .unwrap();

        assert!(!actual.sampled_chance);
        assert_eq!(actual.chance_event_index, None);
        assert_eq!(actual.transition.state, expected.state);
        assert_eq!(actual.transition.events, expected.events);
        assert_eq!(context.event_index(), 0);
    }

    #[test]
    fn search_chance_uses_the_exact_engine_distribution_for_each_sample_seed() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = engine.new_game(Mode::Standard, 808).unwrap();
        let action = Action::DraftPick { offer_index: 0 };
        for base_seed in 0..64u64 {
            let mut context = SearchChanceContext::new(base_seed);
            let sampled = engine
                .apply_for_search(&state, action.clone(), &mut context)
                .unwrap();
            let expected_seed = mix_search_seed(base_seed ^ 0x4348_414e_4345_5f31);
            let mut expected_state = state.clone();
            *expected_state.rng_mut() = crate::DeterministicRng::new(expected_seed);
            let expected = engine.apply(&expected_state, action.clone()).unwrap();
            assert_eq!(sampled.transition.state, expected.state);
            assert_eq!(sampled.transition.events, expected.events);
        }
    }
    #[test]
    fn trickster_sampling_rook_does_not_gain_overtake_cross() {
        // Live overtake helper checks attacker.type on the original board piece;
        // a trickster sampling rook movement is not a board rook, so the cross
        // must not apply even while overtake is armed for its owner.
        let engine = Engine::bundled_20260818().unwrap();
        let mut board = crate::Board::empty();
        for (square, owner, kind) in [
            ("a1", Player::White, PieceKind::Trickster),
            ("a2", Player::White, PieceKind::Pawn),
            ("e1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
        ] {
            board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let probe_state = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::White,
            3311,
        );
        let granted = engine
            .grant_card(&probe_state, 1, "overtake", Player::White)
            .unwrap()
            .state;
        let activated = engine
            .apply(&granted, use_card_without_target(1))
            .unwrap()
            .state;
        // Force the trickster's movement kind to rook for this turn.
        let mut forced = activated.clone();
        let trickster_id = forced.board.get("a1".parse().unwrap()).unwrap().id;
        forced
            .card_runtime
            .trickster_move_kinds
            .insert(trickster_id, PieceKind::Rook);
        let actions = engine.legal_actions(&forced).unwrap();
        assert!(
            !actions.contains(&movement("a1", "a3")),
            "trickster-as-rook must not cross friendly pieces"
        );
    }
    #[test]
    fn nullification_blocks_only_same_kind_enemy_captures() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Rook),
                ("d8", Player::Black, PieceKind::Rook),
                ("g7", Player::Black, PieceKind::Bishop),
            ],
            Player::White,
            951,
        );
        let granted = engine
            .grant_card(&state, 1, "nullification", Player::White)
            .unwrap()
            .state;
        let targets = engine.legal_actions(&granted).unwrap();
        assert!(targets.contains(&use_card(1, "d4")));
        assert!(!targets.contains(&use_card(1, "d8")));
        let protected = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        let rook_id = protected.board.get("d4".parse().unwrap()).unwrap().id;
        assert!(
            protected
                .card_runtime
                .nullification_piece_ids
                .contains(&rook_id)
        );
        let black = engine
            .apply(&protected, movement("a1", "b1"))
            .unwrap()
            .state;
        let moves = engine.legal_actions(&black).unwrap();
        assert!(!moves.contains(&movement("d8", "d4")));
        assert!(moves.contains(&movement("g7", "d4")));
    }
    #[test]
    fn vanguard_gives_only_the_unique_front_pawn_quiet_diagonals() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Pawn),
                ("a3", Player::White, PieceKind::Pawn),
            ],
            Player::White,
            952,
        );
        let granted = engine
            .grant_card(&state, 1, "vanguard", Player::White)
            .unwrap()
            .state;
        let moves = engine.legal_actions(&granted).unwrap();
        assert!(moves.contains(&movement("d4", "c5")));
        assert!(moves.contains(&movement("d4", "e5")));
        assert!(!moves.contains(&movement("a3", "b4")));

        let mut tied = granted.clone();
        tied.board.set(
            "f4".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn).with_id(99)),
        );
        let tied_moves = engine.legal_actions(&tied).unwrap();
        assert!(!tied_moves.contains(&movement("d4", "c5")));
        assert!(!tied_moves.contains(&movement("f4", "e5")));
    }
    #[test]
    fn infiltration_submerges_piece_reaching_isolated_enemy_home_rank() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("b7", Player::White, PieceKind::Rook),
            ],
            Player::White,
            952,
        );
        state.card_runtime.infiltration[Player::White.index()] = true;
        let transitioned = engine.apply(&state, movement("b7", "b8")).unwrap().state;
        let rook = transitioned.board.get("b8".parse().unwrap()).unwrap();
        assert!(
            transitioned
                .card_runtime
                .submerged_piece_ids
                .contains(&rook.id)
        );
    }

    #[test]
    fn resolve_grants_extra_move_to_pawn_after_friendly_pawn_captured() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = minimal_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c4", Player::White, PieceKind::Pawn),
                ("e2", Player::White, PieceKind::Pawn),
                ("e5", Player::Black, PieceKind::Knight),
            ],
            Player::Black,
            953,
        );
        state.card_runtime.resolve[Player::White.index()] = true;
        // Black captures White pawn on c4 with knight
        let after_capture = engine.apply(&state, movement("e5", "c4")).unwrap().state;
        assert!(after_capture.card_runtime.resolve_ready[Player::White.index()]);
        assert_eq!(after_capture.turn, Player::White);

        // White moves pawn e2->e3
        let after_pawn_move = engine
            .apply(&after_capture, movement("e2", "e3"))
            .unwrap()
            .state;
        // Resolve gave extra action, so White's turn continues!
        assert_eq!(after_pawn_move.turn, Player::White);
        assert!(!after_pawn_move.card_runtime.resolve_ready[Player::White.index()]);
    }

    #[test]
    fn metal_restricts_range_and_applies_three_owner_turns_of_cooldown() {
        let engine = Engine::bundled_20260915().unwrap();
        let state = september_state(
            &[
                ("h1", Player::White, PieceKind::King),
                ("d4", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1001,
        );
        let granted = engine
            .grant_card(&state, 1, crate::METAL, Player::White)
            .unwrap()
            .state;
        assert!(
            engine
                .legal_actions(&granted)
                .unwrap()
                .contains(&use_card(1, "d4"))
        );
        let metalized = engine.apply(&granted, use_card(1, "d4")).unwrap().state;
        assert!(has_move(&engine, &metalized, "d4", "d5"));
        assert!(!has_move(&engine, &metalized, "d4", "d8"));

        let mut cooled = engine
            .apply(&metalized, movement("d4", "d5"))
            .unwrap()
            .state;
        let rook_id = cooled.board.get("d5".parse().unwrap()).unwrap().id;
        assert_eq!(cooled.card_runtime.metal_cooldowns.get(&rook_id), Some(&3));
        cooled.turn = Player::White;
        cooled.actions_remaining = 1;
        assert!(!has_move(&engine, &cooled, "d5", "d6"));
        engine.tick_card_statuses_after_turn(&mut cooled, Player::White);
        engine.tick_card_statuses_after_turn(&mut cooled, Player::White);
        engine.tick_card_statuses_after_turn(&mut cooled, Player::White);
        assert!(!cooled.card_runtime.metal_cooldowns.contains_key(&rook_id));
        assert!(has_move(&engine, &cooled, "d5", "d6"));
        assert!(!has_move(&engine, &cooled, "d5", "d8"));
    }

    #[test]
    fn false_start_advances_each_friendly_file_as_far_as_its_path_allows() {
        let engine = Engine::bundled_20260915().unwrap();
        let state = september_state(
            &[
                ("a1", Player::White, PieceKind::Rook),
                ("a2", Player::White, PieceKind::Pawn),
                ("b2", Player::White, PieceKind::Pawn),
                ("h1", Player::White, PieceKind::King),
                ("a4", Player::Black, PieceKind::Bishop),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1002,
        );
        let granted = engine
            .grant_card(&state, 2, crate::FALSE_START, Player::White)
            .unwrap()
            .state;
        let advanced = engine
            .apply(&granted, use_card_without_target(2))
            .unwrap()
            .state;
        assert_eq!(
            advanced.board.get("a2".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        assert_eq!(
            advanced.board.get("a3".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        assert_eq!(
            advanced.board.get("b4".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        assert_eq!(
            advanced.board.get("h3".parse().unwrap()).unwrap().kind,
            PieceKind::King
        );
        assert_eq!(
            advanced.board.get("a4".parse().unwrap()).unwrap().owner,
            Player::Black
        );
    }

    #[test]
    fn locust_swarm_grants_one_adjacent_hurdle_jump_to_unmoved_non_pawns() {
        let engine = Engine::bundled_20260915().unwrap();
        let state = september_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("c3", Player::White, PieceKind::Knight),
                ("c4", Player::White, PieceKind::Pawn),
                ("c6", Player::White, PieceKind::Pawn),
                ("e3", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1003,
        );
        let granted = engine
            .grant_card(&state, 3, crate::LOCUST_SWARM, Player::White)
            .unwrap()
            .state;
        let marked = engine
            .apply(&granted, use_card_without_target(3))
            .unwrap()
            .state;
        let knight_id = marked.board.get("c3".parse().unwrap()).unwrap().id;
        assert!(
            marked
                .card_runtime
                .locust_swarm_piece_ids
                .contains(&knight_id)
        );
        assert!(has_move(&engine, &marked, "c3", "c5"));
        assert!(!has_move(&engine, &marked, "c3", "f3"));

        let mut moved = engine.apply(&marked, movement("c3", "c5")).unwrap().state;
        assert!(
            !moved
                .card_runtime
                .locust_swarm_piece_ids
                .contains(&knight_id)
        );
        moved.turn = Player::White;
        moved.actions_remaining = 1;
        assert!(!has_move(&engine, &moved, "c5", "c7"));
    }

    #[test]
    fn thief_transforms_a_queen_leaps_and_is_arrested_after_losing_submerge() {
        let engine = Engine::bundled_20260915().unwrap();
        let state = september_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("c3", Player::White, PieceKind::Queen),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1005,
        );
        let granted = engine
            .grant_card(&state, 5, THIEF, Player::White)
            .unwrap()
            .state;
        let transformed = engine
            .apply(&granted, use_card_without_target(5))
            .unwrap()
            .state;
        let thief = transformed.board.get("c3".parse().unwrap()).unwrap();
        assert_eq!(thief.kind, PieceKind::Thief);
        assert!(
            transformed
                .card_runtime
                .submerged_piece_ids
                .contains(&thief.id)
        );
        assert!(has_move(&engine, &transformed, "c3", "c6"));
        assert!(!has_move(&engine, &transformed, "c3", "d4"));

        let mut exposed = september_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("c3", Player::White, PieceKind::Thief),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1006,
        );
        let thief_id = exposed.board.get("c3".parse().unwrap()).unwrap().id;
        assert!(!exposed.card_runtime.submerged_piece_ids.contains(&thief_id));
        exposed = engine.apply(&exposed, movement("a1", "a2")).unwrap().state;
        assert!(exposed.board.find_by_id(thief_id).is_none());
    }

    #[test]
    fn brutus_can_capture_its_friendly_king() {
        let engine = Engine::bundled_20260915().unwrap();
        let state = september_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("a3", Player::White, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1007,
        );
        let granted = engine
            .grant_card(&state, 6, BRUTUS, Player::White)
            .unwrap()
            .state;
        assert_eq!(
            granted.board.get("a3".parse().unwrap()).unwrap().kind,
            PieceKind::Brutus
        );
        assert!(has_move(&engine, &granted, "a3", "a1"));
        let betrayed = engine.apply(&granted, movement("a3", "a1")).unwrap().state;
        assert_eq!(
            betrayed.status,
            GameStatus::Won {
                winner: Player::Black,
                reason: WinReason::KingCaptured
            }
        );
    }

    #[test]
    fn big_bishop_opening_creates_a_two_by_two_diagonal_large_piece() {
        let engine = Engine::bundled_20260915().unwrap();
        let state = september_state(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
            1008,
        );
        let granted = engine
            .grant_card(&state, 7, BIG_BISHOP, Player::White)
            .unwrap()
            .state;
        let mut opened = engine
            .apply(&granted, use_card_without_target(7))
            .unwrap()
            .state;
        let bishop = opened.board.get("f2".parse().unwrap()).unwrap();
        assert_eq!(bishop.kind, PieceKind::BigBishop);
        assert_eq!(
            opened
                .card_runtime
                .large_pieces
                .get(&bishop.id)
                .unwrap()
                .anchor,
            "f2".parse().unwrap()
        );
        opened.board.set("f3".parse().unwrap(), None);
        assert!(has_move(&engine, &opened, "f2", "e3"));
        assert!(!has_move(&engine, &opened, "f2", "f3"));
    }

    #[test]
    fn recurrence_marks_a_random_non_pawn_and_respawns_it_on_the_home_rank() {
        let engine = Engine::bundled_20260915().unwrap();
        let mut state = september_state(
            &[
                ("a2", Player::White, PieceKind::Pawn),
                ("c3", Player::White, PieceKind::Bishop),
                ("c8", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::Black,
            1004,
        );
        state.card_runtime.democracy[Player::White.index()] = true;
        let granted = engine
            .grant_card(&state, 4, crate::RECURRENCE, Player::White)
            .unwrap()
            .state;
        let bishop_id = granted.board.get("c3".parse().unwrap()).unwrap().id;
        assert!(
            granted
                .card_runtime
                .recurrence_piece_ids
                .contains(&bishop_id)
        );

        let captured = engine.apply(&granted, movement("c8", "c3")).unwrap().state;
        let (square, bishop) = captured.board.find_by_id(bishop_id).unwrap();
        assert_eq!(square.rank(), 0);
        assert_eq!(bishop.kind, PieceKind::Bishop);
        assert!(captured.card_runtime.pending_recurrences.is_empty());
        assert!(
            !captured
                .card_runtime
                .recurrence_piece_ids
                .contains(&bishop_id)
        );
    }
}

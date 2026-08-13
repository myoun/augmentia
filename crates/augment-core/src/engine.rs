use crate::{
    Action, ActionTarget, CardActivation, CardInstance, CardPhase, ChoiceContinuation,
    ChoiceRegistry, DeathmatchState, DraftPhase, DraftState, DrawReason, EMPTY_LUNCHBOX,
    EffectCommand, EffectRegistry, EmptyLunchboxStatus, EnPassantState, Event,
    ForcedMoveContinuation, ForcedMoveSource, GENEVA_CONVENTION, GamePhase, GameState, GameStatus,
    IMPLEMENTED_CARDS, KING_OF_THE_HILL, Mode, MoveAction, PARRY, PLATFORM, PendingChoice, Piece,
    PieceKind, PlatformRuleState, Player, RulesetManifest, Square, Transition, WinReason,
    implemented_card,
};
use std::collections::BTreeSet;
use std::fmt;

pub struct Engine {
    ruleset: RulesetManifest,
    effects: EffectRegistry,
    choices: ChoiceRegistry,
}

const MAX_EFFECT_EVENTS_PER_RESOLUTION: usize = 4_096;
const MAX_AUTOMATIC_TURN_ENDS_PER_TRANSITION: usize = 256;
const DRAFT_BALANCE_MAX_HALF_STAR_GAP: u64 = 2;
const DRAFT_BALANCE_EXTRA_ATTEMPTS: usize = 2;

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
        RulesetManifest::bundled_20260807().map(Self::new)
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
        if self.ruleset.mode(mode.id()).is_none() {
            return Err(EngineError::UnknownMode(mode.id().to_owned()));
        }
        if mode != Mode::Standard {
            return Err(EngineError::UnsupportedMode(mode.id().to_owned()));
        }

        let mut state = GameState::standard(self.ruleset.id.clone(), mode, seed);
        state.reset_repetition_history();
        self.start_draft(&mut state, DraftPhase::Opening, Player::White, None, None);
        Ok(state)
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
        match definition_id {
            KING_OF_THE_HILL => next.card_runtime.hill_king[owner.index()] = true,
            GENEVA_CONVENTION => next.card_runtime.geneva_convention[owner.index()] = true,
            _ => {}
        }
        let mut events = vec![Event::CardGranted {
            owner,
            instance_id,
            definition_id: definition_id.to_owned(),
        }];
        self.resolve_effect_queue_from(&mut next, &mut events, 0)?;
        Ok(Transition {
            state: next,
            events,
        })
    }

    /// Install a global RULE card. RULE cards are not owned and do not contribute stars.
    pub fn apply_rule(
        &self,
        state: &GameState,
        definition_id: &str,
    ) -> Result<Transition, EngineError> {
        self.ensure_ruleset(state)?;
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

        let mut next = state.clone();
        next.applied_rules.push(definition_id.to_owned());
        next.applied_rules.sort();
        let current_turn = next.shared_turn_count();
        next.card_runtime.platform = Some(PlatformRuleState {
            cell: None,
            spawned_at: current_turn,
            next_at: current_turn.saturating_add(5),
            triggered_piece_ids: BTreeSet::new(),
            nonce: 0,
        });
        let mut events = vec![Event::RuleApplied {
            definition_id: definition_id.to_owned(),
        }];
        self.spawn_platform(&mut next, current_turn, &mut events);
        Ok(Transition {
            state: next,
            events,
        })
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
        let resume_turn = resume_turn.unwrap_or(Player::White);
        let offers = self.draw_draft_offers(state, player, phase, balance_target);
        state.turn = player;
        state.phase = GamePhase::Draft(DraftState {
            phase,
            offers,
            resume_turn,
        });
    }

    fn apply_card_definition(&self, state: &mut GameState, definition_id: &str, owner: Player) {
        match definition_id {
            KING_OF_THE_HILL => state.card_runtime.hill_king[owner.index()] = true,
            GENEVA_CONVENTION => state.card_runtime.geneva_convention[owner.index()] = true,
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
                self.apply_card_definition(state, &definition_id, player);
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
        if !delayed && definition.activation == CardActivation::Passive {
            self.apply_card_definition(&mut next, &definition_id, actor);
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
                definition_id,
            },
        ];

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
        if legal_actions.binary_search(&action).is_err() {
            return Err(EngineError::IllegalAction(format!("{action:?}")));
        }

        match action.clone() {
            Action::Move(movement) => self.apply_move(state, action, movement),
            Action::UseCard {
                instance_id,
                target,
            } => self.apply_active_card(state, action, instance_id, target),
            Action::DraftPick { offer_index } => self.apply_draft_pick(state, action, offer_index),
            Action::ResolveChoice {
                choice_id,
                option_index,
            } => self.apply_choice(state, action, choice_id, option_index),
        }
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

        let target_square = match target {
            ActionTarget::PieceAt(square) => square,
            _ => {
                return Err(EngineError::Invariant(
                    "implemented active card requires a piece target".to_owned(),
                ));
            }
        };
        let target_piece = next.board.get(target_square).ok_or_else(|| {
            EngineError::Invariant("legal card target has no target piece".to_owned())
        })?;
        match definition_id.as_str() {
            PARRY => {
                next.card_runtime.parry_piece_ids.insert(target_piece.id);
            }
            EMPTY_LUNCHBOX => {
                next.card_runtime.empty_lunchboxes.insert(
                    target_piece.id,
                    EmptyLunchboxStatus {
                        by: actor,
                        deadline_turn: next.turns_taken[target_piece.owner.index()]
                            .saturating_add(3),
                    },
                );
            }
            _ => {
                return Err(EngineError::Invariant(format!(
                    "unsupported active card {definition_id:?} was legal"
                )));
            }
        }
        next.quiet_ply = 0;
        next.mark_deathmatch_progress();
        let mut events = vec![
            Event::ActionApplied { actor, action },
            Event::CardUsed {
                owner: actor,
                instance_id,
                definition_id,
            },
        ];
        self.resolve_effect_queue_from(&mut next, &mut events, 0)?;
        self.resolve_missing_king(&mut next, &mut events);
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

    fn apply_move(
        &self,
        state: &GameState,
        action: Action,
        movement: MoveAction,
    ) -> Result<Transition, EngineError> {
        let mut next = state.clone();
        let actor = next.turn;
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
        let moving_piece = next
            .board
            .get(movement.from)
            .ok_or_else(|| EngineError::Invariant("legal move has no source piece".to_owned()))?;
        let destination_piece = next.board.get(movement.to);
        let en_passant_capture = (moving_piece.kind == PieceKind::Pawn
            && destination_piece.is_none()
            && movement.from.file() != movement.to.file())
        .then_some(next.en_passant)
        .flatten()
        .filter(|en_passant| {
            en_passant.target == movement.to
                && en_passant.pawn_owner != actor
                && en_passant.captured.rank() == movement.from.rank()
                && en_passant.captured.file() == movement.to.file()
        })
        .and_then(|en_passant| {
            next.board
                .get(en_passant.captured)
                .filter(|piece| {
                    piece.kind == PieceKind::Pawn && piece.owner == en_passant.pawn_owner
                })
                .map(|piece| (en_passant.captured, piece))
        });
        let capture_square = en_passant_capture
            .map(|(square, _)| square)
            .unwrap_or(movement.to);
        let target_piece = destination_piece.or(en_passant_capture.map(|(_, piece)| piece));
        let irreversible = moving_piece.kind == PieceKind::Pawn || target_piece.is_some();
        let mut moving_survived = true;
        let mut events = vec![Event::ActionApplied {
            actor,
            action: action.clone(),
        }];

        next.board.set(movement.from, None);
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
                self.clear_removed_piece_statuses(&mut next, moving_piece.id);
                let mut retaliator = captured;
                retaliator.capture_count = retaliator.capture_count.saturating_add(1);
                next.board.set(movement.from, Some(retaliator));
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
            } else {
                self.clear_removed_piece_statuses(&mut next, captured.id);
                let mut placed_piece = moving_piece;
                if let Some(promotion) = movement.promotion {
                    placed_piece.kind = promotion;
                }
                placed_piece.capture_count = placed_piece.capture_count.saturating_add(1);
                next.board.set(movement.to, Some(placed_piece));
                events.push(Event::PieceCaptured {
                    actor,
                    piece: captured,
                    at: capture_square,
                });
            }
        } else {
            let mut placed_piece = moving_piece;
            if let Some(promotion) = movement.promotion {
                placed_piece.kind = promotion;
            }
            next.board.set(movement.to, Some(placed_piece));
        }

        let previous_en_passant = next.en_passant;
        next.en_passant = if moving_piece.kind == PieceKind::Pawn
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

        self.resolve_effect_queue_from(&mut next, &mut events, 0)?;

        self.resolve_missing_king(&mut next, &mut events);
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

    fn clear_removed_piece_statuses(&self, state: &mut GameState, piece_id: u32) {
        state.card_runtime.parry_piece_ids.remove(&piece_id);
        state.card_runtime.empty_lunchboxes.remove(&piece_id);
    }

    fn resolve_missing_king(&self, state: &mut GameState, _events: &mut Vec<Event>) {
        if state.status != GameStatus::Ongoing || !self.ruleset.base_rules.king_capture_wins {
            return;
        }
        if !state.has_king(Player::White) {
            state.status = GameStatus::Won {
                winner: Player::Black,
                reason: WinReason::KingCaptured,
            };
        } else if !state.has_king(Player::Black) {
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
        let interval_full_turns = self.ruleset.base_rules.deathmatch_quiet_move_count.max(1);
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
        self.resolve_missing_king(state, events);
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
        self.resolve_empty_lunchboxes_after_turn(state, actor, events);
        if state.status != GameStatus::Ongoing {
            self.push_game_ended(state, events);
            return Ok(());
        }
        self.tick_platform_after_turn(state, events);
        if self.tick_deathmatch_after_turn(state, actor) {
            self.push_game_ended(state, events);
            return Ok(());
        }
        self.start_deathmatch_if_due(state, events);
        events.push(Event::TurnEnded { player: actor });

        let defender = actor.opponent();
        state.turn = defender;
        state.phase = GamePhase::Playing;
        state.actions_remaining = 1;
        state.cards_used_this_turn[actor.index()] = 0;
        state.cards_used_this_turn[defender.index()] = 0;
        let next_turn_event_start = events.len();
        events.push(Event::TurnChanged { player: defender });
        self.activate_cards_for_turn(state, defender, events);
        events.push(Event::TurnStarted {
            player: defender,
            actions_remaining: state.actions_remaining,
        });
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
            return self.finish_turn_chain(state, defender, events, automatic_turn_ends + 1);
        }
        if self.start_due_milestone_draft(state, defender, events) {
            return Ok(());
        }
        self.resolve_no_action_loss(state, defender, events);

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
            .find(|(_, piece)| piece.owner == moving_color && piece.kind == PieceKind::King)
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

    fn generate_card_actions(&self, state: &GameState, actions: &mut Vec<Action>) {
        for card in &state.cards {
            let definition = implemented_card(&card.definition_id);
            if card.owner != state.turn
                || card.charges == 0
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
                PARRY => {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn
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
                            && piece.kind != PieceKind::King
                            && !state.card_runtime.empty_lunchboxes.contains_key(&piece.id)
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn generate_move_actions(&self, state: &GameState) -> Vec<Action> {
        let mut actions = Vec::new();
        for (from, piece) in state.board.iter() {
            if piece.owner != state.turn {
                continue;
            }
            self.generate_moves_for_piece(state, from, piece, &mut actions);
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
        actions: &mut Vec<Action>,
    ) {
        match piece.kind {
            PieceKind::Pawn => self.generate_pawn_moves(state, from, piece, actions),
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
                &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
                actions,
            ),
            PieceKind::Rook => self.generate_slider_moves(
                state,
                from,
                piece,
                &[(-1, 0), (1, 0), (0, -1), (0, 1)],
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
                }
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
        let (direction, double_step_ranks, promotion_rank) = match piece.owner {
            Player::White => (1, [1, 0], 7),
            Player::Black => (-1, [6, 7], 0),
        };

        if let Some(one_step) = from.offset(0, direction)
            && state.board.get(one_step).is_none()
        {
            push_pawn_move(from, one_step, promotion_rank, actions);
            if double_step_ranks.contains(&from.rank())
                && let Some(two_steps) = from.offset(0, direction * 2)
                && state.board.get(two_steps).is_none()
            {
                actions.push(Action::Move(MoveAction::new(from, two_steps, None)));
            }
        }

        for file_delta in [-1, 1] {
            let Some(target) = from.offset(file_delta, direction) else {
                continue;
            };
            let regular_capture = state
                .board
                .get(target)
                .is_some_and(|target_piece| self.can_capture(state, piece, target_piece));
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
                                && self.can_capture(state, piece, captured)
                        })
            });
            if regular_capture || en_passant_capture {
                push_pawn_move(from, target, promotion_rank, actions);
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
            match state.board.get(target) {
                None => actions.push(Action::Move(MoveAction::new(from, target, None))),
                Some(target_piece) if self.can_capture(state, piece, target_piece) => {
                    actions.push(Action::Move(MoveAction::new(from, target, None)));
                }
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

    fn can_capture(&self, state: &GameState, attacker: Piece, target: Piece) -> bool {
        attacker.owner != target.owner
            && !(attacker.kind == PieceKind::Queen
                && target.kind == PieceKind::Pawn
                && state.card_runtime.geneva_convention[target.owner.index()])
    }
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

fn push_pawn_move(from: Square, to: Square, promotion_rank: u8, actions: &mut Vec<Action>) {
    if to.rank() == promotion_rank {
        for promotion in [
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
        ] {
            actions.push(Action::Move(MoveAction::new(from, to, Some(promotion))));
        }
    } else {
        actions.push(Action::Move(MoveAction::new(from, to, None)));
    }
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
    use crate::{BUNDLED_RULESET_ID, Board, CardInstance, ChoiceHandler, EffectHandler};
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

    fn minimal_state(pieces: &[(&str, Player, PieceKind)], turn: Player, seed: u64) -> GameState {
        let mut board = Board::empty();
        for (square, owner, kind) in pieces {
            board.set(square.parse().unwrap(), Some(Piece::new(*owner, *kind)));
        }
        GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, turn, seed)
    }

    fn standard_play_state(seed: u64) -> GameState {
        GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, seed)
    }

    fn first_draft_pick(engine: &Engine, state: &GameState) -> Transition {
        engine
            .apply(state, Action::DraftPick { offer_index: 0 })
            .unwrap()
    }

    #[test]
    fn standard_game_runs_white_then_black_opening_draft_before_play() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = engine.new_game(Mode::Standard, 7).unwrap();
        let GamePhase::Draft(draft) = &state.phase else {
            panic!("standard game must start in the opening draft")
        };
        assert_eq!(draft.phase, DraftPhase::Opening);
        assert_eq!(state.turn, Player::White);
        assert_eq!(draft.offers.len(), 3);
        assert_eq!(
            draft.offers.iter().cloned().collect::<BTreeSet<_>>(),
            [KING_OF_THE_HILL, GENEVA_CONVENTION, PARRY]
                .into_iter()
                .map(str::to_owned)
                .collect()
        );
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
        assert_eq!(
            engine
                .legal_actions(&after_black)
                .unwrap()
                .iter()
                .filter(|action| matches!(action, Action::Move(_)))
                .count(),
            20
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
        assert!(!candidates.contains(EMPTY_LUNCHBOX));
        assert!(!candidates.contains(PLATFORM));
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

        let transition = engine.apply(&state, movement("g8", "f6")).unwrap();
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

        let transition = engine.apply(&state, movement("g8", "f6")).unwrap();
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
        state = first_draft_pick(&engine, &state).state;

        let white_card = state
            .cards
            .iter()
            .find(|card| card.owner == Player::White && card.definition_id == PARRY)
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
            .find(|card| card.owner == Player::White && card.definition_id == PARRY)
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
        state = engine
            .grant_card(&state, 77, PARRY, Player::White)
            .unwrap()
            .state;

        let transition = engine.apply(&state, movement("g8", "f6")).unwrap();
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
        assert_eq!(state.fingerprint(), 0xa609_c6df_568e_9f3e);
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
        assert_eq!(state.fingerprint(), 0xcbb9_b584_9fd9_534a);
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
        assert_eq!(state.fingerprint(), 0x05fa_8474_ccda_c939);
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
}

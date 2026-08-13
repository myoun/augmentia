use crate::{
    Action, ActionTarget, CardActivation, CardInstance, DrawReason, EMPTY_LUNCHBOX, EffectCommand,
    EffectRegistry, EmptyLunchboxStatus, Event, ForcedMoveSource, GENEVA_CONVENTION, GamePhase,
    GameState, GameStatus, KING_OF_THE_HILL, Mode, MoveAction, PARRY, PLATFORM, Piece, PieceKind,
    PlatformRuleState, Player, RulesetManifest, Square, Transition, WinReason, implemented_card,
};
use std::collections::BTreeSet;
use std::fmt;

pub struct Engine {
    ruleset: RulesetManifest,
    effects: EffectRegistry,
}

impl Engine {
    pub fn new(ruleset: RulesetManifest) -> Self {
        Self {
            ruleset,
            effects: EffectRegistry::default(),
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

    pub fn new_game(&self, mode: Mode, seed: u64) -> Result<GameState, EngineError> {
        if self.ruleset.mode(mode.id()).is_none() {
            return Err(EngineError::UnknownMode(mode.id().to_owned()));
        }
        Ok(GameState::standard(self.ruleset.id.clone(), mode, seed))
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
        self.apply_card_reactions(&mut next, &mut events)?;
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

    pub fn legal_actions(&self, state: &GameState) -> Result<Vec<Action>, EngineError> {
        self.ensure_ruleset(state)?;
        if state.status != GameStatus::Ongoing {
            return Ok(Vec::new());
        }
        match &state.phase {
            GamePhase::Playing => {
                let mut actions = self.generate_move_actions(state);
                self.generate_card_actions(state, &mut actions);
                actions.sort();
                actions.dedup();
                Ok(actions)
            }
            GamePhase::ForcedMove { piece_id, .. } => {
                Ok(self.generate_piece_move_actions(state, *piece_id))
            }
            GamePhase::Draft { .. } => Err(EngineError::UnsupportedPhase("draft")),
            GamePhase::AwaitingChoice(_) => Err(EngineError::UnsupportedPhase("choice")),
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
            _ => Err(EngineError::UnsupportedPhase("non-move action")),
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
        next.cards[card_index].charges = next.cards[card_index].charges.saturating_sub(1);

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
        let mut events = vec![
            Event::ActionApplied { actor, action },
            Event::CardUsed {
                owner: actor,
                instance_id,
                definition_id,
            },
        ];
        self.apply_card_reactions(&mut next, &mut events)?;
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
        let was_forced = matches!(next.phase, GamePhase::ForcedMove { .. });
        let moving_piece = next
            .board
            .get(movement.from)
            .ok_or_else(|| EngineError::Invariant("legal move has no source piece".to_owned()))?;
        let target_piece = next.board.get(movement.to);
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
            next.board.set(movement.to, None);
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
                    at: movement.to,
                });
            }
        } else {
            let mut placed_piece = moving_piece;
            if let Some(promotion) = movement.promotion {
                placed_piece.kind = promotion;
            }
            next.board.set(movement.to, Some(placed_piece));
        }

        next.ply = next.ply.saturating_add(1);
        next.quiet_ply = if irreversible {
            0
        } else {
            next.quiet_ply.saturating_add(1)
        };

        self.apply_card_reactions(&mut next, &mut events)?;

        self.resolve_missing_king(&mut next, &mut events);
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
        if was_forced {
            next.phase = GamePhase::Playing;
        }
        self.finish_turn(&mut next, actor, &mut events);
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

    fn apply_card_reactions(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let stimuli = events.clone();
        let mut cards = state.cards.clone();
        cards.sort_by_key(|card| (card.owner.index(), card.instance_id));

        for event in &stimuli {
            for card in &cards {
                let Some(handler) = self.effects.get(&card.definition_id) else {
                    continue;
                };
                for command in handler.react(card, event, state) {
                    self.reduce_effect(state, &command)?;
                    events.push(Event::EffectApplied {
                        source_card_instance: card.instance_id,
                        command: format!("{command:?}"),
                    });
                }
            }
        }
        Ok(())
    }

    fn reduce_effect(
        &self,
        state: &mut GameState,
        command: &EffectCommand,
    ) -> Result<(), EngineError> {
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
            }
            EffectCommand::RemovePiece { square } => {
                if let Some(removed) = state.board.get(*square) {
                    self.clear_removed_piece_statuses(state, removed.id);
                }
                state.board.set(*square, None);
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
        Ok(())
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

    fn finish_turn(&self, state: &mut GameState, actor: Player, events: &mut Vec<Event>) {
        if state.status != GameStatus::Ongoing {
            self.push_game_ended(state, events);
            return;
        }

        state.turns_taken[actor.index()] = state.turns_taken[actor.index()].saturating_add(1);
        self.resolve_empty_lunchboxes_after_turn(state, actor, events);
        if state.status != GameStatus::Ongoing {
            self.push_game_ended(state, events);
            return;
        }
        self.tick_platform_after_turn(state, events);

        let defender = actor.opponent();
        state.turn = defender;
        events.push(Event::TurnChanged { player: defender });
        if self.ruleset.base_rules.immobility_loses && self.generate_move_actions(state).is_empty()
        {
            state.status = GameStatus::Won {
                winner: actor,
                reason: WinReason::NoLegalActions,
            };
        }

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
            if card.owner != state.turn || card.charges == 0 {
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
        let (direction, start_rank, promotion_rank) = match piece.owner {
            Player::White => (1, 1, 7),
            Player::Black => (-1, 6, 0),
        };

        if let Some(one_step) = from.offset(0, direction)
            && state.board.get(one_step).is_none()
        {
            push_pawn_move(from, one_step, promotion_rank, actions);
            if from.rank() == start_rank
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
            if state
                .board
                .get(target)
                .is_some_and(|target_piece| self.can_capture(state, piece, target_piece))
            {
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
    UnknownCard(String),
    CardIsRule(String),
    CardIsNotRule(String),
    DuplicateCardInstance(u32),
    RuleAlreadyApplied(String),
    GameAlreadyEnded,
    UnsupportedPhase(&'static str),
    IllegalAction(String),
    InvalidEffect(String),
    Invariant(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RulesetMismatch { engine, state } => {
                write!(f, "ruleset mismatch: engine={engine:?}, state={state:?}")
            }
            Self::UnknownMode(mode) => write!(f, "unknown mode {mode:?}"),
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
            Self::Invariant(message) => write!(f, "engine invariant failed: {message}"),
        }
    }
}

impl std::error::Error for EngineError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BUNDLED_RULESET_ID, Board, CardInstance, EffectHandler};
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

    #[test]
    fn standard_position_has_twenty_ordered_actions() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = engine.new_game(Mode::Standard, 1).unwrap();
        let actions = engine.legal_actions(&state).unwrap();
        assert_eq!(actions.len(), 20);
        assert!(actions.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn e2_e4_is_a_deterministic_pure_transition() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = engine.new_game(Mode::Standard, 99).unwrap();
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
        let mut state = engine.new_game(Mode::Standard, 1).unwrap();
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
        let mut state = engine.new_game(Mode::Standard, 1).unwrap();
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
        assert_eq!(state.fingerprint(), 0x9cec_7a38_7f28_f764);
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
        assert_eq!(state.fingerprint(), 0xa47c_6c3a_b134_050d);
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
                source: ForcedMoveSource::Platform
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
        assert_eq!(state.fingerprint(), 0xe4b2_93f4_30d7_1fb7);
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
}

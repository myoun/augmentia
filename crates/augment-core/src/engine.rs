use crate::{
    Action, Board, DrawReason, EffectCommand, EffectRegistry, Event, GamePhase, GameState,
    GameStatus, Mode, MoveAction, Piece, PieceKind, Player, RulesetManifest, Square, Transition,
    WinReason,
};
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

    pub fn legal_actions(&self, state: &GameState) -> Result<Vec<Action>, EngineError> {
        self.ensure_ruleset(state)?;
        if state.status != GameStatus::Ongoing {
            return Ok(Vec::new());
        }
        match state.phase {
            GamePhase::Playing => Ok(self.generate_move_actions(state)),
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

        let mut next = state.clone();
        let actor = next.turn;
        let mut events = Vec::new();
        let mut captured_king = false;
        let irreversible;

        match action.clone() {
            Action::Move(movement) => {
                let moving_piece = next.board.get(movement.from).ok_or_else(|| {
                    EngineError::Invariant("legal move has no source piece".to_owned())
                })?;
                let captured = next.board.set(movement.to, None);
                next.board.set(movement.from, None);

                let mut placed_piece = moving_piece;
                if let Some(promotion) = movement.promotion {
                    placed_piece.kind = promotion;
                }
                next.board.set(movement.to, Some(placed_piece));

                irreversible = moving_piece.kind == PieceKind::Pawn || captured.is_some();
                events.push(Event::ActionApplied {
                    actor,
                    action: action.clone(),
                });
                if let Some(piece) = captured {
                    captured_king = piece.kind == PieceKind::King;
                    events.push(Event::PieceCaptured {
                        actor,
                        piece,
                        at: movement.to,
                    });
                }
            }
            _ => return Err(EngineError::UnsupportedPhase("non-move action")),
        }

        next.ply = next.ply.saturating_add(1);
        next.quiet_ply = if irreversible {
            0
        } else {
            next.quiet_ply.saturating_add(1)
        };

        self.apply_card_reactions(&mut next, &mut events)?;

        let defender = actor.opponent();
        if next.status == GameStatus::Ongoing
            && self.ruleset.base_rules.king_capture_wins
            && captured_king
            && !next.has_king(defender)
        {
            next.status = GameStatus::Won {
                winner: actor,
                reason: WinReason::KingCaptured,
            };
        }

        if next.status == GameStatus::Ongoing {
            next.turn = defender;
            events.push(Event::TurnChanged { player: defender });

            if self.ruleset.base_rules.immobility_loses
                && self.generate_move_actions(&next).is_empty()
            {
                next.status = GameStatus::Won {
                    winner: actor,
                    reason: WinReason::NoLegalActions,
                };
            }
        }

        if next.status == GameStatus::Ongoing {
            let repetition_count = next.record_current_position();
            if repetition_count >= self.ruleset.base_rules.repetition_count {
                let [white, black] = next.half_star_totals;
                next.status = match white.cmp(&black) {
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

        if next.status != GameStatus::Ongoing {
            events.push(Event::GameEnded {
                status: next.status.clone(),
            });
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
                state.board.set(*square, Some(*piece));
            }
            EffectCommand::RemovePiece { square } => {
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

    fn generate_move_actions(&self, state: &GameState) -> Vec<Action> {
        let mut actions = Vec::new();
        for (from, piece) in state.board.iter() {
            if piece.owner != state.turn {
                continue;
            }
            match piece.kind {
                PieceKind::Pawn => {
                    self.generate_pawn_moves(&state.board, from, piece, &mut actions)
                }
                PieceKind::Knight => self.generate_leaper_moves(
                    &state.board,
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
                    &mut actions,
                ),
                PieceKind::Bishop => self.generate_slider_moves(
                    &state.board,
                    from,
                    piece,
                    &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
                    &mut actions,
                ),
                PieceKind::Rook => self.generate_slider_moves(
                    &state.board,
                    from,
                    piece,
                    &[(-1, 0), (1, 0), (0, -1), (0, 1)],
                    &mut actions,
                ),
                PieceKind::Queen => self.generate_slider_moves(
                    &state.board,
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
                    &mut actions,
                ),
                PieceKind::King => self.generate_leaper_moves(
                    &state.board,
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
                    &mut actions,
                ),
            }
        }
        actions.sort();
        actions
    }

    fn generate_pawn_moves(
        &self,
        board: &Board,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let (direction, start_rank, promotion_rank) = match piece.owner {
            Player::White => (1, 1, 7),
            Player::Black => (-1, 6, 0),
        };

        if let Some(one_step) = from.offset(0, direction)
            && board.get(one_step).is_none()
        {
            push_pawn_move(from, one_step, promotion_rank, actions);
            if from.rank() == start_rank
                && let Some(two_steps) = from.offset(0, direction * 2)
                && board.get(two_steps).is_none()
            {
                actions.push(Action::Move(MoveAction::new(from, two_steps, None)));
            }
        }

        for file_delta in [-1, 1] {
            let Some(target) = from.offset(file_delta, direction) else {
                continue;
            };
            if board
                .get(target)
                .is_some_and(|target_piece| target_piece.owner != piece.owner)
            {
                push_pawn_move(from, target, promotion_rank, actions);
            }
        }
    }

    fn generate_leaper_moves(
        &self,
        board: &Board,
        from: Square,
        piece: Piece,
        offsets: &[(i8, i8)],
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in offsets {
            let Some(target) = from.offset(file_delta, rank_delta) else {
                continue;
            };
            if board
                .get(target)
                .is_none_or(|target_piece| target_piece.owner != piece.owner)
            {
                actions.push(Action::Move(MoveAction::new(from, target, None)));
            }
        }
    }

    fn generate_slider_moves(
        &self,
        board: &Board,
        from: Square,
        piece: Piece,
        directions: &[(i8, i8)],
        actions: &mut Vec<Action>,
    ) {
        for &(file_delta, rank_delta) in directions {
            let mut cursor = from;
            while let Some(target) = cursor.offset(file_delta, rank_delta) {
                match board.get(target) {
                    None => actions.push(Action::Move(MoveAction::new(from, target, None))),
                    Some(target_piece) if target_piece.owner != piece.owner => {
                        actions.push(Action::Move(MoveAction::new(from, target, None)));
                        break;
                    }
                    Some(_) => break,
                }
                cursor = target;
            }
        }
    }
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
    use crate::{BUNDLED_RULESET_ID, CardInstance, EffectHandler};
    use std::str::FromStr;

    fn movement(from: &str, to: &str) -> Action {
        Action::Move(MoveAction::new(
            Square::from_str(from).unwrap(),
            Square::from_str(to).unwrap(),
            None,
        ))
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
        assert_eq!(
            left.state.board.get("e4".parse().unwrap()),
            Some(Piece::new(Player::White, PieceKind::Pawn))
        );
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

use super::*;
pub(super) const ID: &str = crate::THIEF;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let queens: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Queen && !royal(state, *p))
        .collect();
    if queens.is_empty() {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant("no thief queen".into()));
    }
    let index = if queens.len() == 1 {
        0
    } else {
        let (roll, index) = state.rng_mut().web_choice_index(queens.len());
        events.push(Event::RandomSample {
            purpose: "acquire.opening.thief.target".into(),
            roll_53: roll,
        });
        index.unwrap()
    };
    let (square, piece) = queens[index];
    engine.opening_transform(state, square, PieceKind::Thief, true, false)?;
    state.card_runtime.submerged_piece_ids.insert(piece.id);
    state.card_runtime.wanted_piece_ids.insert(piece.id);
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Queen && !royal(state, p)),
        "비왕실 아군 퀸이 필요합니다.",
    )
}
impl Engine {
    pub(in crate::engine) fn start_opening_optional_move(
        &self,
        state: &mut GameState,
        piece_id: u32,
        reason: &str,
        events: &mut Vec<Event>,
    ) -> bool {
        if self.generate_piece_move_actions(state, piece_id).is_empty() {
            return false;
        }
        state.phase = GamePhase::ForcedMove {
            piece_id,
            source: ForcedMoveSource::Effect(reason.to_owned()),
            continuation: if state.actions_remaining > 0 {
                ForcedMoveContinuation::ResumeTurn
            } else {
                ForcedMoveContinuation::FinishTurn
            },
        };
        events.push(Event::ForcedMoveStarted {
            player: state.turn,
            piece_id,
            source: reason.to_owned(),
        });
        true
    }

    pub(in crate::engine) fn generate_current_thief_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        out: &mut Vec<Action>,
    ) {
        for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            for n in 1..=3 {
                let Some(to) = from.offset(df * n, dr * n) else {
                    continue;
                };
                let jumped = (1..n).any(|k| {
                    from.offset(df * k, dr * k)
                        .is_some_and(|s| self.board_piece_at(state, s).is_some())
                });
                match self.board_piece_at(state, to) {
                    None => out.push(Action::Move(MoveAction::new(from, to, None))),
                    Some(target) if !jumped && self.can_capture(state, piece, target) => {
                        out.push(Action::Move(MoveAction::new(from, to, None)))
                    }
                    _ => {}
                }
            }
        }
    }
}

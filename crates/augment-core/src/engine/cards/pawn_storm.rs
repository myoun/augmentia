use super::*;
pub(super) const ID: &str = crate::PAWN_STORM;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let direction = if actor == Player::White { 1 } else { -1 };
    let mut entries = squares_target(target)?
        .into_iter()
        .filter_map(|square| state.board.get(square).map(|piece| (square, piece)))
        .collect::<Vec<_>>();
    entries.sort_by_key(|(square, _)| {
        if actor == Player::White {
            7 - square.rank()
        } else {
            square.rank()
        }
    });
    for (from, mut piece) in entries {
        if let Some(to) = from.offset(0, direction)
            && state.board.get(to).is_none()
        {
            state.board.set(from, None);
            if to.rank() == if actor == Player::White { 7 } else { 0 } {
                piece.kind = PieceKind::Queen;
            }
            state.board.set(to, Some(piece));
            state.card_runtime.moved_piece_ids.insert(piece.id);
            engine.mark_fresh_no_capture(state, piece);
        }
    }
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    let direction = if actor == Player::White { 1 } else { -1 };
    let candidates = state
        .board
        .iter()
        .filter(|(square, piece)| {
            piece.owner == actor
                && piece.kind == PieceKind::Pawn
                && square
                    .offset(0, direction)
                    .is_some_and(|to| state.board.get(to).is_none())
        })
        .map(|(square, _)| square)
        .collect::<Vec<_>>();
    for square in &candidates {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(vec![*square]),
        });
    }
    if candidates.len() > 1 {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(candidates),
        });
    }
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let id = ID;
    denied(
        EligibilityReasonCode::UnsupportedPredicate,
        id,
        "공식 획득 조건의 런타임 매핑이 아직 검증되지 않아 획득을 허용하지 않습니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

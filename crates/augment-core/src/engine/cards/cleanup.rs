use super::*;
pub(super) const ID: &str = crate::CLEANUP;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let squares = squares_target(target)?.to_vec();
    for square in squares {
        if let Some(piece) = state.board.get(square) {
            engine.delete_piece_for_card(state, square, piece, actor.opponent(), CLEANUP, events);
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
    for (square, piece) in state.board.iter() {
        if piece.owner == actor
            && !state.card_runtime.piece_status.contains_key(&piece.id)
            && !matches!(
                piece.kind,
                PieceKind::Monster | PieceKind::Scarecrow | PieceKind::Football
            )
        {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![square]),
            });
        }
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

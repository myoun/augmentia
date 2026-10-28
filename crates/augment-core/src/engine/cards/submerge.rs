use super::*;
pub(super) const ID: &str = crate::SUBMERGE;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    _actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    state.card_runtime.submerged_piece_ids.insert(piece.id);
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    let push_piece = |actions: &mut Vec<Action>, square| {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        })
    };
    for (square, piece) in state.board.iter() {
        if piece.owner == actor
            && !engine.is_royal_piece(state, piece)
            && !state.card_runtime.submerged_piece_ids.contains(&piece.id)
            && !has_adjacent_enemy(state, square, actor)
        {
            push_piece(actions, square);
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

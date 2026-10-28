use super::*;
pub(super) const ID: &str = crate::TROJAN_HORSE;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    _actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    state.card_runtime.trojan_piece_ids.insert(piece.id);
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
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
            && piece.kind == PieceKind::Knight
            && !state.card_runtime.trojan_piece_ids.contains(&piece.id)
        {
            push_piece(actions, square);
        }
    }
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let rt = &state.card_runtime;
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    require(
        has_own(&|p| p.kind == Knight && !rt.trojan_piece_ids.contains(&p.id)),
        "트로이 목마 효과가 없는 아군 나이트가 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

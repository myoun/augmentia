use super::*;
pub(super) const ID: &str = crate::COUNTERATTACK;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn play(
    _: &Engine,
    state: &mut GameState,
    _: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, pawn) = piece_target(state, target)?;
    state.card_runtime.counterattack_ready.insert(pawn.id);
    Ok(())
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, pawn) in state.board.iter() {
        if pawn.owner == state.turn
            && pawn.kind == PieceKind::Pawn
            && !engine.is_royal_piece(state, pawn)
        {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn),
        "아군 폰이 필요합니다.",
    )
}

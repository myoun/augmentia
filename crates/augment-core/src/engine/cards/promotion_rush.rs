use super::*;
pub(super) const ID: &str = crate::PROMOTION_RUSH;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn play(
    _: &Engine,
    state: &mut GameState,
    _: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    state.card_runtime.promotion_rush_piece_ids.insert(piece.id);
    state
        .card_runtime
        .promotion_rush_capture_cleared
        .remove(&piece.id);
    Ok(())
}
fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, piece) in web_ordered_pieces(state) {
        if piece.owner == card.owner
            && piece.kind != PieceKind::Pawn
            && !state.card_runtime.large_pieces.contains_key(&piece.id)
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
            .any(|(_, p)| p.owner == owner && p.kind != PieceKind::Pawn),
        "아군 비폰 기물이 필요합니다.",
    )
}

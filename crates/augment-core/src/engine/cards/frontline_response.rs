use super::*;
pub(super) const ID: &str = crate::FRONTLINE_RESPONSE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state
        .card_runtime
        .frontline_piece_ids
        .extend(state.board.iter().filter_map(|(_, piece)| {
            (piece.owner == owner && piece.kind == PieceKind::Rook).then_some(piece.id)
        }));
    Ok(())
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Rook),
        "아군 룩이 필요합니다.",
    )
}

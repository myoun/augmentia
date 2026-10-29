use super::*;
pub(super) const ID: &str = crate::LOCUST_SWARM;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    for (_, piece) in state.board.iter() {
        if piece.owner == owner
            && matches!(
                piece.kind,
                PieceKind::Knight | PieceKind::Bishop | PieceKind::Camel | PieceKind::Rook
            )
            && !state.card_runtime.moved_piece_ids.contains(&piece.id)
        {
            state.card_runtime.locust_swarm_piece_ids.insert(piece.id);
        }
    }
    Ok(())
}
pub(super) fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

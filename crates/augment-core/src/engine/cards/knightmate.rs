use super::*;
pub(super) const ID: &str = crate::KNIGHTMATE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let _ = (&engine, &events);
    let king = web_ordered_pieces(state)
        .into_iter()
        .find(|(_, p)| p.owner == owner && royal(state, *p));
    let knights: Vec<_> = state
        .board
        .iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Knight)
        .map(|(s, _)| s)
        .collect();
    if let Some((square, piece)) = king {
        engine.opening_transform(state, square, PieceKind::RoyalKnight, true, false)?;
        state.card_runtime.royal_piece_ids.insert(piece.id);
    }
    for square in knights {
        engine.opening_transform(state, square, PieceKind::Man, true, false)?;
    }
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
            .any(|(_, p)| p.owner == owner && royal(state, p)),
        "아군 왕실 기물이 필요합니다.",
    )
}

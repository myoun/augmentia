use super::*;
pub(super) const ID: &str = crate::LAST_STAND;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let _ = (&engine, &events);
    let targets: Vec<_> = state
        .board
        .iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
        .map(|(s, _)| s)
        .collect();
    for square in targets {
        engine.opening_transform(state, square, PieceKind::Fanatic, false, true)?;
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
            .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn),
        "아군 폰이 필요합니다.",
    )
}

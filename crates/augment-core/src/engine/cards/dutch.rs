use super::*;
pub(super) const ID: &str = crate::DUTCH;
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
        .filter(|(_, p)| {
            p.owner == owner
                && matches!(
                    p.kind,
                    PieceKind::Rook | PieceKind::Bishop | PieceKind::Knight
                )
        })
        .map(|(s, _)| s)
        .collect();
    for square in targets {
        engine.opening_transform(state, square, PieceKind::Windmill, false, false)?;
    }
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    require(
        state.board.iter().any(|(_, p)| {
            p.owner == owner
                && matches!(
                    p.kind,
                    PieceKind::Rook | PieceKind::Bishop | PieceKind::Knight
                )
        }),
        "아군 룩이나 비숍 또는 나이트가 필요합니다.",
    )
}

use super::*;
pub(super) const ID: &str = crate::GUARD;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let _ = (&engine, &events);
    let files: BTreeSet<_> = state
        .board
        .iter()
        .filter(|(_, p)| p.owner == owner && royal(state, *p))
        .map(|(s, _)| s.file())
        .collect();
    let targets: Vec<_> = state
        .board
        .iter()
        .filter(|(s, p)| {
            p.owner == owner
                && matches!(p.kind, PieceKind::Pawn | PieceKind::Fanatic)
                && files.contains(&s.file())
        })
        .map(|(s, _)| s)
        .collect();
    for square in targets {
        engine.opening_transform(state, square, PieceKind::Guard, true, false)?;
    }
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    let files: BTreeSet<_> = state
        .board
        .iter()
        .filter(|(_, p)| p.owner == owner && royal(state, *p))
        .map(|(s, _)| s.file())
        .collect();
    require(
        state.board.iter().any(|(s, p)| {
            p.owner == owner
                && matches!(p.kind, PieceKind::Pawn | PieceKind::Fanatic)
                && files.contains(&s.file())
        }),
        "왕실 기물과 같은 파일에 아군 폰이 필요합니다.",
    )
}

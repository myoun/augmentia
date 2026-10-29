use super::*;
pub(super) const ID: &str = crate::QUEEN_CAVALRY;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let _ = (&engine, &events);
    let file = if state.rule_runtime.enabled.contains(crate::DIAGONAL_CHESS) {
        if owner == Player::White { 0 } else { 7 }
    } else {
        3
    };
    let target = state
        .board
        .iter()
        .filter(|(s, p)| p.owner == owner && p.kind == PieceKind::Pawn && s.file() == file)
        .min_by_key(|(s, _)| {
            if owner == Player::White {
                s.rank()
            } else {
                7 - s.rank()
            }
        })
        .map(|(s, _)| s);
    if let Some(square) = target {
        let kind = if state.card_runtime.monochrome_chess {
            PieceKind::Camel
        } else {
            PieceKind::Knight
        };
        engine.opening_transform(state, square, kind, true, false)?;
    }
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    let file = if state.rule_runtime.enabled.contains(crate::DIAGONAL_CHESS) {
        if owner == Player::White { 0 } else { 7 }
    } else {
        3
    };
    require(
        state
            .board
            .iter()
            .any(|(s, p)| p.owner == owner && p.kind == PieceKind::Pawn && s.file() == file),
        "해당 파일에 아군 폰이 필요합니다.",
    )
}

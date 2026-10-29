use super::*;
pub(super) const ID: &str = crate::BIG_ROOK;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let horde = state
        .board
        .iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
        .count()
        > 8;
    let minor = if horde {
        PieceKind::Pawn
    } else if state.card_runtime.monochrome_chess {
        PieceKind::Camel
    } else {
        PieceKind::Knight
    };
    let anchor = opening_square(owner, "g2", "g8");
    let layout = [
        (anchor, PieceKind::BigRook, false),
        (opening_square(owner, "f3", "f6"), minor, true),
        (opening_square(owner, "g3", "g6"), PieceKind::Pawn, true),
        (opening_square(owner, "h3", "h6"), PieceKind::Pawn, true),
    ];
    engine.install_card_layout(
        state,
        owner,
        &layout,
        &large_piece_cells(anchor),
        ID,
        events,
    );
    engine.resolve_missing_king(state, events);
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

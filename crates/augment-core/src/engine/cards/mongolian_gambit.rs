use super::*;
pub(super) const ID: &str = crate::MONGOLIAN_GAMBIT;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let pieces = state
        .board
        .iter()
        .filter(|(_, piece)| {
            piece.owner == owner
                && !engine.is_royal_piece(state, *piece)
                && piece.kind != PieceKind::Merchant
        })
        .collect::<Vec<_>>();
    for (square, mut piece) in pieces {
        engine.clear_piece_kind_runtime(state, piece.id);
        engine.clear_queens_gambit_protection(state, piece.id);
        piece.kind = PieceKind::Knight;
        state.board.set(square, Some(piece));
        state.card_runtime.moved_piece_ids.insert(piece.id);
    }
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    require(
        has_own(&|p| !royal(state, p) && p.kind != Merchant),
        "상인이 아닌 비왕실 아군 기물이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

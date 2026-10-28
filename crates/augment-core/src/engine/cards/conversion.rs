use super::*;
pub(super) const ID: &str = crate::CONVERSION;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let knights = state
        .board
        .iter()
        .filter(|(_, piece)| piece.owner == owner && piece.kind == PieceKind::Knight)
        .collect::<Vec<_>>();
    for (square, mut piece) in knights {
        engine.clear_queens_gambit_protection(state, piece.id);
        piece.kind = PieceKind::Bishop;
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
    let count = |owner, kind| {
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .count()
    };
    let own = |kind| count(player, kind);
    require(own(Knight) > 0, "아군 나이트가 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

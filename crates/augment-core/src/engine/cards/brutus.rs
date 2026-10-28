use super::*;
pub(super) const ID: &str = crate::BRUTUS;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let candidates = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| {
            piece.owner == owner
                && piece.kind == PieceKind::Rook
                && !engine.is_royal_piece(state, *piece)
        })
        .collect::<Vec<_>>();
    let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
    events.push(Event::RandomSample {
        purpose: "brutus-target".to_owned(),
        roll_53: roll,
    });
    if let Some(index) = index {
        let (square, mut piece) = candidates[index];
        engine.clear_piece_kind_runtime(state, piece.id);
        piece.kind = PieceKind::Brutus;
        state.board.set(square, Some(piece));
        engine.mark_transformed_piece(state, piece);
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
        has_own(&|p| p.kind == Rook && !royal(state, p)),
        "비왕실 아군 룩이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

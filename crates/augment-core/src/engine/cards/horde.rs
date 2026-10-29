use super::*;
pub(super) const ID: &str = crate::HORDE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let Some((from, king)) = web_ordered_pieces(state)
        .into_iter()
        .find(|(_, p)| p.owner == owner && royal(state, *p))
    else {
        return Ok(());
    };
    let home = opening_square(owner, "e1", "e8");
    if state.board.get(home).is_some_and(|p| p.owner != owner) {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant(
            "horde king home is occupied by the enemy".into(),
        ));
    }
    let removed: Vec<_> = state
        .board
        .iter()
        .filter(|(_, p)| p.owner == owner && p.id != king.id)
        .collect();
    for (s, p) in removed {
        engine.delete_piece_for_card(state, s, p, owner, ID, events);
    }
    state.board.set(from, None);
    state.board.set(home, Some(king));
    state.card_runtime.moved_piece_ids.insert(king.id);
    let ranks = if owner == Player::White {
        [0, 1, 2, 3]
    } else {
        [7, 6, 5, 4]
    };
    let extra = if owner == Player::White { 4 } else { 3 };
    let mut cells = Vec::new();
    for rank in ranks {
        for file in 0..8 {
            cells.push(Square::new(file, rank).unwrap());
        }
    }
    for file in [1, 2, 5, 6] {
        cells.push(Square::new(file, extra).unwrap());
    }
    for square in cells {
        if state.board.get(square).is_none() && !engine.rule_cell_collapsed(state, square) {
            let piece = Piece::new(owner, PieceKind::Pawn).with_id(state.allocate_piece_id());
            state.board.set(square, Some(piece));
            state.card_runtime.piece_origins.insert(piece.id, square);
        }
    }
    let collapsed: Vec<_> = state
        .board
        .iter()
        .filter(|(s, p)| p.owner == owner && engine.rule_cell_collapsed(state, *s))
        .collect();
    for (s, p) in collapsed {
        engine.delete_piece_for_card(state, s, p, owner, ID, events);
    }
    if state
        .cards
        .iter()
        .any(|c| c.owner == owner.opponent() && c.definition_id == crate::PAWN_CHAIN)
    {
        let pawns = state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
            .map(|(s, p)| (s, p.id))
            .collect();
        state
            .card_runtime
            .october
            .pawn_chain_links
            .extend(super::pawn_chain::pawn_links(pawns, Some(owner)));
    }
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

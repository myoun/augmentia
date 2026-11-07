use super::*;
pub(super) const ID: &str = crate::SUMMON_COLOSSUS;

/// Summon Colossus sacrifices the six central own pawns and places the colossus
/// body on the opening anchor.
///
/// This re-enabled card keeps the effect it had before being disabled upstream; the
/// 2026-11-02 change was to its name and availability only (main bundle
/// 183228-183267 for the summon, 5052-5101 for the anchor and sacrifice rules).
pub(in crate::engine) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let anchor = opening_square(actor, "d3", "d7");
    let pawns = web_ordered_pieces(state)
        .into_iter()
        .filter(|(square, piece)| {
            piece.owner == actor && piece.kind == PieceKind::Pawn && !matches!(square.file(), 0 | 7)
        })
        .collect::<Vec<_>>();
    if pawns.len() != 6 {
        return Err(EngineError::Invariant(
            "summon colossus requires the six non-rook-file pawns".to_owned(),
        ));
    }
    let sacrifice_ids: BTreeSet<_> = pawns.iter().map(|(_, piece)| piece.id).collect();
    for cell in large_piece_cells(anchor) {
        if state
            .board
            .get(cell)
            .is_some_and(|piece| !sacrifice_ids.contains(&piece.id))
        {
            return Err(EngineError::Invariant(
                "colossus footprint is blocked".to_owned(),
            ));
        }
    }
    for (square, pawn) in pawns {
        engine.remove_piece_for_card(state, square, pawn, actor.opponent(), ID, events);
    }
    let colossus = Piece::new(actor, PieceKind::Colossus).with_id(state.allocate_piece_id());
    state.board.set(anchor, Some(colossus));
    state.card_runtime.large_pieces.insert(
        colossus.id,
        LargePieceStatus {
            anchor,
            hp: 3,
            max_hp: 3,
        },
    );
    engine.register_spawned_piece(state, anchor, colossus, true);
    Ok(())
}

pub(in crate::engine) fn eligibility(
    state: &GameState,
    actor: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    let pawns = state
        .board
        .iter()
        .filter(|(square, piece)| {
            piece.owner == actor && piece.kind == PieceKind::Pawn && !matches!(square.file(), 0 | 7)
        })
        .map(|(_, piece)| piece.id)
        .collect::<BTreeSet<_>>();
    require(
        pawns.len() == 6
            && large_piece_cells(opening_square(actor, "d3", "d7"))
                .into_iter()
                .all(|square| {
                    state
                        .board
                        .get(square)
                        .is_none_or(|piece| pawns.contains(&piece.id))
                }),
        "거신병을 소환할 폰 6개와 빈 자리가 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    #[test]
    fn summon_colossus_sacrifices_six_pawns_and_places_the_body() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(&[
            ("b2", Player::White, PieceKind::Pawn),
            ("c2", Player::White, PieceKind::Pawn),
            ("d2", Player::White, PieceKind::Pawn),
            ("e2", Player::White, PieceKind::Pawn),
            ("f2", Player::White, PieceKind::Pawn),
            ("g2", Player::White, PieceKind::Pawn),
            ("a2", Player::White, PieceKind::Pawn),
            ("h2", Player::White, PieceKind::Pawn),
        ]);
        let after = engine
            .grant_card(&before, 1, crate::SUMMON_COLOSSUS, Player::White)
            .unwrap()
            .state;
        // The six central pawns are gone; the rook-file pawns stay.
        assert!(after.board.get("a2".parse().unwrap()).is_some());
        assert!(after.board.get("h2".parse().unwrap()).is_some());
        for square in ["b2", "c2", "d2", "e2", "f2", "g2"] {
            assert!(
                after.board.get(square.parse().unwrap()).is_none(),
                "{square} must be sacrificed"
            );
        }
        let body = after
            .board
            .iter()
            .find(|(_, piece)| piece.kind == PieceKind::Colossus)
            .expect("the colossus is summoned");
        assert_eq!(body.1.owner, Player::White);
        assert_eq!(after.card_runtime.large_pieces[&body.1.id].hp, 3);
    }
}

use super::*;
pub(super) const ID: &str = crate::ASSEMBLY;
pub(super) fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.october.assembly[owner.index()] = true;
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let rt = &state.card_runtime;
    let i = player.index();
    if let Some(result) = enabled(rt.october.assembly[i]) {
        return result;
    }
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn settle_assembly_capture(
        &self,
        state: &mut GameState,
        original: Piece,
        events: &[Event],
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID
            || !state.card_runtime.october.assembly[original.owner.index()]
        {
            return;
        }
        let promoted = events.iter().any(|event| matches!(event, Event::PieceCaptured { actor, piece, .. }
            if *actor == original.owner && piece.owner != original.owner
            && matches!((original.kind, piece.kind), (PieceKind::Rook, PieceKind::Bishop) | (PieceKind::Bishop, PieceKind::Rook))));
        if promoted && let Some((square, mut piece)) = state.board.find_by_id(original.id) {
            piece.kind = PieceKind::Queen;
            state.board.set(square, Some(piece));
            state.card_runtime.moved_piece_ids.insert(piece.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::interactions::*;
    use super::*;
    #[test]
    fn assembly_is_passive_promotes_only_real_enemy_cross_captures_and_preserves_identity() {
        let engine = Engine::bundled_latest().unwrap();
        for (attacker, victim, expected) in [
            (PieceKind::Rook, PieceKind::Bishop, PieceKind::Queen),
            (PieceKind::Bishop, PieceKind::Rook, PieceKind::Queen),
            (PieceKind::Rook, PieceKind::Pawn, PieceKind::Rook),
        ] {
            let (from, to) = if attacker == PieceKind::Bishop {
                ("c3", "e5")
            } else {
                ("c3", "c5")
            };
            let state = position(
                &[
                    ("a1", Player::White, PieceKind::King),
                    ("h8", Player::Black, PieceKind::King),
                    (from, Player::White, attacker),
                    (to, Player::Black, victim),
                ],
                Player::White,
            );
            let id = state.board.get(from.parse().unwrap()).unwrap().id;
            let state = engine
                .grant_card(&state, 1, crate::ASSEMBLY, Player::White)
                .unwrap()
                .state;
            assert_eq!(state.cards[0].charges, 0);
            let next = engine.apply(&state, movement(from, to)).unwrap().state;
            let piece = next.board.get(to.parse().unwrap()).unwrap();
            assert_eq!((piece.id, piece.kind), (id, expected));
        }
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Rook),
                ("c5", Player::Black, PieceKind::Bishop),
            ],
            Player::White,
        );
        let mut state = engine
            .grant_card(&state, 1, crate::ASSEMBLY, Player::White)
            .unwrap()
            .state;
        let bishop = state.board.get("c5".parse().unwrap()).unwrap();
        state.card_runtime.shielded_piece_ids.insert(bishop.id);
        let next = engine.apply(&state, movement("c3", "c5")).unwrap().state;
        assert_eq!(
            next.board.get("c3".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        // No actual bishop capture occurred when a shield was consumed.
        assert!(next.board.find_by_id(bishop.id).is_some());
    }
}

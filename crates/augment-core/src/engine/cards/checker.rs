use super::*;
pub(super) const ID: &str = crate::CHECKER;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    engine.install_checker(state, owner);
    Ok(())
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn install_checker(&self, state: &mut GameState, owner: Player) {
        let rank = if owner == Player::White { 2 } else { 5 };
        let cells = [0, 1, 6, 7].map(|file| Square::new(file, rank).unwrap());
        // acquireSeptember27OpeningPieces is all-or-nothing and quietly settles
        // the passive if any prescribed square is occupied/reserved.
        if !cells.iter().all(|square| {
            self.card_landing_open(state, *square)
                && !self.d4_restricts(state, owner, false, *square)
        }) {
            return;
        }
        for square in cells {
            let piece = Piece::new(owner, PieceKind::Checker).with_id(state.allocate_piece_id());
            state.board.set(square, Some(piece));
            self.register_spawned_piece(state, square, piece, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::target_procedures::*;
    use super::*;
    #[test]
    fn checker_poison_capture_stops_the_jump_chain_and_shield_contact_does_not_poison() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Checker),
                ("d4", Player::Black, PieceKind::Pawn),
                ("f6", Player::Black, PieceKind::Pawn),
            ],
            Player::White,
        );
        let pawn = before.board.get("d4".parse().unwrap()).unwrap();
        let checker = before.board.get("c3".parse().unwrap()).unwrap();
        before.card_runtime.poisoned_piece_ids.insert(pawn.id);
        let after = engine.apply(&before, movement("c3", "e5")).unwrap().state;
        assert_eq!(after.turn, Player::Black);
        assert_eq!(
            after.card_runtime.poison_stun_turns.get(&checker.id),
            Some(&2)
        );
        assert!(after.board.get("f6".parse().unwrap()).is_some());
        assert!(has_move(&engine, &after, "f6", "e5"));
        before.card_runtime.shielded_piece_ids.insert(pawn.id);
        let after = engine.apply(&before, movement("c3", "e5")).unwrap().state;
        assert!(
            !after
                .card_runtime
                .poison_stun_turns
                .contains_key(&checker.id)
        );
        assert!(after.board.find_by_id(pawn.id).is_some());
    }
    #[test]
    fn checker_acquisition_adds_four_allies_and_preserves_original_pawns_for_both_colors() {
        let engine = Engine::bundled_latest().unwrap();
        for owner in [Player::White, Player::Black] {
            let before = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
            let next = engine
                .acquire_for_setup(&before, CHECKER, owner)
                .unwrap()
                .state;
            assert_eq!(next.board.iter().count(), 36);
            assert_eq!(
                next.board
                    .iter()
                    .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Checker)
                    .count(),
                4
            );
            let rank = if owner == Player::White { 2 } else { 5 };
            for file in [0, 1, 6, 7] {
                let square = Square::new(file, rank).unwrap();
                let piece = next.board.get(square).unwrap();
                assert_eq!(piece.kind, PieceKind::Checker);
                assert_eq!(next.card_runtime.piece_origins[&piece.id], square);
                assert!(next.card_runtime.moved_piece_ids.contains(&piece.id));
            }
            assert_eq!(
                next.board
                    .iter()
                    .filter(|(_, p)| p.kind == PieceKind::Pawn)
                    .count(),
                16
            );
            assert_eq!(
                (next.ply, next.rng_state(), next.cards[0].charges),
                (0, before.rng_state(), 0)
            );
        }
    }
    #[test]
    fn checker_acquisition_is_all_or_nothing_when_a_spawn_cell_is_reserved() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
        before.card_runtime.barricades.insert("a3".parse().unwrap());
        let next = engine
            .grant_card(&before, 1, CHECKER, Player::White)
            .unwrap()
            .state;
        assert!(!next.board.contains(Player::White, PieceKind::Checker));
        assert_eq!(next.board, before.board);
        assert_eq!(next.cards[0].charges, 0);
    }
}

use super::*;
pub(super) const ID: &str = crate::FALSE_START;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    engine.install_false_start(state, owner);
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    let own_royal = has_own(&|p| royal(state, p));
    require(own_royal, "배치할 아군 왕실 기물이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn install_false_start(&self, state: &mut GameState, owner: Player) {
        fn root(parent: &[usize; 8], mut file: usize) -> usize {
            while parent[file] != file {
                file = parent[file];
            }
            file
        }
        let own: Vec<_> = state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner)
            .map(|(square, piece)| {
                let cells = state.card_runtime.large_pieces.get(&piece.id).map_or_else(
                    || vec![square],
                    |large| large_piece_occupied_cells(state, piece.id, large.anchor),
                );
                (square, piece, cells)
            })
            .collect();
        let mut parent = [0, 1, 2, 3, 4, 5, 6, 7];
        for (square, _, cells) in &own {
            for cell in cells {
                parent[root(&parent, usize::from(cell.file()))] =
                    root(&parent, usize::from(square.file()));
            }
        }
        let direction = if owner == Player::White { 1 } else { -1 };
        let mut shifts = [0_i8; 8];
        let mut blocked = [false; 8];
        for step in 1..=2 {
            let mut denied = [false; 8];
            for (square, _, cells) in &own {
                let group = root(&parent, usize::from(square.file()));
                if cells.iter().any(|cell| {
                    cell.offset(0, direction * step)
                        .is_some_and(|to| self.d4_restricts(state, owner, false, to))
                }) {
                    denied[group] = true;
                }
                if cells.iter().any(|cell| {
                    cell.offset(0, direction * step).is_none_or(|to| {
                        self.board_piece_at(state, to)
                            .is_some_and(|p| p.owner != owner)
                            || state.card_runtime.barricades.contains(&to)
                            || state.card_runtime.black_holes.contains(&to)
                            || state.card_runtime.crown_pos == Some(to)
                            || (state.card_runtime.collapsed_files_ranks
                                && (matches!(to.file(), 0 | 7) || matches!(to.rank(), 0 | 7)))
                            || state
                                .card_runtime
                                .pending_lobsters
                                .iter()
                                .any(|p| p.square == to)
                            || state
                                .card_runtime
                                .pending_scarecrows
                                .iter()
                                .any(|p| p.square == to)
                            || state
                                .card_runtime
                                .quantum_positions
                                .values()
                                .any(|p| *p == to)
                    })
                }) {
                    blocked[group] = true;
                }
            }
            for (square, _, _) in &own {
                let group = root(&parent, usize::from(square.file()));
                if !blocked[group] && !denied[group] {
                    shifts[group] = step;
                }
            }
        }
        for (square, _, _) in &own {
            state.board.set(*square, None);
        }
        for (square, piece, _) in own {
            let group = root(&parent, usize::from(square.file()));
            let to = square
                .offset(0, direction * shifts[group])
                .expect("checked false-start destination");
            state.board.set(to, Some(piece));
            if let Some(large) = state.card_runtime.large_pieces.get_mut(&piece.id) {
                large.anchor = large
                    .anchor
                    .offset(0, direction * shifts[group])
                    .expect("checked large-body shift");
            }
        }
        state.en_passant = None;
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::acquisition::*;
    use super::*;
    #[test]
    fn false_start_moves_before_first_move_and_preserves_identity_and_unmoved_flags() {
        let engine = Engine::bundled_latest().unwrap();
        for owner in [Player::White, Player::Black] {
            let before = game();
            let (from, to) = if owner == Player::White {
                ("e2", "e4")
            } else {
                ("e7", "e5")
            };
            let pawn = before.board.get(from.parse().unwrap()).unwrap();
            let next = engine
                .grant_card(&before, 1, crate::FALSE_START, owner)
                .unwrap()
                .state;
            assert_eq!(next.board.get(to.parse().unwrap()).unwrap().id, pawn.id);
            assert!(!next.card_runtime.moved_piece_ids.contains(&pawn.id));
            assert_eq!(
                next.card_runtime.piece_origins.get(&pawn.id),
                Some(&from.parse().unwrap())
            );
            assert_eq!(
                (next.ply, next.turn, next.rng_state()),
                (0, Player::White, before.rng_state())
            );
        }
    }
    #[test]
    fn false_start_links_large_body_files_and_respects_reserved_cells() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = game();
        before.board.set(
            "g5".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Knight).with_id(100)),
        );
        let mut before = engine
            .grant_card(&before, 1, BIG_BISHOP, Player::White)
            .unwrap()
            .state;
        before
            .card_runtime
            .black_holes
            .insert("b3".parse().unwrap());
        let next = engine
            .grant_card(&before, 2, crate::FALSE_START, Player::White)
            .unwrap()
            .state;
        assert_eq!(
            next.board.get("f3".parse().unwrap()).unwrap().kind,
            PieceKind::BigBishop
        );
        assert_eq!(
            next.board.get("g4".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        assert_eq!(
            next.board.get("b2".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        let large = next.board.get("f3".parse().unwrap()).unwrap();
        assert_eq!(
            next.card_runtime.large_pieces[&large.id].anchor,
            "f3".parse().unwrap()
        );
    }
}

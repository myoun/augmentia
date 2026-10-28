use super::*;
pub(super) const ID: &str = crate::DIAGONAL_CHESS;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_diagonal_chess(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn install_diagonal_chess(
        &self,
        state: &mut GameState,
        _events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let layout = [
            (Player::White, PieceKind::Pawn, "a1"),
            (Player::White, PieceKind::King, "b1"),
            (Player::White, PieceKind::Bishop, "c1"),
            (Player::White, PieceKind::Rook, "d1"),
            (Player::White, PieceKind::Pawn, "e1"),
            (Player::White, PieceKind::Queen, "a2"),
            (Player::White, PieceKind::Pawn, "b2"),
            (Player::White, PieceKind::Knight, "c2"),
            (Player::White, PieceKind::Pawn, "d2"),
            (Player::White, PieceKind::Bishop, "a3"),
            (Player::White, PieceKind::Knight, "b3"),
            (Player::White, PieceKind::Pawn, "c3"),
            (Player::White, PieceKind::Rook, "a4"),
            (Player::White, PieceKind::Pawn, "b4"),
            (Player::White, PieceKind::Pawn, "a5"),
            (Player::Black, PieceKind::Pawn, "h8"),
            (Player::Black, PieceKind::Queen, "g8"),
            (Player::Black, PieceKind::Bishop, "f8"),
            (Player::Black, PieceKind::Rook, "e8"),
            (Player::Black, PieceKind::Pawn, "d8"),
            (Player::Black, PieceKind::King, "h7"),
            (Player::Black, PieceKind::Pawn, "g7"),
            (Player::Black, PieceKind::Knight, "f7"),
            (Player::Black, PieceKind::Pawn, "e7"),
            (Player::Black, PieceKind::Bishop, "h6"),
            (Player::Black, PieceKind::Knight, "g6"),
            (Player::Black, PieceKind::Pawn, "f6"),
            (Player::Black, PieceKind::Rook, "h5"),
            (Player::Black, PieceKind::Pawn, "g5"),
            (Player::Black, PieceKind::Pawn, "h4"),
        ];
        let before = state.board.clone();
        let mut used = BTreeSet::new();
        let mut placements = Vec::new();
        for (owner, kind, square) in layout {
            let Some((_, piece)) = web_ordered_pieces(state)
                .into_iter()
                .find(|(_, p)| p.owner == owner && p.kind == kind && !used.contains(&p.id))
            else {
                return Err(EngineError::Invariant(
                    "대각선 체스의 기본 기물 구성이 필요합니다.".into(),
                ));
            };
            used.insert(piece.id);
            placements.push((square.parse::<Square>().unwrap(), piece));
        }
        state.board = Board::empty();
        for (square, piece) in placements {
            state.board.set(square, Some(piece));
            state.card_runtime.piece_origins.insert(piece.id, square);
            state.card_runtime.moved_piece_ids.remove(&piece.id);
        }
        for (_, piece) in before.iter() {
            if !used.contains(&piece.id) {
                self.clear_removed_piece_statuses(state, piece.id);
            }
        }
        // The authoritative layout deliberately contains fifteen pieces per side.
        state.en_passant = None;
        Ok(())
    }
}

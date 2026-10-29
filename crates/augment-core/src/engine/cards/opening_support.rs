use super::*;

impl Engine {
    pub(in crate::engine) fn opening_transform(
        &self,
        state: &mut GameState,
        square: Square,
        kind: PieceKind,
        moved: bool,
        fresh: bool,
    ) -> Result<(), EngineError> {
        if fresh {
            self.transform_current_piece(state, square, kind, moved)?;
        } else {
            let mut piece = state
                .board
                .get(square)
                .ok_or_else(|| EngineError::Invariant("missing opening transform target".into()))?;
            if piece.kind != kind {
                self.clear_piece_kind_runtime(state, piece.id);
                self.clear_queens_gambit_protection(state, piece.id);
            }
            piece.kind = kind;
            state.board.set(square, Some(piece));
            if moved {
                state.card_runtime.moved_piece_ids.insert(piece.id);
            }
            if kind == PieceKind::Windmill {
                state
                    .card_runtime
                    .windmill_modes
                    .insert(piece.id, WindmillMode::Bishop);
            }
        }
        Ok(())
    }
}

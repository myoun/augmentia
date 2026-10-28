use super::*;
pub(super) const ID: &str = crate::MONOCHROME_CHESS;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_monochrome_chess(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn install_monochrome_chess(
        &self,
        state: &mut GameState,
        _events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        state.card_runtime.monochrome_chess = true;
        let pieces: Vec<_> = state.board.iter().collect();
        for (square, mut piece) in pieces {
            state
                .rule_runtime
                .mono_shades
                .insert(piece.id, (square.file() + square.rank()) % 2);
            if piece.kind == PieceKind::Knight {
                piece.kind = PieceKind::Camel;
                state.board.set(square, Some(piece));
                state.card_runtime.moved_piece_ids.insert(piece.id);
            }
        }
        Ok(())
    }
}

use super::*;
pub(super) const ID: &str = crate::ELEPHANT_ESCAPE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let _ = (&engine, &events);
    let diagonal = state.rule_runtime.enabled.contains(crate::DIAGONAL_CHESS);
    let names = if diagonal {
        if owner == Player::White {
            ["a6", "f1"]
        } else {
            ["h3", "c8"]
        }
    } else if owner == Player::White {
        ["a3", "h3"]
    } else {
        ["a6", "h6"]
    };
    let squares = names.map(|s| s.parse::<Square>().unwrap());
    if squares.iter().all(|s| {
        engine.card_landing_open(state, *s) && !engine.d4_restricts(state, owner, false, *s)
    }) {
        for square in squares {
            let piece = Piece::new(owner, PieceKind::Alfil).with_id(state.allocate_piece_id());
            state.board.set(square, Some(piece));
            engine.register_spawned_piece(state, square, piece, true);
            state
                .rule_runtime
                .piece_fresh_until
                .insert(piece.id, rule_halfmove(state) + 2);
        }
    }
    Ok(())
}
pub(super) fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

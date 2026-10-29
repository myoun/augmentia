use super::*;
pub(super) const ID: &str = crate::WINDMILL;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let [a, b] = two_square_target(target)?;
    let pieces = [state.board.get(a), state.board.get(b)];
    if a == b
        || pieces.iter().any(|p| p.is_none_or(|p| p.owner != actor))
        || !pieces.iter().flatten().any(|p| p.kind == PieceKind::Bishop)
        || !pieces.iter().flatten().any(|p| p.kind == PieceKind::Rook)
    {
        return Err(EngineError::Invariant(
            "windmill requires one bishop and one rook".into(),
        ));
    }
    // The captured authority uses a selected pair and preserves moved flags in V19.
    for square in [a, b] {
        engine.transform_current_piece(state, square, PieceKind::Windmill, false)?;
    }
    Ok(())
}
pub(super) fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (bishop, _) in state
        .board
        .iter()
        .filter(|(_, p)| p.owner == card.owner && p.kind == PieceKind::Bishop)
    {
        for (rook, _) in state
            .board
            .iter()
            .filter(|(_, p)| p.owner == card.owner && p.kind == PieceKind::Rook)
        {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![bishop, rook]),
            });
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![rook, bishop]),
            });
        }
    }
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    require(
        [PieceKind::Bishop, PieceKind::Rook]
            .into_iter()
            .all(|kind| {
                state
                    .board
                    .iter()
                    .any(|(_, p)| p.owner == owner && p.kind == kind)
            }),
        "아군 비숍과 룩이 필요합니다.",
    )
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

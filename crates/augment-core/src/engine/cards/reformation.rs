use super::*;
pub(super) const ID: &str = crate::REFORMATION;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    if *target != ActionTarget::None {
        return Err(EngineError::Invariant(
            "reformation takes no selection".into(),
        ));
    }
    let bishops: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| p.owner == actor && p.kind == PieceKind::Bishop)
        .map(|(s, _)| s)
        .collect();
    if bishops.is_empty() {
        return Err(EngineError::Invariant("no bishops to transform".into()));
    }
    for square in bishops {
        engine.transform_current_piece(state, square, PieceKind::Protestant, true)?;
    }
    Ok(())
}
pub(super) fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if state
        .board
        .iter()
        .any(|(_, p)| p.owner == card.owner && p.kind == PieceKind::Bishop)
    {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
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
            .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Bishop),
        "아군 비숍이 필요합니다.",
    )
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

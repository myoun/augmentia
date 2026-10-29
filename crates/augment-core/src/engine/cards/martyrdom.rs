use super::*;
pub(super) const ID: &str = crate::MARTYRDOM;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let _ = (&engine, &events);
    let bishops: Vec<_> = state
        .board
        .iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Bishop)
        .collect();
    for (square, piece) in bishops {
        engine.delete_piece_for_card(state, square, piece, owner, ID, events);
    }
    for (_, p) in state
        .board
        .iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
    {
        state.card_runtime.shielded_piece_ids.insert(p.id);
    }
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    require(
        [PieceKind::Bishop, PieceKind::Pawn].iter().all(|k| {
            state
                .board
                .iter()
                .any(|(_, p)| p.owner == owner && p.kind == *k)
        }),
        "아군 비숍과 폰이 필요합니다.",
    )
}

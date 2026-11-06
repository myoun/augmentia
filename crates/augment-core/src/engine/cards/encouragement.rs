use super::*;
pub(super) const ID: &str = crate::ENCOURAGEMENT;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.encouragement[owner.index()] = true;
    Ok(())
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    if let Some(ineligible) = enabled(state.card_runtime.encouragement[owner.index()]) {
        return ineligible;
    }
    require(
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == owner && (royal(state, p) || p.kind == PieceKind::Merchant)),
        "아군 왕실 기물이 필요합니다.",
    )
}

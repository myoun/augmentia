use super::*;
pub(super) const ID: &str = crate::VANGUARD;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.vanguard[owner.index()] = true;
    Ok(())
}
pub(super) fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

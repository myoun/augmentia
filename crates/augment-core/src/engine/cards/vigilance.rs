use super::*;
pub(super) const ID: &str = crate::VIGILANCE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.vigilance[owner.index()] = true;
    Ok(())
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    if let Some(ineligible) = enabled(state.card_runtime.vigilance[owner.index()]) {
        return ineligible;
    }
    CardEligibility::Eligible
}

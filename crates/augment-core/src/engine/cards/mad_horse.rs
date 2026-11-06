use super::*;
pub(super) const ID: &str = crate::MAD_HORSE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.mad_horse[owner.index()] = true;
    Ok(())
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    enabled(state.card_runtime.mad_horse[owner.index()]).unwrap_or(CardEligibility::Eligible)
}

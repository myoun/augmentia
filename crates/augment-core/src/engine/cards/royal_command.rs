use super::*;
pub(super) const ID: &str = crate::ROYAL_COMMAND;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(
    ID,
    turn_effect_card::play,
    turn_effect_card::actions,
    eligibility,
);
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    turn_effect_card::eligibility(state, owner, ID)
}

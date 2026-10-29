use super::*;
pub(super) const ID: &str = crate::ROCKET;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, None, eligibility);
pub(super) fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

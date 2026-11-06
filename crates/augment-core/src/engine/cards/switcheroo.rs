use super::*;
pub(super) const ID: &str = crate::SWITCHEROO;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(
    ID,
    super::turn_effect_card::play,
    super::turn_effect_card::actions,
    eligibility,
);
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    super::turn_effect_card::eligibility(state, owner, ID)
}

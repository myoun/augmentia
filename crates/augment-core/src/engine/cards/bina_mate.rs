use super::*;
pub(super) const ID: &str = crate::BINA_MATE;
pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let rt = &state.card_runtime;
    let i = player.index();
    if let Some(result) = enabled(rt.bina_mate[i]) {
        return result;
    }
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, None, eligibility);

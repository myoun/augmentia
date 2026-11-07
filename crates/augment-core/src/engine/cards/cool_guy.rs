use super::*;
pub(super) const ID: &str = crate::COOL_GUY;
pub(super) fn install(
    _engine: &Engine,
    state: &mut GameState,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.cool_guy_pieces = BTreeSet::new();
    Ok(())
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    // Upstream `v19DraftRequirementMet` allows ALWAYS unconditionally.
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, eligibility);

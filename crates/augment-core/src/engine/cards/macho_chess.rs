use super::*;
pub(super) const ID: &str = crate::MACHO_CHESS;
pub(super) fn install(
    _engine: &Engine,
    state: &mut GameState,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.macho_chess = true;
    Ok(())
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    // Upstream `v19DraftRequirementMet` allows ANY unconditionally.
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, eligibility);

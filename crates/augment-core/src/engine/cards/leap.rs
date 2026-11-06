// PORT_STUB: leap is registered but not yet ported. Remove this file's stub body.
use super::*;
pub(super) const ID: &str = crate::LEAP;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, None, eligibility);

fn pending<T>(what: &str) -> Result<T, EngineError> {
    Err(EngineError::Invariant(format!("{what} is not ported yet")))
}

fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

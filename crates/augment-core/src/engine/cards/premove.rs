// PORT_STUB: premove is registered but not yet ported. Remove this file's stub body.
use super::*;
pub(super) const ID: &str = crate::PREMOVE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn pending<T>(what: &str) -> Result<T, EngineError> {
    Err(EngineError::Invariant(format!("{what} is not ported yet")))
}

fn play(
    _: &Engine,
    _: &mut GameState,
    _: Player,
    _: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    pending("this card")
}

fn actions(_: &Engine, _: &GameState, _: &CardInstance, _: &mut Vec<Action>) {}

fn install(_: &Engine, _: &mut GameState, _: &mut Vec<Event>) -> Result<(), EngineError> {
    pending("this rule")
}

fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

use super::transform_support::{self, TransformSource, TransformSpec};
use super::*;
pub(super) const ID: &str = crate::OFFERING;
const SPEC: TransformSpec = TransformSpec {
    source: TransformSource::Kind(PieceKind::Rook),
    kind: PieceKind::Offering,
    allow_royal: false,
};
/// Offering is one of the November expansion's on-acquire random transforms: it
/// converts a random own rook when acquired (OCTOBER27_RANDOM_TRANSFORMS, main
/// bundle 1768, resolved at 41613-41630).
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    transform_support::acquire_random_transform(
        engine,
        state,
        owner,
        SPEC,
        "acquire.offering.target",
        events,
    )?;
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    transform_support::eligibility(state, owner, SPEC)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

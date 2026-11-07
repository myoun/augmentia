use super::transform_support::{self, TransformSource, TransformSpec};
use super::*;
pub(super) const ID: &str = crate::KILLER;
const SPEC: TransformSpec = TransformSpec {
    source: TransformSource::Kind(PieceKind::Bishop),
    kind: PieceKind::Killer,
    allow_royal: false,
};
/// The 2026-10-27 balance pass turned the explicit-target piece cards into
/// passives that transform a random own piece on acquisition. Upstream keeps the
/// behavior in `OCTOBER27_RANDOM_TRANSFORMS` (main bundle 1768) and resolves it
/// at acquisition time (41613-41630).
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
        "acquire.killer.target",
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

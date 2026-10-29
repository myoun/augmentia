use super::transform_support::{self, TransformSource, TransformSpec};
use super::*;
pub(super) const ID: &str = crate::JESTER;
const SPEC: TransformSpec = TransformSpec {
    source: TransformSource::Kind(PieceKind::Queen),
    kind: PieceKind::Jester,
    allow_royal: true,
};
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, piece) = piece_target(state, target)?;
    if !SPEC.accepts(state, actor, piece) {
        return Err(EngineError::Invariant(
            "invalid piece transformation target".into(),
        ));
    }
    engine.transform_current_piece(state, square, SPEC.kind, true)?;
    Ok(())
}
pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    out: &mut Vec<Action>,
) {
    transform_support::actions(engine, state, card, SPEC, out);
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    transform_support::eligibility(state, owner, SPEC)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

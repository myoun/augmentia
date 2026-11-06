use super::*;
pub(super) const ID: &str = crate::INERTIA;
pub(super) const BEHAVIOR: CardBehavior =
    CardBehavior::active(ID, play, target_status::actions, eligibility);
fn play(
    _: &Engine,
    state: &mut GameState,
    _: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    state.card_runtime.inertia_piece_ids.insert(piece.id);
    Ok(())
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    target_status::eligibility(state, owner, ID)
}

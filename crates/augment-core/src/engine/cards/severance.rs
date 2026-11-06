use super::*;
pub(super) const ID: &str = crate::SEVERANCE;
pub(super) const BEHAVIOR: CardBehavior =
    CardBehavior::active(ID, play, target_status::actions, eligibility);
fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    state.card_runtime.severed.insert(
        piece.id,
        TimedPieceStatus {
            by: actor,
            remaining_turns: 2,
        },
    );
    Ok(())
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    target_status::eligibility(state, owner, ID)
}

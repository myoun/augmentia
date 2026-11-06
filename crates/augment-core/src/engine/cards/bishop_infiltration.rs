use super::*;
pub(super) const ID: &str = crate::BISHOP_INFILTRATION;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.bishop_infiltration_due[actor.index()] =
        Some(state.turns_taken[actor.index()].saturating_add(3));
    Ok(())
}
fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if state.card_runtime.bishop_infiltration_due[card.owner.index()].is_none() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    enabled(state.card_runtime.bishop_infiltration_due[owner.index()].is_some())
        .unwrap_or(CardEligibility::Eligible)
}

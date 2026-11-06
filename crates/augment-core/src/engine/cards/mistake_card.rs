use super::*;
pub(super) const ID: &str = crate::MISTAKE_CARD;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.mistake_card[actor.opponent().index()] = true;
    Ok(())
}
fn actions(_: &Engine, _: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    out.push(Action::UseCard {
        instance_id: card.instance_id,
        target: ActionTarget::None,
    });
}
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

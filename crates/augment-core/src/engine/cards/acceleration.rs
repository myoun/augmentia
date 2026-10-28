use super::*;
pub(super) const ID: &str = crate::ACCELERATION;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_acceleration(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn install_acceleration(
        &self,
        state: &mut GameState,
        _events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        state.card_runtime.acceleration = true;
        state.rule_runtime.acceleration_after =
            Some((state.turns_taken[1] + 2 + u32::from(state.turn == Player::Black)) * 2 + 1);
        Ok(())
    }
}

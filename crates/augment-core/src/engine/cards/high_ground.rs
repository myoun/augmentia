use super::*;
pub(super) const ID: &str = crate::HIGH_GROUND;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_high_ground(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn install_high_ground(
        &self,
        state: &mut GameState,
        _events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        for rank in 3..=4 {
            for file in 2..=5 {
                state
                    .card_runtime
                    .high_ground_squares
                    .insert(Square::new(file, rank).unwrap());
            }
        }
        Ok(())
    }
}

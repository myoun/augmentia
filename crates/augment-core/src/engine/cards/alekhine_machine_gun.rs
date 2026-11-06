use super::*;

pub(super) const ID: &str = crate::ALEKHINE_MACHINE_GUN;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let ids = engine
        .alekhine_formation(state, actor)
        .ok_or_else(|| EngineError::Invariant("missing Alekhine formation".into()))?;
    for id in ids {
        state.card_runtime.protected_piece_ids.insert(id);
        if state.card_runtime.queens_gambit_protection.contains(&id) {
            state
                .card_runtime
                .queens_gambit_previous_protection
                .insert(id);
        }
    }
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if engine.alekhine_formation(state, card.owner).is_some() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

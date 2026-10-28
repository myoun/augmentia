use super::*;
pub(super) const ID: &str = crate::SYMMETRY;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.symmetry[actor.index()] = true;
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    _state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let push_none = |actions: &mut Vec<Action>| {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        })
    };
    push_none(actions);
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let rt = &state.card_runtime;
    let i = player.index();
    if let Some(result) = enabled(rt.symmetry[i]) {
        return result;
    }
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

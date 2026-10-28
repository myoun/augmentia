use super::*;
pub(super) const ID: &str = crate::DISASSEMBLY;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.disassembly[actor.index()] = true;
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
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let id = ID;
    denied(
        EligibilityReasonCode::UnsupportedPredicate,
        id,
        "공식 획득 조건의 런타임 매핑이 아직 검증되지 않아 획득을 허용하지 않습니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

use super::*;
pub(super) const ID: &str = crate::REVERSE_PAWNS;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let opponent = actor.opponent();
    state.card_runtime.reverse_pawns_charges[opponent.index()] = 3;
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
    use PieceKind::*;
    let count = |owner, kind| {
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .count()
    };
    require(count(player.opponent(), Pawn) > 0, "상대 폰이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

use super::*;
pub(super) const ID: &str = crate::PORTAL_GUN;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let [sq1, sq2] = two_square_target(target)?;
    state.card_runtime.pending_portals.push(PendingPortal {
        owner: actor,
        squares: [sq1, sq2],
        due_turn: state.turns_taken[actor.index()].saturating_add(1),
    });
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let empties = (0..64)
        .filter_map(Square::from_index)
        .filter(|sq| state.board.get(*sq).is_none())
        .collect::<Vec<_>>();
    for i in 0..empties.len() {
        for j in (i + 1)..empties.len() {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![empties[i], empties[j]]),
            });
        }
    }
}

pub(super) fn eligibility(
    state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    require(
        empty_square_count(state) >= 2,
        "빈 칸이 2개 이상 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

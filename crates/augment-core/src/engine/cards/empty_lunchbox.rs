use super::*;
pub(super) const ID: &str = crate::EMPTY_LUNCHBOX;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let target = target.clone();
    let (_, target_piece) = piece_target(&*state, &target)?;
    state.card_runtime.empty_lunchboxes.insert(
        target_piece.id,
        EmptyLunchboxStatus {
            by: actor,
            deadline_turn: state.turns_taken[target_piece.owner.index()].saturating_add(3),
        },
    );
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    for (square, piece) in state.board.iter() {
        if piece.owner == state.turn.opponent()
            && !engine.is_royal_piece(state, piece)
            && !state.card_runtime.empty_lunchboxes.contains_key(&piece.id)
        {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    // Upstream `v19DraftRequirementMet` allows ANY unconditionally.
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

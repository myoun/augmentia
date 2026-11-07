use super::*;
pub(super) const ID: &str = crate::OUTPOST;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    _actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    state
        .card_runtime
        .outpost_protected_piece_ids
        .insert(piece.id);
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    let push_piece = |actions: &mut Vec<Action>, square| {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        })
    };
    for (square, piece) in state.board.iter() {
        let in_opponent_half = match actor {
            Player::White => square.rank() >= 4,
            Player::Black => square.rank() <= 3,
        };
        if piece.owner == actor
            && !engine.is_royal_piece(state, piece)
            && in_opponent_half
            && !state
                .card_runtime
                .outpost_protected_piece_ids
                .contains(&piece.id)
        {
            push_piece(actions, square);
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

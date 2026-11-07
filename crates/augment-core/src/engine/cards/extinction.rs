use super::*;
pub(super) const ID: &str = crate::EXTINCTION;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, selected) = piece_target(state, target)?;
    let kind = selected.kind;
    let victims = state
        .board
        .iter()
        .filter(|(_, piece)| piece.kind == kind)
        .collect::<Vec<_>>();
    for (square, piece) in victims {
        engine.delete_piece_for_card(state, square, piece, actor, EXTINCTION, events);
    }
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    for (square, piece) in state.board.iter() {
        if piece.owner == actor
            && !engine.is_royal_piece(state, piece)
            && matches!(
                piece.kind,
                PieceKind::Knight | PieceKind::Bishop | PieceKind::Camel
            )
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

use super::*;
pub(super) const ID: &str = crate::ASSASSIN;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    _actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let definition_id = ID.to_owned();
    let target = target.clone();
    let (target_square, mut target_piece) = piece_target(&*state, &target)?;
    engine.clear_queens_gambit_protection(state, target_piece.id);
    target_piece.kind = if definition_id == ASSASSIN {
        PieceKind::Assassin
    } else {
        PieceKind::Grasshopper
    };
    state.board.set(target_square, Some(target_piece));
    state.card_runtime.moved_piece_ids.insert(target_piece.id);
    engine.mark_fresh_no_capture(state, target_piece);
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    for (square, piece) in state.board.iter() {
        if piece.owner == state.turn && piece.kind == PieceKind::Knight {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
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
    let own = |kind| count(player, kind);
    require(own(Knight) > 0, "아군 나이트가 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

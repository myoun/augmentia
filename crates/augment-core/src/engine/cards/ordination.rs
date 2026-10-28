use super::*;
pub(super) const ID: &str = crate::ORDINATION;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    _actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let id = ID;
    let (square, mut piece) = piece_target(state, target)?;
    engine.clear_piece_kind_runtime(state, piece.id);
    piece.kind = if id == ORDINATION {
        PieceKind::Cardinal
    } else {
        PieceKind::Unicorn
    };
    state.board.set(square, Some(piece));
    engine.mark_transformed_piece(state, piece);
    engine.resolve_gomoku_victory(state);
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    let mut push = |target| {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target,
        })
    };
    let kind = if card.definition_id == ORDINATION {
        PieceKind::Bishop
    } else {
        PieceKind::Rook
    };
    for (square, piece) in state.board.iter() {
        if piece.owner == actor && piece.kind == kind {
            push(ActionTarget::PieceAt(square));
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
    require(own(Bishop) > 0, "아군 비숍 하나가 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

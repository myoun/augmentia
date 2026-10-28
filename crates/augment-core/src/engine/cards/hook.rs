use super::*;
pub(super) const ID: &str = crate::HOOK;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let definition_id = ID.to_owned();
    let target = target.clone();
    let [queen_square, sacrifice_square] = two_square_target(&target)?;
    let mut queen = state
        .board
        .get(queen_square)
        .ok_or_else(|| EngineError::Invariant("piece card has no selected queen".to_owned()))?;
    let sacrificed = state
        .board
        .get(sacrifice_square)
        .ok_or_else(|| EngineError::Invariant("piece card has no sacrifice".to_owned()))?;
    state.board.set(sacrifice_square, None);
    engine.clear_removed_piece_statuses(state, sacrificed.id);
    events.push(Event::PieceRemoved {
        actor: actor.opponent(),
        piece: sacrificed,
        at: sacrifice_square,
        cause: definition_id.clone(),
    });
    engine.resolve_reaper_deaths(
        state,
        &[(sacrifice_square, sacrificed)],
        actor.opponent(),
        events,
    );
    engine.clear_queens_gambit_protection(state, queen.id);
    queen.kind = if definition_id == AMAZON {
        PieceKind::Amazon
    } else {
        PieceKind::Hook
    };
    state.board.set(queen_square, Some(queen));
    state.card_runtime.moved_piece_ids.insert(queen.id);
    engine.mark_fresh_no_capture(state, queen);
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let sacrifice_kind = if card.definition_id == AMAZON {
        PieceKind::Knight
    } else {
        PieceKind::Rook
    };
    let queens = state
        .board
        .iter()
        .filter(|(_, piece)| piece.owner == state.turn && piece.kind == PieceKind::Queen)
        .collect::<Vec<_>>();
    let sacrifices = state
        .board
        .iter()
        .filter(|(_, piece)| piece.owner == state.turn && piece.kind == sacrifice_kind)
        .collect::<Vec<_>>();
    for (queen_square, _) in &queens {
        for (sacrifice_square, _) in &sacrifices {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![*queen_square, *sacrifice_square]),
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
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    let own_queen = has_own(&|p| queen(state, p));
    require(own_queen && own(Rook) > 0, "아군 퀸과 룩이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

use super::*;
pub(super) const ID: &str = crate::ARCHBISHOP;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let id = ID;
    let [knight_square, base_square] = two_square_target(target)?;
    let knight = state
        .board
        .get(knight_square)
        .ok_or_else(|| EngineError::Invariant("missing compound sacrifice".into()))?;
    let mut base = state
        .board
        .get(base_square)
        .ok_or_else(|| EngineError::Invariant("missing compound base".into()))?;
    // Removal does not capture or trigger resurrection/capture reactions.
    engine.delete_piece_for_card(state, knight_square, knight, actor, id, events);
    base.kind = if id == crate::ARCHBISHOP {
        PieceKind::Archbishop
    } else {
        PieceKind::Chancellor
    };
    state.board.set(base_square, Some(base));
    // applyAmazonV7 marks identity moved, retaining statuses and capture count.
    // It does not spawn a new identity or impose a fresh-capture delay.
    state.card_runtime.moved_piece_ids.insert(base.id);
    engine.resolve_gomoku_victory(state);
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let eligible = |piece: Piece, kind| {
        piece.owner == state.turn && piece.kind == kind && !engine.is_royal_piece(state, piece)
    };
    for (knight_square, _) in state
        .board
        .iter()
        .filter(|(_, p)| eligible(*p, PieceKind::Knight))
    {
        for (base_square, _) in state
            .board
            .iter()
            .filter(|(_, p)| eligible(*p, PieceKind::Bishop))
        {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![knight_square, base_square]),
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
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    let id = ID;
    let base = if id == crate::ARCHBISHOP {
        Bishop
    } else {
        Rook
    };
    require(
        has_own(&|p| p.kind == Knight && !royal(state, p))
            && has_own(&|p| p.kind == base && !royal(state, p)),
        "비왕실 아군 나이트와 변환할 비숍/룩이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

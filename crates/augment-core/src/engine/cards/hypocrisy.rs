use super::*;
pub(super) const ID: &str = crate::HYPOCRISY;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let opponent = actor.opponent();
    let ActionTarget::Squares(squares) = target else {
        return Err(EngineError::Invariant(
            "hypocrisy requires four squares".to_owned(),
        ));
    };
    for &square in squares {
        let pawn = Piece::new(opponent, PieceKind::Pawn).with_id(state.allocate_piece_id());
        state.board.set(square, Some(pawn));
        engine.register_spawned_piece(state, square, pawn, true);
    }
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let squares = (0_u8..64)
        .filter_map(Square::from_index)
        .filter(|square| {
            state.board.get(*square).is_none() && !state.card_runtime.black_holes.contains(square)
        })
        .take(4)
        .collect::<Vec<_>>();
    if squares.len() == 4 {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(squares),
        });
    }
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

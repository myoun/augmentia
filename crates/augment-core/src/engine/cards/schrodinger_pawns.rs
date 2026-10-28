use super::*;
pub(super) const ID: &str = crate::SCHRODINGER_PAWNS;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let mut candidates = web_ordered_pieces(state)
        .into_iter()
        .filter_map(|(square, piece)| {
            let direction = if actor == Player::White { -1 } else { 1 };
            square
                .offset(0, direction)
                .filter(|behind| state.board.get(*behind).is_none())
                .filter(|_| {
                    piece.owner == actor
                        && piece.kind == PieceKind::Pawn
                        && !state.card_runtime.quantum_positions.contains_key(&piece.id)
                })
                .map(|behind| (piece, behind))
        })
        .collect::<Vec<_>>();
    for _ in 0..candidates.len().min(4) {
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        events.push(Event::RandomSample {
            purpose: "schrodinger-pawns".to_owned(),
            roll_53: roll,
        });
        let (piece, behind) = candidates.remove(index.expect("non-empty candidates"));
        state
            .card_runtime
            .quantum_positions
            .insert(piece.id, behind);
        engine.mark_fresh_no_capture(state, piece);
    }
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    let push_none = |actions: &mut Vec<Action>| {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        })
    };
    if state.board.iter().any(|(square, piece)| {
        let direction = if actor == Player::White { -1 } else { 1 };
        piece.owner == actor
            && piece.kind == PieceKind::Pawn
            && !state.card_runtime.quantum_positions.contains_key(&piece.id)
            && square
                .offset(0, direction)
                .is_some_and(|behind| state.board.get(behind).is_none())
    }) {
        push_none(actions);
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

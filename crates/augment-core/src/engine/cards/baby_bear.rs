use super::*;
pub(super) const ID: &str = crate::BABY_BEAR;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, queen) = piece_target(state, target)?;
    engine.remove_piece_for_card(state, square, queen, actor, BABY_BEAR, events);
    let baby_bear = Piece::new(actor, PieceKind::BabyBear).with_id(state.allocate_piece_id());
    state.board.set(square, Some(baby_bear));
    state
        .card_runtime
        .baby_bear_grow_turns
        .insert(baby_bear.id, state.shared_turn_count().saturating_add(7));
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("baby-bear-spawn:{}:{}", baby_bear.id, square),
    });
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    for (square, piece) in state.board.iter() {
        if piece.owner == actor && piece.kind == PieceKind::Queen {
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
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    require(
        has_own(&|p| queen(state, p) && !royal(state, p)),
        "비왕실 아군 퀸이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

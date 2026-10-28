use super::*;
pub(super) const ID: &str = crate::REPLAY;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let opponent = actor.opponent();
    let replay = state.card_runtime.move_replays[actor.index()]
        .clone()
        .ok_or_else(|| EngineError::Invariant("replay has no recorded move".to_owned()))?;
    if !replay_move_matches_state(state, &replay) {
        return Err(EngineError::Invariant(
            "replay cells no longer match the recorded move".to_owned(),
        ));
    }
    for cell in &replay.cells {
        state.board.set(cell.square, cell.before);
    }
    if state.en_passant == replay.en_passant_after {
        state.en_passant = replay.en_passant_before;
    }
    let mut current_runtime = state.card_runtime.clone();
    current_runtime.move_replays = [None, None];
    if current_runtime == *replay.runtime_after {
        let other_replay = state.card_runtime.move_replays[opponent.index()].clone();
        state.card_runtime = (*replay.runtime_before).clone();
        // Playing Replay only consumes the current player's stored move. The other
        // player's independent capture remains available, as in normalizeColorValues.
        state.card_runtime.move_replays[opponent.index()] = other_replay;
    } else {
        restore_replay_piece_runtime(&mut state.card_runtime, &replay);
        restore_replay_color_runtime(&mut state.card_runtime, &replay, actor);
    }
    state.card_runtime.move_replays[actor.index()] = None;
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
    if state.card_runtime.move_replays[actor.index()]
        .as_ref()
        .is_some_and(|replay| replay_move_matches_state(state, replay))
    {
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

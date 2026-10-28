use super::*;
pub(super) const ID: &str = crate::HORSE_RIDING;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.horse_riding[actor.index()] = true;
    engine.resolve_gomoku_victory(state);
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
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
    if !state.card_runtime.horse_riding[actor.index()]
        && state
            .board
            .iter()
            .any(|(_, p)| p.owner == actor && engine.is_royal_piece(state, p))
    {
        push(ActionTarget::None);
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
    let own_royal = has_own(&|p| royal(state, p));
    require(own_royal, "효과를 받을 아군 왕실 기물이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::target_procedures::*;
    use super::*;
    #[test]
    fn horse_riding_is_temporary_active_and_keeps_king_identity_and_knights() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(
            &[
                ("d4", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Knight),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        let id = before.board.get("d4".parse().unwrap()).unwrap().id;
        let mut state = engine
            .grant_card(&before, 1, HORSE_RIDING, Player::White)
            .unwrap()
            .state;
        assert!(!state.card_runtime.horse_riding[0]);
        assert!(!has_move(&engine, &state, "d4", "f5"));
        state = engine.apply(&state, use_none()).unwrap().state;
        assert_eq!(
            state.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::King
        );
        assert!(state.board.contains(Player::White, PieceKind::Knight));
        assert!(has_move(&engine, &state, "d4", "f5"));
        assert!(has_move(&engine, &state, "d4", "e4"));
        state.actions_remaining = 2;
        state = engine.apply(&state, movement("d4", "f5")).unwrap().state;
        assert_eq!(state.turn, Player::White);
        assert_eq!(state.board.get("f5".parse().unwrap()).unwrap().id, id);
        assert!(state.card_runtime.horse_riding[0]);
        state = engine.apply(&state, movement("a2", "b4")).unwrap().state;
        assert_eq!(state.turn, Player::Black);
        assert!(!state.card_runtime.horse_riding[0]);
        state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        assert!(!has_move(&engine, &state, "f5", "h6"));
    }
}

use super::*;
pub(super) const ID: &str = crate::REVERSAL;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let id = ID;
    let (square, piece) = piece_target(state, target)?;
    // rememberCapture:false: a sacrifice does not count as a capture.
    engine.delete_piece_for_card(state, square, piece, actor, id, events);
    state.card_runtime.reversal[actor.index()] = true;
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
    if !state.card_runtime.reversal[actor.index()] {
        for (square, piece) in state.board.iter() {
            if piece.owner == actor
                && crate::latest::latest_minor_kind(piece.kind)
                && !engine.is_royal_piece(state, piece)
            {
                push(ActionTarget::PieceAt(square));
            }
        }
    }
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let rt = &state.card_runtime;
    let i = player.index();
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    require(
        !rt.reversal[i]
            && has_own(&|p| crate::latest::latest_minor_kind(p.kind) && !royal(state, p)),
        "희생할 비왕실 아군 마이너 기물이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::target_procedures::*;
    use super::*;
    #[test]
    fn reversal_sacrifices_one_native_minor_and_expires_after_own_turn() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(
            &[
                ("h1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("a2", Player::White, PieceKind::Knight),
                ("d4", Player::White, PieceKind::Rook),
                ("f1", Player::White, PieceKind::Bishop),
            ],
            Player::White,
        );
        let mut state = engine
            .acquire_for_setup(&before, crate::REVERSAL, Player::White)
            .unwrap()
            .state;
        assert!(!engine.legal_actions(&state).unwrap().contains(&use_none()));
        let knight = state.board.get("a2".parse().unwrap()).unwrap();
        state.card_runtime.recurrence_piece_ids.insert(knight.id);
        let used = engine.apply(&state, use_piece("a2")).unwrap();
        assert!(used.state.board.get("a2".parse().unwrap()).is_none());
        assert!(used.state.card_runtime.pending_recurrences.is_empty());
        assert!(
            !used
                .events
                .iter()
                .any(|e| matches!(e, Event::PieceCaptured { .. }))
        );
        assert_eq!(
            (used.state.turn, used.state.ply, used.state.rng_state()),
            (Player::White, 0, state.rng_state())
        );
        assert!(has_move(&engine, &used.state, "d4", "e5"));
        assert!(!has_move(&engine, &used.state, "d4", "d5"));
        assert!(has_move(&engine, &used.state, "f1", "f3"));
        assert!(!has_move(&engine, &used.state, "f1", "g2"));
        let mut after = engine
            .apply(&used.state, movement("d4", "e5"))
            .unwrap()
            .state;
        assert!(!after.card_runtime.reversal[0]);
        after = engine.apply(&after, movement("h8", "h7")).unwrap().state;
        assert!(has_move(&engine, &after, "e5", "e6"));
    }
    #[test]
    fn reversal_rejects_major_royal_enemy_and_empty_targets_transactionally() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Wizard),
                ("d4", Player::White, PieceKind::Rook),
                ("f4", Player::Black, PieceKind::Bishop),
            ],
            Player::White,
        );
        state = engine
            .grant_card(&state, 1, crate::REVERSAL, Player::White)
            .unwrap()
            .state;
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("c3"))
        );
        let wizard = state.board.get("c3".parse().unwrap()).unwrap();
        state.card_runtime.royal_piece_ids.insert(wizard.id);
        let unchanged = state.clone();
        for target in ["a1", "c3", "d4", "f4", "b2"] {
            assert!(engine.apply(&state, use_piece(target)).is_err());
        }
        assert_eq!(state, unchanged);
    }
}

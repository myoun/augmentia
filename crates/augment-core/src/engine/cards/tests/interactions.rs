use super::*;

pub(in crate::engine::cards) fn position(
    pieces: &[(&str, Player, PieceKind)],
    turn: Player,
) -> GameState {
    let mut board = crate::Board::empty();
    for (square, owner, kind) in pieces {
        board.set(square.parse().unwrap(), Some(Piece::new(*owner, *kind)));
    }
    let mut state = GameState::from_board(crate::LATEST_RULESET_ID, Mode::Standard, board, turn, 1);
    state.completed_drafts = [true; 3];
    state
}
pub(in crate::engine::cards) fn base() -> GameState {
    position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
        ],
        Player::White,
    )
}
pub(in crate::engine::cards) fn movement(from: &str, to: &str) -> Action {
    Action::Move(MoveAction::new(
        from.parse().unwrap(),
        to.parse().unwrap(),
        None,
    ))
}
pub(in crate::engine::cards) fn card(target: &str) -> Action {
    Action::UseCard {
        instance_id: 1,
        target: ActionTarget::PieceAt(target.parse().unwrap()),
    }
}
pub(in crate::engine::cards) fn has_move(
    engine: &Engine,
    state: &GameState,
    from: &str,
    to: &str,
) -> bool {
    engine
        .legal_actions(state)
        .unwrap()
        .contains(&movement(from, to))
}

#[test]
pub(in crate::engine::cards) fn six_restored_transform_contracts_keep_identity_owner_turn_and_reset_recruiter_captures()
 {
    let engine = Engine::bundled_latest().unwrap();
    for (id, before_kind, after_kind) in [
        (ASSASSIN, PieceKind::Knight, PieceKind::Assassin),
        (DRAGON, PieceKind::Rook, PieceKind::Dragon),
        (KNIGHTMASTER, PieceKind::Knight, PieceKind::Knightmaster),
        (LOG, PieceKind::Pawn, PieceKind::Log),
        (STANDARD_BEARER, PieceKind::Pawn, PieceKind::StandardBearer),
        (LOCAL_CONSCRIPTION, PieceKind::Queen, PieceKind::Recruiter),
    ] {
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, before_kind),
                ("f4", Player::Black, before_kind),
            ],
            Player::White,
        );
        let mut piece = state.board.get("d4".parse().unwrap()).unwrap();
        piece.capture_count = 3;
        state.board.set("d4".parse().unwrap(), Some(piece));
        let state = engine
            .acquire_for_setup(&state, id, Player::White)
            .unwrap()
            .state;
        let actions = engine.legal_actions(&state).unwrap();
        assert!(actions.contains(&card("d4")), "{id}");
        assert!(!actions.contains(&card("f4")), "enemy target: {id}");
        assert!(!actions.contains(&card("a1")), "wrong kind: {id}");
        let next = engine.apply(&state, card("d4")).unwrap();
        let transformed = next.state.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(
            (transformed.id, transformed.owner, transformed.kind),
            (piece.id, Player::White, after_kind)
        );
        assert_eq!(
            transformed.capture_count,
            if id == LOCAL_CONSCRIPTION { 0 } else { 3 }
        );
        assert_eq!(next.state.turn, Player::White);
        assert_eq!(next.state.ply, 0);
        assert_eq!(next.state.cards[0].charges, 0);
        assert!(
            !next
                .events
                .iter()
                .any(|event| matches!(event, Event::RandomSample { .. }))
        );
    }
}

#[test]
pub(in crate::engine::cards) fn effect_runtime_changes_search_and_repetition_identity() {
    let state = base();
    let mut changed = state.clone();
    changed.card_runtime.effects.assembly[0] = true;
    assert_ne!(state.position_key(), changed.position_key());
    assert_ne!(state.fingerprint(), changed.fingerprint());
}

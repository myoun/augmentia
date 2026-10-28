use super::*;

pub(in crate::engine::cards) fn flag_test_state() -> GameState {
    let mut board = crate::Board::empty();
    for (square, owner, kind) in [
        ("h1", Player::White, PieceKind::King),
        ("b1", Player::White, PieceKind::Rook),
        ("a2", Player::Black, PieceKind::Rook),
        ("h8", Player::Black, PieceKind::King),
    ] {
        board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
    }
    let mut state = GameState::from_board(
        crate::BUNDLED_RULESET_ID,
        Mode::Standard,
        board,
        Player::Black,
        915,
    );
    state.card_runtime.capture_the_flag = Some(crate::CaptureTheFlagState {
        flags: ["a1".parse().unwrap(), "a8".parse().unwrap()],
        occupations: [None, None],
    });
    state
}

pub(in crate::engine::cards) fn flag_move(from: &str, to: &str) -> Action {
    Action::Move(MoveAction::new(
        from.parse().unwrap(),
        to.parse().unwrap(),
        None,
    ))
}

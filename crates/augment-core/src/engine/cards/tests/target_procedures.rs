use super::*;

pub(in crate::engine::cards) fn position(
    pieces: &[(&str, Player, PieceKind)],
    turn: Player,
) -> GameState {
    let mut board = crate::Board::empty();
    for (square, owner, kind) in pieces {
        board.set(square.parse().unwrap(), Some(Piece::new(*owner, *kind)));
    }
    let mut state = GameState::from_board(crate::LATEST_RULESET_ID, Mode::Standard, board, turn, 7);
    state.completed_drafts = [true; 3];
    state
}
pub(in crate::engine::cards) fn movement(from: &str, to: &str) -> Action {
    Action::Move(MoveAction::new(
        from.parse().unwrap(),
        to.parse().unwrap(),
        None,
    ))
}
pub(in crate::engine::cards) fn use_none() -> Action {
    Action::UseCard {
        instance_id: 1,
        target: ActionTarget::None,
    }
}
pub(in crate::engine::cards) fn use_piece(square: &str) -> Action {
    Action::UseCard {
        instance_id: 1,
        target: ActionTarget::PieceAt(square.parse().unwrap()),
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
pub(in crate::engine::cards) fn has_moves_for(
    engine: &Engine,
    state: &GameState,
    source: &str,
) -> bool {
    let source: Square = source.parse().unwrap();
    engine
        .legal_actions(state)
        .unwrap()
        .iter()
        .any(|a| matches!(a, Action::Move(m) if m.from == source))
}

#[test]
pub(in crate::engine::cards) fn poisoned_pawn_samples_at_most_three_distinct_unpoisoned_pawns_with_exact_draw_count()
 {
    let engine = Engine::bundled_latest().unwrap();
    for count in 1..=5 {
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        for file in 0..count {
            state.board.set(
                Square::new(file, 1).unwrap(),
                Some(Piece::new(Player::White, PieceKind::Pawn).with_id(100 + u32::from(file))),
            );
        }
        state = engine
            .grant_card(&state, 1, crate::POISONED_PAWN, Player::White)
            .unwrap()
            .state;
        let used = engine.apply(&state, use_none()).unwrap();
        assert_eq!(
            used.state.card_runtime.poisoned_piece_ids.len(),
            usize::from(count.min(3))
        );
        assert_eq!(
            used.events
                .iter()
                .filter(|e| matches!(e, Event::RandomSample { .. }))
                .count(),
            usize::from(count.min(3))
        );
        assert_eq!(used, engine.apply(&state, use_none()).unwrap());
        assert!(engine.apply(&state, use_piece("a2")).is_err());
    }
    let mut state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
    state.completed_drafts = [true; 3];
    let prior = state.board.get("a2".parse().unwrap()).unwrap().id;
    state.card_runtime.poisoned_piece_ids.insert(prior);
    state = engine
        .grant_card(&state, 1, crate::POISONED_PAWN, Player::White)
        .unwrap()
        .state;
    let used = engine.apply(&state, use_none()).unwrap().state;
    assert_eq!(used.card_runtime.poisoned_piece_ids.len(), 4);
    assert!(used.card_runtime.poisoned_piece_ids.contains(&prior));
}

#[test]
pub(in crate::engine::cards) fn friendly_poison_capture_does_not_stun_and_poison_blocks_piece_abilities()
 {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("c3", Player::White, PieceKind::Knight),
            ("d5", Player::White, PieceKind::Pawn),
        ],
        Player::White,
    );
    let knight = state.board.get("c3".parse().unwrap()).unwrap();
    let pawn = state.board.get("d5".parse().unwrap()).unwrap();
    state.card_runtime.mad_horse[0] = true;
    state.card_runtime.poisoned_piece_ids.insert(pawn.id);
    let next = engine.apply(&state, movement("c3", "d5")).unwrap().state;
    assert!(!next.card_runtime.poison_stun_turns.contains_key(&knight.id));
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("c3", Player::White, PieceKind::Wizard),
        ],
        Player::White,
    );
    let wizard = state.board.get("c3".parse().unwrap()).unwrap();
    state.card_runtime.wizard_mana.insert(wizard.id, 6);
    assert!(
        engine.legal_actions(&state).unwrap().iter().any(
            |a| matches!(a, Action::UsePieceAbility { piece_id, .. } if *piece_id == wizard.id)
        )
    );
    state.card_runtime.poison_stun_turns.insert(wizard.id, 2);
    assert!(
        !engine.legal_actions(&state).unwrap().iter().any(
            |a| matches!(a, Action::UsePieceAbility { piece_id, .. } if *piece_id == wizard.id)
        )
    );
}

#[test]
pub(in crate::engine::cards) fn ordination_needs_one_bishop_without_sacrifice_and_pegasus_targets_a_rook()
 {
    let engine = Engine::bundled_latest().unwrap();
    for (id, source, target) in [
        (ORDINATION, PieceKind::Bishop, PieceKind::Cardinal),
        (PEGASUS, PieceKind::Rook, PieceKind::Unicorn),
    ] {
        let before = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, source),
                ("b1", Player::White, PieceKind::Knight),
                ("f4", Player::Black, source),
            ],
            Player::White,
        );
        let state = engine
            .acquire_for_setup(&before, id, Player::White)
            .unwrap()
            .state;
        let id_before = state.board.get("d4".parse().unwrap()).unwrap().id;
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("d4"))
        );
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("f4"))
        );
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("b1"))
        );
        let next = engine.apply(&state, use_piece("d4")).unwrap();
        let piece = next.state.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(
            (piece.id, piece.owner, piece.kind),
            (id_before, Player::White, target)
        );
        assert_eq!(next.state.board.iter().count(), before.board.iter().count());
        assert!(
            !next
                .events
                .iter()
                .any(|e| matches!(e, Event::PieceRemoved { .. } | Event::PieceCaptured { .. }))
        );
    }
}

use super::target_procedures::{has_move, movement, position, use_none, use_piece};
use super::*;

#[test]
fn twins_can_bind_a_royal_and_swap_ids_after_a_move() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Rook),
            ("d2", Player::White, PieceKind::Rook),
        ],
        Player::White,
    );
    let king = state.board.get("a1".parse().unwrap()).unwrap();
    let rook = state.board.get("d4".parse().unwrap()).unwrap();
    let state = engine
        .grant_card(&state, 1, crate::TWINS, Player::White)
        .unwrap()
        .state;
    let choice = Action::UseCard {
        instance_id: 1,
        target: ActionTarget::Squares(vec!["d4".parse().unwrap(), "a1".parse().unwrap()]),
    };
    let state = engine.apply(&state, choice).unwrap().state;
    let after = engine.apply(&state, movement("d4", "d5")).unwrap().state;
    assert_eq!(after.board.get("a1".parse().unwrap()).unwrap().id, rook.id);
    assert_eq!(after.board.get("d5".parse().unwrap()).unwrap().id, king.id);
    assert!(after.card_runtime.moved_piece_ids.contains(&king.id));
}

#[test]
fn twins_delay_swap_until_file_surge_chain_finishes() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("c3", Player::White, PieceKind::Knight),
            ("d2", Player::White, PieceKind::Knight),
        ],
        Player::White,
    );
    let first = state.board.get("c3".parse().unwrap()).unwrap();
    let second = state.board.get("d2".parse().unwrap()).unwrap();
    state.card_runtime.file_surge[Player::White.index()] = true;
    state = engine
        .grant_card(&state, 1, crate::TWINS, Player::White)
        .unwrap()
        .state;
    let choice = Action::UseCard {
        instance_id: 1,
        target: ActionTarget::Squares(vec!["c3".parse().unwrap(), "d2".parse().unwrap()]),
    };
    state = engine.apply(&state, choice).unwrap().state;
    state = engine.apply(&state, movement("c3", "a4")).unwrap().state;
    assert!(matches!(state.phase, GamePhase::ForcedMove { piece_id, .. } if piece_id == first.id));
    assert_eq!(state.board.get("a4".parse().unwrap()).unwrap().id, first.id);
    assert_eq!(
        state.board.get("d2".parse().unwrap()).unwrap().id,
        second.id
    );
    state = engine.apply(&state, movement("a4", "b6")).unwrap().state;
    assert_eq!(state.board.get("d2".parse().unwrap()).unwrap().id, first.id);
    assert_eq!(
        state.board.get("b6".parse().unwrap()).unwrap().id,
        second.id
    );
    assert!(state.card_runtime.twin_swap_pending.is_empty());
}

#[test]
fn e4_teleports_to_color_specific_square_and_expires_after_turn() {
    let engine = Engine::bundled_latest().unwrap();
    for (owner, source, target) in [(Player::White, "a2", "e4"), (Player::Black, "h7", "e5")] {
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                (source, owner, PieceKind::Rook),
                (target, owner.opponent(), PieceKind::Knight),
            ],
            owner,
        );
        let rook = state.board.get(source.parse().unwrap()).unwrap();
        assert!(!has_move(&engine, &state, source, target));
        state = engine
            .grant_card(&state, 1, crate::E4, owner)
            .unwrap()
            .state;
        state = engine.apply(&state, use_none()).unwrap().state;
        assert!(has_move(&engine, &state, source, target));
        state = engine
            .apply(&state, movement(source, target))
            .unwrap()
            .state;
        assert_eq!(
            state.board.get(target.parse().unwrap()).unwrap().id,
            rook.id
        );
        assert_eq!(state.turn, owner.opponent());
        assert!(!state.card_runtime.e4_active[owner.index()]);
    }
}

#[test]
fn e4_gives_wizard_a_capture_but_does_not_allow_slime_teleport() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b2", Player::White, PieceKind::Wizard),
            ("c2", Player::White, PieceKind::Slime),
            ("e4", Player::Black, PieceKind::Knight),
        ],
        Player::White,
    );
    state = engine
        .grant_card(&state, 1, crate::E4, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, use_none()).unwrap().state;
    assert!(has_move(&engine, &state, "b2", "e4"));
    assert!(!has_move(&engine, &state, "c2", "e4"));
    let after = engine.apply(&state, movement("b2", "e4")).unwrap().state;
    assert_eq!(
        after.board.get("e4".parse().unwrap()).unwrap().kind,
        PieceKind::Wizard
    );
}

#[test]
fn mad_horse_rejects_protected_allies_and_chains_friendly_captures() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("a8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Knight),
            ("f5", Player::White, PieceKind::Pawn),
            ("h6", Player::White, PieceKind::Pawn),
        ],
        Player::White,
    );
    let pawn = state.board.get("f5".parse().unwrap()).unwrap();
    let knight = state.board.get("d4".parse().unwrap()).unwrap();
    state.card_runtime.shielded_piece_ids.insert(pawn.id);
    state = engine
        .grant_card(&state, 1, crate::MAD_HORSE, Player::White)
        .unwrap()
        .state;
    assert!(!has_move(&engine, &state, "d4", "f5"));
    state.card_runtime.shielded_piece_ids.remove(&pawn.id);
    state = engine.apply(&state, movement("d4", "f5")).unwrap().state;
    assert!(matches!(state.phase, GamePhase::ForcedMove { piece_id, .. } if piece_id == knight.id));
    state = engine.apply(&state, movement("f5", "h6")).unwrap().state;
    assert!(matches!(state.phase, GamePhase::ForcedMove { piece_id, .. } if piece_id == knight.id));
    state = engine.apply(&state, movement("h6", "g8")).unwrap().state;
    assert_eq!(state.turn, Player::Black);
    assert_eq!(
        state.board.get("g8".parse().unwrap()).unwrap().id,
        knight.id
    );
}

#[test]
fn reposition_blocks_all_captures_then_forces_the_first_mover_to_move_again() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Rook),
            ("d6", Player::Black, PieceKind::Pawn),
            ("c4", Player::White, PieceKind::Bishop),
            ("f7", Player::Black, PieceKind::Knight),
        ],
        Player::White,
    );
    let rook = state.board.get("d4".parse().unwrap()).unwrap();
    state = engine
        .grant_card(&state, 1, crate::REPOSITION, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, use_none()).unwrap().state;
    assert!(!has_move(&engine, &state, "d4", "d6"));
    assert!(!has_move(&engine, &state, "c4", "f7"));
    state = engine.apply(&state, movement("d4", "d5")).unwrap().state;
    assert!(matches!(state.phase, GamePhase::ForcedMove { piece_id, .. } if piece_id == rook.id));
    assert!(!has_move(&engine, &state, "d5", "d6"));
    assert!(!has_move(&engine, &state, "a1", "b1"));
    state = engine.apply(&state, movement("d5", "e5")).unwrap().state;
    assert_eq!(state.turn, Player::Black);
    assert!(state.card_runtime.reposition.is_none());
}

#[test]
fn reposition_finishes_turn_if_the_first_piece_has_no_legal_followup() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b2", Player::White, PieceKind::Pawn),
            ("b4", Player::White, PieceKind::Pawn),
            ("c4", Player::Black, PieceKind::Pawn),
        ],
        Player::White,
    );
    let state = engine
        .grant_card(&state, 1, crate::REPOSITION, Player::White)
        .unwrap()
        .state;
    let state = engine.apply(&state, use_none()).unwrap().state;
    let after = engine.apply(&state, movement("b2", "b3")).unwrap().state;
    assert_eq!(after.turn, Player::Black);
    assert!(after.card_runtime.reposition.is_none());
}

#[test]
fn sacrifice_uses_current_values_and_preserves_existing_shield() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b2", Player::White, PieceKind::Knight),
            ("c2", Player::White, PieceKind::Camel),
            ("c8", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    let camel = state.board.get("c2".parse().unwrap()).unwrap();
    state.card_runtime.shielded_piece_ids.insert(camel.id);
    state = engine
        .grant_card(&state, 1, crate::SACRIFICE, Player::White)
        .unwrap()
        .state;
    let choice = |a: &str, b: &str| Action::UseCard {
        instance_id: 1,
        target: ActionTarget::Squares(vec![a.parse().unwrap(), b.parse().unwrap()]),
    };
    let actions = engine.legal_actions(&state).unwrap();
    assert!(actions.contains(&choice("b2", "c2")));
    assert!(!actions.contains(&choice("c2", "b2")));
    state = engine.apply(&state, choice("b2", "c2")).unwrap().state;
    assert!(state.board.get("b2".parse().unwrap()).is_none());
    state = engine.apply(&state, movement("a1", "b1")).unwrap().state;
    state = engine.apply(&state, movement("c8", "c2")).unwrap().state;
    assert!(state.card_runtime.shielded_piece_ids.contains(&camel.id));
    assert!(
        state
            .card_runtime
            .sacrifice_protection
            .contains_key(&camel.id)
    );
    state = engine.apply(&state, movement("b1", "a1")).unwrap().state;
    state = engine.apply(&state, movement("h8", "g8")).unwrap().state;
    state = engine.apply(&state, movement("a1", "b1")).unwrap().state;
    assert!(
        !state
            .card_runtime
            .sacrifice_protection
            .contains_key(&camel.id)
    );
    assert!(state.card_runtime.shielded_piece_ids.contains(&camel.id));
}

#[test]
fn scarecrow_sacrifices_piece_then_becomes_capturable_after_three_owner_starts() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Pawn),
            ("d8", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    let pawn = state.board.get("d4".parse().unwrap()).unwrap();
    state = engine
        .grant_card(&state, 1, crate::SCARECROW, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, use_piece("d4")).unwrap().state;
    let scarecrow = state.board.get("d4".parse().unwrap()).unwrap();
    assert_ne!(scarecrow.id, pawn.id);
    assert_eq!(scarecrow.kind, PieceKind::Scarecrow);
    assert!(
        state
            .card_runtime
            .scarecrow_reservations
            .contains_key(&scarecrow.id)
    );
    state = engine.apply(&state, movement("a1", "b1")).unwrap().state;
    state = engine.apply(&state, movement("d8", "d4")).unwrap().state;
    assert_eq!(
        state.board.get("d4".parse().unwrap()).unwrap().id,
        scarecrow.id
    );
    state = engine.apply(&state, movement("b1", "a1")).unwrap().state;
    state = engine.apply(&state, movement("h8", "g8")).unwrap().state;
    state = engine.apply(&state, movement("a1", "b1")).unwrap().state;
    state = engine.apply(&state, movement("g8", "h8")).unwrap().state;
    assert!(
        !state
            .card_runtime
            .scarecrow_reservations
            .contains_key(&scarecrow.id)
    );
    assert!(
        !state
            .card_runtime
            .protected_piece_ids
            .contains(&scarecrow.id)
    );
    state = engine.apply(&state, movement("b1", "a1")).unwrap().state;
    assert!(has_move(&engine, &state, "d8", "d4"));
    state = engine.apply(&state, movement("d8", "d4")).unwrap().state;
    assert_eq!(
        state.board.get("d4".parse().unwrap()).unwrap().owner,
        Player::Black
    );
}

#[test]
fn free_castling_selects_aligned_rook_and_crushes_both_destinations() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("b1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b5", Player::White, PieceKind::Rook),
            ("b2", Player::White, PieceKind::Knight),
            ("b3", Player::Black, PieceKind::Pawn),
            ("f5", Player::White, PieceKind::Rook),
        ],
        Player::White,
    );
    let king = state.board.get("b1".parse().unwrap()).unwrap();
    let rook = state.board.get("b5".parse().unwrap()).unwrap();
    let victim = state.board.get("b3".parse().unwrap()).unwrap();
    state.card_runtime.protected_piece_ids.insert(victim.id);
    state.card_runtime.imperial_studies[Player::White.index()] = true;
    state = engine
        .grant_card(&state, 1, crate::CASTLING, Player::White)
        .unwrap()
        .state;
    let actions = engine.legal_actions(&state).unwrap();
    assert!(actions.contains(&use_piece("b5")));
    assert!(actions.contains(&use_piece("f5")));
    let result = engine.apply(&state, use_piece("b5")).unwrap();
    assert_eq!(
        result.state.board.get("b3".parse().unwrap()).unwrap().id,
        king.id
    );
    assert_eq!(
        result.state.board.get("b2".parse().unwrap()).unwrap().id,
        rook.id
    );
    assert!(
        result
            .state
            .card_runtime
            .castled_piece_ids
            .contains(&king.id)
    );
    assert!(result.state.card_runtime.imperial_moves[&king.id].contains(&PieceKind::Pawn));
    assert_eq!(
        result
            .events
            .iter()
            .filter(|e| matches!(e, Event::PieceCaptured { .. }))
            .count(),
        2
    );
    assert_eq!(result.state.turn, Player::White);
}

#[test]
fn emergency_evacuation_resolves_selected_order_and_blocks_only_moved_pieces() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("c2", Player::White, PieceKind::Queen),
            ("c3", Player::White, PieceKind::Rook),
            ("c5", Player::Black, PieceKind::Knight),
        ],
        Player::White,
    );
    let queen = state.board.get("c2".parse().unwrap()).unwrap();
    let rook = state.board.get("c3".parse().unwrap()).unwrap();
    let state = engine
        .grant_card(&state, 1, crate::EMERGENCY_EVACUATION, Player::White)
        .unwrap()
        .state;
    let choice = |s: &[&str]| Action::UseCard {
        instance_id: 1,
        target: ActionTarget::Squares(s.iter().map(|p| p.parse().unwrap()).collect()),
    };
    assert!(
        engine
            .legal_actions(&state)
            .unwrap()
            .contains(&choice(&["c2", "c3"]))
    );
    let reversed = engine.apply(&state, choice(&["c3", "c2"])).unwrap().state;
    assert_eq!(
        reversed.board.get("c3".parse().unwrap()).unwrap().id,
        rook.id
    );
    assert_eq!(
        reversed.board.get("c1".parse().unwrap()).unwrap().id,
        queen.id
    );
    let mut moved = engine.apply(&state, choice(&["c2", "c3"])).unwrap().state;
    assert_eq!(moved.board.get("c2".parse().unwrap()).unwrap().id, rook.id);
    assert_eq!(moved.board.get("c1".parse().unwrap()).unwrap().id, queen.id);
    assert!(!has_move(&engine, &moved, "c2", "c5"));
    moved = engine.apply(&moved, movement("c2", "c3")).unwrap().state;
    moved = engine.apply(&moved, movement("h8", "g8")).unwrap().state;
    assert!(has_move(&engine, &moved, "c3", "c5"));
}

#[test]
fn emergency_evacuation_into_black_hole_resolves_hazard_after_moving() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d6", Player::White, PieceKind::Rook),
        ],
        Player::White,
    );
    let state = engine.apply_rule(&state, crate::BLACK_HOLE).unwrap().state;
    let state = engine
        .grant_card(&state, 1, crate::EMERGENCY_EVACUATION, Player::White)
        .unwrap()
        .state;
    let action = Action::UseCard {
        instance_id: 1,
        target: ActionTarget::Squares(vec!["d6".parse().unwrap()]),
    };
    let after = engine.apply(&state, action).unwrap().state;
    assert!(after.board.get("d6".parse().unwrap()).is_none());
    assert!(after.board.get("d5".parse().unwrap()).is_none());
}

#[test]
fn vortex_shuffles_only_eligible_enemy_pieces_with_one_roll_per_shuffle_step() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a8", Player::Black, PieceKind::Rook),
            ("b8", Player::Black, PieceKind::Knight),
            ("c8", Player::Black, PieceKind::Slime),
            ("d8", Player::Black, PieceKind::BigBishop),
            ("a7", Player::Black, PieceKind::Pawn),
        ],
        Player::White,
    );
    let a = state.board.get("a8".parse().unwrap()).unwrap();
    let b = state.board.get("b8".parse().unwrap()).unwrap();
    let slime = state.board.get("c8".parse().unwrap()).unwrap();
    let large = state.board.get("d8".parse().unwrap()).unwrap();
    state = engine
        .grant_card(&state, 1, crate::VORTEX, Player::White)
        .unwrap()
        .state;
    let mut expected_rng = state.clone();
    let (_, index) = expected_rng.rng_mut().web_choice_index(2);
    let result = engine.apply(&state, use_none()).unwrap();
    assert_eq!(
        result
            .events
            .iter()
            .filter(|e| matches!(e, Event::RandomSample { purpose, .. } if purpose == "vortex"))
            .count(),
        1
    );
    let expected_a = if index == Some(0) { b.id } else { a.id };
    assert_eq!(
        result.state.board.get("a8".parse().unwrap()).unwrap().id,
        expected_a
    );
    assert_eq!(
        result.state.board.get("c8".parse().unwrap()).unwrap().id,
        slime.id
    );
    assert_eq!(
        result.state.board.get("d8".parse().unwrap()).unwrap().id,
        large.id
    );
}

#[test]
fn loyalist_selects_nonroyal_and_moves_to_empty_square_near_king() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("h5", Player::White, PieceKind::Rook),
            ("g5", Player::White, PieceKind::Slime),
            ("b1", Player::White, PieceKind::Pawn),
        ],
        Player::White,
    );
    let rook = state.board.get("h5".parse().unwrap()).unwrap();
    state = engine
        .grant_card(&state, 1, crate::LOYALIST, Player::White)
        .unwrap()
        .state;
    let actions = engine.legal_actions(&state).unwrap();
    assert!(actions.contains(&use_piece("h5")));
    assert!(!actions.contains(&use_piece("a1")));
    assert!(!actions.contains(&use_piece("g5")));
    state = engine.apply(&state, use_piece("h5")).unwrap().state;
    assert!(has_move(&engine, &state, "h5", "b2"));
    assert!(!has_move(&engine, &state, "h5", "b1"));
    state = engine.apply(&state, movement("h5", "b2")).unwrap().state;
    assert_eq!(state.board.get("b2".parse().unwrap()).unwrap().id, rook.id);
    assert!(state.card_runtime.loyalist_piece_ids.contains(&rook.id));
    assert_eq!(state.turn, Player::Black);
}

#[test]
fn bishop_infiltration_jumps_to_empty_squares_without_jump_capture_until_third_start() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("c1", Player::White, PieceKind::Bishop),
            ("d2", Player::White, PieceKind::Pawn),
            ("f4", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    assert!(!has_move(&engine, &state, "c1", "e3"));
    state = engine
        .grant_card(&state, 1, crate::BISHOP_INFILTRATION, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, use_none()).unwrap().state;
    assert!(has_move(&engine, &state, "c1", "e3"));
    assert!(!has_move(&engine, &state, "c1", "f4"));
    assert!(has_move(&engine, &state, "c1", "g5"));
    for (w_from, w_to, b_from, b_to) in [
        ("a1", "b1", "h8", "g8"),
        ("b1", "a1", "g8", "h8"),
        ("a1", "b1", "h8", "g8"),
    ] {
        state = engine.apply(&state, movement(w_from, w_to)).unwrap().state;
        state = engine.apply(&state, movement(b_from, b_to)).unwrap().state;
    }
    assert!(!has_move(&engine, &state, "c1", "e3"));
    assert!(state.card_runtime.bishop_infiltration_due[Player::White.index()].is_none());
}

#[test]
fn bishop_infiltration_jump_can_land_in_black_hole_and_trigger_it() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b2", Player::White, PieceKind::Bishop),
            ("c3", Player::White, PieceKind::Pawn),
        ],
        Player::White,
    );
    let state = engine.apply_rule(&state, crate::BLACK_HOLE).unwrap().state;
    let state = engine
        .grant_card(&state, 1, crate::BISHOP_INFILTRATION, Player::White)
        .unwrap()
        .state;
    let state = engine.apply(&state, use_none()).unwrap().state;
    assert!(has_move(&engine, &state, "b2", "d4"));
    let after = engine.apply(&state, movement("b2", "d4")).unwrap().state;
    assert!(after.board.get("b2".parse().unwrap()).is_none());
    assert!(after.board.get("d4".parse().unwrap()).is_none());
}

#[test]
fn switcheroo_swaps_royal_and_pawn_consumes_turn_and_resets_quiet_count() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("e2", Player::White, PieceKind::Pawn),
            ("f2", Player::White, PieceKind::Pawn),
            ("g2", Player::White, PieceKind::Pawn),
        ],
        Player::White,
    );
    let king = state.board.get("a1".parse().unwrap()).unwrap();
    let pawn = state.board.get("g2".parse().unwrap()).unwrap();
    let e = state.board.get("e2".parse().unwrap()).unwrap();
    let f = state.board.get("f2".parse().unwrap()).unwrap();
    state
        .card_runtime
        .effects
        .pawn_chain_links
        .push([e.id, f.id]);
    state.quiet_ply = 17;
    state = engine
        .grant_card(&state, 1, crate::SWITCHEROO, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, use_none()).unwrap().state;
    assert!(!has_move(&engine, &state, "a1", "e2"));
    assert!(!has_move(&engine, &state, "a1", "f2"));
    assert!(has_move(&engine, &state, "a1", "g2"));
    state = engine.apply(&state, movement("a1", "g2")).unwrap().state;
    assert_eq!(state.board.get("g2".parse().unwrap()).unwrap().id, king.id);
    assert_eq!(state.board.get("a1".parse().unwrap()).unwrap().id, pawn.id);
    assert_eq!(state.turn, Player::Black);
    assert_eq!(state.quiet_ply, 0);
}

#[test]
fn dice_samples_six_kinds_and_locks_the_opponent_for_two_turns() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h1", Player::White, PieceKind::Rook),
            ("h8", Player::Black, PieceKind::King),
            ("d6", Player::Black, PieceKind::Pawn),
            ("a8", Player::Black, PieceKind::Rook),
            ("f8", Player::Black, PieceKind::Bishop),
            ("g8", Player::Black, PieceKind::Knight),
            ("e8", Player::Black, PieceKind::Queen),
        ],
        Player::White,
    );
    state = engine
        .grant_card(&state, 1, crate::DICE, Player::White)
        .unwrap()
        .state;
    let used = engine.apply(&state, use_none()).unwrap();
    assert_eq!(
        used.events
            .iter()
            .filter(|e| matches!(e, Event::RandomSample { purpose, .. } if purpose == "dice"))
            .count(),
        1
    );
    let lock = used.state.card_runtime.dice_locks[Player::Black.index()].unwrap();
    assert_eq!(lock.remaining_turns, 2);
    state = engine
        .apply(&used.state, movement("a1", "b1"))
        .unwrap()
        .state;
    let locked_square = state
        .board
        .iter()
        .find_map(|(square, p)| (p.owner == Player::Black && p.kind == lock.kind).then_some(square))
        .unwrap();
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .iter()
            .any(|a| matches!(a, Action::Move(m) if m.from == locked_square))
    );
    let quiet = engine
        .legal_actions(&state)
        .unwrap()
        .into_iter()
        .find(|a| matches!(a, Action::Move(m) if state.board.get(m.to).is_none()))
        .unwrap();
    state = engine.apply(&state, quiet).unwrap().state;
    assert_eq!(
        state.card_runtime.dice_locks[Player::Black.index()]
            .unwrap()
            .remaining_turns,
        1
    );
    state = engine.apply(&state, movement("b1", "c1")).unwrap().state;
    let quiet = engine
        .legal_actions(&state)
        .unwrap()
        .into_iter()
        .find(|a| matches!(a, Action::Move(m) if state.board.get(m.to).is_none()))
        .unwrap();
    state = engine.apply(&state, quiet).unwrap().state;
    assert!(state.card_runtime.dice_locks[Player::Black.index()].is_none());
}

#[test]
fn suicide_bomber_targets_pawn_family_and_explosion_bypasses_shields() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Fanatic),
            ("e4", Player::White, PieceKind::Guard),
            ("c4", Player::White, PieceKind::Knight),
            ("c5", Player::White, PieceKind::Bishop),
            ("e5", Player::White, PieceKind::Pawn),
            ("f2", Player::White, PieceKind::StandardBearer),
            ("d8", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    let knight = state.board.get("c4".parse().unwrap()).unwrap();
    let bishop = state.board.get("c5".parse().unwrap()).unwrap();
    let gambit_pawn = state.board.get("e5".parse().unwrap()).unwrap();
    state.card_runtime.shielded_piece_ids.insert(knight.id);
    state.card_runtime.protected_piece_ids.insert(bishop.id);
    state
        .card_runtime
        .protected_piece_ids
        .insert(gambit_pawn.id);
    state
        .card_runtime
        .queens_gambit_protection
        .insert(gambit_pawn.id);
    state = engine
        .grant_card(&state, 1, crate::SUICIDE_BOMBER, Player::White)
        .unwrap()
        .state;
    let actions = engine.legal_actions(&state).unwrap();
    assert!(actions.contains(&use_piece("d4")));
    assert!(actions.contains(&use_piece("f2")));
    assert!(!actions.contains(&use_piece("c4")));
    state = engine.apply(&state, use_piece("d4")).unwrap().state;
    state = engine.apply(&state, movement("a1", "b1")).unwrap().state;
    state = engine.apply(&state, movement("d8", "d4")).unwrap().state;
    assert!(state.board.get("d4".parse().unwrap()).is_none());
    assert!(state.board.get("c4".parse().unwrap()).is_none());
    assert!(state.board.get("e4".parse().unwrap()).is_none());
    assert_eq!(
        state.board.get("e5".parse().unwrap()).unwrap().id,
        gambit_pawn.id
    );
    assert_eq!(
        state.board.get("c5".parse().unwrap()).unwrap().id,
        bishop.id
    );
}

#[test]
fn vigilance_protects_royal_through_next_enemy_turn_after_capture() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Pawn),
            ("b2", Player::White, PieceKind::Pawn),
            ("d8", Player::Black, PieceKind::Rook),
            ("a8", Player::Black, PieceKind::Rook),
        ],
        Player::Black,
    );
    let king = state.board.get("a1".parse().unwrap()).unwrap();
    state = engine
        .grant_card(&state, 1, crate::VIGILANCE, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, movement("d8", "d4")).unwrap().state;
    assert_eq!(
        state.card_runtime.vigilance_protection[&king.id].remaining_turns,
        1
    );
    state = engine.apply(&state, movement("b2", "b3")).unwrap().state;
    let hit = engine.apply(&state, movement("a8", "a1")).unwrap();
    assert_eq!(
        hit.state.board.get("a1".parse().unwrap()).unwrap().id,
        king.id
    );
    assert!(hit.state.board.get("a8".parse().unwrap()).is_some());
    assert!(
        !hit.state
            .card_runtime
            .vigilance_protection
            .contains_key(&king.id)
    );
}

#[test]
fn encouragement_protects_orthogonal_neighbors_and_does_not_consume_shields() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a2", Player::White, PieceKind::Rook),
            ("b2", Player::White, PieceKind::Knight),
            ("a8", Player::Black, PieceKind::Rook),
            ("d4", Player::Black, PieceKind::Bishop),
        ],
        Player::Black,
    );
    let rook = state.board.get("a2".parse().unwrap()).unwrap();
    state.card_runtime.shielded_piece_ids.insert(rook.id);
    state = engine
        .grant_card(&state, 1, crate::ENCOURAGEMENT, Player::White)
        .unwrap()
        .state;
    assert!(has_move(&engine, &state, "a8", "a2"));
    assert!(has_move(&engine, &state, "d4", "b2"));
    let hit = engine.apply(&state, movement("a8", "a2")).unwrap();
    assert_eq!(
        hit.state.board.get("a2".parse().unwrap()).unwrap().id,
        rook.id
    );
    assert!(hit.state.board.get("a8".parse().unwrap()).is_some());
    assert!(hit.state.card_runtime.shielded_piece_ids.contains(&rook.id));
}

#[test]
fn basic_training_adds_pawn_steps_and_respects_special_capture_rules() {
    let engine = Engine::bundled_latest().unwrap();
    for kind in [PieceKind::Recruiter, PieceKind::Missionary] {
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, kind),
                ("e5", Player::Black, PieceKind::Pawn),
                ("b2", Player::White, PieceKind::Pawn),
            ],
            Player::White,
        );
        state = engine
            .grant_card(&state, 1, crate::BASIC_TRAINING, Player::White)
            .unwrap()
            .state;
        let actions = engine.legal_actions(&state).unwrap();
        assert!(actions.contains(&use_piece("d4")));
        assert!(!actions.contains(&use_piece("b2")));
        state = engine.apply(&state, use_piece("d4")).unwrap().state;
        assert!(has_move(&engine, &state, "d4", "d5"));
        assert!(has_move(&engine, &state, "d4", "e5"));
        let next = engine.apply(&state, movement("d4", "e5")).unwrap().state;
        if kind == PieceKind::Missionary {
            assert_eq!(
                next.board.get("d4".parse().unwrap()).unwrap().kind,
                PieceKind::Missionary
            );
            assert_eq!(
                next.board.get("e5".parse().unwrap()).unwrap().owner,
                Player::White
            );
        } else {
            assert_eq!(
                next.board.get("e5".parse().unwrap()).unwrap().kind,
                PieceKind::Recruiter
            );
        }
    }
}

#[test]
fn metal_targets_unprotected_ranged_pieces_and_resists_capture() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Rook),
            ("b1", Player::White, PieceKind::Knight),
            ("d8", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    state = engine
        .grant_card(&state, 1, crate::METAL, Player::White)
        .unwrap()
        .state;
    let actions = engine.legal_actions(&state).unwrap();
    assert!(actions.contains(&use_piece("d4")));
    assert!(!actions.contains(&use_piece("b1")));
    state = engine.apply(&state, use_piece("d4")).unwrap().state;
    let rook = state.board.get("d4".parse().unwrap()).unwrap();
    state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
    state = engine.apply(&state, movement("d8", "d4")).unwrap().state;
    assert_eq!(state.board.get("d4".parse().unwrap()).unwrap().id, rook.id);
    assert!(state.board.get("d8".parse().unwrap()).is_some());
    state = engine.apply(&state, movement("d4", "d5")).unwrap().state;
    assert_eq!(state.card_runtime.metal_cooldowns.get(&rook.id), Some(&3));
    let mut insight = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Rook),
        ],
        Player::Black,
    );
    let target = insight.board.get("d4".parse().unwrap()).unwrap();
    insight.card_runtime.metalized_piece_ids.insert(target.id);
    insight.card_runtime.metal_cooldowns.insert(target.id, 2);
    insight = engine
        .grant_card(&insight, 1, crate::INSIGHT, Player::Black)
        .unwrap()
        .state;
    insight = engine.apply(&insight, use_none()).unwrap().state;
    assert!(
        !insight
            .card_runtime
            .metalized_piece_ids
            .contains(&target.id)
    );
    assert!(
        !insight
            .card_runtime
            .metal_cooldowns
            .contains_key(&target.id)
    );
}

#[test]
fn promotion_rush_adds_queen_moves_blocks_capture_and_expires_on_move_or_turn_end() {
    let engine = Engine::bundled_latest().unwrap();
    let initial = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Knight),
            ("d6", Player::Black, PieceKind::Pawn),
        ],
        Player::White,
    );
    let piece = initial.board.get("d4".parse().unwrap()).unwrap();
    let granted = engine
        .grant_card(&initial, 1, crate::PROMOTION_RUSH, Player::White)
        .unwrap()
        .state;
    let active = engine.apply(&granted, use_piece("d4")).unwrap().state;
    assert!(has_move(&engine, &active, "d4", "d5"));
    assert!(!has_move(&engine, &active, "d4", "d6"));
    let moved = engine.apply(&active, movement("d4", "d5")).unwrap().state;
    assert!(
        !moved
            .card_runtime
            .promotion_rush_piece_ids
            .contains(&piece.id)
    );
    let expired = engine.apply(&active, movement("a1", "b1")).unwrap().state;
    assert!(
        !expired
            .card_runtime
            .promotion_rush_piece_ids
            .contains(&piece.id)
    );
}

/// Chameleon Mutation is an OPENING passive from the 2026-10-27 balance pass: it
/// marks up to three random own pieces on acquisition, and a marked piece adopts
/// the kind of whatever it captures.
#[test]
fn chameleon_marks_on_pickup_and_the_marked_piece_adopts_the_captured_type() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Rook),
            ("d6", Player::Black, PieceKind::Bishop),
        ],
        Player::White,
    );
    state = engine
        .grant_card(&state, 1, crate::CHAMELEON_MUTATION, Player::White)
        .unwrap()
        .state;
    let rook = state.board.get("d4".parse().unwrap()).unwrap();
    assert!(
        state.card_runtime.chameleon_piece_ids.contains(&rook.id),
        "the only non-royal candidate must be marked"
    );
    state = engine.apply(&state, movement("d4", "d6")).unwrap().state;
    assert_eq!(
        state.board.get("d6".parse().unwrap()).unwrap().kind,
        PieceKind::Bishop,
        "the marked piece adopts the captured kind"
    );
}

#[test]
fn alekhine_requires_queen_nearest_home_on_same_file_and_grants_permanent_protection() {
    let engine = Engine::bundled_latest().unwrap();
    for (owner, queen, rook1, rook2) in [
        (Player::White, "d1", "d3", "d5"),
        (Player::Black, "d8", "d6", "d4"),
    ] {
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                (queen, owner, PieceKind::Queen),
                (rook1, owner, PieceKind::Rook),
                (rook2, owner, PieceKind::Rook),
            ],
            owner,
        );
        state = engine
            .grant_card(&state, 1, crate::ALEKHINE_MACHINE_GUN, owner)
            .unwrap()
            .state;
        assert!(engine.legal_actions(&state).unwrap().contains(&use_none()));
        state = engine.apply(&state, use_none()).unwrap().state;
        for square in [queen, rook1, rook2] {
            let piece = state.board.get(square.parse().unwrap()).unwrap();
            assert!(state.card_runtime.protected_piece_ids.contains(&piece.id));
            assert!(!state.card_runtime.shielded_piece_ids.contains(&piece.id));
        }
    }
}

#[test]
fn counterattack_retaliates_once_and_locks_the_pawn_for_its_next_turn() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("h1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Pawn),
            ("d8", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    let pawn = state.board.get("d4".parse().unwrap()).unwrap();
    state = engine
        .grant_card(&state, 1, crate::COUNTERATTACK, Player::White)
        .unwrap()
        .state;
    assert!(
        engine
            .legal_actions(&state)
            .unwrap()
            .contains(&use_piece("d4"))
    );
    state = engine.apply(&state, use_piece("d4")).unwrap().state;
    state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
    let transition = engine.apply(&state, movement("d8", "d4")).unwrap();
    assert_eq!(
        transition
            .state
            .board
            .get("d8".parse().unwrap())
            .unwrap()
            .id,
        pawn.id
    );
    assert!(
        !transition
            .state
            .card_runtime
            .counterattack_ready
            .contains(&pawn.id)
    );
    assert!(
        transition
            .state
            .card_runtime
            .counterattack_locked
            .contains(&pawn.id)
    );
    assert!(
        !transition
            .events
            .iter()
            .any(|e| matches!(e, Event::RandomSample { purpose, .. } if purpose == "parry"))
    );
    assert!(!has_move(&engine, &transition.state, "d8", "d7"));
    let state = engine
        .apply(&transition.state, movement("g1", "f1"))
        .unwrap()
        .state;
    assert!(!state.card_runtime.counterattack_locked.contains(&pawn.id));
}

#[test]
fn disarm_targets_visible_enemy_nonroyals_and_expires_after_their_turn() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a8", Player::Black, PieceKind::Rook),
            ("a5", Player::White, PieceKind::Pawn),
            ("b8", Player::Black, PieceKind::Knight),
        ],
        Player::White,
    );
    let hidden = state.board.get("b8".parse().unwrap()).unwrap();
    state.card_runtime.stealth_piece_ids.insert(hidden.id);
    state = engine
        .grant_card(&state, 1, crate::DISARM, Player::White)
        .unwrap()
        .state;
    let actions = engine.legal_actions(&state).unwrap();
    assert!(actions.contains(&use_piece("a8")));
    assert!(!actions.contains(&use_piece("b8")));
    assert!(!actions.contains(&use_piece("h8")));
    state = engine.apply(&state, use_piece("a8")).unwrap().state;
    state = engine.apply(&state, movement("a1", "b1")).unwrap().state;
    assert!(!has_move(&engine, &state, "a8", "a5"));
    assert!(has_move(&engine, &state, "a8", "a7"));
    state = engine.apply(&state, movement("h8", "g8")).unwrap().state;
    let rook = state.board.get("a8".parse().unwrap()).unwrap();
    assert!(!state.card_runtime.disarmed.contains_key(&rook.id));
}

#[test]
fn inertia_and_severance_accept_only_current_ranged_targets() {
    let engine = Engine::bundled_latest().unwrap();
    for id in [crate::INERTIA, crate::SEVERANCE] {
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d6", Player::Black, PieceKind::Rook),
                ("c6", Player::Black, PieceKind::Knight),
            ],
            Player::White,
        );
        state = engine
            .grant_card(&state, 1, id, Player::White)
            .unwrap()
            .state;
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("d6"))
        );
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("c6"))
        );
        state = engine.apply(&state, use_piece("d6")).unwrap().state;
        state = engine.apply(&state, movement("a1", "b1")).unwrap().state;
        if id == crate::INERTIA {
            assert!(!has_move(&engine, &state, "d6", "d5"));
            assert!(has_move(&engine, &state, "d6", "d4"));
        } else {
            assert!(has_move(&engine, &state, "d6", "d5"));
            assert!(!has_move(&engine, &state, "d6", "d4"));
        }
    }
}

#[test]
fn ranged_classification_counts_allied_monsters_for_berserker() {
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d6", Player::Black, PieceKind::Berserker),
        ],
        Player::White,
    );
    for file in 0..7 {
        state.board.set(
            Square::new(file, 6).unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Pawn).with_id(100 + u32::from(file))),
        );
    }
    let berserker = state.board.get("d6".parse().unwrap()).unwrap();
    assert!(super::wanted::ranged(
        &state,
        berserker,
        "d6".parse().unwrap()
    ));
    state.board.set(
        "a6".parse().unwrap(),
        Some(Piece::new(Player::Black, PieceKind::Monster).with_id(200)),
    );
    assert!(!super::wanted::ranged(
        &state,
        berserker,
        "d6".parse().unwrap()
    ));
}

#[test]
fn random_middle_effects_consume_one_roll_and_apply_to_the_only_candidate() {
    let engine = Engine::bundled_latest().unwrap();
    for id in [crate::EVASION, crate::FANATICAL_RITUAL] {
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d6", Player::Black, PieceKind::Rook),
            ],
            Player::White,
        );
        let king = state.board.get("a1".parse().unwrap()).unwrap();
        let rook = state.board.get("d6".parse().unwrap()).unwrap();
        state = engine
            .grant_card(&state, 1, id, Player::White)
            .unwrap()
            .state;
        let action = Action::UseCard {
            instance_id: 1,
            target: ActionTarget::None,
        };
        let transition = engine.apply(&state, action.clone()).unwrap();
        assert_eq!(transition, engine.apply(&state, action).unwrap());
        assert_eq!(
            transition
                .events
                .iter()
                .filter(|event| matches!(event, Event::RandomSample { .. }))
                .count(),
            1
        );
        if id == crate::EVASION {
            assert!(
                transition
                    .state
                    .card_runtime
                    .evasion_piece_ids
                    .contains(&king.id)
            );
        }
        if id == crate::ROYAL_SHIELD {
            assert!(
                transition
                    .state
                    .card_runtime
                    .shielded_piece_ids
                    .contains(&king.id)
            );
        }
        if id == crate::FANATICAL_RITUAL {
            let changed = transition.state.board.find_by_id(rook.id).unwrap().1;
            assert_eq!(
                (changed.owner, changed.kind),
                (Player::Black, PieceKind::Fanatic)
            );
            assert!(
                transition
                    .state
                    .card_runtime
                    .moved_piece_ids
                    .contains(&rook.id)
            );
        }
    }
}

#[test]
fn freeze_selects_three_distinct_nonroyals_and_counts_enemy_turns() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a8", Player::Black, PieceKind::Rook),
            ("b8", Player::Black, PieceKind::Knight),
            ("c8", Player::Black, PieceKind::Bishop),
            ("d8", Player::Black, PieceKind::Queen),
        ],
        Player::White,
    );
    let mut state = engine
        .grant_card(&state, 1, crate::FREEZE, Player::White)
        .unwrap()
        .state;
    let action = Action::UseCard {
        instance_id: 1,
        target: ActionTarget::None,
    };
    let result = engine.apply(&state, action).unwrap();
    assert_eq!(result.state.card_runtime.frozen.len(), 3);
    assert_eq!(
        result
            .events
            .iter()
            .filter(|e| matches!(e, Event::RandomSample { .. }))
            .count(),
        3
    );
    state = result.state;
    let frozen = state
        .card_runtime
        .frozen
        .keys()
        .copied()
        .collect::<Vec<_>>();
    engine.tick_card_statuses_after_turn(&mut state, Player::White);
    assert!(
        frozen
            .iter()
            .all(|id| state.card_runtime.frozen[id].remaining_turns == 3)
    );
    for expected in [2, 1] {
        engine.tick_card_statuses_after_turn(&mut state, Player::Black);
        assert!(
            frozen
                .iter()
                .all(|id| state.card_runtime.frozen[id].remaining_turns == expected)
        );
    }
    engine.tick_card_statuses_after_turn(&mut state, Player::Black);
    assert!(
        frozen
            .iter()
            .all(|id| !state.card_runtime.frozen.contains_key(id))
    );
}

#[test]
fn en_passant_bang_captures_adjacent_nonpawn_and_expires_on_turn_end() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d5", Player::White, PieceKind::Pawn),
            ("e5", Player::Black, PieceKind::Knight),
        ],
        Player::White,
    );
    let victim = state.board.get("e5".parse().unwrap()).unwrap();
    let state = engine
        .grant_card(&state, 1, crate::EN_PASSANT_BANG, Player::White)
        .unwrap()
        .state;
    let state = engine
        .apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None,
            },
        )
        .unwrap()
        .state;
    assert!(has_move(&engine, &state, "d5", "e6"));
    let done = engine.apply(&state, movement("d5", "e6")).unwrap().state;
    assert!(done.board.find_by_id(victim.id).is_none());
    assert!(!done.card_runtime.en_passant_frenzy[Player::White.index()]);
}

#[test]
fn snipe_requires_pawn_screen_and_tracks_reversal() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b2", Player::White, PieceKind::Bishop),
            ("c3", Player::White, PieceKind::Pawn),
            ("e5", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    let state = engine
        .grant_card(&state, 1, crate::SNIPE, Player::White)
        .unwrap()
        .state;
    let state = engine
        .apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None,
            },
        )
        .unwrap()
        .state;
    assert!(has_move(&engine, &state, "b2", "e5"));
    let mut nonpawn = state.clone();
    let mut screen = nonpawn.board.get("c3".parse().unwrap()).unwrap();
    screen.kind = PieceKind::Knight;
    nonpawn.board.set("c3".parse().unwrap(), Some(screen));
    assert!(!has_move(&engine, &nonpawn, "b2", "e5"));
    let mut reversed = state;
    reversed.card_runtime.reversal[0] = true;
    reversed.board.set("c3".parse().unwrap(), None);
    reversed.board.set(
        "b3".parse().unwrap(),
        Some(Piece::new(Player::White, PieceKind::Pawn).with_id(90)),
    );
    reversed.board.set(
        "b5".parse().unwrap(),
        Some(Piece::new(Player::Black, PieceKind::Rook).with_id(91)),
    );
    assert!(has_move(&engine, &reversed, "b2", "b5"));
}

#[test]
fn turn_effects_target_the_intended_color_and_expire_at_turn_boundaries() {
    let engine = Engine::bundled_latest().unwrap();
    for id in [crate::SOCIALISM, crate::TAUNT, crate::ROYAL_COMMAND] {
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("b1", Player::White, PieceKind::Knight),
                ("b8", Player::Black, PieceKind::Knight),
            ],
            Player::White,
        );
        let state = engine
            .grant_card(&state, 1, id, Player::White)
            .unwrap()
            .state;
        let mut state = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        match id {
            crate::SOCIALISM => assert_eq!(state.card_runtime.socialism[1], 1),
            crate::TAUNT => assert_eq!(state.card_runtime.taunt[1], 1),
            _ => {
                let status = state.card_runtime.royal_command[0].unwrap();
                assert_eq!((status.starts_at_turn, status.expires_at_turn), (1, 3));
            }
        }
        engine.tick_middle_statuses_after_turn(&mut state, Player::White, &mut Vec::new());
        if id == crate::SOCIALISM {
            assert_eq!(state.card_runtime.socialism[1], 1);
        }
        if id == crate::TAUNT {
            assert_eq!(state.card_runtime.taunt[1], 1);
        }
        engine.tick_middle_statuses_after_turn(&mut state, Player::Black, &mut Vec::new());
        if id == crate::SOCIALISM {
            assert_eq!(state.card_runtime.socialism[1], 0);
        }
        if id == crate::TAUNT {
            assert_eq!(state.card_runtime.taunt[1], 0);
        }
    }
}

#[test]
fn frontline_only_protects_rooks_present_at_acquisition_while_in_own_half() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(
        &[
            ("h1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a2", Player::White, PieceKind::Rook),
            ("a8", Player::Black, PieceKind::Rook),
            ("c8", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    let mut state = engine
        .acquire_for_setup(&before, crate::FRONTLINE_RESPONSE, Player::White)
        .unwrap()
        .state;
    let rook = state.board.get("a2".parse().unwrap()).unwrap();
    let attacker = state.board.get("a8".parse().unwrap()).unwrap();
    assert!(!engine.can_capture(&state, attacker, rook));
    state.board.set("a2".parse().unwrap(), None);
    state.board.set("a5".parse().unwrap(), Some(rook));
    assert!(engine.can_capture(&state, attacker, rook));
    let newer = Piece::new(Player::White, PieceKind::Rook).with_id(99);
    state.board.set("c2".parse().unwrap(), Some(newer));
    let other = state.board.get("c8".parse().unwrap()).unwrap();
    assert!(engine.can_capture(&state, other, newer));
}

#[test]
fn blizzard_freezes_both_kings_until_third_caster_turn_start() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("c2", Player::White, PieceKind::Rook),
            ("h6", Player::White, PieceKind::Rook),
            ("b8", Player::Black, PieceKind::Knight),
        ],
        Player::White,
    );
    let white_king = state.board.get("a1".parse().unwrap()).unwrap();
    let black_king = state.board.get("h8".parse().unwrap()).unwrap();
    let state = engine
        .grant_card(&state, 1, crate::ICE_KING, Player::White)
        .unwrap()
        .state;
    let mut state = engine
        .apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None,
            },
        )
        .unwrap()
        .state;
    assert_eq!(state.card_runtime.ice_king.len(), 2);
    assert!(!engine.can_capture(
        &state,
        state.board.get("h6".parse().unwrap()).unwrap(),
        black_king
    ));
    assert!(!has_move(&engine, &state, "a1", "a2"));
    for turn in 1..=3 {
        state.turns_taken[0] = turn;
        engine.settle_card_turn_start(&mut state, Player::White);
        assert_eq!(
            state.card_runtime.ice_king.contains_key(&white_king.id),
            turn < 3
        );
        assert_eq!(
            state.card_runtime.ice_king.contains_key(&black_king.id),
            turn < 3
        );
    }
}

#[test]
fn ice_sheet_targets_ranged_and_special_pieces_and_limits_each_direction() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d6", Player::Black, PieceKind::Rook),
            ("b6", Player::Black, PieceKind::MagicGirl),
            ("a2", Player::White, PieceKind::Pawn),
        ],
        Player::White,
    );
    let rook = state.board.get("d6".parse().unwrap()).unwrap();
    let magic = state.board.get("b6".parse().unwrap()).unwrap();
    let state = engine
        .grant_card(&state, 1, crate::ICE_SHEET, Player::White)
        .unwrap()
        .state;
    let mut state = engine
        .apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None,
            },
        )
        .unwrap()
        .state;
    assert!(state.card_runtime.ice_sheet.contains_key(&rook.id));
    assert!(state.card_runtime.ice_sheet.contains_key(&magic.id));
    state.turn = Player::Black;
    assert!(!has_move(&engine, &state, "d6", "d5"));
    assert!(has_move(&engine, &state, "d6", "d1"));
    for remaining in [2, 1] {
        engine.tick_middle_statuses_after_turn(&mut state, Player::Black, &mut Vec::new());
        assert_eq!(
            state.card_runtime.ice_sheet[&rook.id].remaining_turns,
            remaining
        );
    }
    engine.tick_middle_statuses_after_turn(&mut state, Player::Black, &mut Vec::new());
    assert!(!state.card_runtime.ice_sheet.contains_key(&rook.id));
}

#[test]
fn mistake_card_applies_to_next_opponent_turn_and_never_reverses_large_body_capture() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::BigRook),
            ("d8", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    let big = state.board.get("d4".parse().unwrap()).unwrap();
    state
        .card_runtime
        .insert_large_body(big.id, "d4".parse().unwrap(), PieceHealth::new(2));
    state = engine
        .grant_card(&state, 1, crate::MISTAKE_CARD, Player::White)
        .unwrap()
        .state;
    state = engine
        .apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None,
            },
        )
        .unwrap()
        .state;
    assert!(state.card_runtime.mistake_card[Player::Black.index()]);
    state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
    let result = engine.apply(&state, movement("d8", "d4")).unwrap();
    assert!(
        !result
            .events
            .iter()
            .any(|e| matches!(e, Event::RandomSample { purpose, .. } if purpose == "mistake-card"))
    );
    assert!(!result.state.card_runtime.mistake_card[Player::Black.index()]);
}

#[test]
fn feudal_contract_accepts_either_selection_order_and_counterstrikes() {
    let engine = Engine::bundled_latest().unwrap();
    for reverse in [false, true] {
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Pawn),
                ("a2", Player::White, PieceKind::Rook),
                ("d6", Player::Black, PieceKind::Rook),
            ],
            Player::White,
        );
        let guardian = state.board.get("a2".parse().unwrap()).unwrap();
        state = engine
            .grant_card(&state, 1, crate::FEUDAL_CONTRACT, Player::White)
            .unwrap()
            .state;
        let pair = if reverse { ["a2", "d4"] } else { ["d4", "a2"] };
        state = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Squares(
                        pair.iter().map(|s| s.parse().unwrap()).collect(),
                    ),
                },
            )
            .unwrap()
            .state;
        state = engine.apply(&state, movement("a1", "b1")).unwrap().state;
        state = engine.apply(&state, movement("d6", "d4")).unwrap().state;
        assert_eq!(
            state.board.get("d4".parse().unwrap()).unwrap().id,
            guardian.id
        );
        assert!(state.board.get("a2".parse().unwrap()).is_none());
        assert!(state.card_runtime.feudal_contracts.is_empty());
        assert!(state.card_runtime.moved_piece_ids.contains(&guardian.id));
    }
}

#[test]
fn solidarity_swaps_nonroyal_minors_across_the_board_and_expires() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b2", Player::White, PieceKind::Knight),
            ("g6", Player::White, PieceKind::Bishop),
            ("d4", Player::White, PieceKind::Rook),
        ],
        Player::White,
    );
    let knight = state.board.get("b2".parse().unwrap()).unwrap();
    let bishop = state.board.get("g6".parse().unwrap()).unwrap();
    state = engine
        .grant_card(&state, 1, crate::SOLIDARITY, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, use_none()).unwrap().state;
    assert!(has_move(&engine, &state, "b2", "g6"));
    assert!(!has_move(&engine, &state, "b2", "d4"));
    state = engine.apply(&state, movement("b2", "g6")).unwrap().state;
    assert_eq!(
        state.board.get("g6".parse().unwrap()).unwrap().id,
        knight.id
    );
    assert_eq!(
        state.board.get("b2".parse().unwrap()).unwrap().id,
        bishop.id
    );
    assert_eq!(state.turn, Player::Black);
    assert!(!state.card_runtime.solidarity_active[Player::White.index()]);
}

#[test]
fn panic_resolves_two_ordered_pieces_on_opponents_turn() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b6", Player::Black, PieceKind::Knight),
            ("g7", Player::Black, PieceKind::Bishop),
            ("a2", Player::White, PieceKind::Rook),
        ],
        Player::White,
    );
    let knight = state.board.get("b6".parse().unwrap()).unwrap();
    let bishop = state.board.get("g7".parse().unwrap()).unwrap();
    state = engine
        .grant_card(&state, 1, crate::PANIC, Player::White)
        .unwrap()
        .state;
    state = engine
        .apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::Squares(vec!["b6".parse().unwrap(), "g7".parse().unwrap()]),
            },
        )
        .unwrap()
        .state;
    assert_eq!(
        state.board.find_by_id(knight.id).unwrap().0,
        "b6".parse().unwrap()
    );
    let transition = engine.apply(&state, movement("a2", "a3")).unwrap();
    state = transition.state;
    assert_eq!(state.turn, Player::Black);
    assert_ne!(
        state.board.find_by_id(knight.id).unwrap().0,
        "b6".parse().unwrap()
    );
    assert_ne!(
        state.board.find_by_id(bishop.id).unwrap().0,
        "g7".parse().unwrap()
    );
    assert!(state.card_runtime.pending_panic.is_empty());
    assert_eq!(
        transition
            .events
            .iter()
            .filter(
                |event| matches!(event, Event::RandomSample { purpose, .. } if purpose == "panic")
            )
            .count(),
        2
    );
}

#[test]
fn panic_keeps_random_draw_for_a_target_captured_before_trigger() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b6", Player::Black, PieceKind::Knight),
            ("g7", Player::Black, PieceKind::Bishop),
            ("b2", Player::White, PieceKind::Rook),
        ],
        Player::White,
    );
    state = engine
        .grant_card(&state, 1, crate::PANIC, Player::White)
        .unwrap()
        .state;
    state = engine
        .apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::Squares(vec!["b6".parse().unwrap(), "g7".parse().unwrap()]),
            },
        )
        .unwrap()
        .state;
    let transition = engine.apply(&state, movement("b2", "b6")).unwrap();
    assert_eq!(
        transition
            .events
            .iter()
            .filter(
                |event| matches!(event, Event::RandomSample { purpose, .. } if purpose == "panic")
            )
            .count(),
        2
    );
    assert!(transition.state.card_runtime.pending_panic.is_empty());
}

#[test]
fn blockade_places_blocking_walls_until_third_owner_start() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("d4", Player::White, PieceKind::King),
            ("d5", Player::White, PieceKind::Pawn),
            ("a2", Player::White, PieceKind::Rook),
            ("h8", Player::Black, PieceKind::King),
            ("e8", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    state = engine
        .grant_card(&state, 1, crate::BLOCKADE, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, use_none()).unwrap().state;
    assert_eq!(state.card_runtime.blockade_walls.len(), 7);
    assert!(
        !state
            .card_runtime
            .barricades
            .contains(&"d5".parse().unwrap())
    );
    assert!(
        state
            .card_runtime
            .barricades
            .contains(&"e5".parse().unwrap())
    );
    let mut without_deadline = state.clone();
    without_deadline.card_runtime.blockade_walls.clear();
    assert_ne!(state.position_key(), without_deadline.position_key());
    state.turn = Player::Black;
    assert!(has_move(&engine, &state, "e8", "e6"));
    assert!(!has_move(&engine, &state, "e8", "e5"));
    assert!(!has_move(&engine, &state, "e8", "e4"));
    for turn in 1..=3 {
        state.turns_taken[Player::White.index()] = turn;
        engine.settle_card_turn_start(&mut state, Player::White);
        assert_eq!(
            state.card_runtime.blockade_walls.len(),
            if turn < 3 { 7 } else { 0 }
        );
    }
    assert!(has_move(&engine, &state, "e8", "e5"));
    assert!(has_move(&engine, &state, "e8", "e4"));
}

#[test]
fn blockade_wall_keeps_deadline_after_conveyor_moves_it() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d2", Player::White, PieceKind::Rook),
        ],
        Player::White,
    );
    state = engine
        .grant_card(&state, 1, crate::BLOCKADE, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, use_none()).unwrap().state;
    let old = "a2".parse().unwrap();
    let new = "a3".parse().unwrap();
    engine.rotate_rule_ring(&mut state, &[old, new], &mut Vec::new());
    assert!(!state.card_runtime.barricades.contains(&old));
    assert!(state.card_runtime.barricades.contains(&new));
    assert_eq!(
        state.card_runtime.blockade_walls.get(&new),
        Some(&(Player::White, 3))
    );
    state.turns_taken[Player::White.index()] = 3;
    engine.settle_card_turn_start(&mut state, Player::White);
    assert!(!state.card_runtime.barricades.contains(&new));
}

#[test]
fn blockade_prevents_large_body_overlap_not_only_anchor_landing() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("a8", Player::Black, PieceKind::King),
            ("g8", Player::Black, PieceKind::BigRook),
        ],
        Player::Black,
    );
    let large = state.board.get("g8".parse().unwrap()).unwrap();
    state
        .card_runtime
        .insert_large_body(large.id, "g8".parse().unwrap(), PieceHealth::new(2));
    let wall = "h6".parse().unwrap();
    state.card_runtime.barricades.insert(wall);
    state
        .card_runtime
        .blockade_walls
        .insert(wall, (Player::White, 3));
    assert!(!engine.large_landing_is_legal(&state, large, "g7".parse().unwrap(), 2));
    assert!(!has_move(&engine, &state, "g8", "g7"));
}

#[test]
fn roller_records_rook_travel_and_retains_road_after_turn() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a2", Player::White, PieceKind::Rook),
        ],
        Player::White,
    );
    state = engine
        .grant_card(&state, 1, crate::ROLLER, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, use_none()).unwrap().state;
    state = engine.apply(&state, movement("a2", "d2")).unwrap().state;
    for square in ["a2", "b2", "c2", "d2"] {
        assert!(
            state
                .card_runtime
                .highway_cells
                .contains(&square.parse().unwrap())
        );
    }
    assert_eq!(state.card_runtime.highway_cells.len(), 4);
    assert!(!state.card_runtime.roller_active[Player::White.index()]);
    assert_eq!(state.turn, Player::Black);
    let mut without_road = state.clone();
    without_road.card_runtime.highway_cells.clear();
    assert_ne!(state.position_key(), without_road.position_key());
}

#[test]
fn roller_roads_allow_orthogonal_travel_only_through_connected_road_cells() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b2", Player::White, PieceKind::Knight),
        ],
        Player::White,
    );
    state.card_runtime.highway_cells = ["b2", "c2", "d2"]
        .into_iter()
        .map(|s| s.parse().unwrap())
        .collect();
    assert!(has_move(&engine, &state, "b2", "d2"));
    assert!(!has_move(&engine, &state, "b2", "e2"));
    state
        .card_runtime
        .highway_cells
        .remove(&"c2".parse().unwrap());
    assert!(!has_move(&engine, &state, "b2", "d2"));
}

#[test]
fn roller_marks_teleport_endpoints_without_filling_intervening_squares() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a2", Player::White, PieceKind::Rook),
        ],
        Player::White,
    );
    state.card_runtime.roller_active[Player::White.index()] = true;
    state.card_runtime.e4_active[Player::White.index()] = true;
    state = engine.apply(&state, movement("a2", "e4")).unwrap().state;
    assert_eq!(
        state.card_runtime.highway_cells,
        ["a2", "e4"]
            .into_iter()
            .map(|s| s.parse().unwrap())
            .collect()
    );
}

#[test]
fn roller_marks_castling_rooks_path() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("e1", Player::White, PieceKind::King),
            ("h1", Player::White, PieceKind::Rook),
            ("a8", Player::Black, PieceKind::King),
        ],
        Player::White,
    );
    state.card_runtime.roller_active[Player::White.index()] = true;
    state = engine.apply(&state, movement("e1", "g1")).unwrap().state;
    assert_eq!(
        state.card_runtime.highway_cells,
        ["f1", "g1", "h1"]
            .into_iter()
            .map(|s| s.parse().unwrap())
            .collect()
    );
}

#[test]
fn highway_trail_does_not_grant_guard_a_long_range_capture() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b2", Player::White, PieceKind::Guard),
            ("e2", Player::Black, PieceKind::Pawn),
        ],
        Player::White,
    );
    state.card_runtime.highway_cells = ["b2", "c2", "d2", "e2"]
        .into_iter()
        .map(|s| s.parse().unwrap())
        .collect();
    assert!(has_move(&engine, &state, "b2", "d2"));
    assert!(!has_move(&engine, &state, "b2", "e2"));
}

#[test]
fn roller_records_both_segments_of_portal_travel() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("c1", Player::White, PieceKind::Rook),
        ],
        Player::White,
    );
    let mut state = engine.apply_rule(&state, crate::PORTAL).unwrap().state;
    state.card_runtime.roller_active[Player::White.index()] = true;
    state = engine.apply(&state, movement("c1", "f7")).unwrap().state;
    let expected = ["c1", "c2", "c3", "f6", "f7"]
        .into_iter()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(state.card_runtime.highway_cells, expected);
}

#[test]
fn roller_does_not_create_road_when_a_shield_stops_the_rook() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("h1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a2", Player::White, PieceKind::Rook),
            ("a5", Player::Black, PieceKind::Pawn),
        ],
        Player::White,
    );
    let pawn = state.board.get("a5".parse().unwrap()).unwrap();
    state.card_runtime.shielded_piece_ids.insert(pawn.id);
    state.card_runtime.roller_active[Player::White.index()] = true;
    state = engine.apply(&state, movement("a2", "a5")).unwrap().state;
    assert!(state.card_runtime.highway_cells.is_empty());
    assert!(state.board.get("a2".parse().unwrap()).is_some());
}

#[test]
fn chimera_uses_current_ordered_major_pool_and_resets_piece_abilities() {
    let engine = Engine::bundled_latest().unwrap();
    // Source: tricksterMovementTypes filtered by v19MajorTypes, in source order.
    let expected = [
        PieceKind::Queen,
        PieceKind::Rook,
        PieceKind::Herald,
        PieceKind::PrimeMinister,
        PieceKind::Amazon,
        PieceKind::Jester,
        PieceKind::Hook,
        PieceKind::Man,
        PieceKind::Assassin,
        PieceKind::Reaper,
        PieceKind::Windmill,
        PieceKind::Bear,
        PieceKind::MagicGirl,
        PieceKind::Berserker,
        PieceKind::Siren,
        PieceKind::Undead,
        PieceKind::Hedgehog,
        PieceKind::Princess,
        PieceKind::Octopus,
        PieceKind::Grappler,
        PieceKind::RevolvingDoor,
        PieceKind::Chancellor,
        PieceKind::Archbishop,
        PieceKind::Killer,
    ];
    for (index, kind) in expected.into_iter().enumerate() {
        let seed = (0..10000)
            .find(|seed| {
                crate::DeterministicRng::new(*seed)
                    .web_choice_index(expected.len())
                    .1
                    == Some(index)
            })
            .unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("a2", Player::White, PieceKind::Rook),
            ],
            Player::White,
        );
        *state.rng_mut() = crate::DeterministicRng::new(seed);
        let piece = state.board.get("a2".parse().unwrap()).unwrap();
        state.card_runtime.chimera_piece_ids.insert(piece.id);
        state.card_runtime.shielded_piece_ids.insert(piece.id);
        state
            .card_runtime
            .windmill_modes
            .insert(piece.id, WindmillMode::Rook);
        state.card_runtime.reaper_souls.insert(piece.id, 9);
        let after = engine.apply(&state, movement("a2", "a3")).unwrap().state;
        assert_eq!(
            after.board.find_by_id(piece.id).unwrap().1.kind,
            kind,
            "pool index {index}"
        );
        assert!(!after.card_runtime.shielded_piece_ids.contains(&piece.id));
        assert_eq!(
            after.card_runtime.windmill_modes.get(&piece.id),
            (kind == PieceKind::Windmill).then_some(&WindmillMode::Bishop)
        );
        assert_eq!(
            after.card_runtime.reaper_souls.get(&piece.id),
            (kind == PieceKind::Reaper).then_some(&0)
        );
        assert_eq!(
            after.card_runtime.bear_retaliations.get(&piece.id),
            matches!(kind, PieceKind::Bear | PieceKind::Hedgehog).then_some(&2)
        );
    }
}

#[test]
fn chimera_selects_nonroyal_major_and_marks_it_moved() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a2", Player::White, PieceKind::Rook),
            ("b2", Player::White, PieceKind::Knight),
        ],
        Player::White,
    );
    state = engine
        .grant_card(&state, 1, crate::CHIMERA, Player::White)
        .unwrap()
        .state;
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .contains(&use_piece("b2"))
    );
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .contains(&use_piece("a1"))
    );
    let rook = state.board.get("a2".parse().unwrap()).unwrap();
    state = engine.apply(&state, use_piece("a2")).unwrap().state;
    assert!(state.card_runtime.chimera_piece_ids.contains(&rook.id));
    assert!(state.card_runtime.moved_piece_ids.contains(&rook.id));
}

#[test]
fn file_conveyor_waits_for_the_end_of_the_next_owner_turn() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b2", Player::White, PieceKind::Rook),
            ("c2", Player::White, PieceKind::Knight),
            ("g7", Player::Black, PieceKind::Knight),
        ],
        Player::White,
    );
    let rook = state.board.get("b2".parse().unwrap()).unwrap();
    state = engine
        .grant_card(&state, 1, crate::FILE_CONVEYOR, Player::White)
        .unwrap()
        .state;
    state = engine
        .apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::Square("b6".parse().unwrap()),
            },
        )
        .unwrap()
        .state;
    assert_eq!(
        state.card_runtime.file_conveyors,
        vec![(1, Player::White, 2)]
    );
    state = engine.apply(&state, movement("c2", "d4")).unwrap().state;
    assert_eq!(
        state.board.find_by_id(rook.id).unwrap().0,
        "b2".parse().unwrap()
    );
    state = engine.apply(&state, movement("g7", "f5")).unwrap().state;
    assert_eq!(
        state.board.find_by_id(rook.id).unwrap().0,
        "b2".parse().unwrap()
    );
    state = engine.apply(&state, movement("d4", "e6")).unwrap().state;
    assert_eq!(
        state.board.find_by_id(rook.id).unwrap().0,
        "b3".parse().unwrap()
    );
}

#[test]
fn file_conveyor_moves_chains_and_does_not_wrap_at_board_edge() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b3", Player::White, PieceKind::Rook),
            ("b4", Player::White, PieceKind::Bishop),
            ("b7", Player::Black, PieceKind::Knight),
            ("b8", Player::Black, PieceKind::Rook),
        ],
        Player::White,
    );
    let rook = state.board.get("b3".parse().unwrap()).unwrap();
    let bishop = state.board.get("b4".parse().unwrap()).unwrap();
    let edge = state.board.get("b8".parse().unwrap()).unwrap();
    let ring: Vec<_> = (0..8).map(|rank| Square::new(1, rank).unwrap()).collect();
    engine.rotate_conveyor(&mut state, &ring, true, &mut Vec::new());
    assert_eq!(
        state.board.find_by_id(rook.id).unwrap().0,
        "b4".parse().unwrap()
    );
    assert_eq!(
        state.board.find_by_id(bishop.id).unwrap().0,
        "b5".parse().unwrap()
    );
    assert_eq!(
        state.board.find_by_id(edge.id).unwrap().0,
        "b8".parse().unwrap()
    );
    assert!(state.board.get("b1".parse().unwrap()).is_none());
}

#[test]
fn file_conveyor_uses_black_direction_and_same_file_replaces_previous_owner() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b5", Player::Black, PieceKind::Rook),
        ],
        Player::Black,
    );
    let rook = state.board.get("b5".parse().unwrap()).unwrap();
    state
        .card_runtime
        .file_conveyors
        .push((1, Player::White, 8));
    state = engine
        .grant_card(&state, 1, crate::FILE_CONVEYOR, Player::Black)
        .unwrap()
        .state;
    state = engine
        .apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::Square("b1".parse().unwrap()),
            },
        )
        .unwrap()
        .state;
    assert_eq!(
        state.card_runtime.file_conveyors,
        vec![(1, Player::Black, 2)]
    );
    state.turns_taken[Player::Black.index()] = 2;
    engine.settle_file_conveyors(&mut state, Player::Black, &mut Vec::new());
    assert_eq!(
        state.board.find_by_id(rook.id).unwrap().0,
        "b4".parse().unwrap()
    );
}

#[test]
fn trolley_balanced_sampling_matches_pinned_source_for_multiple_seeds() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
        ],
        Player::White,
    );
    for (square, id, owner, kind) in [
        ("a2", 10, Player::White, PieceKind::Rook),
        ("b2", 11, Player::White, PieceKind::Pawn),
        ("c2", 12, Player::White, PieceKind::Bishop),
        ("d2", 13, Player::White, PieceKind::Knight),
        ("e2", 14, Player::White, PieceKind::Pawn),
        ("f2", 15, Player::White, PieceKind::Pawn),
        ("a7", 20, Player::Black, PieceKind::Rook),
        ("b7", 21, Player::Black, PieceKind::Knight),
        ("c7", 22, Player::Black, PieceKind::Pawn),
        ("d7", 23, Player::Black, PieceKind::Bishop),
        ("e7", 24, Player::Black, PieceKind::Pawn),
        ("f7", 25, Player::Black, PieceKind::Rook),
        ("g7", 26, Player::Black, PieceKind::Queen),
        ("h7", 27, Player::Black, PieceKind::Bishop),
    ] {
        state.board.set(
            square.parse().unwrap(),
            Some(Piece::new(owner, kind).with_id(id)),
        );
    }
    for (seed, expected) in [
        (0, [vec![22, 23], vec![20, 21, 27, 11, 10, 14]]),
        (1, [vec![20, 27, 24, 10, 15], vec![21]]),
        (42, [vec![23, 24], vec![20, 25, 11, 10]]),
        (0xdead_beef, [vec![26, 22, 14, 12, 13], vec![23]]),
    ] {
        assert_eq!(
            super::super::trolley::bundles(&engine, &state, Player::Black, seed),
            Some(expected),
            "seed {seed}"
        );
    }
}

#[test]
fn trolley_uses_one_seed_ends_turn_and_opponent_selects_mixed_casualties() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = GameState::standard(crate::LATEST_RULESET_ID, crate::Mode::Standard, 59);
    state.completed_drafts = [true; 3];
    state = engine
        .grant_card(&state, 1, crate::TROLLEY, Player::White)
        .unwrap()
        .state;
    let mut expected_rng = *state.rng_mut();
    let (_, seed) = expected_rng.web_choice_index(1_usize << 32);
    let expected =
        super::super::trolley::bundles(&engine, &state, Player::Black, seed.unwrap() as u32)
            .unwrap();
    let transition = engine.apply(&state, use_none()).unwrap();
    let mut after = transition.state;
    assert_eq!(after.turn, Player::Black);
    assert_eq!(after.rng_mut().state(), expected_rng.state());
    assert_eq!(
        transition
            .events
            .iter()
            .filter(
                |e| matches!(e, Event::RandomSample { purpose, .. } if purpose == "trolley-seed")
            )
            .count(),
        1
    );
    let choice = match &after.phase {
        GamePhase::AwaitingChoice(choice) => choice.clone(),
        other => panic!("expected trolley choice, got {other:?}"),
    };
    assert_eq!(choice.owner, Player::Black);
    assert_eq!(choice.source_card_instance, 1);
    let dilemma = after.card_runtime.active_trolley.clone().unwrap();
    assert_eq!(dilemma.bundles, expected);
    let mixed = dilemma
        .bundles
        .iter()
        .position(|bundle| {
            bundle
                .iter()
                .filter_map(|id| after.board.find_by_id(*id))
                .any(|(_, p)| p.owner == Player::White)
        })
        .unwrap();
    let selected = dilemma.bundles[mixed].clone();
    let surviving = dilemma.bundles[1 - mixed].clone();
    assert!(
        selected
            .iter()
            .filter_map(|id| after.board.find_by_id(*id))
            .any(|(_, p)| p.owner == Player::Black)
    );
    after = engine
        .apply(
            &after,
            Action::ResolveChoice {
                choice_id: choice.id,
                option_index: mixed as u16,
            },
        )
        .unwrap()
        .state;
    assert!(
        selected
            .iter()
            .all(|id| after.board.find_by_id(*id).is_none())
    );
    assert!(
        surviving
            .iter()
            .all(|id| after.board.find_by_id(*id).is_some())
    );
    assert_eq!(after.turn, Player::Black);
    assert!(matches!(after.phase, GamePhase::Playing));
    assert_eq!(after.actions_remaining, 1);
}

#[test]
fn trolley_has_no_action_when_no_balanced_bundles_are_possible() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
        ],
        Player::White,
    );
    let state = engine
        .grant_card(&state, 1, crate::TROLLEY, Player::White)
        .unwrap()
        .state;
    assert!(!engine.legal_actions(&state).unwrap().contains(&use_none()));
}

#[test]
fn trolley_relaxes_ideal_score_ranges_when_only_a_less_balanced_pair_exists() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
        ],
        Player::White,
    );
    for (square, id, owner, kind) in [
        ("a2", 12, Player::White, PieceKind::Bishop),
        ("b7", 21, Player::Black, PieceKind::Bishop),
        ("c7", 26, Player::Black, PieceKind::Queen),
    ] {
        state.board.set(
            square.parse().unwrap(),
            Some(Piece::new(owner, kind).with_id(id)),
        );
    }
    assert_eq!(
        super::super::trolley::bundles(&engine, &state, Player::Black, 0),
        Some([vec![21], vec![26, 12]])
    );
}

use super::acquisition::{game, position};
use super::*;

#[test]
fn opening_mass_transforms_preserve_identity_and_the_source_moved_policy() {
    let engine = Engine::bundled_latest().unwrap();
    for owner in [Player::White, Player::Black] {
        for (id, from, to, moved, fresh) in [
            (
                crate::DUTCH,
                PieceKind::Rook,
                PieceKind::Windmill,
                false,
                false,
            ),
            (
                crate::LAST_STAND,
                PieceKind::Pawn,
                PieceKind::Fanatic,
                false,
                true,
            ),
            (
                crate::PAWN_CONVERSION,
                PieceKind::Pawn,
                PieceKind::BerolinaPawn,
                false,
                true,
            ),
            (
                crate::APPRENTICE_KNIGHTS,
                PieceKind::Pawn,
                PieceKind::Squire,
                true,
                true,
            ),
        ] {
            let before = game();
            let square = match (owner, from) {
                (Player::White, PieceKind::Rook) => "a1",
                (Player::Black, PieceKind::Rook) => "a8",
                (Player::White, _) => "b2",
                _ => "b7",
            }
            .parse()
            .unwrap();
            let original = before.board.get(square).unwrap();
            let after = engine.acquire_for_setup(&before, id, owner).unwrap().state;
            let transformed = after.board.get(square).unwrap();
            assert_eq!(
                (transformed.id, transformed.kind),
                (original.id, to),
                "{id}"
            );
            assert_eq!(
                after.card_runtime.moved_piece_ids.contains(&original.id),
                moved,
                "{id}"
            );
            assert_eq!(
                after
                    .rule_runtime
                    .piece_fresh_until
                    .contains_key(&original.id),
                fresh,
                "{id}"
            );
            assert_eq!(after.rng_state(), before.rng_state());
        }
    }
}

#[test]
fn guard_uses_royal_current_file_and_cavalry_selects_only_homeward_pawn() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("c1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("c2", Player::White, PieceKind::Pawn),
        ("c4", Player::White, PieceKind::Fanatic),
        ("d2", Player::White, PieceKind::Pawn),
        ("d4", Player::White, PieceKind::Pawn),
    ]);
    let guard = engine
        .acquire_for_setup(&before, crate::GUARD, Player::White)
        .unwrap()
        .state;
    for square in ["c2", "c4"] {
        assert_eq!(
            guard.board.get(square.parse().unwrap()).unwrap().kind,
            PieceKind::Guard
        );
    }
    let cavalry = engine
        .acquire_for_setup(&before, crate::QUEEN_CAVALRY, Player::White)
        .unwrap()
        .state;
    assert_eq!(
        cavalry.board.get("d2".parse().unwrap()).unwrap().kind,
        PieceKind::Knight
    );
    assert_eq!(
        cavalry.board.get("d4".parse().unwrap()).unwrap().kind,
        PieceKind::Pawn
    );
    assert!(cavalry.rule_runtime.piece_fresh_until.is_empty());
}

#[test]
fn elephant_spawn_is_atomic_and_uses_diagonal_rule_squares() {
    let engine = Engine::bundled_latest().unwrap();
    for owner in [Player::White, Player::Black] {
        let before = game();
        let result = engine
            .acquire_for_setup(&before, crate::ELEPHANT_ESCAPE, owner)
            .unwrap()
            .state;
        let rank = if owner == Player::White { 2 } else { 5 };
        for file in [0, 7] {
            let p = result.board.get(Square::new(file, rank).unwrap()).unwrap();
            assert_eq!(p.kind, PieceKind::Alfil);
            assert!(result.rule_runtime.piece_fresh_until.contains_key(&p.id));
        }
        let mut blocked = before.clone();
        blocked
            .card_runtime
            .barricades
            .insert(Square::new(0, rank).unwrap());
        let result = engine
            .grant_card(&blocked, 1, crate::ELEPHANT_ESCAPE, owner)
            .unwrap()
            .state;
        assert_eq!(result.board, blocked.board);
    }
    let mut before = position(&[
        ("h1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
    ]);
    before
        .rule_runtime
        .enabled
        .insert(crate::DIAGONAL_CHESS.into());
    let after = engine
        .acquire_for_setup(&before, crate::ELEPHANT_ESCAPE, Player::White)
        .unwrap()
        .state;
    for s in ["a6", "f1"] {
        assert_eq!(
            after.board.get(s.parse().unwrap()).unwrap().kind,
            PieceKind::Alfil
        );
    }
}

#[test]
fn knightmate_retains_royal_identity_and_martyrdom_is_a_sacrifice() {
    let engine = Engine::bundled_latest().unwrap();
    let before = game();
    let king = before.board.get("e1".parse().unwrap()).unwrap();
    let after = engine
        .acquire_for_setup(&before, crate::KNIGHTMATE, Player::White)
        .unwrap()
        .state;
    assert_eq!(
        after.board.find_by_id(king.id).unwrap().1.kind,
        PieceKind::RoyalKnight
    );
    assert!(after.card_runtime.royal_piece_ids.contains(&king.id));
    assert_eq!(
        after.board.get("b1".parse().unwrap()).unwrap().kind,
        PieceKind::Man
    );
}

/// Martyrdom is upstream-disabled from the 2026-11-02 snapshot. Its sacrifice and
/// shield behavior is retained for the historical rulesets, so the effect itself
/// is still exercised here directly rather than through a card grant.
#[test]
fn martyrdom_sacrifices_bishops_and_shields_pawns_without_capture_credit() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("c3", Player::White, PieceKind::Bishop),
        ("f3", Player::White, PieceKind::Bishop),
        ("d2", Player::White, PieceKind::Pawn),
    ]);
    let mut state = before.clone();
    let mut events = Vec::new();
    crate::engine::cards::martyrdom::acquire(&engine, &mut state, Player::White, &mut events)
        .unwrap();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Event::PieceCaptured { .. }))
    );
    assert_eq!(
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == Player::White && p.kind == PieceKind::Bishop)
            .count(),
        0
    );
    assert_eq!(
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == Player::White
                && p.kind == PieceKind::Pawn
                && state.card_runtime.shielded_piece_ids.contains(&p.id))
            .count(),
        1
    );
    // And the card itself is refused by the current profile.
    assert!(
        engine
            .acquire_for_setup(&before, crate::MARTYRDOM, Player::White)
            .is_err(),
        "an upstream-disabled card must not be granted"
    );
}

#[test]
fn random_rook_transforms_choose_without_rng_for_a_single_target() {
    let engine = Engine::bundled_latest().unwrap();
    for (card, kind) in [
        (crate::HERALD, PieceKind::Herald),
        (crate::PRINCESS, PieceKind::Princess),
    ] {
        let before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("c3", Player::White, PieceKind::Rook),
        ]);
        let old = before.board.get("c3".parse().unwrap()).unwrap();
        let after = engine
            .acquire_for_setup(&before, card, Player::White)
            .unwrap();
        assert_eq!(after.state.rng_state(), before.rng_state());
        let piece = after.state.board.get("c3".parse().unwrap()).unwrap();
        assert_eq!((piece.id, piece.kind), (old.id, kind));
        assert!(after.state.card_runtime.moved_piece_ids.contains(&piece.id));
        assert!(
            !after
                .state
                .rule_runtime
                .piece_fresh_until
                .contains_key(&piece.id)
        );
    }
}
#[test]
fn locust_crosses_a_distant_hurdle_and_rocket_only_augments_an_unmoved_rook() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("d4", Player::White, PieceKind::Knight),
        ("d6", Player::Black, PieceKind::Pawn),
        ("c2", Player::White, PieceKind::Queen),
        ("f2", Player::White, PieceKind::Rook),
    ]);
    let after = engine
        .acquire_for_setup(&before, crate::LOCUST_SWARM, Player::White)
        .unwrap()
        .state;
    let moves = engine.legal_actions(&after).unwrap();
    assert!(moves.contains(&acquisition::movement("d4", "d7")));
    let queen = after.board.get("c2".parse().unwrap()).unwrap();
    assert!(
        !after
            .card_runtime
            .locust_swarm_piece_ids
            .contains(&queen.id)
    );
    let after = engine
        .acquire_for_setup(&before, crate::ROCKET, Player::White)
        .unwrap()
        .state;
    assert!(
        engine
            .legal_actions(&after)
            .unwrap()
            .contains(&acquisition::movement("f2", "e4"))
    );
    let mut moved = after.clone();
    let rook = moved.board.get("f2".parse().unwrap()).unwrap();
    moved.card_runtime.moved_piece_ids.insert(rook.id);
    assert!(
        !engine
            .legal_actions(&moved)
            .unwrap()
            .contains(&acquisition::movement("f2", "e4"))
    );
}

#[test]
fn horde_and_big_rook_install_layouts_without_fresh_capture_delay() {
    let engine = Engine::bundled_latest().unwrap();
    for owner in [Player::White, Player::Black] {
        let before = game();
        let king = before
            .board
            .get(
                (if owner == Player::White { "e1" } else { "e8" })
                    .parse()
                    .unwrap(),
            )
            .unwrap();
        let horde = engine
            .acquire_for_setup(&before, crate::HORDE, owner)
            .unwrap()
            .state;
        assert_eq!(horde.board.find_by_id(king.id).unwrap().1.id, king.id);
        assert_eq!(
            horde
                .board
                .iter()
                .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
                .count(),
            35
        );
        assert!(
            horde
                .board
                .iter()
                .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
                .all(|(_, p)| !horde
                    .card_runtime
                    .fresh_no_capture_until
                    .contains_key(&p.id))
        );
        let big = engine
            .acquire_for_setup(&before, crate::BIG_ROOK, owner)
            .unwrap()
            .state;
        let anchor = if owner == Player::White { "g2" } else { "g8" };
        let rook = big.board.get(anchor.parse().unwrap()).unwrap();
        assert_eq!(rook.kind, PieceKind::BigRook);
        assert_eq!(big.card_runtime.health(rook.id).unwrap().hp, 2);
        assert!(!big.card_runtime.moved_piece_ids.contains(&rook.id));
        assert!(
            !big.card_runtime
                .fresh_no_capture_until
                .contains_key(&rook.id)
        );
    }
}

#[test]
fn big_rook_loses_the_hit_cell_then_disappears_on_a_second_hit() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("a1", Player::White, PieceKind::King),
        ("a8", Player::Black, PieceKind::King),
        ("g5", Player::White, PieceKind::Rook),
        ("h5", Player::White, PieceKind::Rook),
        ("g8", Player::Black, PieceKind::BigRook),
    ]);
    let big = state.board.get("g8".parse().unwrap()).unwrap();
    state
        .card_runtime
        .insert_large_body(big.id, "g8".parse().unwrap(), PieceHealth::new(2));
    let first = engine
        .apply(&state, acquisition::movement("g5", "g7"))
        .unwrap()
        .state;
    assert_eq!(
        first.board.get("g7".parse().unwrap()).unwrap().owner,
        Player::White
    );
    assert_eq!(
        first.card_runtime.large_missing_cells.get(&big.id),
        Some(&2)
    );
    assert_eq!(
        engine
            .board_piece_at(&first, "h7".parse().unwrap())
            .unwrap()
            .id,
        big.id
    );
    assert!(first.board.find_by_id(big.id).is_some());
    let mut next = first;
    next.turn = Player::White;
    let second = engine
        .apply(&next, acquisition::movement("h5", "h7"))
        .unwrap()
        .state;
    assert!(second.board.find_by_id(big.id).is_none());
    assert_eq!(
        second.board.get("h7".parse().unwrap()).unwrap().owner,
        Player::White
    );
    assert!(
        !second
            .card_runtime
            .large_missing_cells
            .contains_key(&big.id)
    );
}

#[test]
fn big_rook_fracture_matches_all_four_source_cell_indices() {
    let engine = Engine::bundled_latest().unwrap();
    for (index, hit) in ["g8", "h8", "g7", "h7"].into_iter().enumerate() {
        let mut state = position(&[
            ("a1", Player::White, PieceKind::King),
            ("a8", Player::Black, PieceKind::King),
            ("g8", Player::Black, PieceKind::BigRook),
        ]);
        let big = state.board.get("g8".parse().unwrap()).unwrap();
        let anchor = "g8".parse().unwrap();
        state
            .card_runtime
            .insert_large_body(big.id, anchor, PieceHealth::new(2));
        let original = state.clone();
        assert_eq!(
            engine.fracture_large_cell(&mut state, big, hit.parse().unwrap()),
            Some(false)
        );
        assert_eq!(state.card_runtime.large_missing_cells[&big.id], index as u8);
        assert!(
            engine
                .board_piece_at(&state, hit.parse().unwrap())
                .is_none()
        );
        assert_ne!(state.position_key(), original.position_key());
        assert_ne!(state.fingerprint(), original.fingerprint());
        let remaining = large_piece_occupied_cells(&state, big.id, anchor);
        assert_eq!(remaining.len(), 3);
        assert_eq!(
            engine.fracture_large_cell(&mut state, big, remaining[0]),
            Some(true)
        );
    }
}

#[test]
fn fractured_big_rook_can_move_when_its_original_storage_cell_was_hit() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("a1", Player::White, PieceKind::King),
        ("a8", Player::Black, PieceKind::King),
        ("e7", Player::White, PieceKind::Knight),
        ("g8", Player::Black, PieceKind::BigRook),
    ]);
    let big = state.board.get("g8".parse().unwrap()).unwrap();
    state
        .card_runtime
        .insert_large_body(big.id, "g8".parse().unwrap(), PieceHealth::new(2));
    let first = engine
        .apply(&state, acquisition::movement("e7", "g8"))
        .unwrap()
        .state;
    assert_eq!(
        first.board.find_by_id(big.id).unwrap().0,
        "h8".parse().unwrap()
    );
    assert_eq!(
        first.card_runtime.large_missing_cells.get(&big.id),
        Some(&0)
    );
    let moved = engine
        .apply(&first, acquisition::movement("h8", "f8"))
        .unwrap()
        .state;
    assert_eq!(
        moved.card_runtime.large_anchor(big.id).unwrap(),
        "f8".parse().unwrap()
    );
    assert_eq!(
        moved.board.find_by_id(big.id).unwrap().0,
        "g8".parse().unwrap()
    );
    assert!(
        engine
            .board_piece_at(&moved, "f8".parse().unwrap())
            .is_none()
    );
    assert_eq!(
        engine
            .board_piece_at(&moved, "f7".parse().unwrap())
            .unwrap()
            .id,
        big.id
    );
}

#[test]
fn big_rook_landing_fractures_one_overlapping_cell_and_defeats_two() {
    let engine = Engine::bundled_latest().unwrap();
    for (origin, landing, survives) in [("c5", "e5", true), ("c4", "e4", false)] {
        let mut state = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            (origin, Player::White, PieceKind::BigRook),
            ("f4", Player::Black, PieceKind::BigRook),
        ]);
        let moving = state.board.get(origin.parse().unwrap()).unwrap();
        let target = state.board.get("f4".parse().unwrap()).unwrap();
        for (piece, square) in [(moving, origin), (target, "f4")] {
            state.card_runtime.insert_large_body(
                piece.id,
                square.parse().unwrap(),
                PieceHealth::new(2),
            );
        }
        let after = engine
            .apply(&state, acquisition::movement(origin, landing))
            .unwrap()
            .state;
        assert_eq!(after.board.find_by_id(target.id).is_some(), survives);
        assert_eq!(
            after
                .card_runtime
                .large_missing_cells
                .contains_key(&target.id),
            survives
        );
    }
}

#[test]
fn colossus_sector_counts_each_large_body_cell_in_the_same_attack() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("c4", Player::White, PieceKind::Colossus),
        ("f4", Player::Black, PieceKind::BigRook),
    ]);
    let colossus = state.board.get("c4".parse().unwrap()).unwrap();
    let target = state.board.get("f4".parse().unwrap()).unwrap();
    state
        .card_runtime
        .insert_large_body(target.id, "f4".parse().unwrap(), PieceHealth::new(2));
    let selection = ActionTarget::Squares(
        ["f4", "g4", "h4", "h3"]
            .into_iter()
            .map(|square| square.parse().unwrap())
            .collect(),
    );
    let action = Action::UsePieceAbility {
        piece_id: colossus.id,
        ability: PieceAbility::ColossusSector,
        target: selection.clone(),
    };
    let after = engine
        .apply_piece_ability(
            &state,
            action,
            colossus.id,
            PieceAbility::ColossusSector,
            selection,
        )
        .unwrap()
        .state;
    assert!(after.board.find_by_id(target.id).is_none());
}

#[test]
fn wizard_area_damage_uses_every_occupied_large_body_cell() {
    let engine = Engine::bundled_latest().unwrap();
    for (squares, survives) in [(vec!["f4"], true), (vec!["f4", "g4"], false)] {
        let mut state = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("f4", Player::Black, PieceKind::BigRook),
        ]);
        let big = state.board.get("f4".parse().unwrap()).unwrap();
        state
            .card_runtime
            .insert_large_body(big.id, "f4".parse().unwrap(), PieceHealth::new(2));
        state.card_runtime.delayed_hazards.push(DelayedHazard {
            kind: DelayedHazardKind::Meteor,
            owner: Player::White,
            trigger_after: Player::Black,
            squares: squares.into_iter().map(|s| s.parse().unwrap()).collect(),
        });
        engine.resolve_delayed_hazards(&mut state, Player::Black, &mut Vec::new());
        assert_eq!(state.board.find_by_id(big.id).is_some(), survives);
    }
}

#[test]
fn big_rook_castles_with_king_for_both_colors_and_replaces_allied_landing_pieces() {
    let engine = Engine::bundled_latest().unwrap();
    for (owner, king_from, king_to, rook_from, rook_to, pawn_square) in [
        (Player::White, "e1", "g1", "g2", "e2", "f2"),
        (Player::White, "e1", "c1", "a2", "d2", "e2"),
        (Player::Black, "e8", "g8", "g8", "e8", "f7"),
        (Player::Black, "e8", "c8", "a8", "d8", "e7"),
    ] {
        let enemy_king = if owner == Player::White {
            ("h8", Player::Black, PieceKind::King)
        } else {
            ("h1", Player::White, PieceKind::King)
        };
        let mut state = position(&[
            enemy_king,
            (king_from, owner, PieceKind::King),
            (rook_from, owner, PieceKind::BigRook),
            (pawn_square, owner, PieceKind::Pawn),
        ]);
        state.turn = owner;
        let rook = state.board.get(rook_from.parse().unwrap()).unwrap();
        let pawn = state.board.get(pawn_square.parse().unwrap()).unwrap();
        state.card_runtime.insert_large_body(
            rook.id,
            rook_from.parse().unwrap(),
            PieceHealth::new(2),
        );
        let action = acquisition::movement(king_from, king_to);
        assert!(
            engine.legal_actions(&state).unwrap().contains(&action),
            "{owner:?}"
        );
        let after = engine.apply(&state, action).unwrap().state;
        assert_eq!(
            after.board.get(king_to.parse().unwrap()).unwrap().kind,
            PieceKind::King
        );
        assert_eq!(
            after.card_runtime.large_anchor(rook.id).unwrap(),
            rook_to.parse().unwrap()
        );
        assert!(after.board.find_by_id(rook.id).is_some());
        assert!(after.board.find_by_id(pawn.id).is_none());
        assert!(after.card_runtime.moved_piece_ids.contains(&rook.id));
        let mut moved_rook = state.clone();
        moved_rook.card_runtime.moved_piece_ids.insert(rook.id);
        assert!(
            !engine
                .legal_actions(&moved_rook)
                .unwrap()
                .contains(&acquisition::movement(king_from, king_to))
        );
    }
}

#[test]
fn holdout_waits_28_fullmoves_and_keeps_identity_even_if_piece_type_changes() {
    let engine = Engine::bundled_latest().unwrap();
    for owner in [Player::White, Player::Black] {
        let mut before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", owner, PieceKind::Pawn),
        ]);
        before.turn = Player::White;
        let id = before.board.get("d4".parse().unwrap()).unwrap().id;
        let mut after = engine
            .acquire_for_setup(&before, crate::HOLDOUT, owner)
            .unwrap()
            .state;
        assert_eq!(after.rng_state(), before.rng_state());
        let due = 56 + u32::from(owner == Player::Black);
        assert_eq!(after.card_runtime.holdout_promotions[&id], due);
        after.turns_taken[1] = 27;
        after.turn = Player::Black;
        engine.resolve_holdout_promotions(&mut after, owner);
        assert_eq!(after.board.find_by_id(id).unwrap().1.kind, PieceKind::Pawn);
        after.board.get("d4".parse().unwrap()).unwrap();
        let mut changed = after.board.get("d4".parse().unwrap()).unwrap();
        changed.kind = PieceKind::Fanatic;
        after.board.set("d4".parse().unwrap(), Some(changed));
        after.turns_taken[1] = 28;
        after.turn = owner;
        engine.resolve_holdout_promotions(&mut after, owner);
        assert_eq!(after.board.find_by_id(id).unwrap().1.kind, PieceKind::Queen);
        assert!(!after.card_runtime.holdout_promotions.contains_key(&id));
    }
}

#[test]
fn holdout_promotion_retains_gambit_protection_and_triggers_spy() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("d1", Player::White, PieceKind::Queen),
        ("d2", Player::White, PieceKind::Pawn),
    ]);
    let pawn = before.board.get("d2".parse().unwrap()).unwrap();
    let state = engine
        .acquire_for_setup(&before, crate::QUEENS_GAMBIT, Player::White)
        .unwrap()
        .state;
    let mut state = engine
        .acquire_for_setup(&state, crate::HOLDOUT, Player::White)
        .unwrap()
        .state;
    state.card_runtime.spies.insert(pawn.id, Player::Black);
    state.turns_taken[1] = 28;
    engine.resolve_holdout_promotions(&mut state, Player::White);
    let promoted = state.board.find_by_id(pawn.id).unwrap().1;
    assert_eq!(
        (promoted.kind, promoted.owner),
        (PieceKind::Queen, Player::Black)
    );
    assert!(
        state
            .card_runtime
            .queens_gambit_protection
            .contains(&pawn.id)
    );
    assert!(state.card_runtime.protected_piece_ids.contains(&pawn.id));
    assert!(!state.card_runtime.spies.contains_key(&pawn.id));
}

#[test]
fn queens_gambit_sacrifices_selected_queen_and_protects_its_file_and_an_occupied_distant_file() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("d1", Player::White, PieceKind::Queen),
        ("d2", Player::White, PieceKind::Pawn),
        ("g2", Player::White, PieceKind::Pawn),
        ("e2", Player::White, PieceKind::Pawn),
    ]);
    let after = engine
        .acquire_for_setup(&before, crate::QUEENS_GAMBIT, Player::White)
        .unwrap();
    assert_eq!(after.state.rng_state(), before.rng_state());
    assert!(after.state.board.get("d1".parse().unwrap()).is_none());
    for square in ["d2", "g2"] {
        let p = after.state.board.get(square.parse().unwrap()).unwrap();
        assert!(after.state.card_runtime.protected_piece_ids.contains(&p.id));
    }
    let p = after.state.board.get("e2".parse().unwrap()).unwrap();
    assert!(!after.state.card_runtime.protected_piece_ids.contains(&p.id));
    assert!(
        !after
            .events
            .iter()
            .any(|e| matches!(e, Event::PieceCaptured { .. }))
    );
}

#[test]
fn queens_gambit_preserves_existing_protection_and_tracks_later_timed_protection() {
    let engine = Engine::bundled_latest().unwrap();
    let mut before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("d1", Player::White, PieceKind::Queen),
        ("d2", Player::White, PieceKind::Pawn),
        ("g2", Player::White, PieceKind::Pawn),
    ]);
    let permanent = before.board.get("d2".parse().unwrap()).unwrap();
    let timed = before.board.get("g2".parse().unwrap()).unwrap();
    before.card_runtime.protected_piece_ids.insert(permanent.id);
    before
        .card_runtime
        .remember_permanent_before_timed_protection(permanent.id);
    before.card_runtime.last_resistance.insert(
        permanent.id,
        TimedPieceStatus {
            by: Player::White,
            remaining_turns: 1,
        },
    );
    let mut after = engine
        .acquire_for_setup(&before, crate::QUEENS_GAMBIT, Player::White)
        .unwrap()
        .state;
    assert!(
        after
            .card_runtime
            .queens_gambit_protection
            .contains(&permanent.id)
    );
    assert!(
        after
            .card_runtime
            .queens_gambit_previous_protection
            .contains(&permanent.id)
    );
    let mut without_prior = after.clone();
    without_prior
        .card_runtime
        .queens_gambit_previous_protection
        .remove(&permanent.id);
    assert_ne!(after.position_key(), without_prior.position_key());
    assert_ne!(after.fingerprint(), without_prior.fingerprint());
    after.card_runtime.last_resistance.insert(
        timed.id,
        TimedPieceStatus {
            by: Player::White,
            remaining_turns: 1,
        },
    );
    engine.clear_queens_gambit_protection(&mut after, permanent.id);
    engine.clear_queens_gambit_protection(&mut after, timed.id);
    assert!(
        after
            .card_runtime
            .protected_piece_ids
            .contains(&permanent.id)
    );
    assert!(after.card_runtime.protected_piece_ids.contains(&timed.id));
    assert!(
        !after
            .card_runtime
            .queens_gambit_protection
            .contains(&timed.id)
    );
    engine.tick_middle_statuses_after_turn(&mut after, Player::White, &mut Vec::new());
    assert!(!after.card_runtime.protected_piece_ids.contains(&timed.id));
    assert!(
        after
            .card_runtime
            .protected_piece_ids
            .contains(&permanent.id)
    );
}

#[test]
fn merchant_only_bypasses_gambit_protection_when_no_other_protection_exists() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("e1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("d7", Player::Black, PieceKind::Pawn),
    ]);
    state = engine
        .acquire_for_setup(&state, crate::MERCHANT_GUILD, Player::White)
        .unwrap()
        .state;
    let merchant = state.board.get("e3".parse().unwrap()).unwrap();
    let pawn = state.board.get("d7".parse().unwrap()).unwrap();
    state.card_runtime.merchant_gold.insert(merchant.id, 2);
    state.card_runtime.protected_piece_ids.insert(pawn.id);
    state.card_runtime.queens_gambit_protection.insert(pawn.id);
    assert!(engine.current_merchant_target(&state, merchant, pawn));
    state
        .card_runtime
        .queens_gambit_previous_protection
        .insert(pawn.id);
    assert!(!engine.current_merchant_target(&state, merchant, pawn));
    state
        .card_runtime
        .queens_gambit_previous_protection
        .remove(&pawn.id);
    state.card_runtime.sacrifice_protection.insert(
        pawn.id,
        TimedPieceStatus {
            by: Player::Black,
            remaining_turns: 1,
        },
    );
    assert!(!engine.current_merchant_target(&state, merchant, pawn));
}

#[test]
fn d4_restricts_opponents_and_own_royal_for_both_colors() {
    let engine = Engine::bundled_latest().unwrap();
    for owner in [Player::White, Player::Black] {
        let before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
        ]);
        let after = engine
            .acquire_for_setup(&before, crate::D4, owner)
            .unwrap()
            .state;
        let square = (if owner == Player::White { "d4" } else { "d5" })
            .parse()
            .unwrap();
        assert!(engine.d4_restricts(&after, owner, true, square));
        assert!(!engine.d4_restricts(&after, owner, false, square));
        assert!(engine.d4_restricts(&after, owner.opponent(), false, square));
    }
    let before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("c4", Player::Black, PieceKind::Rook),
    ]);
    let mut after = engine
        .acquire_for_setup(&before, crate::D4, Player::White)
        .unwrap()
        .state;
    after.turn = Player::Black;
    assert!(
        !engine
            .legal_actions(&after)
            .unwrap()
            .contains(&acquisition::movement("c4", "d4"))
    );
}

#[test]
fn vanguard_respects_unique_front_pawn_and_large_body_occupancy() {
    let engine = Engine::bundled_latest().unwrap();
    let mut before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("d4", Player::White, PieceKind::Pawn),
        ("c6", Player::White, PieceKind::BigRook),
    ]);
    let big = before.board.get("c6".parse().unwrap()).unwrap();
    before
        .card_runtime
        .insert_large_body(big.id, "c6".parse().unwrap(), PieceHealth::new(2));
    let mut state = engine
        .acquire_for_setup(&before, crate::VANGUARD, Player::White)
        .unwrap()
        .state;
    let legal = engine.legal_actions(&state).unwrap();
    assert!(!legal.contains(&acquisition::movement("d4", "c5")));
    assert!(legal.contains(&acquisition::movement("d4", "e5")));
    state.board.set(
        "f4".parse().unwrap(),
        Some(Piece::new(Player::White, PieceKind::Pawn).with_id(99)),
    );
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .contains(&acquisition::movement("d4", "e5"))
    );
}

#[test]
fn false_start_can_cross_but_not_land_on_an_enemy_d4_square() {
    let engine = Engine::bundled_latest().unwrap();
    for (from, expected) in [("d3", "d4"), ("d4", "d6")] {
        let before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            (from, Player::White, PieceKind::Pawn),
        ]);
        let pawn = before.board.get(from.parse().unwrap()).unwrap();
        let state = engine
            .acquire_for_setup(&before, crate::D4, Player::Black)
            .unwrap()
            .state;
        let state = engine
            .acquire_for_setup(&state, crate::FALSE_START, Player::White)
            .unwrap()
            .state;
        assert_eq!(
            state.board.find_by_id(pawn.id).unwrap().0,
            expected.parse().unwrap()
        );
    }
}

#[test]
fn d4_blocks_deferred_lobster_and_scarecrow_arrivals() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
    ]);
    let mut state = engine
        .acquire_for_setup(&before, crate::D4, Player::Black)
        .unwrap()
        .state;
    let square = "d5".parse().unwrap();
    state
        .card_runtime
        .pending_lobsters
        .push(crate::PendingLobster {
            owner: Player::White,
            square,
            due_ply: 0,
        });
    engine.resolve_after_counted_move(&mut state, &mut Vec::new());
    assert!(state.board.get(square).is_none());
    assert!(state.card_runtime.pending_lobsters.is_empty());
    state
        .card_runtime
        .pending_scarecrows
        .push(crate::PendingScarecrow {
            owner: Player::White,
            square,
            remaining_ply: 1,
        });
    engine.tick_middle_statuses_after_turn(&mut state, Player::White, &mut Vec::new());
    assert!(state.board.get(square).is_none());
    assert!(state.card_runtime.pending_scarecrows.is_empty());
}

#[test]
fn calling_card_marks_a_nonroyal_and_removes_another_enemy_on_capture() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("d3", Player::White, PieceKind::Rook),
        ("d4", Player::Black, PieceKind::Rook),
        ("b7", Player::Black, PieceKind::Pawn),
    ]);
    let target = before.board.get("d4".parse().unwrap()).unwrap();
    let after = engine
        .acquire_for_setup(&before, crate::CALLING_CARD, Player::White)
        .unwrap()
        .state;
    assert_eq!(after.rng_state(), before.rng_state());
    assert_eq!(
        after.card_runtime.calling_cards.get(&target.id),
        Some(&Player::White)
    );
    let result = engine
        .apply(&after, acquisition::movement("d3", "d4"))
        .unwrap();
    assert!(result.state.board.get("b7".parse().unwrap()).is_none());
    assert!(
        result.events.iter().any(
            |e| matches!(e,Event::RandomSample{purpose,..} if purpose=="calling-card-followup")
        )
    );
}

#[test]
fn wanted_tracks_ranged_target_and_arrests_it_after_exposure_at_own_turn_end() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("c4", Player::Black, PieceKind::Rook),
        ("d4", Player::White, PieceKind::Pawn),
    ]);
    let rook = before.board.get("c4".parse().unwrap()).unwrap();
    let mut state = engine
        .acquire_for_setup(&before, crate::WANTED, Player::White)
        .unwrap()
        .state;
    assert_eq!(state.rng_state(), before.rng_state());
    assert!(state.card_runtime.wanted_piece_ids.contains(&rook.id));
    assert_ne!(state.position_key(), before.position_key());
    state.turn = Player::Black;
    let result = engine
        .apply(&state, acquisition::movement("c4", "c5"))
        .unwrap();
    assert!(result.state.board.find_by_id(rook.id).is_none());
    assert!(
        !result
            .state
            .card_runtime
            .wanted_piece_ids
            .contains(&rook.id)
    );
}

#[test]
fn iron_monarch_extra_move_is_bound_to_the_royal_and_may_be_skipped() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("e2", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("a2", Player::White, PieceKind::Rook),
        ("e3", Player::Black, PieceKind::Squire),
    ]);
    let mut state = engine
        .acquire_for_setup(&before, crate::IRON_MONARCH, Player::White)
        .unwrap()
        .state;
    state.card_runtime.platform = Some(PlatformRuleState {
        cell: Some("e3".parse().unwrap()),
        spawned_at: 0,
        next_at: 99,
        triggered_piece_ids: Default::default(),
        nonce: 0,
    });
    let after = engine
        .apply(&state, acquisition::movement("e2", "e3"))
        .unwrap()
        .state;
    assert_eq!(after.turn, Player::White);
    assert!(
        matches!(&after.phase, GamePhase::ForcedMove { source: ForcedMoveSource::Effect(id), .. } if id == crate::IRON_MONARCH)
    );
    let moves = engine.legal_actions(&after).unwrap();
    assert!(moves.contains(&Action::SkipOptionalMove));
    assert!(
        moves
            .iter()
            .filter_map(|a| if let Action::Move(m) = a {
                Some(m.from)
            } else {
                None
            })
            .all(|s| s == "e3".parse().unwrap())
    );
    let done = engine
        .apply(&after, Action::SkipOptionalMove)
        .unwrap()
        .state;
    assert_eq!(done.turn, Player::Black);
}
#[test]
fn thief_jump_grants_optional_extra_move_but_cannot_capture_over_a_hurdle() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("d4", Player::White, PieceKind::Queen),
        ("d5", Player::White, PieceKind::Pawn),
        ("d7", Player::Black, PieceKind::Rook),
    ]);
    let id = before.board.get("d4".parse().unwrap()).unwrap().id;
    let mut state = engine
        .acquire_for_setup(&before, crate::THIEF, Player::White)
        .unwrap()
        .state;
    state.card_runtime.platform = Some(PlatformRuleState {
        cell: Some("d6".parse().unwrap()),
        spawned_at: 0,
        next_at: 99,
        triggered_piece_ids: Default::default(),
        nonce: 0,
    });
    assert!(state.card_runtime.wanted_piece_ids.contains(&id));
    let legal = engine.legal_actions(&state).unwrap();
    assert!(legal.contains(&acquisition::movement("d4", "d6")));
    assert!(!legal.contains(&acquisition::movement("d4", "d7")));
    let after = engine
        .apply(&state, acquisition::movement("d4", "d6"))
        .unwrap()
        .state;
    assert_eq!(after.turn, Player::White);
    assert!(
        matches!(&after.phase, GamePhase::ForcedMove { source: ForcedMoveSource::Effect(id), .. } if id == crate::THIEF)
    );
    let extra = engine.legal_actions(&after).unwrap();
    assert!(!extra.contains(&acquisition::movement("d6", "d4")));
    assert!(!extra.contains(&acquisition::movement("d6", "d3")));
    assert!(extra.contains(&acquisition::movement("d6", "c6")));
    let mut no_path = after.clone();
    no_path.card_runtime.thief_turn_paths.clear();
    assert_ne!(after.position_key(), no_path.position_key());
    assert!(
        engine
            .legal_actions(&after)
            .unwrap()
            .contains(&Action::SkipOptionalMove)
    );
    let done = engine
        .apply(&after, Action::SkipOptionalMove)
        .unwrap()
        .state;
    assert_eq!(done.turn, Player::Black);
    assert!(done.card_runtime.thief_turn_paths.is_empty());
}

#[test]
fn merchant_stays_put_when_buying_and_uses_current_gold_and_combat_prices() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("e1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("a7", Player::Black, PieceKind::Pawn),
        ("b7", Player::Black, PieceKind::Knight),
    ]);
    let king = before.board.get("e1".parse().unwrap()).unwrap();
    let mut after = engine
        .acquire_for_setup(&before, crate::MERCHANT_GUILD, Player::White)
        .unwrap()
        .state;
    assert_eq!(after.board.get("e3".parse().unwrap()).unwrap().id, king.id);
    assert_eq!(after.card_runtime.merchant_gold[&king.id], 1);
    assert!(
        !engine
            .legal_actions(&after)
            .unwrap()
            .contains(&acquisition::movement("e3", "a7"))
    );
    after.card_runtime.merchant_gold.insert(king.id, 2);
    assert!(
        engine
            .legal_actions(&after)
            .unwrap()
            .contains(&acquisition::movement("e3", "a7"))
    );
    assert!(
        !engine
            .legal_actions(&after)
            .unwrap()
            .contains(&acquisition::movement("e3", "b7"))
    );
    let pawn = after.board.get("a7".parse().unwrap()).unwrap();
    let done = engine
        .apply(&after, acquisition::movement("e3", "a7"))
        .unwrap()
        .state;
    assert_eq!(done.board.get("e3".parse().unwrap()).unwrap().id, king.id);
    assert_eq!(done.board.get("a7".parse().unwrap()).unwrap().id, pawn.id);
    assert_eq!(
        done.board.get("a7".parse().unwrap()).unwrap().owner,
        Player::White
    );
    assert_eq!(done.card_runtime.merchant_gold[&king.id], 0);
    assert_eq!(done.turn, Player::Black);
}

#[test]
fn merchant_cannot_buy_shielded_royal_but_unshielded_purchase_wins() {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("e1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("a7", Player::Black, PieceKind::Pawn),
    ]);
    let merchant = before.board.get("e1".parse().unwrap()).unwrap();
    let royal = before.board.get("h8".parse().unwrap()).unwrap();
    let mut state = engine
        .acquire_for_setup(&before, crate::MERCHANT_GUILD, Player::White)
        .unwrap()
        .state;
    state.card_runtime.merchant_gold.insert(merchant.id, 20);
    state.card_runtime.democracy[Player::Black.index()] = true;
    state.card_runtime.shielded_piece_ids.insert(royal.id);
    let buy = acquisition::movement("e3", "h8");
    assert!(!engine.legal_actions(&state).unwrap().contains(&buy));
    state.card_runtime.shielded_piece_ids.remove(&royal.id);
    assert!(engine.legal_actions(&state).unwrap().contains(&buy));
    let done = engine.apply(&state, buy).unwrap().state;
    assert_eq!(
        done.status,
        GameStatus::Won {
            winner: Player::White,
            reason: WinReason::CardEffect("merchant-purchase".into()),
        }
    );
    assert!(done.board.get("h8".parse().unwrap()).is_none());
}

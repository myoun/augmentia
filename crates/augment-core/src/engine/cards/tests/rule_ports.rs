//! Current RULE contracts exercised through public Engine transitions.
use super::*;
use serde_json::Value;

fn fixtures() -> Value {
    serde_json::from_str(include_str!(
        "../../../../tests/fixtures/current-rules.json"
    ))
    .unwrap()
}
fn position(extra: &[(&str, Player, PieceKind)]) -> GameState {
    let mut board = Board::empty();
    for (s, owner, kind) in [
        ("b2", Player::White, PieceKind::King),
        ("g7", Player::Black, PieceKind::King),
    ]
    .into_iter()
    .chain(extra.iter().copied())
    {
        board.set(s.parse().unwrap(), Some(Piece::new(owner, kind)));
    }
    let mut state = GameState::from_board(
        crate::LATEST_RULESET_ID,
        Mode::Standard,
        board,
        Player::White,
        31,
    );
    state.completed_drafts = [true; 3];
    state
}
fn movement(a: &str, b: &str) -> Action {
    Action::Move(MoveAction::new(
        a.parse().unwrap(),
        b.parse().unwrap(),
        None,
    ))
}
fn raw_kind(value: &str) -> PieceKind {
    let name = match value {
        "eagle" => "Alibaba",
        "pegasus" => "Unicorn",
        other => {
            return serde_json::from_value(Value::String(format!(
                "{}{}",
                other[0..1].to_uppercase(),
                &other[1..]
            )))
            .unwrap();
        }
    };
    serde_json::from_value(Value::String(name.into())).unwrap()
}

#[test]
fn all_960_layouts_match_pinned_source_and_keep_piece_identities() {
    for (i, layout) in fixtures()["chess960"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let expected: Vec<_> = layout
            .as_array()
            .unwrap()
            .iter()
            .map(|v| raw_kind(v.as_str().unwrap()))
            .collect();
        assert_eq!(
            super::super::chess_960::back_rank(i).as_slice(),
            expected,
            "index {i}"
        );
    }
    let engine = Engine::bundled_latest().unwrap();
    let mut state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
    state.completed_drafts = [true; 3];
    let before: BTreeSet<_> = state.board.iter().map(|(_, p)| p.id).collect();
    let state = engine.apply_rule(&state, crate::CHESS_960).unwrap().state;
    assert_eq!(before, state.board.iter().map(|(_, p)| p.id).collect());
    for f in 0..8 {
        assert_eq!(
            state.board.get(Square::new(f, 0).unwrap()).unwrap().kind,
            state.board.get(Square::new(f, 7).unwrap()).unwrap().kind
        );
    }
}

#[test]
fn twelve_seeded_chaos_armies_match_complete_65_type_source_pool() {
    let fixtures = fixtures();
    assert_eq!(super::super::chess_n_pow_30::POOL.len(), 65);
    for case in fixtures["chaos"].as_array().unwrap() {
        let (value, armies) =
            super::super::chess_n_pow_30::seeded_armies(case["seed"].as_u64().unwrap() as u32)
                .unwrap();
        assert_eq!(u64::from(value), case["totalValue"].as_u64().unwrap());
        for (i, color) in ["white", "black"].into_iter().enumerate() {
            let expected: Vec<_> = case[color]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| raw_kind(v.as_str().unwrap()))
                .collect();
            assert_eq!(armies[i], expected, "seed {} {color}", case["seed"]);
        }
    }
}

#[test]
fn every_rule_installs_is_registered_and_rejects_duplicate_installation_atomically() {
    let engine = Engine::bundled_latest().unwrap();
    for id in [
        crate::ACCELERATION,
        crate::BLACK_HOLE,
        crate::CHESS_344200,
        crate::CHESS_960,
        crate::CHESS_N_POW_30,
        crate::CONVEYOR,
        crate::CROWN,
        crate::DIAGONAL_CHESS,
        crate::FOOTBALL,
        crate::HIGH_GROUND,
        crate::HIGHWAY,
        crate::MISTAKE,
        crate::MONOCHROME_CHESS,
        crate::MONSTER_RULE,
        crate::PERIODIC_COLLAPSE,
        crate::PORTAL,
        crate::REVELATION,
        crate::RULE_BOMBS,
        crate::WINTER_KINGDOM,
    ] {
        let mut base = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 31);
        base.completed_drafts = [true; 3];
        let fingerprint = base.fingerprint();
        let next = engine
            .apply_rule(&base, id)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(base.fingerprint(), fingerprint);
        assert!(next.state.applied_rules.iter().any(|r| r == id));
        assert!(engine.apply_rule(&next.state, id).is_err());
        engine.legal_actions(&next.state).unwrap();
    }
}

#[test]
fn home_shuffle_stays_in_each_army_and_diagonal_layout_matches_source() {
    let engine = Engine::bundled_latest().unwrap();
    let mut base = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 31);
    base.completed_drafts = [true; 3];
    let next = engine.apply_rule(&base, crate::CHESS_344200).unwrap().state;
    assert_eq!(next.board.iter().count(), 32);
    for (s, p) in next.board.iter() {
        assert!(if p.owner == Player::White {
            s.rank() <= 1
        } else {
            s.rank() >= 6
        });
    }
    let diagonal = engine
        .apply_rule(&base, crate::DIAGONAL_CHESS)
        .unwrap()
        .state;
    assert_eq!(diagonal.board.iter().count(), 30);
    for row in fixtures()["diagonal"].as_array().unwrap() {
        let p = diagonal
            .board
            .get(row[2].as_str().unwrap().parse().unwrap())
            .unwrap();
        assert_eq!(p.kind, raw_kind(row[1].as_str().unwrap()));
        assert_eq!(p.owner.to_string(), row[0].as_str().unwrap());
    }
}

#[test]
fn high_ground_highway_and_monochrome_change_legal_moves() {
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[
        ("d1", Player::White, PieceKind::Rook),
        ("d4", Player::Black, PieceKind::Pawn),
    ]);
    let state = engine.apply_rule(&base, crate::HIGH_GROUND).unwrap().state;
    assert_eq!(state.card_runtime.high_ground_squares.len(), 8);
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .contains(&movement("d1", "d4"))
    );
    let base = position(&[("b3", Player::White, PieceKind::Pawn)]);
    let highway = engine.apply_rule(&base, crate::HIGHWAY).unwrap().state;
    assert!(
        engine
            .legal_actions(&highway)
            .unwrap()
            .contains(&movement("b3", "b6"))
    );
    let base = position(&[("d4", Player::White, PieceKind::Knight)]);
    let mono = engine
        .apply_rule(&base, crate::MONOCHROME_CHESS)
        .unwrap()
        .state;
    assert_eq!(
        mono.board.get("d4".parse().unwrap()).unwrap().kind,
        PieceKind::Camel
    );
    for action in engine.legal_actions(&mono).unwrap() {
        if let Action::Move(m) = action {
            assert_eq!(
                (m.from.file() + m.from.rank()) % 2,
                (m.to.file() + m.to.rank()) % 2
            );
        }
    }
}

#[test]
fn portal_entry_captures_both_cells_and_slider_continues_after_exit() {
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[
        ("b1", Player::White, PieceKind::Knight),
        ("c3", Player::Black, PieceKind::Pawn),
        ("f6", Player::Black, PieceKind::Bishop),
    ]);
    let state = engine.apply_rule(&base, crate::PORTAL).unwrap().state;
    let next = engine.apply(&state, movement("b1", "c3")).unwrap();
    assert!(next.state.board.get("c3".parse().unwrap()).is_none());
    assert_eq!(
        next.state.board.get("f6".parse().unwrap()).unwrap().kind,
        PieceKind::Knight
    );
    assert_eq!(
        next.events
            .iter()
            .filter(|e| matches!(e, Event::PieceCaptured { .. }))
            .count(),
        2
    );
    let base = position(&[("c1", Player::White, PieceKind::Rook)]);
    let state = engine.apply_rule(&base, crate::PORTAL).unwrap().state;
    assert!(
        engine
            .legal_actions(&state)
            .unwrap()
            .contains(&movement("c1", "f7"))
    );
    assert!(
        engine
            .apply(&state, movement("c1", "f7"))
            .unwrap()
            .state
            .board
            .get("f7".parse().unwrap())
            .is_some()
    );
}

#[test]
fn conveyor_and_periodic_collapse_wait_for_completed_black_turn() {
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[("a1", Player::White, PieceKind::Rook)]);
    let state = engine.apply_rule(&base, crate::CONVEYOR).unwrap().state;
    let state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
    assert!(state.board.get("a1".parse().unwrap()).is_some());
    let state = engine.apply(&state, movement("g7", "f7")).unwrap().state;
    assert!(state.board.get("a2".parse().unwrap()).is_some());
    let mut base = position(&[("a1", Player::White, PieceKind::Rook)]);
    base.turns_taken = [19, 19];
    let state = engine
        .apply_rule(&base, crate::PERIODIC_COLLAPSE)
        .unwrap()
        .state;
    let state = engine.apply(&state, movement("b2", "c3")).unwrap().state;
    assert_eq!(state.rule_runtime.collapse_depth, 0);
    let state = engine.apply(&state, movement("g7", "f6")).unwrap().state;
    assert_eq!(state.rule_runtime.collapse_depth, 1);
    assert!(state.board.get("a1".parse().unwrap()).is_none());
}

#[test]
fn football_has_neutral_kick_actions_and_king_capture_uses_normal_terminal_checks() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = engine
        .apply_rule(
            &position(&[("d3", Player::White, PieceKind::Rook)]),
            crate::FOOTBALL,
        )
        .unwrap()
        .state;
    state.card_runtime.football_pos = Some("d4".parse().unwrap());
    assert!(
        engine
            .legal_actions(&state)
            .unwrap()
            .contains(&movement("d4", "d6"))
    );
    let batch = engine.legal_action_batch(&state).unwrap();
    let action = batch.iter().find(|a| **a == movement("d4", "d6")).unwrap();
    let next = batch.apply(action).unwrap();
    assert_eq!(
        next.state.card_runtime.football_pos,
        Some("d6".parse().unwrap())
    );
    assert!(next.state.board.get("d3".parse().unwrap()).is_some());
}

#[test]
fn winter_previews_six_nonroyals_then_freezes_and_disables_for_small_armies() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 31);
    state.completed_drafts = [true; 3];
    state.turns_taken = [2, 2];
    let state = engine
        .apply_rule(&state, crate::WINTER_KINGDOM)
        .unwrap()
        .state;
    let state = engine.apply(&state, movement("a2", "a3")).unwrap().state;
    assert_eq!(state.rule_runtime.winter_preview.len(), 6);
    assert!(state.rule_runtime.winter_frozen.is_empty());
    let mut state = engine.apply(&state, movement("a7", "a6")).unwrap().state;
    assert_eq!(state.rule_runtime.winter_frozen.len(), 6);
    assert!(state.rule_runtime.winter_frozen.iter().all(|id| {
        state
            .board
            .find_by_id(*id)
            .is_some_and(|(_, p)| !engine.is_royal_piece(&state, p))
    }));
    state.board = position(&[]).board;
    state.turns_taken = [5, 5];
    state.rule_runtime.winter_frozen.clear();
    let state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
    let state = engine.apply(&state, movement("g7", "f7")).unwrap().state;
    assert!(state.rule_runtime.winter_disabled);
    assert!(state.rule_runtime.winter_frozen.is_empty());
}

#[test]
fn material_revelation_waits_for_sixty_full_moves_and_rule_state_is_search_identity() {
    let engine = Engine::bundled_latest().unwrap();
    let mut base = position(&[("d3", Player::White, PieceKind::Queen)]);
    base.turns_taken = [59, 59];
    let state = engine.apply_rule(&base, crate::REVELATION).unwrap().state;
    let state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
    assert_eq!(state.status, GameStatus::Ongoing);
    let state = engine.apply(&state, movement("g7", "f7")).unwrap().state;
    assert!(matches!(
        state.status,
        GameStatus::Won {
            winner: Player::White,
            ..
        }
    ));
    let state = engine
        .apply_rule(&position(&[]), crate::ACCELERATION)
        .unwrap()
        .state;
    let mut changed = state.clone();
    changed.rule_runtime.acceleration_after = Some(123);
    assert_ne!(state.position_key(), changed.position_key());
}

#[test]
fn acceleration_matches_source_delays_and_extra_actions_do_not_end_a_turn() {
    let engine = Engine::bundled_latest().unwrap();
    for case in fixtures()["acceleration"].as_array().unwrap() {
        let mut base = position(&[]);
        base.turns_taken = [case["black"].as_u64().unwrap() as u32; 2];
        base.turn = if case["turn"] == "white" {
            Player::White
        } else {
            Player::Black
        };
        let state = engine.apply_rule(&base, crate::ACCELERATION).unwrap().state;
        assert_eq!(
            state.rule_runtime.acceleration_after,
            Some(case["due"].as_u64().unwrap() as u32)
        );
    }
    let mut state = engine
        .apply_rule(&position(&[]), crate::ACCELERATION)
        .unwrap()
        .state;
    state.turns_taken = [2, 2];
    let state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
    assert_eq!(state.turn, Player::Black);
    assert_eq!(state.actions_remaining, 2);
    let state = engine.apply(&state, movement("g7", "f7")).unwrap().state;
    assert_eq!(state.turn, Player::Black);
    assert_eq!(state.turns_taken[1], 2);
    let state = engine.apply(&state, movement("f7", "f6")).unwrap().state;
    assert_eq!(state.turn, Player::White);
    assert_eq!(state.turns_taken[1], 3);
    assert_eq!(state.actions_remaining, 2);
}

#[test]
fn black_holes_remove_initial_occupants_and_new_arrivals() {
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[
        ("d4", Player::White, PieceKind::Bishop),
        ("e3", Player::White, PieceKind::Rook),
    ]);
    let state = engine.apply_rule(&base, crate::BLACK_HOLE).unwrap().state;
    assert!(state.board.get("d4".parse().unwrap()).is_none());
    let state = engine.apply(&state, movement("e3", "e4")).unwrap().state;
    assert!(state.board.get("e4".parse().unwrap()).is_none());
    assert_eq!(state.status, GameStatus::Ongoing);
}

#[test]
fn crown_is_immobile_transfers_on_capture_and_counts_full_moves() {
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[
        ("d2", Player::White, PieceKind::Rook),
        ("d7", Player::Black, PieceKind::Rook),
    ]);
    let mut state = engine.apply_rule(&base, crate::CROWN).unwrap().state;
    state.card_runtime.crown_pos = Some("d4".parse().unwrap());
    let state = engine.apply(&state, movement("d2", "d4")).unwrap().state;
    let holder = state.board.get("d4".parse().unwrap()).unwrap();
    assert_eq!(holder.kind, PieceKind::Crown);
    assert_eq!(state.rule_runtime.crown_count, 0);
    let state = engine.apply(&state, movement("d7", "d4")).unwrap().state;
    let holder = state.board.get("d4".parse().unwrap()).unwrap();
    assert_eq!(holder.kind, PieceKind::Crown);
    assert_eq!(holder.owner, Player::Black);
    assert_eq!(state.rule_runtime.crown_count, 0);
    let mut state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .iter()
            .any(|a| matches!(a, Action::Move(m) if m.from == "d4".parse().unwrap()))
    );
    state.turns_taken = [10, 10];
    state.rule_runtime.crown_count = 9;
    state.rule_runtime.crown_last_fullmove = 10;
    state.actions_remaining = 2;
    let state = engine.apply(&state, movement("g7", "f7")).unwrap().state;
    assert_eq!(state.rule_runtime.crown_count, 9);
    let state = engine.apply(&state, movement("f7", "f6")).unwrap().state;
    assert!(matches!(
        state.status,
        GameStatus::Won {
            winner: Player::Black,
            ..
        }
    ));
}

#[test]
fn bombs_choose_distinct_empty_cells_and_explode_a_whole_cross() {
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[("d4", Player::White, PieceKind::Pawn)]);
    let state = engine.apply_rule(&base, crate::RULE_BOMBS).unwrap().state;
    assert_eq!(state.card_runtime.rule_bombs.len(), 3);
    assert!(
        state
            .card_runtime
            .rule_bombs
            .iter()
            .all(|s| matches!(s.rank(), 3 | 4) && s.file() != 4 && state.board.get(*s).is_none())
    );
    let mut state = engine
        .apply_rule(
            &position(&[
                ("d3", Player::White, PieceKind::Rook),
                ("a4", Player::Black, PieceKind::Bishop),
                ("d8", Player::Black, PieceKind::Knight),
            ]),
            crate::RULE_BOMBS,
        )
        .unwrap()
        .state;
    state.card_runtime.rule_bombs = BTreeSet::from(["d4".parse().unwrap()]);
    state.rule_runtime.bomb_ignore.clear();
    let state = engine.apply(&state, movement("d3", "d4")).unwrap().state;
    for square in ["d4", "a4", "d8"] {
        assert!(state.board.get(square.parse().unwrap()).is_none());
    }
    assert!(state.card_runtime.rule_bombs.is_empty());
}

#[test]
fn portal_blocks_original_ray_and_falls_back_when_exit_is_occupied_by_an_ally() {
    let engine = Engine::bundled_latest().unwrap();
    let state = engine
        .apply_rule(
            &position(&[("c1", Player::White, PieceKind::Rook)]),
            crate::PORTAL,
        )
        .unwrap()
        .state;
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .contains(&movement("c1", "c4"))
    );
    let state = engine
        .apply_rule(
            &position(&[
                ("c1", Player::White, PieceKind::Rook),
                ("f6", Player::White, PieceKind::Pawn),
            ]),
            crate::PORTAL,
        )
        .unwrap()
        .state;
    let state = engine.apply(&state, movement("c1", "c3")).unwrap().state;
    assert_eq!(
        state.board.get("c3".parse().unwrap()).unwrap().kind,
        PieceKind::Rook
    );
    assert_eq!(
        state.board.get("f6".parse().unwrap()).unwrap().kind,
        PieceKind::Pawn
    );
}

#[test]
fn monster_is_neutral_and_moves_once_every_three_completed_halfmoves() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = engine
        .apply_rule(&position(&[]), crate::MONSTER_RULE)
        .unwrap()
        .state;
    let id = state.card_runtime.monster_rule.unwrap();
    let start = state.board.find_by_id(id).unwrap().0;
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .iter()
            .any(|a| matches!(a, Action::Move(m) if m.from == start))
    );
    state.actions_remaining = 2;
    let state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
    assert_eq!(state.board.find_by_id(id).unwrap().0, start);
    let state = engine.apply(&state, movement("c2", "c1")).unwrap().state;
    assert_eq!(state.board.find_by_id(id).unwrap().0, start);
    let state = engine.apply(&state, movement("g7", "f7")).unwrap().state;
    assert_eq!(state.board.find_by_id(id).unwrap().0, start);
    let state = engine.apply(&state, movement("c1", "b1")).unwrap().state;
    assert_ne!(state.board.find_by_id(id).unwrap().0, start);
}

#[test]
fn source_don_routes_and_combat_values_match() {
    let fixture = fixtures();
    let square = |v: &Value| {
        Square::new(
            v["col"].as_u64().unwrap() as u8,
            7 - v["row"].as_u64().unwrap() as u8,
        )
        .unwrap()
    };
    for route in fixture["don_routes"].as_array().unwrap() {
        let targets: Vec<_> = route["targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(square)
            .collect();
        let expected: Vec<_> = route["path"]
            .as_array()
            .unwrap()
            .iter()
            .map(square)
            .collect();
        assert_eq!(
            super::super::chaos_pieces::don_path(
                square(&route["from"]),
                &targets,
                &BTreeSet::new(),
                false
            )
            .unwrap(),
            expected
        );
    }
    for (name, value) in fixture["combat_values"].as_object().unwrap() {
        if ["core", "colossusCore", "colossus-core"].contains(&name.as_str()) {
            continue;
        }
        assert_eq!(
            u64::from(super::super::revelation::combat_value(raw_kind(name))),
            value.as_u64().unwrap(),
            "{name}"
        );
    }
    for name in fixture["minor_types"].as_array().unwrap() {
        assert!(crate::latest::latest_minor_kind(raw_kind(
            name.as_str().unwrap()
        )));
    }
    for name in fixture["major_types"].as_array().unwrap() {
        assert!(crate::latest::latest_major_kind(raw_kind(
            name.as_str().unwrap()
        )));
    }
}

#[test]
fn chaos_dependency_pull_tower_barrier_and_medium_have_executable_moves() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("d2", Player::White, PieceKind::Grappler),
        ("d6", Player::Black, PieceKind::Rook),
        ("a3", Player::White, PieceKind::TransmissionTower),
        ("h7", Player::Black, PieceKind::Barrier),
    ]);
    state
        .rule_runtime
        .enabled
        .insert(crate::CHESS_N_POW_30.into());
    assert!(
        engine
            .legal_actions(&state)
            .unwrap()
            .contains(&movement("d2", "a4"))
    );
    let state = engine.apply(&state, movement("d2", "d6")).unwrap().state;
    assert_eq!(
        state.board.get("d2".parse().unwrap()).unwrap().kind,
        PieceKind::Grappler
    );
    let rook = state.board.get("d3".parse().unwrap()).unwrap();
    assert_eq!(rook.kind, PieceKind::Rook);
    assert!(state.rule_runtime.grappler_bound.contains_key(&rook.id));
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .iter()
            .any(|a| matches!(a, Action::Move(m) if m.from == "d3".parse().unwrap()))
    );
    let state = engine.apply(&state, movement("g7", "f7")).unwrap().state;
    assert!(state.rule_runtime.grappler_bound.is_empty());
    let mut state = position(&[
        ("g1", Player::White, PieceKind::Rook),
        ("h7", Player::Black, PieceKind::Barrier),
    ]);
    state
        .rule_runtime
        .enabled
        .insert(crate::CHESS_N_POW_30.into());
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .contains(&movement("g1", "g7"))
    );
    let mut state = position(&[
        ("d2", Player::White, PieceKind::Medium),
        ("a1", Player::White, PieceKind::Rook),
        ("a5", Player::Black, PieceKind::Bishop),
    ]);
    state
        .rule_runtime
        .enabled
        .insert(crate::CHESS_N_POW_30.into());
    let state = engine.apply(&state, movement("a1", "a5")).unwrap().state;
    let state = engine.apply(&state, movement("g7", "f7")).unwrap().state;
    assert!(
        engine
            .legal_actions(&state)
            .unwrap()
            .contains(&movement("d2", "f4"))
    );
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .contains(&movement("d2", "d4"))
    );
}

#[test]
fn rule_position_keys_normalize_inert_clocks_but_distinguish_scheduled_outcomes() {
    let engine = Engine::bundled_latest().unwrap();
    let state = engine
        .apply_rule(&position(&[]), crate::HIGHWAY)
        .unwrap()
        .state;
    let mut later = state.clone();
    later.turns_taken = [10; 2];
    assert_eq!(state.position_key(), later.position_key());
    let state = engine
        .apply_rule(&position(&[]), crate::REVELATION)
        .unwrap()
        .state;
    let mut later = state.clone();
    later.turns_taken = [10; 2];
    assert_ne!(state.position_key(), later.position_key());
}

#[test]
fn mistake_uses_twenty_percent_and_preserves_remaining_actions() {
    let engine = Engine::bundled_latest().unwrap();
    let mut outcomes = [false; 2];
    for seed in 0..32 {
        let mut state = position(&[
            ("a1", Player::White, PieceKind::Rook),
            ("a5", Player::Black, PieceKind::Bishop),
        ]);
        *state.rng_mut() = crate::DeterministicRng::new(seed);
        state.actions_remaining = 2;
        let state = engine.apply_rule(&state, crate::MISTAKE).unwrap().state;
        let mut expected = crate::DeterministicRng::new(seed);
        let (_, reversed) = expected.web_probability(1, 5);
        outcomes[usize::from(reversed)] = true;
        let next = engine.apply(&state, movement("a1", "a5")).unwrap();
        let square = if reversed { "a1" } else { "a5" };
        assert_eq!(
            next.state.board.get(square.parse().unwrap()).unwrap().owner,
            if reversed {
                Player::Black
            } else {
                Player::White
            }
        );
        assert_eq!(next.state.turn, Player::White);
        assert_eq!(next.state.actions_remaining, 1);
        assert_eq!(next.state.rng_state(), expected.state());
    }
    assert_eq!(outcomes, [true; 2]);
}

#[test]
fn full_board_neutral_spawn_failure_is_atomic() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[]);
    for index in 0..64 {
        let square = Square::from_index(index).unwrap();
        if state.board.get(square).is_none() {
            let p = Piece::new(Player::White, PieceKind::Pawn).with_id(state.allocate_piece_id());
            state.board.set(square, Some(p));
        }
    }
    let before = state.fingerprint();
    for id in [crate::FOOTBALL, crate::MONSTER_RULE] {
        assert!(engine.apply_rule(&state, id).is_err());
        assert_eq!(state.fingerprint(), before);
    }
}

#[test]
fn berolina_double_step_exposes_its_midpoint_to_en_passant_and_promotes() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("c2", Player::White, PieceKind::BerolinaPawn),
        ("c4", Player::Black, PieceKind::Pawn),
    ]);
    state
        .rule_runtime
        .enabled
        .insert(crate::CHESS_N_POW_30.into());
    let state = engine.apply(&state, movement("c2", "e4")).unwrap().state;
    assert_eq!(state.rule_runtime.berolina_rights.len(), 1);
    let state = engine.apply(&state, movement("c4", "d3")).unwrap().state;
    assert!(state.board.get("e4".parse().unwrap()).is_none());
    assert_eq!(
        state.board.get("d3".parse().unwrap()).unwrap().kind,
        PieceKind::Pawn
    );
    assert!(state.rule_runtime.berolina_rights.is_empty());
    let mut state = position(&[("b7", Player::White, PieceKind::BerolinaPawn)]);
    state
        .rule_runtime
        .enabled
        .insert(crate::CHESS_N_POW_30.into());
    let state = engine
        .apply(
            &state,
            Action::Move(MoveAction::new(
                "b7".parse().unwrap(),
                "a8".parse().unwrap(),
                Some(PieceKind::Queen),
            )),
        )
        .unwrap()
        .state;
    assert_eq!(
        state.board.get("a8".parse().unwrap()).unwrap().kind,
        PieceKind::Queen
    );
}

#[test]
fn revolving_door_rotates_neighbors_and_killer_automatically_captures_a_royal() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("d4", Player::White, PieceKind::RevolvingDoor),
        ("c5", Player::White, PieceKind::Pawn),
    ]);
    state
        .rule_runtime
        .enabled
        .insert(crate::CHESS_N_POW_30.into());
    let state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
    assert!(state.board.get("c5".parse().unwrap()).is_none());
    assert_eq!(
        state.board.get("d5".parse().unwrap()).unwrap().kind,
        PieceKind::Pawn
    );
    let mut state = position(&[("d4", Player::White, PieceKind::Killer)]);
    let king = state.board.get("g7".parse().unwrap()).unwrap();
    state.board.set("g7".parse().unwrap(), None);
    state.board.set("f6".parse().unwrap(), Some(king));
    state
        .rule_runtime
        .enabled
        .insert(crate::CHESS_N_POW_30.into());
    let state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
    assert!(matches!(
        state.status,
        GameStatus::Won {
            winner: Player::White,
            ..
        }
    ));
    assert_eq!(
        state.board.get("f6".parse().unwrap()).unwrap().kind,
        PieceKind::Killer
    );
}

#[test]
fn beacon_removes_six_neighbors_and_wins_on_its_third_owner_start() {
    let engine = Engine::bundled_latest().unwrap();
    let mut entries = vec![("d4", Player::White, PieceKind::Beacon)];
    entries.extend(
        ["c3", "c4", "c5", "d3", "d5", "e3"]
            .into_iter()
            .map(|s| (s, Player::White, PieceKind::Pawn)),
    );
    let mut state = position(&entries);
    state
        .rule_runtime
        .enabled
        .insert(crate::CHESS_N_POW_30.into());
    let mut state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
    assert_eq!(state.board.iter().count(), 3);
    assert_eq!(state.rule_runtime.beacon_due.len(), 1);
    assert_eq!(state.status, GameStatus::Ongoing);
    state.turns_taken = [2; 2];
    let state = engine.apply(&state, movement("g7", "f7")).unwrap().state;
    assert!(
        matches!(state.status, GameStatus::Won {winner: Player::White, reason: WinReason::CardEffect(ref id)} if id == "beacon")
    );
}

#[test]
fn don_quixote_automatically_steps_and_rampages_through_windmills() {
    let engine = Engine::bundled_latest().unwrap();
    for windmill in [false, true] {
        let mut pieces = vec![("a1", Player::White, PieceKind::DonQuixote)];
        if windmill {
            pieces.push(("h8", Player::Black, PieceKind::Windmill));
        }
        let mut state = position(&pieces);
        let id = state.board.get("a1".parse().unwrap()).unwrap().id;
        state
            .rule_runtime
            .enabled
            .insert(crate::CHESS_N_POW_30.into());
        let state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
        let state = engine.apply(&state, movement("g7", "f7")).unwrap().state;
        assert_ne!(state.board.find_by_id(id).unwrap().0, "a1".parse().unwrap());
        assert!(
            !state
                .board
                .iter()
                .any(|(_, p)| p.kind == PieceKind::Windmill)
        );
        assert_eq!(state.turn, Player::White);
        assert_eq!(state.turns_taken, [1; 2]);
    }
}

#[test]
fn mistake_also_reverses_checker_jumps_and_the_first_portal_capture() {
    let engine = Engine::bundled_latest().unwrap();
    let seed = (0..100)
        .find(|seed| crate::DeterministicRng::new(*seed).web_probability(1, 5).1)
        .unwrap();
    let mut state = position(&[
        ("d3", Player::White, PieceKind::Checker),
        ("e4", Player::Black, PieceKind::Pawn),
    ]);
    *state.rng_mut() = crate::DeterministicRng::new(seed);
    let state = engine.apply_rule(&state, crate::MISTAKE).unwrap().state;
    let state = engine.apply(&state, movement("d3", "f5")).unwrap().state;
    assert_eq!(
        state.board.get("d3".parse().unwrap()).unwrap().owner,
        Player::Black
    );
    assert!(state.board.get("e4".parse().unwrap()).is_none());
    assert!(state.board.get("f5".parse().unwrap()).is_none());
    let mut state = position(&[
        ("b1", Player::White, PieceKind::Knight),
        ("c3", Player::Black, PieceKind::Pawn),
        ("f6", Player::Black, PieceKind::Bishop),
    ]);
    *state.rng_mut() = crate::DeterministicRng::new(seed);
    let state = engine.apply_rule(&state, crate::MISTAKE).unwrap().state;
    let state = engine.apply_rule(&state, crate::PORTAL).unwrap().state;
    let state = engine.apply(&state, movement("b1", "c3")).unwrap().state;
    assert_eq!(
        state.board.get("b1".parse().unwrap()).unwrap().kind,
        PieceKind::Pawn
    );
    assert!(state.board.get("c3".parse().unwrap()).is_none());
    assert_eq!(
        state.board.get("f6".parse().unwrap()).unwrap().kind,
        PieceKind::Bishop
    );
}

#[test]
fn football_hits_a_shield_without_removing_the_target_or_moving_the_ball() {
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[
        ("d3", Player::White, PieceKind::Rook),
        ("d6", Player::Black, PieceKind::Bishop),
    ]);
    let mut state = engine.apply_rule(&base, crate::FOOTBALL).unwrap().state;
    state.card_runtime.football_pos = Some("d4".parse().unwrap());
    let victim = state.board.get("d6".parse().unwrap()).unwrap();
    state.card_runtime.shielded_piece_ids.insert(victim.id);
    let state = engine.apply(&state, movement("d4", "d6")).unwrap().state;
    assert!(state.board.find_by_id(victim.id).is_some());
    assert!(!state.card_runtime.shielded_piece_ids.contains(&victim.id));
    assert_eq!(state.card_runtime.football_pos, Some("d4".parse().unwrap()));
    assert_eq!(state.turn, Player::Black);
}

#[test]
fn environmental_loss_of_both_royals_is_a_draw() {
    let engine = Engine::bundled_latest().unwrap();
    let mut board = Board::empty();
    board.set(
        "d4".parse().unwrap(),
        Some(Piece::new(Player::White, PieceKind::King)),
    );
    board.set(
        "e4".parse().unwrap(),
        Some(Piece::new(Player::Black, PieceKind::King)),
    );
    let state = GameState::from_board(
        crate::LATEST_RULESET_ID,
        Mode::Standard,
        board,
        Player::White,
        31,
    );
    assert!(matches!(
        engine
            .apply_rule(&state, crate::BLACK_HOLE)
            .unwrap()
            .state
            .status,
        GameStatus::Draw { .. }
    ));
}

#[test]
fn native_dependency_behaviors_work_in_edited_positions_without_a_chaos_rule() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(&[
        ("d4", Player::White, PieceKind::RevolvingDoor),
        ("c5", Player::White, PieceKind::Pawn),
        ("a3", Player::White, PieceKind::TransmissionTower),
    ]);
    assert!(
        engine
            .legal_actions(&state)
            .unwrap()
            .contains(&movement("d4", "a4"))
    );
    let state = engine.apply(&state, movement("b2", "c2")).unwrap().state;
    assert_eq!(
        state.board.get("d5".parse().unwrap()).unwrap().kind,
        PieceKind::Pawn
    );
}

#[test]
fn portal_capture_carries_the_crown_to_its_actual_exit() {
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[
        ("b1", Player::White, PieceKind::Knight),
        ("c3", Player::Black, PieceKind::Rook),
    ]);
    let mut state = engine.apply_rule(&base, crate::CROWN).unwrap().state;
    state.card_runtime.crown_pos = Some("c3".parse().unwrap());
    engine.settle_rule_crown(&mut state, false);
    let state = engine.apply_rule(&state, crate::PORTAL).unwrap().state;
    let state = engine.apply(&state, movement("b1", "c3")).unwrap().state;
    let holder = state.board.get("f6".parse().unwrap()).unwrap();
    assert_eq!(holder.kind, PieceKind::Crown);
    assert_eq!(state.rule_runtime.crown_id, Some(holder.id));
    assert!(state.card_runtime.crown_pos.is_none());
    let mut delayed = state.clone();
    delayed.rule_runtime.crown_last_fullmove += 1;
    assert_ne!(state.position_key(), delayed.position_key());
}

#[test]
fn medium_samples_a_simultaneous_bomb_capture_batch_instead_of_always_remembering_the_last_victim()
{
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[
        ("h1", Player::White, PieceKind::Medium),
        ("d3", Player::White, PieceKind::Rook),
        ("a4", Player::Black, PieceKind::Bishop),
        ("d8", Player::Black, PieceKind::Knight),
    ]);
    let mut base = engine.apply_rule(&base, crate::RULE_BOMBS).unwrap().state;
    base.card_runtime.rule_bombs = BTreeSet::from(["d4".parse().unwrap()]);
    base.rule_runtime.bomb_ignore.clear();
    let mut remembered = BTreeSet::new();
    for seed in 0..32 {
        let mut state = base.clone();
        *state.rng_mut() = crate::DeterministicRng::new(seed);
        let next = engine.apply(&state, movement("d3", "d4")).unwrap();
        remembered.insert(next.state.rule_runtime.medium_memory.unwrap().effective);
        assert_eq!(next.events.iter().filter(|e| matches!(e, Event::RandomSample {purpose, ..} if purpose == "rule.medium.capture-batch")).count(), 1);
        assert!(state.rule_runtime.medium_memory.is_none());
    }
    assert_eq!(
        remembered,
        BTreeSet::from([PieceKind::Rook, PieceKind::Bishop, PieceKind::Knight])
    );
}

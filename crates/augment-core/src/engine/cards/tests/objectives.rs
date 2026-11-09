use super::*;
#[test]
pub(in crate::engine::cards) fn objective_geometry_matches_pinned_upstream_fixtures() {
    let e = Engine::bundled_latest().unwrap();
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/effects-objectives.json"
    ))
    .unwrap();
    for case in data["masks"].as_array().unwrap() {
        let kind: PieceKind = serde_json::from_value(case["kind"].clone()).unwrap();
        let mut s = position(&[("d4", Player::White, kind)]);
        let piece = at(&s, "d4");
        s.card_runtime.reversal[0] = case["flags"]["reversal"].as_bool().unwrap_or(false);
        s.card_runtime.magic_girl_surge[0] = case["flags"]["surge"].as_bool().unwrap_or(false);
        if case["flags"]["queen"] == true {
            s.board.set(
                "e1".parse().unwrap(),
                Some(Piece::new(Player::White, PieceKind::Queen).with_id(100)),
            );
        }
        if case["flags"]["army"] == true {
            for i in 0..6 {
                s.board.set(
                    Square::new(i, 1).unwrap(),
                    Some(Piece::new(Player::White, PieceKind::Pawn).with_id(100 + i as u32)),
                );
            }
        }
        if case["learned"] == "knight" {
            s.card_runtime
                .trickster_move_kinds
                .insert(piece.id, PieceKind::Knight);
        }
        if let Some(loss) = case["loss"].as_u64() {
            s.card_runtime
                .effects
                .grudge_losses
                .insert(piece.id, loss as u8);
        }
        assert_eq!(
            grudge_movement_mask(piece, &s.board, &s.card_runtime, 0),
            case["mask"].as_u64().unwrap() as u8,
            "{case}"
        );
    }
    for case in data["restrictions"].as_array().unwrap() {
        let kind: PieceKind = serde_json::from_value(case["kind"].clone()).unwrap();
        let mut s = position(&[("d4", Player::White, kind)]);
        let piece = at(&s, "d4");
        s.card_runtime
            .effects
            .grudge_losses
            .insert(piece.id, case["loss"].as_u64().unwrap() as u8);
        let mut actions = (0..64)
            .map(|i| {
                Action::Move(MoveAction::new(
                    "d4".parse().unwrap(),
                    Square::from_index(i).unwrap(),
                    None,
                ))
            })
            .collect();
        e.filter_card_state_moves(&s, "d4".parse().unwrap(), piece, &mut actions);
        let actual: BTreeSet<_> = actions
            .into_iter()
            .filter_map(|a| {
                if let Action::Move(m) = a {
                    Some(m.to.to_string())
                } else {
                    None
                }
            })
            .collect();
        let expected: BTreeSet<_> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect();
        assert_eq!(actual, expected, "{case}");
    }
    for case in data["graphs"].as_array().unwrap() {
        let pawns = case["squares"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str().unwrap().parse().unwrap(), i as u32 + 1))
            .collect();
        let horde = case["horde"].as_str().map(|s| {
            if s == "white" {
                Player::White
            } else {
                Player::Black
            }
        });
        let expected: Vec<[u32; 2]> = serde_json::from_value(case["expected"].clone()).unwrap();
        assert_eq!(pawn_links(pawns, horde), expected, "{case}");
    }
}

#[test]
pub(in crate::engine::cards) fn swaps_bypass_grudge_geometry_but_check_both_pawn_chain_destinations()
 {
    let e = Engine::bundled_latest().unwrap();
    let mut s = position(&[
        ("a2", Player::White, PieceKind::Dragon),
        ("c2", Player::White, PieceKind::Pawn),
        ("d4", Player::White, PieceKind::Pawn),
    ]);
    let dragon = at(&s, "a2");
    s.card_runtime.effects.grudge_losses.insert(dragon.id, 4);
    s.card_runtime
        .effects
        .pawn_chain_links
        .push([at(&s, "c2").id, at(&s, "d4").id]);
    assert!(!has_move(&e, &s, "a2", "b4"));
    assert!(!has_move(&e, &s, "a2", "c2"));
    assert!(has_move(&e, &s, "a2", "d4"));
}
pub(in crate::engine::cards) fn position(extra: &[(&str, Player, PieceKind)]) -> GameState {
    let mut board = crate::Board::empty();
    for (s, o, k) in [
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
    ]
    .iter()
    .chain(extra)
    {
        board.set(s.parse().unwrap(), Some(Piece::new(*o, *k)));
    }
    let mut s = GameState::from_board(
        crate::LATEST_RULESET_ID,
        Mode::Standard,
        board,
        Player::White,
        17,
    );
    s.completed_drafts = [true; 3];
    s
}
pub(in crate::engine::cards) fn at(s: &GameState, q: &str) -> Piece {
    s.board.get(q.parse().unwrap()).unwrap()
}
pub(in crate::engine::cards) fn movement(a: &str, b: &str) -> Action {
    Action::Move(MoveAction::new(
        a.parse().unwrap(),
        b.parse().unwrap(),
        None,
    ))
}
pub(in crate::engine::cards) fn has_move(e: &Engine, s: &GameState, a: &str, b: &str) -> bool {
    e.legal_actions(s).unwrap().contains(&movement(a, b))
}
pub(in crate::engine::cards) fn arm(e: &Engine, s: &mut GameState) {
    *s = e
        .grant_card(s, 1, crate::PROPHECY, Player::White)
        .unwrap()
        .state;
    *s = e
        .apply(
            s,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None,
            },
        )
        .unwrap()
        .state;
}

#[test]
pub(in crate::engine::cards) fn royal_check_respects_actual_identity_immunity_and_checker_jump_geometry()
 {
    let e = Engine::bundled_latest().unwrap();
    let mut s = position(&[("h2", Player::White, PieceKind::Rook)]);
    let king = at(&s, "h8");
    s.card_runtime.frozen.insert(
        king.id,
        FrozenStatus {
            source: Player::White,
            remaining_turns: 2,
        },
    );
    assert!(!e.card_royal_checked(&s, Player::White));
    s.card_runtime.frozen.clear();
    s.card_runtime.shielded_piece_ids.insert(king.id);
    assert!(e.card_royal_checked(&s, Player::White));
    let mut checker = position(&[("g5", Player::White, PieceKind::Checker)]);
    let king = at(&checker, "h8");
    checker.board.set("h8".parse().unwrap(), None);
    checker.board.set("f6".parse().unwrap(), Some(king));
    assert!(e.card_royal_checked(&checker, Player::White));
}

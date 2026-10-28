use super::*;

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
    let mut state = GameState::from_board(
        crate::LATEST_RULESET_ID,
        Mode::Standard,
        board,
        Player::White,
        7,
    );
    state.completed_drafts = [true; 3];
    state
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
pub(in crate::engine::cards) fn use_at(
    s: &mut GameState,
    engine: &Engine,
    id: &str,
    target: ActionTarget,
) {
    *s = engine.grant_card(s, 1, id, Player::White).unwrap().state;
    *s = engine
        .apply(
            s,
            Action::UseCard {
                instance_id: 1,
                target,
            },
        )
        .unwrap()
        .state;
}
pub(in crate::engine::cards) fn single(q: &str) -> ActionTarget {
    ActionTarget::PieceAt(q.parse().unwrap())
}
pub(in crate::engine::cards) fn pair(a: &str, b: &str) -> ActionTarget {
    ActionTarget::Squares(vec![a.parse().unwrap(), b.parse().unwrap()])
}
pub(in crate::engine::cards) fn handoff(engine: &Engine, s: &mut GameState) {
    let actor = s.turn;
    engine.finish_turn(s, actor, &mut Vec::new()).unwrap();
}
pub(in crate::engine::cards) fn three_owner_starts(engine: &Engine, s: &mut GameState) {
    for _ in 0..6 {
        handoff(engine, s);
    }
}
pub(in crate::engine::cards) fn has_move(engine: &Engine, s: &GameState, a: &str, b: &str) -> bool {
    engine.legal_actions(s).unwrap().contains(&movement(a, b))
}

#[test]
pub(in crate::engine::cards) fn reservations_affect_repetition_and_reject_reserved_initial_landing()
{
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[
        ("d1", Player::White, PieceKind::Queen),
        ("c1", Player::White, PieceKind::Bishop),
        ("d6", Player::Black, PieceKind::Rook),
    ]);
    for (id, target) in [
        (crate::SEAL, single("d6")),
        (crate::GREEK_GIFT, single("c1")),
        (crate::TABOO, pair("d1", "e5")),
        (crate::ASCENSION, pair("d1", "e5")),
    ] {
        let mut s = base.clone();
        use_at(&mut s, &engine, id, target);
        let mut cleared = s.clone();
        cleared.card_runtime.october = crate::state::OctoberRuntimeState::default();
        assert_ne!(s.position_key(), cleared.position_key());
        assert_ne!(s.fingerprint(), cleared.fingerprint());
    }
    let mut s = engine
        .grant_card(&base, 1, crate::ASCENSION, Player::White)
        .unwrap()
        .state;
    s.card_runtime.barricades.insert("e5".parse().unwrap());
    assert!(
        engine
            .apply(
                &s,
                Action::UseCard {
                    instance_id: 1,
                    target: pair("d1", "e5")
                }
            )
            .is_err()
    );
    let royal = at(&s, "d1");
    s.card_runtime.royal_piece_ids.insert(royal.id);
    assert!(
        !engine
            .legal_actions(&s)
            .unwrap()
            .iter()
            .any(|a| matches!(a, Action::UseCard { .. }))
    );
}

#[test]
pub(in crate::engine::cards) fn countdowns_distinguish_positions_and_search_branches_keep_reservations_isolated()
 {
    let engine = Engine::bundled_latest().unwrap();
    let base = position(&[
        ("d1", Player::White, PieceKind::Queen),
        ("c1", Player::White, PieceKind::Bishop),
        ("d6", Player::Black, PieceKind::Rook),
    ]);
    for (id, target) in [
        (crate::SEAL, single("d6")),
        (crate::GREEK_GIFT, single("c1")),
        (crate::TABOO, pair("d1", "e5")),
        (crate::ASCENSION, pair("d1", "e5")),
    ] {
        let mut s = base.clone();
        use_at(&mut s, &engine, id, target);
        let before = s.clone();
        let fingerprint = s.fingerprint();
        let expected = engine.apply(&s, movement("a1", "a2")).unwrap();
        let mut chance = SearchChanceContext::new(19);
        let actual = engine
            .apply_for_search(&s, movement("a1", "a2"), &mut chance)
            .unwrap();
        assert!(!actual.sampled_chance);
        assert_eq!(chance.event_index(), 0);
        assert_eq!(
            actual.transition.state.fingerprint(),
            expected.state.fingerprint()
        );
        assert_eq!(s.fingerprint(), fingerprint);
        assert_eq!(s, before);
        if matches!(id, crate::SEAL | crate::ASCENSION) {
            let mut later = s.clone();
            later.turns_taken[0] += 1;
            assert_ne!(s.position_key(), later.position_key());
        }
    }
    assert!(
        crate::check_card_eligibility(
            &position(&[]),
            crate::TABOO,
            Player::White,
            crate::AcquisitionContext::Draft
        )
        .is_eligible()
    );
    assert!(
        !crate::check_card_eligibility(
            &position(&[]),
            crate::TABOO,
            Player::White,
            crate::AcquisitionContext::Setup
        )
        .is_eligible()
    );
}

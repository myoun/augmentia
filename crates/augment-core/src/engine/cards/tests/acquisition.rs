use super::*;

pub(in crate::engine::cards) fn game() -> GameState {
    let mut state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
    state.completed_drafts = [true; 3];
    state
}
pub(in crate::engine::cards) fn position(pieces: &[(&str, Player, PieceKind)]) -> GameState {
    let mut board = crate::Board::empty();
    for (square, owner, kind) in pieces {
        board.set(square.parse().unwrap(), Some(Piece::new(*owner, *kind)));
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
pub(in crate::engine::cards) fn movement(from: &str, to: &str) -> Action {
    Action::Move(MoveAction::new(
        from.parse().unwrap(),
        to.parse().unwrap(),
        None,
    ))
}
pub(in crate::engine::cards) fn passive(engine: &Engine, id: &str) -> Transition {
    engine.grant_card(&game(), 1, id, Player::White).unwrap()
}

#[test]
pub(in crate::engine::cards) fn opening_flags_install_at_acquisition_without_a_charge_action_or_rng()
 {
    let engine = Engine::bundled_latest().unwrap();
    for id in [FAST_GROWTH, FIANCHETTO, FIELD_PROMOTION] {
        let before = game();
        let transition = engine.grant_card(&before, 1, id, Player::White).unwrap();
        let state = &transition.state;
        assert_eq!(
            (
                state.turn,
                state.ply,
                state.actions_remaining,
                state.rng_state()
            ),
            (before.turn, 0, 1, before.rng_state())
        );
        assert_eq!(state.cards[0].charges, 0);
        assert!(state.card_runtime.opening_auto_card_ids.is_empty());
        assert!(match id {
            FAST_GROWTH => state.card_runtime.fast_growth[0],
            FIANCHETTO => state.card_runtime.fianchetto[0],
            _ => state.card_runtime.field_promotion[0],
        });
        assert!(
            !engine
                .legal_actions(state)
                .unwrap()
                .iter()
                .any(|a| matches!(a, Action::UseCard { instance_id: 1, .. }))
        );
        let next = engine.apply(state, movement("e2", "e4")).unwrap();
        assert!(
            !next
                .events
                .iter()
                .any(|e| matches!(e, Event::CardUsed { .. }))
        );
    }
}

#[test]
pub(in crate::engine::cards) fn london_and_bishop_install_immediately_without_fresh_capture_delay()
{
    let engine = Engine::bundled_latest().unwrap();
    let london = passive(&engine, LONDON_SYSTEM).state;
    assert_eq!(
        london.board.get("d2".parse().unwrap()).unwrap().kind,
        PieceKind::Knight
    );
    assert_eq!(
        london.board.get("f4".parse().unwrap()).unwrap().kind,
        PieceKind::Bishop
    );
    let home_pawn = london.board.get("a2".parse().unwrap()).unwrap();
    assert!(!london.card_runtime.moved_piece_ids.contains(&home_pawn.id));
    assert!(
        engine
            .legal_actions(&london)
            .unwrap()
            .contains(&movement("a2", "a4"))
    );
    for (_, piece) in london
        .board
        .iter()
        .filter(|(_, p)| p.owner == Player::White)
    {
        assert!(
            !london
                .card_runtime
                .fresh_no_capture_until
                .contains_key(&piece.id)
        );
    }
    let mut before = game();
    before.board.set(
        "f4".parse().unwrap(),
        Some(Piece::new(Player::Black, PieceKind::Bishop).with_id(100)),
    );
    let bishop = engine
        .grant_card(&before, 1, BIG_BISHOP, Player::White)
        .unwrap()
        .state;
    let large = bishop.board.get("f2".parse().unwrap()).unwrap();
    assert_eq!(large.kind, PieceKind::BigBishop);
    assert_eq!(bishop.card_runtime.health(large.id).unwrap().hp, 2);
    assert!(
        engine
            .legal_actions(&bishop)
            .unwrap()
            .contains(&movement("h3", "f4"))
    );
}

#[test]
pub(in crate::engine::cards) fn passive_draft_pick_applies_before_opponent_offer_and_is_not_applied_twice()
 {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = game();
    state.phase = GamePhase::Draft(DraftState {
        phase: DraftPhase::Opening,
        offers: vec![OTHERWORLD.into()],
        resume_turn: Player::White,
    });
    let picked = engine
        .apply(&state, Action::DraftPick { offer_index: 0 })
        .unwrap()
        .state;
    assert_eq!(
        picked
            .board
            .iter()
            .filter(|(_, p)| p.owner == Player::White && p.kind == PieceKind::Pawn)
            .count(),
        7
    );
    assert_eq!(picked.ply, 0);
    assert_eq!(picked.turn, Player::Black);
    assert!(picked.card_runtime.opening_auto_card_ids.is_empty());
    let GamePhase::Draft(mut draft) = picked.phase.clone() else {
        panic!("black opening draft");
    };
    draft.offers = vec![SPRINT.into()];
    let mut picked = picked;
    picked.phase = GamePhase::Draft(draft);
    let completed = engine
        .apply(&picked, Action::DraftPick { offer_index: 0 })
        .unwrap()
        .state;
    assert_eq!(completed.card_runtime.pending_otherworld.len(), 1);
    assert_eq!(completed.current_repetition_count(), 1);
    assert_eq!(completed.automatic_draw.as_ref().unwrap().quiet_turns, 0);
    assert!(
        !completed
            .automatic_draw
            .as_ref()
            .unwrap()
            .progress_this_turn
    );
    let action = engine
        .legal_actions(&completed)
        .unwrap()
        .into_iter()
        .find(|a| matches!(a, Action::Move(_)))
        .unwrap();
    let played = engine.apply(&completed, action).unwrap();
    assert_eq!(played.state.card_runtime.pending_otherworld.len(), 1);
    assert!(!played.events.iter().any(
        |e| matches!(e, Event::CardUsed { definition_id, .. } if definition_id == OTHERWORLD)
    ));
}

#[test]
pub(in crate::engine::cards) fn end_passive_waits_for_unlock_then_installs_without_use_card() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = game();
    state.phase = GamePhase::Draft(DraftState {
        phase: DraftPhase::End,
        offers: vec![crate::FINAL_WEAPON.into()],
        resume_turn: Player::White,
    });
    state = engine
        .apply(&state, Action::DraftPick { offer_index: 0 })
        .unwrap()
        .state;
    let GamePhase::Draft(mut draft) = state.phase.clone() else {
        panic!("black end draft");
    };
    draft.offers = vec![crate::GOMOKU.into()];
    state.phase = GamePhase::Draft(draft);
    state = engine
        .apply(&state, Action::DraftPick { offer_index: 0 })
        .unwrap()
        .state;
    // The END passive is owned but not yet armed; both players must finish their
    // unlock turn before it installs.
    assert!(!state.card_runtime.final_weapon[Player::White.index()]);
    state = engine.apply(&state, movement("g1", "f3")).unwrap().state;
    assert!(!state.card_runtime.final_weapon[Player::White.index()]);
    let transition = engine.apply(&state, movement("g8", "f6")).unwrap();
    assert!(transition.state.card_runtime.final_weapon[Player::White.index()]);
    assert!(
        transition
            .state
            .cards
            .iter()
            .find(|c| c.definition_id == crate::FINAL_WEAPON)
            .unwrap()
            .available_after_turn
            .is_none()
    );
    assert!(!transition.events.iter().any(
        |e| matches!(e, Event::CardUsed { definition_id, .. } if definition_id == crate::FINAL_WEAPON)
    ));
}

#[test]
pub(in crate::engine::cards) fn failed_otherworld_acquisition_is_transactional_and_passives_cannot_be_clicked()
 {
    let engine = Engine::bundled_latest().unwrap();
    let before = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
    ]);
    let unchanged = before.clone();
    assert!(
        engine
            .grant_card(&before, 1, OTHERWORLD, Player::White)
            .is_err()
    );
    assert_eq!(before, unchanged);
    let mut state = passive(&engine, FIELD_PROMOTION).state;
    state.cards[0].charges = 1;
    assert!(
        !engine
            .legal_actions(&state)
            .unwrap()
            .iter()
            .any(|a| matches!(a, Action::UseCard { instance_id: 1, .. }))
    );
    state.card_runtime.opening_auto_card_ids.insert(1);
    assert!(engine.legal_actions(&state).is_err());
}

#[test]
pub(in crate::engine::cards) fn fast_growth_also_restricts_the_second_capture_field_promotion_choice()
 {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("a1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("d2", Player::White, PieceKind::Pawn),
        ("e3", Player::Black, PieceKind::Pawn),
    ]);
    let mut pawn = state.board.get("d2".parse().unwrap()).unwrap();
    pawn.capture_count = 1;
    state.board.set("d2".parse().unwrap(), Some(pawn));
    state = engine
        .grant_card(&state, 1, FAST_GROWTH, Player::White)
        .unwrap()
        .state;
    state = engine
        .grant_card(&state, 2, FIELD_PROMOTION, Player::White)
        .unwrap()
        .state;
    state = engine.apply(&state, movement("d2", "e3")).unwrap().state;
    let GamePhase::AwaitingChoice(choice) = &state.phase else {
        panic!("second-capture promotion");
    };
    assert_eq!(choice.options, ["bishop", "knight"]);
    let next = engine
        .apply(
            &state,
            Action::ResolveChoice {
                choice_id: choice.id,
                option_index: 1,
            },
        )
        .unwrap()
        .state;
    assert_eq!(
        next.board.get("e3".parse().unwrap()).unwrap().kind,
        PieceKind::Knight
    );
}

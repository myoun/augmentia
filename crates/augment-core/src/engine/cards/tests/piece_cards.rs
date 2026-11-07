use super::*;
use serde_json::Value;

fn fixtures() -> Value {
    serde_json::from_str(include_str!(
        "../../../../tests/fixtures/current-piece-cards.json"
    ))
    .unwrap()
}
fn kind(name: &str) -> PieceKind {
    let pascal = name
        .split('-')
        .map(|word| format!("{}{}", word[..1].to_uppercase(), &word[1..]))
        .collect::<String>();
    serde_json::from_value(Value::String(pascal)).unwrap()
}
fn position(extra: &[(&str, Player, PieceKind)]) -> GameState {
    let mut board = Board::empty();
    for (s, owner, kind) in [
        ("h1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
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
fn use_piece(square: &str) -> Action {
    Action::UseCard {
        instance_id: 1,
        target: ActionTarget::PieceAt(square.parse().unwrap()),
    }
}
fn use_pair(a: &str, b: &str) -> Action {
    Action::UseCard {
        instance_id: 1,
        target: ActionTarget::Squares(vec![a.parse().unwrap(), b.parse().unwrap()]),
    }
}
fn use_none() -> Action {
    Action::UseCard {
        instance_id: 1,
        target: ActionTarget::None,
    }
}
fn movement(a: &str, b: &str) -> Action {
    Action::Move(MoveAction::new(
        a.parse().unwrap(),
        b.parse().unwrap(),
        None,
    ))
}
fn acquire(engine: &Engine, state: &GameState, id: &str) -> GameState {
    engine
        .acquire_for_setup(state, id, Player::White)
        .unwrap()
        .state
}
fn moves(engine: &Engine, state: &GameState, from: &str) -> BTreeSet<String> {
    engine
        .legal_actions(state)
        .unwrap()
        .into_iter()
        .filter_map(|a| match a {
            Action::Move(m) if m.from == from.parse().unwrap() => Some(m.to.to_string()),
            _ => None,
        })
        .collect()
}

#[test]
fn all_single_target_cards_match_952_pinned_selection_contracts() {
    let engine = Engine::bundled_latest().unwrap();
    for case in fixtures()["selected"].as_array().unwrap() {
        let color = if case["color"] == "white" {
            Player::White
        } else {
            Player::Black
        };
        let mut state = position(&[
            ("b2", Player::White, PieceKind::Queen),
            ("b3", Player::White, PieceKind::Rook),
            ("c2", Player::White, PieceKind::Bishop),
            ("c3", Player::White, PieceKind::Knight),
            ("d4", color, kind(case["type"].as_str().unwrap())),
        ]);
        if case["royal"] == true {
            let id = state.board.get("d4".parse().unwrap()).unwrap().id;
            state.card_runtime.royal_piece_ids.insert(id);
        }
        let state = acquire(&engine, &state, case["card"].as_str().unwrap());
        assert_eq!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("d4")),
            case["allowed"].as_bool().unwrap(),
            "{case}"
        );
    }
}

#[test]
fn grappler_and_windmill_pair_order_and_royal_contracts_match_source() {
    let engine = Engine::bundled_latest().unwrap();
    for case in fixtures()["pairs"].as_array().unwrap() {
        for card in [crate::GRAPPLER, crate::WINDMILL] {
            let mut state = position(&[
                ("d4", Player::White, PieceKind::Queen),
                ("b2", Player::White, PieceKind::Bishop),
                ("c2", Player::White, PieceKind::Rook),
                ("d2", Player::Black, PieceKind::Knight),
            ]);
            let grant = acquire(&engine, &state, card);
            state = grant;
            if case["royal"] == true {
                let ids: Vec<_> = state.board.iter().map(|(_, p)| p.id).collect();
                state.card_runtime.royal_piece_ids.extend(ids);
            }
            let action = use_pair(
                case["squares"][0].as_str().unwrap(),
                case["squares"][1].as_str().unwrap(),
            );
            assert_eq!(
                engine.legal_actions(&state).unwrap().contains(&action),
                case[card].as_bool().unwrap(),
                "{card} {case}"
            );
        }
    }
}

#[test]
fn transformations_keep_identity_capture_count_generic_status_and_the_same_turn() {
    let engine = Engine::bundled_latest().unwrap();
    // Barrier and Killer became on-acquire passives in the 2026-10-27 balance
    // pass, so they no longer expose a card action; their acquisition path is
    // covered by `acquired_random_transform_cards_convert_on_pickup_and_keep_status`
    // below. Beacon is upstream-disabled from 2026-11-02.
    for (card, from, to) in [
        (crate::CAMPFIRE, PieceKind::Knight, PieceKind::Campfire),
        (
            crate::CONSTITUTIONAL_MONARCHY,
            PieceKind::Queen,
            PieceKind::PrimeMinister,
        ),
        (crate::DON_QUIXOTE, PieceKind::Rook, PieceKind::DonQuixote),
        (crate::EASTERN_POLICY, PieceKind::Camel, PieceKind::Cannon),
        (crate::IDOL, PieceKind::Queen, PieceKind::Idol),
        (crate::JESTER, PieceKind::Queen, PieceKind::Jester),
        (crate::MEDIUM, PieceKind::Knight, PieceKind::Medium),
        (crate::REAPER, PieceKind::Queen, PieceKind::Reaper),
        (
            crate::REVOLVING_DOOR,
            PieceKind::Rook,
            PieceKind::RevolvingDoor,
        ),
        (
            crate::TRANSMISSION_TOWER,
            PieceKind::Rook,
            PieceKind::TransmissionTower,
        ),
        (crate::WIZARD, PieceKind::Queen, PieceKind::Wizard),
    ] {
        let mut state = position(&[("d4", Player::White, from)]);
        let mut p = state.board.get("d4".parse().unwrap()).unwrap();
        p.capture_count = 7;
        state.board.set("d4".parse().unwrap(), Some(p));
        state.card_runtime.shielded_piece_ids.insert(p.id);
        state.card_runtime.poisoned_piece_ids.insert(p.id);
        let state = acquire(&engine, &state, card);
        let before = state.fingerprint();
        let next = engine.apply(&state, use_piece("d4")).unwrap();
        assert_eq!(state.fingerprint(), before);
        let changed = next.state.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(
            (changed.id, changed.capture_count, changed.kind),
            (p.id, 7, to),
            "{card}"
        );
        assert!(next.state.card_runtime.shielded_piece_ids.contains(&p.id));
        assert!(next.state.card_runtime.poisoned_piece_ids.contains(&p.id));
        assert!(next.state.card_runtime.moved_piece_ids.contains(&p.id));
        assert_eq!(next.state.turn, Player::White);
        assert_eq!(next.state.actions_remaining, state.actions_remaining);
        assert_eq!(next.state.ply, state.ply);
        assert!(
            !next
                .events
                .iter()
                .any(|e| matches!(e, Event::TurnEnded { .. }))
        );
        assert_eq!(next.state.cards[0].charges, 0);
    }
}

/// The 2026-10-27 balance pass converted these explicit-target piece cards into
/// on-acquire passives that transform a random own piece of a fixed source kind.
/// Upstream keeps the mapping in `OCTOBER27_RANDOM_TRANSFORMS` and resolves it
/// when the card is acquired, so the piece changes without any card action.
#[test]
fn acquired_random_transform_cards_convert_on_pickup_and_keep_status() {
    let engine = Engine::bundled_latest().unwrap();
    for (card, from, to) in [
        (crate::BARRIER, PieceKind::Rook, PieceKind::Barrier),
        (crate::BERSERKER, PieceKind::Rook, PieceKind::Berserker),
        (crate::KILLER, PieceKind::Bishop, PieceKind::Killer),
        (crate::SIREN, PieceKind::Rook, PieceKind::Siren),
        (crate::UNDEAD, PieceKind::Queen, PieceKind::Undead),
        (crate::HEDGEHOG, PieceKind::Queen, PieceKind::Hedgehog),
    ] {
        let mut state = position(&[("d4", Player::White, from)]);
        let mut p = state.board.get("d4".parse().unwrap()).unwrap();
        p.capture_count = 7;
        state.board.set("d4".parse().unwrap(), Some(p));
        state.card_runtime.shielded_piece_ids.insert(p.id);
        let state = acquire(&engine, &state, card);
        // Acquisition itself performs the transformation: no card action needed.
        assert!(
            !state
                .cards
                .iter()
                .any(|instance| instance.definition_id == card && instance.charges > 0),
            "{card} must not expose a usable charge"
        );
        let changed = state.board.get("d4".parse().unwrap()).unwrap();
        assert_eq!(
            (changed.id, changed.capture_count, changed.kind),
            (p.id, 7, to),
            "{card}"
        );
        assert!(
            state.card_runtime.shielded_piece_ids.contains(&p.id),
            "{card} must keep generic status"
        );
        assert!(
            state.card_runtime.moved_piece_ids.contains(&p.id),
            "{card} must mark the transformed piece moved"
        );
    }
}

#[test]
fn chameleon_mutation_marks_up_to_three_random_non_royal_pieces_on_pickup() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(&[
        ("b2", Player::White, PieceKind::Queen),
        ("b3", Player::White, PieceKind::Rook),
        ("c2", Player::White, PieceKind::Bishop),
        ("c3", Player::White, PieceKind::Knight),
        ("e4", Player::White, PieceKind::Pawn),
    ]);
    let state = acquire(&engine, &state, crate::CHAMELEON_MUTATION);
    assert_eq!(
        state.card_runtime.chameleon_piece_ids.len(),
        3,
        "the acquisition marks at most three pieces"
    );
    // The king is never marked.
    let king = state.board.get("a1".parse().unwrap()).unwrap();
    assert!(!state.card_runtime.chameleon_piece_ids.contains(&king.id));
}

#[test]
fn current_native_geometry_matches_32_pinned_authority_fixtures() {
    let engine = Engine::bundled_latest().unwrap();
    for case in fixtures()["native"].as_array().unwrap() {
        let color = if case["color"] == "white" {
            Player::White
        } else {
            Player::Black
        };
        let mut entries = vec![("d4", color, kind(case["kind"].as_str().unwrap()))];
        for p in case["blockers"].as_array().unwrap() {
            entries.push((
                p["square"].as_str().unwrap(),
                if p["color"] == "white" {
                    Player::White
                } else {
                    Player::Black
                },
                kind(p["type"].as_str().unwrap()),
            ));
        }
        let mut state = position(&entries);
        state.turn = color;
        let id = state.board.get("d4".parse().unwrap()).unwrap().id;
        state
            .card_runtime
            .windmill_modes
            .insert(id, WindmillMode::Bishop);
        let expected: BTreeSet<_> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            moves(&engine, &state, "d4"),
            expected,
            "{} {color}",
            case["kind"]
        );
    }
}

#[test]
fn grappler_sacrifice_is_not_a_capture_and_the_new_piece_can_pull_after_fresh_delay() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("d3", Player::White, PieceKind::Queen),
        ("b2", Player::White, PieceKind::Knight),
        ("d7", Player::Black, PieceKind::Rook),
    ]);
    let minor = state.board.get("b2".parse().unwrap()).unwrap();
    state.card_runtime.recurrence_piece_ids.insert(minor.id);
    let state = acquire(&engine, &state, crate::GRAPPLER);
    let next = engine.apply(&state, use_pair("d3", "b2")).unwrap();
    assert!(next.state.board.get("b2".parse().unwrap()).is_none());
    assert!(next.state.card_runtime.pending_recurrences.is_empty());
    assert!(
        !next
            .events
            .iter()
            .any(|e| matches!(e, Event::PieceCaptured { .. }))
    );
    let state = engine
        .apply(&next.state, movement("h1", "g1"))
        .unwrap()
        .state;
    let state = engine.apply(&state, movement("h8", "g8")).unwrap().state;
    let state = engine.apply(&state, movement("d3", "d7")).unwrap().state;
    assert_eq!(
        state.board.get("d4".parse().unwrap()).unwrap().kind,
        PieceKind::Rook
    );
    assert_eq!(
        state.board.get("d3".parse().unwrap()).unwrap().kind,
        PieceKind::Grappler
    );
}

#[test]
fn windmill_transforms_only_the_selected_pair_and_preserves_moved_flags() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(&[
        ("b2", Player::White, PieceKind::Bishop),
        ("c2", Player::White, PieceKind::Rook),
        ("b3", Player::White, PieceKind::Bishop),
        ("c3", Player::White, PieceKind::Rook),
    ]);
    let state = acquire(&engine, &state, crate::WINDMILL);
    let ids: Vec<_> = ["b2", "c2"]
        .map(|s| state.board.get(s.parse().unwrap()).unwrap().id)
        .into();
    let state = engine.apply(&state, use_pair("b2", "c2")).unwrap().state;
    for id in ids {
        assert_eq!(
            state.board.find_by_id(id).unwrap().1.kind,
            PieceKind::Windmill
        );
        assert!(!state.card_runtime.moved_piece_ids.contains(&id));
        assert_eq!(state.card_runtime.windmill_modes[&id], WindmillMode::Bishop);
    }
    assert_eq!(
        state.board.get("b3".parse().unwrap()).unwrap().kind,
        PieceKind::Bishop
    );
    assert_eq!(
        state.board.get("c3".parse().unwrap()).unwrap().kind,
        PieceKind::Rook
    );
}

#[test]
fn reformation_transforms_all_own_bishops_and_protestant_passes_blockers() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(&[
        ("d4", Player::White, PieceKind::Bishop),
        ("a2", Player::White, PieceKind::Bishop),
        ("e5", Player::White, PieceKind::Pawn),
        ("f6", Player::Black, PieceKind::Pawn),
        ("g7", Player::Black, PieceKind::Bishop),
    ]);
    let state = engine
        .apply(&acquire(&engine, &state, crate::REFORMATION), use_none())
        .unwrap()
        .state;
    assert_eq!(
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == Player::White && p.kind == PieceKind::Protestant)
            .count(),
        2
    );
    let state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
    let state = engine.apply(&state, movement("h8", "g8")).unwrap().state;
    assert!(moves(&engine, &state, "d4").contains("f6"));
    assert!(moves(&engine, &state, "d4").contains("g7"));
    assert!(!moves(&engine, &state, "d4").contains("e5"));
}

#[test]
fn eagle_keeps_original_identities_spawns_at_most_one_extra_per_knight_and_blocks_fresh_captures() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(&[
        ("b2", Player::White, PieceKind::Knight),
        ("f2", Player::White, PieceKind::Knight),
    ]);
    let ids: BTreeSet<_> = state
        .board
        .iter()
        .filter(|(_, p)| p.kind == PieceKind::Knight)
        .map(|(_, p)| p.id)
        .collect();
    let next = engine
        .apply(&acquire(&engine, &state, crate::EAGLE), use_none())
        .unwrap();
    assert_eq!(
        next.state
            .board
            .iter()
            .filter(|(_, p)| p.kind == PieceKind::Alibaba)
            .count(),
        4
    );
    for id in ids {
        assert_eq!(
            next.state.board.find_by_id(id).unwrap().1.kind,
            PieceKind::Alibaba
        );
    }
    assert_eq!(
        next.events
            .iter()
            .filter(
                |e| matches!(e,Event::RandomSample{purpose,..} if purpose=="card.eagle.placement")
            )
            .count(),
        2
    );
    assert_eq!(next.state.rule_runtime.piece_fresh_until.len(), 4);
}

#[test]
fn fresh_piece_capture_delay_survives_the_opponent_turn_and_same_type_conversion_keeps_old_status()
{
    let engine = Engine::bundled_latest().unwrap();
    let state = position(&[
        ("d4", Player::White, PieceKind::Queen),
        ("d6", Player::Black, PieceKind::Pawn),
    ]);
    let mut state = engine
        .apply(&acquire(&engine, &state, crate::JESTER), use_piece("d4"))
        .unwrap()
        .state;
    let id = state.board.get("d4".parse().unwrap()).unwrap().id;
    assert_eq!(state.rule_runtime.piece_fresh_until[&id], 2);
    state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
    assert_eq!(state.rule_runtime.piece_fresh_until[&id], 2);
    state = engine.apply(&state, movement("h8", "g8")).unwrap().state;
    assert!(!state.rule_runtime.piece_fresh_until.contains_key(&id));
    let state = position(&[("d4", Player::White, PieceKind::Campfire)]);
    let state = engine
        .apply(&acquire(&engine, &state, crate::CAMPFIRE), use_piece("d4"))
        .unwrap()
        .state;
    assert!(state.rule_runtime.piece_fresh_until.is_empty());
}

#[test]
fn meteor_footprints_match_all_64_source_anchor_cases() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[("d4", Player::White, PieceKind::Wizard)]);
    let id = state.board.get("d4".parse().unwrap()).unwrap().id;
    state.card_runtime.wizard_mana.insert(id, 5);
    for case in fixtures()["meteor"].as_array().unwrap() {
        let next = engine
            .apply(
                &state,
                Action::UsePieceAbility {
                    piece_id: id,
                    ability: PieceAbility::WizardMeteor,
                    target: ActionTarget::Square(case["at"].as_str().unwrap().parse().unwrap()),
                },
            )
            .unwrap();
        let cells: BTreeSet<_> = next.state.card_runtime.delayed_hazards[0]
            .squares
            .iter()
            .map(|s| s.to_string())
            .collect();
        let expected: BTreeSet<_> = case["cells"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect();
        assert_eq!(cells, expected, "{}", case["at"]);
        assert_eq!(next.state.card_runtime.wizard_mana[&id], 2);
    }
}

#[test]
fn reaper_counts_only_allied_enemy_captures_and_executes_on_the_second_soul() {
    assert_eq!(fixtures()["reaper_target"], 2);
    assert_eq!(fixtures()["reaper_allied_loss"], true);
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("d4", Player::White, PieceKind::Reaper),
        ("c4", Player::White, PieceKind::Pawn),
        ("e4", Player::White, PieceKind::Pawn),
        ("c8", Player::Black, PieceKind::Rook),
        ("e8", Player::Black, PieceKind::Rook),
    ]);
    let id = state.board.get("d4".parse().unwrap()).unwrap().id;
    state.turn = Player::Black;
    let state = engine.apply(&state, movement("c8", "c4")).unwrap().state;
    assert_eq!(state.card_runtime.reaper_souls[&id], 1);
    let state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
    let state = engine.apply(&state, movement("e8", "e4")).unwrap().state;
    assert_eq!(state.card_runtime.reaper_souls[&id], 2);
    assert!(matches!(
        state.status,
        GameStatus::Won {
            winner: Player::White,
            ..
        }
    ));
    assert_eq!(state.board.get("h8".parse().unwrap()).unwrap().id, id);
    let state = position(&[
        ("d4", Player::White, PieceKind::Reaper),
        ("c4", Player::Black, PieceKind::Pawn),
        ("c8", Player::White, PieceKind::Rook),
    ]);
    let id = state.board.get("d4".parse().unwrap()).unwrap().id;
    let state = engine.apply(&state, movement("c8", "c4")).unwrap().state;
    assert_eq!(state.card_runtime.reaper_souls.get(&id), None);
}

#[test]
fn wizard_hazards_and_sacrifices_do_not_grant_reaper_souls() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("d4", Player::White, PieceKind::Reaper),
        ("c4", Player::White, PieceKind::Pawn),
        ("b6", Player::Black, PieceKind::Wizard),
    ]);
    let reaper = state.board.get("d4".parse().unwrap()).unwrap().id;
    let wizard = state.board.get("b6".parse().unwrap()).unwrap().id;
    state.card_runtime.reaper_souls.insert(reaper, 1);
    state.card_runtime.wizard_mana.insert(wizard, 1);
    state.turn = Player::Black;
    let state = engine
        .apply(
            &state,
            Action::UsePieceAbility {
                piece_id: wizard,
                ability: PieceAbility::WizardLightning,
                target: ActionTarget::Square("c4".parse().unwrap()),
            },
        )
        .unwrap()
        .state;
    let state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
    assert!(state.board.get("c4".parse().unwrap()).is_none());
    assert_eq!(state.card_runtime.reaper_souls[&reaper], 1);
    assert_eq!(state.status, GameStatus::Ongoing);
}

#[test]
fn medium_remembers_a_capture_before_the_card_was_acquired() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(&[
        ("a2", Player::White, PieceKind::Rook),
        ("b2", Player::Black, PieceKind::Bishop),
        ("d4", Player::White, PieceKind::Knight),
    ]);
    let state = engine.apply(&state, movement("a2", "b2")).unwrap().state;
    assert_eq!(
        state.rule_runtime.medium_memory.unwrap().effective,
        PieceKind::Bishop
    );
    let state = engine.apply(&state, movement("h8", "g8")).unwrap().state;
    let state = engine
        .apply(&acquire(&engine, &state, crate::MEDIUM), use_piece("d4"))
        .unwrap()
        .state;
    assert!(moves(&engine, &state, "d4").contains("f6"));
    assert!(!moves(&engine, &state, "d4").contains("d5"));
}

#[test]
fn idol_grants_one_encore_per_turn_and_rests_the_first_mover() {
    let engine = Engine::bundled_latest().unwrap();
    let state = position(&[
        ("d4", Player::White, PieceKind::Queen),
        ("c4", Player::White, PieceKind::Rook),
        ("b2", Player::White, PieceKind::Bishop),
    ]);
    let state = engine
        .apply(&acquire(&engine, &state, crate::IDOL), use_piece("d4"))
        .unwrap()
        .state;
    let state = engine.apply(&state, movement("c4", "c5")).unwrap().state;
    assert_eq!(state.turn, Player::White);
    assert_eq!(state.actions_remaining, 1);
    assert!(moves(&engine, &state, "c5").is_empty());
    let state = engine.apply(&state, movement("b2", "a3")).unwrap().state;
    assert_eq!(state.turn, Player::Black);
}

#[test]
fn wizard_mana_gain_caps_at_five_and_time_stop_spends_mana_without_a_turn_end() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("d4", Player::White, PieceKind::Queen),
        ("a2", Player::White, PieceKind::Pawn),
        ("a8", Player::Black, PieceKind::Rook),
    ]);
    state = engine
        .apply(&acquire(&engine, &state, crate::WIZARD), use_piece("d4"))
        .unwrap()
        .state;
    let id = state.board.get("d4".parse().unwrap()).unwrap().id;
    assert_eq!(state.card_runtime.wizard_mana[&id], 0);
    state = engine.apply(&state, movement("h1", "g1")).unwrap().state;
    state = engine.apply(&state, movement("a8", "a2")).unwrap().state;
    assert_eq!(state.card_runtime.wizard_mana[&id], 1);
    state.card_runtime.wizard_mana.insert(id, 5);
    let next = engine
        .apply(
            &state,
            Action::UsePieceAbility {
                piece_id: id,
                ability: PieceAbility::WizardTimeStop,
                target: ActionTarget::None,
            },
        )
        .unwrap();
    assert_eq!(next.state.card_runtime.wizard_mana[&id], 0);
    assert_eq!(next.state.turn, Player::White);
    assert!(
        !next
            .events
            .iter()
            .any(|e| matches!(e, Event::TurnEnded { .. }))
    );
    let next = engine
        .apply(&next.state, movement("g1", "f1"))
        .unwrap()
        .state;
    assert_eq!(next.turn, Player::White);
}

#[test]
fn eagle_prefers_a_farther_safe_cell_and_never_falls_back_to_an_attacked_cell() {
    let engine = Engine::bundled_latest().unwrap();
    let mut board = Board::empty();
    for i in 0..64 {
        board.set(
            Square::from_index(i).unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn)),
        );
    }
    for square in ["e4", "c1"] {
        board.set(square.parse().unwrap(), None);
    }
    for (square, owner, kind) in [
        ("h1", Player::White, PieceKind::King),
        ("h8", Player::Black, PieceKind::King),
        ("f6", Player::Black, PieceKind::Knight),
        ("d4", Player::White, PieceKind::Knight),
    ] {
        board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
    }
    let mut state = GameState::from_board(
        crate::LATEST_RULESET_ID,
        Mode::Standard,
        board,
        Player::White,
        31,
    );
    state.completed_drafts = [true; 3];
    let next = engine
        .apply(&acquire(&engine, &state, crate::EAGLE), use_none())
        .unwrap();
    let expected = &fixtures()["eagle"][0]["expected"][0];
    assert_eq!(expected, "c1");
    assert_eq!(
        next.state.board.get("c1".parse().unwrap()).unwrap().kind,
        PieceKind::Alibaba
    );
    assert!(next.state.board.get("e4".parse().unwrap()).is_none());
    let mut full = state.clone();
    let pawn = Piece::new(Player::White, PieceKind::Pawn).with_id(full.allocate_piece_id());
    full.board.set("c1".parse().unwrap(), Some(pawn));
    let next = engine
        .apply(&acquire(&engine, &full, crate::EAGLE), use_none())
        .unwrap();
    assert!(next.state.board.get("e4".parse().unwrap()).is_none());
    assert_eq!(
        next.state
            .board
            .iter()
            .filter(|(_, p)| p.kind == PieceKind::Alibaba)
            .count(),
        1
    );
    assert!(
        !next
            .events
            .iter()
            .any(|e| matches!(e, Event::RandomSample { .. }))
    );
}

#[test]
fn medium_complex_borrowed_movement_matches_pinned_base_memory_geometry() {
    let engine = Engine::bundled_latest().unwrap();
    for case in fixtures()["borrowed"].as_array().unwrap() {
        let color = if case["color"] == "white" {
            Player::White
        } else {
            Player::Black
        };
        let mut state = position(&[("d4", color, PieceKind::Medium)]);
        state.turn = color;
        for blocker in case["blockers"].as_array().unwrap() {
            let square = blocker["square"].as_str().unwrap().parse().unwrap();
            let owner = if blocker["color"] == "white" {
                Player::White
            } else {
                Player::Black
            };
            let piece = Piece::new(owner, kind(blocker["type"].as_str().unwrap()))
                .with_id(state.allocate_piece_id());
            state.board.set(square, Some(piece));
        }
        let source = match case["type"].as_str().unwrap() {
            "eagle" => PieceKind::Alibaba,
            other => kind(other),
        };
        state.rule_runtime.medium_memory = Some(ParrotMemoryStatus {
            source,
            effective: source,
        });
        state.rule_runtime.medium_windmill_mode = match case["mode"].as_str() {
            Some("rook") => Some(WindmillMode::Rook),
            Some("bishop") => Some(WindmillMode::Bishop),
            _ => None,
        };
        let expected: BTreeSet<_> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(moves(&engine, &state, "d4"), expected, "{case}");
    }
}

#[test]
fn medium_keeps_captured_windmill_mode_and_trickster_actual_movement() {
    let engine = Engine::bundled_latest().unwrap();
    for source in [PieceKind::Windmill, PieceKind::Trickster] {
        let mut state = position(&[
            ("a2", Player::White, PieceKind::Rook),
            ("b2", Player::Black, source),
            ("d4", Player::White, PieceKind::Knight),
        ]);
        let id = state.board.get("b2".parse().unwrap()).unwrap().id;
        state
            .card_runtime
            .windmill_modes
            .insert(id, WindmillMode::Rook);
        state
            .card_runtime
            .trickster_move_kinds
            .insert(id, PieceKind::Knight);
        let state = engine.apply(&state, movement("a2", "b2")).unwrap().state;
        let state = engine.apply(&state, movement("h8", "g8")).unwrap().state;
        let state = engine
            .apply(&acquire(&engine, &state, crate::MEDIUM), use_piece("d4"))
            .unwrap()
            .state;
        if source == PieceKind::Windmill {
            assert_eq!(
                state.rule_runtime.medium_windmill_mode,
                Some(WindmillMode::Rook)
            );
            assert!(moves(&engine, &state, "d4").contains("d6"));
        } else {
            assert_eq!(
                state.rule_runtime.medium_memory.unwrap().source,
                PieceKind::Knight
            );
            assert!(moves(&engine, &state, "d4").contains("f5"));
        }
    }
}

#[test]
fn medium_borrowed_checker_jump_captures_without_transforming_the_medium() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("d4", Player::White, PieceKind::Medium),
        ("e5", Player::Black, PieceKind::Pawn),
    ]);
    state.rule_runtime.medium_memory = Some(ParrotMemoryStatus {
        source: PieceKind::Checker,
        effective: PieceKind::Checker,
    });
    let state = engine.apply(&state, movement("d4", "f6")).unwrap().state;
    assert_eq!(
        state.board.get("f6".parse().unwrap()).unwrap().kind,
        PieceKind::Medium
    );
    assert!(state.board.get("e5".parse().unwrap()).is_none());
    assert_eq!(
        state.rule_runtime.medium_memory.unwrap().source,
        PieceKind::Pawn
    );
}

#[test]
fn medium_borrowed_dragon_swap_preserves_both_piece_identities() {
    let engine = Engine::bundled_latest().unwrap();
    let mut state = position(&[
        ("d4", Player::White, PieceKind::Medium),
        ("b2", Player::White, PieceKind::Pawn),
    ]);
    let medium = state.board.get("d4".parse().unwrap()).unwrap();
    let pawn = state.board.get("b2".parse().unwrap()).unwrap();
    state.rule_runtime.medium_memory = Some(ParrotMemoryStatus {
        source: PieceKind::Dragon,
        effective: PieceKind::Dragon,
    });
    let next = engine.apply(&state, movement("d4", "b2")).unwrap();
    assert_eq!(next.state.board.get("b2".parse().unwrap()), Some(medium));
    assert_eq!(next.state.board.get("d4".parse().unwrap()), Some(pawn));
    assert_eq!(next.state.turn, Player::Black);
    assert!(
        !next
            .events
            .iter()
            .any(|e| matches!(e, Event::PieceCaptured { .. }))
    );
}

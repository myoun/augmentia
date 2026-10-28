//! v19 compound pieces: native knight sacrifice plus a bishop/rook transformation.
//! Source: COMPOUND_CARD_SPECS, compoundCardTargets, amazonSelectionContract,
//! applyAmazonV7. Legal actions use the UI prefix order: knight, then base piece.
use super::*;

fn compound_base(id: &str) -> PieceKind {
    if id == crate::ARCHBISHOP {
        PieceKind::Bishop
    } else {
        PieceKind::Rook
    }
}

impl Engine {
    pub(super) fn generate_compound_card_actions(
        &self,
        state: &GameState,
        card: &CardInstance,
        actions: &mut Vec<Action>,
    ) {
        let eligible = |piece: Piece, kind| {
            piece.owner == state.turn && piece.kind == kind && !self.is_royal_piece(state, piece)
        };
        for (knight_square, _) in state
            .board
            .iter()
            .filter(|(_, p)| eligible(*p, PieceKind::Knight))
        {
            for (base_square, _) in state
                .board
                .iter()
                .filter(|(_, p)| eligible(*p, compound_base(&card.definition_id)))
            {
                actions.push(Action::UseCard {
                    instance_id: card.instance_id,
                    target: ActionTarget::Squares(vec![knight_square, base_square]),
                });
            }
        }
    }

    pub(super) fn apply_compound_card(
        &self,
        state: &mut GameState,
        actor: Player,
        id: &str,
        target: &ActionTarget,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let [knight_square, base_square] = two_square_target(target)?;
        let knight = state
            .board
            .get(knight_square)
            .ok_or_else(|| EngineError::Invariant("missing compound sacrifice".into()))?;
        let mut base = state
            .board
            .get(base_square)
            .ok_or_else(|| EngineError::Invariant("missing compound base".into()))?;
        // Removal does not capture or trigger resurrection/capture reactions.
        self.delete_piece_for_card(state, knight_square, knight, actor, id, events);
        base.kind = if id == crate::ARCHBISHOP {
            PieceKind::Archbishop
        } else {
            PieceKind::Chancellor
        };
        state.board.set(base_square, Some(base));
        // applyAmazonV7 marks identity moved, retaining statuses and capture count.
        // It does not spawn a new identity or impose a fresh-capture delay.
        state.card_runtime.moved_piece_ids.insert(base.id);
        self.resolve_gomoku_victory(state);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(pieces: &[(&str, Player, PieceKind)]) -> GameState {
        let mut board = crate::Board::empty();
        for (square, owner, kind) in pieces {
            board.set(square.parse().unwrap(), Some(Piece::new(*owner, *kind)));
        }
        let mut state = GameState::from_board(
            crate::LATEST_RULESET_ID,
            Mode::Standard,
            board,
            Player::White,
            42,
        );
        state.completed_drafts = [true; 3];
        state
    }
    fn setup(base: PieceKind) -> GameState {
        position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("b1", Player::White, PieceKind::Knight),
            ("d4", Player::White, base),
        ])
    }
    fn use_pair(first: &str, second: &str) -> Action {
        Action::UseCard {
            instance_id: 1,
            target: ActionTarget::Squares(vec![first.parse().unwrap(), second.parse().unwrap()]),
        }
    }
    fn movement(from: &str, to: &str) -> Action {
        Action::Move(MoveAction::new(
            from.parse().unwrap(),
            to.parse().unwrap(),
            None,
        ))
    }
    fn moves(engine: &Engine, state: &GameState, source: &str) -> BTreeSet<Square> {
        engine
            .legal_actions(state)
            .unwrap()
            .into_iter()
            .filter_map(|a| match a {
                Action::Move(m) if m.from == source.parse().unwrap() => Some(m.to),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn compound_moves_match_pinned_upstream_fixtures() {
        let engine = Engine::bundled_latest().unwrap();
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/october-compound.json"))
                .unwrap();
        for fixture in data["fixtures"].as_array().unwrap() {
            let owner = if fixture["owner"] == "white" {
                Player::White
            } else {
                Player::Black
            };
            let kind = if fixture["kind"] == "archbishop" {
                PieceKind::Archbishop
            } else {
                PieceKind::Chancellor
            };
            let from: Square = fixture["from"].as_str().unwrap().parse().unwrap();
            let mut state = position(&[(fixture["from"].as_str().unwrap(), owner, kind)]);
            for (i, blocker) in fixture["blockers"].as_array().unwrap().iter().enumerate() {
                let color = if blocker["owner"] == "white" {
                    Player::White
                } else {
                    Player::Black
                };
                state.board.set(
                    blocker["square"].as_str().unwrap().parse().unwrap(),
                    Some(Piece::new(color, PieceKind::Pawn).with_id(100 + i as u32)),
                );
            }
            let mut actions = Vec::new();
            engine.generate_moves_for_piece(
                &state,
                from,
                state.board.get(from).unwrap(),
                &mut actions,
            );
            let actual: BTreeSet<_> = actions
                .into_iter()
                .filter_map(|a| match a {
                    Action::Move(m) => Some(m.to.to_string()),
                    _ => None,
                })
                .collect();
            let expected: BTreeSet<_> = fixture["expected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect();
            assert_eq!(
                actual, expected,
                "{} {} {}",
                fixture["kind"], fixture["owner"], fixture["from"]
            );
        }
    }

    #[test]
    fn compound_cards_keep_base_identity_statuses_and_turn_without_capture_reactions() {
        let engine = Engine::bundled_latest().unwrap();
        for (id, base, result) in [
            (crate::ARCHBISHOP, PieceKind::Bishop, PieceKind::Archbishop),
            (crate::CHANCELLOR, PieceKind::Rook, PieceKind::Chancellor),
        ] {
            let mut before = setup(base);
            let (square, mut target) = (
                "d4".parse().unwrap(),
                before.board.get("d4".parse().unwrap()).unwrap(),
            );
            target.capture_count = 3;
            before.board.set(square, Some(target));
            before.card_runtime.shielded_piece_ids.insert(target.id);
            let knight = before.board.get("b1".parse().unwrap()).unwrap();
            before.card_runtime.recurrence_piece_ids.insert(knight.id);
            let state = engine
                .grant_card(&before, 1, id, Player::White)
                .unwrap()
                .state;
            let transition = engine.apply(&state, use_pair("b1", "d4")).unwrap();
            let after = &transition.state;
            let transformed = after.board.get(square).unwrap();
            assert_eq!(
                (transformed.id, transformed.owner, transformed.capture_count),
                (target.id, target.owner, 3)
            );
            assert_eq!(transformed.kind, result);
            assert!(after.board.get("b1".parse().unwrap()).is_none());
            assert!(after.card_runtime.shielded_piece_ids.contains(&target.id));
            assert!(after.card_runtime.moved_piece_ids.contains(&target.id));
            assert!(
                !after
                    .card_runtime
                    .fresh_no_capture_until
                    .contains_key(&target.id)
            );
            assert!(after.card_runtime.pending_recurrences.is_empty());
            assert!(!after.card_runtime.magic_girl_surge[0]);
            assert_eq!(
                (after.turn, after.ply, after.actions_remaining),
                (state.turn, state.ply, state.actions_remaining)
            );
            assert_eq!(after.rng_state(), state.rng_state());
            assert_eq!(after.cards[0].charges, 0);
            assert!(transition.events.iter().any(|e| matches!(e, Event::PieceRemoved { piece, cause, .. } if piece.id == knight.id && cause == id)));
            assert!(
                !transition
                    .events
                    .iter()
                    .any(|e| matches!(e, Event::PieceCaptured { .. }))
            );
        }
    }

    #[test]
    fn compound_targets_require_both_native_nonroyal_allies_and_reject_invalid_actions_atomically()
    {
        let engine = Engine::bundled_latest().unwrap();
        for (id, base) in [
            (crate::ARCHBISHOP, PieceKind::Bishop),
            (crate::CHANCELLOR, PieceKind::Rook),
        ] {
            let mut before = setup(base);
            before.board.set(
                "c1".parse().unwrap(),
                Some(Piece::new(Player::Black, PieceKind::Knight).with_id(100)),
            );
            let state = engine
                .grant_card(&before, 1, id, Player::White)
                .unwrap()
                .state;
            let actions = engine.legal_actions(&state).unwrap();
            assert_eq!(
                actions
                    .iter()
                    .filter(|a| matches!(a, Action::UseCard { .. }))
                    .count(),
                1
            );
            for invalid in [
                use_pair("b1", "b1"),
                use_pair("c1", "d4"),
                use_pair("b1", "a1"),
                use_pair("b1", "e4"),
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            ] {
                let fingerprint = state.fingerprint();
                assert!(engine.apply(&state, invalid).is_err());
                assert_eq!(state.fingerprint(), fingerprint);
            }
            for square in ["b1", "d4"] {
                let mut royal = state.clone();
                royal
                    .card_runtime
                    .royal_piece_ids
                    .insert(royal.board.get(square.parse().unwrap()).unwrap().id);
                assert!(
                    !engine
                        .legal_actions(&royal)
                        .unwrap()
                        .contains(&use_pair("b1", "d4"))
                );
                assert!(
                    !crate::check_card_eligibility(
                        &royal,
                        id,
                        Player::White,
                        crate::AcquisitionContext::Setup
                    )
                    .is_eligible()
                );
            }
            let mut missing = before.clone();
            missing.board.set("b1".parse().unwrap(), None);
            assert!(
                !crate::check_card_eligibility(
                    &missing,
                    id,
                    Player::White,
                    crate::AcquisitionContext::Draft
                )
                .is_eligible()
            );
        }
    }

    #[test]
    fn compound_movement_is_exact_union_of_native_slider_and_knight_with_blockers() {
        let engine = Engine::bundled_latest().unwrap();
        for (kind, base) in [
            (PieceKind::Archbishop, PieceKind::Bishop),
            (PieceKind::Chancellor, PieceKind::Rook),
        ] {
            for owner in [Player::White, Player::Black] {
                let mut state = position(&[
                    ("a1", owner, PieceKind::King),
                    ("h8", owner.opponent(), PieceKind::King),
                    ("d4", owner, kind),
                    ("e4", owner, PieceKind::Pawn),
                    ("e5", owner, PieceKind::Pawn),
                    ("b4", owner.opponent(), PieceKind::Pawn),
                    ("b6", owner.opponent(), PieceKind::Pawn),
                    ("f5", owner.opponent(), PieceKind::Pawn),
                ]);
                state.turn = owner;
                let mut native = state.clone();
                let mut piece = native.board.get("d4".parse().unwrap()).unwrap();
                piece.kind = base;
                native.board.set("d4".parse().unwrap(), Some(piece));
                let mut expected = moves(&engine, &native, "d4");
                piece.kind = PieceKind::Knight;
                native.board.set("d4".parse().unwrap(), Some(piece));
                expected.extend(moves(&engine, &native, "d4"));
                let actual = moves(&engine, &state, "d4");
                assert_eq!(actual, expected, "{kind:?} {owner:?}");
                assert!(
                    actual.contains(&"f5".parse().unwrap()),
                    "knight capture jumps blockers"
                );
                assert!(!actual.contains(&"e5".parse().unwrap()));
            }
        }
    }

    #[test]
    fn compound_piece_can_capture_immediately_after_transformation() {
        let engine = Engine::bundled_latest().unwrap();
        for (id, base) in [
            (crate::ARCHBISHOP, PieceKind::Bishop),
            (crate::CHANCELLOR, PieceKind::Rook),
        ] {
            let mut before = setup(base);
            before.board.set(
                "f5".parse().unwrap(),
                Some(Piece::new(Player::Black, PieceKind::Pawn).with_id(100)),
            );
            let state = engine
                .grant_card(&before, 1, id, Player::White)
                .unwrap()
                .state;
            let after = engine.apply(&state, use_pair("b1", "d4")).unwrap().state;
            let capture = engine.apply(&after, movement("d4", "f5")).unwrap();
            assert!(
                capture
                    .events
                    .iter()
                    .any(|e| matches!(e, Event::PieceCaptured { piece, .. } if piece.id == 100))
            );
        }
    }

    #[test]
    fn compound_attacks_capture_royals_and_release_initiative() {
        let engine = Engine::bundled_latest().unwrap();
        for (kind, target) in [
            (PieceKind::Archbishop, "g7"),
            (PieceKind::Chancellor, "d7"),
            (PieceKind::Archbishop, "f5"),
            (PieceKind::Chancellor, "f5"),
        ] {
            let mut state = position(&[
                ("a1", Player::White, PieceKind::King),
                (target, Player::Black, PieceKind::King),
                ("d4", Player::White, kind),
            ]);
            state.card_runtime.initiative[1] = Some(InitiativeStatus {
                by: Player::White,
                start_turn: 0,
                limit: 8,
            });
            engine.resolve_october_initiative_checks(&mut state);
            assert!(state.card_runtime.initiative[1].is_none());
            let after = engine.apply(&state, movement("d4", target)).unwrap().state;
            assert!(matches!(
                after.status,
                GameStatus::Won {
                    winner: Player::White,
                    ..
                }
            ));
        }
    }

    #[test]
    fn compounds_are_major_ranged_and_piece_roundtrip_keeps_identity_and_hash() {
        let engine = Engine::bundled_latest().unwrap();
        for kind in [PieceKind::Archbishop, PieceKind::Chancellor] {
            assert!(crate::latest::latest_major_kind(kind));
            assert!(!crate::latest::latest_minor_kind(kind));
            let state = setup(kind);
            let piece = state.board.get("d4".parse().unwrap()).unwrap();
            assert!(engine.is_ranged_piece(&state, piece));
            let bytes = serde_json::to_vec(&piece).unwrap();
            let restored: Piece = serde_json::from_slice(&bytes).unwrap();
            let mut decoded = state.clone();
            decoded.board.set("d4".parse().unwrap(), Some(restored));
            assert_eq!(decoded.fingerprint(), state.fingerprint());
            assert_eq!(decoded.position_key(), state.position_key());
            assert_eq!(decoded.board.get("d4".parse().unwrap()), Some(piece));
        }
    }
}

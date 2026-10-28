//! Bounded v19 ports. Source contracts and ranges live in cards/ported.json.
use super::*;

impl Engine {
    pub(super) fn resolve_october_acquisitions(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID
            || !matches!(state.phase, GamePhase::Playing)
        {
            return;
        }
        for owner in [Player::White, Player::Black] {
            if !std::mem::take(&mut state.card_runtime.october.iqp_pending[owner.index()]) {
                continue;
            }
            let targets: Vec<_> = state
                .board
                .iter()
                .filter(|(square, piece)| {
                    piece.owner == owner.opponent()
                        && piece.kind == PieceKind::Pawn
                        && matches!(square.file(), 2 | 4)
                })
                .collect();
            for (square, piece) in targets {
                // v8RemoveCurrentPieces(... rememberCapture:false): direct removal,
                // without crediting a capturing piece or scheduling capture reactions.
                self.delete_piece_for_card(state, square, piece, owner, crate::IQP, events);
            }
        }
    }

    pub(super) fn resolve_october_turn_start(&self, state: &mut GameState, owner: Player) {
        if state.ruleset_id == crate::LATEST_RULESET_ID
            && state.status == GameStatus::Ongoing
            && state.card_runtime.october.hope[owner.index()]
            && state
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == owner)
                .count()
                <= 2
        {
            state.status = GameStatus::Won {
                winner: owner,
                reason: WinReason::CardEffect(crate::HOPE.into()),
            };
        }
    }

    pub(super) fn resolve_october_assembly(
        &self,
        state: &mut GameState,
        original: Piece,
        events: &[Event],
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID
            || !state.card_runtime.october.assembly[original.owner.index()]
        {
            return;
        }
        let promoted = events.iter().any(|event| matches!(event, Event::PieceCaptured { actor, piece, .. }
            if *actor == original.owner && piece.owner != original.owner
            && matches!((original.kind, piece.kind), (PieceKind::Rook, PieceKind::Bishop) | (PieceKind::Bishop, PieceKind::Rook))));
        if promoted && let Some((square, mut piece)) = state.board.find_by_id(original.id) {
            piece.kind = PieceKind::Queen;
            state.board.set(square, Some(piece));
            state.card_runtime.moved_piece_ids.insert(piece.id);
        }
    }

    pub(super) fn resolve_october_initiative_checks(&self, state: &mut GameState) {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            return;
        }
        for target in [Player::White, Player::Black] {
            let Some(status) = state.card_runtime.initiative[target.index()] else {
                continue;
            };
            let king_squares: Vec<_> = state
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == target && self.is_royal_piece(state, *piece))
                .map(|(square, _)| square)
                .collect();
            let mut probe = state.clone();
            // Disable castling in an attack query to avoid recursively checking
            // castling safety. Keep the real target identity and its protections.
            probe.card_runtime.moved_piece_ids.extend(
                state
                    .board
                    .iter()
                    .filter(|(_, piece)| self.is_royal_piece(state, *piece))
                    .map(|(_, piece)| piece.id),
            );
            let checked = probe.board.iter().filter(|(_, piece)| piece.owner == status.by).any(|(from, piece)| {
                let mut actions = Vec::new();
                self.generate_moves_for_piece(&probe, from, piece, &mut actions);
                actions.iter().any(|action| matches!(action, Action::Move(movement) if king_squares.contains(&movement.to)))
            });
            if checked {
                state.card_runtime.initiative[target.index()] = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(pieces: &[(&str, Player, PieceKind)], turn: Player) -> GameState {
        let mut board = crate::Board::empty();
        for (square, owner, kind) in pieces {
            board.set(square.parse().unwrap(), Some(Piece::new(*owner, *kind)));
        }
        let mut state =
            GameState::from_board(crate::LATEST_RULESET_ID, Mode::Standard, board, turn, 1);
        state.completed_drafts = [true; 3];
        state
    }
    fn base() -> GameState {
        position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
        )
    }
    fn movement(from: &str, to: &str) -> Action {
        Action::Move(MoveAction::new(
            from.parse().unwrap(),
            to.parse().unwrap(),
            None,
        ))
    }
    fn card(target: &str) -> Action {
        Action::UseCard {
            instance_id: 1,
            target: ActionTarget::PieceAt(target.parse().unwrap()),
        }
    }
    fn has_move(engine: &Engine, state: &GameState, from: &str, to: &str) -> bool {
        engine
            .legal_actions(state)
            .unwrap()
            .contains(&movement(from, to))
    }

    #[test]
    fn six_restored_transform_contracts_keep_identity_owner_turn_and_reset_recruiter_captures() {
        let engine = Engine::bundled_latest().unwrap();
        for (id, before_kind, after_kind) in [
            (ASSASSIN, PieceKind::Knight, PieceKind::Assassin),
            (DRAGON, PieceKind::Rook, PieceKind::Dragon),
            (KNIGHTMASTER, PieceKind::Knight, PieceKind::Knightmaster),
            (LOG, PieceKind::Pawn, PieceKind::Log),
            (STANDARD_BEARER, PieceKind::Pawn, PieceKind::StandardBearer),
            (LOCAL_CONSCRIPTION, PieceKind::Queen, PieceKind::Recruiter),
        ] {
            let mut state = position(
                &[
                    ("a1", Player::White, PieceKind::King),
                    ("h8", Player::Black, PieceKind::King),
                    ("d4", Player::White, before_kind),
                    ("f4", Player::Black, before_kind),
                ],
                Player::White,
            );
            let mut piece = state.board.get("d4".parse().unwrap()).unwrap();
            piece.capture_count = 3;
            state.board.set("d4".parse().unwrap(), Some(piece));
            let state = engine
                .acquire_for_setup(&state, id, Player::White)
                .unwrap()
                .state;
            let actions = engine.legal_actions(&state).unwrap();
            assert!(actions.contains(&card("d4")), "{id}");
            assert!(!actions.contains(&card("f4")), "enemy target: {id}");
            assert!(!actions.contains(&card("a1")), "wrong kind: {id}");
            let next = engine.apply(&state, card("d4")).unwrap();
            let transformed = next.state.board.get("d4".parse().unwrap()).unwrap();
            assert_eq!(
                (transformed.id, transformed.owner, transformed.kind),
                (piece.id, Player::White, after_kind)
            );
            assert_eq!(
                transformed.capture_count,
                if id == LOCAL_CONSCRIPTION { 0 } else { 3 }
            );
            assert_eq!(next.state.turn, Player::White);
            assert_eq!(next.state.ply, 0);
            assert_eq!(next.state.cards[0].charges, 0);
            assert!(
                !next
                    .events
                    .iter()
                    .any(|event| matches!(event, Event::RandomSample { .. }))
            );
        }
    }

    #[test]
    fn initiative_lasts_eight_owner_turns_and_releases_on_capture_or_real_check() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::Black, PieceKind::Rook),
                ("d2", Player::White, PieceKind::Pawn),
            ],
            Player::Black,
        );
        let state = engine
            .grant_card(&state, 1, INITIATIVE, Player::White)
            .unwrap()
            .state;
        assert_eq!(state.card_runtime.initiative[1].unwrap().limit, 8);
        assert!(!has_move(&engine, &state, "d4", "d2"));
        let mut expired = state.clone();
        expired.turns_taken[1] = 8;
        assert!(has_move(&engine, &expired, "d4", "d2"));
        let checking = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("g2", Player::White, PieceKind::Rook),
            ],
            Player::White,
        );
        let checking = engine
            .grant_card(&checking, 1, INITIATIVE, Player::White)
            .unwrap()
            .state;
        assert!(checking.card_runtime.initiative[1].is_some());
        let checked = engine.apply(&checking, movement("g2", "h2")).unwrap().state;
        assert!(checked.card_runtime.initiative[1].is_none());
        let mut protected = checking.clone();
        let king = protected.board.get("h8".parse().unwrap()).unwrap();
        protected.card_runtime.shielded_piece_ids.insert(king.id);
        // A consumable shield is a reaction and does not block the direct attack query.
        let checked = engine
            .apply(&protected, movement("g2", "h2"))
            .unwrap()
            .state;
        assert!(checked.card_runtime.initiative[1].is_none());
        let mut frozen = checking;
        frozen.card_runtime.frozen.insert(
            king.id,
            FrozenStatus {
                source: Player::White,
                remaining_turns: 3,
            },
        );
        let frozen = engine.apply(&frozen, movement("g2", "h2")).unwrap().state;
        assert!(frozen.card_runtime.initiative[1].is_some());
    }

    #[test]
    fn latest_highlander_requires_four_distinct_types_and_a_literal_king() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = base();
        for (square, kind) in [("b2", PieceKind::Knight), ("c2", PieceKind::Bishop)] {
            state.board.set(
                square.parse().unwrap(),
                Some(Piece::new(Player::White, kind).with_id(10 + square.as_bytes()[0] as u32)),
            );
        }
        let state = engine
            .grant_card(&state, 1, crate::HIGHLANDER, Player::White)
            .unwrap()
            .state;
        assert_eq!(state.status, GameStatus::Ongoing);
        let mut four = state.clone();
        four.board.set(
            "d2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook).with_id(200)),
        );
        engine.resolve_highlander_victory(&mut four);
        assert!(matches!(
            four.status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));
        let mut duplicate = state.clone();
        duplicate.board.set(
            "d2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Knight).with_id(200)),
        );
        engine.resolve_highlander_victory(&mut duplicate);
        assert_eq!(duplicate.status, GameStatus::Ongoing);
        let mut no_king = four;
        no_king.status = GameStatus::Ongoing;
        let mut king = no_king.board.get("a1".parse().unwrap()).unwrap();
        king.kind = PieceKind::RoyalKnight;
        no_king.board.set("a1".parse().unwrap(), Some(king));
        engine.resolve_highlander_victory(&mut no_king);
        assert_eq!(no_king.status, GameStatus::Ongoing);
    }

    #[test]
    fn latest_overwhelm_blocks_queen_capture_but_allows_king_capture() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("c4", Player::White, PieceKind::Queen),
                ("d4", Player::Black, PieceKind::King),
            ],
            Player::Black,
        );
        let state = engine
            .grant_card(&state, 1, OVERWHELM, Player::White)
            .unwrap()
            .state;
        assert!(!has_move(&engine, &state, "d4", "c4"));
        let state = position(
            &[
                ("c4", Player::White, PieceKind::King),
                ("d4", Player::Black, PieceKind::King),
            ],
            Player::Black,
        );
        let state = engine
            .grant_card(&state, 1, OVERWHELM, Player::White)
            .unwrap()
            .state;
        assert!(has_move(&engine, &state, "d4", "c4"));
        assert!(matches!(
            engine
                .apply(&state, movement("d4", "c4"))
                .unwrap()
                .state
                .status,
            GameStatus::Won {
                winner: Player::Black,
                ..
            }
        ));
    }

    #[test]
    fn rook_lift_requires_non_capture_and_caps_chain_at_three_retained_moves() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("d2", Player::White, PieceKind::King),
                ("e7", Player::Black, PieceKind::King),
                ("b1", Player::White, PieceKind::Rook),
            ],
            Player::White,
        );
        let mut state = engine
            .grant_card(&state, 1, ROOK_LIFT, Player::White)
            .unwrap()
            .state;
        for (index, (from, to)) in [("b1", "a1"), ("a1", "a8"), ("a8", "h8"), ("h8", "h1")]
            .into_iter()
            .enumerate()
        {
            state = engine.apply(&state, movement(from, to)).unwrap().state;
            assert_eq!(
                state.turn,
                if index < 3 {
                    Player::White
                } else {
                    Player::Black
                }
            );
        }
        assert!(state.card_runtime.october.rook_lift_chain.is_none());
        let mut capture = position(
            &[
                ("d2", Player::White, PieceKind::King),
                ("e7", Player::Black, PieceKind::King),
                ("b1", Player::White, PieceKind::Rook),
                ("a1", Player::Black, PieceKind::Bishop),
            ],
            Player::White,
        );
        capture = engine
            .grant_card(&capture, 1, ROOK_LIFT, Player::White)
            .unwrap()
            .state;
        capture = engine.apply(&capture, movement("b1", "a1")).unwrap().state;
        assert_eq!(capture.turn, Player::Black);
    }

    #[test]
    fn man_accepts_both_sides_native_minors_and_rejects_royals_and_majors() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Knight),
                ("f4", Player::Black, PieceKind::Wizard),
                ("b2", Player::White, PieceKind::Queen),
            ],
            Player::White,
        );
        let mut state = engine
            .grant_card(&state, 1, crate::MAN, Player::White)
            .unwrap()
            .state;
        let wizard = state.board.get("f4".parse().unwrap()).unwrap();
        state.card_runtime.wizard_mana.insert(wizard.id, 4);
        assert!(engine.legal_actions(&state).unwrap().contains(&card("d4")));
        assert!(!engine.legal_actions(&state).unwrap().contains(&card("b2")));
        assert!(!engine.legal_actions(&state).unwrap().contains(&card("a1")));
        let next = engine.apply(&state, card("f4")).unwrap().state;
        let man = next.board.get("f4".parse().unwrap()).unwrap();
        assert_eq!(
            (man.id, man.owner, man.kind),
            (wizard.id, Player::Black, PieceKind::Man)
        );
        assert!(!next.card_runtime.wizard_mana.contains_key(&wizard.id));
        state.card_runtime.royal_piece_ids.insert(wizard.id);
        assert!(!engine.legal_actions(&state).unwrap().contains(&card("f4")));
    }

    #[test]
    fn iqp_removes_current_enemy_c_e_pawns_after_draft_without_capture_reactions() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 1);
        // Move an original d-pawn into the c-file: current location is authoritative.
        let d = state.board.get("d7".parse().unwrap()).unwrap();
        state.board.set("d7".parse().unwrap(), None);
        state.board.set("c5".parse().unwrap(), Some(d));
        let c = state.board.get("c7".parse().unwrap()).unwrap();
        state.card_runtime.shielded_piece_ids.insert(c.id);
        state.card_runtime.recurrence_piece_ids.insert(c.id);
        state.phase = GamePhase::Draft(DraftState {
            phase: DraftPhase::Opening,
            offers: vec![crate::IQP.into()],
            resume_turn: Player::White,
        });
        let picked = engine
            .apply(&state, Action::DraftPick { offer_index: 0 })
            .unwrap();
        assert!(picked.state.board.get("c7".parse().unwrap()).is_some());
        assert!(picked.state.card_runtime.october.iqp_pending[0]);
        let next = engine
            .apply(&picked.state, Action::DraftPick { offer_index: 0 })
            .unwrap();
        assert!(next.state.board.get("c7".parse().unwrap()).is_none());
        assert!(next.state.board.get("e7".parse().unwrap()).is_none());
        assert!(next.state.board.get("c5".parse().unwrap()).is_none());
        assert!(next.state.board.get("c2".parse().unwrap()).is_some());
        assert!(next.state.card_runtime.pending_recurrences.is_empty());
        assert_eq!(next.state.card_runtime.october.iqp_pending, [false; 2]);
    }

    #[test]
    fn hope_wins_at_own_turn_start_and_never_on_acquisition_or_opponent_start() {
        let engine = Engine::bundled_latest().unwrap();
        let state = engine
            .grant_card(&base(), 1, crate::HOPE, Player::White)
            .unwrap()
            .state;
        assert_eq!(state.status, GameStatus::Ongoing);
        let state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert_eq!(state.status, GameStatus::Ongoing);
        let state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        assert_eq!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect(crate::HOPE.into())
            }
        );
    }

    #[test]
    fn hope_counts_allied_pieces_and_waits_for_standard_end_draft_unlock() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("b2", Player::White, PieceKind::Pawn),
                ("c2", Player::White, PieceKind::Pawn),
            ],
            Player::Black,
        );
        let mut state = engine
            .grant_card(&state, 1, crate::HOPE, Player::White)
            .unwrap()
            .state;
        engine.resolve_october_turn_start(&mut state, Player::White);
        assert_eq!(state.status, GameStatus::Ongoing);
        state.board.set("b2".parse().unwrap(), None);
        engine.resolve_october_turn_start(&mut state, Player::White);
        assert!(matches!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));

        let mut state = base();
        state.phase = GamePhase::Draft(DraftState {
            phase: DraftPhase::End,
            offers: vec![crate::HOPE.into()],
            resume_turn: Player::White,
        });
        state = engine
            .apply(&state, Action::DraftPick { offer_index: 0 })
            .unwrap()
            .state;
        assert!(!state.card_runtime.october.hope[0]);
        state = engine
            .apply(&state, Action::DraftPick { offer_index: 0 })
            .unwrap()
            .state;
        assert_eq!(state.status, GameStatus::Ongoing);
        assert!(!state.card_runtime.october.hope[0]);
        state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert_eq!(state.status, GameStatus::Ongoing);
        state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        assert!(state.card_runtime.october.hope[0]);
        assert!(matches!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));
    }

    #[test]
    fn assembly_is_passive_promotes_only_real_enemy_cross_captures_and_preserves_identity() {
        let engine = Engine::bundled_latest().unwrap();
        for (attacker, victim, expected) in [
            (PieceKind::Rook, PieceKind::Bishop, PieceKind::Queen),
            (PieceKind::Bishop, PieceKind::Rook, PieceKind::Queen),
            (PieceKind::Rook, PieceKind::Pawn, PieceKind::Rook),
        ] {
            let (from, to) = if attacker == PieceKind::Bishop {
                ("c3", "e5")
            } else {
                ("c3", "c5")
            };
            let state = position(
                &[
                    ("a1", Player::White, PieceKind::King),
                    ("h8", Player::Black, PieceKind::King),
                    (from, Player::White, attacker),
                    (to, Player::Black, victim),
                ],
                Player::White,
            );
            let id = state.board.get(from.parse().unwrap()).unwrap().id;
            let state = engine
                .grant_card(&state, 1, crate::ASSEMBLY, Player::White)
                .unwrap()
                .state;
            assert_eq!(state.cards[0].charges, 0);
            let next = engine.apply(&state, movement(from, to)).unwrap().state;
            let piece = next.board.get(to.parse().unwrap()).unwrap();
            assert_eq!((piece.id, piece.kind), (id, expected));
        }
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Rook),
                ("c5", Player::Black, PieceKind::Bishop),
            ],
            Player::White,
        );
        let mut state = engine
            .grant_card(&state, 1, crate::ASSEMBLY, Player::White)
            .unwrap()
            .state;
        let bishop = state.board.get("c5".parse().unwrap()).unwrap();
        state.card_runtime.shielded_piece_ids.insert(bishop.id);
        let next = engine.apply(&state, movement("c3", "c5")).unwrap().state;
        assert_eq!(
            next.board.get("c3".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        // No actual bishop capture occurred when a shield was consumed.
        assert!(next.board.find_by_id(bishop.id).is_some());
    }

    #[test]
    fn october_runtime_changes_search_and_repetition_identity() {
        let state = base();
        let mut changed = state.clone();
        changed.card_runtime.october.assembly[0] = true;
        assert_ne!(state.position_key(), changed.position_key());
        assert_ne!(state.fingerprint(), changed.fingerprint());
    }
}

//! October acquisition timing. Passive installation never goes through a player action.
use super::*;

impl Engine {
    pub(super) fn apply_acquired_card_definition(
        &self,
        state: &mut GameState,
        id: &str,
        owner: Player,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            self.apply_card_definition(state, id, owner, events);
            return Ok(());
        }
        match id {
            crate::GRUDGE | crate::PAWN_CHAIN | crate::PENTA_CHECK => {
                self.acquire_october_objective(state, id, owner, events)
            }
            FAST_GROWTH | FIANCHETTO | FIELD_PROMOTION => {
                self.apply_opening_card(state, owner, id, &ActionTarget::None, events)?;
            }
            crate::FALSE_START => self.apply_latest_false_start(state, owner),
            LONDON_SYSTEM => self.apply_latest_london(state, owner, events),
            BIG_BISHOP => self.apply_latest_big_bishop(state, owner, events),
            CHECKER => self.apply_latest_checker_acquisition(state, owner),
            OTHERWORLD => {
                let pawns: Vec<_> = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
                    .collect();
                // applyOpeningPassiveAcquisition.choose: no draw for zero/one target.
                // A previously offered target may disappear before a draft settles.
                if pawns.is_empty() {
                    if matches!(state.phase, GamePhase::Draft(_)) {
                        return Ok(());
                    }
                    return Err(EngineError::Invariant(
                        "이세계에 사용할 아군 폰이 없습니다.".into(),
                    ));
                }
                let index = if pawns.len() == 1 {
                    0
                } else {
                    let (roll, index) = state.rng_mut().web_choice_index(pawns.len());
                    events.push(Event::RandomSample {
                        purpose: "acquire.opening.otherworld.target".into(),
                        roll_53: roll,
                    });
                    index.expect("nonempty pawn candidates")
                };
                let (square, pawn) = pawns[index];
                let origin = state
                    .card_runtime
                    .piece_origins
                    .get(&pawn.id)
                    .copied()
                    .unwrap_or(square);
                self.delete_piece_for_card(state, square, pawn, owner, id, events);
                state
                    .card_runtime
                    .pending_otherworld
                    .push(PendingOtherworld {
                        piece_id: pawn.id,
                        owner,
                        square,
                        origin,
                        due_ply: state.ply.saturating_add(28),
                    });
            }
            crate::BLUE_JEANS => {
                let targets: Vec<_> = state
                    .board
                    .iter()
                    .filter(|(_, p)| crate::latest::latest_major_kind(p.kind))
                    .collect();
                for (square, piece) in targets {
                    self.delete_piece_for_card(state, square, piece, owner, id, events);
                }
            }
            crate::GOMOKU => state.card_runtime.gomoku[owner.index()] = true,
            _ => self.apply_card_definition(state, id, owner, events),
        }
        self.resolve_october_acquisitions(state, events);
        self.resolve_gomoku_victory(state);
        Ok(())
    }

    fn latest_install_layout(
        &self,
        state: &mut GameState,
        owner: Player,
        layout: &[(Square, PieceKind, bool)],
        footprint: &[Square],
        cause: &str,
        events: &mut Vec<Event>,
    ) {
        let targets: BTreeSet<_> = layout
            .iter()
            .map(|(s, _, _)| *s)
            .chain(footprint.iter().copied())
            .collect();
        let removed: Vec<_> = state
            .board
            .iter()
            .filter(|(square, piece)| {
                (cause == LONDON_SYSTEM && piece.owner == owner)
                    || targets.contains(square)
                    || state
                        .card_runtime
                        .large_pieces
                        .get(&piece.id)
                        .is_some_and(|large| {
                            large_piece_cells(large.anchor)
                                .iter()
                                .any(|cell| targets.contains(cell))
                        })
            })
            .collect();
        for (square, piece) in removed {
            self.delete_piece_for_card(state, square, piece, owner, cause, events);
        }
        for &(square, kind, moved) in layout {
            let piece = Piece::new(owner, kind).with_id(state.allocate_piece_id());
            state.board.set(square, Some(piece));
            state.card_runtime.piece_origins.insert(piece.id, square);
            if moved {
                state.card_runtime.moved_piece_ids.insert(piece.id);
            }
            // openingPassiveSkipsFreshCapture: London/BISHOP can capture immediately.
            if kind == PieceKind::BigBishop {
                state.card_runtime.large_pieces.insert(
                    piece.id,
                    LargePieceStatus {
                        anchor: square,
                        hp: 2,
                        max_hp: 2,
                    },
                );
            }
        }
        state.en_passant = None;
        let alive = |color| {
            state
                .board
                .iter()
                .any(|(_, p)| p.owner == color && self.is_royal_piece(state, p))
        };
        if !alive(Player::White) && !alive(Player::Black) {
            state.status = GameStatus::Draw {
                reason: DrawReason::CardEffect(cause.into()),
            };
        }
    }

    fn apply_latest_london(&self, state: &mut GameState, owner: Player, events: &mut Vec<Event>) {
        // londonLayout: the five untouched home pawns and home royals retain unmoved identity.
        let white = [
            ("a1", PieceKind::Rook, false),
            ("d1", PieceKind::Queen, false),
            ("e1", PieceKind::King, false),
            ("h1", PieceKind::Rook, false),
            ("f4", PieceKind::Bishop, true),
            ("d3", PieceKind::Bishop, true),
            ("f3", PieceKind::Knight, true),
            ("d2", PieceKind::Knight, true),
            ("a2", PieceKind::Pawn, false),
            ("b2", PieceKind::Pawn, false),
            ("c3", PieceKind::Pawn, true),
            ("d4", PieceKind::Pawn, true),
            ("e3", PieceKind::Pawn, true),
            ("f2", PieceKind::Pawn, false),
            ("g2", PieceKind::Pawn, false),
            ("h2", PieceKind::Pawn, false),
        ];
        let layout: Vec<_> = white
            .iter()
            .map(|(name, kind, moved)| {
                let white: Square = name.parse().expect("London layout");
                let square = if owner == Player::White {
                    white
                } else {
                    Square::new(white.file(), 7 - white.rank()).unwrap()
                };
                (square, *kind, *moved)
            })
            .collect();
        self.latest_install_layout(state, owner, &layout, &[], LONDON_SYSTEM, events);
    }

    fn apply_latest_big_bishop(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let horde = state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
            .count()
            > 8;
        let minor = if horde {
            PieceKind::Pawn
        } else if state.card_runtime.monochrome_chess {
            PieceKind::Camel
        } else {
            PieceKind::Knight
        };
        let anchor = opening_square(owner, "f2", "f8");
        let layout = [
            (anchor, PieceKind::BigBishop, false),
            (opening_square(owner, "f3", "f6"), PieceKind::Pawn, true),
            (opening_square(owner, "g3", "g6"), PieceKind::Pawn, true),
            (opening_square(owner, "h3", "h6"), minor, true),
        ];
        self.latest_install_layout(
            state,
            owner,
            &layout,
            &large_piece_cells(anchor),
            BIG_BISHOP,
            events,
        );
    }

    fn apply_latest_false_start(&self, state: &mut GameState, owner: Player) {
        fn root(parent: &[usize; 8], mut file: usize) -> usize {
            while parent[file] != file {
                file = parent[file];
            }
            file
        }
        let own: Vec<_> = state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner)
            .map(|(square, piece)| {
                let cells = state
                    .card_runtime
                    .large_pieces
                    .get(&piece.id)
                    .map_or_else(|| vec![square], |large| large_piece_cells(large.anchor));
                (square, piece, cells)
            })
            .collect();
        let mut parent = [0, 1, 2, 3, 4, 5, 6, 7];
        for (square, _, cells) in &own {
            for cell in cells {
                parent[root(&parent, usize::from(cell.file()))] =
                    root(&parent, usize::from(square.file()));
            }
        }
        let direction = if owner == Player::White { 1 } else { -1 };
        let mut shifts = [0_i8; 8];
        let mut blocked = [false; 8];
        for step in 1..=2 {
            for (square, _, cells) in &own {
                let group = root(&parent, usize::from(square.file()));
                if cells.iter().any(|cell| {
                    cell.offset(0, direction * step).is_none_or(|to| {
                        self.board_piece_at(state, to)
                            .is_some_and(|p| p.owner != owner)
                            || state.card_runtime.barricades.contains(&to)
                            || state.card_runtime.black_holes.contains(&to)
                            || state.card_runtime.crown_pos == Some(to)
                            || (state.card_runtime.collapsed_files_ranks
                                && (matches!(to.file(), 0 | 7) || matches!(to.rank(), 0 | 7)))
                            || state
                                .card_runtime
                                .pending_lobsters
                                .iter()
                                .any(|p| p.square == to)
                            || state
                                .card_runtime
                                .pending_scarecrows
                                .iter()
                                .any(|p| p.square == to)
                            || state
                                .card_runtime
                                .quantum_positions
                                .values()
                                .any(|p| *p == to)
                    })
                }) {
                    blocked[group] = true;
                }
            }
            for (square, _, _) in &own {
                let group = root(&parent, usize::from(square.file()));
                if !blocked[group] {
                    shifts[group] = step;
                }
            }
        }
        for (square, _, _) in &own {
            state.board.set(*square, None);
        }
        for (square, piece, _) in own {
            let group = root(&parent, usize::from(square.file()));
            let to = square
                .offset(0, direction * shifts[group])
                .expect("checked false-start destination");
            state.board.set(to, Some(piece));
            if let Some(large) = state.card_runtime.large_pieces.get_mut(&piece.id) {
                large.anchor = to;
            }
        }
        state.en_passant = None;
    }

    pub(super) fn latest_recurrence_targets(
        &self,
        state: &GameState,
        owner: Player,
    ) -> Vec<(Square, Piece)> {
        web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, p)| {
                p.owner == owner
                    && !matches!(
                        p.kind,
                        PieceKind::Pawn
                            | PieceKind::Monster
                            | PieceKind::Crown
                            | PieceKind::Football
                    )
                    && !self.is_royal_piece(state, *p)
                    && !state.card_runtime.recurrence_piece_ids.contains(&p.id)
            })
            .collect()
    }

    pub(super) fn apply_latest_recurrence(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let candidates = self.latest_recurrence_targets(state, owner);
        if candidates.is_empty() {
            return Err(EngineError::Invariant(
                "회귀를 부여할 기물이 없습니다.".into(),
            ));
        }
        // Active recurrence requests a draw even when there is only one candidate.
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        events.push(Event::RandomSample {
            purpose: "card.internal.recurrence.piece".into(),
            roll_53: roll,
        });
        state
            .card_runtime
            .recurrence_piece_ids
            .insert(candidates[index.unwrap()].1.id);
        Ok(())
    }

    pub(super) fn resolve_latest_gomoku(&self, state: &mut GameState) {
        let winners: Vec<_> = [Player::White, Player::Black]
            .into_iter()
            .filter(|owner| {
                state.card_runtime.gomoku[owner.index()]
                    && (0..8).any(|file| {
                        let mut count = 0;
                        (0..8).any(|rank| {
                            count = if state
                                .board
                                .get(Square::new(file, rank).unwrap())
                                .is_some_and(|p| p.owner == *owner)
                            {
                                count + 1
                            } else {
                                0
                            };
                            count >= 5
                        })
                    })
            })
            .collect();
        match winners.as_slice() {
            [winner] => {
                state.status = GameStatus::Won {
                    winner: *winner,
                    reason: WinReason::CardEffect(crate::GOMOKU.into()),
                }
            }
            [_, _] => {
                state.status = GameStatus::Draw {
                    reason: DrawReason::CardEffect(crate::GOMOKU.into()),
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game() -> GameState {
        let mut state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
        state.completed_drafts = [true; 3];
        state
    }
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
            7,
        );
        state.completed_drafts = [true; 3];
        state
    }
    fn movement(from: &str, to: &str) -> Action {
        Action::Move(MoveAction::new(
            from.parse().unwrap(),
            to.parse().unwrap(),
            None,
        ))
    }
    fn passive(engine: &Engine, id: &str) -> Transition {
        engine.grant_card(&game(), 1, id, Player::White).unwrap()
    }

    #[test]
    fn opening_flags_install_at_acquisition_without_a_charge_action_or_rng() {
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
    fn false_start_moves_before_first_move_and_preserves_identity_and_unmoved_flags() {
        let engine = Engine::bundled_latest().unwrap();
        for owner in [Player::White, Player::Black] {
            let before = game();
            let (from, to) = if owner == Player::White {
                ("e2", "e4")
            } else {
                ("e7", "e5")
            };
            let pawn = before.board.get(from.parse().unwrap()).unwrap();
            let next = engine
                .grant_card(&before, 1, crate::FALSE_START, owner)
                .unwrap()
                .state;
            assert_eq!(next.board.get(to.parse().unwrap()).unwrap().id, pawn.id);
            assert!(!next.card_runtime.moved_piece_ids.contains(&pawn.id));
            assert_eq!(
                next.card_runtime.piece_origins.get(&pawn.id),
                Some(&from.parse().unwrap())
            );
            assert_eq!(
                (next.ply, next.turn, next.rng_state()),
                (0, Player::White, before.rng_state())
            );
        }
    }

    #[test]
    fn london_and_bishop_install_immediately_without_fresh_capture_delay() {
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
        assert_eq!(bishop.card_runtime.large_pieces[&large.id].hp, 2);
        assert!(
            engine
                .legal_actions(&bishop)
                .unwrap()
                .contains(&movement("h3", "f4"))
        );
    }

    #[test]
    fn false_start_links_large_body_files_and_respects_reserved_cells() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = game();
        before.board.set(
            "g5".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Knight).with_id(100)),
        );
        let mut before = engine
            .grant_card(&before, 1, BIG_BISHOP, Player::White)
            .unwrap()
            .state;
        before
            .card_runtime
            .black_holes
            .insert("b3".parse().unwrap());
        let next = engine
            .grant_card(&before, 2, crate::FALSE_START, Player::White)
            .unwrap()
            .state;
        assert_eq!(
            next.board.get("f3".parse().unwrap()).unwrap().kind,
            PieceKind::BigBishop
        );
        assert_eq!(
            next.board.get("g4".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        assert_eq!(
            next.board.get("b2".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        let large = next.board.get("f3".parse().unwrap()).unwrap();
        assert_eq!(
            next.card_runtime.large_pieces[&large.id].anchor,
            "f3".parse().unwrap()
        );
    }

    #[test]
    fn otherworld_removes_now_skips_single_target_rng_and_returns_at_28_moves() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d5", Player::White, PieceKind::Pawn),
        ]);
        let pawn = before.board.get("d5".parse().unwrap()).unwrap();
        before
            .card_runtime
            .piece_origins
            .insert(pawn.id, "e2".parse().unwrap());
        let transition = engine
            .grant_card(&before, 1, OTHERWORLD, Player::White)
            .unwrap();
        let mut state = transition.state;
        assert!(state.board.get("d5".parse().unwrap()).is_none());
        assert_eq!(state.rng_state(), before.rng_state());
        assert_eq!(state.card_runtime.pending_otherworld[0].due_ply, 28);
        state.ply = 27;
        engine.resolve_after_counted_move(&mut state, &mut Vec::new());
        assert!(state.board.get("d5".parse().unwrap()).is_none());
        state.ply = 28;
        engine.resolve_after_counted_move(&mut state, &mut Vec::new());
        let wizard = state.board.get("d5".parse().unwrap()).unwrap();
        assert_eq!((wizard.id, wizard.kind), (pawn.id, PieceKind::Wizard));
        assert_eq!(
            state.card_runtime.piece_origins[&wizard.id],
            "e2".parse().unwrap()
        );
        let mut two = before.clone();
        two.board.set(
            "c4".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn).with_id(100)),
        );
        let result = engine
            .grant_card(&two, 1, OTHERWORLD, Player::White)
            .unwrap();
        assert_eq!(
            result
                .events
                .iter()
                .filter(|e| matches!(e, Event::RandomSample { .. }))
                .count(),
            1
        );
        assert_eq!(
            result,
            engine
                .grant_card(&two, 1, OTHERWORLD, Player::White)
                .unwrap()
        );
    }

    #[test]
    fn passive_draft_pick_applies_before_opponent_offer_and_is_not_applied_twice() {
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
    fn blue_jeans_removes_both_sides_current_majors_without_capture_credit() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Rook),
            ("f5", Player::Black, PieceKind::Amazon),
            ("b3", Player::White, PieceKind::Octopus),
            ("c3", Player::White, PieceKind::Wizard),
            ("c4", Player::Black, PieceKind::Recruiter),
            ("f4", Player::Black, PieceKind::Trickster),
        ]);
        let result = engine
            .grant_card(&before, 1, crate::BLUE_JEANS, Player::White)
            .unwrap();
        for square in ["d4", "f5", "b3"] {
            assert!(result.state.board.get(square.parse().unwrap()).is_none());
        }
        for square in ["a1", "h8", "c3", "c4", "f4"] {
            assert!(result.state.board.get(square.parse().unwrap()).is_some());
        }
        assert!(
            !result
                .events
                .iter()
                .any(|e| matches!(e, Event::PieceCaptured { .. }))
        );
        assert_eq!(result.state.cards[0].charges, 0);
    }

    #[test]
    fn end_passive_waits_for_unlock_then_installs_without_use_card() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = game();
        state.phase = GamePhase::Draft(DraftState {
            phase: DraftPhase::End,
            offers: vec![crate::BLUE_JEANS.into()],
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
        assert!(state.board.get("d1".parse().unwrap()).is_some());
        state = engine.apply(&state, movement("g1", "f3")).unwrap().state;
        assert!(state.board.get("d1".parse().unwrap()).is_some());
        let transition = engine.apply(&state, movement("g8", "f6")).unwrap();
        assert!(transition.state.board.get("d1".parse().unwrap()).is_none());
        assert!(
            transition
                .state
                .cards
                .iter()
                .find(|c| c.definition_id == crate::BLUE_JEANS)
                .unwrap()
                .available_after_turn
                .is_none()
        );
        assert!(!transition.events.iter().any(|e| matches!(e, Event::CardUsed { definition_id, .. } if definition_id == crate::BLUE_JEANS)));
    }

    #[test]
    fn gomoku_is_checked_on_acquisition_and_simultaneous_lines_draw() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
        ]);
        for rank in 1..5 {
            let piece = Piece::new(Player::White, PieceKind::Pawn).with_id(100 + rank as u32);
            before.board.set(Square::new(0, rank).unwrap(), Some(piece));
        }
        let won = engine
            .grant_card(&before, 1, crate::GOMOKU, Player::White)
            .unwrap();
        assert_eq!(
            won.state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect(crate::GOMOKU.into())
            }
        );
        let mut both = before;
        for rank in 3..7 {
            both.board.set(
                Square::new(7, rank).unwrap(),
                Some(Piece::new(Player::Black, PieceKind::Pawn).with_id(200 + rank as u32)),
            );
        }
        both.card_runtime.gomoku = [true; 2];
        engine.resolve_gomoku_victory(&mut both);
        assert_eq!(
            both.status,
            GameStatus::Draw {
                reason: DrawReason::CardEffect(crate::GOMOKU.into())
            }
        );
    }

    #[test]
    fn recurrence_is_active_does_not_fire_on_grant_and_draws_even_for_one_target() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("c3", Player::White, PieceKind::Knight),
            ("b2", Player::White, PieceKind::Pawn),
        ]);
        let granted = engine
            .grant_card(&before, 1, crate::RECURRENCE, Player::White)
            .unwrap();
        assert_eq!(granted.state.rng_state(), before.rng_state());
        assert!(granted.state.card_runtime.recurrence_piece_ids.is_empty());
        assert_eq!(granted.state.cards[0].charges, 1);
        let used = engine
            .apply(
                &granted.state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .unwrap();
        let knight = used.state.board.get("c3".parse().unwrap()).unwrap();
        assert_eq!(
            used.state.card_runtime.recurrence_piece_ids,
            [knight.id].into_iter().collect()
        );
        assert_eq!(
            used.events
                .iter()
                .filter(|e| matches!(e, Event::RandomSample { .. }))
                .count(),
            1
        );
        assert_eq!(used.state.ply, 0);
        assert_eq!(used.state.cards[0].charges, 0);
    }

    #[test]
    fn failed_otherworld_acquisition_is_transactional_and_passives_cannot_be_clicked() {
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
    fn fast_growth_also_restricts_the_second_capture_field_promotion_choice() {
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
}

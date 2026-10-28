//! Scoped v19 target/procedure ports; original contracts are indexed in cards/ported.json.
use super::*;

impl Engine {
    fn latest_poison_targets(&self, state: &GameState, actor: Player) -> Vec<(Square, Piece)> {
        web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, p)| {
                p.owner == actor
                    && p.kind == PieceKind::Pawn
                    && !state.card_runtime.poisoned_piece_ids.contains(&p.id)
            })
            .collect()
    }

    pub(super) fn apply_latest_changed_effect(
        &self,
        state: &mut GameState,
        actor: Player,
        id: &str,
        target: &ActionTarget,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        match id {
            HORSE_RIDING => state.card_runtime.horse_riding[actor.index()] = true,
            crate::REVERSAL => {
                let (square, piece) = piece_target(state, target)?;
                // rememberCapture:false: a sacrifice does not count as a capture.
                self.delete_piece_for_card(state, square, piece, actor, id, events);
                state.card_runtime.reversal[actor.index()] = true;
            }
            crate::POISONED_PAWN => {
                let mut candidates = self.latest_poison_targets(state, actor);
                if candidates.is_empty() {
                    return Err(EngineError::Invariant(
                        "독을 부여할 아군 폰이 없습니다.".into(),
                    ));
                }
                for ordinal in 0..candidates.len().min(3) {
                    // The active card draws once per pick, including a final singleton.
                    let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                    events.push(Event::RandomSample {
                        purpose: format!("card.v19.poisoned-pawn.{ordinal}"),
                        roll_53: roll,
                    });
                    let (_, piece) = candidates.remove(index.expect("nonempty poison candidates"));
                    state.card_runtime.poisoned_piece_ids.insert(piece.id);
                }
            }
            ORDINATION | PEGASUS => {
                let (square, mut piece) = piece_target(state, target)?;
                self.clear_piece_kind_runtime(state, piece.id);
                piece.kind = if id == ORDINATION {
                    PieceKind::Cardinal
                } else {
                    PieceKind::Unicorn
                };
                state.board.set(square, Some(piece));
                self.mark_transformed_piece(state, piece);
            }
            _ => {
                return Err(EngineError::Invariant(format!(
                    "unknown October effect {id}"
                )));
            }
        }
        self.resolve_gomoku_victory(state);
        Ok(())
    }

    pub(super) fn generate_latest_changed_effect_actions(
        &self,
        state: &GameState,
        card: &CardInstance,
        actions: &mut Vec<Action>,
    ) -> bool {
        let actor = state.turn;
        let mut push = |target| {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target,
            })
        };
        match card.definition_id.as_str() {
            HORSE_RIDING => {
                if !state.card_runtime.horse_riding[actor.index()]
                    && state
                        .board
                        .iter()
                        .any(|(_, p)| p.owner == actor && self.is_royal_piece(state, p))
                {
                    push(ActionTarget::None);
                }
            }
            crate::REVERSAL => {
                if !state.card_runtime.reversal[actor.index()] {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == actor
                            && crate::latest::latest_minor_kind(piece.kind)
                            && !self.is_royal_piece(state, piece)
                        {
                            push(ActionTarget::PieceAt(square));
                        }
                    }
                }
            }
            crate::POISONED_PAWN => {
                if !self.latest_poison_targets(state, actor).is_empty() {
                    push(ActionTarget::None);
                }
            }
            ORDINATION | PEGASUS => {
                let kind = if card.definition_id == ORDINATION {
                    PieceKind::Bishop
                } else {
                    PieceKind::Rook
                };
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor && piece.kind == kind {
                        push(ActionTarget::PieceAt(square));
                    }
                }
            }
            _ => return false,
        }
        true
    }

    pub(super) fn apply_latest_checker_acquisition(&self, state: &mut GameState, owner: Player) {
        let rank = if owner == Player::White { 2 } else { 5 };
        let cells = [0, 1, 6, 7].map(|file| Square::new(file, rank).unwrap());
        // acquireSeptember27OpeningPieces is all-or-nothing and quietly settles
        // the passive if any prescribed square is occupied/reserved.
        if !cells
            .iter()
            .all(|square| self.latest_checker_square_open(state, *square))
        {
            return;
        }
        for square in cells {
            let piece = Piece::new(owner, PieceKind::Checker).with_id(state.allocate_piece_id());
            state.board.set(square, Some(piece));
            self.register_spawned_piece(state, square, piece, true);
        }
    }

    pub(super) fn latest_checker_square_open(&self, state: &GameState, square: Square) -> bool {
        self.board_piece_at(state, square).is_none()
            && !state.card_runtime.barricades.contains(&square)
            && !state.card_runtime.black_holes.contains(&square)
            && state.card_runtime.crown_pos != Some(square)
            && !(state.card_runtime.collapsed_files_ranks
                && (matches!(square.file(), 0 | 7) || matches!(square.rank(), 0 | 7)))
            && !state
                .card_runtime
                .quantum_positions
                .values()
                .any(|s| *s == square)
            && !state
                .card_runtime
                .pending_lobsters
                .iter()
                .any(|p| p.square == square)
            && !state
                .card_runtime
                .pending_scarecrows
                .iter()
                .any(|p| p.square == square)
            && !state
                .card_runtime
                .pending_portals
                .iter()
                .any(|p| p.squares.contains(&square))
            && !state
                .card_runtime
                .portals
                .is_some_and(|p| p.contains(&square))
            && !state
                .card_runtime
                .rule_portals
                .is_some_and(|p| p.contains(&square))
    }

    pub(super) fn apply_latest_poison_from_events(
        &self,
        state: &mut GameState,
        attacker_id: u32,
        poisoned_before: &BTreeSet<u32>,
        events: &[Event],
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID || poisoned_before.is_empty() {
            return;
        }
        let Some((_, attacker)) = state.board.find_by_id(attacker_id) else {
            return;
        };
        let start = events
            .iter()
            .rposition(|e| matches!(e, Event::ActionApplied { .. }))
            .unwrap_or(0);
        let poisoned = events[start..].iter().any(|event| {
            let (actor, piece) = match event {
                Event::PieceCaptured { actor, piece, .. } => (*actor, piece),
                Event::PieceRemoved {
                    actor,
                    piece,
                    cause,
                    ..
                } if cause == "colossus-sector" => (*actor, piece),
                _ => return false,
            };
            actor == attacker.owner
                && piece.owner != attacker.owner
                && poisoned_before.contains(&piece.id)
        });
        if poisoned {
            // Capture turn is ticked immediately on handoff, leaving two future
            // own turns. Poison blocks action, but does not grant frozen immunity.
            let remaining = state
                .card_runtime
                .poison_stun_turns
                .entry(attacker.id)
                .or_default();
            *remaining = (*remaining).max(3);
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
            GameState::from_board(crate::LATEST_RULESET_ID, Mode::Standard, board, turn, 7);
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
    fn use_none() -> Action {
        Action::UseCard {
            instance_id: 1,
            target: ActionTarget::None,
        }
    }
    fn use_piece(square: &str) -> Action {
        Action::UseCard {
            instance_id: 1,
            target: ActionTarget::PieceAt(square.parse().unwrap()),
        }
    }
    fn has_move(engine: &Engine, state: &GameState, from: &str, to: &str) -> bool {
        engine
            .legal_actions(state)
            .unwrap()
            .contains(&movement(from, to))
    }
    fn has_moves_for(engine: &Engine, state: &GameState, source: &str) -> bool {
        let source: Square = source.parse().unwrap();
        engine
            .legal_actions(state)
            .unwrap()
            .iter()
            .any(|a| matches!(a, Action::Move(m) if m.from == source))
    }

    #[test]
    fn horse_riding_is_temporary_active_and_keeps_king_identity_and_knights() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(
            &[
                ("d4", Player::White, PieceKind::King),
                ("a2", Player::White, PieceKind::Knight),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        let id = before.board.get("d4".parse().unwrap()).unwrap().id;
        let mut state = engine
            .grant_card(&before, 1, HORSE_RIDING, Player::White)
            .unwrap()
            .state;
        assert!(!state.card_runtime.horse_riding[0]);
        assert!(!has_move(&engine, &state, "d4", "f5"));
        state = engine.apply(&state, use_none()).unwrap().state;
        assert_eq!(
            state.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::King
        );
        assert!(state.board.contains(Player::White, PieceKind::Knight));
        assert!(has_move(&engine, &state, "d4", "f5"));
        assert!(has_move(&engine, &state, "d4", "e4"));
        state.actions_remaining = 2;
        state = engine.apply(&state, movement("d4", "f5")).unwrap().state;
        assert_eq!(state.turn, Player::White);
        assert_eq!(state.board.get("f5".parse().unwrap()).unwrap().id, id);
        assert!(state.card_runtime.horse_riding[0]);
        state = engine.apply(&state, movement("a2", "b4")).unwrap().state;
        assert_eq!(state.turn, Player::Black);
        assert!(!state.card_runtime.horse_riding[0]);
        state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        assert!(!has_move(&engine, &state, "f5", "h6"));
    }

    #[test]
    fn reversal_sacrifices_one_native_minor_and_expires_after_own_turn() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(
            &[
                ("h1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("a2", Player::White, PieceKind::Knight),
                ("d4", Player::White, PieceKind::Rook),
                ("f1", Player::White, PieceKind::Bishop),
            ],
            Player::White,
        );
        let mut state = engine
            .acquire_for_setup(&before, crate::REVERSAL, Player::White)
            .unwrap()
            .state;
        assert!(!engine.legal_actions(&state).unwrap().contains(&use_none()));
        let knight = state.board.get("a2".parse().unwrap()).unwrap();
        state.card_runtime.recurrence_piece_ids.insert(knight.id);
        let used = engine.apply(&state, use_piece("a2")).unwrap();
        assert!(used.state.board.get("a2".parse().unwrap()).is_none());
        assert!(used.state.card_runtime.pending_recurrences.is_empty());
        assert!(
            !used
                .events
                .iter()
                .any(|e| matches!(e, Event::PieceCaptured { .. }))
        );
        assert_eq!(
            (used.state.turn, used.state.ply, used.state.rng_state()),
            (Player::White, 0, state.rng_state())
        );
        assert!(has_move(&engine, &used.state, "d4", "e5"));
        assert!(!has_move(&engine, &used.state, "d4", "d5"));
        assert!(has_move(&engine, &used.state, "f1", "f3"));
        assert!(!has_move(&engine, &used.state, "f1", "g2"));
        let mut after = engine
            .apply(&used.state, movement("d4", "e5"))
            .unwrap()
            .state;
        assert!(!after.card_runtime.reversal[0]);
        after = engine.apply(&after, movement("h8", "h7")).unwrap().state;
        assert!(has_move(&engine, &after, "e5", "e6"));
    }

    #[test]
    fn reversal_rejects_major_royal_enemy_and_empty_targets_transactionally() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Wizard),
                ("d4", Player::White, PieceKind::Rook),
                ("f4", Player::Black, PieceKind::Bishop),
            ],
            Player::White,
        );
        state = engine
            .grant_card(&state, 1, crate::REVERSAL, Player::White)
            .unwrap()
            .state;
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("c3"))
        );
        let wizard = state.board.get("c3".parse().unwrap()).unwrap();
        state.card_runtime.royal_piece_ids.insert(wizard.id);
        let unchanged = state.clone();
        for target in ["a1", "c3", "d4", "f4", "b2"] {
            assert!(engine.apply(&state, use_piece(target)).is_err());
        }
        assert_eq!(state, unchanged);
    }

    #[test]
    fn poisoned_pawn_samples_at_most_three_distinct_unpoisoned_pawns_with_exact_draw_count() {
        let engine = Engine::bundled_latest().unwrap();
        for count in 1..=5 {
            let mut state = position(
                &[
                    ("a1", Player::White, PieceKind::King),
                    ("h8", Player::Black, PieceKind::King),
                ],
                Player::White,
            );
            for file in 0..count {
                state.board.set(
                    Square::new(file, 1).unwrap(),
                    Some(Piece::new(Player::White, PieceKind::Pawn).with_id(100 + u32::from(file))),
                );
            }
            state = engine
                .grant_card(&state, 1, crate::POISONED_PAWN, Player::White)
                .unwrap()
                .state;
            let used = engine.apply(&state, use_none()).unwrap();
            assert_eq!(
                used.state.card_runtime.poisoned_piece_ids.len(),
                usize::from(count.min(3))
            );
            assert_eq!(
                used.events
                    .iter()
                    .filter(|e| matches!(e, Event::RandomSample { .. }))
                    .count(),
                usize::from(count.min(3))
            );
            assert_eq!(used, engine.apply(&state, use_none()).unwrap());
            assert!(engine.apply(&state, use_piece("a2")).is_err());
        }
        let mut state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
        state.completed_drafts = [true; 3];
        let prior = state.board.get("a2".parse().unwrap()).unwrap().id;
        state.card_runtime.poisoned_piece_ids.insert(prior);
        state = engine
            .grant_card(&state, 1, crate::POISONED_PAWN, Player::White)
            .unwrap()
            .state;
        let used = engine.apply(&state, use_none()).unwrap().state;
        assert_eq!(used.card_runtime.poisoned_piece_ids.len(), 4);
        assert!(used.card_runtime.poisoned_piece_ids.contains(&prior));
    }

    #[test]
    fn poison_stuns_two_future_own_turns_but_does_not_protect_the_capturer() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("h1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("a5", Player::Black, PieceKind::Pawn),
                ("b5", Player::Black, PieceKind::Rook),
            ],
            Player::White,
        );
        let pawn = state.board.get("a5".parse().unwrap()).unwrap();
        let rook = state.board.get("a1".parse().unwrap()).unwrap();
        state.card_runtime.poisoned_piece_ids.insert(pawn.id);
        state = engine.apply(&state, movement("a1", "a5")).unwrap().state;
        assert_eq!(state.card_runtime.poison_stun_turns.get(&rook.id), Some(&2));
        assert!(!state.card_runtime.frozen.contains_key(&rook.id));
        assert!(has_move(&engine, &state, "b5", "a5"));
        for (from, to, remaining) in [
            ("h8", "g8", 2),
            ("h1", "h2", 1),
            ("g8", "h8", 1),
            ("h2", "h3", 0),
            ("h8", "g8", 0),
        ] {
            state = engine.apply(&state, movement(from, to)).unwrap().state;
            assert_eq!(
                state
                    .card_runtime
                    .poison_stun_turns
                    .get(&rook.id)
                    .copied()
                    .unwrap_or(0),
                remaining
            );
            if state.turn == Player::White && remaining > 0 {
                assert!(!has_moves_for(&engine, &state, "a5"));
            }
        }
        assert!(has_move(&engine, &state, "a5", "a6"));
    }

    #[test]
    fn checker_poison_capture_stops_the_jump_chain_and_shield_contact_does_not_poison() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Checker),
                ("d4", Player::Black, PieceKind::Pawn),
                ("f6", Player::Black, PieceKind::Pawn),
            ],
            Player::White,
        );
        let pawn = before.board.get("d4".parse().unwrap()).unwrap();
        let checker = before.board.get("c3".parse().unwrap()).unwrap();
        before.card_runtime.poisoned_piece_ids.insert(pawn.id);
        let after = engine.apply(&before, movement("c3", "e5")).unwrap().state;
        assert_eq!(after.turn, Player::Black);
        assert_eq!(
            after.card_runtime.poison_stun_turns.get(&checker.id),
            Some(&2)
        );
        assert!(after.board.get("f6".parse().unwrap()).is_some());
        assert!(has_move(&engine, &after, "f6", "e5"));
        before.card_runtime.shielded_piece_ids.insert(pawn.id);
        let after = engine.apply(&before, movement("c3", "e5")).unwrap().state;
        assert!(
            !after
                .card_runtime
                .poison_stun_turns
                .contains_key(&checker.id)
        );
        assert!(after.board.find_by_id(pawn.id).is_some());
    }

    #[test]
    fn friendly_poison_capture_does_not_stun_and_poison_blocks_piece_abilities() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Knight),
                ("d5", Player::White, PieceKind::Pawn),
            ],
            Player::White,
        );
        let knight = state.board.get("c3".parse().unwrap()).unwrap();
        let pawn = state.board.get("d5".parse().unwrap()).unwrap();
        state.card_runtime.mad_horse[0] = true;
        state.card_runtime.poisoned_piece_ids.insert(pawn.id);
        let next = engine.apply(&state, movement("c3", "d5")).unwrap().state;
        assert!(!next.card_runtime.poison_stun_turns.contains_key(&knight.id));
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Wizard),
            ],
            Player::White,
        );
        let wizard = state.board.get("c3".parse().unwrap()).unwrap();
        state.card_runtime.wizard_mana.insert(wizard.id, 6);
        assert!(engine.legal_actions(&state).unwrap().iter().any(
            |a| matches!(a, Action::UsePieceAbility { piece_id, .. } if *piece_id == wizard.id)
        ));
        state.card_runtime.poison_stun_turns.insert(wizard.id, 2);
        assert!(!engine.legal_actions(&state).unwrap().iter().any(
            |a| matches!(a, Action::UsePieceAbility { piece_id, .. } if *piece_id == wizard.id)
        ));
    }

    #[test]
    fn checker_acquisition_adds_four_allies_and_preserves_original_pawns_for_both_colors() {
        let engine = Engine::bundled_latest().unwrap();
        for owner in [Player::White, Player::Black] {
            let before = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
            let next = engine
                .acquire_for_setup(&before, CHECKER, owner)
                .unwrap()
                .state;
            assert_eq!(next.board.iter().count(), 36);
            assert_eq!(
                next.board
                    .iter()
                    .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Checker)
                    .count(),
                4
            );
            let rank = if owner == Player::White { 2 } else { 5 };
            for file in [0, 1, 6, 7] {
                let square = Square::new(file, rank).unwrap();
                let piece = next.board.get(square).unwrap();
                assert_eq!(piece.kind, PieceKind::Checker);
                assert_eq!(next.card_runtime.piece_origins[&piece.id], square);
                assert!(next.card_runtime.moved_piece_ids.contains(&piece.id));
            }
            assert_eq!(
                next.board
                    .iter()
                    .filter(|(_, p)| p.kind == PieceKind::Pawn)
                    .count(),
                16
            );
            assert_eq!(
                (next.ply, next.rng_state(), next.cards[0].charges),
                (0, before.rng_state(), 0)
            );
        }
    }

    #[test]
    fn checker_acquisition_is_all_or_nothing_when_a_spawn_cell_is_reserved() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
        before.card_runtime.barricades.insert("a3".parse().unwrap());
        let next = engine
            .grant_card(&before, 1, CHECKER, Player::White)
            .unwrap()
            .state;
        assert!(!next.board.contains(Player::White, PieceKind::Checker));
        assert_eq!(next.board, before.board);
        assert_eq!(next.cards[0].charges, 0);
    }

    #[test]
    fn otherworld_return_is_not_a_reserved_spawn_cell_for_checker_or_false_start() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("a3", Player::White, PieceKind::Pawn),
            ],
            Player::White,
        );
        let mut before = engine
            .grant_card(&before, 1, OTHERWORLD, Player::White)
            .unwrap()
            .state;
        let checker = engine
            .grant_card(&before, 2, CHECKER, Player::White)
            .unwrap()
            .state;
        assert_eq!(
            checker
                .board
                .iter()
                .filter(|(_, p)| p.kind == PieceKind::Checker)
                .count(),
            4
        );
        before.board.set(
            "a2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn).with_id(100)),
        );
        let shifted = engine
            .grant_card(&before, 2, crate::FALSE_START, Player::White)
            .unwrap()
            .state;
        assert_eq!(
            shifted.board.get("a4".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
    }

    #[test]
    fn ordination_needs_one_bishop_without_sacrifice_and_pegasus_targets_a_rook() {
        let engine = Engine::bundled_latest().unwrap();
        for (id, source, target) in [
            (ORDINATION, PieceKind::Bishop, PieceKind::Cardinal),
            (PEGASUS, PieceKind::Rook, PieceKind::Unicorn),
        ] {
            let before = position(
                &[
                    ("a1", Player::White, PieceKind::King),
                    ("h8", Player::Black, PieceKind::King),
                    ("d4", Player::White, source),
                    ("b1", Player::White, PieceKind::Knight),
                    ("f4", Player::Black, source),
                ],
                Player::White,
            );
            let state = engine
                .acquire_for_setup(&before, id, Player::White)
                .unwrap()
                .state;
            let id_before = state.board.get("d4".parse().unwrap()).unwrap().id;
            assert!(
                engine
                    .legal_actions(&state)
                    .unwrap()
                    .contains(&use_piece("d4"))
            );
            assert!(
                !engine
                    .legal_actions(&state)
                    .unwrap()
                    .contains(&use_piece("f4"))
            );
            assert!(
                !engine
                    .legal_actions(&state)
                    .unwrap()
                    .contains(&use_piece("b1"))
            );
            let next = engine.apply(&state, use_piece("d4")).unwrap();
            let piece = next.state.board.get("d4".parse().unwrap()).unwrap();
            assert_eq!(
                (piece.id, piece.owner, piece.kind),
                (id_before, Player::White, target)
            );
            assert_eq!(next.state.board.iter().count(), before.board.iter().count());
            assert!(
                !next
                    .events
                    .iter()
                    .any(|e| matches!(e, Event::PieceRemoved { .. } | Event::PieceCaptured { .. }))
            );
        }
    }

    #[test]
    fn pegasus_uses_the_latest_end_draft_lock_before_rook_target_actions_appear() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
        state.phase = GamePhase::Draft(DraftState {
            phase: DraftPhase::End,
            offers: vec![PEGASUS.into()],
            resume_turn: Player::White,
        });
        state = engine
            .apply(&state, Action::DraftPick { offer_index: 0 })
            .unwrap()
            .state;
        let GamePhase::Draft(mut draft) = state.phase.clone() else {
            panic!("black draft");
        };
        draft.offers = vec![crate::HOPE.into()];
        state.phase = GamePhase::Draft(draft);
        state = engine
            .apply(&state, Action::DraftPick { offer_index: 0 })
            .unwrap()
            .state;
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("a1"))
        );
        state = engine.apply(&state, movement("g1", "f3")).unwrap().state;
        state = engine.apply(&state, movement("g8", "f6")).unwrap().state;
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("a1"))
        );
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .contains(&use_piece("b1"))
        );
    }
}

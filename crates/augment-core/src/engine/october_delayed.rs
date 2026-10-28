//! Scoped October reservations. Timers count completed owner turns, not actions.
use super::*;
use crate::state::{PendingAscension, PendingSeal, PendingTaboo};

impl Engine {
    pub(super) fn generate_october_delayed_actions(
        &self,
        state: &GameState,
        card: &CardInstance,
        actions: &mut Vec<Action>,
    ) {
        for (square, piece) in state.board.iter() {
            let allowed = !self.is_royal_piece(state, piece)
                && match card.definition_id.as_str() {
                    crate::SEAL => {
                        piece.owner != state.turn
                            && !matches!(
                                piece.kind,
                                PieceKind::King
                                    | PieceKind::Monster
                                    | PieceKind::Football
                                    | PieceKind::Crown
                            )
                    }
                    crate::GREEK_GIFT => {
                        piece.owner == state.turn && piece.kind == PieceKind::Bishop
                    }
                    _ => piece.owner == state.turn && piece.kind == PieceKind::Queen,
                };
            if !allowed {
                continue;
            }
            if matches!(card.definition_id.as_str(), crate::ASCENSION | crate::TABOO) {
                for index in 0..64 {
                    let landing = Square::from_index(index).unwrap();
                    if self.latest_checker_square_open(state, landing) {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Squares(vec![square, landing]),
                        });
                    }
                }
            } else {
                actions.push(Action::UseCard {
                    instance_id: card.instance_id,
                    target: ActionTarget::PieceAt(square),
                });
            }
        }
    }

    pub(super) fn apply_october_delayed_card(
        &self,
        state: &mut GameState,
        actor: Player,
        id: &str,
        target: &ActionTarget,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        match id {
            crate::SEAL => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.october.pending_seals.push(PendingSeal {
                    by: actor,
                    piece_id: piece.id,
                    due_turn: state.turns_taken[actor.index()].saturating_add(1),
                });
            }
            crate::GREEK_GIFT => {
                let (square, piece) = piece_target(state, target)?;
                self.delete_piece_for_card(state, square, piece, actor, id, events);
                state.card_runtime.october.greek_gift_pending[actor.index()] = true;
            }
            crate::ASCENSION => {
                let [square, landing] = two_square_target(target)?;
                let piece = state
                    .board
                    .get(square)
                    .ok_or_else(|| EngineError::Invariant("missing ascension queen".into()))?;
                let mut saved = CardRuntimeState::default();
                end::restore_piece_runtime(&mut saved, &state.card_runtime, &[piece.id]);
                self.delete_piece_for_card(state, square, piece, actor, id, events);
                state
                    .card_runtime
                    .october
                    .pending_ascensions
                    .push(PendingAscension {
                        piece,
                        square: landing,
                        due_turn: state.turns_taken[actor.index()].saturating_add(3),
                        runtime: Box::new(saved),
                    });
            }
            crate::TABOO => {
                let [square, landing] = two_square_target(target)?;
                let piece = state
                    .board
                    .get(square)
                    .ok_or_else(|| EngineError::Invariant("missing taboo queen".into()))?;
                state.card_runtime.october.pending_taboo.push(PendingTaboo {
                    owner: actor,
                    piece_id: piece.id,
                    square: landing,
                });
            }
            _ => unreachable!(),
        }
        self.resolve_gomoku_victory(state);
        Ok(())
    }

    pub(super) fn release_october_bounds_after_move(
        &self,
        state: &mut GameState,
        moving: Piece,
        before: &crate::Board,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            return;
        }
        let relocated = before
            .find_by_id(moving.id)
            .zip(state.board.find_by_id(moving.id))
            .is_some_and(|((old, _), (new, _))| old != new);
        if relocated {
            let release: Vec<_> = state
                .board
                .iter()
                .filter(|(_, p)| p.id != moving.id && p.owner == moving.owner)
                .map(|(_, p)| p.id)
                .collect();
            for id in release {
                state.card_runtime.october.bound_piece_ids.remove(&id);
            }
        }
    }

    pub(super) fn resolve_october_delayed_start(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID || state.status != GameStatus::Ongoing {
            return;
        }
        let seals = std::mem::take(&mut state.card_runtime.october.pending_seals);
        for seal in seals {
            if seal.by != owner || state.turns_taken[owner.index()] < seal.due_turn {
                state.card_runtime.october.pending_seals.push(seal);
            } else if let Some((_, piece)) = state.board.find_by_id(seal.piece_id)
                && piece.owner == owner.opponent()
                && !self.is_royal_piece(state, piece)
                && piece.kind != PieceKind::King
            {
                state.card_runtime.october.bound_piece_ids.insert(piece.id);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("seal-bound:{}", piece.id),
                });
            }
        }
        let pending = std::mem::take(&mut state.card_runtime.october.pending_ascensions);
        for entry in pending {
            if entry.piece.owner != owner || state.turns_taken[owner.index()] < entry.due_turn {
                state.card_runtime.october.pending_ascensions.push(entry);
                continue;
            }
            // Landing clears alternate quantum occupancy without killing its real piece.
            state
                .card_runtime
                .quantum_positions
                .retain(|_, s| *s != entry.square);
            self.crush_october_landing(state, entry.square, owner, crate::ASCENSION, events);
            state.card_runtime.barricades.remove(&entry.square);
            if !self.october_collapsed_cell(state, entry.square)
                && self.board_piece_at(state, entry.square).is_none()
            {
                state.board.set(entry.square, Some(entry.piece));
                end::restore_piece_runtime(
                    &mut state.card_runtime,
                    &entry.runtime,
                    &[entry.piece.id],
                );
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("ascension-return:{}:{}", entry.piece.id, entry.square),
                });
            }
        }
        self.resolve_gomoku_victory(state);
    }

    fn october_collapsed_cell(&self, state: &GameState, square: Square) -> bool {
        state.card_runtime.collapsed_files_ranks
            && (matches!(square.file(), 0 | 7) || matches!(square.rank(), 0 | 7))
    }

    fn crush_october_landing(
        &self,
        state: &mut GameState,
        square: Square,
        owner: Player,
        cause: &str,
        events: &mut Vec<Event>,
    ) {
        if let Some((anchor, piece)) = self.board_piece_entry_at(state, square) {
            // Reuse the current core's large-body hit abstraction. Cell fracture
            // parity remains outside these scoped ports, as for other landings.
            if let Some(large) = state.card_runtime.large_pieces.get_mut(&piece.id) {
                large.hp = large.hp.saturating_sub(1);
                if large.hp > 0 {
                    return;
                }
            }
            self.remove_piece_for_card(state, anchor, piece, owner, cause, events);
        }
    }

    pub(super) fn resolve_october_delayed_end(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID || state.status != GameStatus::Ongoing {
            return;
        }
        if std::mem::take(&mut state.card_runtime.october.greek_gift_pending[actor.index()]) {
            let kings: Vec<_> = state
                .board
                .iter()
                .filter(|(_, p)| p.owner == actor.opponent() && self.is_royal_piece(state, *p))
                .flat_map(|(s, p)| {
                    if state.card_runtime.large_pieces.contains_key(&p.id) {
                        large_piece_cells(s)
                    } else {
                        vec![s]
                    }
                })
                .collect();
            let victims: Vec<_> = state
                .board
                .iter()
                .filter(|(s, p)| {
                    p.owner == actor.opponent()
                        && p.kind == PieceKind::Pawn
                        && !self.is_royal_piece(state, *p)
                        && kings.iter().any(|king| {
                            s.file()
                                .abs_diff(king.file())
                                .max(s.rank().abs_diff(king.rank()))
                                == 1
                        })
                })
                .collect();
            for (s, p) in victims {
                self.remove_piece_for_card(state, s, p, actor, crate::GREEK_GIFT, events);
            }
            self.resolve_missing_king(state, events);
        }
        let pending = std::mem::take(&mut state.card_runtime.october.pending_taboo);
        for entry in pending {
            if entry.owner != actor {
                state.card_runtime.october.pending_taboo.push(entry);
                continue;
            }
            if state.status != GameStatus::Ongoing {
                continue;
            }
            let Some((source, piece)) = state.board.find_by_id(entry.piece_id) else {
                continue;
            };
            if piece.owner != actor {
                continue;
            }
            // Follow the identity even after a move or transformation this turn.
            self.delete_piece_for_card(state, source, piece, actor, crate::TABOO, events);
            self.resolve_missing_king(state, events);
            if state.status != GameStatus::Ongoing {
                continue;
            }
            let mut seen = BTreeSet::new();
            for square in [source, entry.square] {
                if !seen.insert(square) {
                    continue;
                }
                self.crush_october_landing(state, square, actor, crate::TABOO, events);
                // Source removes movable wall/football/monster installations.
                state.card_runtime.barricades.remove(&square);
                if self.latest_checker_square_open(state, square) {
                    let monster =
                        Piece::new(actor, PieceKind::Monster).with_id(state.allocate_piece_id());
                    state.board.set(square, Some(monster));
                    state.card_runtime.moved_piece_ids.insert(monster.id);
                    state.card_runtime.piece_origins.insert(monster.id, square);
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("taboo-monster:{}:{}", monster.id, square),
                    });
                }
            }
            self.resolve_missing_king(state, events);
        }
        self.resolve_gomoku_victory(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(extra: &[(&str, Player, PieceKind)]) -> GameState {
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
    fn at(s: &GameState, q: &str) -> Piece {
        s.board.get(q.parse().unwrap()).unwrap()
    }
    fn movement(a: &str, b: &str) -> Action {
        Action::Move(MoveAction::new(
            a.parse().unwrap(),
            b.parse().unwrap(),
            None,
        ))
    }
    fn use_at(s: &mut GameState, engine: &Engine, id: &str, target: ActionTarget) {
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
    fn single(q: &str) -> ActionTarget {
        ActionTarget::PieceAt(q.parse().unwrap())
    }
    fn pair(a: &str, b: &str) -> ActionTarget {
        ActionTarget::Squares(vec![a.parse().unwrap(), b.parse().unwrap()])
    }
    fn handoff(engine: &Engine, s: &mut GameState) {
        let actor = s.turn;
        engine.finish_turn(s, actor, &mut Vec::new()).unwrap();
    }
    fn three_owner_starts(engine: &Engine, s: &mut GameState) {
        for _ in 0..6 {
            handoff(engine, s);
        }
    }
    fn has_move(engine: &Engine, s: &GameState, a: &str, b: &str) -> bool {
        engine.legal_actions(s).unwrap().contains(&movement(a, b))
    }

    #[test]
    fn seal_tracks_identity_at_next_owner_start_and_ally_move_releases_without_immunity() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d5", Player::Black, PieceKind::Rook),
            ("c3", Player::White, PieceKind::Bishop),
        ]);
        let id = at(&state, "d5").id;
        use_at(&mut state, &engine, crate::SEAL, single("d5"));
        assert!(state.card_runtime.october.bound_piece_ids.is_empty());
        state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert!(has_move(&engine, &state, "d5", "d4"));
        state = engine.apply(&state, movement("d5", "d4")).unwrap().state;
        assert!(state.card_runtime.october.bound_piece_ids.contains(&id));
        assert!(state.card_runtime.october.pending_seals.is_empty());
        assert!(
            has_move(&engine, &state, "c3", "d4"),
            "bound grants no capture protection"
        );
        state = engine.apply(&state, movement("a2", "a3")).unwrap().state;
        assert!(!has_move(&engine, &state, "d4", "d3"));
        state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        assert!(!state.card_runtime.october.bound_piece_ids.contains(&id));
    }

    #[test]
    fn seal_rechecks_dead_converted_and_royal_targets_without_hitting_replacements() {
        let engine = Engine::bundled_latest().unwrap();
        for variant in 0..3 {
            let mut s = position(&[("d5", Player::Black, PieceKind::Rook)]);
            let id = at(&s, "d5").id;
            use_at(&mut s, &engine, crate::SEAL, single("d5"));
            let mut p = at(&s, "d5");
            match variant {
                0 => p.id = 100,
                1 => p.owner = Player::White,
                _ => {
                    s.card_runtime.royal_piece_ids.insert(id);
                }
            }
            s.board.set("d5".parse().unwrap(), Some(p));
            s.turns_taken[0] = 1;
            engine.resolve_october_delayed_start(&mut s, Player::White, &mut Vec::new());
            assert!(s.card_runtime.october.bound_piece_ids.is_empty());
            assert!(s.card_runtime.october.pending_seals.is_empty());
        }
    }

    #[test]
    fn bound_checker_has_no_forced_jump_and_bound_rook_cannot_be_swapped_or_castled() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("d4", Player::White, PieceKind::Checker),
            ("e5", Player::Black, PieceKind::Pawn),
            ("c2", Player::White, PieceKind::Dragon),
            ("h1", Player::White, PieceKind::Rook),
        ]);
        for q in ["d4", "h1"] {
            s.card_runtime.october.bound_piece_ids.insert(at(&s, q).id);
        }
        assert!(!has_move(&engine, &s, "d4", "f6"));
        assert!(!has_move(&engine, &s, "c2", "h1"));
        assert!(has_move(&engine, &s, "a1", "a2"));
        let white_king = at(&s, "a1");
        s.board.set("a1".parse().unwrap(), None);
        s.board.set("e1".parse().unwrap(), Some(white_king));
        assert!(!has_move(&engine, &s, "e1", "g1"));
    }

    #[test]
    fn greek_gift_sacrifice_is_noncapture_and_waits_for_actual_turn_end() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("c1", Player::White, PieceKind::Bishop),
            ("g7", Player::Black, PieceKind::Pawn),
            ("h7", Player::Black, PieceKind::Pawn),
            ("g8", Player::Black, PieceKind::Pawn),
            ("b7", Player::Black, PieceKind::Pawn),
            ("g6", Player::White, PieceKind::Pawn),
            ("f8", Player::Black, PieceKind::Knight),
        ]);
        let bishop = at(&s, "c1");
        let protected = at(&s, "g7");
        let royal = at(&s, "g8");
        s.card_runtime.recurrence_piece_ids.insert(bishop.id);
        s.card_runtime.shielded_piece_ids.insert(protected.id);
        s.card_runtime.royal_piece_ids.insert(royal.id);
        use_at(&mut s, &engine, crate::GREEK_GIFT, single("c1"));
        assert!(s.board.get("c1".parse().unwrap()).is_none());
        assert!(s.card_runtime.pending_recurrences.is_empty());
        s.actions_remaining = 2;
        s = engine.apply(&s, movement("a1", "a2")).unwrap().state;
        assert!(s.card_runtime.october.greek_gift_pending[0]);
        assert_eq!(at(&s, "g7"), protected);
        s = engine.apply(&s, movement("a2", "a3")).unwrap().state;
        for q in ["g7", "h7"] {
            assert!(s.board.get(q.parse().unwrap()).is_none());
        }
        for q in ["g8", "b7", "g6"] {
            assert!(s.board.get(q.parse().unwrap()).is_some());
        }
        assert!(!s.card_runtime.october.greek_gift_pending[0]);
    }

    #[test]
    fn taboo_follows_moving_queen_crushes_landing_and_spawns_neutral_monsters() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
        let queen = at(&s, "d1");
        use_at(&mut s, &engine, crate::TABOO, pair("d1", "e5"));
        assert_eq!(at(&s, "d1"), queen);
        assert!(s.board.get("e5".parse().unwrap()).is_none());
        s.board.set(
            "e5".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Pawn).with_id(100)),
        );
        s.card_runtime.recurrence_piece_ids.insert(queen.id);
        s = engine.apply(&s, movement("d1", "d4")).unwrap().state;
        assert!(s.board.find_by_id(queen.id).is_none());
        assert!(s.card_runtime.pending_recurrences.is_empty());
        for q in ["d4", "e5"] {
            assert_eq!(at(&s, q).kind, PieceKind::Monster);
        }
        assert!(s.card_runtime.october.pending_taboo.is_empty());
        s.turn = Player::White;
        s.card_runtime.royal_command[0] = Some(crate::RoyalCommandStatus {
            starts_at_turn: 0,
            expires_at_turn: 100,
        });
        assert!(
            !has_move(&engine, &s, "d4", "c4"),
            "neutral monsters do not inherit the summoner's grants"
        );
    }

    #[test]
    fn taboo_deduplicates_landings_and_cancels_when_identity_is_gone_or_enemy() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
        use_at(&mut s, &engine, crate::TABOO, pair("d1", "d4"));
        let out = engine.apply(&s, movement("d1", "d4")).unwrap();
        assert_eq!(
            out.state
                .board
                .iter()
                .filter(|(_, p)| p.kind == PieceKind::Monster)
                .count(),
            1
        );
        for convert in [false, true] {
            let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
            use_at(&mut s, &engine, crate::TABOO, pair("d1", "d4"));
            let mut queen = at(&s, "d1");
            queen.owner = Player::Black;
            s.board.set("d1".parse().unwrap(), convert.then_some(queen));
            handoff(&engine, &mut s);
            assert!(!s.board.iter().any(|(_, p)| p.kind == PieceKind::Monster));
            assert!(s.card_runtime.october.pending_taboo.is_empty());
        }
    }

    #[test]
    fn ascension_returns_on_third_owner_start_with_frozen_metadata_and_original_identity() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
        let mut queen = at(&s, "d1");
        queen.capture_count = 4;
        s.board.set("d1".parse().unwrap(), Some(queen));
        s.card_runtime.shielded_piece_ids.insert(queen.id);
        s.card_runtime.frozen.insert(
            queen.id,
            FrozenStatus {
                source: Player::Black,
                remaining_turns: 2,
            },
        );
        s.card_runtime.recurrence_piece_ids.insert(queen.id);
        use_at(&mut s, &engine, crate::ASCENSION, pair("d1", "e5"));
        assert!(s.board.find_by_id(queen.id).is_none());
        assert!(!s.card_runtime.shielded_piece_ids.contains(&queen.id));
        let pending = &s.card_runtime.october.pending_ascensions[0];
        assert!(pending.runtime.october.pending_ascensions.is_empty());
        assert_eq!(pending.due_turn, 3);
        for _ in 0..4 {
            handoff(&engine, &mut s);
            assert!(s.board.find_by_id(queen.id).is_none());
        }
        s.board.set(
            "e5".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Pawn).with_id(100)),
        );
        s.card_runtime.shielded_piece_ids.insert(100);
        handoff(&engine, &mut s);
        handoff(&engine, &mut s);
        assert_eq!(at(&s, "e5"), queen);
        assert!(s.card_runtime.recurrence_piece_ids.contains(&queen.id));
        assert!(s.card_runtime.october.pending_ascensions.is_empty());
        assert!(s.card_runtime.shielded_piece_ids.contains(&queen.id));
        assert_eq!(s.card_runtime.frozen[&queen.id].remaining_turns, 2);
        assert!(!s.card_runtime.moved_piece_ids.contains(&queen.id));
        assert!(
            !s.card_runtime
                .fresh_no_capture_until
                .contains_key(&queen.id)
        );
    }

    #[test]
    fn ascension_extra_actions_do_not_advance_return_and_collapsed_landing_consumes_reservation() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
        use_at(&mut s, &engine, crate::ASCENSION, pair("d1", "a5"));
        s.actions_remaining = 2;
        s = engine.apply(&s, movement("a1", "b2")).unwrap().state;
        assert_eq!(s.turn, Player::White);
        assert_eq!(s.turns_taken[0], 0);
        assert_eq!(s.card_runtime.october.pending_ascensions[0].due_turn, 3);
        let black = at(&s, "h8");
        s.board.set("h8".parse().unwrap(), None);
        s.board.set("g7".parse().unwrap(), Some(black));
        s.card_runtime.collapsed_files_ranks = true;
        three_owner_starts(&engine, &mut s);
        assert!(s.card_runtime.october.pending_ascensions.is_empty());
        assert!(s.board.get("a5".parse().unwrap()).is_none());
    }

    #[test]
    fn reservations_affect_repetition_and_reject_reserved_initial_landing() {
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
    fn countdowns_distinguish_positions_and_search_branches_keep_reservations_isolated() {
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
}

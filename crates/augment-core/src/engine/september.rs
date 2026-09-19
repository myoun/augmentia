use super::*;

impl Engine {
    /// The published major family uses identity, not Trickster's sampled movement kind.
    fn majesty_is_major(kind: PieceKind) -> bool {
        matches!(
            kind,
            PieceKind::Queen
                | PieceKind::Rook
                | PieceKind::Amazon
                | PieceKind::Man
                | PieceKind::Colossus
                | PieceKind::BigRook
                | PieceKind::Herald
                | PieceKind::Jester
                | PieceKind::Recruiter
                | PieceKind::Hook
                | PieceKind::PrimeMinister
                | PieceKind::Assassin
                | PieceKind::Wizard
                | PieceKind::Windmill
                | PieceKind::Crown
                | PieceKind::Bear
                | PieceKind::MagicGirl
                | PieceKind::Berserker
                | PieceKind::Siren
                | PieceKind::Trickster
                | PieceKind::Reaper
                | PieceKind::Undead
        )
    }

    pub(super) fn majesty_destination_blocked(
        &self,
        state: &GameState,
        piece: Piece,
        to: Square,
    ) -> bool {
        let identity = state.board.find_by_id(piece.id).map_or(piece, |(_, p)| p);
        let owner = identity.owner.opponent();
        if !state.card_runtime.majesty[owner.index()] || !Self::majesty_is_major(identity.kind) {
            return false;
        }
        let cells = if matches!(identity.kind, PieceKind::Colossus | PieceKind::BigRook)
            || state.card_runtime.large_pieces.contains_key(&identity.id)
        {
            large_piece_cells(to)
        } else {
            vec![to]
        };
        state.board.iter().any(|(square, royal)| {
            royal.owner == owner
                && (matches!(
                    royal.kind,
                    PieceKind::King | PieceKind::RoyalKnight | PieceKind::DarkWizard
                ) || state.card_runtime.royal_piece_ids.contains(&royal.id))
                && cells.iter().any(|cell| {
                    cell.file()
                        .abs_diff(square.file())
                        .max(cell.rank().abs_diff(square.rank()))
                        == 1
                })
        })
    }

    pub(super) fn install_capture_the_flag(&self, state: &mut GameState, events: &mut Vec<Event>) {
        let mut flags = [Square::new(0, 0).unwrap(), Square::new(0, 7).unwrap()];
        for owner in [Player::White, Player::Black] {
            let (roll, file) = state.rng_mut().web_choice_index(8);
            flags[owner.index()] = Square::new(
                file.unwrap() as u8,
                if owner == Player::White { 0 } else { 7 },
            )
            .unwrap();
            events.push(Event::RandomSample {
                purpose: format!("capture-the-flag:{owner}"),
                roll_53: roll,
            });
        }
        state.card_runtime.capture_the_flag = Some(crate::CaptureTheFlagState {
            flags,
            occupations: [None, None],
        });
    }

    /// Source septemberAdvanceFlags: deadlines count the flag owner's completed turns.
    pub(super) fn advance_capture_flags(
        &self,
        state: &mut GameState,
        actor: Player,
        completed: bool,
    ) {
        if state.status != GameStatus::Ongoing {
            return;
        }
        let Some(rule) = &mut state.card_runtime.capture_the_flag else {
            return;
        };
        let mut defeated = Vec::new();
        for owner in [Player::White, Player::Black] {
            let index = owner.index();
            let occupant = state
                .board
                .get(rule.flags[index])
                .filter(|piece| piece.owner != owner);
            let Some(piece) = occupant else {
                rule.occupations[index] = None;
                continue;
            };
            match rule.occupations[index] {
                Some(previous) if previous.piece_id == piece.id => {
                    if state.turns_taken[index] > previous.after_owner_turn {
                        defeated.push(owner);
                    }
                }
                _ => {
                    let before = state.turns_taken[index]
                        .saturating_sub(u32::from(completed && actor == owner));
                    rule.occupations[index] = Some(crate::FlagOccupation {
                        piece_id: piece.id,
                        after_owner_turn: before.saturating_add(u32::from(actor == owner)),
                    });
                }
            }
        }
        if defeated.len() == 2 {
            state.status = GameStatus::Draw {
                reason: DrawReason::CardEffect(crate::CAPTURE_THE_FLAG.to_owned()),
            };
        } else if let Some(owner) = defeated.first() {
            state.status = GameStatus::Won {
                winner: owner.opponent(),
                reason: WinReason::CardEffect(crate::CAPTURE_THE_FLAG.to_owned()),
            };
        }
    }

    /// Published 09.15: every allied kind, including pawns, must occur only once.
    pub(super) fn resolve_highlander_victory(&self, state: &mut GameState) {
        if state.status != GameStatus::Ongoing {
            return;
        }
        for owner in [Player::White, Player::Black] {
            if !state.cards.iter().any(|card| {
                card.owner == owner
                    && card.definition_id == crate::HIGHLANDER
                    && card.available_after_turn.is_none()
            }) {
                continue;
            }
            let mut types = BTreeSet::new();
            let eligible = state
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == owner)
                .all(|(_, piece)| types.insert(piece.kind));
            if eligible && !types.is_empty() {
                state.status = GameStatus::Won {
                    winner: owner,
                    reason: WinReason::CardEffect(crate::HIGHLANDER.to_owned()),
                };
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flag_test_state() -> GameState {
        let mut board = crate::Board::empty();
        for (square, owner, kind) in [
            ("h1", Player::White, PieceKind::King),
            ("b1", Player::White, PieceKind::Rook),
            ("a2", Player::Black, PieceKind::Rook),
            ("h8", Player::Black, PieceKind::King),
        ] {
            board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let mut state = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::Black,
            915,
        );
        state.card_runtime.capture_the_flag = Some(crate::CaptureTheFlagState {
            flags: ["a1".parse().unwrap(), "a8".parse().unwrap()],
            occupations: [None, None],
        });
        state
    }

    fn flag_move(from: &str, to: &str) -> Action {
        Action::Move(MoveAction::new(
            from.parse().unwrap(),
            to.parse().unwrap(),
            None,
        ))
    }

    #[test]
    fn majesty_destination_filter_direct_diagnostic() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut board = crate::Board::empty();
        board.set(
            "e1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King)),
        );
        board.set(
            "e2".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Queen)),
        );
        let mut state = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::Black,
            915,
        );
        state.card_runtime.majesty[0] = true;
        let queen = state.board.get("e2".parse().unwrap()).unwrap();
        assert!(engine.majesty_destination_blocked(&state, queen, "e2".parse().unwrap()));
        assert!(!engine.majesty_destination_blocked(&state, queen, "e1".parse().unwrap()));
        assert!(!engine.majesty_destination_blocked(&state, queen, "e3".parse().unwrap()));
    }

    #[test]
    fn majesty_blocks_enemy_major_pieces_from_king_adjacent_destinations() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut board = crate::Board::empty();
        for (square, owner, kind) in [
            ("e1", Player::White, PieceKind::King),
            ("a2", Player::Black, PieceKind::Rook),
            ("a4", Player::Black, PieceKind::Pawn),
            ("d8", Player::Black, PieceKind::King),
            ("h2", Player::Black, PieceKind::Bishop),
        ] {
            board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let without = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::Black,
            915,
        );
        // e2/d1 are adjacent to the white king; a pawn is free to go there already.
        assert!(
            engine
                .legal_actions(&without)
                .unwrap()
                .contains(&flag_move("a4", "a3"))
        );
        let with = engine
            .grant_card(&without, 1, crate::MAJESTY, Player::White)
            .unwrap()
            .state;
        assert!(with.card_runtime.majesty[Player::White.index()]);
        let actions = engine.legal_actions(&with).unwrap();
        assert!(!actions.contains(&flag_move("a2", "e2")));
        assert!(!actions.contains(&flag_move("a2", "e1")));
        assert!(actions.contains(&flag_move("a2", "a3")));
        assert!(actions.contains(&flag_move("a2", "b2")));
        // Majors only: pawns and bishops may still step next to the king.
        assert!(actions.contains(&flag_move("a4", "a3")));
        assert!(actions.contains(&flag_move("h2", "g3")));
        assert!(actions.contains(&flag_move("h2", "g1")));
        // Fingerprint and position key must both change with the flag.
        assert_ne!(without.fingerprint(), with.fingerprint());
        assert_ne!(without.position_key(), with.position_key());
        // A swap move re-checks the swap partner: relay-style king swap target of a rook is blocked.
        let royal_knight = with.clone();
        assert!(engine.apply(&with, flag_move("a2", "b2")).is_ok());
        let _ = royal_knight;
    }

    #[test]
    fn capture_the_flag_defender_has_one_turn_to_liberate() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = flag_test_state();
        let occupied = engine.apply(&state, flag_move("a2", "a1")).unwrap().state;
        assert_eq!(occupied.status, GameStatus::Ongoing);
        let lost = engine
            .apply(&occupied, flag_move("h1", "g1"))
            .unwrap()
            .state;
        assert_eq!(
            lost.status,
            GameStatus::Won {
                winner: Player::Black,
                reason: WinReason::CardEffect("capture-the-flag".to_owned()),
            }
        );
        let freed = engine
            .apply(&occupied, flag_move("b1", "a1"))
            .unwrap()
            .state;
        assert_eq!(freed.status, GameStatus::Ongoing);
        assert!(freed.card_runtime.capture_the_flag.unwrap().occupations[0].is_none());
        let fingerprint = occupied.fingerprint();
        assert!(engine.apply(&occupied, flag_move("a2", "a1")).is_err());
        assert_eq!(occupied.fingerprint(), fingerprint);
    }

    #[test]
    fn capture_the_flag_installs_seeded_home_flags() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = GameState::standard(crate::BUNDLED_RULESET_ID, Mode::Standard, 915);
        let first = engine.apply_rule(&state, "capture-the-flag").unwrap();
        let second = engine.apply_rule(&state, "capture-the-flag").unwrap();
        assert_eq!(first.state.fingerprint(), second.state.fingerprint());
        assert_eq!(first.events, second.events);
        assert_ne!(state.fingerprint(), first.state.fingerprint());
        let flags = first.state.card_runtime.capture_the_flag.as_ref().unwrap();
        assert_eq!(flags.flags[0].rank(), 0);
        assert_eq!(flags.flags[1].rank(), 7);
        assert_eq!(flags.occupations, [None, None]);
        assert_eq!(
            first
                .events
                .iter()
                .filter(|e| matches!(e, Event::RandomSample { .. }))
                .count(),
            2
        );
        assert!(engine.apply_rule(&first.state, "capture-the-flag").is_err());
    }

    #[test]
    fn capture_the_flag_fingerprint_and_replay_restore_deadline() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = flag_test_state();
        state.actions_remaining = 2;
        let occupied = engine.apply(&state, flag_move("a2", "a1")).unwrap().state;
        let flags = occupied.card_runtime.capture_the_flag.as_ref().unwrap();
        assert!(flags.occupations[0].is_some());
        let mut changed = occupied.clone();
        changed
            .card_runtime
            .capture_the_flag
            .as_mut()
            .unwrap()
            .occupations[0]
            .as_mut()
            .unwrap()
            .after_owner_turn += 1;
        assert_ne!(changed.position_key(), occupied.position_key());
        assert_ne!(changed.fingerprint(), occupied.fingerprint());
        let mut changed_cell = occupied.clone();
        changed_cell
            .card_runtime
            .capture_the_flag
            .as_mut()
            .unwrap()
            .flags[0] = "c1".parse().unwrap();
        assert_ne!(changed_cell.fingerprint(), occupied.fingerprint());
        let ready = engine
            .grant_card(&occupied, 915, crate::REPLAY, Player::Black)
            .unwrap()
            .state;
        let restored = engine
            .apply(
                &ready,
                Action::UseCard {
                    instance_id: 915,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            restored.card_runtime.capture_the_flag,
            state.card_runtime.capture_the_flag
        );
        assert_eq!(restored.board, state.board);
    }

    #[test]
    fn highlander_unique_types_win_but_duplicate_pawns_do_not() {
        let engine = Engine::bundled_20260818().unwrap();
        for (duplicate_pawn, wins) in [(false, true), (true, false)] {
            let mut board = crate::Board::empty();
            for (square, owner, kind) in [
                ("a1", Player::White, PieceKind::King),
                ("b2", Player::White, PieceKind::Pawn),
                (
                    "c2",
                    Player::White,
                    if duplicate_pawn {
                        PieceKind::Pawn
                    } else {
                        PieceKind::Bishop
                    },
                ),
                ("h8", Player::Black, PieceKind::King),
            ] {
                board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
            }
            let state = GameState::from_board(
                crate::BUNDLED_RULESET_ID,
                Mode::Standard,
                board,
                Player::White,
                915,
            );
            let next = engine
                .grant_card(&state, 1, "highlander", Player::White)
                .unwrap()
                .state;
            assert_eq!(
                matches!(
                    next.status,
                    GameStatus::Won {
                        winner: Player::White,
                        ..
                    }
                ),
                wins
            );
        }
    }
}

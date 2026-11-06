use super::*;

pub(super) const ID: &str = crate::LOBSTER;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Upstream `LOBSTER_SUMMON_HALF_TURNS = 2`: the summon lands after the next
/// counted move, which the core models as `due_ply = state.ply + 2`.
const SUMMON_HALF_TURNS: u32 = 2;

/// A lobster may be reserved on an empty square that no other card, terrain or
/// pending spawn has claimed. Collapsed rings are not installable squares.
fn reserve_open(engine: &Engine, state: &GameState, owner: Player, square: Square) -> bool {
    engine.card_landing_open(state, square)
        && !engine.rule_cell_collapsed(state, square)
        && !engine.d4_restricts(state, owner, false, square)
}

fn candidates(engine: &Engine, state: &GameState, owner: Player) -> Vec<Square> {
    let mut squares = Vec::new();
    for index in 0..64 {
        let Some(square) = Square::from_index(index) else {
            continue;
        };
        if reserve_open(engine, state, owner, square) {
            squares.push(square);
        }
    }
    squares
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let square = square_target(target)?;
    if !reserve_open(engine, state, actor, square) {
        return Err(EngineError::Invariant(
            "lobster requires an empty, unreserved square".to_owned(),
        ));
    }
    state
        .card_runtime
        .pending_lobsters
        .push(crate::PendingLobster {
            owner: actor,
            square,
            due_ply: state.ply.saturating_add(SUMMON_HALF_TURNS),
        });
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for square in candidates(engine, state, card.owner) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Square(square),
        });
    }
}

fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::LOBSTER.to_owned(),
            owner: Player::White,
            half_stars: 6,
            charges: 1,
            available_after_turn: None,
        }
    }

    fn cell(name: &str) -> Square {
        name.parse().unwrap()
    }

    fn grant(state: &GameState) -> GameState {
        let mut next = state.clone();
        next.add_card(card());
        next
    }

    fn cast(engine: &Engine, state: &GameState, square: &str) -> GameState {
        let mut next = state.clone();
        next.cards[0].charges = 0;
        play(
            engine,
            &mut next,
            Player::White,
            1,
            &ActionTarget::Square(cell(square)),
            &mut Vec::new(),
        )
        .unwrap();
        next
    }

    fn offered(engine: &Engine, state: &GameState) -> Vec<Square> {
        let mut actions = Vec::new();
        super::actions(engine, state, &state.cards[0], &mut actions);
        actions
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::Square(square),
                    ..
                } => Some(square),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn lobster_reserves_the_chosen_square_for_the_next_counted_move() {
        let engine = Engine::bundled_latest().unwrap();
        let state = grant(&position(&[("d4", Player::White, PieceKind::Rook)]));
        let state = cast(&engine, &state, "e5");
        assert_eq!(state.card_runtime.pending_lobsters.len(), 1);
        let pending = state.card_runtime.pending_lobsters[0];
        assert_eq!(pending.owner, Player::White);
        assert_eq!(pending.square, cell("e5"));
        assert_eq!(pending.due_ply, 2);
        // Reserving does not spawn anything yet.
        assert!(state.board.get(cell("e5")).is_none());
        // The reservation blocks its square for every other landing path.
        assert!(!engine.card_landing_open(&state, cell("e5")));
    }

    #[test]
    fn lobster_only_offers_squares_that_are_empty_and_unreserved() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = grant(&position(&[("d4", Player::White, PieceKind::Rook)]));
        state.card_runtime.barricades.insert(cell("a3"));
        state.card_runtime.black_holes.insert(cell("b3"));
        state.card_runtime.crown_pos = Some(cell("c3"));
        state
            .card_runtime
            .pending_lobsters
            .push(crate::PendingLobster {
                owner: Player::Black,
                square: cell("e5"),
                due_ply: 9,
            });
        state
            .card_runtime
            .pending_scarecrows
            .push(crate::PendingScarecrow {
                owner: Player::White,
                square: cell("f5"),
                remaining_ply: 1,
            });
        let offered = offered(&engine, &state);
        // position() seeds kings on a1/h8, so those are occupied too.
        for blocked in ["d4", "a1", "h8", "a3", "b3", "c3", "e5", "f5"] {
            assert!(
                !offered.contains(&cell(blocked)),
                "{blocked} must be blocked"
            );
        }
        assert!(offered.contains(&cell("a2")));
        assert!(offered.contains(&cell("h7")));
        // 64 squares minus 3 pieces, 3 terrain claims and 2 pending spawns.
        assert_eq!(offered.len(), 64 - 8);
    }

    #[test]
    fn lobster_rejects_occupied_reserved_or_off_board_targets() {
        let engine = Engine::bundled_latest().unwrap();
        let state = grant(&position(&[("d4", Player::White, PieceKind::Rook)]));
        for bad in [
            // Occupied by a piece.
            ActionTarget::Square(cell("d4")),
            // Reserved by another pending lobster.
            ActionTarget::Square(cell("e5")),
        ] {
            let mut next = if bad == ActionTarget::Square(cell("e5")) {
                let mut reserved = state.clone();
                reserved
                    .card_runtime
                    .pending_lobsters
                    .push(crate::PendingLobster {
                        owner: Player::Black,
                        square: cell("e5"),
                        due_ply: 9,
                    });
                reserved
            } else {
                state.clone()
            };
            let before = next.clone();
            assert!(
                play(&engine, &mut next, Player::White, 1, &bad, &mut Vec::new()).is_err(),
                "{bad:?}"
            );
            assert_eq!(next, before);
        }
        // A collapsed border cell is no longer a legal reserve.
        let mut collapsed = state.clone();
        collapsed.rule_runtime.collapse_depth = 1;
        assert!(offered(&engine, &collapsed).iter().all(|square| {
            square.file() != 0 && square.file() != 7 && square.rank() != 0 && square.rank() != 7
        }));
        assert!(
            play(
                &engine,
                &mut collapsed.clone(),
                Player::White,
                1,
                &ActionTarget::Square(cell("a1")),
                &mut Vec::new()
            )
            .is_err()
        );
        // A non-square target is rejected too.
        assert!(
            play(
                &engine,
                &mut state.clone(),
                Player::White,
                1,
                &ActionTarget::None,
                &mut Vec::new()
            )
            .is_err()
        );
    }

    #[test]
    fn lobster_reservation_blocks_other_cards_but_is_deterministic() {
        let engine = Engine::bundled_latest().unwrap();
        let state = grant(&position(&[("d4", Player::White, PieceKind::Rook)]));
        let first = cast(&engine, &state, "e5");
        let second = cast(&engine, &state, "e5");
        assert_eq!(first, second);
        assert_eq!(first.position_key(), second.position_key());

        // A second lobster cannot reuse the same square...
        let mut twice = first.clone();
        assert!(
            play(
                &engine,
                &mut twice,
                Player::White,
                1,
                &ActionTarget::Square(cell("e5")),
                &mut Vec::new()
            )
            .is_err()
        );
        // ...but may reserve a different open square.
        assert!(
            play(
                &engine,
                &mut twice,
                Player::White,
                1,
                &ActionTarget::Square(cell("g6")),
                &mut Vec::new()
            )
            .is_ok()
        );
        assert_eq!(twice.card_runtime.pending_lobsters.len(), 2);
    }

    #[test]
    fn lobster_is_spawned_by_the_existing_counted_move_consumer_once_due() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = cast(
            &engine,
            &grant(&position(&[("d5", Player::White, PieceKind::Rook)])),
            "e5",
        );
        // Still pending at the reservation ply.
        engine.resolve_after_counted_move(&mut state, &mut Vec::new());
        assert!(state.board.get(cell("e5")).is_none());
        assert_eq!(state.card_runtime.pending_lobsters.len(), 1);
        // Still pending one ply later: the reservation is two plies out.
        state.ply += 1;
        engine.resolve_after_counted_move(&mut state, &mut Vec::new());
        assert!(state.board.get(cell("e5")).is_none());
        assert_eq!(state.card_runtime.pending_lobsters.len(), 1);
        // Due once the reserved ply arrives.
        state.ply += 1;
        let mut events = Vec::new();
        engine.resolve_after_counted_move(&mut state, &mut events);
        let spawned = state.board.get(cell("e5")).unwrap();
        assert_eq!(spawned.kind, PieceKind::Lobster);
        assert_eq!(spawned.owner, Player::White);
        assert!(state.card_runtime.pending_lobsters.is_empty());
        assert!(events.iter().any(|event| matches!(
            event,
            Event::EffectApplied { command, .. } if command.starts_with("lobster-spawn:")
        )));
        // The reservation no longer blocks adjacent squares for landing.
        assert!(engine.card_landing_open(&state, cell("e4")));
    }

    /// End-to-end through the engine's real turn loop: play the promoted card,
    /// then let two counted moves pass so the existing consumer spawns it.
    #[test]
    fn lobster_spawns_through_the_wired_engine_turn_loop() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b1", Player::White, PieceKind::Knight),
            ("g8", Player::Black, PieceKind::Knight),
        ]);
        state = engine
            .grant_card(&state, 1, crate::LOBSTER, Player::White)
            .unwrap()
            .state;
        state = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Square(cell("e5")),
                },
            )
            .unwrap()
            .state;
        assert_eq!(state.card_runtime.pending_lobsters.len(), 1);
        let due = state.card_runtime.pending_lobsters[0].due_ply;
        assert_eq!(due, state.ply + 2);

        // Walk balanced moves until the pawn lands, never past its deadline.
        for _ in 0..4 {
            if state.board.get(cell("e5")).is_some() {
                break;
            }
            let action = if state.turn == Player::White {
                Action::Move(crate::MoveAction::new(cell("b1"), cell("c3"), None))
            } else {
                Action::Move(crate::MoveAction::new(cell("g8"), cell("f6"), None))
            };
            if !engine.legal_actions(&state).unwrap().contains(&action) {
                break;
            }
            state = engine.apply(&state, action).unwrap().state;
        }
        let spawned = state
            .board
            .get(cell("e5"))
            .expect("the lobster must spawn once its reserved ply arrives");
        assert_eq!(spawned.kind, PieceKind::Lobster);
        assert_eq!(spawned.owner, Player::White);
        assert!(state.card_runtime.pending_lobsters.is_empty());
    }
}

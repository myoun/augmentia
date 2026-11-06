use super::*;
use crate::latest::latest_major_kind;

pub(super) const ID: &str = crate::GALE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Owner turns between playing gale and the wind arriving (`triggerTurn + 3`
/// upstream, `gale_due_turns` in the core runtime).
const DELAY_TURNS: u32 = 3;

fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let due = state.turns_taken[actor.index()].saturating_add(DELAY_TURNS);
    state.card_runtime.gale_due_turns[actor.index()].push(due);
    Ok(())
}

fn actions(_: &Engine, _: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    out.push(Action::UseCard {
        instance_id: card.instance_id,
        target: ActionTarget::None,
    });
}

fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

/// Every board cell a piece occupies; large bodies cover more than their anchor.
fn body_cells(state: &GameState, square: Square, piece: Piece) -> Vec<Square> {
    state
        .card_runtime
        .large_pieces
        .get(&piece.id)
        .map_or_else(|| vec![square], |large| large_piece_cells(large.anchor))
}

/// Latest-contract major predicate: the catalog major family, minus a promoted
/// royal (upstream `regencyHeir !== true`).
fn major(state: &GameState, piece: Piece) -> bool {
    latest_major_kind(piece.kind) && !state.card_runtime.royal_piece_ids.contains(&piece.id)
}

impl Engine {
    /// Resolve every gale that has come due for `owner` at that owner's turn end.
    ///
    /// The core's previous settlement (piece_effects.rs, `tick_middle_statuses_after_turn`)
    /// implemented the 0915 MIDDLE rule, "remove every piece with no adjacent
    /// ally". The latest contract reclassifies gale as END: three owner turns
    /// later, remove every *major* piece that has no other major piece within the
    /// surrounding eight squares. Majors of either colour are judged, matching
    /// upstream `resolvePendingGalesAfterTurn`'s September-18 branch.
    pub(in crate::engine) fn settle_gale(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let current_turn = state.turns_taken[owner.index()];
        let due = state.card_runtime.gale_due_turns[owner.index()]
            .iter()
            .any(|due| *due <= current_turn);
        state.card_runtime.gale_due_turns[owner.index()].retain(|due| *due > current_turn);
        if !due {
            return;
        }

        let majors: Vec<(Square, Piece, Vec<Square>)> = state
            .board
            .iter()
            .filter(|(_, piece)| major(state, *piece))
            .map(|(square, piece)| (square, piece, body_cells(state, square, piece)))
            .collect();
        let doomed: Vec<(Square, Piece)> = majors
            .iter()
            .filter(|(_, piece, cells)| {
                !majors.iter().any(|(_, other, other_cells)| {
                    other.id != piece.id
                        && cells.iter().any(|a| {
                            other_cells.iter().any(|b| {
                                a.file().abs_diff(b.file()) <= 1 && a.rank().abs_diff(b.rank()) <= 1
                            })
                        })
                })
            })
            .map(|(square, piece, _)| (*square, *piece))
            .collect();
        for (square, piece) in doomed {
            // A blown-away piece is a capture for its opponent, not for the caster.
            let capturer = piece.owner.opponent();
            self.remove_piece_for_card(state, square, piece, capturer, ID, events);
        }
        events.push(Event::EffectApplied {
            source_card_instance: 0,
            command: format!("gale:{current_turn}"),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, handoff, position};
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::GALE.to_owned(),
            owner: Player::White,
            half_stars: 7,
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

    fn cast(engine: &Engine, state: &GameState) -> GameState {
        let mut next = state.clone();
        // `apply_active_card` decrements charges before calling the card handler.
        next.cards[0].charges = 0;
        play(
            engine,
            &mut next,
            Player::White,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        )
        .unwrap();
        next
    }

    #[test]
    fn gale_schedules_three_owner_turns_ahead_without_touching_the_board() {
        let engine = Engine::bundled_latest().unwrap();
        let state = grant(&position(&[("d4", Player::White, PieceKind::Rook)]));
        let before = state.board.clone();
        let state = cast(&engine, &state);
        assert_eq!(state.card_runtime.gale_due_turns[0], vec![3]);
        assert!(state.card_runtime.gale_due_turns[1].is_empty());
        assert_eq!(state.board, before);
        // The card is a pure scheduler: no pieces disappear on use.
        assert!(state.board.get(cell("d4")).is_some());
    }

    #[test]
    fn gale_waits_for_the_third_owner_turn_and_ignores_other_boundaries() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = cast(
            &engine,
            &grant(&position(&[
                ("d4", Player::White, PieceKind::Rook),
                ("c3", Player::White, PieceKind::Knight),
                ("h4", Player::Black, PieceKind::Queen),
            ])),
        );
        let count = state.board.iter().count();
        // Owner-turn settlement runs after `turns_taken` has been incremented, so
        // the third completed white turn is `turns_taken[0] == 3`.
        for _ in 0..4 {
            handoff(&engine, &mut state);
        }
        assert_eq!(state.turns_taken[0], 2);
        assert_eq!(state.board.iter().count(), count);
        assert_eq!(state.card_runtime.gale_due_turns[0], vec![3]);
        // The opponent's boundary must not fire it.
        engine.settle_gale(&mut state, Player::Black, &mut Vec::new());
        assert_eq!(state.board.iter().count(), count);
        for _ in 0..2 {
            handoff(&engine, &mut state);
        }
        assert_eq!(state.turns_taken[0], 3);
        engine.settle_gale(&mut state, Player::White, &mut Vec::new());
        assert!(state.card_runtime.gale_due_turns[0].is_empty());
        // d4 is an isolated white major and h4 an isolated black major: both blow
        // away. c3 is a minor and stays.
        assert!(state.board.get(cell("d4")).is_none());
        assert!(state.board.get(cell("h4")).is_none());
        assert!(state.board.get(cell("c3")).is_some());
    }

    #[test]
    fn gale_spares_majors_that_cluster_and_removes_only_isolated_ones() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = cast(
            &engine,
            &grant(&position(&[
                // Adjacent majors: both survive.
                ("b4", Player::White, PieceKind::Rook),
                ("c5", Player::White, PieceKind::Rook),
                // Lone major: removed.
                ("g7", Player::White, PieceKind::Queen),
                // A lone minor is never a target.
                ("a3", Player::White, PieceKind::Bishop),
            ])),
        );
        state.turns_taken[0] = 3;
        let mut events = Vec::new();
        engine.settle_gale(&mut state, Player::White, &mut events);
        assert!(state.board.get(cell("b4")).is_some());
        assert!(state.board.get(cell("c5")).is_some());
        assert!(state.board.get(cell("a3")).is_some());
        assert!(state.board.get(cell("g7")).is_none());
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::PieceRemoved { .. }))
                .count(),
            1
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::EffectApplied { command, .. } if command.starts_with("gale:")
        )));
        // Settling again does nothing: the due entry was consumed.
        engine.settle_gale(&mut state, Player::White, &mut Vec::new());
        assert!(state.board.get(cell("b4")).is_some());
    }

    #[test]
    fn gale_counts_enemy_majors_as_neighbours_and_spares_the_royal_promotion() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = cast(
            &engine,
            &grant(&position(&[
                // An enemy major next door protects this rook.
                ("d4", Player::White, PieceKind::Rook),
                ("e5", Player::Black, PieceKind::Queen),
                // A royal-flagged queen is not a catalog major.
                ("a4", Player::White, PieceKind::Queen),
            ])),
        );
        let royal = at(&state, "a4");
        state.card_runtime.royal_piece_ids.insert(royal.id);
        state.turns_taken[0] = 3;
        engine.settle_gale(&mut state, Player::White, &mut Vec::new());
        assert!(state.board.get(cell("d4")).is_some());
        assert!(state.board.get(cell("e5")).is_some());
        assert!(state.board.get(cell("a4")).is_some());
    }

    #[test]
    fn gale_stacks_two_schedules_and_is_deterministic() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = grant(&position(&[("d4", Player::White, PieceKind::Rook)]));
        state.cards[0].charges = 0;
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        )
        .unwrap();
        state.cards[0].charges = 0;
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(state.card_runtime.gale_due_turns[0], vec![3, 3]);

        let settle = |state: &GameState| {
            let mut next = state.clone();
            next.turns_taken[0] = 3;
            let mut events = Vec::new();
            engine.settle_gale(&mut next, Player::White, &mut events);
            (next, events)
        };
        let first = settle(&state);
        let second = settle(&state);
        assert_eq!(first, second);
        assert!(first.0.card_runtime.gale_due_turns[0].is_empty());
    }

    /// End-to-end through the engine's real turn loop: play the promoted card,
    /// then let the wired turn boundary resolve it after three owner turns.
    #[test]
    fn gale_resolves_through_the_wired_engine_boundary() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d4", Player::White, PieceKind::Rook),
            ("b1", Player::White, PieceKind::Knight),
            ("g7", Player::Black, PieceKind::Knight),
        ]);
        state = engine
            .grant_card(&state, 1, crate::GALE, Player::White)
            .unwrap()
            .state;
        state = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        assert_eq!(state.card_runtime.gale_due_turns[0], vec![3]);

        // Neither the caster's own turn end nor the opponent's turns fire it.
        while state.turns_taken[0] < 2 {
            handoff(&engine, &mut state);
        }
        assert_eq!(state.turns_taken[0], 2);
        assert!(state.board.get(cell("d4")).is_some());
        assert_eq!(state.card_runtime.gale_due_turns[0], vec![3]);

        // The third completed owner turn is the due boundary: the isolated white
        // rook blows away, the friendly minor and the enemy major stay because
        // both are not isolated majors.
        while state.turns_taken[0] < 3 {
            handoff(&engine, &mut state);
        }
        assert_eq!(state.turns_taken[0], 3);
        assert!(state.card_runtime.gale_due_turns[0].is_empty());
        assert!(state.board.get(cell("d4")).is_none());
        assert!(state.board.get(cell("b1")).is_some());
    }
}

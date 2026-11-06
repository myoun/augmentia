use super::*;

pub(super) const ID: &str = crate::COLLAPSE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Maximum number of rings the collapse ladder can consume (`v9ApplyCollapse`
/// clamps `depth` to 4).
const MAX_DEPTH: u8 = 4;

/// Cells of the ring at `depth`, measured inward from the board edge.
///
/// This is the Rust equivalent of `v9CollapseRing`: `depth == 0` is the current
/// outer border, `depth == 1` the next ring, and so on.
fn ring(depth: u8) -> Vec<Square> {
    let mut cells = Vec::new();
    for index in 0..64 {
        let Some(square) = Square::from_index(index) else {
            continue;
        };
        let inset = square
            .file()
            .min(7 - square.file())
            .min(square.rank())
            .min(7 - square.rank());
        if inset == depth {
            cells.push(square);
        }
    }
    cells
}

fn current_depth(state: &GameState) -> u8 {
    state.rule_runtime.collapse_depth
}

/// The collapse card is armed by stamping its own instance's
/// `available_after_turn` slot with the owner's completed-turn index.
///
/// `charges` is already 0 by the time `play` runs, so the availability gate can
/// never block the card again; the field is per-card, part of position identity,
/// and cleared at the owner's turn boundary, which makes it the one persisted
/// per-card slot the card can use without touching shared state.
fn armed(card: &CardInstance) -> bool {
    card.charges == 0 && card.available_after_turn.is_some()
}

fn armed_for(state: &GameState, owner: Player) -> bool {
    state
        .cards
        .iter()
        .any(|card| card.definition_id == ID && card.owner == owner && armed(card))
}

fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    instance_id: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    if current_depth(state) >= MAX_DEPTH || ring(current_depth(state)).is_empty() {
        return Err(EngineError::Invariant(
            "collapse has no outer border left".to_owned(),
        ));
    }
    if armed_for(state, actor) {
        return Err(EngineError::Invariant(
            "collapse is already scheduled for this owner".to_owned(),
        ));
    }
    let card = state
        .cards
        .iter_mut()
        .find(|card| card.instance_id == instance_id && card.owner == actor)
        .ok_or_else(|| EngineError::Invariant("collapse card instance is missing".to_owned()))?;
    card.available_after_turn = Some(state.turns_taken[actor.index()]);
    Ok(())
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    let depth = current_depth(state);
    if depth < MAX_DEPTH && !ring(depth).is_empty() && !armed_for(state, card.owner) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

impl Engine {
    /// Collapse the current outer ring after the owner's next move completes.
    ///
    /// The owner arms the card during their turn; settlement must run at that
    /// owner's turn end (after the move has been applied, before the board is
    /// handed over). Mirror `v9ApplyCollapse`: remove every piece on the ring,
    /// drop ring terrain and spawn reservations, advance the collapse depth, and
    /// flip the legacy border flag.
    pub(in crate::engine) fn settle_collapse(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let due: Vec<u32> = state
            .cards
            .iter()
            .filter(|card| {
                card.definition_id == ID
                    && card.owner == owner
                    && armed(card)
                    && card
                        .available_after_turn
                        .is_some_and(|due| state.turns_taken[owner.index()] >= due)
            })
            .map(|card| card.instance_id)
            .collect();
        if due.is_empty() {
            return;
        }
        for instance_id in due {
            if let Some(card) = state
                .cards
                .iter_mut()
                .find(|card| card.instance_id == instance_id)
            {
                card.available_after_turn = None;
            }
        }

        let depth = current_depth(state);
        let cells = ring(depth);
        if cells.is_empty() {
            return;
        }
        let ring_set: BTreeSet<Square> = cells.iter().copied().collect();

        // Large bodies can straddle the ring from an anchor outside it, so
        // decide membership from every occupied cell and remove each body once.
        let mut victims: Vec<(Square, Piece)> = Vec::new();
        let mut seen = BTreeSet::new();
        for (square, piece) in state.board.iter() {
            if !seen.insert(piece.id) {
                continue;
            }
            let occupied = state
                .card_runtime
                .large_pieces
                .get(&piece.id)
                .map_or_else(|| vec![square], |large| large_piece_cells(large.anchor));
            if occupied.iter().any(|cell| ring_set.contains(cell)) {
                victims.push((square, piece));
            }
        }
        for (square, piece) in victims {
            self.remove_piece_for_card(state, square, piece, owner, ID, events);
        }

        // The ring is gone: terrain, hazards and spawn reservations on it go too.
        for square in &cells {
            state.card_runtime.remove_barricade(*square);
            state.card_runtime.black_holes.remove(square);
            state.card_runtime.rule_bombs.remove(square);
            state.card_runtime.high_ground_squares.remove(square);
            state.card_runtime.highway_cells.remove(square);
            state
                .card_runtime
                .quantum_positions
                .retain(|_, s| s != square);
        }
        state
            .card_runtime
            .pending_lobsters
            .retain(|pending| !ring_set.contains(&pending.square));
        state
            .card_runtime
            .pending_scarecrows
            .retain(|pending| !ring_set.contains(&pending.square));
        state
            .card_runtime
            .pending_portals
            .retain(|pending| !pending.squares.iter().any(|s| ring_set.contains(s)));
        if state
            .card_runtime
            .football_pos
            .is_some_and(|square| ring_set.contains(&square))
        {
            state.card_runtime.football_pos = None;
        }
        if state
            .card_runtime
            .crown_pos
            .is_some_and(|square| ring_set.contains(&square))
        {
            state.card_runtime.crown_pos = None;
        }

        state.rule_runtime.collapse_depth = depth.saturating_add(1).min(MAX_DEPTH);
        state.card_runtime.collapsed_files_ranks = true;
        events.push(Event::EffectApplied {
            source_card_instance: 0,
            command: format!("collapse:{}:{}", depth, cells.len()),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::COLLAPSE.to_owned(),
            owner: Player::White,
            half_stars: 9,
            charges: 1,
            available_after_turn: None,
        }
    }

    fn arm(engine: &Engine, state: &GameState) -> GameState {
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

    fn cell(name: &str) -> Square {
        name.parse().unwrap()
    }

    fn offered(engine: &Engine, state: &GameState) -> bool {
        let mut actions = Vec::new();
        super::actions(engine, state, &state.cards[0], &mut actions);
        !actions.is_empty()
    }

    fn grant(state: &GameState) -> GameState {
        let mut next = state.clone();
        next.add_card(card());
        next
    }

    #[test]
    fn collapse_arms_without_touching_the_board_and_refuses_a_second_arm() {
        let engine = Engine::bundled_latest().unwrap();
        let state = grant(&position(&[("d4", Player::White, PieceKind::Rook)]));
        let before = state.clone();
        let armed_state = arm(&engine, &state);
        // Arming is bookkeeping only; the board and pieces are untouched.
        assert_eq!(armed_state.board, before.board);
        assert!(!armed_state.card_runtime.collapsed_files_ranks);
        assert_eq!(armed_state.rule_runtime.collapse_depth, 0);
        assert!(armed(&armed_state.cards[0]));
        assert!(!offered(&engine, &armed_state));
        // A second collapse cannot be queued while one is pending.
        let mut twice = armed_state.clone();
        twice.cards.push(CardInstance {
            instance_id: 2,
            ..card()
        });
        assert!(
            play(
                &engine,
                &mut twice,
                Player::White,
                2,
                &ActionTarget::None,
                &mut Vec::new()
            )
            .is_err()
        );
    }

    #[test]
    fn collapse_settles_at_the_owner_boundary_and_removes_only_the_outer_ring() {
        let engine = Engine::bundled_latest().unwrap();
        // position() seeds a1/h8 kings, so every extra piece here is interior
        // apart from the two border pieces below.
        let state = grant(&position(&[
            ("h4", Player::White, PieceKind::Queen),
            ("a5", Player::Black, PieceKind::Rook),
            ("d4", Player::White, PieceKind::Rook),
        ]));
        let armed_state = arm(&engine, &state);
        // The opponent's boundary must not fire the owner's collapse.
        let mut waiting = armed_state.clone();
        engine.settle_collapse(&mut waiting, Player::Black, &mut Vec::new());
        assert!(!waiting.card_runtime.collapsed_files_ranks);
        assert_eq!(waiting.board.iter().count(), 5);
        assert!(armed(&waiting.cards[0]));

        let mut next = waiting.clone();
        let mut events = Vec::new();
        engine.settle_collapse(&mut next, Player::White, &mut events);
        assert!(next.card_runtime.collapsed_files_ranks);
        assert_eq!(next.rule_runtime.collapse_depth, 1);
        // h4 and a5 sat on the border ring and died; the kings on a1/h8 did too.
        assert!(next.board.get(cell("h4")).is_none());
        assert!(next.board.get(cell("a5")).is_none());
        assert!(next.board.get(cell("a1")).is_none());
        assert!(next.board.get(cell("h8")).is_none());
        // Interior pieces survive.
        assert!(next.board.get(cell("d4")).is_some());
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::PieceRemoved { .. }))
                .count(),
            4
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::EffectApplied { command, .. } if command.starts_with("collapse:")
        )));
        // The ring is blocked for later landings and movement.
        assert!(engine.rule_cell_collapsed(&next, cell("a1")));
        assert!(!engine.rule_cell_collapsed(&next, cell("d4")));
        assert!(!engine.card_landing_open(&next, cell("h4")));
        assert!(armed(&next.cards[0]));
        // Settling is idempotent: the marker was consumed.
        engine.settle_collapse(&mut next, Player::White, &mut Vec::new());
        assert_eq!(next.rule_runtime.collapse_depth, 1);
    }

    #[test]
    fn collapse_clears_ring_terrain_and_spawn_reservations() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = grant(&position(&[("d4", Player::White, PieceKind::Rook)]));
        state.card_runtime.barricades.insert(cell("a5"));
        state
            .card_runtime
            .blockade_walls
            .insert(cell("a5"), (Player::White, 3));
        state.card_runtime.rule_bombs.insert(cell("h5"));
        state.card_runtime.black_holes.insert(cell("e1"));
        state
            .card_runtime
            .pending_lobsters
            .push(crate::PendingLobster {
                owner: Player::White,
                square: cell("b1"),
                due_ply: 2,
            });
        state
            .card_runtime
            .pending_scarecrows
            .push(crate::PendingScarecrow {
                owner: Player::White,
                square: cell("g1"),
                remaining_ply: 1,
            });
        state
            .card_runtime
            .pending_portals
            .push(crate::PendingPortal {
                owner: Player::White,
                squares: [cell("c1"), cell("d4")],
                due_turn: 1,
            });
        // One reservation climbs onto the inner board and must survive.
        state
            .card_runtime
            .pending_lobsters
            .push(crate::PendingLobster {
                owner: Player::White,
                square: cell("d5"),
                due_ply: 2,
            });
        state
            .card_runtime
            .pending_scarecrows
            .push(crate::PendingScarecrow {
                owner: Player::White,
                square: cell("d6"),
                remaining_ply: 1,
            });
        let armed_state = arm(&engine, &state);
        let mut next = armed_state;
        engine.settle_collapse(&mut next, Player::White, &mut Vec::new());
        for square in [cell("a5"), cell("h5"), cell("e1")] {
            assert!(!next.card_runtime.barricades.contains(&square));
            assert!(!next.card_runtime.blockade_walls.contains_key(&square));
            assert!(!next.card_runtime.rule_bombs.contains(&square));
            assert!(!next.card_runtime.black_holes.contains(&square));
        }
        assert!(
            !next
                .card_runtime
                .pending_lobsters
                .iter()
                .any(|pending| pending.square == cell("b1"))
        );
        assert!(
            !next
                .card_runtime
                .pending_scarecrows
                .iter()
                .any(|pending| pending.square == cell("g1"))
        );
        // The portal paired a ring cell with an interior cell, so it is dropped.
        assert!(next.card_runtime.pending_portals.is_empty());
        assert!(
            next.card_runtime
                .pending_lobsters
                .iter()
                .any(|pending| pending.square == cell("d5"))
        );
        assert!(
            next.card_runtime
                .pending_scarecrows
                .iter()
                .any(|pending| pending.square == cell("d6"))
        );
    }

    #[test]
    fn collapse_caps_at_four_rings_and_stops_being_offered() {
        let engine = Engine::bundled_latest().unwrap();
        let state = grant(&position(&[("d4", Player::White, PieceKind::Rook)]));
        assert!(offered(&engine, &state));
        let mut next = state;
        for depth in 0..MAX_DEPTH {
            assert_eq!(next.rule_runtime.collapse_depth, depth);
            let armed_state = arm(&engine, &next);
            next = armed_state;
            engine.settle_collapse(&mut next, Player::White, &mut Vec::new());
            next.cards[0].charges = 1;
            next.cards[0].available_after_turn = None;
        }
        assert_eq!(next.rule_runtime.collapse_depth, MAX_DEPTH);
        assert!(ring(MAX_DEPTH).is_empty());
        assert!(!offered(&engine, &next));
        // Playing at the cap fails without touching state.
        let mut capped = next.clone();
        capped.cards[0].charges = 0;
        let before = capped.clone();
        assert!(
            play(
                &engine,
                &mut capped,
                Player::White,
                1,
                &ActionTarget::None,
                &mut Vec::new()
            )
            .is_err()
        );
        assert_eq!(capped, before);
    }
}

use super::*;

pub(super) const ID: &str = crate::BARRICADE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// The three cells a barricade covers when it is centred on `center`.
///
/// Upstream `barricadeFootprint` grows the line out from the chosen centre, so
/// the chosen square is always the middle cell of the wall.
fn footprint(center: Square, vertical: bool) -> Option<[Square; 3]> {
    let (file_delta, rank_delta) = if vertical { (0, 1) } else { (1, 0) };
    Some([
        center.offset(-file_delta, -rank_delta)?,
        center,
        center.offset(file_delta, rank_delta)?,
    ])
}

/// A wall cell must be empty and free of every placement reservation.
fn open(engine: &Engine, state: &GameState, owner: Player, square: Square) -> bool {
    engine.card_landing_open(state, square) && !engine.d4_restricts(state, owner, false, square)
}

/// One unit orthogonal step, or `None` when the two squares do not form one.
fn unit_step(from: Square, to: Square) -> Option<(i8, i8)> {
    let file_delta = i16::from(to.file()) - i16::from(from.file());
    let rank_delta = i16::from(to.rank()) - i16::from(from.rank());
    if !(-1..=1).contains(&file_delta) || !(-1..=1).contains(&rank_delta) {
        return None;
    }
    let step = (file_delta as i8, rank_delta as i8);
    ((step.0 == 0) != (step.1 == 0)).then_some(step)
}

/// Validate a three-cell run and report the axis it was authored on.
fn checked_run(
    engine: &Engine,
    state: &GameState,
    owner: Player,
    target: &ActionTarget,
) -> Result<[Square; 3], EngineError> {
    let ActionTarget::Squares(squares) = target else {
        return Err(EngineError::Invariant(
            "barricade requires three wall cells".to_owned(),
        ));
    };
    let [first, middle, last] = squares.as_slice() else {
        return Err(EngineError::Invariant(
            "barricade requires three wall cells".to_owned(),
        ));
    };
    let cells = [*first, *middle, *last];
    let step = unit_step(cells[0], cells[1])
        .ok_or_else(|| EngineError::Invariant("barricade cells must be contiguous".to_owned()))?;
    if unit_step(cells[1], cells[2]) != Some(step)
        || footprint(cells[1], step.0 == 0) != Some(cells)
    {
        return Err(EngineError::Invariant(
            "barricade cells must form a straight run of three".to_owned(),
        ));
    }
    if !cells
        .iter()
        .all(|square| open(engine, state, owner, *square))
    {
        return Err(EngineError::Invariant(
            "barricade requires three empty cells".to_owned(),
        ));
    }
    Ok(cells)
}

fn placed_runs(engine: &Engine, state: &GameState, owner: Player) -> Vec<[Square; 3]> {
    let mut runs = Vec::new();
    for index in 0..64 {
        let Some(center) = Square::from_index(index) else {
            continue;
        };
        for vertical in [false, true] {
            let Some(cells) = footprint(center, vertical) else {
                continue;
            };
            if cells
                .iter()
                .all(|square| open(engine, state, owner, *square))
            {
                runs.push(cells);
            }
        }
    }
    runs
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let cells = checked_run(engine, state, actor, target)?;
    for square in cells {
        state.card_runtime.barricades.insert(square);
    }
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("barricade:{}:{}:{}", cells[0], cells[1], cells[2]),
    });
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for cells in placed_runs(engine, state, card.owner) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(cells.to_vec()),
        });
    }
}

fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{has_move, position};
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::BARRICADE.to_owned(),
            owner: Player::White,
            half_stars: 5,
            charges: 1,
            available_after_turn: None,
        }
    }

    fn target(cells: [Square; 3]) -> ActionTarget {
        ActionTarget::Squares(cells.to_vec())
    }

    fn cell(name: &str) -> Square {
        name.parse().unwrap()
    }

    fn offered(engine: &Engine, state: &GameState) -> Vec<[Square; 3]> {
        let mut actions = Vec::new();
        super::actions(engine, state, &card(), &mut actions);
        actions
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::Squares(squares),
                    ..
                } => Some([squares[0], squares[1], squares[2]]),
                _ => None,
            })
            .collect()
    }

    fn place(
        engine: &Engine,
        state: &GameState,
        cells: [Square; 3],
    ) -> Result<GameState, EngineError> {
        let mut next = state.clone();
        let mut events = Vec::new();
        play(
            engine,
            &mut next,
            Player::White,
            1,
            &target(cells),
            &mut events,
        )?;
        Ok(next)
    }

    #[test]
    fn barricade_offers_horizontal_and_vertical_runs_over_open_cells_only() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(&[("d4", Player::White, PieceKind::Rook)]);
        let runs = offered(&engine, &state);
        assert!(runs.contains(&[cell("e4"), cell("f4"), cell("g4")]));
        assert!(runs.contains(&[cell("e4"), cell("e5"), cell("e6")]));
        // The occupied centre and both of its runs are unavailable.
        assert!(!runs.contains(&[cell("c4"), cell("d4"), cell("e4")]));
        assert!(!runs.contains(&[cell("d3"), cell("d4"), cell("d5")]));
        // A three-cell run cannot start on an edge rank or file.
        assert!(!runs.contains(&[cell("a1"), cell("b1"), cell("c1")]));
    }

    #[test]
    fn barricade_places_three_walls_and_blocks_their_cells() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(&[("d1", Player::White, PieceKind::Rook)]);
        let run = [cell("c4"), cell("d4"), cell("e4")];
        let state = place(&engine, &state, run).unwrap();
        for square in run {
            assert!(state.card_runtime.barricades.contains(&square));
            assert!(!engine.card_landing_open(&state, square));
        }
        assert_eq!(state.card_runtime.barricades.len(), 3);
        // The d-file rook stops below the wall and cannot slide past it.
        assert!(has_move(&engine, &state, "d1", "d3"));
        assert!(!has_move(&engine, &state, "d1", "d4"));
        assert!(!has_move(&engine, &state, "d1", "d5"));
        // Other files are unaffected.
        assert!(has_move(&engine, &state, "d1", "h1"));
    }

    #[test]
    fn barricade_is_deterministic_and_rejects_malformed_cells() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(&[("d1", Player::White, PieceKind::Rook)]);
        let run = [cell("c4"), cell("d4"), cell("e4")];
        let mut first_events = Vec::new();
        let mut first = state.clone();
        play(
            &engine,
            &mut first,
            Player::White,
            1,
            &target(run),
            &mut first_events,
        )
        .unwrap();
        let mut second_events = Vec::new();
        let mut second = state.clone();
        play(
            &engine,
            &mut second,
            Player::White,
            1,
            &target(run),
            &mut second_events,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first_events, second_events);

        for malformed in [
            // Too short.
            ActionTarget::Squares(vec![cell("c4"), cell("d4")]),
            // Not a straight line.
            ActionTarget::Squares(vec![cell("c4"), cell("d4"), cell("d5")]),
            // Reversed runs are rejected: the middle cell must be authored second.
            ActionTarget::Squares(vec![cell("e4"), cell("d4"), cell("c4")]),
            // A non-square target carries no wall cells.
            ActionTarget::None,
            ActionTarget::PieceAt(cell("d4")),
        ] {
            let mut events = Vec::new();
            let mut next = state.clone();
            assert!(
                play(
                    &engine,
                    &mut next,
                    Player::White,
                    1,
                    &malformed,
                    &mut events
                )
                .is_err()
            );
            assert!(next.card_runtime.barricades.is_empty());
            assert!(events.is_empty());
        }

        // A run through an occupied cell fails without touching state or events.
        let mut blocked = state.clone();
        blocked.board.set(
            cell("d4"),
            Some(Piece::new(Player::White, PieceKind::Knight)),
        );
        let before = blocked.clone();
        let mut events = Vec::new();
        assert!(
            play(
                &engine,
                &mut blocked,
                Player::White,
                1,
                &target(run),
                &mut events
            )
            .is_err()
        );
        assert_eq!(blocked, before);
        assert!(events.is_empty());
    }

    #[test]
    fn barricade_never_overlaps_walls_or_pending_spawns() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("d1", Player::White, PieceKind::Rook)]);
        state.card_runtime.barricades.insert(cell("e4"));
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
            .pending_portals
            .push(crate::PendingPortal {
                owner: Player::White,
                squares: [cell("f4"), cell("h1")],
                due_turn: 1,
            });
        let runs = offered(&engine, &state);
        assert!(!runs.iter().any(|run| run.contains(&cell("e4"))));
        assert!(!runs.iter().any(|run| run.contains(&cell("d5"))));
        assert!(!runs.iter().any(|run| run.contains(&cell("f4"))));

        // The same reservations make a direct placement fail.
        let mut events = Vec::new();
        let mut next = state.clone();
        assert!(
            play(
                &engine,
                &mut next,
                Player::White,
                1,
                &target([cell("d4"), cell("e4"), cell("f4")]),
                &mut events
            )
            .is_err()
        );
        assert_eq!(next.card_runtime.barricades.len(), 1);
    }

    /// End-to-end through the public engine path: grant, then apply the same
    /// action the engine itself offers from `legal_actions`.
    #[test]
    fn barricade_plays_through_the_public_engine_api() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(&[("d1", Player::White, PieceKind::Rook)]);
        let state = engine
            .grant_card(&state, 1, crate::BARRICADE, Player::White)
            .unwrap()
            .state;
        let legal = engine.legal_actions(&state).unwrap();
        let action = legal
            .iter()
            .find(|action| matches!(action, Action::UseCard { instance_id: 1, .. }))
            .cloned()
            .expect("a promoted barricade must offer a placement");
        let played = engine.apply(&state, action.clone()).unwrap();
        // The engine frames a card use with ActionApplied then CardUsed.
        assert!(played.events.iter().any(|event| matches!(
            event,
            Event::CardUsed { definition_id, .. } if definition_id == crate::BARRICADE
        )));
        assert_eq!(played.state.card_runtime.barricades.len(), 3);
        // Playing consumes the single charge, so the card is no longer offered.
        assert!(
            !engine
                .legal_actions(&played.state)
                .unwrap()
                .iter()
                .any(|candidate| matches!(candidate, Action::UseCard { instance_id: 1, .. }))
        );
        // Same computation, same result.
        assert_eq!(played, engine.apply(&state, action).unwrap());
    }
}

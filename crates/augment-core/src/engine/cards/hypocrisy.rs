use super::*;
pub(super) const ID: &str = crate::HYPOCRISY;

/// Hypocrisy always summons exactly four pawns (`min = max = 4`, main bundle 19867).
const TARGETS: usize = 4;

pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let opponent = actor.opponent();
    let squares = squares_target(target)?;
    if squares.len() != TARGETS {
        return Err(EngineError::Invariant(
            "hypocrisy requires four squares".to_owned(),
        ));
    }
    let mut seen = BTreeSet::new();
    for &square in &squares {
        if !seen.insert(square) {
            return Err(EngineError::Invariant(
                "hypocrisy squares must be distinct".to_owned(),
            ));
        }
        if !open_placement_square(engine, state, square) {
            return Err(EngineError::Invariant(
                "hypocrisy square must be an open placement square".to_owned(),
            ));
        }
    }
    for &square in &squares {
        let pawn = Piece::new(opponent, PieceKind::Pawn).with_id(state.allocate_piece_id());
        state.board.set(square, Some(pawn));
        engine.register_spawned_piece(state, square, pawn, true);
    }
    Ok(())
}

/// A square Hypocrisy may summon onto.
///
/// Mirrors the predicate at main bundle 19869: the cell must accept an opponent
/// placement, and must not be a black hole or a collapsed cell. `card_landing_open`
/// already covers the crown, barricade and reservation cases.
fn open_placement_square(engine: &Engine, state: &GameState, square: Square) -> bool {
    engine.card_landing_open(state, square) && !state.card_runtime.black_holes.contains(&square)
}

/// Every combination of four open squares is offered.
///
/// Upstream enumerates all open cells and accepts any four of them; taking only the
/// first four hides every alternative placement.
pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let open = (0_u8..64)
        .filter_map(Square::from_index)
        .filter(|square| open_placement_square(engine, state, *square))
        .collect::<Vec<_>>();
    if open.len() < TARGETS {
        return;
    }
    // Enumerate combinations in board order, which keeps the order deterministic.
    for a in 0..open.len() {
        for b in a + 1..open.len() {
            for c in b + 1..open.len() {
                for d in c + 1..open.len() {
                    actions.push(Action::UseCard {
                        instance_id: card.instance_id,
                        target: ActionTarget::Squares(vec![open[a], open[b], open[c], open[d]]),
                    });
                }
            }
        }
    }
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let id = ID;
    denied(
        EligibilityReasonCode::UnsupportedPredicate,
        id,
        "공식 획득 조건의 런타임 매핑이 아직 검증되지 않아 획득을 허용하지 않습니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::HYPOCRISY.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        }
    }

    fn offered(state: &GameState) -> Vec<Vec<Square>> {
        let engine = Engine::bundled_latest().unwrap();
        let mut out = Vec::new();
        actions(&engine, state, &card(), &mut out);
        out.into_iter()
            .map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::Squares(squares),
                    ..
                } => squares,
                other => panic!("unexpected action {other:?}"),
            })
            .collect()
    }

    #[test]
    fn hypocrisy_offers_four_square_combinations_not_just_the_first_four() {
        // A nearly empty board leaves many open squares, so alternatives exist.
        let state = position(&[]);
        let offered = offered(&state);
        assert!(!offered.is_empty(), "open squares must produce targets");
        // Every target is exactly four distinct squares.
        for squares in &offered {
            assert_eq!(squares.len(), TARGETS);
            assert_eq!(squares.iter().collect::<BTreeSet<_>>().len(), TARGETS);
        }
        // The first four open squares are not the only placement offered.
        let first_four: Vec<Square> = (0_u8..64)
            .filter_map(Square::from_index)
            .filter(|square| state.board.get(*square).is_none())
            .take(TARGETS)
            .collect();
        assert!(
            offered.iter().any(|squares| *squares != first_four),
            "the generator must offer more than the first four open squares"
        );
    }

    #[test]
    fn hypocrisy_summons_exactly_four_enemy_pawns() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        let before = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == Player::Black && piece.kind == PieceKind::Pawn)
            .count();
        let squares = vec![
            "c3".parse().unwrap(),
            "d3".parse().unwrap(),
            "e3".parse().unwrap(),
            "f3".parse().unwrap(),
        ];
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::Squares(squares.clone()),
            &mut Vec::new(),
        )
        .unwrap();
        let after = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == Player::Black && piece.kind == PieceKind::Pawn)
            .count();
        assert_eq!(after, before + TARGETS, "four enemy pawns appear");
        for square in squares {
            assert_eq!(
                state.board.get(square).unwrap().owner,
                Player::Black,
                "the summoned pawns belong to the opponent"
            );
        }
    }

    #[test]
    fn hypocrisy_rejects_wrong_counts_duplicates_and_occupied_squares() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("c3", Player::White, PieceKind::Rook)]);
        for target in [
            // Too few.
            ActionTarget::Squares(vec!["d3".parse().unwrap()]),
            // Duplicates.
            ActionTarget::Squares(vec![
                "d3".parse().unwrap(),
                "d3".parse().unwrap(),
                "e3".parse().unwrap(),
                "f3".parse().unwrap(),
            ]),
            // Occupied.
            ActionTarget::Squares(vec![
                "c3".parse().unwrap(),
                "d3".parse().unwrap(),
                "e3".parse().unwrap(),
                "f3".parse().unwrap(),
            ]),
        ] {
            let before = state.clone();
            assert!(
                play(
                    &engine,
                    &mut state,
                    Player::White,
                    1,
                    &target,
                    &mut Vec::new()
                )
                .is_err(),
                "{target:?} must be rejected"
            );
            assert_eq!(state, before, "{target:?} mutated state");
        }
    }
}

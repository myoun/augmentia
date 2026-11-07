use super::*;
pub(super) const ID: &str = crate::SPY;

/// Most enemy pawns one cast may mark, from the catalog selection (1-2).
const MAX_TARGETS: usize = 2;

pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let squares = squares_target(target)?;
    if squares.len() > MAX_TARGETS {
        return Err(EngineError::Invariant(
            "spy marks at most two pawns".to_owned(),
        ));
    }
    let mut seen = BTreeSet::new();
    let mut chosen = Vec::new();
    for square in squares {
        if !seen.insert(square) {
            return Err(EngineError::Invariant(
                "spy targets must be distinct".to_owned(),
            ));
        }
        let Some(piece) = state.board.get(square) else {
            return Err(EngineError::Invariant(
                "spy target square is empty".to_owned(),
            ));
        };
        if piece.owner != actor.opponent() || piece.kind != PieceKind::Pawn {
            return Err(EngineError::Invariant(
                "spy only marks enemy pawns".to_owned(),
            ));
        }
        chosen.push(piece.id);
    }
    for piece_id in chosen {
        state.card_runtime.spies.insert(piece_id, actor);
    }
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    let opponent = actor.opponent();
    let candidates = state
        .board
        .iter()
        .filter(|(_, piece)| piece.owner == opponent && piece.kind == PieceKind::Pawn)
        .map(|(square, _)| square)
        .collect::<Vec<_>>();
    // One or two enemy pawns: upstream marks every entry of the chosen list
    // (main bundle 46905-46915), so both sizes are valid and only the pair forms were
    // previously missing.
    for i in 0..candidates.len() {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(vec![candidates[i]]),
        });
        for j in i + 1..candidates.len() {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![candidates[i], candidates[j]]),
            });
        }
    }
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let count = |owner, kind| {
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .count()
    };
    require(count(player.opponent(), Pawn) > 0, "상대 폰이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::SPY.to_owned(),
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

    /// Three enemy pawns must yield three singles and three pairs.
    #[test]
    fn spy_enumerates_one_and_two_enemy_pawns() {
        let state = position(&[
            ("b7", Player::Black, PieceKind::Pawn),
            ("c7", Player::Black, PieceKind::Pawn),
            ("d7", Player::Black, PieceKind::Pawn),
            ("e2", Player::White, PieceKind::Pawn),
        ]);
        let offered = offered(&state);
        assert_eq!(offered.iter().filter(|s| s.len() == 1).count(), 3);
        assert_eq!(offered.iter().filter(|s| s.len() == 2).count(), 3);
        assert_eq!(offered.len(), 6);
    }

    #[test]
    fn spy_marks_the_chosen_pawns_and_rejects_bad_targets() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b7", Player::Black, PieceKind::Pawn),
            ("c7", Player::Black, PieceKind::Pawn),
            ("e2", Player::White, PieceKind::Pawn),
        ]);
        let b7 = state.board.get("b7".parse().unwrap()).unwrap();
        let c7 = state.board.get("c7".parse().unwrap()).unwrap();
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::Squares(vec!["b7".parse().unwrap(), "c7".parse().unwrap()]),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(state.card_runtime.spies.get(&b7.id), Some(&Player::White));
        assert_eq!(state.card_runtime.spies.get(&c7.id), Some(&Player::White));

        for target in [
            // An allied pawn is not a valid target.
            ActionTarget::Squares(vec!["e2".parse().unwrap()]),
            // Duplicates.
            ActionTarget::Squares(vec!["b7".parse().unwrap(), "b7".parse().unwrap()]),
            // More than two.
            ActionTarget::Squares(vec![
                "b7".parse().unwrap(),
                "c7".parse().unwrap(),
                "e2".parse().unwrap(),
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

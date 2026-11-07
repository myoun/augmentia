use super::*;
pub(super) const ID: &str = crate::DEATH_SQUAD;

/// Most pawns one cast may convert. The October 27 balance pass replaced the
/// whole-file conversion with "up to three chosen own pawns" (main bundle
/// 43459-43467).
const MAX_TARGETS: usize = 3;

/// Fanatic conversion of one to three chosen allied pawns.
///
/// Upstream validates that every selected square holds an own pawn, then converts
/// each in place while keeping piece identity and marking it moved
/// (`applyDeathSquadV4`, main bundle 43474-43494). Nothing is captured.
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let squares = squares_target(target)?;
    if squares.is_empty() || squares.len() > MAX_TARGETS {
        return Err(EngineError::Invariant(
            "death squad needs one to three pawn targets".to_owned(),
        ));
    }
    let mut seen = BTreeSet::new();
    let mut chosen = Vec::new();
    for square in squares {
        if !seen.insert(square) {
            return Err(EngineError::Invariant(
                "death squad targets must be distinct".to_owned(),
            ));
        }
        let Some(piece) = state.board.get(square) else {
            return Err(EngineError::Invariant(
                "death squad target square is empty".to_owned(),
            ));
        };
        if piece.owner != actor || piece.kind != PieceKind::Pawn {
            return Err(EngineError::Invariant(
                "death squad only converts allied pawns".to_owned(),
            ));
        }
        chosen.push((square, piece));
    }
    for (square, piece) in chosen {
        engine.clear_queens_gambit_protection(state, piece.id);
        let converted = engine.transform_current_piece(state, square, PieceKind::Fanatic, true)?;
        debug_assert_eq!(converted.id, piece.id, "death squad preserves identity");
        events.push(Event::EffectApplied {
            source_card_instance: 0,
            command: format!("death-squad:{}:{}", converted.id, square),
        });
    }
    Ok(())
}

/// Every combination of one to three allied pawns is offered.
fn actions(_engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    let pawns = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| piece.owner == card.owner && piece.kind == PieceKind::Pawn)
        .map(|(square, _)| square)
        .collect::<Vec<_>>();
    for i in 0..pawns.len() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(vec![pawns[i]]),
        });
        for j in i + 1..pawns.len() {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![pawns[i], pawns[j]]),
            });
            for k in j + 1..pawns.len() {
                out.push(Action::UseCard {
                    instance_id: card.instance_id,
                    target: ActionTarget::Squares(vec![pawns[i], pawns[j], pawns[k]]),
                });
            }
        }
    }
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == owner && piece.kind == PieceKind::Pawn),
        "광신도로 바꿀 아군 폰이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, position, single};
    use super::*;

    fn use_card(
        engine: &Engine,
        state: &mut GameState,
        actor: Player,
        target: ActionTarget,
    ) -> Result<(), EngineError> {
        play(engine, state, actor, 1, &target, &mut Vec::new())
    }

    fn targets(squares: &[&str]) -> ActionTarget {
        ActionTarget::Squares(squares.iter().map(|s| s.parse().unwrap()).collect())
    }

    #[test]
    fn death_squad_converts_up_to_three_chosen_pawns() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d2", Player::White, PieceKind::Pawn),
            ("d4", Player::White, PieceKind::Pawn),
            ("e2", Player::White, PieceKind::Pawn),
            ("f2", Player::White, PieceKind::Pawn),
        ]);
        let d2 = at(&state, "d2");
        let e2 = at(&state, "e2");
        let f2 = at(&state, "f2");
        use_card(&engine, &mut state, Player::White, targets(&["d2", "e2"])).unwrap();
        assert_eq!(
            state.board.get("d2".parse().unwrap()).unwrap().kind,
            PieceKind::Fanatic
        );
        assert_eq!(
            state.board.get("e2".parse().unwrap()).unwrap().kind,
            PieceKind::Fanatic
        );
        // Identity preserved, other pawns untouched.
        assert_eq!(state.board.get("d2".parse().unwrap()).unwrap().id, d2.id);
        assert_eq!(state.board.get("e2".parse().unwrap()).unwrap().id, e2.id);
        assert_eq!(
            state.board.get("f2".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        assert_eq!(state.board.get("f2".parse().unwrap()).unwrap().id, f2.id);
        assert!(!state.card_runtime.moved_piece_ids.contains(&f2.id));
        assert!(state.card_runtime.moved_piece_ids.contains(&d2.id));
        assert_eq!(state.turn, Player::White, "PRESERVE_TURN");
    }

    #[test]
    fn death_squad_accepts_one_target_and_rejects_more_than_three() {
        let engine = Engine::bundled_latest().unwrap();
        for squares in [vec!["d2"], vec!["d2", "d4", "e2", "f2"]] {
            let mut state = position(&[
                ("d2", Player::White, PieceKind::Pawn),
                ("d4", Player::White, PieceKind::Pawn),
                ("e2", Player::White, PieceKind::Pawn),
                ("f2", Player::White, PieceKind::Pawn),
            ]);
            let before = state.clone();
            let result = use_card(&engine, &mut state, Player::White, targets(&squares));
            if squares.len() <= MAX_TARGETS {
                assert!(result.is_ok(), "{squares:?} must be accepted");
            } else {
                assert!(result.is_err(), "{squares:?} must be rejected");
                assert_eq!(state, before, "a rejected target must not mutate");
            }
        }
    }

    #[test]
    fn death_squad_rejects_foreign_missing_or_duplicate_targets() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d2", Player::White, PieceKind::Pawn),
            ("h7", Player::Black, PieceKind::Pawn),
        ]);
        for target in [
            single("h7"),
            single("a1"),
            targets(&["d2", "d2"]),
            ActionTarget::None,
        ] {
            let before = state.clone();
            assert!(
                use_card(&engine, &mut state, Player::White, target.clone()).is_err(),
                "{target:?} must be rejected"
            );
            assert_eq!(state, before, "{target:?} mutated the board");
        }
    }

    #[test]
    fn death_squad_offers_every_combination_of_one_to_three_pawns() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(&[
            ("b2", Player::White, PieceKind::Pawn),
            ("b5", Player::White, PieceKind::Pawn),
            ("g2", Player::White, PieceKind::Pawn),
        ]);
        let card = CardInstance {
            instance_id: 1,
            definition_id: crate::DEATH_SQUAD.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        };
        let mut out = Vec::new();
        actions(&engine, &state, &card, &mut out);
        // Three pawns: 3 singles + 3 pairs + 1 triple.
        assert_eq!(out.len(), 7);
        assert!(eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible());
    }
}

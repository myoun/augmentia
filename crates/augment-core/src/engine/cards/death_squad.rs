use super::*;
pub(super) const ID: &str = crate::DEATH_SQUAD;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Fanatic conversion of every allied pawn on the selected file.
///
/// The upstream contract (L198337-198364) selects one square, reads only its
/// file, then converts every own pawn in that file to a fanatic while keeping
/// piece identity and marking the piece moved. It is not a capture: the pawns
/// stay on their squares.
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let file = square_target(target)?.file();
    let pawns = state
        .board
        .iter()
        .filter(|(square, piece)| {
            square.file() == file && piece.owner == actor && piece.kind == PieceKind::Pawn
        })
        .collect::<Vec<_>>();
    if pawns.is_empty() {
        return Err(EngineError::Invariant(
            "death squad has no allied pawn in the selected file".to_owned(),
        ));
    }
    for (square, piece) in pawns {
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

/// Squares that currently hold at least one allied pawn are offered, one per
/// file, mirroring the 0915 action generator. The selected square only needs to
/// name the file, so any occupied ally pawn square on that file is equivalent.
fn actions(_engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    let mut offered_files = BTreeSet::new();
    for (square, piece) in state.board.iter() {
        if piece.owner == card.owner
            && piece.kind == PieceKind::Pawn
            && offered_files.insert(square.file())
        {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Square(square),
            });
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

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, movement, pair, position, single};
    use super::*;

    fn use_card(
        engine: &Engine,
        state: &mut GameState,
        actor: Player,
        target: ActionTarget,
    ) -> Result<(), EngineError> {
        play(engine, state, actor, 1, &target, &mut Vec::new())
    }

    #[test]
    fn death_squad_converts_every_allied_pawn_in_the_selected_file() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d2", Player::White, PieceKind::Pawn),
            ("d4", Player::White, PieceKind::Pawn),
            ("d6", Player::Black, PieceKind::Pawn),
            ("e2", Player::White, PieceKind::Pawn),
        ]);
        let d2 = at(&state, "d2");
        let d4 = at(&state, "d4");
        let e2 = at(&state, "e2");
        let enemy = at(&state, "d6");
        use_card(&engine, &mut state, Player::White, single("d2")).unwrap();
        assert_eq!(
            state.board.get("d2".parse().unwrap()).unwrap().kind,
            PieceKind::Fanatic
        );
        assert_eq!(
            state.board.get("d4".parse().unwrap()).unwrap().kind,
            PieceKind::Fanatic
        );
        // Identity is preserved.
        assert_eq!(state.board.get("d2".parse().unwrap()).unwrap().id, d2.id);
        assert_eq!(state.board.get("d4".parse().unwrap()).unwrap().id, d4.id);
        // Another file and the opponent are untouched.
        assert_eq!(state.board.get("e2".parse().unwrap()).unwrap().id, e2.id);
        assert_eq!(
            state.board.get("e2".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        assert_eq!(state.board.get("d6".parse().unwrap()).unwrap().id, enemy.id);
        assert_eq!(
            state.board.get("d6".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
        // Converted pawns count as moved (upstream sets attributes.moved).
        assert!(state.card_runtime.moved_piece_ids.contains(&d2.id));
        assert!(state.card_runtime.moved_piece_ids.contains(&d4.id));
        assert!(!state.card_runtime.moved_piece_ids.contains(&e2.id));
        // PRESERVE_TURN.
        assert_eq!(state.turn, Player::White);
    }

    #[test]
    fn death_squad_accepts_any_square_of_the_file_and_keeps_the_piece_count() {
        let engine = Engine::bundled_latest().unwrap();
        for target in [single("d4"), ActionTarget::Square("d7".parse().unwrap())] {
            let mut state = position(&[
                ("d2", Player::White, PieceKind::Pawn),
                ("d4", Player::White, PieceKind::Pawn),
            ]);
            let before = state.board.iter().count();
            use_card(&engine, &mut state, Player::White, target.clone()).unwrap();
            assert_eq!(state.board.iter().count(), before);
            assert_eq!(
                state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.kind == PieceKind::Fanatic)
                    .count(),
                2,
                "target {target:?} must select the file, not a single piece"
            );
        }
    }

    #[test]
    fn death_squad_rejects_an_empty_or_foreign_file_without_mutating_state() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d2", Player::White, PieceKind::Pawn),
            ("h7", Player::Black, PieceKind::Pawn),
        ]);
        for target in [
            ActionTarget::Square("a1".parse().unwrap()),
            single("h7"),
            ActionTarget::None,
            pair("d2", "h7"),
        ] {
            let before = state.clone();
            assert!(
                use_card(&engine, &mut state, Player::White, target.clone()).is_err(),
                "target {target:?} must be rejected"
            );
            assert_eq!(
                state, before,
                "rejected target {target:?} mutated the board"
            );
        }
        assert_eq!(
            state
                .board
                .iter()
                .filter(|(_, piece)| piece.kind == PieceKind::Fanatic)
                .count(),
            0
        );
        let _ = movement("d2", "d4");
    }

    #[test]
    fn death_squad_offers_one_action_per_file_with_an_allied_pawn() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(&[
            ("b2", Player::White, PieceKind::Pawn),
            ("b5", Player::White, PieceKind::Pawn),
            ("g2", Player::White, PieceKind::Pawn),
            ("c6", Player::Black, PieceKind::Pawn),
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
        let files = out
            .iter()
            .map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::Square(square),
                    ..
                } => square.file(),
                other => panic!("unexpected action {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(files, vec![1, 6], "one candidate per allied pawn file");

        // Eligibility needs at least one allied pawn; the opponent's does not count.
        assert!(eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible());
        let mut empty = position(&[("c6", Player::Black, PieceKind::Pawn)]);
        assert!(!eligibility(&empty, Player::White, AcquisitionContext::Draft).is_eligible());
        empty.board.set("a2".parse().unwrap(), None);
        assert!(
            empty
                .board
                .iter()
                .all(|(_, p)| p.kind != PieceKind::Pawn || p.owner == Player::Black)
        );
    }
}

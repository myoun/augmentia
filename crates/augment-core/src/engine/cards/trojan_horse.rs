use super::*;
pub(super) const ID: &str = crate::TROJAN_HORSE;

/// Trojan Horse selects an allied knight and an allied pawn, removes the pawn into
/// the knight, and marks the knight. It now consumes the turn.
///
/// Mirrors `trojanSelectionContract` / `applyTrojanHorseV11` (main bundle
/// 47063-47082): the knight must be an unmarked own knight, the pawn an own
/// non-royal pawn, and the pawn is removed without being a capture.
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let [knight_square, pawn_square] = two_square_target(target)?;
    let knight = state
        .board
        .get(knight_square)
        .ok_or_else(|| EngineError::Invariant("missing trojan knight".into()))?;
    if knight.owner != actor
        || knight.kind != PieceKind::Knight
        || state.card_runtime.trojan_piece_ids.contains(&knight.id)
    {
        return Err(EngineError::Invariant(
            "trojan horse needs an unmarked allied knight".into(),
        ));
    }
    let pawn = state
        .board
        .get(pawn_square)
        .ok_or_else(|| EngineError::Invariant("missing trojan pawn".into()))?;
    if pawn.owner != actor || pawn.kind != PieceKind::Pawn || engine.is_royal_piece(state, pawn) {
        return Err(EngineError::Invariant(
            "trojan horse needs an allied non-royal pawn".into(),
        ));
    }
    state.card_runtime.trojan_piece_ids.insert(knight.id);
    // The pawn is consumed into the knight. Upstream removes it through the
    // entry-removal path, so this is not a capture and grants no capture credit.
    engine.delete_piece_for_card(state, pawn_square, pawn, actor, ID, events);
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("trojan-horse:{}:{}", knight.id, pawn_square),
    });
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    out: &mut Vec<Action>,
) {
    let actor = state.turn;
    let knights = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| {
            piece.owner == actor
                && piece.kind == PieceKind::Knight
                && !state.card_runtime.trojan_piece_ids.contains(&piece.id)
        })
        .map(|(square, _)| square)
        .collect::<Vec<_>>();
    let pawns = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| {
            piece.owner == actor
                && piece.kind == PieceKind::Pawn
                && !_engine.is_royal_piece(state, *piece)
        })
        .map(|(square, _)| square)
        .collect::<Vec<_>>();
    for knight in &knights {
        for pawn in &pawns {
            if knight == pawn {
                continue;
            }
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![*knight, *pawn]),
            });
        }
    }
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    let rt = &state.card_runtime;
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    require(
        has_own(&|p| p.kind == PieceKind::Knight && !rt.trojan_piece_ids.contains(&p.id))
            && has_own(&|p| p.kind == PieceKind::Pawn),
        "아군 나이트와 폰이 하나씩 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, position};
    use super::*;

    fn use_card(
        engine: &Engine,
        state: &mut GameState,
        actor: Player,
        target: ActionTarget,
    ) -> Result<(), EngineError> {
        play(engine, state, actor, 1, &target, &mut Vec::new())
    }

    fn pair(a: &str, b: &str) -> ActionTarget {
        ActionTarget::Squares(vec![a.parse().unwrap(), b.parse().unwrap()])
    }

    #[test]
    fn trojan_horse_marks_the_knight_and_consumes_the_pawn_without_capture_credit() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b1", Player::White, PieceKind::Knight),
            ("d2", Player::White, PieceKind::Pawn),
        ]);
        let knight = at(&state, "b1");
        let pawn = at(&state, "d2");
        use_card(&engine, &mut state, Player::White, pair("b1", "d2")).unwrap();
        assert!(state.card_runtime.trojan_piece_ids.contains(&knight.id));
        assert!(
            state.board.get("d2".parse().unwrap()).is_none(),
            "the pawn is consumed"
        );
        assert!(
            state.board.get("b1".parse().unwrap()).is_some(),
            "the knight stays put"
        );
        debug_assert_ne!(knight.id, pawn.id);
    }

    #[test]
    fn trojan_horse_rejects_a_marked_knight_a_foreign_pawn_or_one_target() {
        let engine = Engine::bundled_latest().unwrap();
        let base = position(&[
            ("b1", Player::White, PieceKind::Knight),
            ("g1", Player::White, PieceKind::Knight),
            ("d2", Player::White, PieceKind::Pawn),
            ("d7", Player::Black, PieceKind::Pawn),
        ]);
        // Mark the first knight so it can no longer be chosen.
        let marked = at(&base, "b1");
        let mut marked_state = base.clone();
        marked_state.card_runtime.trojan_piece_ids.insert(marked.id);
        for (state, target) in [
            (marked_state, pair("b1", "d2")),
            (base.clone(), pair("g1", "d7")),
            (
                base.clone(),
                ActionTarget::Squares(vec!["g1".parse().unwrap()]),
            ),
        ] {
            let mut state = state;
            let before = state.clone();
            assert!(
                use_card(&engine, &mut state, Player::White, target.clone()).is_err(),
                "{target:?} must be rejected"
            );
            assert_eq!(state, before, "{target:?} mutated state");
        }
    }

    /// The 2026-10-27 balance pass moved Trojan Horse to END_TURN: using it completes the
    /// actor's turn instead of leaving it open (`usesTurnEndingTrojan` +
    /// `finishV13CompletedCardTurn`, bundle 7250773 and 9579552).
    #[test]
    fn trojan_horse_ends_the_turn() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b1", Player::White, PieceKind::Knight),
            ("d2", Player::White, PieceKind::Pawn),
        ]);
        // Give the actor a second action so an unended turn would be observable.
        state.actions_remaining = 2;
        state = engine
            .grant_card(&state, 1, crate::TROJAN_HORSE, Player::White)
            .expect("trojan-horse is supported")
            .state;
        assert_eq!(state.turn, Player::White);

        let after = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Squares(vec![
                        "b1".parse().unwrap(),
                        "d2".parse().unwrap(),
                    ]),
                },
            )
            .expect("the declaration is legal");

        assert!(
            after
                .events
                .iter()
                .any(|event| matches!(event, Event::TurnEnded { .. })),
            "using trojan-horse must end the turn: {:?}",
            after.events
        );
        assert_eq!(
            after.state.turn,
            Player::Black,
            "the turn passes to the opponent"
        );
        assert_eq!(after.state.cards[0].charges, 0);
    }
}

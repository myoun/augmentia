use super::*;
pub(super) const ID: &str = crate::ULTIMATUM;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Half-turns of countdown. Live `applyUltimatumV5` (JS:199066-199078) stores
/// `duePly: currentPly + 8`, i.e. four full moves after the declaration.
const ULTIMATUM_HALF_TURNS: u32 = 8;

fn play(
    _engine: &Engine,
    state: &mut GameState,
    _actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    if target != &ActionTarget::None {
        return Err(EngineError::Invariant(
            "ultimatum must not have a target".into(),
        ));
    }
    // A second declaration replaces the pending one, like the live handler
    // filtering out the previous `ultimatum` effect before pushing a new one.
    state.card_runtime.ultimatum = Some(state.ply.saturating_add(ULTIMATUM_HALF_TURNS));
    Ok(())
}

fn actions(_engine: &Engine, _state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    out.push(Action::UseCard {
        instance_id: card.instance_id,
        target: ActionTarget::None,
    });
}

fn eligibility(state: &GameState, player: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == player && !royal(state, piece)),
        "킹이 아닌 아군 기물이 필요합니다.",
    )
}

// Settlement already lives in `Engine::tick_end_statuses_after_turn`
// (engine/turn_effects.rs L264-279): once `state.ply` reaches the deadline it
// removes every non-royal piece whose id is absent from
// `card_runtime.moved_piece_ids` and clears the deadline. Do not add a second
// settlement path here.
//
// Known deviation from live `resolveUltimatum` (JS:343743-343776), which tracks
// ids moved *during* the four-move window: the current engine has no
// window-scoped movement set, so it uses the ever-moved set
// (`moved_piece_ids`). A piece that moved before the card was declared is
// therefore spared, and a piece that already moved once cannot be removed.
// The live capture-owner attribution (`victim.color === by ? opponent : by`) is
// also collapsed to the acting player in that shared settlement.

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::ULTIMATUM.to_owned(),
            owner: Player::White,
            half_stars: 9,
            charges: 1,
            available_after_turn: None,
        }
    }

    fn declare(engine: &Engine, state: &mut GameState) {
        let actor = state.turn;
        super::play(
            engine,
            state,
            actor,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        )
        .unwrap();
    }

    /// One completed turn: the move that ended it already advanced `ply`, which
    /// is what the shared settlement compares against the deadline.
    fn tick(engine: &Engine, state: &mut GameState) {
        state.ply = state.ply.saturating_add(1);
        let actor = state.turn;
        engine.tick_end_statuses_after_turn(state, actor, &mut Vec::new());
    }

    #[test]
    fn ultimatum_sets_a_four_move_deadline_and_offers_a_targetless_action() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("b1", Player::White, PieceKind::Rook)]);
        state.add_card(card());
        state.ply = 3;
        declare(&engine, &mut state);
        assert_eq!(state.card_runtime.ultimatum, Some(11));

        let mut out = Vec::new();
        super::actions(&engine, &state, &state.cards[0].clone(), &mut out);
        assert_eq!(
            out,
            vec![Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None
            }]
        );
        assert!(
            engine
                .apply(
                    &state,
                    Action::UseCard {
                        instance_id: 1,
                        target: single("b1"),
                    },
                )
                .is_err(),
            "ultimatum takes no target"
        );
    }

    #[test]
    fn ultimatum_expires_after_eight_plies_and_removes_only_unmoved_non_royal_pieces() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("c1", Player::White, PieceKind::Bishop),
            ("b8", Player::Black, PieceKind::Rook),
            ("c8", Player::Black, PieceKind::Bishop),
        ]);
        let moved = at(&state, "b1");
        let enemy_moved = at(&state, "b8");
        state.card_runtime.moved_piece_ids.insert(moved.id);
        state.card_runtime.moved_piece_ids.insert(enemy_moved.id);
        declare(&engine, &mut state);

        // The settlement compares the deadline against `state.ply`, which the
        // engine advances on every completed half-turn.
        tick(&engine, &mut state);
        assert_eq!(state.card_runtime.ultimatum, Some(8));
        assert!(state.board.get("c1".parse().unwrap()).is_some());

        for _ in 0..7 {
            tick(&engine, &mut state);
        }
        assert_eq!(state.card_runtime.ultimatum, None, "deadline clears");
        assert!(state.board.get("b1".parse().unwrap()).is_some());
        assert!(state.board.get("b8".parse().unwrap()).is_some());
        assert!(state.board.get("c1".parse().unwrap()).is_none());
        assert!(state.board.get("c8".parse().unwrap()).is_none());
        // Both kings are exempt from `is_ultimatum_target`.
        assert!(state.board.get("a1".parse().unwrap()).is_some());
        assert!(state.board.get("h8".parse().unwrap()).is_some());
    }

    #[test]
    fn ultimatum_counts_down_once_per_turn_and_clears_only_when_due() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("c1", Player::White, PieceKind::Bishop)]);
        declare(&engine, &mut state);
        let deadline = state.card_runtime.ultimatum.unwrap();
        for expected in 1..=8 {
            tick(&engine, &mut state);
            if expected < 8 {
                assert_eq!(
                    state.card_runtime.ultimatum,
                    Some(deadline),
                    "deadline unchanged until it is due"
                );
                assert!(
                    state.board.get("c1".parse().unwrap()).is_some(),
                    "nothing is removed before the deadline"
                );
            }
        }
        assert_eq!(state.card_runtime.ultimatum, None);
        assert!(state.board.get("c1".parse().unwrap()).is_none());
    }

    #[test]
    fn ultimatum_eligibility_requires_a_non_king_piece() {
        let state = position(&[]);
        assert!(
            super::eligibility(&state, Player::White, AcquisitionContext::Draft)
                .reason()
                .is_some(),
            "a lone king is not enough"
        );
        let with_piece = position(&[("b1", Player::White, PieceKind::Rook)]);
        assert!(
            super::eligibility(&with_piece, Player::White, AcquisitionContext::Draft).is_eligible()
        );
    }
}

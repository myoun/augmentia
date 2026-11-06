use super::*;

pub(super) const ID: &str = crate::RESOLVE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

/// Arming the card is the whole acquisition: the shared engine already owns the
/// resolute-move contract and only needs the owner flag.
///
/// `septemberNotePawnCapture` (main-RjrZ5fFP.js:1952-1958) arms one credited pawn
/// move when a friendly pawn is captured while `resolve[color]` is set;
/// `septemberConsumeResolveMove` (L1959-1971) spends it on the first friendly pawn
/// move of that turn; `septemberUseResolveCredit` (L1972-1982) keeps the turn.
/// Core does the same from `record_new_capture_reactions` (engine.rs:638-645) and
/// the credited branch in `apply_move` (engine.rs:4143-4152).
fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.resolve[owner.index()] = true;
    Ok(())
}

/// Draw requirement is `ANY`, and the effect is legal in any position (it only
/// arms a future capture reaction).
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    enabled(state.card_runtime.resolve[owner.index()]).unwrap_or(CardEligibility::Eligible)
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, handoff, movement, position};
    use super::*;

    fn armed(engine: &Engine, mut state: GameState) -> GameState {
        acquire(engine, &mut state, Player::White, &mut Vec::new()).unwrap();
        assert!(state.card_runtime.resolve[Player::White.index()]);
        state
    }

    #[test]
    fn resolve_acquire_arms_only_its_owner() {
        let engine = Engine::bundled_latest().unwrap();
        let state = armed(&engine, position(&[]));
        assert!(state.card_runtime.resolve[Player::White.index()]);
        assert!(!state.card_runtime.resolve[Player::Black.index()]);
        assert!(!state.card_runtime.resolve_ready[Player::White.index()]);
        // The flag alone grants nothing until a friendly pawn actually dies.
        assert!(!state.card_runtime.resolve_ready[Player::Black.index()]);
    }

    #[test]
    fn resolve_gives_the_first_pawn_move_a_free_turn_after_a_pawn_loss() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = armed(
            &engine,
            position(&[
                ("c4", Player::White, PieceKind::Pawn),
                ("e2", Player::White, PieceKind::Pawn),
                ("e5", Player::Black, PieceKind::Knight),
            ]),
        );
        state.turn = Player::Black;
        let after_capture = engine.apply(&state, movement("e5", "c4")).unwrap().state;
        assert!(
            after_capture.card_runtime.resolve_ready[Player::White.index()],
            "a captured friendly pawn must arm the credit"
        );
        assert_eq!(after_capture.turn, Player::White);
        let turns_before = after_capture.turns_taken;

        let after_pawn_move = engine
            .apply(&after_capture, movement("e2", "e3"))
            .unwrap()
            .state;
        // The credit is spent, so the turn did not pass.
        assert_eq!(after_pawn_move.turn, Player::White);
        assert_eq!(after_pawn_move.turns_taken, turns_before);
        assert!(!after_pawn_move.card_runtime.resolve_ready[Player::White.index()]);
    }

    #[test]
    fn resolve_ignores_losses_of_non_pawns_and_moves_of_non_pawns() {
        let engine = Engine::bundled_latest().unwrap();
        // A captured knight arms nothing.
        let mut state = armed(
            &engine,
            position(&[
                ("c4", Player::White, PieceKind::Knight),
                ("e2", Player::White, PieceKind::Pawn),
                ("e5", Player::Black, PieceKind::Knight),
            ]),
        );
        state.turn = Player::Black;
        let lost_knight = engine.apply(&state, movement("e5", "c4")).unwrap().state;
        assert!(!lost_knight.card_runtime.resolve_ready[Player::White.index()]);

        // An armed credit is not spent by a non-pawn move: the turn ends normally.
        let mut ready = armed(
            &engine,
            position(&[
                ("c4", Player::White, PieceKind::Pawn),
                ("d1", Player::White, PieceKind::Rook),
                ("e5", Player::Black, PieceKind::Knight),
            ]),
        );
        ready.turn = Player::Black;
        let armed_state = engine.apply(&ready, movement("e5", "c4")).unwrap().state;
        assert!(armed_state.card_runtime.resolve_ready[Player::White.index()]);
        let after_rook = engine
            .apply(&armed_state, movement("d1", "d2"))
            .unwrap()
            .state;
        assert_eq!(after_rook.turn, Player::Black);
        assert!(
            after_rook.card_runtime.resolve_ready[Player::White.index()],
            "only a pawn move may spend the credit"
        );
    }

    #[test]
    fn resolve_without_the_card_never_arms_a_credit() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("c4", Player::White, PieceKind::Pawn),
            ("e2", Player::White, PieceKind::Pawn),
            ("e5", Player::Black, PieceKind::Knight),
        ]);
        state.turn = Player::Black;
        let after_capture = engine.apply(&state, movement("e5", "c4")).unwrap().state;
        assert!(!after_capture.card_runtime.resolve_ready[Player::White.index()]);
        let after_pawn_move = engine
            .apply(&after_capture, movement("e2", "e3"))
            .unwrap()
            .state;
        assert_eq!(after_pawn_move.turn, Player::Black);
        // Both sides completed one turn each: the pawn move was not free.
        assert_eq!(after_pawn_move.turns_taken, [1, 1]);
        let _ = at(&after_pawn_move, "e3");
    }

    #[test]
    fn resolve_eligibility_rejects_a_second_arming() {
        let engine = Engine::bundled_latest().unwrap();
        let state = armed(&engine, position(&[]));
        assert!(
            !eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible(),
            "an already armed Resolve must not be re-acquired"
        );
        assert!(eligibility(&state, Player::Black, AcquisitionContext::Draft).is_eligible());
        let mut unarmed = position(&[]);
        handoff(&engine, &mut unarmed);
        assert!(eligibility(&unarmed, Player::White, AcquisitionContext::Draft).is_eligible());
    }
}

use super::*;
pub(super) const ID: &str = crate::ANTICHESS;

/// Consecutive own turns of losing material before the opponent wins.
const LOSS_STREAK_TO_WIN: u8 = 3;

pub(super) fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.antichess[owner.index()] = Some(AntichessProgress {
        count: 0,
        last_turn: u32::MAX,
    });
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    if let Some(ineligible) = enabled(state.card_runtime.antichess[owner.index()].is_some()) {
        return ineligible;
    }
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    /// Record that `attacker` captured at least one of `victim_color`'s pieces.
    ///
    /// Upstream tracks a per-side streak (`recordWorkerOctober27EnemyCaptures`,
    /// main bundle 29572-29593): the count increases when the victim loses
    /// material on consecutive own turns and resets on a gap. Reaching three
    /// consecutive losses hands the win to the victim.
    pub(in crate::engine) fn record_antichess_loss(
        &self,
        state: &mut GameState,
        attacker: Player,
        victim: Player,
    ) {
        let Some(progress) = state.card_runtime.antichess[victim.index()].as_mut() else {
            return;
        };
        // Only losses on the attacker's own turn count, matching the upstream
        // `boardState.turn === attacker.color` guard.
        if state.turn != attacker {
            return;
        }
        let turn = state.turns_taken[attacker.index()];
        if progress.last_turn == turn {
            return;
        }
        progress.count = if progress.last_turn == turn.saturating_sub(1) {
            progress.count.saturating_add(1)
        } else {
            1
        };
        progress.last_turn = turn;
        let streak = progress.count;
        if streak >= LOSS_STREAK_TO_WIN {
            state.status = GameStatus::Won {
                winner: victim,
                reason: WinReason::CardEffect(ID.to_owned()),
            };
        }
    }

    /// Reset the streak when the victim completes a turn without losing material.
    pub(in crate::engine) fn tick_antichess_turn(&self, state: &mut GameState, owner: Player) {
        let Some(progress) = state.card_runtime.antichess[owner.index()].as_mut() else {
            return;
        };
        let completed = state.turns_taken[owner.index()];
        // A streak survives only if the loss happened on the immediately preceding
        // opponent turn.
        if progress.last_turn != completed {
            progress.count = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    fn granted(engine: &Engine) -> GameState {
        let before = position(&[("a1", Player::White, PieceKind::King)]);
        engine
            .grant_card(&before, 1, crate::ANTICHESS, Player::White)
            .expect("antichess is supported")
            .state
    }

    /// The streak advances only on consecutive opponent turns and resets on a gap.
    #[test]
    fn antichess_streak_counts_consecutive_losses_and_resets_on_a_gap() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = granted(&engine);
        // Loss on opponent turn 0 -> count 1.
        state.turn = Player::Black;
        state.turns_taken[Player::Black.index()] = 0;
        engine.record_antichess_loss(&mut state, Player::Black, Player::White);
        let progress = state.card_runtime.antichess[Player::White.index()].unwrap();
        assert_eq!((progress.count, progress.last_turn), (1, 0));

        // Same turn again does not double-count.
        engine.record_antichess_loss(&mut state, Player::Black, Player::White);
        assert_eq!(
            state.card_runtime.antichess[Player::White.index()]
                .unwrap()
                .count,
            1
        );

        // A skipped turn breaks the streak, so the next loss restarts at one.
        state.turns_taken[Player::Black.index()] = 5;
        engine.record_antichess_loss(&mut state, Player::Black, Player::White);
        assert_eq!(
            state.card_runtime.antichess[Player::White.index()]
                .unwrap()
                .count,
            1
        );
    }

    #[test]
    fn three_consecutive_losses_win_the_game_for_the_victim() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = granted(&engine);
        state.turn = Player::Black;
        for turn in 0..3 {
            state.turns_taken[Player::Black.index()] = turn;
            engine.record_antichess_loss(&mut state, Player::Black, Player::White);
        }
        assert_eq!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect(crate::ANTICHESS.to_owned()),
            }
        );
    }

    #[test]
    fn antichess_is_a_passive_that_rejects_a_second_acquisition() {
        let engine = Engine::bundled_latest().unwrap();
        let state = granted(&engine);
        assert!(state.card_runtime.antichess[Player::White.index()].is_some());
        // Eligibility rejects a second copy; `grant_card` itself does not run the
        // eligibility gate, so this is asserted through it.
        assert!(
            !eligibility(&state, Player::White, AcquisitionContext::Setup).is_eligible(),
            "a second acquisition must be ineligible"
        );
    }
}

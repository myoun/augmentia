use super::*;
pub(super) const ID: &str = crate::PENTA_CHECK;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.effects.penta_check[owner.index()] = Some(PentaCheckProgress {
        count: 0,
        opponent_checked: engine.card_royal_checked(state, owner),
    });
    Ok(())
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PentaCheckProgress {
    pub count: u8,
    pub opponent_checked: bool,
}

impl Engine {
    pub(in crate::engine) fn settle_penta_check_turn_start(
        &self,
        state: &mut GameState,
        owner: Player,
    ) {
        if state.ruleset_id == crate::LATEST_RULESET_ID
            && state.status == GameStatus::Ongoing
            && state.card_runtime.effects.penta_check[owner.index()].is_some_and(|p| p.count >= 5)
        {
            state.status = GameStatus::Won {
                winner: owner,
                reason: WinReason::CardEffect(crate::PENTA_CHECK.into()),
            };
        }
    }
}

impl Engine {
    pub(in crate::engine) fn observe_penta_checks(
        &self,
        state: &mut GameState,
        checked: [bool; 2],
    ) {
        for owner in [Player::White, Player::Black] {
            if let Some(progress) = &mut state.card_runtime.effects.penta_check[owner.index()] {
                if checked[owner.index()] && !progress.opponent_checked {
                    progress.count = (progress.count + 1).min(5);
                }
                progress.opponent_checked = checked[owner.index()];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::objectives::*;
    use super::*;
    #[test]
    fn penta_check_counts_five_new_checks_and_waits_until_owner_turn_start() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[("a2", Player::White, PieceKind::Rook)]);
        s = e
            .grant_card(&s, 1, crate::PENTA_CHECK, Player::White)
            .unwrap()
            .state;
        for (i, (from, to, king_from, king_to)) in [
            ("a2", "h2", "h8", "g8"),
            ("h2", "g2", "g8", "f8"),
            ("g2", "f2", "f8", "e8"),
            ("f2", "e2", "e8", "d8"),
            ("e2", "d2", "d8", "c8"),
        ]
        .into_iter()
        .enumerate()
        {
            s = e.apply(&s, movement(from, to)).unwrap().state;
            assert_eq!(
                s.card_runtime.effects.penta_check[0].unwrap().count,
                (i + 1) as u8
            );
            assert_eq!(s.status, GameStatus::Ongoing);
            s = e.apply(&s, movement(king_from, king_to)).unwrap().state;
            if i < 4 {
                assert_eq!(s.status, GameStatus::Ongoing);
            } else {
                assert_eq!(
                    s.status,
                    GameStatus::Won {
                        winner: Player::White,
                        reason: WinReason::CardEffect(crate::PENTA_CHECK.into())
                    }
                );
            }
        }
    }
    #[test]
    fn penta_check_does_not_count_initial_or_sustained_checks_and_observes_opponent_moves() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[("h2", Player::White, PieceKind::Rook)]);
        s = e
            .grant_card(&s, 1, crate::PENTA_CHECK, Player::White)
            .unwrap()
            .state;
        assert_eq!(s.card_runtime.effects.penta_check[0].unwrap().count, 0);
        s.actions_remaining = 2;
        s = e.apply(&s, movement("h2", "h3")).unwrap().state;
        assert_eq!(s.card_runtime.effects.penta_check[0].unwrap().count, 0);
        s = e.apply(&s, movement("h3", "g3")).unwrap().state;
        assert!(
            !s.card_runtime.effects.penta_check[0]
                .unwrap()
                .opponent_checked
        );
        s = e.apply(&s, movement("h8", "g8")).unwrap().state;
        assert_eq!(s.card_runtime.effects.penta_check[0].unwrap().count, 1);
    }
}

use super::*;
pub(super) const ID: &str = crate::PROPHECY;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.effects.prophecy_remaining[actor.index()] = Some(6);
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    if state.card_runtime.effects.prophecy_remaining[state.turn.index()].is_none()
        && state
            .board
            .iter()
            .any(|(_, p)| p.owner == state.turn && engine.is_royal_piece(state, p))
    {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let rt = &state.card_runtime;
    let i = player.index();
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    let own_royal = has_own(&|p| royal(state, p));
    if let Some(result) = enabled({
        rt.prophecy[i].is_some()
            || (state.ruleset_id == crate::LATEST_RULESET_ID
                && rt.effects.prophecy_remaining[i].is_some())
    }) {
        return result;
    }
    require(own_royal, "효과를 받을 아군 왕실 기물이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

impl Engine {
    pub(in crate::engine) fn cancel_prophecies(&self, state: &mut GameState) {
        if state.ruleset_id == crate::LATEST_RULESET_ID {
            state.card_runtime.effects.prophecy_remaining = [None; 2];
        }
    }
}

impl Engine {
    pub(in crate::engine) fn complete_prophecy_turn(
        &self,
        state: &mut GameState,
        _incoming: Player,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID || state.status != GameStatus::Ongoing {
            return;
        }
        self.settle_check_objectives(state);
        for owner in [Player::White, Player::Black] {
            if let Some(remaining) =
                &mut state.card_runtime.effects.prophecy_remaining[owner.index()]
            {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    state.card_runtime.effects.prophecy_remaining[owner.index()] = None;
                    state.status = GameStatus::Won {
                        winner: owner,
                        reason: WinReason::CardEffect(crate::PROPHECY.into()),
                    };
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::objectives::*;
    use super::*;
    #[test]
    fn prophecy_counts_six_completed_turns_and_extra_actions_do_not_tick() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[]);
        arm(&e, &mut s);
        s.actions_remaining = 2;
        s = e.apply(&s, movement("a1", "a2")).unwrap().state;
        assert_eq!(s.card_runtime.effects.prophecy_remaining[0], Some(6));
        for (i, (a, b)) in [
            ("a2", "a3"),
            ("h8", "h7"),
            ("a3", "b3"),
            ("h7", "g7"),
            ("b3", "c3"),
            ("g7", "f7"),
        ]
        .into_iter()
        .enumerate()
        {
            s = e.apply(&s, movement(a, b)).unwrap().state;
            if i < 5 {
                assert_eq!(s.status, GameStatus::Ongoing);
                assert_eq!(
                    s.card_runtime.effects.prophecy_remaining[0],
                    Some(5 - i as u8)
                );
            }
        }
        assert_eq!(
            s.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect(crate::PROPHECY.into())
            }
        );
    }
    #[test]
    fn prophecy_cancels_on_check_or_removal_before_last_tick_including_sacrifice() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("a2", Player::White, PieceKind::Rook),
            ("a7", Player::Black, PieceKind::Pawn),
        ]);
        arm(&e, &mut s);
        s.card_runtime.effects.prophecy_remaining[0] = Some(1);
        s = e.apply(&s, movement("a2", "a7")).unwrap().state;
        assert_eq!(s.status, GameStatus::Ongoing);
        assert_eq!(s.card_runtime.effects.prophecy_remaining, [None; 2]);
        let mut s = position(&[("a2", Player::White, PieceKind::Rook)]);
        arm(&e, &mut s);
        s.card_runtime.effects.prophecy_remaining[0] = Some(1);
        s = e.apply(&s, movement("a2", "h2")).unwrap().state;
        assert_eq!(s.status, GameStatus::Ongoing);
        assert_eq!(s.card_runtime.effects.prophecy_remaining, [None; 2]);
        let mut s = position(&[("c1", Player::White, PieceKind::Bishop)]);
        arm(&e, &mut s);
        s = e
            .grant_card(&s, 2, crate::GREEK_GIFT, Player::White)
            .unwrap()
            .state;
        s = e
            .apply(
                &s,
                Action::UseCard {
                    instance_id: 2,
                    target: ActionTarget::PieceAt("c1".parse().unwrap()),
                },
            )
            .unwrap()
            .state;
        assert_eq!(s.card_runtime.effects.prophecy_remaining, [None; 2]);
    }
    #[test]
    fn prophecy_started_while_checked_cancels_immediately_and_search_is_transactional() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[("h2", Player::White, PieceKind::Rook)]);
        arm(&e, &mut s);
        assert_eq!(s.cards[0].charges, 0);
        assert_eq!(s.card_runtime.effects.prophecy_remaining, [None; 2]);
        let mut quiet = position(&[]);
        arm(&e, &mut quiet);
        let before = quiet.clone();
        let mut chance = SearchChanceContext::new(31);
        let actual = e
            .apply_for_search(&quiet, movement("a1", "a2"), &mut chance)
            .unwrap();
        assert!(!actual.sampled_chance);
        assert_eq!(
            actual
                .transition
                .state
                .card_runtime
                .effects
                .prophecy_remaining[0],
            Some(5)
        );
        assert_eq!(quiet, before);
        let mut changed = quiet.clone();
        changed.card_runtime.effects.prophecy_remaining[0] = Some(4);
        assert_ne!(changed.position_key(), quiet.position_key());
    }
}

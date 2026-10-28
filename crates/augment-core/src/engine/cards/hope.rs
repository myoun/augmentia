use super::*;
pub(super) const ID: &str = crate::HOPE;
pub(super) fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.october.hope[owner.index()] = true;
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let rt = &state.card_runtime;
    let i = player.index();
    if let Some(result) = enabled(rt.october.hope[i]) {
        return result;
    }
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn settle_hope_turn_start(&self, state: &mut GameState, owner: Player) {
        if state.ruleset_id == crate::LATEST_RULESET_ID
            && state.status == GameStatus::Ongoing
            && state.card_runtime.october.hope[owner.index()]
            && state
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == owner && piece.kind != PieceKind::Monster)
                .count()
                <= 2
        {
            state.status = GameStatus::Won {
                winner: owner,
                reason: WinReason::CardEffect(crate::HOPE.into()),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::interactions::*;
    use super::*;
    #[test]
    fn hope_wins_at_own_turn_start_and_never_on_acquisition_or_opponent_start() {
        let engine = Engine::bundled_latest().unwrap();
        let state = engine
            .grant_card(&base(), 1, crate::HOPE, Player::White)
            .unwrap()
            .state;
        assert_eq!(state.status, GameStatus::Ongoing);
        let state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert_eq!(state.status, GameStatus::Ongoing);
        let state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        assert_eq!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect(crate::HOPE.into())
            }
        );
    }
    #[test]
    fn hope_counts_allied_pieces_and_waits_for_standard_end_draft_unlock() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("b2", Player::White, PieceKind::Pawn),
                ("c2", Player::White, PieceKind::Pawn),
            ],
            Player::Black,
        );
        let mut state = engine
            .grant_card(&state, 1, crate::HOPE, Player::White)
            .unwrap()
            .state;
        engine.settle_card_turn_start(&mut state, Player::White);
        assert_eq!(state.status, GameStatus::Ongoing);
        state.board.set("b2".parse().unwrap(), None);
        engine.settle_card_turn_start(&mut state, Player::White);
        assert!(matches!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));

        let mut state = base();
        state.phase = GamePhase::Draft(DraftState {
            phase: DraftPhase::End,
            offers: vec![crate::HOPE.into()],
            resume_turn: Player::White,
        });
        state = engine
            .apply(&state, Action::DraftPick { offer_index: 0 })
            .unwrap()
            .state;
        assert!(!state.card_runtime.october.hope[0]);
        state = engine
            .apply(&state, Action::DraftPick { offer_index: 0 })
            .unwrap()
            .state;
        assert_eq!(state.status, GameStatus::Ongoing);
        assert!(!state.card_runtime.october.hope[0]);
        state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert_eq!(state.status, GameStatus::Ongoing);
        state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        assert!(state.card_runtime.october.hope[0]);
        assert!(matches!(
            state.status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));
    }
}

use super::*;
pub(super) const ID: &str = crate::ROOK_LIFT;
pub(super) fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.rook_lift[owner.index()] = true;
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let rt = &state.card_runtime;
    let i = player.index();
    let count = |owner, kind| {
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .count()
    };
    let own = |kind| count(player, kind);
    if let Some(result) = enabled(rt.rook_lift[i]) {
        return result;
    }
    require(own(Rook) > 0, "아군 룩이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::interactions::*;
    use super::*;
    #[test]
    fn rook_lift_requires_non_capture_and_caps_chain_at_three_retained_moves() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("d2", Player::White, PieceKind::King),
                ("e7", Player::Black, PieceKind::King),
                ("b1", Player::White, PieceKind::Rook),
            ],
            Player::White,
        );
        let mut state = engine
            .grant_card(&state, 1, ROOK_LIFT, Player::White)
            .unwrap()
            .state;
        for (index, (from, to)) in [("b1", "a1"), ("a1", "a8"), ("a8", "h8"), ("h8", "h1")]
            .into_iter()
            .enumerate()
        {
            state = engine.apply(&state, movement(from, to)).unwrap().state;
            assert_eq!(
                state.turn,
                if index < 3 {
                    Player::White
                } else {
                    Player::Black
                }
            );
        }
        assert!(state.card_runtime.october.rook_lift_chain.is_none());
        let mut capture = position(
            &[
                ("d2", Player::White, PieceKind::King),
                ("e7", Player::Black, PieceKind::King),
                ("b1", Player::White, PieceKind::Rook),
                ("a1", Player::Black, PieceKind::Bishop),
            ],
            Player::White,
        );
        capture = engine
            .grant_card(&capture, 1, ROOK_LIFT, Player::White)
            .unwrap()
            .state;
        capture = engine.apply(&capture, movement("b1", "a1")).unwrap().state;
        assert_eq!(capture.turn, Player::Black);
    }
}

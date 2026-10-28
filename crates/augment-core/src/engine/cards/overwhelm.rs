use super::*;
pub(super) const ID: &str = crate::OVERWHELM;
pub(super) fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.overwhelm[owner.index()] = true;
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let rt = &state.card_runtime;
    let i = player.index();
    if let Some(result) = enabled(rt.overwhelm[i]) {
        return result;
    }
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::interactions::*;
    use super::*;
    #[test]
    fn latest_overwhelm_blocks_queen_capture_but_allows_king_capture() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("c4", Player::White, PieceKind::Queen),
                ("d4", Player::Black, PieceKind::King),
            ],
            Player::Black,
        );
        let state = engine
            .grant_card(&state, 1, OVERWHELM, Player::White)
            .unwrap()
            .state;
        assert!(!has_move(&engine, &state, "d4", "c4"));
        let state = position(
            &[
                ("c4", Player::White, PieceKind::King),
                ("d4", Player::Black, PieceKind::King),
            ],
            Player::Black,
        );
        let state = engine
            .grant_card(&state, 1, OVERWHELM, Player::White)
            .unwrap()
            .state;
        assert!(has_move(&engine, &state, "d4", "c4"));
        assert!(matches!(
            engine
                .apply(&state, movement("d4", "c4"))
                .unwrap()
                .state
                .status,
            GameStatus::Won {
                winner: Player::Black,
                ..
            }
        ));
    }
}

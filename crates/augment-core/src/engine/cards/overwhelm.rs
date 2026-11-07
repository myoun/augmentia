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

    /// The November expansion widened Overwhelm: every enemy minor piece is now
    /// barred from capturing our queen, while majors and the king still may.
    #[test]
    fn latest_overwhelm_blocks_minor_captures_of_the_queen_but_allows_majors() {
        let engine = Engine::bundled_latest().unwrap();
        // A knight (minor) may not take the queen.
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("c4", Player::White, PieceKind::Queen),
                ("d6", Player::Black, PieceKind::Knight),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::Black,
        );
        let state = engine
            .grant_card(&state, 1, OVERWHELM, Player::White)
            .unwrap()
            .state;
        assert!(
            !has_move(&engine, &state, "d6", "c4"),
            "minor capture blocked"
        );

        // A rook (major) still may.
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("c4", Player::White, PieceKind::Queen),
                ("d4", Player::Black, PieceKind::Rook),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::Black,
        );
        let state = engine
            .grant_card(&state, 1, OVERWHELM, Player::White)
            .unwrap()
            .state;
        assert!(
            has_move(&engine, &state, "d4", "c4"),
            "major capture allowed"
        );

        // The enemy king is not a minor piece either, so it may still capture.
        let state = position(
            &[
                ("c4", Player::White, PieceKind::Queen),
                ("a1", Player::White, PieceKind::King),
                ("d4", Player::Black, PieceKind::King),
            ],
            Player::Black,
        );
        let state = engine
            .grant_card(&state, 1, OVERWHELM, Player::White)
            .unwrap()
            .state;
        assert!(
            has_move(&engine, &state, "d4", "c4"),
            "king capture allowed"
        );
    }
}

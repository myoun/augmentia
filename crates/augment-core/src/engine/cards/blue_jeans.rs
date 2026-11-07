use super::*;
pub(super) const ID: &str = crate::BLUE_JEANS;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let id = ID;
    let targets: Vec<_> = state
        .board
        .iter()
        .filter(|(_, p)| crate::latest::latest_major_kind(p.kind))
        .collect();
    for (square, piece) in targets {
        engine.delete_piece_for_card(state, square, piece, owner, id, events);
    }
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    _player: Player,
    context: AcquisitionContext,
) -> CardEligibility {
    return require(
        context == AcquisitionContext::Draft
            || state
                .board
                .iter()
                .any(|(_, p)| crate::latest::latest_major_kind(p.kind)),
        "추방할 메이저 피스가 필요합니다.",
    );
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::acquisition::*;
    use super::*;

    /// Blue Jeans is upstream-disabled from the 2026-11-02 snapshot, so the
    /// current profile rejects it even though the behavior below is retained for
    /// the historical rulesets.
    #[test]
    fn blue_jeans_is_rejected_in_the_current_profile() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Rook),
        ]);
        let rejected = engine.grant_card(&before, 1, crate::BLUE_JEANS, Player::White);
        assert!(
            rejected.is_err(),
            "an upstream-disabled card must not be granted"
        );
    }

    /// The acquisition effect itself still removes both sides' current majors
    /// without recording captures, which the historical snapshots rely on.
    #[test]
    fn blue_jeans_removal_effect_drops_both_sides_majors_without_capture_credit() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Rook),
            ("f5", Player::Black, PieceKind::Amazon),
            ("b3", Player::White, PieceKind::Octopus),
            ("c3", Player::White, PieceKind::Wizard),
            ("c4", Player::Black, PieceKind::Recruiter),
            ("f4", Player::Black, PieceKind::Trickster),
        ]);
        let mut events = Vec::new();
        acquire(&engine, &mut before, Player::White, &mut events).unwrap();
        for square in ["d4", "f5", "b3"] {
            assert!(before.board.get(square.parse().unwrap()).is_none());
        }
        for square in ["a1", "h8", "c3", "c4", "f4"] {
            assert!(before.board.get(square.parse().unwrap()).is_some());
        }
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Event::PieceCaptured { .. })),
            "the removal must not count as a capture"
        );
    }
}

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
    #[test]
    fn blue_jeans_removes_both_sides_current_majors_without_capture_credit() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Rook),
            ("f5", Player::Black, PieceKind::Amazon),
            ("b3", Player::White, PieceKind::Octopus),
            ("c3", Player::White, PieceKind::Wizard),
            ("c4", Player::Black, PieceKind::Recruiter),
            ("f4", Player::Black, PieceKind::Trickster),
        ]);
        let result = engine
            .grant_card(&before, 1, crate::BLUE_JEANS, Player::White)
            .unwrap();
        for square in ["d4", "f5", "b3"] {
            assert!(result.state.board.get(square.parse().unwrap()).is_none());
        }
        for square in ["a1", "h8", "c3", "c4", "f4"] {
            assert!(result.state.board.get(square.parse().unwrap()).is_some());
        }
        assert!(
            !result
                .events
                .iter()
                .any(|e| matches!(e, Event::PieceCaptured { .. }))
        );
        assert_eq!(result.state.cards[0].charges, 0);
    }
}

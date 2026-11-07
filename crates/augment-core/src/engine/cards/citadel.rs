use super::*;
pub(super) const ID: &str = crate::CITADEL;

/// The kingside bishop column: f-file on an 8-wide board, which is column index 5.
const KINGSIDE_BISHOP_FILE: u8 = 5;

/// Citadel converts the owner's kingside bishop into a rook on acquisition.
///
/// Upstream picks the home-row bishop at the fixed kingside column and rewrites
/// its type in place (main bundle 24364-24372), so identity and status survive.
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let home_rank = if owner == Player::White { 0 } else { 7 };
    let Some(square) = Square::new(KINGSIDE_BISHOP_FILE, home_rank) else {
        return Ok(());
    };
    let Some(piece) = state.board.get(square) else {
        return Ok(());
    };
    if piece.owner != owner || piece.kind != PieceKind::Bishop {
        return Ok(());
    }
    engine.transform_current_piece(state, square, PieceKind::Rook, true)?;
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    context: AcquisitionContext,
) -> CardEligibility {
    if context == AcquisitionContext::Draft {
        return CardEligibility::Eligible;
    }
    let home_rank = if owner == Player::White { 0 } else { 7 };
    let available = Square::new(KINGSIDE_BISHOP_FILE, home_rank)
        .and_then(|square| state.board.get(square))
        .is_some_and(|piece| piece.owner == owner && piece.kind == PieceKind::Bishop);
    require(available, "킹사이드 아군 비숍이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    #[test]
    fn citadel_turns_the_kingside_bishop_into_a_rook() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(&[
            ("f1", Player::White, PieceKind::Bishop),
            ("c1", Player::White, PieceKind::Bishop),
        ]);
        let target = before.board.get("f1".parse().unwrap()).unwrap();
        let other = before.board.get("c1".parse().unwrap()).unwrap();
        let after = engine
            .grant_card(&before, 1, crate::CITADEL, Player::White)
            .unwrap()
            .state;
        let converted = after.board.get("f1".parse().unwrap()).unwrap();
        assert_eq!(converted.kind, PieceKind::Rook);
        assert_eq!(converted.id, target.id, "identity is preserved");
        // The queenside bishop is untouched.
        assert_eq!(after.board.get("c1".parse().unwrap()).unwrap().id, other.id);
        assert_eq!(
            after.board.get("c1".parse().unwrap()).unwrap().kind,
            PieceKind::Bishop
        );
    }

    #[test]
    fn citadel_is_a_no_op_without_a_kingside_bishop() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(&[("c1", Player::White, PieceKind::Bishop)]);
        let after = engine
            .grant_card(&before, 1, crate::CITADEL, Player::White)
            .unwrap()
            .state;
        assert_eq!(
            after.board.get("c1".parse().unwrap()).unwrap().kind,
            PieceKind::Bishop
        );
        assert!(after.board.get("f1".parse().unwrap()).is_none());
    }
}

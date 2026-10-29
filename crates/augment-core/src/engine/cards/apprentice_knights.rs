use super::*;
pub(super) const ID: &str = crate::APPRENTICE_KNIGHTS;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let _ = (&engine, &events);
    let diagonal = state.rule_runtime.enabled.contains(crate::DIAGONAL_CHESS);
    let origins = if diagonal {
        if owner == Player::White {
            ["b4", "d2"]
        } else {
            ["g5", "e7"]
        }
    } else if owner == Player::White {
        ["b2", "g2"]
    } else {
        ["b7", "g7"]
    };
    let targets: Vec<_> = state
        .board
        .iter()
        .filter(|(s, p)| {
            p.owner == owner
                && p.kind == PieceKind::Pawn
                && origins.iter().any(|o| {
                    state
                        .card_runtime
                        .piece_origins
                        .get(&p.id)
                        .copied()
                        .unwrap_or(*s)
                        == o.parse::<Square>().unwrap()
                })
        })
        .map(|(s, _)| s)
        .collect();
    for square in targets {
        engine.opening_transform(state, square, PieceKind::Squire, true, true)?;
    }
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    let origins = if owner == Player::White {
        ["b2", "g2"]
    } else {
        ["b7", "g7"]
    };
    require(
        state.board.iter().any(|(s, p)| {
            p.owner == owner
                && p.kind == PieceKind::Pawn
                && origins.iter().any(|o| {
                    state
                        .card_runtime
                        .piece_origins
                        .get(&p.id)
                        .copied()
                        .unwrap_or(s)
                        == o.parse::<Square>().unwrap()
                })
        }),
        "원래 b 또는 g 파일의 아군 폰이 필요합니다.",
    )
}

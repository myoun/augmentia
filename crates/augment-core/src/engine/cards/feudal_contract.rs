use super::*;
pub(super) const ID: &str = crate::FEUDAL_CONTRACT;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn pawn_family(piece: Piece) -> bool {
    matches!(
        piece.kind,
        PieceKind::Pawn
            | PieceKind::BerolinaPawn
            | PieceKind::Fanatic
            | PieceKind::Squire
            | PieceKind::StandardBearer
    )
}
fn guardian(piece: Piece) -> bool {
    !pawn_family(piece)
        && !matches!(
            piece.kind,
            PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop | PieceKind::Football
        )
}
fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let [a, b] = two_square_target(target)?;
    let a = state
        .board
        .get(a)
        .ok_or_else(|| EngineError::Invariant("feudal target disappeared".into()))?;
    let b = state
        .board
        .get(b)
        .ok_or_else(|| EngineError::Invariant("feudal target disappeared".into()))?;
    let (pawn, guardian) = if pawn_family(a) { (a, b) } else { (b, a) };
    state
        .card_runtime
        .feudal_contracts
        .retain(|c| c.pawn_id != pawn.id && c.guardian_id != guardian.id);
    state.card_runtime.feudal_contracts.push(FeudalContract {
        owner: actor,
        pawn_id: pawn.id,
        guardian_id: guardian.id,
    });
    Ok(())
}
fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    let pieces = web_ordered_pieces(state);
    for &(pawn_sq, pawn) in &pieces {
        if pawn.owner != card.owner || !pawn_family(pawn) {
            continue;
        }
        for &(guardian_sq, piece) in &pieces {
            if piece.owner != card.owner || !guardian(piece) {
                continue;
            }
            for pair in [[pawn_sq, guardian_sq], [guardian_sq, pawn_sq]] {
                out.push(Action::UseCard {
                    instance_id: card.instance_id,
                    target: ActionTarget::Squares(pair.to_vec()),
                });
            }
        }
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
            && state.board.iter().any(|(_, p)| {
                p.owner == owner
                    && p.kind != PieceKind::Pawn
                    && p.kind != PieceKind::Merchant
                    && !royal(state, p)
            }),
        "아군 폰과 비왕실 기물이 필요합니다.",
    )
}

use super::*;
use crate::engine::piece_effects::aligned_distance;
pub(super) const ID: &str = crate::CASTLING;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    engine.apply_free_castling(state, actor, target, events)
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    let king = web_ordered_pieces(state)
        .into_iter()
        .find(|(_, p)| p.owner == card.owner && engine.is_royal_piece(state, *p));
    if let Some((king_square, _)) = king {
        for (square, piece) in web_ordered_pieces(state) {
            if piece.owner == card.owner
                && piece.kind == PieceKind::Rook
                && aligned_distance(king_square, square).is_some_and(|(_, _, d)| d >= 3)
            {
                out.push(Action::UseCard {
                    instance_id: card.instance_id,
                    target: ActionTarget::PieceAt(square),
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
            .any(|(_, p)| p.owner == owner && (royal(state, p) || p.kind == PieceKind::Merchant))
            && state
                .board
                .iter()
                .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Rook),
        "아군 왕실 기물과 룩이 필요합니다.",
    )
}

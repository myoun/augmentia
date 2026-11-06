use super::*;
pub(super) const ID: &str = crate::BASIC_TRAINING;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn allowed(piece: Piece) -> bool {
    !matches!(
        piece.kind,
        PieceKind::Pawn
            | PieceKind::King
            | PieceKind::Queen
            | PieceKind::PrimeMinister
            | PieceKind::Jester
            | PieceKind::Guard
            | PieceKind::Amazon
            | PieceKind::Man
            | PieceKind::Idol
            | PieceKind::BabyBear
            | PieceKind::Bear
            | PieceKind::Football
    )
}
fn play(
    _: &Engine,
    state: &mut GameState,
    _: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    state.card_runtime.basic_training_piece_ids.insert(piece.id);
    Ok(())
}
fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, piece) in web_ordered_pieces(state) {
        if piece.owner == card.owner
            && allowed(piece)
            && !state
                .card_runtime
                .basic_training_piece_ids
                .contains(&piece.id)
        {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == owner && !royal(state, piece) && allowed(piece)),
        "훈련할 아군 기물이 필요합니다.",
    )
}

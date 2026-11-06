use super::*;
pub(super) const ID: &str = crate::LOYALIST;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn play(
    _: &Engine,
    state: &mut GameState,
    _: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    state.card_runtime.loyalist_piece_ids.insert(piece.id);
    Ok(())
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, piece) in web_ordered_pieces(state) {
        if piece.owner == card.owner
            && !engine.is_royal_piece(state, piece)
            && !engine.piece_has_ability(state, piece, PieceKind::Slime)
            && !matches!(
                piece.kind,
                PieceKind::Merchant
                    | PieceKind::Football
                    | PieceKind::Monster
                    | PieceKind::Colossus
                    | PieceKind::BigRook
                    | PieceKind::BigBishop
            )
            && !state.card_runtime.loyalist_piece_ids.contains(&piece.id)
        {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

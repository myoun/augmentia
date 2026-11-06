use super::*;
pub(super) const ID: &str = crate::SUICIDE_BOMBER;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn candidates(state: &GameState, owner: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| {
            p.owner == owner
                && matches!(
                    p.kind,
                    PieceKind::Pawn
                        | PieceKind::BerolinaPawn
                        | PieceKind::Fanatic
                        | PieceKind::Squire
                        | PieceKind::StandardBearer
                )
                && !state
                    .card_runtime
                    .feudal_contracts
                    .iter()
                    .any(|bond| bond.pawn_id == p.id || bond.guardian_id == p.id)
        })
        .collect()
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
    state.card_runtime.explosive_piece_ids.insert(piece.id);
    Ok(())
}
fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, _) in candidates(state, card.owner) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        });
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        !candidates(state, owner).is_empty(),
        "자폭 효과를 부여할 폰 계열 기물이 필요합니다.",
    )
}

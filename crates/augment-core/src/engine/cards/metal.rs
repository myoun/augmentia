use super::*;
pub(super) const ID: &str = crate::METAL;
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
    state.card_runtime.metalized_piece_ids.insert(piece.id);
    state.card_runtime.metal_cooldowns.remove(&piece.id);
    Ok(())
}
fn candidates(state: &GameState, owner: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(square, p)| {
            p.owner == owner
                && !royal(state, *p)
                && !state.card_runtime.metalized_piece_ids.contains(&p.id)
                && super::wanted::ranged(state, *p, *square)
        })
        .collect()
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
        "금속화할 원거리 기물이 필요합니다.",
    )
}

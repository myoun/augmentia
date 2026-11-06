use super::*;
pub(super) const ID: &str = crate::ICE_KING;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn kings(engine: &Engine, state: &GameState) -> Vec<Piece> {
    let pieces = web_ordered_pieces(state);
    [Player::White, Player::Black]
        .into_iter()
        .filter_map(|owner| {
            pieces.iter().find_map(|(_, piece)| {
                (piece.owner == owner && engine.is_royal_piece(state, *piece)).then_some(*piece)
            })
        })
        .collect()
}
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let targets = kings(engine, state);
    if targets.len() != 2 {
        return Err(EngineError::Invariant("blizzard needs both kings".into()));
    }
    let due = state.turns_taken[actor.index()].saturating_add(3);
    for piece in targets {
        state.card_runtime.ice_king.insert(piece.id, (actor, due));
    }
    Ok(())
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if kings(engine, state).len() == 2 {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

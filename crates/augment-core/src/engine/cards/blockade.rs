use super::*;
pub(super) const ID: &str = crate::BLOCKADE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn royal_square(engine: &Engine, state: &GameState, owner: Player) -> Option<Square> {
    web_ordered_pieces(state)
        .into_iter()
        .find_map(|(square, piece)| {
            (piece.owner == owner && engine.is_royal_piece(state, piece)).then_some(square)
        })
}
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let square = royal_square(engine, state, actor)
        .ok_or_else(|| EngineError::Invariant("blockade requires an allied royal".into()))?;
    let due = state.turns_taken[actor.index()].saturating_add(3);
    for (df, dr) in [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ] {
        if let Some(to) = square.offset(df, dr)
            && engine.card_landing_open(state, to)
            && !engine.d4_restricts(state, actor, false, to)
        {
            state.card_runtime.barricades.insert(to);
            state.card_runtime.blockade_walls.insert(to, (actor, due));
        }
    }
    Ok(())
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if royal_square(engine, state, card.owner).is_some() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

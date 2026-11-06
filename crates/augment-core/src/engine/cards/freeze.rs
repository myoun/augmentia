use super::*;
pub(super) const ID: &str = crate::FREEZE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn candidates(engine: &Engine, state: &GameState, actor: Player) -> Vec<(Square, Piece)> {
    let enemy = actor.opponent();
    let army = state
        .board
        .iter()
        .filter(|(_, p)| {
            p.owner == enemy && !matches!(p.kind, PieceKind::Football | PieceKind::Scarecrow)
        })
        .count();
    if army <= 4 {
        return Vec::new();
    }
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| {
            p.owner == enemy
                && !engine.is_royal_piece(state, *p)
                && !matches!(p.kind, PieceKind::Football | PieceKind::Scarecrow)
                && !state.card_runtime.frozen.contains_key(&p.id)
                && !state.rule_runtime.winter_frozen.contains(&p.id)
        })
        .collect()
}
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    if *target != ActionTarget::None {
        return Err(EngineError::Invariant("freeze takes no target".into()));
    }
    let mut pool = candidates(engine, state, actor);
    if pool.is_empty() {
        return Err(EngineError::Invariant("no freeze candidates".into()));
    }
    for ordinal in 0..pool.len().min(3) {
        let (roll, index) = state.rng_mut().web_choice_index(pool.len());
        events.push(Event::RandomSample {
            purpose: format!("card.v19.freeze.{ordinal}"),
            roll_53: roll,
        });
        let (_, piece) = pool.remove(index.expect("nonempty freeze pool"));
        state.card_runtime.frozen.insert(
            piece.id,
            FrozenStatus {
                source: actor,
                remaining_turns: 3,
            },
        );
    }
    Ok(())
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if !candidates(engine, state, state.turn).is_empty() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

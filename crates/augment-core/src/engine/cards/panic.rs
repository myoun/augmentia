use super::*;
pub(super) const ID: &str = crate::PANIC;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn candidates(engine: &Engine, state: &GameState, owner: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| {
            p.owner == owner.opponent()
                && !engine.is_royal_piece(state, *p)
                && !engine.piece_has_ability(state, *p, PieceKind::Slime)
                && !matches!(
                    p.kind,
                    PieceKind::Colossus
                        | PieceKind::BigRook
                        | PieceKind::BigBishop
                        | PieceKind::Football
                )
                && !state.card_runtime.piece_status.contains_key(&p.id)
                && !state
                    .card_runtime
                    .pending_panic
                    .iter()
                    .any(|pending| pending.piece_ids.contains(&p.id))
        })
        .collect()
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
    let first = state
        .board
        .get(a)
        .ok_or_else(|| EngineError::Invariant("panic target disappeared".into()))?;
    let second = state
        .board
        .get(b)
        .ok_or_else(|| EngineError::Invariant("panic target disappeared".into()))?;
    state.card_runtime.pending_panic.push(PendingPanic {
        owner: actor.opponent(),
        piece_ids: vec![first.id, second.id],
    });
    Ok(())
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    let targets = candidates(engine, state, card.owner);
    for (i, &(a, _)) in targets.iter().enumerate() {
        for (j, &(b, _)) in targets.iter().enumerate() {
            if i != j {
                out.push(Action::UseCard {
                    instance_id: card.instance_id,
                    target: ActionTarget::Squares(vec![a, b]),
                });
            }
        }
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    static ENGINE: std::sync::OnceLock<Engine> = std::sync::OnceLock::new();
    let engine = ENGINE.get_or_init(|| Engine::bundled_latest().expect("bundled latest ruleset"));
    require(
        candidates(engine, state, owner).len() >= 2,
        "상대 비왕실 기물 두 개가 필요합니다.",
    )
}

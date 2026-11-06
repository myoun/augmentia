use super::*;
pub(super) const ID: &str = crate::SACRIFICE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn value(state: &GameState, piece: Piece) -> Option<u16> {
    if royal(state, piece)
        || matches!(
            piece.kind,
            PieceKind::Merchant | PieceKind::Football | PieceKind::Vip
        )
    {
        return None;
    }
    Some(super::revelation::combat_value(piece.kind))
}
fn pairs(state: &GameState, owner: Player) -> Vec<(Square, Square)> {
    let pieces: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter_map(|(square, p)| {
            (p.owner == owner)
                .then(|| value(state, p).map(|v| (square, p, v)))
                .flatten()
        })
        .collect();
    let mut pairs = Vec::new();
    for (source, _, source_value) in &pieces {
        for (target, target_piece, target_value) in &pieces {
            if source != target
                && target_piece.kind != PieceKind::Barrier
                && target_value < source_value
            {
                pairs.push((*source, *target));
            }
        }
    }
    pairs
}
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let [source, destination] = two_square_target(target)?;
    let victim = state
        .board
        .get(source)
        .ok_or_else(|| EngineError::Invariant("sacrifice source disappeared".into()))?;
    let beneficiary = state
        .board
        .get(destination)
        .ok_or_else(|| EngineError::Invariant("sacrifice target disappeared".into()))?;
    state
        .card_runtime
        .remember_permanent_before_timed_protection(beneficiary.id);
    state
        .card_runtime
        .protected_piece_ids
        .insert(beneficiary.id);
    state.card_runtime.sacrifice_protection.insert(
        beneficiary.id,
        TimedPieceStatus {
            by: actor,
            remaining_turns: 3,
        },
    );
    engine.delete_piece_for_card(state, source, victim, actor, ID, events);
    Ok(())
}
fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (source, destination) in pairs(state, card.owner) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(vec![source, destination]),
        });
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    let values: Vec<_> = state
        .board
        .iter()
        .filter_map(|(_, p)| (p.owner == owner).then(|| value(state, p)).flatten())
        .collect();
    require(
        values
            .iter()
            .any(|source| values.iter().any(|target| target < source)),
        "희생할 기물과 더 약한 보호 대상이 필요합니다.",
    )
}

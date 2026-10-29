use super::*;
pub(super) const ID: &str = crate::GRAPPLER;
fn base(state: &GameState, owner: Player, p: Piece) -> bool {
    p.owner == owner && p.kind == PieceKind::Queen && !royal(state, p)
}
fn sacrifice(state: &GameState, owner: Player, p: Piece) -> bool {
    p.owner == owner
        && p.kind != PieceKind::Monster
        && crate::latest::latest_minor_kind(p.kind)
        && !royal(state, p)
}
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let [queen, minor] = two_square_target(target)?;
    let q = state
        .board
        .get(queen)
        .ok_or_else(|| EngineError::Invariant("missing grappler queen".into()))?;
    let m = state
        .board
        .get(minor)
        .ok_or_else(|| EngineError::Invariant("missing grappler sacrifice".into()))?;
    if queen == minor || !base(state, actor, q) || !sacrifice(state, actor, m) {
        return Err(EngineError::Invariant("invalid grappler selection".into()));
    }
    engine.delete_piece_for_card(state, minor, m, actor, ID, events);
    engine.transform_current_piece(state, queen, PieceKind::Grappler, true)?;
    Ok(())
}
pub(super) fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (q, _) in state
        .board
        .iter()
        .filter(|(_, p)| base(state, card.owner, *p))
    {
        for (m, _) in state
            .board
            .iter()
            .filter(|(_, p)| sacrifice(state, card.owner, *p))
        {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![q, m]),
            });
        }
    }
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    require(
        state.board.iter().any(|(_, p)| base(state, owner, p))
            && state.board.iter().any(|(_, p)| sacrifice(state, owner, p)),
        "비왕실 아군 퀸과 희생할 마이너가 필요합니다.",
    )
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

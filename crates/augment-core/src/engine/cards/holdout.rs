use super::*;
pub(super) const ID: &str = crate::HOLDOUT;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let candidates: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn && !royal(state, *p))
        .collect();
    if candidates.is_empty() {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant("no holdout pawn".into()));
    }
    let index = if candidates.len() == 1 {
        0
    } else {
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        events.push(Event::RandomSample {
            purpose: "acquire.opening.holdout.target".into(),
            roll_53: roll,
        });
        index.unwrap()
    };
    let due = rule_halfmove(state).saturating_add(56 + u32::from(state.turn != owner));
    state
        .card_runtime
        .holdout_promotions
        .insert(candidates[index].1.id, due);
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn && !royal(state, p)),
        "비왕실 아군 폰이 필요합니다.",
    )
}

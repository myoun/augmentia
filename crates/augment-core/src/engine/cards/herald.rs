use super::*;
pub(super) const ID: &str = crate::HERALD;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let targets: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Rook && !royal(state, *p))
        .collect();
    if targets.is_empty() {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant(
            "no applicable opening target".into(),
        ));
    }
    let n = if targets.len() == 1 {
        0
    } else {
        let (roll, index) = state.rng_mut().web_choice_index(targets.len());
        events.push(Event::RandomSample {
            purpose: "acquire.opening.herald.target".into(),
            roll_53: roll,
        });
        index.unwrap()
    };
    engine.opening_transform(state, targets[n].0, PieceKind::Herald, true, false)
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
            .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Rook && !royal(state, p)),
        "비왕실 아군 룩이 필요합니다.",
    )
}

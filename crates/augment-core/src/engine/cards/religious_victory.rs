use super::*;
pub(super) const ID: &str = crate::RELIGIOUS_VICTORY;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.religious_victory[owner.index()] = true;
    engine.resolve_religious_victory(state);
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let rt = &state.card_runtime;
    let i = player.index();
    let count = |owner, kind| {
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .count()
    };
    let own = |kind| count(player, kind);
    if let Some(result) = enabled(rt.religious_victory[i]) {
        return result;
    }
    require(own(Bishop) >= 2, "아군 비숍 2개가 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

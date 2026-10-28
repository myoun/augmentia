use super::*;
pub(super) const ID: &str = crate::FAST_GROWTH;
pub(super) fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let target = &ActionTarget::None;
    let actor = owner;
    let none = || {
        (target == &ActionTarget::None)
            .then_some(())
            .ok_or_else(|| EngineError::Invariant(format!("{ID} must not have a target")))
    };
    none()?;
    state.card_runtime.fast_growth[actor.index()] = true;
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let count = |owner, kind| {
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .count()
    };
    let own = |kind| count(player, kind);
    require(own(Pawn) > 0, "아군 폰이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

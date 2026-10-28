use super::*;
pub(super) const ID: &str = crate::RACING_KING;
pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    let own_royal = has_own(&|p| royal(state, p));
    require(own_royal, "효과를 받을 아군 왕실 기물이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, None, eligibility);

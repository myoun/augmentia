use super::*;
pub(super) const ID: &str = crate::MOVING;
pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let id = ID;
    denied(
        EligibilityReasonCode::UnsupportedPredicate,
        id,
        "공식 획득 조건의 런타임 매핑이 아직 검증되지 않아 획득을 허용하지 않습니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, None, eligibility);

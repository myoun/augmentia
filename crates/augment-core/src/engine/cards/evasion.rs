use super::*;
pub(super) const ID: &str = crate::EVASION;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(
    ID,
    random_status::play,
    random_status::actions,
    random_status::eligibility,
);

use super::*;
pub(super) const ID: &str = crate::D4;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, None, eligibility);
pub(super) fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}
impl Engine {
    pub(in crate::engine) fn d4_restricts(
        &self,
        state: &GameState,
        owner: Player,
        royal: bool,
        square: Square,
    ) -> bool {
        state.ruleset_id == crate::LATEST_RULESET_ID
            && state.cards.iter().any(|card| {
                card.definition_id == ID
                    && (card.owner != owner || royal)
                    && square.file() == 3
                    && square.rank() == if card.owner == Player::White { 3 } else { 4 }
            })
    }
}

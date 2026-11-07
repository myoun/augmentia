use super::*;
pub(super) const ID: &str = crate::PLATFORM;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let current_turn = state.shared_turn_count();
    state.card_runtime.platform = Some(PlatformRuleState {
        cell: None,
        spawned_at: current_turn,
        next_at: current_turn.saturating_add(5),
        triggered_piece_ids: BTreeSet::new(),
        nonce: 0,
    });
    engine.spawn_platform(state, current_turn, events);
    Ok(())
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    // Upstream `v19DraftRequirementMet` allows ANY unconditionally.
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, eligibility);

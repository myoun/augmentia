use super::*;
pub(super) const ID: &str = crate::INSIGHT;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    for (_, piece) in state.board.iter() {
        if piece.owner == actor {
            state.card_runtime.frozen.remove(&piece.id);
            state.card_runtime.disarmed.remove(&piece.id);
            state.card_runtime.severed.remove(&piece.id);
            state.card_runtime.inertia_piece_ids.remove(&piece.id);
            state.card_runtime.ice_sheet.remove(&piece.id);
            state.card_runtime.witch_trials.remove(&piece.id);
            state.card_runtime.poison_stun_turns.remove(&piece.id);
            state.card_runtime.calling_cards.remove(&piece.id);
            state.card_runtime.empty_lunchboxes.remove(&piece.id);
            let had_potion_manner = state.card_runtime.potion_manner_piece_ids.remove(&piece.id);
            if had_potion_manner || state.applied_rules.iter().any(|rule| rule == COOL_GUY) {
                state
                    .card_runtime
                    .manner_captured_last_piece_ids
                    .remove(&piece.id);
            }
            if state
                .card_runtime
                .potion_saturation_piece_ids
                .remove(&piece.id)
                || state.card_runtime.saturation_rule
            {
                state
                    .card_runtime
                    .saturation_capture_counts
                    .remove(&piece.id);
            }
        } else {
            state.card_runtime.shielded_piece_ids.remove(&piece.id);
            state.card_runtime.protected_piece_ids.remove(&piece.id);
            state.card_runtime.sacrifice_protection.remove(&piece.id);
            state.card_runtime.last_resistance.remove(&piece.id);
            state.card_runtime.coronation_protection.remove(&piece.id);
            state
                .card_runtime
                .queens_gambit_protection
                .remove(&piece.id);
            state
                .card_runtime
                .queens_gambit_previous_protection
                .remove(&piece.id);
            state.card_runtime.evasion_piece_ids.remove(&piece.id);
            state.card_runtime.parry_piece_ids.remove(&piece.id);
            state.card_runtime.loyalist_piece_ids.remove(&piece.id);
            state.card_runtime.submerged_piece_ids.remove(&piece.id);
            state.card_runtime.ghost_piece_ids.remove(&piece.id);
            state.card_runtime.poisoned_piece_ids.remove(&piece.id);
            state.card_runtime.explosive_piece_ids.remove(&piece.id);
            state.card_runtime.chimera_piece_ids.remove(&piece.id);
            state.card_runtime.chameleon_piece_ids.remove(&piece.id);
            state
                .card_runtime
                .basic_training_piece_ids
                .remove(&piece.id);
            state
                .card_runtime
                .potion_basic_training_piece_ids
                .remove(&piece.id);
            state.card_runtime.trojan_piece_ids.remove(&piece.id);
            state.card_runtime.stealth_piece_ids.remove(&piece.id);
            state.card_runtime.imperial_moves.remove(&piece.id);
            state.card_runtime.quantum_positions.remove(&piece.id);
            state.card_runtime.stake_statuses.remove(&piece.id);
        }
    }
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    _state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let push_none = |actions: &mut Vec<Action>| {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        })
    };
    push_none(actions);
}

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

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

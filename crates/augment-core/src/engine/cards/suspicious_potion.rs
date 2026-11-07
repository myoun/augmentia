use super::*;
pub(super) const ID: &str = crate::SUSPICIOUS_POTION;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    let royal = engine.is_royal_piece(state, piece);
    let ranged = matches!(
        piece.kind,
        PieceKind::MagicGirl | PieceKind::Berserker | PieceKind::Trickster
    ) || is_ranged_piece(piece.kind);
    // Keep this in the same order as SUSPICIOUS_POTION_EFFECTS in the 0818 client;
    // web_choice_index therefore consumes the same roll for the same target.
    let effects = suspicious_potion_effects(royal, piece.kind, ranged);
    let (roll, index) = state.rng_mut().web_choice_index(effects.len());
    events.push(Event::RandomSample {
        purpose: "suspicious-potion".to_owned(),
        roll_53: roll,
    });
    match effects[index.unwrap_or(0)] {
        "sacrifice-protection" => {
            if state.ruleset_id == crate::LATEST_RULESET_ID {
                state
                    .card_runtime
                    .remember_permanent_before_timed_protection(piece.id);
            }
            state.card_runtime.protected_piece_ids.insert(piece.id);
            state.card_runtime.sacrifice_protection.insert(
                piece.id,
                TimedPieceStatus {
                    by: piece.owner,
                    remaining_turns: 3,
                },
            );
        }
        "last-resistance" => {
            if state.ruleset_id == crate::LATEST_RULESET_ID {
                state
                    .card_runtime
                    .remember_permanent_before_timed_protection(piece.id);
            }
            state.card_runtime.protected_piece_ids.insert(piece.id);
            state.card_runtime.last_resistance.insert(
                piece.id,
                TimedPieceStatus {
                    by: piece.owner,
                    remaining_turns: 3,
                },
            );
        }
        "coronation-protection" => {
            if state.ruleset_id == crate::LATEST_RULESET_ID {
                state
                    .card_runtime
                    .remember_permanent_before_timed_protection(piece.id);
            }
            state.card_runtime.protected_piece_ids.insert(piece.id);
            state
                .card_runtime
                .coronation_protection
                .insert(piece.id, state.turns_taken[piece.owner.index()]);
        }
        "shield" => {
            state.card_runtime.shielded_piece_ids.insert(piece.id);
        }
        "evasion" => {
            state.card_runtime.evasion_piece_ids.insert(piece.id);
        }
        "parry" => {
            state.card_runtime.parry_piece_ids.insert(piece.id);
        }
        "basic-training" => {
            state.card_runtime.basic_training_piece_ids.insert(piece.id);
            state
                .card_runtime
                .potion_basic_training_piece_ids
                .insert(piece.id);
        }
        "stealth" => {
            state.card_runtime.stealth_piece_ids.insert(piece.id);
        }
        "loyalist" => {
            state.card_runtime.loyalist_piece_ids.insert(piece.id);
        }
        "ghost" => {
            state.card_runtime.ghost_piece_ids.insert(piece.id);
        }
        "chameleon" => {
            state.card_runtime.chameleon_piece_ids.insert(piece.id);
        }
        "submerge" => {
            state.card_runtime.submerged_piece_ids.insert(piece.id);
        }
        "freeze" => {
            state.card_runtime.frozen.insert(
                piece.id,
                FrozenStatus {
                    source: actor.opponent(),
                    remaining_turns: 3,
                },
            );
        }
        "chimera" => {
            state.card_runtime.chimera_piece_ids.insert(piece.id);
        }
        "witch-trial" => {
            state.card_runtime.witch_trials.insert(
                piece.id,
                TimedPieceStatus {
                    by: piece.owner.opponent(),
                    remaining_turns: 3,
                },
            );
        }
        "stake" => {
            state.card_runtime.stake_statuses.insert(
                piece.id,
                TimedPieceStatus {
                    by: piece.owner,
                    remaining_turns: 4,
                },
            );
        }
        "calling-card" => {
            state
                .card_runtime
                .calling_cards
                .insert(piece.id, piece.owner.opponent());
        }
        "empty-lunchbox" => {
            state.card_runtime.empty_lunchboxes.insert(
                piece.id,
                EmptyLunchboxStatus {
                    by: piece.owner.opponent(),
                    deadline_turn: state.turns_taken[piece.owner.index()].saturating_add(3),
                },
            );
        }
        "poison-stun" => {
            state
                .card_runtime
                .poison_stun_turns
                .entry(piece.id)
                .and_modify(|remaining| *remaining = (*remaining).max(3))
                .or_insert(3);
        }
        "disarm" => {
            state.card_runtime.disarmed.insert(piece.id, 1);
        }
        "manner-no-capture" => {
            state.card_runtime.potion_manner_piece_ids.insert(piece.id);
            state
                .card_runtime
                .manner_captured_last_piece_ids
                .insert(piece.id);
        }
        "saturation-no-capture" => {
            state
                .card_runtime
                .potion_saturation_piece_ids
                .insert(piece.id);
            state
                .card_runtime
                .saturation_capture_counts
                .insert(piece.id, 3);
        }
        "explosive" => {
            state.card_runtime.explosive_piece_ids.insert(piece.id);
        }
        "inertia" => {
            state.card_runtime.inertia_piece_ids.insert(piece.id);
        }
        "poisoned-pawn" => {
            state.card_runtime.poisoned_piece_ids.insert(piece.id);
        }
        "queens-gambit-protection" => {
            if state.ruleset_id == crate::LATEST_RULESET_ID {
                super::queens_gambit::apply_protection(&mut state.card_runtime, piece.id);
            } else {
                state.card_runtime.protected_piece_ids.insert(piece.id);
                state.card_runtime.queens_gambit_protection.insert(piece.id);
            }
        }
        "trojan-horse" => {
            state.card_runtime.trojan_piece_ids.insert(piece.id);
        }
        "severance" => {
            state.card_runtime.severed.insert(
                piece.id,
                TimedPieceStatus {
                    by: piece.owner.opponent(),
                    remaining_turns: 2,
                },
            );
        }
        _ => unreachable!(),
    }
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let push_piece = |actions: &mut Vec<Action>, square| {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        })
    };
    for (square, piece) in state.board.iter() {
        if !state.card_runtime.piece_status.contains_key(&piece.id)
            && !matches!(
                piece.kind,
                PieceKind::Monster | PieceKind::Scarecrow | PieceKind::Football
            )
        {
            push_piece(actions, square);
        }
    }
}

pub(super) fn eligibility(
    state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    require(
        state.board.iter().any(|(_, piece)| !royal(state, piece)),
        "물약을 적용할 기물이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

use super::*;
use crate::*;

impl Engine {
    pub(super) fn resolve_end_after_move(
        &self,
        state: &mut GameState,
        piece: Piece,
        movement: MoveAction,
        captured: Option<(Square, Piece)>,
        events: &mut Vec<Event>,
    ) {
        let actor = piece.owner;
        if state.card_runtime.moving[actor.index()] {
            let (last_piece, count) = state.card_runtime.moving_consecutive[actor.index()];
            if last_piece == Some(piece.id) {
                let new_count = count.saturating_add(1);
                state.card_runtime.moving_consecutive[actor.index()] = (Some(piece.id), new_count);
                if new_count >= 5 {
                    state.card_runtime.evasion_piece_ids.insert(piece.id);
                }
            } else {
                state.card_runtime.moving_consecutive[actor.index()] = (Some(piece.id), 1);
            }
        }

        if captured.is_some() && state.card_runtime.frenzy_pawns.contains(&piece.id) {
            state.phase = GamePhase::ForcedMove {
                piece_id: piece.id,
                source: ForcedMoveSource::Effect(FRENZY.to_owned()),
                continuation: ForcedMoveContinuation::FinishTurn,
            };
        }

        if state.card_runtime.charge_pawns.remove(&piece.id) && movement.promotion.is_none() {
            if let Some((sq, p)) = state.board.find_by_id(piece.id) {
                self.remove_piece_for_card(state, sq, p, actor, CHARGE, events);
            }
        }

        if state.card_runtime.desperado_pieces.remove(&piece.id) {
            if let Some((sq, p)) = state.board.find_by_id(piece.id) {
                self.remove_piece_for_card(state, sq, p, actor, DESPERADO, events);
            }
        }

        if state.card_runtime.coronation[actor.index()]
            && movement.promotion == Some(PieceKind::Queen)
        {
            if state.ruleset_id == crate::LATEST_RULESET_ID {
                state
                    .card_runtime
                    .remember_permanent_before_timed_protection(piece.id);
            }
            state.card_runtime.protected_piece_ids.insert(piece.id);
            state
                .card_runtime
                .coronation_protection
                .insert(piece.id, state.turns_taken[actor.index()]);
        }

        if state.card_runtime.underpromotion[actor.index()]
            && matches!(
                movement.promotion,
                Some(PieceKind::Knight | PieceKind::Bishop | PieceKind::Rook)
            )
        {
            state.phase = GamePhase::ForcedMove {
                piece_id: piece.id,
                source: ForcedMoveSource::Effect(UNDERPROMOTION.to_owned()),
                continuation: ForcedMoveContinuation::FinishTurn,
            };
        }

        if let Some(spy_owner) = state.card_runtime.spies.remove(&piece.id)
            && movement.promotion.is_some()
        {
            if let Some((sq, mut promoted)) = state.board.find_by_id(piece.id) {
                promoted.owner = spy_owner;
                state.board.set(sq, Some(promoted));
            }
        }

        if state.card_runtime.racing_king[actor.index()]
            && self.is_royal_piece(state, piece)
            && movement.to.rank() == if actor == Player::White { 7 } else { 0 }
        {
            state.status = GameStatus::Won {
                winner: actor,
                reason: WinReason::CardEffect("racing-king".to_owned()),
            };
        }

        // Vanish removes one piece per side at each turn end. The current contract
        // differs from the old single-side rule, so the card owns the settlement.
        self.settle_vanish(state, actor, events);
        self.resolve_missing_king(state, events);

        for player in [Player::White, Player::Black] {
            if state.ruleset_id != crate::LATEST_RULESET_ID
                && let Some(due_ply) = state.card_runtime.prophecy[player.index()]
            {
                if captured.is_some() {
                    state.card_runtime.prophecy[player.index()] = None;
                } else if state.ply >= due_ply {
                    state.status = GameStatus::Won {
                        winner: player,
                        reason: WinReason::CardEffect("prophecy".to_owned()),
                    };
                }
            }
        }

        if state.card_runtime.bina_mate[actor.index()] {
            if let Some((king_sq, _)) = state
                .board
                .iter()
                .find(|(_, p)| p.owner == actor.opponent() && self.is_royal_piece(state, *p))
            {
                let mut attackers = 0;
                for (_, friendly) in state.board.iter().filter(|(_, p)| p.owner == actor) {
                    if self
                        .generate_piece_move_actions(state, friendly.id)
                        .iter()
                        .any(|a| matches!(a, Action::Move(m) if m.to == king_sq))
                    {
                        attackers += 1;
                    }
                }
                if attackers >= 2 {
                    state.status = GameStatus::Won {
                        winner: actor,
                        reason: WinReason::CardEffect("bina-mate".to_owned()),
                    };
                }
            }
        }
    }

    pub(super) fn tick_end_statuses_after_turn(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        state.card_runtime.zugzwang[actor.index()] = false;
        state.card_runtime.reversal[actor.index()] = false;
        if state.ruleset_id == crate::LATEST_RULESET_ID {
            state.card_runtime.horse_riding[actor.index()] = false;
        }

        let mut remaining_bribes = Vec::new();
        for bribe in std::mem::take(&mut state.card_runtime.bribe_reverts) {
            if state.turns_taken[actor.index()] >= bribe.due_turn {
                if let Some((sq, mut piece)) = state.board.find_by_id(bribe.piece_id) {
                    self.clear_queens_gambit_protection(state, piece.id);
                    piece.kind = PieceKind::Knight;
                    state.board.set(sq, Some(piece));
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("bribe-revert:{}:{}", piece.id, sq),
                    });
                }
            } else {
                remaining_bribes.push(bribe);
            }
        }
        state.card_runtime.bribe_reverts = remaining_bribes;

        let mut remaining_necro = Vec::new();
        for necro in std::mem::take(&mut state.card_runtime.necromancy_reverts) {
            if state.ply >= necro.due_ply {
                if let Some((sq, mut piece)) = state.board.find_by_id(necro.piece_id) {
                    self.clear_queens_gambit_protection(state, piece.id);
                    piece.kind = PieceKind::Pawn;
                    state.board.set(sq, Some(piece));
                }
            } else {
                remaining_necro.push(necro);
            }
        }
        state.card_runtime.necromancy_reverts = remaining_necro;

        let mut remaining_vip = Vec::new();
        for vip in std::mem::take(&mut state.card_runtime.vip_pending) {
            if state.ply >= vip.due_ply {
                if let Some((sq, mut piece)) = state.board.find_by_id(vip.piece_id) {
                    self.clear_queens_gambit_protection(state, piece.id);
                    piece.kind = PieceKind::Vip;
                    state.board.set(sq, Some(piece));
                }
            } else {
                remaining_vip.push(vip);
            }
        }
        state.card_runtime.vip_pending = remaining_vip;

        let due_stakes = state
            .card_runtime
            .stake_statuses
            .iter_mut()
            .filter_map(|(piece_id, status)| {
                if state
                    .board
                    .find_by_id(*piece_id)
                    .is_some_and(|(_, piece)| piece.owner == actor)
                {
                    status.remaining_turns = status.remaining_turns.saturating_sub(1);
                    (status.remaining_turns == 0).then_some(*piece_id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for piece_id in due_stakes {
            state.card_runtime.stake_statuses.remove(&piece_id);
            state.card_runtime.shielded_piece_ids.insert(piece_id);
        }

        // ICBM fires at the owner's next turn start, which the delayed turn-start
        // settlement handles; it must not fire at the end of the caster's turn.

        if let Some(due_ply) = state.card_runtime.ultimatum {
            if state.ply >= due_ply {
                let unmoved = state
                    .board
                    .iter()
                    .filter(|(_, p)| {
                        !self.is_royal_piece(state, *p)
                            && !state.card_runtime.moved_piece_ids.contains(&p.id)
                    })
                    .collect::<Vec<_>>();
                for (sq, victim) in unmoved {
                    self.remove_piece_for_card(state, sq, victim, actor, ULTIMATUM, events);
                }
                state.card_runtime.ultimatum = None;
            }
        }
    }
}

pub(in crate::engine) fn replay_move_matches_state(
    state: &GameState,
    replay: &MoveReplayState,
) -> bool {
    replay
        .cells
        .iter()
        .all(|cell| state.board.get(cell.square) == cell.after)
        && replay_piece_runtime_matches(&state.card_runtime, replay)
}

fn replay_piece_ids(replay: &MoveReplayState) -> Vec<u32> {
    let mut ids = replay
        .cells
        .iter()
        .flat_map(|cell| [cell.before, cell.after])
        .flatten()
        .map(|piece| piece.id)
        .collect::<Vec<_>>();
    ids.extend(
        replay
            .runtime_before
            .vigilance_protection
            .keys()
            .chain(replay.runtime_after.vigilance_protection.keys())
            .copied()
            .filter(|id| {
                replay.runtime_before.vigilance_protection.get(id)
                    != replay.runtime_after.vigilance_protection.get(id)
            }),
    );
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn replay_piece_runtime_matches(current: &CardRuntimeState, replay: &MoveReplayState) -> bool {
    let after = &*replay.runtime_after;
    macro_rules! sets_match {
        ($id:expr; $($field:ident),+ $(,)?) => {
            $(if current.$field.contains(&$id) != after.$field.contains(&$id) { return false; })+
        };
    }
    macro_rules! maps_match {
        ($id:expr; $($field:ident),+ $(,)?) => {
            $(if current.$field.get(&$id) != after.$field.get(&$id) { return false; })+
        };
    }
    for id in replay_piece_ids(replay) {
        if current.effects.grudged_piece_ids.contains(&id)
            != after.effects.grudged_piece_ids.contains(&id)
            || current.effects.grudge_losses.get(&id) != after.effects.grudge_losses.get(&id)
        {
            return false;
        }
        if current.effects.bound_piece_ids.contains(&id)
            != after.effects.bound_piece_ids.contains(&id)
        {
            return false;
        }
        sets_match!(id;
            moved_piece_ids, castled_piece_ids, parry_piece_ids, shielded_piece_ids, protected_piece_ids,
            idol_rest_piece_ids, loyalist_piece_ids, royal_piece_ids,
            basic_training_piece_ids, promotion_rush_piece_ids, promotion_rush_capture_cleared, evacuation_no_capture, twin_swap_pending, evasion_piece_ids,
            ghost_piece_ids, inertia_piece_ids, chameleon_piece_ids, chimera_piece_ids,
            poisoned_piece_ids, queens_gambit_protection, queens_gambit_previous_protection, frontline_piece_ids, counterattack_ready, counterattack_locked, potion_basic_training_piece_ids,
            potion_manner_piece_ids, manner_captured_last_piece_ids,
            potion_saturation_piece_ids, explosive_piece_ids, trojan_piece_ids,
            stealth_piece_ids, submerged_piece_ids, wanted_piece_ids, charge_pawns, desperado_pieces,
            field_promoted_piece_ids, locust_swarm_piece_ids, metalized_piece_ids,
            recurrence_piece_ids, nullification_piece_ids
        );
        maps_match!(id;
            fresh_no_capture_until, disarmed, frozen, empty_lunchboxes,
            herald_jump_locked_until, windmill_modes, log_statuses, wizard_mana,
            reaper_souls, idol_used_turn, piece_origins, piece_status, calling_cards,
            holdout_promotions, merchant_gold, ice_sheet, severed, imperial_moves,
            sacrifice_protection, last_resistance, coronation_protection, thief_turn_paths, large_missing_cells, ice_king, vigilance_protection, scarecrow_reservations,
            poison_stun_turns, saturation_capture_counts, quantum_positions,
            witch_trials, baby_bear_grow_turns, bear_retaliations,
            trickster_move_kinds, stake_statuses, bunkers, spies, metal_cooldowns
        );
    }
    true
}

pub(in crate::engine) fn restore_replay_piece_runtime(
    current: &mut CardRuntimeState,
    replay: &MoveReplayState,
) {
    restore_piece_runtime(current, &replay.runtime_before, &replay_piece_ids(replay));
}

pub(super) fn restore_piece_runtime(
    current: &mut CardRuntimeState,
    before: &CardRuntimeState,
    ids: &[u32],
) {
    macro_rules! restore_sets {
        ($id:expr; $($field:ident),+ $(,)?) => {
            $(if before.$field.contains(&$id) {
                current.$field.insert($id);
            } else {
                current.$field.remove(&$id);
            })+
        };
    }
    macro_rules! restore_maps {
        ($id:expr; $($field:ident),+ $(,)?) => {
            $(if let Some(value) = before.$field.get(&$id).cloned() {
                current.$field.insert($id, value);
            } else {
                current.$field.remove(&$id);
            })+
        };
    }
    for id in ids.iter().copied() {
        if before.effects.grudged_piece_ids.contains(&id) {
            current.effects.grudged_piece_ids.insert(id);
        } else {
            current.effects.grudged_piece_ids.remove(&id);
        }
        if let Some(loss) = before.effects.grudge_losses.get(&id) {
            current.effects.grudge_losses.insert(id, *loss);
        } else {
            current.effects.grudge_losses.remove(&id);
        }
        if before.effects.bound_piece_ids.contains(&id) {
            current.effects.bound_piece_ids.insert(id);
        } else {
            current.effects.bound_piece_ids.remove(&id);
        }
        restore_sets!(id;
            moved_piece_ids, castled_piece_ids, parry_piece_ids, shielded_piece_ids, protected_piece_ids,
            idol_rest_piece_ids, loyalist_piece_ids, royal_piece_ids,
            basic_training_piece_ids, promotion_rush_piece_ids, promotion_rush_capture_cleared, evacuation_no_capture, twin_swap_pending, evasion_piece_ids,
            ghost_piece_ids, inertia_piece_ids, chameleon_piece_ids, chimera_piece_ids,
            poisoned_piece_ids, queens_gambit_protection, queens_gambit_previous_protection, frontline_piece_ids, counterattack_ready, counterattack_locked, potion_basic_training_piece_ids,
            potion_manner_piece_ids, manner_captured_last_piece_ids,
            potion_saturation_piece_ids, explosive_piece_ids, trojan_piece_ids,
            stealth_piece_ids, submerged_piece_ids, wanted_piece_ids, charge_pawns, desperado_pieces,
            field_promoted_piece_ids, locust_swarm_piece_ids, metalized_piece_ids,
            recurrence_piece_ids, nullification_piece_ids
        );
        restore_maps!(id;
            fresh_no_capture_until, disarmed, frozen, empty_lunchboxes,
            herald_jump_locked_until, windmill_modes, log_statuses, wizard_mana,
            reaper_souls, idol_used_turn, piece_origins, piece_status, calling_cards,
            holdout_promotions, merchant_gold, ice_sheet, severed, imperial_moves,
            sacrifice_protection, last_resistance, coronation_protection, thief_turn_paths, large_missing_cells, ice_king, vigilance_protection, scarecrow_reservations,
            poison_stun_turns, saturation_capture_counts, quantum_positions,
            witch_trials, baby_bear_grow_turns, bear_retaliations,
            trickster_move_kinds, stake_statuses, bunkers, spies, metal_cooldowns
        );
    }

    let touches = |id: u32| ids.binary_search(&id).is_ok();
    current
        .siren_exposure
        .retain(|(siren, target), _| !touches(*siren) && !touches(*target));
    current.siren_exposure.extend(
        before
            .siren_exposure
            .iter()
            .filter(|((siren, target), _)| touches(*siren) || touches(*target))
            .map(|(key, value)| (*key, *value)),
    );
    current
        .feudal_contracts
        .retain(|entry| !touches(entry.pawn_id) && !touches(entry.guardian_id));
    current.feudal_contracts.extend(
        before
            .feudal_contracts
            .iter()
            .filter(|entry| touches(entry.pawn_id) || touches(entry.guardian_id))
            .copied(),
    );
    current
        .twin_bonds
        .retain(|entry| !touches(entry.first_id) && !touches(entry.second_id));
    current.twin_bonds.extend(
        before
            .twin_bonds
            .iter()
            .filter(|entry| touches(entry.first_id) || touches(entry.second_id))
            .copied(),
    );
}

pub(in crate::engine) fn restore_replay_color_runtime(
    current: &mut CardRuntimeState,
    replay: &MoveReplayState,
    actor: Player,
) {
    let before = &*replay.runtime_before;
    let after = &*replay.runtime_after;
    let index = actor.index();
    macro_rules! restore_color_fields {
        ($($field:ident),+ $(,)?) => {
            $(if current.$field[index] == after.$field[index] {
                current.$field[index] = before.$field[index];
            })+
        };
    }
    restore_color_fields!(
        quantum_pending,
        switcheroo,
        reposition_ready,
        e4_active,
        roller_active,
        solidarity_active,
        moving,
        moving_consecutive,
        exhaustion_last_piece,
        exhaustion_count,
        zugzwang,
    );
    if current.pending_rule_tickets == after.pending_rule_tickets {
        current.pending_rule_tickets = before.pending_rule_tickets.clone();
    }
    if current.file_conveyors == after.file_conveyors {
        current.file_conveyors = before.file_conveyors.clone();
    }
    if current.highway_cells == after.highway_cells {
        current.highway_cells = before.highway_cells.clone();
    }
    if current.crown_pos == after.crown_pos && current.crown_holder == after.crown_holder {
        current.crown_pos = before.crown_pos;
        current.crown_holder = before.crown_holder;
    }
    if current.capture_the_flag == after.capture_the_flag {
        current.capture_the_flag = before.capture_the_flag.clone();
    }
    if current.ultimatum == after.ultimatum {
        current.ultimatum = before.ultimatum;
    }
    if current.pending_portals == after.pending_portals {
        current.pending_portals = before.pending_portals.clone();
    }
    if current.portals == after.portals {
        current.portals = before.portals;
    }
    if current.pending_lobsters == after.pending_lobsters {
        current.pending_lobsters = before.pending_lobsters.clone();
    }
    if current.undead_resurrections == after.undead_resurrections {
        current.undead_resurrections = before.undead_resurrections.clone();
    }
}

//! Historical midgame card dispatch.
use crate::engine::piece_effects::*;
use crate::engine::*;
use crate::*;
impl Engine {
    pub(in crate::engine) fn apply_middle_active_card(
        &self,
        state: &mut GameState,
        actor: Player,
        definition_id: &str,
        target: &ActionTarget,
        events: &mut Vec<Event>,
    ) -> Result<bool, EngineError> {
        let opponent = actor.opponent();
        match definition_id {
            CLEANUP => {
                let squares = squares_target(target)?.to_vec();
                for square in squares {
                    if let Some(piece) = state.board.get(square) {
                        self.delete_piece_for_card(
                            state,
                            square,
                            piece,
                            actor.opponent(),
                            CLEANUP,
                            events,
                        );
                    }
                }
            }
            FRONTLINE_RESPONSE => state.card_runtime.frontline_response[actor.index()] = true,
            RELAY => state.card_runtime.relay[actor.index()] = true,
            SUSPICIOUS_POTION => {
                let (_, piece) = piece_target(state, target)?;
                let royal = self.is_royal_piece(state, piece);
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
                                deadline_turn: state.turns_taken[piece.owner.index()]
                                    .saturating_add(3),
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
                        state.card_runtime.protected_piece_ids.insert(piece.id);
                        state.card_runtime.queens_gambit_protection.insert(piece.id);
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
            }
            BASIC_TRAINING => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.basic_training_piece_ids.insert(piece.id);
            }
            BREAKTHROUGH_ORDER => state.card_runtime.breakthrough_pawns[actor.index()] = true,
            EN_PASSANT_BANG => state.card_runtime.en_passant_frenzy[actor.index()] = true,
            PROMOTION_RUSH => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.promotion_rush_piece_ids.insert(piece.id);
                self.mark_fresh_no_capture(state, piece);
            }
            ARMISTICE => state.card_runtime.armistice_remaining_ply = 4,
            EVASION => {
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor
                            && !state.card_runtime.evasion_piece_ids.contains(&piece.id)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "evasion".to_owned(),
                    roll_53: roll,
                });
                let (_, piece) = candidates[index.ok_or_else(|| {
                    EngineError::Invariant("legal evasion has no candidate".to_owned())
                })?];
                state.card_runtime.evasion_piece_ids.insert(piece.id);
            }
            ICE_SHEET => {
                let targets = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| piece.owner == opponent && is_ranged_piece(piece.kind))
                    .collect::<Vec<_>>();
                for (_, piece) in targets {
                    state.card_runtime.ice_sheet.insert(
                        piece.id,
                        TimedPieceStatus {
                            by: actor,
                            remaining_turns: 3,
                        },
                    );
                }
            }
            INERTIA => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.inertia_piece_ids.insert(piece.id);
            }
            ROYAL_COMMAND => {
                let start = state.turns_taken[actor.index()].saturating_add(1);
                state.card_runtime.royal_command[actor.index()] = Some(RoyalCommandStatus {
                    starts_at_turn: start,
                    expires_at_turn: start.saturating_add(2),
                });
            }
            ROYAL_SHIELD => {
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor
                            && !state.card_runtime.shielded_piece_ids.contains(&piece.id)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "royal-shield".to_owned(),
                    roll_53: roll,
                });
                let (_, piece) = candidates[index.ok_or_else(|| {
                    EngineError::Invariant("legal royal shield has no candidate".to_owned())
                })?];
                state.card_runtime.shielded_piece_ids.insert(piece.id);
            }
            SEVERANCE => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.severed.insert(
                    piece.id,
                    TimedPieceStatus {
                        by: actor,
                        remaining_turns: 2,
                    },
                );
            }
            SOCIALISM => state.card_runtime.socialism[opponent.index()] = 1,
            TAUNT => state.card_runtime.taunt[opponent.index()] = 1,
            ALEKHINE_MACHINE_GUN => {
                for piece_id in self.alekhine_formation(state, actor).ok_or_else(|| {
                    EngineError::Invariant("legal Alekhine card has no formation".to_owned())
                })? {
                    state.card_runtime.shielded_piece_ids.insert(piece_id);
                }
            }
            CHAMELEON_MUTATION => {
                for square in squares_target(target)? {
                    if let Some(piece) = state.board.get(square) {
                        state.card_runtime.chameleon_piece_ids.insert(piece.id);
                    }
                }
            }
            CHIMERA => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.chimera_piece_ids.insert(piece.id);
            }
            FEUDAL_CONTRACT => {
                let [pawn_square, guardian_square] = two_square_target(target)?;
                let pawn = state
                    .board
                    .get(pawn_square)
                    .ok_or_else(|| EngineError::Invariant("feudal pawn disappeared".to_owned()))?;
                let guardian = state.board.get(guardian_square).ok_or_else(|| {
                    EngineError::Invariant("feudal guardian disappeared".to_owned())
                })?;
                state.card_runtime.feudal_contracts.retain(|contract| {
                    contract.pawn_id != pawn.id && contract.guardian_id != guardian.id
                });
                state.card_runtime.feudal_contracts.push(FeudalContract {
                    owner: actor,
                    pawn_id: pawn.id,
                    guardian_id: guardian.id,
                });
            }
            MISTAKE_CARD => state.card_runtime.mistake_card[opponent.index()] = true,
            POISONED_PAWN => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.poisoned_piece_ids.insert(piece.id);
            }
            SACRIFICE => {
                let [sacrifice_square, protected_square] = two_square_target(target)?;
                let sacrificed = state.board.get(sacrifice_square).ok_or_else(|| {
                    EngineError::Invariant("sacrifice target disappeared".to_owned())
                })?;
                let protected = state.board.get(protected_square).ok_or_else(|| {
                    EngineError::Invariant("sacrifice protection target disappeared".to_owned())
                })?;
                state.board.set(sacrifice_square, None);
                self.clear_removed_piece_statuses(state, sacrificed.id);
                events.push(Event::PieceRemoved {
                    actor: opponent,
                    piece: sacrificed,
                    at: sacrifice_square,
                    cause: SACRIFICE.to_owned(),
                });
                if state.ruleset_id == crate::LATEST_RULESET_ID {
                    state
                        .card_runtime
                        .remember_permanent_before_timed_protection(protected.id);
                }
                state.card_runtime.protected_piece_ids.insert(protected.id);
                state.card_runtime.sacrifice_protection.insert(
                    protected.id,
                    TimedPieceStatus {
                        by: actor,
                        remaining_turns: 3,
                    },
                );
            }
            SNIPE => state.card_runtime.bishop_snipe[actor.index()] = true,
            SUICIDE_BOMBER => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.explosive_piece_ids.insert(piece.id);
            }
            crate::OUTPOST => {
                let (_, piece) = piece_target(state, target)?;
                state
                    .card_runtime
                    .outpost_protected_piece_ids
                    .insert(piece.id);
            }
            crate::SYMMETRY => state.card_runtime.symmetry[actor.index()] = true,
            crate::MIRACLE => {
                // Miracle (live JS:91144-91179): every enemy piece currently capturable by a
                // friendly bishop switches owners; every friendly bishop is then sacrificed.
                let bishops = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Bishop)
                    .collect::<Vec<_>>();
                let mut capturable = Vec::new();
                for (bishop_square, bishop) in &bishops {
                    let moves = {
                        let mut candidates = Vec::new();
                        self.generate_slider_moves(
                            state,
                            *bishop_square,
                            *bishop,
                            &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
                            &mut candidates,
                        );
                        candidates
                    };
                    for action in moves {
                        if let Action::Move(move_action) = action {
                            if let Some(target_piece) = state.board.get(move_action.to) {
                                if target_piece.owner == opponent
                                    && !capturable.iter().any(|(square, _): &(Square, Piece)| {
                                        *square == move_action.to
                                    })
                                {
                                    capturable.push((move_action.to, target_piece));
                                }
                            }
                        }
                    }
                }
                for (square, piece) in capturable {
                    let mut converted = piece;
                    converted.owner = actor;
                    state.board.set(square, Some(converted));
                }
                for (square, bishop) in bishops {
                    self.delete_piece_for_card(state, square, bishop, actor, MIRACLE, events);
                }
            }
            TROJAN_HORSE => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.trojan_piece_ids.insert(piece.id);
            }
            CASTLING => self.apply_free_castling(state, actor, target, events)?,
            EMERGENCY_EVACUATION => {
                let mut entries = squares_target(target)?
                    .into_iter()
                    .filter_map(|square| state.board.get(square).map(|piece| (square, piece)))
                    .collect::<Vec<_>>();
                entries.sort_by_key(|(square, _)| {
                    if actor == Player::White {
                        square.rank()
                    } else {
                        7 - square.rank()
                    }
                });
                for (from, piece) in entries {
                    let direction = if actor == Player::White { -1 } else { 1 };
                    if let Some(to) = from.offset(0, direction)
                        && state.board.get(to).is_none()
                    {
                        state.board.set(from, None);
                        state.board.set(to, Some(piece));
                        state.card_runtime.moved_piece_ids.insert(piece.id);
                        self.mark_fresh_no_capture(state, piece);
                    }
                }
            }
            GALE => state.card_runtime.gale_due_turns[actor.index()]
                .push(state.turns_taken[actor.index()].saturating_add(3)),
            PANIC => {
                let piece_ids = squares_target(target)?
                    .into_iter()
                    .filter_map(|square| state.board.get(square).map(|piece| piece.id))
                    .collect();
                state.card_runtime.pending_panic.push(PendingPanic {
                    owner: opponent,
                    piece_ids,
                });
            }
            PAWN_STORM => {
                let direction = if actor == Player::White { 1 } else { -1 };
                let mut entries = squares_target(target)?
                    .into_iter()
                    .filter_map(|square| state.board.get(square).map(|piece| (square, piece)))
                    .collect::<Vec<_>>();
                entries.sort_by_key(|(square, _)| {
                    if actor == Player::White {
                        7 - square.rank()
                    } else {
                        square.rank()
                    }
                });
                for (from, mut piece) in entries {
                    if let Some(to) = from.offset(0, direction)
                        && state.board.get(to).is_none()
                    {
                        state.board.set(from, None);
                        if to.rank() == if actor == Player::White { 7 } else { 0 } {
                            piece.kind = PieceKind::Queen;
                        }
                        state.board.set(to, Some(piece));
                        state.card_runtime.moved_piece_ids.insert(piece.id);
                        self.mark_fresh_no_capture(state, piece);
                    }
                }
            }
            PREMOVE => {
                let squares = squares_target(target)?;
                if squares.len() < 2 || squares.len() > 6 || squares.len() % 2 != 0 {
                    return Err(EngineError::Invariant(
                        "premove needs 1 to 3 move square pairs".to_owned(),
                    ));
                }
                let mut moves = Vec::new();
                for chunk in squares.chunks_exact(2) {
                    moves.push(MoveAction::new(chunk[0], chunk[1], None));
                }
                state.card_runtime.pending_premoves.push(PendingPremove {
                    owner: actor,
                    trigger_player: opponent,
                    trigger_turn: state.turns_taken[opponent.index()].saturating_add(1),
                    moves,
                });
            }
            QUANTUM_MECHANICS => state.card_runtime.quantum_pending[actor.index()] = true,
            SCARECROW => {
                let square = square_target(target)?;
                state
                    .card_runtime
                    .pending_scarecrows
                    .push(PendingScarecrow {
                        owner: actor,
                        square,
                        remaining_ply: 6,
                    });
            }
            SCHRODINGER_PAWNS => {
                let mut candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter_map(|(square, piece)| {
                        let direction = if actor == Player::White { -1 } else { 1 };
                        square
                            .offset(0, direction)
                            .filter(|behind| state.board.get(*behind).is_none())
                            .filter(|_| {
                                piece.owner == actor
                                    && piece.kind == PieceKind::Pawn
                                    && !state.card_runtime.quantum_positions.contains_key(&piece.id)
                            })
                            .map(|behind| (piece, behind))
                    })
                    .collect::<Vec<_>>();
                for _ in 0..candidates.len().min(4) {
                    let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                    events.push(Event::RandomSample {
                        purpose: "schrodinger-pawns".to_owned(),
                        roll_53: roll,
                    });
                    let (piece, behind) = candidates.remove(index.expect("non-empty candidates"));
                    state
                        .card_runtime
                        .quantum_positions
                        .insert(piece.id, behind);
                    self.mark_fresh_no_capture(state, piece);
                }
            }
            STEALTH => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.stealth_piece_ids.insert(piece.id);
            }
            SUBMERGE => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.submerged_piece_ids.insert(piece.id);
            }
            SUBSTITUTION => state.card_runtime.substitution[actor.index()] = true,
            crate::OVERTAKE => state.card_runtime.overtake[actor.index()] = true,
            crate::DISASSEMBLY => state.card_runtime.disassembly[actor.index()] = true,
            TWINS => {
                let [first_square, second_square] = two_square_target(target)?;
                let first = state
                    .board
                    .get(first_square)
                    .ok_or_else(|| EngineError::Invariant("first twin disappeared".to_owned()))?;
                let second = state
                    .board
                    .get(second_square)
                    .ok_or_else(|| EngineError::Invariant("second twin disappeared".to_owned()))?;
                state.card_runtime.twin_bonds.retain(|bond| {
                    ![first.id, second.id].contains(&bond.first_id)
                        && ![first.id, second.id].contains(&bond.second_id)
                });
                state.card_runtime.twin_bonds.push(TwinBond {
                    first_id: first.id,
                    second_id: second.id,
                });
            }
            BLACK_MAGIC => {
                let Some(square) = self
                    .black_magic_spawn_candidates(state, actor)
                    .first()
                    .copied()
                else {
                    return Err(EngineError::Invariant(
                        "black magic has no adjacent open square".to_owned(),
                    ));
                };
                let monster =
                    Piece::new(actor, PieceKind::Monster).with_id(state.allocate_piece_id());
                state.board.set(square, Some(monster));
                self.register_spawned_piece(state, square, monster, true);
            }
            FANATICAL_RITUAL => {
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == opponent && !self.is_royal_piece(state, *piece)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "fanatical-ritual".to_owned(),
                    roll_53: roll,
                });
                let (square, mut piece) = candidates[index.ok_or_else(|| {
                    EngineError::Invariant("legal ritual has no candidate".to_owned())
                })?];
                self.clear_queens_gambit_protection(state, piece.id);
                piece.kind = PieceKind::Fanatic;
                state.board.set(square, Some(piece));
                state.card_runtime.moved_piece_ids.insert(piece.id);
            }
            RANDOM_ROULETTE => {
                let (square, mut piece) = piece_target(state, target)?;
                const OUTCOMES: &[PieceKind] = &[
                    PieceKind::Pawn,
                    PieceKind::Knight,
                    PieceKind::Bishop,
                    PieceKind::Rook,
                    PieceKind::Queen,
                    PieceKind::Colossus,
                    PieceKind::BigRook,
                    PieceKind::Protestant,
                    PieceKind::Herald,
                    PieceKind::Cannon,
                    PieceKind::Fanatic,
                    PieceKind::PrimeMinister,
                    PieceKind::Alibaba,
                    PieceKind::Amazon,
                    PieceKind::Cardinal,
                    PieceKind::Unicorn,
                    PieceKind::Jester,
                    PieceKind::Camel,
                    PieceKind::Log,
                    PieceKind::Hook,
                    PieceKind::Grasshopper,
                    PieceKind::Dragon,
                    PieceKind::Man,
                    PieceKind::Assassin,
                    PieceKind::Knightmaster,
                    PieceKind::StandardBearer,
                    PieceKind::Recruiter,
                    PieceKind::Squire,
                    PieceKind::Checker,
                    PieceKind::CheckerKing,
                    PieceKind::Wizard,
                    PieceKind::Alfil,
                    PieceKind::Ferz,
                    PieceKind::Bat,
                    PieceKind::Guard,
                    PieceKind::Reaper,
                    PieceKind::Idol,
                    PieceKind::BabyBear,
                    PieceKind::Lobster,
                    PieceKind::Missionary,
                    PieceKind::Bear,
                    PieceKind::Windmill,
                ];
                let large_anchors = (0..64)
                    .filter_map(Square::from_index)
                    .filter(|anchor| {
                        let cells = large_piece_cells(*anchor);
                        cells.len() == 4
                            && cells.into_iter().all(|cell| {
                                state
                                    .board
                                    .get(cell)
                                    .is_none_or(|occupant| occupant.id == piece.id)
                            })
                    })
                    .collect::<Vec<_>>();
                let outcomes = OUTCOMES
                    .iter()
                    .copied()
                    .filter(|kind| {
                        *kind != piece.kind
                            && (!matches!(kind, PieceKind::Colossus | PieceKind::BigRook)
                                || !large_anchors.is_empty())
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(outcomes.len());
                events.push(Event::RandomSample {
                    purpose: "random-roulette".to_owned(),
                    roll_53: roll,
                });
                self.clear_removed_piece_statuses(state, piece.id);
                piece.kind = outcomes[index.expect("roulette outcomes are non-empty")];
                let destination = if matches!(piece.kind, PieceKind::Colossus | PieceKind::BigRook)
                {
                    let (roll, selected) = state.rng_mut().web_choice_index(large_anchors.len());
                    events.push(Event::RandomSample {
                        purpose: "random-roulette-large-anchor".to_owned(),
                        roll_53: roll,
                    });
                    large_anchors[selected.expect("large outcome requires an anchor")]
                } else {
                    square
                };
                state.board.set(square, None);
                state.board.set(destination, Some(piece));
                state.card_runtime.moved_piece_ids.insert(piece.id);
                match piece.kind {
                    PieceKind::Colossus => {
                        state.card_runtime.insert_large_body(
                            piece.id,
                            destination,
                            PieceHealth::new(3),
                        );
                    }
                    PieceKind::BigRook => {
                        state.card_runtime.insert_large_body(
                            piece.id,
                            destination,
                            PieceHealth::new(2),
                        );
                    }
                    PieceKind::Wizard => {
                        state.card_runtime.wizard_mana.insert(piece.id, 0);
                    }
                    PieceKind::Windmill => {
                        state
                            .card_runtime
                            .windmill_modes
                            .insert(piece.id, WindmillMode::Bishop);
                    }
                    PieceKind::BabyBear => {
                        state
                            .card_runtime
                            .baby_bear_grow_turns
                            .insert(piece.id, state.shared_turn_count().saturating_add(7));
                    }
                    PieceKind::Bear => {
                        state.card_runtime.bear_retaliations.insert(piece.id, 2);
                    }
                    _ => {}
                }
            }
            SWITCHEROO => state.card_runtime.switcheroo[actor.index()] = true,
            TROLLEY => state.card_runtime.pending_trolley_by[opponent.index()] = Some(actor),
            VORTEX => {
                let mut entries = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == opponent
                            && piece.kind != PieceKind::Pawn
                            && !matches!(piece.kind, PieceKind::BigRook | PieceKind::Colossus)
                            && !self.is_royal_piece(state, *piece)
                    })
                    .collect::<Vec<_>>();
                let mut pieces = entries.iter().map(|(_, piece)| *piece).collect::<Vec<_>>();
                for index in (1..pieces.len()).rev() {
                    let (roll, selected) = state.rng_mut().web_choice_index(index + 1);
                    events.push(Event::RandomSample {
                        purpose: "vortex".to_owned(),
                        roll_53: roll,
                    });
                    pieces.swap(index, selected.expect("shuffle prefix is non-empty"));
                }
                for ((square, _), piece) in entries.drain(..).zip(pieces) {
                    state.board.set(square, Some(piece));
                }
            }
            WITCH_TRIAL => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.witch_trials.insert(
                    piece.id,
                    TimedPieceStatus {
                        by: actor,
                        remaining_turns: 3,
                    },
                );
            }
            BLACK_BOX => {
                let active_cards = [PARRY, DISARM, DICE, FREEZE, REPOSITION, SPRINT];
                let (roll, idx) = state.rng_mut().web_choice_index(active_cards.len());
                events.push(Event::RandomSample {
                    purpose: "black-box".to_owned(),
                    roll_53: roll,
                });
                let chosen = active_cards[idx.unwrap_or(0)];
                let _ = self.apply_middle_active_card(
                    state,
                    actor,
                    chosen,
                    &ActionTarget::None,
                    events,
                );
            }
            WHITE_BOX => {}
            INSIGHT => {
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
                        let had_potion_manner =
                            state.card_runtime.potion_manner_piece_ids.remove(&piece.id);
                        if had_potion_manner
                            || state.applied_rules.iter().any(|rule| rule == COOL_GUY)
                        {
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
            }
            crate::NULLIFICATION => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.nullification_piece_ids.insert(piece.id);
            }
            RULE_TICKET => {
                let _ = self.install_rule_ext(state, ACCELERATION, events);
            }
            _ => return Ok(false),
        }
        self.resolve_religious_victory(state);
        Ok(true)
    }
    pub(in crate::engine) fn generate_middle_card_actions(
        &self,
        state: &GameState,
        card: &CardInstance,
        actions: &mut Vec<Action>,
    ) -> bool {
        let actor = state.turn;
        let opponent = actor.opponent();
        let push_none = |actions: &mut Vec<Action>| {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::None,
            })
        };
        let push_piece = |actions: &mut Vec<Action>, square| {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            })
        };
        match card.definition_id.as_str() {
            CLEANUP => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && !state.card_runtime.piece_status.contains_key(&piece.id)
                        && !matches!(
                            piece.kind,
                            PieceKind::Monster | PieceKind::Scarecrow | PieceKind::Football
                        )
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Squares(vec![square]),
                        });
                    }
                }
            }
            BREAKTHROUGH_ORDER
            | EN_PASSANT_BANG
            | ARMISTICE
            | ROYAL_COMMAND
            | SOCIALISM
            | TAUNT
            | MISTAKE_CARD
            | SNIPE
            | GALE
            | QUANTUM_MECHANICS
            | SUBSTITUTION
            | WHITE_BOX
            | BLACK_BOX
            | INSIGHT
            | RULE_TICKET
            | SWITCHEROO
            | FRONTLINE_RESPONSE
            | RELAY
            | crate::OVERTAKE
            | crate::DISASSEMBLY
            | crate::SYMMETRY
            | crate::MIRACLE => push_none(actions),
            SUSPICIOUS_POTION => {
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
            EVASION | ROYAL_SHIELD => {
                if state.board.iter().any(|(_, piece)| {
                    piece.owner == actor
                        && if card.definition_id == EVASION {
                            !state.card_runtime.evasion_piece_ids.contains(&piece.id)
                        } else {
                            !state.card_runtime.shielded_piece_ids.contains(&piece.id)
                        }
                }) {
                    push_none(actions);
                }
            }
            BASIC_TRAINING => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && !self.is_royal_piece(state, piece)
                        && !matches!(
                            piece.kind,
                            PieceKind::Pawn
                                | PieceKind::Queen
                                | PieceKind::PrimeMinister
                                | PieceKind::Jester
                                | PieceKind::Guard
                                | PieceKind::Amazon
                                | PieceKind::Man
                                | PieceKind::Idol
                        )
                        && !state
                            .card_runtime
                            .basic_training_piece_ids
                            .contains(&piece.id)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            PROMOTION_RUSH => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && piece.kind != PieceKind::Pawn
                        && !self.is_royal_piece(state, piece)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            ICE_SHEET => {
                if state
                    .board
                    .iter()
                    .any(|(_, piece)| piece.owner == opponent && is_ranged_piece(piece.kind))
                {
                    push_none(actions);
                }
            }
            INERTIA | SEVERANCE => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == opponent
                        && is_ranged_piece(piece.kind)
                        && !self.is_royal_piece(state, piece)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            ALEKHINE_MACHINE_GUN => {
                if self.alekhine_formation(state, actor).is_some() {
                    push_none(actions);
                }
            }
            CHAMELEON_MUTATION => {
                let candidates = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor && !self.is_royal_piece(state, *piece)
                    })
                    .map(|(square, _)| square)
                    .collect::<Vec<_>>();
                push_square_combinations(actions, card.instance_id, &candidates, 3);
            }
            CHIMERA => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && matches!(piece.kind, PieceKind::Knight | PieceKind::Bishop)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            FEUDAL_CONTRACT => {
                let pawns = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor
                            && matches!(piece.kind, PieceKind::Pawn | PieceKind::Fanatic)
                    })
                    .collect::<Vec<_>>();
                let guardians = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor
                            && !matches!(piece.kind, PieceKind::Pawn | PieceKind::Fanatic)
                            && !self.is_royal_piece(state, *piece)
                    })
                    .collect::<Vec<_>>();
                for (pawn_square, _) in &pawns {
                    for (guardian_square, _) in &guardians {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Squares(vec![*pawn_square, *guardian_square]),
                        });
                    }
                }
            }
            POISONED_PAWN => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && piece.kind == PieceKind::Pawn
                        && !state.card_runtime.poisoned_piece_ids.contains(&piece.id)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            SACRIFICE => {
                let pieces = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor && !self.is_royal_piece(state, *piece)
                    })
                    .collect::<Vec<_>>();
                for (sacrifice_square, sacrificed) in &pieces {
                    for (protected_square, protected) in &pieces {
                        if sacrifice_square != protected_square
                            && middle_piece_value(protected.kind)
                                < middle_piece_value(sacrificed.kind)
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::Squares(vec![
                                    *sacrifice_square,
                                    *protected_square,
                                ]),
                            });
                        }
                    }
                }
            }
            SUICIDE_BOMBER => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && matches!(piece.kind, PieceKind::Pawn | PieceKind::Fanatic)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            crate::OUTPOST => {
                for (square, piece) in state.board.iter() {
                    let in_opponent_half = match actor {
                        Player::White => square.rank() >= 4,
                        Player::Black => square.rank() <= 3,
                    };
                    if piece.owner == actor
                        && !self.is_royal_piece(state, piece)
                        && in_opponent_half
                        && !state
                            .card_runtime
                            .outpost_protected_piece_ids
                            .contains(&piece.id)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            TROJAN_HORSE => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && piece.kind == PieceKind::Knight
                        && !state.card_runtime.trojan_piece_ids.contains(&piece.id)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            CASTLING => {
                if let Some((king_square, _)) = state
                    .board
                    .iter()
                    .find(|(_, piece)| piece.owner == actor && self.is_royal_piece(state, *piece))
                {
                    for (rook_square, piece) in state.board.iter() {
                        if piece.owner == actor
                            && piece.kind == PieceKind::Rook
                            && aligned_distance(king_square, rook_square)
                                .is_some_and(|distance| distance.2 >= 3)
                        {
                            push_piece(actions, rook_square);
                        }
                    }
                }
            }
            EMERGENCY_EVACUATION => {
                let direction = if actor == Player::White { -1 } else { 1 };
                let candidates = state
                    .board
                    .iter()
                    .filter(|(square, piece)| {
                        piece.owner == actor
                            && square
                                .offset(0, direction)
                                .is_some_and(|to| state.board.get(to).is_none())
                    })
                    .map(|(square, _)| square)
                    .collect::<Vec<_>>();
                push_square_combinations(actions, card.instance_id, &candidates, 3);
            }
            PANIC => {
                let candidates = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == opponent && !self.is_royal_piece(state, *piece)
                    })
                    .map(|(square, _)| square)
                    .collect::<Vec<_>>();
                for first in 0..candidates.len() {
                    for second in first + 1..candidates.len() {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Squares(vec![
                                candidates[first],
                                candidates[second],
                            ]),
                        });
                    }
                }
            }
            PAWN_STORM => {
                let direction = if actor == Player::White { 1 } else { -1 };
                let candidates = state
                    .board
                    .iter()
                    .filter(|(square, piece)| {
                        piece.owner == actor
                            && piece.kind == PieceKind::Pawn
                            && square
                                .offset(0, direction)
                                .is_some_and(|to| state.board.get(to).is_none())
                    })
                    .map(|(square, _)| square)
                    .collect::<Vec<_>>();
                for square in &candidates {
                    actions.push(Action::UseCard {
                        instance_id: card.instance_id,
                        target: ActionTarget::Squares(vec![*square]),
                    });
                }
                if candidates.len() > 1 {
                    actions.push(Action::UseCard {
                        instance_id: card.instance_id,
                        target: ActionTarget::Squares(candidates),
                    });
                }
            }
            PREMOVE => {
                for candidate in self.generate_move_actions(state) {
                    if let Action::Move(movement) = candidate {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Squares(vec![movement.from, movement.to]),
                        });
                    }
                }
            }
            SCARECROW => {
                for index in 0..64 {
                    let square = Square::from_index(index).expect("board square");
                    if state.board.get(square).is_none() {
                        if state
                            .card_runtime
                            .pending_scarecrows
                            .iter()
                            .any(|pending| pending.square == square)
                        {
                            continue;
                        }
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Square(square),
                        });
                    }
                }
            }
            SCHRODINGER_PAWNS => {
                if state.board.iter().any(|(square, piece)| {
                    let direction = if actor == Player::White { -1 } else { 1 };
                    piece.owner == actor
                        && piece.kind == PieceKind::Pawn
                        && !state.card_runtime.quantum_positions.contains_key(&piece.id)
                        && square
                            .offset(0, direction)
                            .is_some_and(|behind| state.board.get(behind).is_none())
                }) {
                    push_none(actions);
                }
            }
            STEALTH => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && matches!(piece.kind, PieceKind::Bishop | PieceKind::Protestant)
                        && !state.card_runtime.stealth_piece_ids.contains(&piece.id)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            SUBMERGE => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && !self.is_royal_piece(state, piece)
                        && !state.card_runtime.submerged_piece_ids.contains(&piece.id)
                        && !has_adjacent_enemy(state, square, actor)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            TWINS => {
                let candidates = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor && !self.is_royal_piece(state, *piece)
                    })
                    .map(|(square, _)| square)
                    .collect::<Vec<_>>();
                for first in 0..candidates.len() {
                    for second in first + 1..candidates.len() {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Squares(vec![
                                candidates[first],
                                candidates[second],
                            ]),
                        });
                    }
                }
            }
            BLACK_MAGIC => {
                if state
                    .board
                    .iter()
                    .any(|(_, piece)| piece.owner == actor && self.is_royal_piece(state, piece))
                    && !self.black_magic_spawn_candidates(state, actor).is_empty()
                {
                    push_none(actions);
                }
            }
            FANATICAL_RITUAL => {
                if state
                    .board
                    .iter()
                    .any(|(_, piece)| piece.owner == opponent && !self.is_royal_piece(state, piece))
                {
                    push_none(actions);
                }
            }
            RANDOM_ROULETTE => {
                for (square, piece) in state.board.iter() {
                    if !self.is_royal_piece(state, piece) && is_major_piece(piece.kind) {
                        push_piece(actions, square);
                    }
                }
            }
            TROLLEY => {
                if self.has_trolley_bundle_pair(state, opponent) {
                    push_none(actions);
                }
            }
            VORTEX => {
                if state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == opponent
                            && piece.kind != PieceKind::Pawn
                            && !matches!(piece.kind, PieceKind::BigRook | PieceKind::Colossus)
                            && !self.is_royal_piece(state, *piece)
                    })
                    .count()
                    >= 2
                {
                    push_none(actions);
                }
            }
            crate::NULLIFICATION => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && !state
                            .card_runtime
                            .nullification_piece_ids
                            .contains(&piece.id)
                    {
                        push_piece(actions, square);
                    }
                }
            }
            WITCH_TRIAL => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == opponent && !self.is_royal_piece(state, piece) {
                        push_piece(actions, square);
                    }
                }
            }
            _ => return false,
        }
        true
    }
}

use super::*;
use crate::*;

const KING_OFFSETS: &[(i8, i8)] = &[
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];
pub(super) const TROLLEY_CHOICE_HANDLER: &str = "middle.trolley";

#[derive(Clone, Copy)]
pub(super) struct MiddleCaptureContext {
    poisoned: bool,
    explosive: bool,
    trojan: bool,
    feudal_guardian_id: Option<u32>,
}

impl Engine {
    pub(super) fn apply_middle_active_card(
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
                        self.remove_piece_for_card(
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
                let pieces = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| piece.owner == actor)
                    .collect::<Vec<_>>();
                let royal = pieces
                    .iter()
                    .copied()
                    .find(|(_, piece)| self.is_royal_piece(state, *piece))
                    .ok_or_else(|| EngineError::Invariant("black magic has no king".to_owned()))?;
                let mut removed = 0_usize;
                for (square, mut piece) in pieces {
                    if piece.id == royal.1.id {
                        self.clear_piece_kind_runtime(state, piece.id);
                        piece.kind = PieceKind::DarkWizard;
                        state.board.set(square, Some(piece));
                        state.card_runtime.moved_piece_ids.insert(piece.id);
                        state.card_runtime.piece_origins.insert(piece.id, square);
                        continue;
                    }
                    self.remove_piece_for_card(
                        state,
                        square,
                        piece,
                        actor.opponent(),
                        BLACK_MAGIC,
                        events,
                    );
                    removed += 1;
                }
                let spawn_count = removed.div_ceil(2);
                let ranks: Vec<u8> = if actor == Player::White {
                    (0..8).collect()
                } else {
                    (0..8).rev().collect()
                };
                let open = ranks
                    .into_iter()
                    .flat_map(|rank| (0..8).filter_map(move |file| Square::new(file, rank)))
                    .filter(|square| state.board.get(*square).is_none())
                    .take(spawn_count)
                    .collect::<Vec<_>>();
                for square in open {
                    let monster =
                        Piece::new(actor, PieceKind::Monster).with_id(state.allocate_piece_id());
                    state.board.set(square, Some(monster));
                    self.mark_fresh_no_capture(state, monster);
                }
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
                        state.card_runtime.large_pieces.insert(
                            piece.id,
                            LargePieceStatus {
                                anchor: destination,
                                hp: 3,
                                max_hp: 3,
                            },
                        );
                    }
                    PieceKind::BigRook => {
                        state.card_runtime.large_pieces.insert(
                            piece.id,
                            LargePieceStatus {
                                anchor: destination,
                                hp: 2,
                                max_hp: 2,
                            },
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
            RULE_TICKET => {
                let _ = self.install_rule_ext(state, ACCELERATION, events);
            }
            _ => return Ok(false),
        }
        self.resolve_religious_victory(state);
        Ok(true)
    }

    pub(super) fn generate_middle_card_actions(
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
                        && !state.card_runtime.large_pieces.contains_key(&piece.id)
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
            BREAKTHROUGH_ORDER | EN_PASSANT_BANG | ARMISTICE | ROYAL_COMMAND | SOCIALISM
            | TAUNT | MISTAKE_CARD | SNIPE | GALE | QUANTUM_MECHANICS | SUBSTITUTION
            | WHITE_BOX | BLACK_BOX | INSIGHT | RULE_TICKET | SWITCHEROO | FRONTLINE_RESPONSE
            | RELAY => push_none(actions),
            SUSPICIOUS_POTION => {
                for (square, piece) in state.board.iter() {
                    if !state.card_runtime.large_pieces.contains_key(&piece.id)
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

    fn alekhine_formation(&self, state: &GameState, actor: Player) -> Option<Vec<u32>> {
        let queens = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Queen);
        for (queen_square, queen) in queens {
            let rooks = state
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Rook)
                .collect::<Vec<_>>();
            for first in 0..rooks.len() {
                for second in first + 1..rooks.len() {
                    let a = rooks[first].0;
                    let b = rooks[second].0;
                    if (a.rank() == queen_square.rank()
                        && b.rank() == queen_square.rank()
                        && (a.file() < queen_square.file()) != (b.file() < queen_square.file()))
                        || (a.file() == queen_square.file()
                            && b.file() == queen_square.file()
                            && (a.rank() < queen_square.rank()) != (b.rank() < queen_square.rank()))
                    {
                        return Some(vec![queen.id, rooks[first].1.id, rooks[second].1.id]);
                    }
                }
            }
        }
        None
    }

    fn apply_free_castling(
        &self,
        state: &mut GameState,
        actor: Player,
        target: &ActionTarget,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let rook_square = square_target(target)?;
        let rook = state
            .board
            .get(rook_square)
            .ok_or_else(|| EngineError::Invariant("castling rook disappeared".to_owned()))?;
        let (king_square, king) = state
            .board
            .iter()
            .find(|(_, piece)| piece.owner == actor && self.is_royal_piece(state, *piece))
            .ok_or_else(|| EngineError::Invariant("castling king disappeared".to_owned()))?;
        let (file_delta, rank_delta, distance) = aligned_distance(king_square, rook_square)
            .ok_or_else(|| EngineError::Invariant("castling pieces are not aligned".to_owned()))?;
        if distance < 3 {
            return Err(EngineError::Invariant(
                "castling rook is too close".to_owned(),
            ));
        }
        let king_to = king_square
            .offset(file_delta * 2, rank_delta * 2)
            .expect("aligned castling destination");
        let rook_to = king_square
            .offset(file_delta, rank_delta)
            .expect("aligned rook destination");
        state.board.set(king_square, None);
        state.board.set(rook_square, None);
        for (square, attacker) in [(king_to, king), (rook_to, rook)] {
            if let Some(victim) = state.board.get(square) {
                state.board.set(square, None);
                self.clear_removed_piece_statuses(state, victim.id);
                events.push(Event::PieceCaptured {
                    actor,
                    piece: victim,
                    at: square,
                });
            }
            state.board.set(square, Some(attacker));
            state.card_runtime.moved_piece_ids.insert(attacker.id);
        }
        state.en_passant = None;
        Ok(())
    }

    pub(super) fn middle_piece_is_exhausted(&self, state: &GameState, piece: Piece) -> bool {
        state.card_runtime.exhaustion[piece.owner.index()]
            && !self.is_royal_piece(state, piece)
            && state.card_runtime.exhaustion_last_piece[piece.owner.index()] == Some(piece.id)
            && state.card_runtime.exhaustion_count[piece.owner.index()] >= 3
    }

    pub(super) fn middle_encouragement_protects(&self, state: &GameState, target: Piece) -> bool {
        if !state.card_runtime.encouragement[target.owner.index()] {
            return false;
        }
        let Some((target_square, _)) = state.board.find_by_id(target.id) else {
            return false;
        };
        state.board.iter().any(|(square, piece)| {
            piece.owner == target.owner
                && self.is_royal_piece(state, piece)
                && piece.id != target.id
                && squares_are_adjacent(square, target_square)
        })
    }

    pub(super) fn middle_submerged_protects(&self, state: &GameState, target: Piece) -> bool {
        state.card_runtime.submerged_piece_ids.contains(&target.id)
            && state
                .board
                .find_by_id(target.id)
                .is_some_and(|(square, piece)| !has_adjacent_enemy(state, square, piece.owner))
    }

    pub(super) fn generate_socialism_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let direction = if piece.owner == Player::White { 1 } else { -1 };
        if let Some(to) = from.offset(0, direction)
            && self.board_piece_at(state, to).is_none()
        {
            actions.push(Action::Move(MoveAction::new(from, to, None)));
        }
        for file_delta in [-1, 1] {
            if let Some(to) = from.offset(file_delta, direction)
                && self
                    .board_piece_at(state, to)
                    .is_some_and(|target| self.can_capture(state, piece, target))
            {
                actions.push(Action::Move(MoveAction::new(from, to, None)));
            }
        }
    }

    pub(super) fn augment_middle_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let owner = piece.owner.index();
        if state
            .card_runtime
            .basic_training_piece_ids
            .contains(&piece.id)
        {
            self.generate_socialism_moves(state, from, piece, actions);
        }
        if state
            .card_runtime
            .promotion_rush_piece_ids
            .contains(&piece.id)
        {
            self.generate_slider_moves(state, from, piece, QUEEN_DIRECTIONS, actions);
        }
        if piece.kind == PieceKind::Pawn {
            if state.card_runtime.pawn_leap[owner] {
                self.generate_leaper_moves(state, from, piece, KNIGHT_OFFSETS, actions);
            }
            if state.card_runtime.retreat[owner] {
                let backward = if piece.owner == Player::White { -1 } else { 1 };
                if let Some(to) = from.offset(0, backward)
                    && self.board_piece_at(state, to).is_none()
                {
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                }
            }
            if state.card_runtime.breakthrough_pawns[owner] {
                let direction = if piece.owner == Player::White { 1 } else { -1 };
                let mut cursor = from;
                while let Some(to) = cursor.offset(0, direction) {
                    if self.board_piece_at(state, to).is_some() {
                        break;
                    }
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                    cursor = to;
                }
            }
            if state.card_runtime.en_passant_frenzy[owner] {
                let direction = if piece.owner == Player::White { 1 } else { -1 };
                for file_delta in [-1, 1] {
                    let Some(adjacent) = from.offset(file_delta, 0) else {
                        continue;
                    };
                    let Some(target_piece) = self.board_piece_at(state, adjacent) else {
                        continue;
                    };
                    let Some(to) = from.offset(file_delta, direction) else {
                        continue;
                    };
                    if self.board_piece_at(state, to).is_none()
                        && self.can_capture(state, piece, target_piece)
                    {
                        actions.push(Action::Move(MoveAction::new(from, to, None)));
                    }
                }
            }
        }
        if self.is_royal_piece(state, piece) && state.card_runtime.horse_riding[owner] {
            self.generate_leaper_moves(state, from, piece, KNIGHT_OFFSETS, actions);
        }
        if piece.kind == PieceKind::Knight
            && state.card_runtime.corner_kick[owner]
            && matches!(from.file(), 0 | 7)
            && matches!(from.rank(), 0 | 7)
        {
            self.generate_slider_moves(state, from, piece, BISHOP_DIRECTIONS, actions);
        }
        if state.card_runtime.royal_command[owner].is_some_and(|status| {
            let turn = state.turns_taken[owner];
            turn >= status.starts_at_turn && turn < status.expires_at_turn
        }) {
            self.generate_leaper_moves(state, from, piece, KING_OFFSETS, actions);
        }
        if let Some(kinds) = state.card_runtime.imperial_moves.get(&piece.id) {
            for kind in kinds {
                self.generate_middle_kind_moves(state, from, piece, *kind, actions);
            }
        }
        if piece.kind == PieceKind::Knight && state.card_runtime.mad_horse[owner] {
            for &(file, rank) in KNIGHT_OFFSETS {
                if let Some(to) = from.offset(file, rank)
                    && self.board_piece_at(state, to).is_some_and(|target| {
                        target.owner == piece.owner && !self.is_royal_piece(state, target)
                    })
                {
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                }
            }
        }
        if state.card_runtime.substitution[owner] {
            for (to, target) in state.board.iter() {
                if target.owner != piece.owner && target.kind == piece.kind {
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                }
            }
        }
        if self.is_royal_piece(state, piece) && state.card_runtime.switcheroo[owner] {
            for (to, target) in state.board.iter() {
                if target.owner == piece.owner && target.kind == PieceKind::Pawn {
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                }
            }
        }
        if piece.kind == PieceKind::Bishop && state.card_runtime.bishop_snipe[owner] {
            for &(file, rank) in BISHOP_DIRECTIONS {
                let mut cursor = from;
                let mut hurdle = false;
                while let Some(to) = cursor.offset(file, rank) {
                    if let Some(target) = self.board_piece_at(state, to) {
                        if hurdle {
                            if self.can_capture(state, piece, target) {
                                actions.push(Action::Move(MoveAction::new(from, to, None)));
                            }
                            break;
                        }
                        hurdle = true;
                    }
                    cursor = to;
                }
            }
        }
    }

    fn generate_middle_kind_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        kind: PieceKind,
        actions: &mut Vec<Action>,
    ) {
        match kind {
            PieceKind::Pawn => self.generate_socialism_moves(state, from, piece, actions),
            PieceKind::Knight => {
                self.generate_leaper_moves(state, from, piece, KNIGHT_OFFSETS, actions)
            }
            PieceKind::Bishop => {
                self.generate_slider_moves(state, from, piece, BISHOP_DIRECTIONS, actions)
            }
            PieceKind::Rook => {
                self.generate_slider_moves(state, from, piece, ROOK_DIRECTIONS, actions)
            }
            PieceKind::Queen => {
                self.generate_slider_moves(state, from, piece, QUEEN_DIRECTIONS, actions)
            }
            _ => {}
        }
    }

    pub(super) fn filter_middle_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let owner = piece.owner.index();
        actions.retain(|action| {
            let Action::Move(movement) = action else {
                return true;
            };
            if state
                .card_runtime
                .pending_scarecrows
                .iter()
                .any(|pending| pending.square == movement.to)
            {
                return false;
            }
            let file_distance = movement.from.file().abs_diff(movement.to.file());
            let rank_distance = movement.from.rank().abs_diff(movement.to.rank());
            if piece.kind == PieceKind::DarkWizard
                && state
                    .card_runtime
                    .piece_origins
                    .get(&piece.id)
                    .is_some_and(|origin| {
                        movement.to.file().abs_diff(origin.file()) > 1
                            || movement.to.rank().abs_diff(origin.rank()) > 1
                    })
            {
                return false;
            }
            if state.card_runtime.inertia_piece_ids.contains(&piece.id)
                && file_distance <= 1
                && rank_distance <= 1
            {
                return false;
            }
            if state.card_runtime.severed.contains_key(&piece.id)
                && (file_distance > 1 || rank_distance > 1)
            {
                return false;
            }
            if state.card_runtime.taunt[owner] > 0 {
                let backwards = if piece.owner == Player::White {
                    movement.to.rank() < from.rank()
                } else {
                    movement.to.rank() > from.rank()
                };
                if backwards {
                    return false;
                }
            }
            if piece.kind == PieceKind::Knight && state.card_runtime.knight_injury[owner] {
                let file_delta = movement.to.file() as i8 - from.file() as i8;
                let rank_delta = movement.to.rank() as i8 - from.rank() as i8;
                if file_delta.unsigned_abs() == 2 && rank_delta.unsigned_abs() == 1 {
                    if from
                        .offset(file_delta.signum(), 0)
                        .is_some_and(|middle| self.board_piece_at(state, middle).is_some())
                    {
                        return false;
                    }
                } else if rank_delta.unsigned_abs() == 2
                    && file_delta.unsigned_abs() == 1
                    && from
                        .offset(0, rank_delta.signum())
                        .is_some_and(|middle| self.board_piece_at(state, middle).is_some())
                {
                    return false;
                }
            }
            true
        });

        if state.card_runtime.ice_sheet.contains_key(&piece.id) {
            let mut maximum = std::collections::BTreeMap::<(i8, i8), u8>::new();
            for action in actions.iter() {
                let Action::Move(movement) = action else {
                    continue;
                };
                let file = movement.to.file() as i8 - from.file() as i8;
                let rank = movement.to.rank() as i8 - from.rank() as i8;
                let direction = (file.signum(), rank.signum());
                maximum
                    .entry(direction)
                    .and_modify(|distance| {
                        *distance = (*distance).max(file.unsigned_abs().max(rank.unsigned_abs()))
                    })
                    .or_insert(file.unsigned_abs().max(rank.unsigned_abs()));
            }
            actions.retain(|action| {
                let Action::Move(movement) = action else {
                    return true;
                };
                let file = movement.to.file() as i8 - from.file() as i8;
                let rank = movement.to.rank() as i8 - from.rank() as i8;
                maximum
                    .get(&(file.signum(), rank.signum()))
                    .is_some_and(|maximum| *maximum == file.unsigned_abs().max(rank.unsigned_abs()))
            });
        }
    }

    pub(super) fn middle_evasion_destinations(
        &self,
        state: &GameState,
        origin: Square,
        attacker_landing: Square,
    ) -> Vec<Square> {
        KING_OFFSETS
            .iter()
            .filter_map(|(file, rank)| origin.offset(*file, *rank))
            .filter(|square| {
                *square != attacker_landing && self.board_piece_at(state, *square).is_none()
            })
            .collect()
    }

    pub(super) fn middle_capture_context(
        &self,
        state: &GameState,
        target: Piece,
    ) -> MiddleCaptureContext {
        MiddleCaptureContext {
            poisoned: state.card_runtime.poisoned_piece_ids.contains(&target.id),
            explosive: state.card_runtime.explosive_piece_ids.contains(&target.id),
            trojan: state.card_runtime.trojan_piece_ids.contains(&target.id),
            feudal_guardian_id: state
                .card_runtime
                .feudal_contracts
                .iter()
                .find(|contract| contract.pawn_id == target.id)
                .map(|contract| contract.guardian_id),
        }
    }

    pub(super) fn note_middle_piece_moved(&self, state: &mut GameState, piece: Piece) {
        let owner = piece.owner.index();
        if !state.card_runtime.exhaustion[owner] {
            return;
        }
        if state.card_runtime.exhaustion_last_piece[owner] == Some(piece.id) {
            state.card_runtime.exhaustion_count[owner] =
                state.card_runtime.exhaustion_count[owner].saturating_add(1);
        } else {
            state.card_runtime.exhaustion_last_piece[owner] = Some(piece.id);
            state.card_runtime.exhaustion_count[owner] = 1;
        }
    }

    pub(super) fn resolve_middle_after_move(
        &self,
        state: &mut GameState,
        original_piece: Piece,
        movement: MoveAction,
        captured: Option<(Square, Piece)>,
        context: Option<MiddleCaptureContext>,
        events: &mut Vec<Event>,
    ) -> bool {
        self.note_middle_piece_moved(state, original_piece);
        state
            .card_runtime
            .quantum_positions
            .remove(&original_piece.id);
        let owner = original_piece.owner;
        let captured_piece = captured.map(|(_, piece)| piece);

        if let Some(captured_piece) = captured_piece {
            if state
                .card_runtime
                .chameleon_piece_ids
                .contains(&original_piece.id)
                && let Some(mut attacker) = state.board.get(movement.to)
            {
                attacker.kind = captured_piece.kind;
                state.board.set(movement.to, Some(attacker));
            }
            if self.is_royal_piece(state, original_piece)
                && state.card_runtime.imperial_studies[owner.index()]
            {
                state
                    .card_runtime
                    .imperial_moves
                    .entry(original_piece.id)
                    .or_default()
                    .insert(captured_piece.kind);
            }
            if original_piece.kind == PieceKind::Queen
                && captured_piece.kind == PieceKind::Queen
                && state.card_runtime.queen_afterimage[owner.index()]
                && state.board.get(movement.from).is_none()
            {
                let afterimage =
                    Piece::new(owner, PieceKind::Queen).with_id(state.allocate_piece_id());
                state.board.set(movement.from, Some(afterimage));
                self.mark_fresh_no_capture(state, afterimage);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("queen-afterimage:{}:{}", afterimage.id, movement.from),
                });
            }
            if state
                .card_runtime
                .witch_trials
                .remove(&original_piece.id)
                .is_some()
            {
                state
                    .card_runtime
                    .shielded_piece_ids
                    .insert(original_piece.id);
            }
        }

        if state
            .card_runtime
            .chimera_piece_ids
            .contains(&original_piece.id)
            && let Some(mut chimera) = state.board.get(movement.to)
        {
            let (roll, selected) = state.rng_mut().web_choice_index(25);
            events.push(Event::RandomSample {
                purpose: "chimera".to_owned(),
                roll_53: roll,
            });
            let selected = selected.unwrap_or(0);
            chimera.kind = if selected == 24 {
                PieceKind::Queen
            } else {
                [
                    PieceKind::Pawn,
                    PieceKind::Knight,
                    PieceKind::Bishop,
                    PieceKind::Rook,
                ][selected % 4]
            };
            state.board.set(movement.to, Some(chimera));
        }

        if state.card_runtime.quantum_pending[owner.index()]
            && captured.is_none()
            && let Some(attacker) = state.board.get(movement.to)
        {
            let candidates = (0..64)
                .filter_map(Square::from_index)
                .filter(|square| state.board.get(*square).is_none())
                .collect::<Vec<_>>();
            let (roll, selected) = state.rng_mut().web_choice_index(candidates.len());
            events.push(Event::RandomSample {
                purpose: "quantum-mechanics".to_owned(),
                roll_53: roll,
            });
            if let Some(selected) = selected {
                state
                    .card_runtime
                    .quantum_positions
                    .insert(attacker.id, candidates[selected]);
            }
            state.card_runtime.quantum_pending[owner.index()] = false;
        }

        if original_piece.kind == PieceKind::Knight
            && state.card_runtime.radical_charge[owner.index()]
        {
            let file = movement.to.file() as i8 - movement.from.file() as i8;
            let rank = movement.to.rank() as i8 - movement.from.rank() as i8;
            let jumped = if file.unsigned_abs() == 2 {
                movement.from.offset(file.signum(), 0)
            } else if rank.unsigned_abs() == 2 {
                movement.from.offset(0, rank.signum())
            } else {
                None
            };
            if let Some(square) = jumped
                && let Some(target) = state.board.get(square)
                && target.owner != owner
            {
                self.remove_piece_for_card(state, square, target, owner, RADICAL_CHARGE, events);
            }
        }

        if let Some(context) = context {
            if context.poisoned
                && let Some(attacker) = state.board.get(movement.to)
            {
                state.card_runtime.frozen.insert(
                    attacker.id,
                    FrozenStatus {
                        source: captured_piece.map_or(owner.opponent(), |piece| piece.owner),
                        remaining_turns: 3,
                    },
                );
            }
            if context.explosive {
                let center = captured.map_or(movement.to, |(square, _)| square);
                let victims = state
                    .board
                    .iter()
                    .filter(|(square, _)| {
                        square.file().abs_diff(center.file()) <= 1
                            && square.rank().abs_diff(center.rank()) <= 1
                    })
                    .filter(|(_, piece)| {
                        !state.card_runtime.frozen.contains_key(&piece.id)
                            && !matches!(
                                piece.kind,
                                PieceKind::Guard | PieceKind::Jester | PieceKind::Monster
                            )
                    })
                    .collect::<Vec<_>>();
                for (square, victim) in victims {
                    self.remove_piece_for_card(
                        state,
                        square,
                        victim,
                        owner,
                        SUICIDE_BOMBER,
                        events,
                    );
                }
            }
            if context.trojan
                && let Some(attacker) = state.board.find_by_id(original_piece.id)
            {
                self.remove_piece_for_card(
                    state,
                    attacker.0,
                    attacker.1,
                    captured_piece.map_or(owner.opponent(), |piece| piece.owner),
                    TROJAN_HORSE,
                    events,
                );
                if state.board.get(movement.to).is_none() {
                    let pawn_owner = captured_piece.map_or(owner.opponent(), |piece| piece.owner);
                    let pawn =
                        Piece::new(pawn_owner, PieceKind::Pawn).with_id(state.allocate_piece_id());
                    state.board.set(movement.to, Some(pawn));
                    self.mark_fresh_no_capture(state, pawn);
                }
            }
            if let Some(guardian_id) = context.feudal_guardian_id
                && let Some((guardian_square, mut guardian)) = state.board.find_by_id(guardian_id)
                && let Some((attacker_square, attacker)) = state.board.find_by_id(original_piece.id)
            {
                state.board.set(guardian_square, None);
                self.remove_piece_for_card(
                    state,
                    attacker_square,
                    attacker,
                    guardian.owner,
                    FEUDAL_CONTRACT,
                    events,
                );
                guardian.capture_count = guardian.capture_count.saturating_add(1);
                state.board.set(attacker_square, Some(guardian));
                state.card_runtime.moved_piece_ids.insert(guardian.id);
            }
        }

        if let Some(bond) = state
            .card_runtime
            .twin_bonds
            .iter()
            .find(|bond| bond.first_id == original_piece.id || bond.second_id == original_piece.id)
            .copied()
        {
            let partner_id = if bond.first_id == original_piece.id {
                bond.second_id
            } else {
                bond.first_id
            };
            if let Some((partner_square, partner)) = state.board.find_by_id(partner_id)
                && let Some(moved) = state.board.get(movement.to)
            {
                state.board.set(partner_square, Some(moved));
                state.board.set(movement.to, Some(partner));
            }
        }

        let captured_backward = captured_piece.is_some()
            && original_piece.kind == PieceKind::Knight
            && state.card_runtime.backward_knight[owner.index()]
            && if owner == Player::White {
                movement.to.rank() < movement.from.rank()
            } else {
                movement.to.rank() > movement.from.rank()
            };
        let captured_friend = captured_piece.is_some_and(|piece| piece.owner == owner);
        let extra_move = captured_backward
            || (original_piece.kind == PieceKind::Knight
                && state.card_runtime.file_surge[owner.index()]
                && matches!(movement.to.file(), 0 | 7))
            || (original_piece.kind == PieceKind::Rook
                && state.card_runtime.rook_lift[owner.index()]
                && matches!(
                    (movement.to.file(), movement.to.rank()),
                    (0, 0) | (0, 7) | (7, 0) | (7, 7)
                ))
            || (original_piece.kind == PieceKind::Knight
                && state.card_runtime.mad_horse[owner.index()]
                && captured_friend);
        if extra_move && state.board.find_by_id(original_piece.id).is_some() {
            state.phase = GamePhase::ForcedMove {
                piece_id: original_piece.id,
                source: ForcedMoveSource::Effect("middle-extra-move".to_owned()),
                continuation: ForcedMoveContinuation::FinishTurn,
            };
            events.push(Event::ForcedMoveStarted {
                player: owner,
                piece_id: original_piece.id,
                source: "middle-extra-move".to_owned(),
            });
        }
        self.resolve_religious_victory(state);
        state.board.find_by_id(original_piece.id).is_some()
    }

    pub(super) fn tick_middle_statuses_after_turn(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        let owner = actor.index();
        state.card_runtime.breakthrough_pawns[owner] = false;
        state.card_runtime.en_passant_frenzy[owner] = false;
        state.card_runtime.bishop_snipe[owner] = false;
        state.card_runtime.quantum_pending[owner] = false;
        state.card_runtime.substitution[owner] = false;
        state.card_runtime.switcheroo[owner] = false;
        state.card_runtime.mistake_card[owner] = false;
        state.card_runtime.socialism[owner] = state.card_runtime.socialism[owner].saturating_sub(1);
        state.card_runtime.taunt[owner] = state.card_runtime.taunt[owner].saturating_sub(1);
        state.card_runtime.armistice_remaining_ply =
            state.card_runtime.armistice_remaining_ply.saturating_sub(1);
        state.card_runtime.stealth_piece_ids.retain(|piece_id| {
            state
                .board
                .find_by_id(*piece_id)
                .is_some_and(|(_, piece)| piece.owner == actor)
        });

        if state.card_runtime.royal_command[owner]
            .is_some_and(|status| state.turns_taken[owner] >= status.expires_at_turn)
        {
            state.card_runtime.royal_command[owner] = None;
        }

        let timed_ids = |entries: &std::collections::BTreeMap<u32, TimedPieceStatus>,
                         state: &GameState| {
            entries
                .keys()
                .copied()
                .filter(|piece_id| {
                    state
                        .board
                        .find_by_id(*piece_id)
                        .is_some_and(|(_, piece)| piece.owner == actor)
                })
                .collect::<Vec<_>>()
        };
        for piece_id in timed_ids(&state.card_runtime.ice_sheet, state) {
            if let Some(status) = state.card_runtime.ice_sheet.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                if status.remaining_turns == 0 {
                    state.card_runtime.ice_sheet.remove(&piece_id);
                }
            }
        }
        for piece_id in timed_ids(&state.card_runtime.severed, state) {
            if let Some(status) = state.card_runtime.severed.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                if status.remaining_turns == 0 {
                    state.card_runtime.severed.remove(&piece_id);
                }
            }
        }
        for piece_id in timed_ids(&state.card_runtime.sacrifice_protection, state) {
            if let Some(status) = state.card_runtime.sacrifice_protection.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                if status.remaining_turns == 0 {
                    state.card_runtime.sacrifice_protection.remove(&piece_id);
                    if !has_piece_protection_source(state, piece_id) {
                        state.card_runtime.protected_piece_ids.remove(&piece_id);
                    }
                }
            }
        }
        for piece_id in timed_ids(&state.card_runtime.last_resistance, state) {
            if let Some(status) = state.card_runtime.last_resistance.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                if status.remaining_turns == 0 {
                    state.card_runtime.last_resistance.remove(&piece_id);
                    if !has_piece_protection_source(state, piece_id) {
                        state.card_runtime.protected_piece_ids.remove(&piece_id);
                    }
                }
            }
        }
        let witch_ids = timed_ids(&state.card_runtime.witch_trials, state);
        for piece_id in witch_ids {
            let expires = if let Some(status) = state.card_runtime.witch_trials.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                status.remaining_turns == 0
            } else {
                false
            };
            if expires {
                state.card_runtime.witch_trials.remove(&piece_id);
                if let Some((square, piece)) = state.board.find_by_id(piece_id) {
                    self.remove_piece_for_card(
                        state,
                        square,
                        piece,
                        piece.owner.opponent(),
                        WITCH_TRIAL,
                        events,
                    );
                }
            }
        }

        let pending = std::mem::take(&mut state.card_runtime.pending_scarecrows);
        for mut scarecrow in pending {
            scarecrow.remaining_ply = scarecrow.remaining_ply.saturating_sub(1);
            if scarecrow.remaining_ply > 0 {
                state.card_runtime.pending_scarecrows.push(scarecrow);
            } else if state.board.get(scarecrow.square).is_none() {
                let piece = Piece::new(scarecrow.owner, PieceKind::Scarecrow)
                    .with_id(state.allocate_piece_id());
                state.board.set(scarecrow.square, Some(piece));
                state.card_runtime.moved_piece_ids.insert(piece.id);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("scarecrow-spawn:{}:{}", piece.id, scarecrow.square),
                });
            }
        }

        let current_turn = state.turns_taken[owner];
        let gale_due = state.card_runtime.gale_due_turns[owner]
            .iter()
            .any(|due| *due <= current_turn);
        state.card_runtime.gale_due_turns[owner].retain(|due| *due > current_turn);
        if gale_due {
            let isolated = state
                .board
                .iter()
                .filter(|(square, piece)| {
                    piece.owner == actor
                        && !KING_OFFSETS.iter().any(|(file, rank)| {
                            square
                                .offset(*file, *rank)
                                .and_then(|near| state.board.get(near))
                                .is_some_and(|near| near.owner == actor)
                        })
                })
                .collect::<Vec<_>>();
            for (square, piece) in isolated {
                self.remove_piece_for_card(state, square, piece, actor.opponent(), GALE, events);
            }
        }

        let grown = state
            .card_runtime
            .baby_bear_grow_turns
            .iter()
            .filter_map(|(piece_id, due)| (state.shared_turn_count() >= *due).then_some(*piece_id))
            .collect::<Vec<_>>();
        for piece_id in grown {
            state.card_runtime.baby_bear_grow_turns.remove(&piece_id);
            if let Some((square, mut piece)) = state.board.find_by_id(piece_id) {
                piece.kind = PieceKind::Bear;
                state.board.set(square, Some(piece));
                state.card_runtime.bear_retaliations.insert(piece.id, 2);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("baby-bear-grow:{}:{}", piece.id, square),
                });
            }
        }
    }

    pub(super) fn resolve_middle_turn_start(
        &self,
        state: &mut GameState,
        player: Player,
        events: &mut Vec<Event>,
    ) {
        let expired_coronation = state
            .card_runtime
            .coronation_protection
            .iter()
            .filter_map(|(piece_id, start_turn)| {
                state
                    .board
                    .find_by_id(*piece_id)
                    .is_some_and(|(_, piece)| {
                        piece.owner == player && state.turns_taken[player.index()] > *start_turn
                    })
                    .then_some(*piece_id)
            })
            .collect::<Vec<_>>();
        for piece_id in expired_coronation {
            state.card_runtime.coronation_protection.remove(&piece_id);
            if !has_piece_protection_source(state, piece_id) {
                state.card_runtime.protected_piece_ids.remove(&piece_id);
            }
        }
        let revealed = state
            .card_runtime
            .submerged_piece_ids
            .iter()
            .copied()
            .filter(|piece_id| {
                state
                    .board
                    .find_by_id(*piece_id)
                    .is_none_or(|(square, piece)| {
                        piece.owner == player && has_adjacent_enemy(state, square, piece.owner)
                    })
            })
            .collect::<Vec<_>>();
        for piece_id in revealed {
            state.card_runtime.submerged_piece_ids.remove(&piece_id);
        }

        let pending = std::mem::take(&mut state.card_runtime.pending_panic);
        for panic in pending {
            if panic.owner != player {
                state.card_runtime.pending_panic.push(panic);
                continue;
            }
            for piece_id in panic.piece_ids {
                let candidates = self
                    .generate_piece_move_actions(state, piece_id)
                    .into_iter()
                    .filter_map(|action| match action {
                        Action::Move(movement)
                            if state.board.get(movement.to).is_none()
                                && movement.promotion.is_none() =>
                        {
                            Some(movement)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let (roll, selected) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "panic".to_owned(),
                    roll_53: roll,
                });
                if let Some(selected) = selected {
                    let movement = candidates[selected];
                    if let Some(piece) = state.board.get(movement.from) {
                        state.board.set(movement.from, None);
                        state.board.set(movement.to, Some(piece));
                        state.card_runtime.moved_piece_ids.insert(piece.id);
                        events.push(Event::EffectApplied {
                            source_card_instance: 0,
                            command: format!(
                                "panic:{}:{}->{}",
                                piece.id, movement.from, movement.to
                            ),
                        });
                    }
                }
            }
        }

        if let Some(by) = state.card_runtime.pending_trolley_by[player.index()].take()
            && let Some(bundles) = self.sample_middle_trolley_bundle_pair(state, player, events)
        {
            let labels = bundles.each_ref().map(|bundle| {
                let pieces = bundle
                    .iter()
                    .filter_map(|piece_id| {
                        state
                            .board
                            .find_by_id(*piece_id)
                            .map(|(_, piece)| format!("{:?}", piece.kind))
                    })
                    .collect::<Vec<_>>();
                format!("희생: {}", pieces.join(" + "))
            });
            state.card_runtime.active_trolley = Some(TrolleyDilemma { by, bundles });
            let choice_id = state.allocate_choice_id();
            state.phase = GamePhase::AwaitingChoice(PendingChoice {
                id: choice_id,
                owner: player,
                source_card_instance: 0,
                handler_id: TROLLEY_CHOICE_HANDLER.to_owned(),
                options: labels.to_vec(),
                continuation: ChoiceContinuation::ResumeTurn,
            });
            events.push(Event::ChoiceRequested {
                choice_id,
                owner: player,
                handler_id: TROLLEY_CHOICE_HANDLER.to_owned(),
                options: labels.to_vec(),
            });
        }
    }

    pub(super) fn resolve_middle_trolley_choice(
        &self,
        state: &mut GameState,
        option_index: usize,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let dilemma = state.card_runtime.active_trolley.take().ok_or_else(|| {
            EngineError::Invariant("trolley choice has no active dilemma".to_owned())
        })?;
        let doomed = dilemma
            .bundles
            .get(option_index)
            .ok_or_else(|| EngineError::Invariant("trolley choice index is invalid".to_owned()))?;
        for piece_id in doomed {
            if let Some((square, piece)) = state.board.find_by_id(*piece_id) {
                self.remove_piece_for_card(state, square, piece, dilemma.by, TROLLEY, events);
            }
        }
        self.resolve_missing_king(state, events);
        Ok(())
    }

    fn has_trolley_bundle_pair(&self, state: &GameState, player: Player) -> bool {
        let pieces = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == player && !self.is_royal_piece(state, *piece))
            .map(|(_, piece)| (piece.id, middle_piece_value(piece.kind)))
            .collect::<Vec<_>>();
        let mut by_score = std::collections::BTreeMap::<u8, Vec<Vec<u32>>>::new();
        collect_trolley_bundles(&pieces, 0, &mut Vec::new(), 0, &mut by_score);
        for low_score in 2..=8 {
            for high_score in low_score..=u8::min(10, low_score + 2) {
                let left = by_score.get(&low_score).into_iter().flatten();
                let right = by_score
                    .get(&high_score)
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                for first in left {
                    for second in &right {
                        if (low_score != high_score || first.as_ptr() != second.as_ptr())
                            && first.iter().all(|piece_id| !second.contains(piece_id))
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn sample_middle_trolley_bundle_pair(
        &self,
        state: &mut GameState,
        player: Player,
        events: &mut Vec<Event>,
    ) -> Option<[Vec<u32>; 2]> {
        let pieces = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == player && !self.is_royal_piece(state, *piece))
            .map(|(_, piece)| (piece.id, middle_piece_value(piece.kind)))
            .collect::<Vec<_>>();
        let mut by_score = std::collections::BTreeMap::<u8, Vec<Vec<u32>>>::new();
        collect_trolley_bundles(&pieces, 0, &mut Vec::new(), 0, &mut by_score);
        let mut pairs = Vec::new();
        for low_score in 2..=8 {
            for high_score in low_score..=u8::min(10, low_score + 2) {
                let left = by_score.get(&low_score).into_iter().flatten();
                let right = by_score
                    .get(&high_score)
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                for first in left {
                    for second in &right {
                        if (low_score != high_score || first.as_ptr() != second.as_ptr())
                            && first.iter().all(|piece_id| !second.contains(piece_id))
                        {
                            pairs.push([first.clone(), (*second).clone()]);
                        }
                    }
                }
            }
        }
        if pairs.is_empty() {
            return None;
        }
        let (roll, index) = state.rng_mut().web_choice_index(pairs.len());
        events.push(Event::RandomSample {
            purpose: "trolley-bundles".to_owned(),
            roll_53: roll,
        });
        Some(pairs[index.unwrap_or(0)].clone())
    }

    pub(super) fn resolve_due_premoves_after_turn(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        let pending = std::mem::take(&mut state.card_runtime.pending_premoves);
        for premove in pending {
            if premove.trigger_player != actor
                || state.turns_taken[actor.index()] < premove.trigger_turn
            {
                state.card_runtime.pending_premoves.push(premove);
                continue;
            }
            let previous_turn = state.turn;
            state.turn = premove.owner;
            state.card_runtime.free_move_no_capture[premove.owner.index()] = true;
            for movement in premove.moves {
                let legal = state.board.get(movement.from).is_some_and(|piece| {
                    piece.owner == premove.owner
                        && self
                            .generate_piece_move_actions(state, piece.id)
                            .contains(&Action::Move(movement))
                });
                if !legal {
                    break;
                }
                let Some(mut piece) = state.board.get(movement.from) else {
                    break;
                };
                state.board.set(movement.from, None);
                if let Some(promotion) = movement.promotion {
                    piece.kind = promotion;
                }
                state.board.set(movement.to, Some(piece));
                state.card_runtime.moved_piece_ids.insert(piece.id);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("premove:{}:{}->{}", piece.id, movement.from, movement.to),
                });
            }
            state.turn = previous_turn;
        }
    }

    pub(super) fn resolve_religious_victory(&self, state: &mut GameState) {
        if state.status != GameStatus::Ongoing {
            return;
        }
        for player in [Player::White, Player::Black] {
            if !state.card_runtime.religious_victory[player.index()] {
                continue;
            }
            let own = state
                .board
                .iter()
                .filter(|(_, piece)| {
                    piece.owner == player
                        && matches!(piece.kind, PieceKind::Bishop | PieceKind::Protestant)
                })
                .count();
            let enemy = state
                .board
                .iter()
                .filter(|(_, piece)| {
                    piece.owner == player.opponent()
                        && matches!(piece.kind, PieceKind::Bishop | PieceKind::Protestant)
                })
                .count();
            if own >= enemy.saturating_add(3) {
                state.status = GameStatus::Won {
                    winner: player,
                    reason: WinReason::CardEffect(RELIGIOUS_VICTORY.to_owned()),
                };
                break;
            }
        }
    }
}

const KNIGHT_OFFSETS: &[(i8, i8)] = &[
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];
const BISHOP_DIRECTIONS: &[(i8, i8)] = &[(-1, -1), (-1, 1), (1, -1), (1, 1)];
const ROOK_DIRECTIONS: &[(i8, i8)] = &[(-1, 0), (1, 0), (0, -1), (0, 1)];
const QUEEN_DIRECTIONS: &[(i8, i8)] = &[
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
];

fn collect_trolley_bundles(
    pieces: &[(u32, u8)],
    start: usize,
    selected: &mut Vec<u32>,
    score: u8,
    by_score: &mut std::collections::BTreeMap<u8, Vec<Vec<u32>>>,
) {
    if !selected.is_empty() && (2..=10).contains(&score) {
        let bucket = by_score.entry(score).or_default();
        if bucket.len() < 240 {
            bucket.push(selected.clone());
        }
    }
    if selected.len() >= 4 || score >= 10 {
        return;
    }
    for index in start..pieces.len() {
        let (piece_id, value) = pieces[index];
        if score.saturating_add(value) > 10 {
            continue;
        }
        selected.push(piece_id);
        collect_trolley_bundles(pieces, index + 1, selected, score + value, by_score);
        selected.pop();
    }
}

fn push_square_combinations(
    actions: &mut Vec<Action>,
    instance_id: u32,
    candidates: &[Square],
    maximum: usize,
) {
    fn walk(
        actions: &mut Vec<Action>,
        instance_id: u32,
        candidates: &[Square],
        maximum: usize,
        start: usize,
        selected: &mut Vec<Square>,
    ) {
        if !selected.is_empty() {
            actions.push(Action::UseCard {
                instance_id,
                target: ActionTarget::Squares(selected.clone()),
            });
        }
        if selected.len() >= maximum {
            return;
        }
        for index in start..candidates.len() {
            selected.push(candidates[index]);
            walk(
                actions,
                instance_id,
                candidates,
                maximum,
                index + 1,
                selected,
            );
            selected.pop();
        }
    }
    walk(
        actions,
        instance_id,
        candidates,
        maximum,
        0,
        &mut Vec::new(),
    );
}

fn squares_target(target: &ActionTarget) -> Result<Vec<Square>, EngineError> {
    match target {
        ActionTarget::Squares(squares) if !squares.is_empty() => Ok(squares.clone()),
        ActionTarget::Square(square) | ActionTarget::PieceAt(square) => Ok(vec![*square]),
        _ => Err(EngineError::Invariant(
            "card requires one or more squares".to_owned(),
        )),
    }
}

fn is_ranged_piece(kind: PieceKind) -> bool {
    matches!(
        kind,
        PieceKind::Bishop
            | PieceKind::Rook
            | PieceKind::Queen
            | PieceKind::Bear
            | PieceKind::Amazon
            | PieceKind::Cardinal
            | PieceKind::Cannon
            | PieceKind::Herald
            | PieceKind::Hook
            | PieceKind::Protestant
            | PieceKind::Jester
            | PieceKind::Windmill
            | PieceKind::BigRook
            | PieceKind::Idol
    )
}

pub(super) fn suspicious_potion_effects(
    royal: bool,
    kind: PieceKind,
    ranged: bool,
) -> Vec<&'static str> {
    [
        ("sacrifice-protection", !royal),
        ("last-resistance", royal),
        ("coronation-protection", true),
        ("shield", true),
        ("evasion", true),
        ("parry", true),
        ("basic-training", !royal && kind != PieceKind::Pawn),
        ("stealth", true),
        ("loyalist", !royal),
        ("ghost", true),
        ("chameleon", !royal),
        ("submerge", true),
        ("freeze", true),
        ("chimera", !royal),
        ("witch-trial", !royal),
        ("stake", true),
        ("calling-card", !royal),
        ("empty-lunchbox", !royal),
        ("poison-stun", true),
        ("disarm", true),
        ("manner-no-capture", true),
        ("saturation-no-capture", true),
        ("explosive", kind == PieceKind::Pawn),
        ("poisoned-pawn", kind == PieceKind::Pawn),
        ("queens-gambit-protection", kind == PieceKind::Pawn),
        ("trojan-horse", kind == PieceKind::Knight),
        ("severance", ranged),
        ("inertia", ranged),
    ]
    .into_iter()
    .filter_map(|(effect, eligible)| eligible.then_some(effect))
    .collect()
}

fn is_major_piece(kind: PieceKind) -> bool {
    matches!(
        kind,
        PieceKind::Queen
            | PieceKind::Rook
            | PieceKind::Amazon
            | PieceKind::Man
            | PieceKind::Colossus
            | PieceKind::BigRook
            | PieceKind::Herald
            | PieceKind::Jester
            | PieceKind::Recruiter
            | PieceKind::Hook
            | PieceKind::PrimeMinister
            | PieceKind::Assassin
            | PieceKind::Wizard
            | PieceKind::Windmill
            | PieceKind::Bear
    )
}

fn has_piece_protection_source(state: &GameState, piece_id: u32) -> bool {
    state
        .card_runtime
        .sacrifice_protection
        .contains_key(&piece_id)
        || state.card_runtime.last_resistance.contains_key(&piece_id)
        || state
            .card_runtime
            .coronation_protection
            .contains_key(&piece_id)
        || state
            .card_runtime
            .queens_gambit_protection
            .contains(&piece_id)
}

fn middle_piece_value(kind: PieceKind) -> u8 {
    match kind {
        PieceKind::Pawn
        | PieceKind::Squire
        | PieceKind::Fanatic
        | PieceKind::Scarecrow
        | PieceKind::BabyBear => 1,
        PieceKind::Knight
        | PieceKind::Bishop
        | PieceKind::Alfil
        | PieceKind::Guard
        | PieceKind::Checker
        | PieceKind::CheckerKing
        | PieceKind::Man
        | PieceKind::Camel
        | PieceKind::Bat => 3,
        PieceKind::Ferz | PieceKind::Lobster => 2,
        PieceKind::Rook
        | PieceKind::Cardinal
        | PieceKind::Protestant
        | PieceKind::Windmill
        | PieceKind::Cannon
        | PieceKind::Herald => 5,
        PieceKind::Queen
        | PieceKind::Amazon
        | PieceKind::PrimeMinister
        | PieceKind::Dragon
        | PieceKind::Wizard
        | PieceKind::Jester
        | PieceKind::Hook
        | PieceKind::Bear => 9,
        PieceKind::King | PieceKind::RoyalKnight | PieceKind::Merchant | PieceKind::DarkWizard => {
            10
        }
        PieceKind::BigRook | PieceKind::Colossus => 12,
        _ => 4,
    }
}

fn aligned_distance(from: Square, to: Square) -> Option<(i8, i8, u8)> {
    let file = to.file() as i8 - from.file() as i8;
    let rank = to.rank() as i8 - from.rank() as i8;
    if file == 0 && rank != 0 {
        Some((0, rank.signum(), rank.unsigned_abs()))
    } else if rank == 0 && file != 0 {
        Some((file.signum(), 0, file.unsigned_abs()))
    } else if file.unsigned_abs() == rank.unsigned_abs() {
        Some((file.signum(), rank.signum(), file.unsigned_abs()))
    } else {
        None
    }
}

fn has_adjacent_enemy(state: &GameState, square: Square, owner: Player) -> bool {
    KING_OFFSETS.iter().any(|(file, rank)| {
        square
            .offset(*file, *rank)
            .and_then(|target| state.board.get(target))
            .is_some_and(|piece| piece.owner != owner)
    })
}

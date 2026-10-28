//! Shared placement and spatial primitives, independent of card dispatch.
use super::*;

impl Engine {
    pub(in crate::engine) fn install_card_layout(
        &self,
        state: &mut GameState,
        owner: Player,
        layout: &[(Square, PieceKind, bool)],
        footprint: &[Square],
        cause: &str,
        events: &mut Vec<Event>,
    ) {
        let targets: BTreeSet<_> = layout
            .iter()
            .map(|(s, _, _)| *s)
            .chain(footprint.iter().copied())
            .collect();
        let removed: Vec<_> = state
            .board
            .iter()
            .filter(|(square, piece)| {
                (cause == LONDON_SYSTEM && piece.owner == owner)
                    || targets.contains(square)
                    || state
                        .card_runtime
                        .large_pieces
                        .get(&piece.id)
                        .is_some_and(|large| {
                            large_piece_cells(large.anchor)
                                .iter()
                                .any(|cell| targets.contains(cell))
                        })
            })
            .collect();
        for (square, piece) in removed {
            self.delete_piece_for_card(state, square, piece, owner, cause, events);
        }
        for &(square, kind, moved) in layout {
            let piece = Piece::new(owner, kind).with_id(state.allocate_piece_id());
            state.board.set(square, Some(piece));
            state.card_runtime.piece_origins.insert(piece.id, square);
            if moved {
                state.card_runtime.moved_piece_ids.insert(piece.id);
            }
            // openingPassiveSkipsFreshCapture: London/BISHOP can capture immediately.
            if kind == PieceKind::BigBishop {
                state.card_runtime.large_pieces.insert(
                    piece.id,
                    LargePieceStatus {
                        anchor: square,
                        hp: 2,
                        max_hp: 2,
                    },
                );
            }
        }
        state.en_passant = None;
        let alive = |color| {
            state
                .board
                .iter()
                .any(|(_, p)| p.owner == color && self.is_royal_piece(state, p))
        };
        if !alive(Player::White) && !alive(Player::Black) {
            state.status = GameStatus::Draw {
                reason: DrawReason::CardEffect(cause.into()),
            };
        }
    }
}

impl Engine {
    pub(in crate::engine) fn card_landing_open(&self, state: &GameState, square: Square) -> bool {
        self.board_piece_at(state, square).is_none()
            && !state.card_runtime.barricades.contains(&square)
            && !state.card_runtime.black_holes.contains(&square)
            && state.card_runtime.crown_pos != Some(square)
            && !(state.card_runtime.collapsed_files_ranks
                && (matches!(square.file(), 0 | 7) || matches!(square.rank(), 0 | 7)))
            && !state
                .card_runtime
                .quantum_positions
                .values()
                .any(|s| *s == square)
            && !state
                .card_runtime
                .pending_lobsters
                .iter()
                .any(|p| p.square == square)
            && !state
                .card_runtime
                .pending_scarecrows
                .iter()
                .any(|p| p.square == square)
            && !state
                .card_runtime
                .pending_portals
                .iter()
                .any(|p| p.squares.contains(&square))
            && !state
                .card_runtime
                .portals
                .is_some_and(|p| p.contains(&square))
            && !state
                .card_runtime
                .rule_portals
                .is_some_and(|p| p.contains(&square))
    }
}

impl Engine {
    pub(in crate::engine) fn collapsed_card_landing(
        &self,
        state: &GameState,
        square: Square,
    ) -> bool {
        state.card_runtime.collapsed_files_ranks
            && (matches!(square.file(), 0 | 7) || matches!(square.rank(), 0 | 7))
    }
}

impl Engine {
    pub(in crate::engine) fn crush_card_landing(
        &self,
        state: &mut GameState,
        square: Square,
        owner: Player,
        cause: &str,
        events: &mut Vec<Event>,
    ) {
        if let Some((anchor, piece)) = self.board_piece_entry_at(state, square) {
            // Reuse the current core's large-body hit abstraction. Cell fracture
            // parity remains outside these scoped ports, as for other landings.
            if let Some(large) = state.card_runtime.large_pieces.get_mut(&piece.id) {
                large.hp = large.hp.saturating_sub(1);
                if large.hp > 0 {
                    return;
                }
            }
            self.remove_piece_for_card(state, anchor, piece, owner, cause, events);
        }
    }
}

impl Engine {
    pub(in crate::engine) fn card_royal_checked(
        &self,
        state: &GameState,
        attacker: Player,
    ) -> bool {
        // v6OpponentKingAttacked uses the first royal in web board order. Keep
        // its actual identity and immunities; disable castling in attack probes.
        let Some((target, royal)) = web_ordered_pieces(state)
            .into_iter()
            .find(|(_, p)| p.owner == attacker.opponent() && self.is_royal_piece(state, *p))
        else {
            return false;
        };
        let mut probe = state.clone();
        probe.card_runtime.moved_piece_ids.extend(
            state
                .board
                .iter()
                .filter(|(_, p)| self.is_royal_piece(state, *p))
                .map(|(_, p)| p.id),
        );
        for (from, piece) in probe.board.iter().filter(|(_, p)| p.owner == attacker) {
            let mut actions = Vec::new();
            self.generate_moves_for_piece(&probe, from, piece, &mut actions);
            if actions.into_iter().any(|a| {
                if let Action::Move(m) = a {
                    m.to == target
                        || (matches!(piece.kind, PieceKind::Checker | PieceKind::CheckerKing)
                            && self
                                .checker_jump_capture(&probe, from, m.to, piece)
                                .is_some_and(|(_, p)| p.id == royal.id))
                        || (probe.card_runtime.large_pieces.contains_key(&piece.id)
                            && self
                                .large_landing_targets(&probe, piece, m.to)
                                .iter()
                                .any(|(_, p)| p.id == royal.id))
                } else {
                    false
                }
            }) {
                return true;
            }
        }
        false
    }
}

impl Engine {
    pub(in crate::engine) fn filter_card_state_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        self.current_rule_move_filter(state, from, piece, actions);

        if state.ruleset_id != crate::LATEST_RULESET_ID {
            return;
        }
        let rt = &state.card_runtime.october;
        if rt.grudge_losses.is_empty() && rt.pawn_chain_links.is_empty() {
            return;
        }
        actions.retain(|a| {
            let Action::Move(m) = a else {
                return true;
            };
            let ally = state
                .board
                .get(m.to)
                .filter(|p| p.owner == piece.owner && p.id != piece.id);
            let swapping = ally.is_some()
                && (self.piece_has_ability(state, piece, PieceKind::Dragon)
                    || state.card_runtime.relay[piece.owner.index()]
                    || state.card_runtime.substitution[piece.owner.index()]
                    || (self.is_royal_piece(state, piece)
                        && state.card_runtime.switcheroo[piece.owner.index()]));
            let castling = self.standard_castling_rook_plan(state, piece, *m);
            let df = from.file().abs_diff(m.to.file());
            let dr = from.rank().abs_diff(m.to.rank());
            let stationary = from == m.to;
            let component = match self.piece_ability_kind(state, piece) {
                PieceKind::Hook | PieceKind::Brutus => 1,
                PieceKind::Cardinal | PieceKind::Protestant => 2,
                _ => {
                    if df == 0 || dr == 0 {
                        1
                    } else if df == dr {
                        2
                    } else if u16::from(df) * u16::from(dr) == 2 {
                        4
                    } else if u16::from(df) * u16::from(dr) == 3 {
                        8
                    } else {
                        0
                    }
                }
            };
            if !swapping
                && !stationary
                && rt
                    .grudge_losses
                    .get(&piece.id)
                    .is_some_and(|loss| loss & component != 0)
            {
                return false;
            }
            for pair in &rt.pawn_chain_links {
                let Some((a, _)) = state.board.find_by_id(pair[0]) else {
                    continue;
                };
                let Some((b, _)) = state.board.find_by_id(pair[1]) else {
                    continue;
                };
                if distance(a, b) > 2 {
                    continue;
                }
                let destination = |id, old| {
                    if id == piece.id {
                        m.to
                    } else if swapping && ally.is_some_and(|p| p.id == id) {
                        from
                    } else if let Some((_, to, rook)) = castling
                        && rook.id == id
                    {
                        to
                    } else {
                        old
                    }
                };
                if distance(destination(pair[0], a), destination(pair[1], b)) > 2 {
                    return false;
                }
            }
            true
        });
    }
}

pub(super) fn distance(a: Square, b: Square) -> u8 {
    a.file().abs_diff(b.file()).max(a.rank().abs_diff(b.rank()))
}

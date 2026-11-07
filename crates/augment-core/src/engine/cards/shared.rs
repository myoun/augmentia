//! Shared placement and spatial primitives, independent of card dispatch.
use super::*;

impl Engine {
    /// Returns whether a fractured body was defeated, or None for other pieces.
    pub(in crate::engine) fn fracture_large_cell(
        &self,
        state: &mut GameState,
        piece: Piece,
        square: Square,
    ) -> Option<bool> {
        if state.ruleset_id != crate::LATEST_RULESET_ID
            || !matches!(piece.kind, PieceKind::BigRook | PieceKind::BigBishop)
        {
            return None;
        }
        let body = state.card_runtime.piece_status.get(&piece.id).copied()?;
        let anchor = body.large?.anchor;
        let index = large_piece_cells(anchor)
            .iter()
            .position(|cell| *cell == square)? as u8;
        if state
            .card_runtime
            .large_missing_cells
            .contains_key(&piece.id)
        {
            return Some(true);
        }
        state
            .card_runtime
            .large_missing_cells
            .insert(piece.id, index);
        // Losing a cell leaves the body alive but brittle: the footprint stays and
        // durability drops to its last hit.
        if let Some(health) = state.card_runtime.health_mut(piece.id) {
            health.hp = 1;
        }
        if let Some((storage, _)) = state.board.find_by_id(piece.id) {
            state.board.set(storage, None);
        }
        let storage = large_piece_occupied_cells(state, piece.id, anchor)[0];
        state.board.set(storage, Some(piece));
        Some(false)
    }

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
                        .large_anchor(piece.id)
                        .is_some_and(|anchor| {
                            large_piece_cells(anchor)
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
            if matches!(kind, PieceKind::BigBishop | PieceKind::BigRook) {
                state
                    .card_runtime
                    .insert_large_body(piece.id, square, PieceHealth::new(2));
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
            if let Some(defeated) = self.fracture_large_cell(state, piece, square) {
                if defeated {
                    self.remove_piece_for_card(state, anchor, piece, owner, cause, events);
                }
                return;
            }
            // Reuse the current core's durability abstraction. Cell fracture parity
            // remains outside these scoped ports, as for other landings. A piece
            // without durability is removed outright; one with durability survives
            // until its last hit.
            if state
                .card_runtime
                .health_mut(piece.id)
                .is_some_and(|health| !health.take_hit())
            {
                return;
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
                        || (probe.card_runtime.piece_status.contains_key(&piece.id)
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
        actions.retain(|action| match action {
            Action::Move(movement) if movement.to != from => {
                !state.card_runtime.barricades.contains(&movement.to)
            }
            _ => true,
        });
        // A royal already inside its own palace may not step out of it.
        if state.card_runtime.palaces.contains_key(&piece.owner) {
            actions.retain(|action| match action {
                Action::Move(movement) => !self.palace_blocks_move(state, piece, from, movement.to),
                _ => true,
            });
        }
        // Chain bonds break when the pair separates beyond Chebyshev distance two.
        if !state.card_runtime.chains.is_empty() {
            actions.retain(|action| match action {
                Action::Move(movement) => {
                    super::chain::chain_allows_move(state, piece.id, movement.to)
                }
                _ => true,
            });
        }
        // Synchronization bans enemy arrivals on the opposite square colour while
        // every friendly piece shares one colour.
        if state
            .cards
            .iter()
            .any(|c| c.definition_id == crate::SYNCHRONIZATION)
        {
            actions.retain(|action| match action {
                Action::Move(movement) => {
                    !self.synchronization_blocks_arrival(state, piece, movement.to)
                }
                _ => true,
            });
        }
        if self.piece_ability_kind(state, piece) == PieceKind::Thief
            && let Some((visited, df, dr)) = state.card_runtime.thief_turn_paths.get(&piece.id)
        {
            actions.retain(|action| match action {
                Action::Move(m) => {
                    let route = self.rule_portal_route(state, from, piece, m.to);
                    let entry = route.map_or(m.to, |r| r.entry);
                    let landing = route.map_or(m.to, |r| r.landing);
                    let reverse = (
                        (from.file() as i8 - entry.file() as i8).signum(),
                        (from.rank() as i8 - entry.rank() as i8).signum(),
                    );
                    reverse != (*df, *dr)
                        && !visited.contains(&entry)
                        && !visited.contains(&landing)
                }
                _ => true,
            });
        }
        if state
            .cards
            .iter()
            .any(|card| card.definition_id == crate::D4)
        {
            actions.retain(|action| match action {
                Action::Move(m) => {
                    let cells = if state.card_runtime.is_large(piece.id) {
                        large_piece_occupied_cells(state, piece.id, m.to)
                    } else {
                        vec![m.to]
                    };
                    !cells.into_iter().any(|s| {
                        self.d4_restricts(state, piece.owner, self.is_royal_piece(state, piece), s)
                    })
                }
                _ => true,
            });
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

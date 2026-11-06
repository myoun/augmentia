use super::*;
pub(super) const ID: &str = crate::CONVEYOR;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_conveyor(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

pub(super) fn ring() -> Vec<Square> {
    (0..8)
        .map(|f| Square::new(f, 7).unwrap())
        .chain((0..7).rev().map(|r| Square::new(7, r).unwrap()))
        .chain((0..7).rev().map(|f| Square::new(f, 0).unwrap()))
        .chain((1..7).map(|r| Square::new(0, r).unwrap()))
        .collect()
}
impl Engine {
    pub(in crate::engine) fn settle_rule_conveyor(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) {
        if !rule_active(state, ID) {
            return;
        }
        self.rotate_rule_ring(state, &ring(), events);
    }
    pub(in crate::engine) fn rotate_rule_ring(
        &self,
        state: &mut GameState,
        ring: &[Square],
        events: &mut Vec<Event>,
    ) {
        self.rotate_conveyor(state, ring, false, events);
    }

    pub(in crate::engine) fn rotate_conveyor(
        &self,
        state: &mut GameState,
        ring: &[Square],
        linear: bool,
        events: &mut Vec<Event>,
    ) {
        let n = ring.len();
        if n == 0 {
            return;
        }
        let pieces: Vec<_> = ring.iter().map(|s| state.board.get(*s)).collect();
        let walls: Vec<_> = ring
            .iter()
            .map(|s| state.card_runtime.barricades.contains(s))
            .collect();
        let blockade_walls: Vec<_> = ring
            .iter()
            .map(|square| state.card_runtime.blockade_walls.get(square).copied())
            .collect();
        let balls: Vec<_> = ring
            .iter()
            .map(|s| state.card_runtime.football_pos == Some(*s))
            .collect();
        let fixed: Vec<_> = ring
            .iter()
            .enumerate()
            .map(|(i, s)| {
                self.board_piece_at(state, *s).is_some_and(|p| {
                    state.card_runtime.large_pieces.contains_key(&p.id)
                        || p.kind == PieceKind::Slime
                        || (state.ruleset_id == crate::LATEST_RULESET_ID
                            && self.piece_has_ability(state, p, PieceKind::Slime))
                        || (linear
                            && i + 1 < n
                            && self.d4_restricts(
                                state,
                                p.owner,
                                self.is_royal_piece(state, p),
                                ring[i + 1],
                            ))
                }) || pieces[i].is_some() && (walls[i] || balls[i])
                    || (linear && walls[i] && balls[i])
            })
            .collect();
        let occupied: Vec<_> = (0..n)
            .map(|i| pieces[i].is_some() || walls[i] || balls[i])
            .collect();
        let mut moving: Vec<_> = (0..n)
            .map(|i| {
                !linear
                    && occupied[i]
                    && !fixed[i]
                    && !fixed[(i + 1) % n]
                    && (!fixed.iter().any(|f| *f) || !occupied[(i + 1) % n])
            })
            .collect();
        if linear {
            for i in (0..n.saturating_sub(1)).rev() {
                moving[i] = occupied[i]
                    && !fixed[i]
                    && !fixed[i + 1]
                    && (!occupied[i + 1] || moving[i + 1]);
            }
        }
        if !linear {
            loop {
                let mut changed = false;
                for i in 0..n {
                    if occupied[i]
                        && !fixed[i]
                        && !fixed[(i + 1) % n]
                        && !moving[i]
                        && moving[(i + 1) % n]
                    {
                        moving[i] = true;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        for i in 0..n {
            if moving[i] {
                state.board.set(ring[i], None);
                state.card_runtime.remove_barricade(ring[i]);
                if balls[i] {
                    state.card_runtime.football_pos = None;
                }
            }
        }
        for i in 0..n {
            if !moving[i] {
                continue;
            }
            let to = ring[(i + 1) % n];
            if let Some(mut p) = pieces[i] {
                state.card_runtime.moved_piece_ids.insert(p.id);
                if (matches!(
                    p.kind,
                    PieceKind::Pawn | PieceKind::Squire | PieceKind::StandardBearer
                ) || (!linear && p.kind == PieceKind::BerolinaPawn))
                    && to.rank() == if p.owner == Player::White { 7 } else { 0 }
                {
                    p.kind = PieceKind::Queen;
                }
                state.board.set(to, Some(p));
            }
            if walls[i] && !self.rule_cell_collapsed(state, to) {
                state.card_runtime.barricades.insert(to);
                if let Some(wall) = blockade_walls[i] {
                    state.card_runtime.blockade_walls.insert(to, wall);
                }
            }
            if balls[i] && !self.rule_cell_collapsed(state, to) {
                state.card_runtime.football_pos = Some(to);
            }
        }
        let shift = |square: Square| {
            ring.iter()
                .position(|s| *s == square)
                .map_or(square, |i| ring[(i + 1) % n])
        };
        for pending in &mut state.card_runtime.pending_lobsters {
            pending.square = shift(pending.square);
        }
        for pending in &mut state.card_runtime.pending_scarecrows {
            pending.square = shift(pending.square);
        }
        if let Some(mut ep) = state.en_passant {
            if let Some(p) = pieces.iter().flatten().find(|p| {
                p.owner == ep.pawn_owner
                    && state.card_runtime.piece_origins.contains_key(&p.id)
                    && pieces
                        .iter()
                        .position(|x| x.is_some_and(|x| x.id == p.id))
                        .is_some_and(|i| ring[i] == ep.captured)
            }) {
                if let Some((square, _)) = state.board.find_by_id(p.id) {
                    ep.captured = square;
                    if let Some(target) = square.offset(
                        0,
                        if ep.pawn_owner == Player::White {
                            -1
                        } else {
                            1
                        },
                    ) {
                        ep.target = target;
                        state.en_passant = Some(ep);
                    } else {
                        state.en_passant = None;
                    }
                }
            }
        }
        let fallen: Vec<_> = state
            .board
            .iter()
            .filter(|(s, _)| self.rule_cell_collapsed(state, *s))
            .collect();
        let memories =
            self.rule_capture_memories(state, &fallen.iter().map(|(_, p)| *p).collect::<Vec<_>>());
        for (s, p) in fallen {
            self.remove_piece_for_card(state, s, p, state.turn, ID, events);
        }
        self.settle_rule_capture_memories(state, &memories, events);
        self.current_rule_arrivals(state, events);
    }
}

impl Engine {
    pub(in crate::engine) fn install_conveyor(
        &self,
        state: &mut GameState,
        _events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        state.card_runtime.conveyor = true;
        Ok(())
    }
}

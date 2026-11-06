//! Piece behavior required by the current 65-type chaos setup.
use super::*;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, VecDeque};

const KNIGHT: &[(i8, i8)] = &[
    (-1, 2),
    (1, 2),
    (-2, 1),
    (2, 1),
    (-2, -1),
    (2, -1),
    (-1, -2),
    (1, -2),
];
fn row_index(s: Square) -> usize {
    usize::from(7 - s.rank()) * 8 + usize::from(s.file())
}
fn square_at(i: usize) -> Square {
    Square::new((i % 8) as u8, 7 - (i / 8) as u8).unwrap()
}

/// Source shortest route uses an admissible minimum-spanning-tree bound and a
/// greedy incumbent. Large armies use the source's nearest-target BFS route.
pub(super) fn don_path(
    from: Square,
    targets: &[Square],
    blocked: &BTreeSet<Square>,
    fast: bool,
) -> Option<Vec<Square>> {
    let start = row_index(from);
    let targets: Vec<_> = targets.iter().map(|s| row_index(*s)).collect();
    if targets.iter().any(|i| blocked.contains(&square_at(*i))) {
        return None;
    }
    let neighbors: Vec<Vec<usize>> = (0..64)
        .map(|i| {
            let s = square_at(i);
            if blocked.contains(&s) && s != from {
                return Vec::new();
            }
            KNIGHT
                .iter()
                .filter_map(|(df, dr)| s.offset(*df, *dr))
                .filter(|s| !blocked.contains(s) || *s == from)
                .map(row_index)
                .collect()
        })
        .collect();
    let mut distance = [[u16::MAX; 64]; 64];
    let mut previous = [[usize::MAX; 64]; 64];
    for source in 0..64 {
        let mut queue = VecDeque::from([source]);
        distance[source][source] = 0;
        while let Some(p) = queue.pop_front() {
            for &n in &neighbors[p] {
                if distance[source][n] == u16::MAX {
                    distance[source][n] = distance[source][p] + 1;
                    previous[source][n] = p;
                    queue.push_back(n);
                }
            }
        }
    }
    if targets.iter().any(|t| distance[start][*t] == u16::MAX) {
        return None;
    }
    let mut bits = [0u64; 64];
    for (i, t) in targets.iter().enumerate() {
        bits[*t] = 1u64 << i;
    }
    let complete = (1u64 << targets.len()) - 1;
    let segment = |a: usize, b: usize| {
        let mut path = Vec::new();
        let mut p = b;
        while p != a {
            path.push(p);
            p = previous[a][p];
        }
        path.reverse();
        path
    };
    let mut mask = bits[start];
    let mut position = start;
    let mut incumbent = Vec::new();
    while mask != complete {
        let next = if fast {
            let mut queue = VecDeque::from([position]);
            let mut seen = BTreeSet::from([position]);
            let mut found = None;
            'search: while let Some(p) = queue.pop_front() {
                for n in &neighbors[p] {
                    if seen.insert(*n) {
                        if bits[*n] & !mask != 0 {
                            found = Some(*n);
                            break 'search;
                        }
                        queue.push_back(*n);
                    }
                }
            }
            found?
        } else {
            *targets
                .iter()
                .filter(|t| bits[**t] & !mask != 0)
                .min_by_key(|t| (distance[position][**t], **t))
                .unwrap()
        };
        for step in segment(position, next) {
            incumbent.push(step);
            mask |= bits[step];
        }
        position = next;
    }
    if fast {
        return Some(incumbent.into_iter().map(square_at).collect());
    }
    let lower_bound = |position: usize, mask: u64| -> u16 {
        let unseen: Vec<_> = targets
            .iter()
            .copied()
            .filter(|t| bits[*t] & !mask != 0)
            .collect();
        if unseen.is_empty() {
            return 0;
        }
        let mut remaining: BTreeMap<usize, u16> = unseen[1..]
            .iter()
            .map(|t| (*t, distance[unseen[0]][*t]))
            .collect();
        let mut tree = 0;
        while !remaining.is_empty() {
            let (&chosen, &cost) = remaining.iter().min_by_key(|(i, c)| (**c, **i)).unwrap();
            tree += cost;
            remaining.remove(&chosen);
            for (i, c) in &mut remaining {
                *c = (*c).min(distance[chosen][*i]);
            }
        }
        tree + unseen.iter().map(|t| distance[position][*t]).min().unwrap()
    };
    #[derive(Clone, Copy)]
    struct Node {
        position: usize,
        mask: u64,
        g: u16,
        parent: Option<usize>,
    }
    let initial = bits[start];
    let h = lower_bound(start, initial);
    if usize::from(h) >= incumbent.len() {
        return Some(incumbent.into_iter().map(square_at).collect());
    }
    let mut nodes = vec![Node {
        position: start,
        mask: initial,
        g: 0,
        parent: None,
    }];
    let mut heap = BinaryHeap::from([Reverse((h, h, 0usize))]);
    let mut best = BTreeMap::from([((start, initial), 0)]);
    while let Some(Reverse((f, _, index))) = heap.pop() {
        let current = nodes[index];
        if best.get(&(current.position, current.mask)) != Some(&current.g) {
            continue;
        }
        if usize::from(f) >= incumbent.len() {
            break;
        }
        if current.mask == complete {
            let mut path = Vec::new();
            let mut i = index;
            while let Some(parent) = nodes[i].parent {
                path.push(square_at(nodes[i].position));
                i = parent;
            }
            path.reverse();
            return Some(path);
        }
        for position in &neighbors[current.position] {
            let mask = current.mask | bits[*position];
            let g = current.g + 1;
            if best.get(&(*position, mask)).is_some_and(|old| *old <= g) {
                continue;
            }
            let h = lower_bound(*position, mask);
            if usize::from(g + h) >= incumbent.len() {
                continue;
            }
            best.insert((*position, mask), g);
            let child_index = nodes.len();
            nodes.push(Node {
                position: *position,
                mask,
                g,
                parent: Some(index),
            });
            heap.push(Reverse((g + h, h, child_index)));
        }
    }
    Some(incumbent.into_iter().map(square_at).collect())
}

impl Engine {
    pub(in crate::engine) fn rule_capture_memories(
        &self,
        state: &GameState,
        victims: &[Piece],
    ) -> Vec<(Option<ParrotMemoryStatus>, Option<WindmillMode>)> {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            return Vec::new();
        }
        victims
            .iter()
            .filter_map(|p| {
                let kind = match p.kind {
                    PieceKind::Colossus | PieceKind::Football | PieceKind::Monster => return None,
                    PieceKind::Medium => {
                        return Some((
                            state.rule_runtime.medium_memory,
                            state.rule_runtime.medium_windmill_mode,
                        ));
                    }
                    PieceKind::Parrot => {
                        return Some((state.card_runtime.parrot_memory[p.owner.index()], None));
                    }
                    PieceKind::BigRook => PieceKind::Rook,
                    PieceKind::BigBishop => PieceKind::Bishop,
                    PieceKind::Trickster => self.piece_ability_kind(state, *p),
                    kind => kind,
                };
                let mode = (kind == PieceKind::Windmill).then(|| {
                    state
                        .card_runtime
                        .windmill_modes
                        .get(&p.id)
                        .copied()
                        .unwrap_or(WindmillMode::Bishop)
                });
                Some((
                    Some(ParrotMemoryStatus {
                        source: kind,
                        effective: kind,
                    }),
                    mode,
                ))
            })
            .collect()
    }
    pub(in crate::engine) fn settle_rule_capture_memories(
        &self,
        state: &mut GameState,
        choices: &[(Option<ParrotMemoryStatus>, Option<WindmillMode>)],
        events: &mut Vec<Event>,
    ) {
        let index = match choices.len() {
            0 => return,
            1 => 0,
            n => rule_random(state, events, n, "rule.medium.capture-batch").unwrap(),
        };
        (
            state.rule_runtime.medium_memory,
            state.rule_runtime.medium_windmill_mode,
        ) = choices[index];
    }

    pub(in crate::engine) fn rule_berolina_capture(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        to: Square,
    ) -> Option<(Square, Piece)> {
        let kind = self.piece_ability_kind(state, piece);
        if !matches!(kind, PieceKind::Pawn | PieceKind::BerolinaPawn)
            || to.rank() as i8 - from.rank() as i8
                != if piece.owner == Player::White { 1 } else { -1 }
            || from.file().abs_diff(to.file()) != u8::from(kind == PieceKind::Pawn)
            || self.board_piece_at(state, to).is_some()
        {
            return None;
        }
        state
            .rule_runtime
            .berolina_rights
            .iter()
            .find_map(|(id, (owner, square, targets))| {
                (*owner != piece.owner && targets.contains(&to))
                    .then(|| state.board.find_by_id(*id))
                    .flatten()
                    .filter(|(s, p)| s == square && self.can_capture(state, piece, *p))
            })
    }

    pub(in crate::engine) fn current_rule_dependency_capture(
        &self,
        state: &mut GameState,
        captured: Piece,
    ) {
        if let Some((memory, mode)) = self
            .rule_capture_memories(state, &[captured])
            .into_iter()
            .next()
        {
            state.rule_runtime.medium_memory = memory;
            state.rule_runtime.medium_windmill_mode = mode;
        }
    }

    pub(in crate::engine) fn rule_grappler_pull(
        &self,
        state: &GameState,
        movement: MoveAction,
    ) -> Option<(Square, Piece)> {
        let piece = state.move_source_piece(movement.from)?;
        if !self.piece_has_ability(state, piece, PieceKind::Grappler) {
            return None;
        }
        let target = self.board_piece_at(state, movement.to)?;
        if target.owner == piece.owner {
            return None;
        }
        let df = movement.to.file() as i8 - movement.from.file() as i8;
        let dr = movement.to.rank() as i8 - movement.from.rank() as i8;
        if df.abs().max(dr.abs()) < 2 || df != 0 && dr != 0 && df.abs() != dr.abs() {
            return None;
        }
        let to = movement.from.offset(df.signum(), dr.signum())?;
        if !self.card_landing_open(state, to)
            || self.rule_cell_collapsed(state, to)
            || target.kind == PieceKind::Slime
        {
            return None;
        }
        if state.card_runtime.large_pieces.contains_key(&target.id) {
            let cells = large_piece_cells(to);
            if cells.len() != 4
                || cells.iter().any(|s| {
                    self.rule_cell_collapsed(state, *s)
                        || self
                            .board_piece_at(state, *s)
                            .is_some_and(|p| p.id != target.id)
                        || state.card_runtime.barricades.contains(s)
                        || state.card_runtime.black_holes.contains(s)
                })
            {
                return None;
            }
        }
        let mut cursor = to;
        while cursor != movement.to {
            if self.board_piece_at(state, cursor).is_some() {
                return None;
            }
            cursor = cursor.offset(df.signum(), dr.signum())?;
        }
        Some((to, target))
    }

    pub(in crate::engine) fn apply_rule_grappler(
        &self,
        state: &GameState,
        action: Action,
        movement: MoveAction,
        to: Square,
        target: Piece,
    ) -> Result<Transition, EngineError> {
        let mut next = state.clone();
        let actor = state.turn;
        let caster = state.move_source_piece(movement.from).unwrap();
        let mut events = vec![Event::ActionApplied { actor, action }];
        let (anchor, _) = self.board_piece_entry_at(state, movement.to).unwrap();
        next.board.set(anchor, None);
        next.board.set(to, Some(target));
        next.card_runtime.moved_piece_ids.insert(target.id);
        if let Some(large) = next.card_runtime.large_pieces.get_mut(&target.id) {
            large.anchor = to;
        }
        next.rule_runtime
            .grappler_bound
            .insert(caster.id, actor.opponent());
        next.rule_runtime
            .grappler_bound
            .insert(target.id, actor.opponent());
        next.en_passant = None;
        next.ply = next.ply.saturating_add(1);
        next.actions_remaining = 0;
        events.push(Event::EffectApplied {
            source_card_instance: 0,
            command: format!("grappler-pull:{}:{}:{to}", target.id, movement.to),
        });
        self.current_rule_arrivals(&mut next, &mut events);
        self.finish_turn(&mut next, actor, &mut events)?;
        Ok(Transition {
            state: next,
            events,
        })
    }

    pub(in crate::engine) fn current_rule_dependency_end(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        if !rule_dependencies_active(state) {
            return;
        }
        state
            .rule_runtime
            .grappler_bound
            .retain(|_, until| *until != actor);
        let killers: Vec<_> = web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, p)| {
                p.owner == actor && self.piece_has_ability(state, *p, PieceKind::Killer)
            })
            .collect();
        for (from, piece) in killers {
            if let Some((to, _)) = web_ordered_pieces(state)
                .into_iter()
                .find(|(_, p)| p.owner != actor && self.is_royal_piece(state, *p))
            {
                let mut probe = state.clone();
                probe.turn = actor;
                let mut moves = Vec::new();
                self.generate_moves_for_piece(&probe, from, piece, &mut moves);
                let action = Action::Move(MoveAction::new(from, to, None));
                if moves.contains(&action) {
                    self.rule_automatic_step(state, actor, action, events);
                }
            }
            if state.status != GameStatus::Ongoing {
                return;
            }
        }
        let doors: Vec<_> = web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, p)| {
                p.owner == actor && self.piece_has_ability(state, *p, PieceKind::RevolvingDoor)
            })
            .collect();
        for (_, piece) in doors {
            let Some((from, _)) = state.board.find_by_id(piece.id) else {
                continue;
            };
            let ring: Option<Vec<_>> = [
                (-1, 1),
                (0, 1),
                (1, 1),
                (1, 0),
                (1, -1),
                (0, -1),
                (-1, -1),
                (-1, 0),
            ]
            .into_iter()
            .map(|(df, dr)| from.offset(df, dr))
            .collect();
            if let Some(ring) = ring {
                self.rotate_rule_ring(state, &ring, events);
            }
        }
        let beacons: Vec<_> = web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, p)| {
                p.owner == actor
                    && self.piece_has_ability(state, *p, PieceKind::Beacon)
                    && !state.rule_runtime.beacon_due.contains_key(&p.id)
            })
            .collect();
        for (from, beacon) in beacons {
            let neighbors: Vec<_> = (-1..=1)
                .flat_map(|df| {
                    (-1..=1).filter_map(move |dr| {
                        if df == 0 && dr == 0 {
                            None
                        } else {
                            from.offset(df, dr)
                        }
                    })
                })
                .collect();
            if neighbors
                .iter()
                .filter(|s| {
                    self.board_piece_at(state, **s).is_some()
                        || state.card_runtime.barricades.contains(s)
                })
                .count()
                < 6
            {
                continue;
            }
            let victims: Vec<_> = neighbors
                .iter()
                .filter_map(|s| self.board_piece_entry_at(state, *s))
                .filter(|(_, p)| !self.is_royal_piece(state, *p))
                .collect();
            let mut removed = BTreeSet::new();
            for (s, p) in victims {
                if removed.insert(p.id) {
                    self.delete_piece_for_card(state, s, p, actor, "beacon", events);
                }
            }
            for s in neighbors {
                state.card_runtime.remove_barricade(s);
                if state.card_runtime.football_pos == Some(s) {
                    state.card_runtime.football_pos = None;
                }
            }
            state
                .rule_runtime
                .beacon_due
                .insert(beacon.id, (actor, rule_halfmove(state) + 5));
        }
    }

    fn rule_automatic_step(
        &self,
        state: &mut GameState,
        actor: Player,
        action: Action,
        events: &mut Vec<Event>,
    ) {
        let Action::Move(m) = action else {
            return;
        };
        let turn = state.turn;
        let actions = state.actions_remaining;
        let turns = state.turns_taken;
        let phase = state.phase.clone();
        let mut probe = state.clone();
        probe.turn = actor;
        probe.actions_remaining = u16::MAX;
        probe.phase = GamePhase::Playing;
        probe.rule_runtime.automatic = true;
        if let Ok(mut result) = self.apply_move(&probe, action, m) {
            result.state.rule_runtime.automatic = false;
            result.state.turn = turn;
            result.state.turns_taken = turns;
            result.state.actions_remaining = actions;
            result.state.phase = phase;
            *state = result.state;
            events.extend(result.events);
        }
    }

    pub(in crate::engine) fn current_rule_dependency_start(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) {
        if !rule_dependencies_active(state) {
            return;
        }
        let actor = state.turn;
        let live_beacons: BTreeSet<_> = state
            .rule_runtime
            .beacon_due
            .iter()
            .filter_map(|(id, (owner, _))| {
                state
                    .board
                    .find_by_id(*id)
                    .filter(|(_, p)| {
                        p.owner == *owner && self.piece_has_ability(state, *p, PieceKind::Beacon)
                    })
                    .map(|_| *id)
            })
            .collect();
        state
            .rule_runtime
            .beacon_due
            .retain(|id, _| live_beacons.contains(id));
        if state
            .rule_runtime
            .beacon_due
            .values()
            .any(|(owner, due)| *owner == actor && rule_halfmove(state) >= *due)
        {
            state.status = GameStatus::Won {
                winner: actor,
                reason: WinReason::CardEffect("beacon".into()),
            };
            return;
        }
        let dons: Vec<_> = web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, p)| {
                p.owner == actor && self.piece_has_ability(state, *p, PieceKind::DonQuixote)
            })
            .map(|(_, p)| p.id)
            .collect();
        if dons.is_empty() {
            return;
        }
        let don_count = dons.len();
        let seed =
            rule_random(state, events, 1usize << 32, "rule.chaos.don-quixote.seed").unwrap() as u32;
        let mut random = super::chess_n_pow_30::SeededRandom(seed);
        for id in dons {
            let Some((from, piece)) = state.board.find_by_id(id) else {
                continue;
            };
            if state.status != GameStatus::Ongoing {
                break;
            }
            let mills: Vec<_> = web_ordered_pieces(state)
                .into_iter()
                .filter(|(_, p)| self.piece_has_ability(state, *p, PieceKind::Windmill))
                .map(|(s, _)| s)
                .collect();
            if mills.is_empty() {
                let mut moves = Vec::new();
                self.generate_moves_for_piece(state, from, piece, &mut moves);
                if moves.is_empty() {
                    continue;
                }
                let quiet: Vec<_> = moves
                    .iter()
                    .filter(|a| {
                        if let Action::Move(m) = a {
                            state.board.get(m.to).is_none_or(|p| {
                                !state.card_runtime.protected_piece_ids.contains(&p.id)
                            })
                        } else {
                            false
                        }
                    })
                    .cloned()
                    .collect();
                if !quiet.is_empty() {
                    moves = quiet;
                }
                let mut roll = random.bounded(40320);
                let mut remaining: Vec<_> = (0..8).collect();
                let mut priority = Vec::new();
                while !remaining.is_empty() {
                    let n = remaining.len();
                    let i = roll as usize % n;
                    roll /= n as u32;
                    priority.push(remaining.remove(i));
                }
                moves.sort_by_key(|a| {
                    let Action::Move(m) = a else {
                        return usize::MAX;
                    };
                    KNIGHT
                        .iter()
                        .position(|(df, dr)| from.offset(*df, *dr) == Some(m.to))
                        .and_then(|n| priority.iter().position(|p| *p == n))
                        .unwrap_or(8)
                });
                self.rule_automatic_step(state, actor, moves.remove(0), events);
                continue;
            }
            let blocked: BTreeSet<_> = state
                .board
                .iter()
                .filter(|(_, p)| self.is_royal_piece(state, *p) || p.kind == PieceKind::Vip)
                .map(|(s, _)| s)
                .collect();
            let fast = mills.len() >= 13
                || don_count >= 2 && mills.len() >= 7
                || don_count >= 4 && mills.len() >= 2;
            let path = don_path(from, &mills, &blocked, fast)
                .or_else(|| don_path(from, &mills, &BTreeSet::new(), fast));
            let Some(path) = path else {
                continue;
            };
            for to in path {
                if state.status != GameStatus::Ongoing {
                    break;
                }
                let Some((source, mut p)) = state.board.find_by_id(id) else {
                    break;
                };
                if let Some((anchor, victim)) = self.board_piece_entry_at(state, to) {
                    self.remove_piece_for_card(state, anchor, victim, actor, "don-quixote", events);
                    p.capture_count = p.capture_count.saturating_add(1);
                }
                state.board.set(source, None);
                state.board.set(to, Some(p));
                state.card_runtime.moved_piece_ids.insert(id);
                state.en_passant = None;
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("don-quixote-rampage:{source}:{to}"),
                });
                self.resolve_missing_king(state, events);
            }
        }
    }
}

impl Engine {
    pub(in crate::engine) fn generate_rule_dependency_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let orthogonal = &[(0, 1), (0, -1), (1, 0), (-1, 0)];
        let diagonal = &[(1, 1), (1, -1), (-1, 1), (-1, -1)];
        match piece.kind {
            PieceKind::Grappler => self.generate_slider_moves(
                state,
                from,
                piece,
                &[
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ],
                actions,
            ),
            PieceKind::RevolvingDoor => {
                self.generate_slider_moves(state, from, piece, orthogonal, actions)
            }
            PieceKind::DonQuixote => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-2, -1),
                    (-2, 1),
                    (-1, -2),
                    (-1, 2),
                    (1, -2),
                    (1, 2),
                    (2, -1),
                    (2, 1),
                ],
                actions,
            ),
            PieceKind::Medium => self.generate_current_medium_moves(state, from, piece, actions),
            PieceKind::Killer => {
                for &(df, dr) in diagonal {
                    let mut cursor = from;
                    for _ in 0..2 {
                        let Some(to) = cursor.offset(df, dr) else {
                            break;
                        };
                        let target = self.movement_occupant(state, to);
                        if target.is_none()
                            || target.is_some_and(|p| self.can_capture(state, piece, p))
                        {
                            actions.push(Action::Move(MoveAction::new(from, to, None)));
                        }
                        if target.is_some() {
                            break;
                        }
                        cursor = to;
                    }
                }
            }
            PieceKind::Beacon => {
                for &(df, dr) in &[
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ] {
                    if let Some(to) = from.offset(df, dr)
                        && self.movement_occupant(state, to).is_none()
                    {
                        actions.push(Action::Move(MoveAction::new(from, to, None)));
                    }
                }
            }
            PieceKind::BerolinaPawn => {
                let dr = if piece.owner == Player::White { 1 } else { -1 };
                for df in [-1, 1] {
                    if let Some(to) = from.offset(df, dr)
                        && self.movement_occupant(state, to).is_none()
                    {
                        self.push_rule_berolina_move(state, from, to, piece, actions);
                        if !state.card_runtime.moved_piece_ids.contains(&piece.id)
                            && matches!(from.rank(), 0 | 1 | 6 | 7)
                            && let Some(far) = to.offset(df, dr)
                            && self.movement_occupant(state, far).is_none()
                        {
                            self.push_rule_berolina_move(state, from, far, piece, actions);
                        }
                    }
                }
                if let Some(to) = from.offset(0, dr)
                    && self
                        .movement_occupant(state, to)
                        .is_some_and(|p| self.can_capture(state, piece, p))
                {
                    self.push_rule_berolina_move(state, from, to, piece, actions);
                }
            }
            PieceKind::TransmissionTower | PieceKind::Barrier => {}
            _ => unreachable!("dependency generator only accepts chaos expansion pieces"),
        }
    }
}

impl Engine {
    fn push_rule_berolina_move(
        &self,
        _state: &GameState,
        from: Square,
        to: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        if to.rank() == if piece.owner == Player::White { 7 } else { 0 } {
            for kind in [
                PieceKind::Queen,
                PieceKind::Rook,
                PieceKind::Bishop,
                PieceKind::Knight,
            ] {
                actions.push(Action::Move(MoveAction::new(from, to, Some(kind))));
            }
        } else {
            actions.push(Action::Move(MoveAction::new(from, to, None)));
        }
    }
}

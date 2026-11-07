//! Shared settlement order and state for the current board rules.
use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuleRuntime {
    pub enabled: BTreeSet<String>,
    pub(crate) ticket_random_record: Option<Vec<(usize, usize)>>,
    pub(crate) ticket_random_replay: Option<(Vec<usize>, usize, bool)>,
    pub automatic: bool,
    pub piece_fresh_until: BTreeMap<u32, u32>,
    pub medium_memory: Option<ParrotMemoryStatus>,
    pub medium_windmill_mode: Option<WindmillMode>,
    pub grappler_bound: BTreeMap<u32, Player>,
    pub beacon_due: BTreeMap<u32, (Player, u32)>,
    pub berolina_rights: BTreeMap<u32, (Player, Square, Vec<Square>)>,
    pub acceleration_after: Option<u32>,
    pub collapse_depth: u8,
    pub collapse_next: Option<u32>,
    /// Pending Bombardment file collapses, one entry per cast.
    pub pending_bombardments: Vec<crate::state::PendingBombardment>,
    pub crown_id: Option<u32>,
    pub crown_square: Option<Square>,
    pub chaos_until_halfmove: Option<u32>,
    pub crown_color: Option<Player>,
    pub crown_count: u8,
    pub crown_last_fullmove: u32,
    pub bomb_ignore: BTreeMap<Square, u32>,
    pub winter_preview_cycle: Option<u32>,
    pub winter_preview: Vec<u32>,
    pub winter_frozen: BTreeSet<u32>,
    pub winter_cycle: u32,
    pub winter_disabled: bool,
    pub monster_last_halfmove: u32,
    pub mono_shades: BTreeMap<u32, u8>,
}

impl RuleRuntime {
    pub(crate) fn position_data(&self, state: &GameState) -> String {
        let mut normalized = self.clone();
        normalized.automatic = false;
        let black = state.turns_taken[Player::Black.index()];
        let halfmove = black.saturating_mul(2) + u32::from(state.turn == Player::Black);
        for due in normalized.piece_fresh_until.values_mut() {
            *due = due.saturating_sub(halfmove);
        }
        for (_, due) in normalized.beacon_due.values_mut() {
            *due = due.saturating_sub(halfmove);
        }
        normalized.acceleration_after = normalized
            .acceleration_after
            .map(|n| n.saturating_sub(halfmove));
        normalized.chaos_until_halfmove = normalized
            .chaos_until_halfmove
            .map(|n| n.saturating_sub(halfmove));
        normalized.collapse_next = normalized.collapse_next.map(|n| n.saturating_sub(black));
        let crown_delta = if normalized.crown_id.is_some() {
            i64::from(black) - i64::from(normalized.crown_last_fullmove)
        } else {
            0
        };
        normalized.crown_last_fullmove = 0;
        normalized.winter_cycle =
            if normalized.enabled.contains(crate::WINTER_KINGDOM) && !normalized.winter_disabled {
                black % 3
            } else {
                0
            };
        normalized.winter_preview_cycle = normalized
            .winter_preview_cycle
            .map(|n| n.saturating_sub(black));
        normalized.monster_last_halfmove = if normalized.enabled.contains(crate::MONSTER_RULE) {
            halfmove % 3
        } else {
            0
        };
        format!(
            "{:?}",
            (
                normalized,
                crown_delta,
                &state.card_runtime.black_holes,
                &state.card_runtime.high_ground_squares,
                &state.card_runtime.highways,
                &state.card_runtime.rule_bombs,
                state.card_runtime.rule_portals,
                state.card_runtime.crown_pos,
                state.card_runtime.football_pos,
                state.card_runtime.monster_rule,
                self.enabled
                    .contains(crate::REVELATION)
                    .then(|| 60u32.saturating_sub(black))
            )
        )
        .replace(", ticket_random_record: None", "")
        .replace(", ticket_random_replay: None", "")
    }
}

pub(super) fn mark_rule(state: &mut GameState, id: &str) {
    state.rule_runtime.enabled.insert(id.to_owned());
}
pub(super) fn rule_active(state: &GameState, id: &str) -> bool {
    state.rule_runtime.enabled.contains(id)
}
pub(super) fn rule_dependencies_active(state: &GameState) -> bool {
    state.ruleset_id == crate::LATEST_RULESET_ID
        && (rule_active(state, crate::CHESS_N_POW_30)
            || !state.rule_runtime.grappler_bound.is_empty()
            || !state.rule_runtime.beacon_due.is_empty()
            || !state.rule_runtime.berolina_rights.is_empty()
            || state.board.iter().any(|(_, p)| {
                matches!(
                    p.kind,
                    PieceKind::Grappler
                        | PieceKind::RevolvingDoor
                        | PieceKind::DonQuixote
                        | PieceKind::Medium
                        | PieceKind::BerolinaPawn
                        | PieceKind::Killer
                        | PieceKind::Beacon
                        | PieceKind::TransmissionTower
                        | PieceKind::Barrier
                )
            }))
}
pub(super) fn rule_halfmove(state: &GameState) -> u32 {
    state.turns_taken[Player::Black.index()].saturating_mul(2)
        + u32::from(state.turn == Player::Black)
}
pub(in crate::engine) fn rule_sample(
    state: &mut GameState,
    len: usize,
) -> (u64, Option<usize>, bool) {
    if let Some((values, cursor, missing)) = &mut state.rule_runtime.ticket_random_replay {
        if len == 0 {
            return (0, None, true);
        }
        let index = if let Some(value) = values.get(*cursor) {
            *value % len
        } else {
            *missing = true;
            0
        };
        *cursor += 1;
        return (0, Some(index), true);
    }
    if len == 0 && state.rule_runtime.ticket_random_record.is_some() {
        return (0, None, true);
    }
    let (roll, index) = state.rng_mut().web_choice_index(len);
    if let Some(record) = &mut state.rule_runtime.ticket_random_record
        && let Some(index) = index
    {
        record.push((len, index));
    }
    (roll, index, false)
}
pub(super) fn rule_random(
    state: &mut GameState,
    events: &mut Vec<Event>,
    len: usize,
    purpose: &str,
) -> Option<usize> {
    let (roll, index, replayed) = rule_sample(state, len);
    if !replayed {
        events.push(Event::RandomSample {
            purpose: purpose.into(),
            roll_53: roll,
        });
    }
    index
}
pub(super) fn rule_shuffle<T>(
    state: &mut GameState,
    events: &mut Vec<Event>,
    values: &mut [T],
    purpose: &str,
) {
    for i in (1..values.len()).rev() {
        let j = rule_random(state, events, i + 1, &format!("{purpose}.{i}")).unwrap();
        values.swap(i, j);
    }
}
pub(super) fn rule_center() -> Vec<Square> {
    ["d4", "d5", "e4", "e5"]
        .map(|s| s.parse().unwrap())
        .to_vec()
}
pub(super) fn rule_eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

impl Engine {
    pub(in crate::engine) fn augment_current_rule_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        if state.rule_runtime.enabled.is_empty()
            && !rule_dependencies_active(state)
            && state.card_runtime.highway_cells.is_empty()
        {
            return;
        }
        self.generate_highway_trail_moves(state, from, piece, actions);
        if matches!(
            self.piece_ability_kind(state, piece),
            PieceKind::Pawn | PieceKind::BerolinaPawn
        ) {
            let dr = if piece.owner == Player::White { 1 } else { -1 };
            for df in -1..=1 {
                if let Some(to) = from.offset(df, dr)
                    && self.rule_berolina_capture(state, from, piece, to).is_some()
                {
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                }
            }
        }
        if state.ruleset_id != crate::LATEST_RULESET_ID
            && rule_active(state, crate::HIGHWAY)
            && state.card_runtime.highways.contains(&from.file())
            && !state.card_runtime.large_pieces.contains_key(&piece.id)
            && !matches!(
                piece.kind,
                PieceKind::Football | PieceKind::Crown | PieceKind::Scarecrow
            )
        {
            self.generate_slider_moves(state, from, piece, &[(0, 1), (0, -1)], actions);
        }
        let towers: Vec<_> = state
            .board
            .iter()
            .filter(|(_, p)| {
                p.owner == piece.owner
                    && p.id != piece.id
                    && self.piece_has_ability(state, *p, PieceKind::TransmissionTower)
            })
            .map(|(s, _)| s)
            .collect();
        for tower in towers {
            for df in -1..=1 {
                for dr in -1..=1 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    if let Some(to) = tower.offset(df, dr)
                        && self.card_landing_open(state, to)
                        && !self.rule_cell_collapsed(state, to)
                    {
                        actions.push(Action::Move(MoveAction::new(from, to, None)));
                    }
                }
            }
        }
        self.augment_rule_portal_moves(state, from, piece, actions);
    }

    fn generate_highway_trail_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let road = |square: Square| {
            state.card_runtime.highway_cells.contains(&square)
                || (rule_active(state, crate::HIGHWAY)
                    && state.card_runtime.highways.contains(&square.file()))
        };
        if state.ruleset_id != crate::LATEST_RULESET_ID
            || (state.card_runtime.highway_cells.is_empty() && !rule_active(state, crate::HIGHWAY))
            || !road(from)
            || state.card_runtime.large_pieces.contains_key(&piece.id)
            || matches!(
                piece.kind,
                PieceKind::Football
                    | PieceKind::Crown
                    | PieceKind::Scarecrow
                    | PieceKind::Colossus
                    | PieceKind::BigRook
                    | PieceKind::BigBishop
            )
        {
            return;
        }
        let capture_allowed = !matches!(
            self.piece_ability_kind(state, piece),
            PieceKind::Guard
                | PieceKind::Recruiter
                | PieceKind::Wizard
                | PieceKind::Herald
                | PieceKind::Merchant
                | PieceKind::Scarecrow
        );
        for (df, dr) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let mut cursor = from;
            while let Some(to) = cursor.offset(df, dr) {
                if !road(to) {
                    break;
                }
                match self.movement_occupant(state, to) {
                    None => actions.push(Action::Move(MoveAction::new(from, to, None))),
                    Some(target) => {
                        let jester_target = piece.kind != PieceKind::Jester
                            || matches!(
                                target.kind,
                                PieceKind::King | PieceKind::RoyalKnight | PieceKind::Merchant
                            );
                        if capture_allowed
                            && jester_target
                            && self.can_capture(state, piece, target)
                        {
                            actions.push(Action::Move(MoveAction::new(from, to, None)));
                        }
                        break;
                    }
                }
                cursor = to;
            }
        }
    }

    pub(in crate::engine) fn rule_cell_collapsed(&self, state: &GameState, square: Square) -> bool {
        let depth = state.rule_runtime.collapse_depth;
        square
            .file()
            .min(7 - square.file())
            .min(square.rank())
            .min(7 - square.rank())
            < depth
    }

    pub(in crate::engine) fn current_rule_arrivals(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) {
        if state.rule_runtime.enabled.is_empty() {
            return;
        }
        self.settle_rule_black_holes(state, events);
        self.settle_rule_bombs(state, events, &[]);
        self.settle_rule_crown(state, false);
    }

    pub(in crate::engine) fn current_rule_move(
        &self,
        state: &mut GameState,
        _movement: MoveAction,
        events: &mut Vec<Event>,
    ) {
        if state.rule_runtime.enabled.is_empty() {
            return;
        }
        self.settle_rule_black_holes(state, events);
        let traversed: Vec<_> = events
            .iter()
            .rev()
            .take_while(|e| !matches!(e, Event::ActionApplied { .. }))
            .find_map(|e| {
                if let Event::EffectApplied { command, .. } = e {
                    command.strip_prefix("portal:").map(|s| {
                        s.split(':')
                            .take(2)
                            .filter_map(|s| s.parse::<Square>().ok())
                            .collect()
                    })
                } else {
                    None
                }
            })
            .unwrap_or_default();
        self.settle_rule_bombs(state, events, &traversed);
        self.settle_rule_crown(state, false);
    }

    pub(in crate::engine) fn current_rule_turn(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        let halfmove = rule_halfmove(state);
        state
            .rule_runtime
            .piece_fresh_until
            .retain(|_, due| *due > halfmove);
        if state.rule_runtime.enabled.is_empty() && !rule_dependencies_active(state) {
            return;
        }
        self.settle_periodic_collapse(state, events);
        if state.status != GameStatus::Ongoing {
            return;
        }
        if actor == Player::Black {
            self.settle_rule_conveyor(state, events);
        }
        self.current_rule_arrivals(state, events);
        self.resolve_missing_king(state, events);
        if state.status != GameStatus::Ongoing {
            return;
        }
        self.settle_rule_monster(state, events);
        self.settle_allied_monsters(state, events);
        self.current_rule_arrivals(state, events);
        self.resolve_missing_king(state, events);
        if state.status != GameStatus::Ongoing {
            return;
        }
        self.settle_rule_winter(state, actor, events);
        self.settle_rule_crown(state, true);
        self.settle_rule_revelation(state);
        self.current_rule_dependency_end(state, actor, events);
        if state.status == GameStatus::Ongoing {
            self.current_rule_dependency_start(state, events);
        }
    }

    pub(in crate::engine) fn current_rule_move_filter(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        if state.rule_runtime.grappler_bound.contains_key(&piece.id) {
            actions.clear();
            return;
        }
        if state.rule_runtime.enabled.is_empty() {
            return;
        }
        if rule_active(state, crate::CROWN) && state.rule_runtime.crown_id == Some(piece.id)
            || state.rule_runtime.winter_frozen.contains(&piece.id)
        {
            actions.clear();
            return;
        }
        actions.retain(|action| {
            let Action::Move(m) = action else {
                return true;
            };
            let landing = self
                .rule_portal_route(state, from, piece, m.to)
                .map_or(m.to, |r| r.landing);
            let invalid_pull = self.piece_has_ability(state, piece, PieceKind::Grappler)
                && self.board_piece_at(state, m.to).is_some()
                && from
                    .file()
                    .abs_diff(m.to.file())
                    .max(from.rank().abs_diff(m.to.rank()))
                    >= 2
                && self.rule_grappler_pull(state, *m).is_none();
            !invalid_pull
                && !self.rule_cell_collapsed(state, landing)
                && (!rule_active(state, crate::MONOCHROME_CHESS)
                    || state
                        .rule_runtime
                        .mono_shades
                        .get(&piece.id)
                        .copied()
                        .unwrap_or((from.file() + from.rank()) % 2)
                        == (landing.file() + landing.rank()) % 2)
        });
    }

    pub(in crate::engine) fn current_rule_capture_allowed(
        &self,
        state: &GameState,
        attacker: Piece,
        target: Piece,
    ) -> bool {
        if state.rule_runtime.enabled.is_empty() && !rule_dependencies_active(state) {
            return true;
        }
        if self.is_royal_piece(state, target)
            && state.board.iter().any(|(_, p)| {
                p.owner == target.owner && self.piece_has_ability(state, p, PieceKind::Barrier)
            })
        {
            return false;
        }
        if state.rule_runtime.winter_frozen.contains(&target.id) {
            return false;
        }
        if let (Some((from, _)), Some((to, _))) = (
            state.board.find_by_id(attacker.id),
            state.board.find_by_id(target.id),
        ) && rule_active(state, crate::HIGH_GROUND)
            && state.card_runtime.high_ground_squares.contains(&to)
            && !state.card_runtime.high_ground_squares.contains(&from)
        {
            return false;
        }
        true
    }
}

impl Engine {
    pub(in crate::engine) fn current_rule_spawn_cell(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
        id: &str,
    ) -> Result<Square, EngineError> {
        let mut candidates: Vec<_> = rule_center()
            .into_iter()
            .filter(|s| self.card_landing_open(state, *s) && !self.rule_cell_collapsed(state, *s))
            .collect();
        if candidates.is_empty() {
            candidates = (0..64)
                .filter_map(Square::from_index)
                .filter(|s| {
                    self.card_landing_open(state, *s) && !self.rule_cell_collapsed(state, *s)
                })
                .collect();
            let distance = |s: Square| {
                (2 * i16::from(s.file()) - 7).unsigned_abs()
                    + (2 * i16::from(s.rank()) - 7).unsigned_abs()
            };
            if let Some(minimum) = candidates.iter().map(|s| distance(*s)).min() {
                candidates.retain(|s| distance(*s) == minimum);
            }
        }
        candidates.sort_by_key(|s| s.to_string());
        let Some(i) = rule_random(state, events, candidates.len(), &format!("rule.{id}.spawn"))
        else {
            return Err(EngineError::Invariant(
                "중립 설치물을 놓을 빈칸이 없습니다.".into(),
            ));
        };
        Ok(candidates[i])
    }
}

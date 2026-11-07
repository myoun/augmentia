use super::*;
pub(super) const ID: &str = crate::RULE_BOMBS;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_rule_bombs(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn settle_rule_bombs(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
        traversed: &[Square],
    ) {
        if !rule_active(state, ID) {
            return;
        }
        let triggered: Vec<_> = state
            .card_runtime
            .rule_bombs
            .iter()
            .copied()
            .filter(|s| {
                traversed.contains(s)
                    || self
                        .board_piece_at(state, *s)
                        .is_some_and(|p| state.rule_runtime.bomb_ignore.get(s) != Some(&p.id))
            })
            .collect();
        let cells: BTreeSet<_> = triggered
            .iter()
            .flat_map(|s| {
                (0..8).flat_map(move |i| {
                    [
                        Square::new(s.file(), i).unwrap(),
                        Square::new(i, s.rank()).unwrap(),
                    ]
                })
            })
            .collect();
        let victims: Vec<_> = state
            .board
            .iter()
            .filter(|(s, p)| {
                cells.contains(s)
                    || state.card_runtime.large_anchor(p.id).is_some_and(|anchor| {
                        large_piece_cells(anchor).iter().any(|s| cells.contains(s))
                    })
            })
            .collect();
        let memories =
            self.rule_capture_memories(state, &victims.iter().map(|(_, p)| *p).collect::<Vec<_>>());
        for (square, piece) in victims {
            self.remove_piece_for_card(state, square, piece, state.turn, ID, events);
        }
        self.settle_rule_capture_memories(state, &memories, events);
        for square in cells {
            state.card_runtime.remove_barricade(square);
        }
        for square in triggered {
            state.card_runtime.rule_bombs.remove(&square);
            state.rule_runtime.bomb_ignore.remove(&square);
        }
    }
}

impl Engine {
    pub(in crate::engine) fn install_rule_bombs(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let mut candidates = Vec::new();
        for file in 0..8 {
            for rank in 3..=4 {
                let s = Square::new(file, rank).unwrap();
                if file != 4
                    && !state.card_runtime.barricades.contains(&s)
                    && !state.card_runtime.black_holes.contains(&s)
                    && state.card_runtime.crown_pos != Some(s)
                    && !self.rule_cell_collapsed(state, s)
                {
                    candidates.push(s);
                }
            }
        }
        candidates.sort_by_key(|s| s.to_string());
        if candidates.len() < 3 {
            return Err(EngineError::Invariant(
                "룰 폭탄을 놓을 후보 칸 세 곳이 필요합니다.".into(),
            ));
        }
        for ordinal in 0..3 {
            let mut pool: Vec<_> = candidates
                .iter()
                .copied()
                .filter(|s| self.board_piece_at(state, *s).is_none())
                .collect();
            if pool.is_empty() {
                pool = candidates.clone();
            }
            let i =
                rule_random(state, events, pool.len(), &format!("rule.bombs.{ordinal}")).unwrap();
            let square = pool[i];
            candidates.retain(|s| *s != square);
            state.card_runtime.rule_bombs.insert(square);
            if let Some(p) = self.board_piece_at(state, square) {
                state.rule_runtime.bomb_ignore.insert(square, p.id);
            }
        }
        Ok(())
    }
}

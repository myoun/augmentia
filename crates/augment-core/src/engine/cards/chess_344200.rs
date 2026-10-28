use super::*;
pub(super) const ID: &str = crate::CHESS_344200;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_chess_344200(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn install_chess_344200(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let before = state.board.clone();
        let mut placements = Vec::new();
        let mut moved_ids = BTreeSet::new();
        for owner in [Player::White, Player::Black] {
            let home = if owner == Player::White { 0 } else { 7 };
            let entries: Vec<_> = web_ordered_pieces(state)
                .into_iter()
                .filter(|(s, p)| {
                    p.owner == owner
                        && if owner == Player::White {
                            s.rank() <= 1
                        } else {
                            s.rank() >= 6
                        }
                })
                .collect();
            if entries.len() < 2 {
                return Err(EngineError::Invariant("재배열할 기물이 부족합니다.".into()));
            }
            let mut royal_cells: Vec<_> = (0..8).map(|f| Square::new(f, home).unwrap()).collect();
            rule_shuffle(
                state,
                events,
                &mut royal_cells,
                &format!("rule.344200.{owner}.royal"),
            );
            let royals: Vec<_> = entries
                .iter()
                .filter(|(_, p)| self.is_royal_piece(state, *p))
                .copied()
                .collect();
            if royals.len() > royal_cells.len() {
                return Err(EngineError::Invariant("왕실 기물이 너무 많습니다.".into()));
            }
            let selected: BTreeSet<_> = royal_cells.iter().copied().take(royals.len()).collect();
            let rows = if owner == Player::White {
                [1, 0]
            } else {
                [7, 6]
            };
            let mut ordinary_cells: Vec<_> = rows
                .into_iter()
                .flat_map(|rank| (0..8).map(move |f| Square::new(f, rank).unwrap()))
                .filter(|s| !selected.contains(s))
                .collect();
            rule_shuffle(
                state,
                events,
                &mut ordinary_cells,
                &format!("rule.344200.{owner}.ordinary"),
            );
            for ((_, piece), square) in royals.into_iter().zip(royal_cells) {
                moved_ids.insert(piece.id);
                placements.push((square, piece));
            }
            for ((_, piece), square) in entries
                .into_iter()
                .filter(|(_, p)| !self.is_royal_piece(state, *p))
                .zip(ordinary_cells)
            {
                moved_ids.insert(piece.id);
                placements.push((square, piece));
            }
        }
        for (square, piece) in before.iter() {
            if moved_ids.contains(&piece.id) {
                state.board.set(square, None);
            }
        }
        for (square, piece) in placements {
            state.board.set(square, Some(piece));
            state.card_runtime.piece_origins.insert(piece.id, square);
        }
        state.en_passant = None;
        Ok(())
    }
}

use super::*;
pub(super) const ID: &str = crate::CROWN;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_crown(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn settle_rule_crown(&self, state: &mut GameState, count: bool) {
        if !rule_active(state, ID) {
            return;
        }
        let old_id = state.rule_runtime.crown_id;
        if old_id.is_some_and(|id| state.board.find_by_id(id).is_none()) {
            state.rule_runtime.crown_id = None;
            state.card_runtime.crown_pos = state.rule_runtime.crown_square;
        }
        let old_id = state.rule_runtime.crown_id;
        let holder = old_id
            .and_then(|id| state.board.find_by_id(id))
            .or_else(|| {
                state
                    .card_runtime
                    .crown_pos
                    .and_then(|s| state.board.get(s).map(|p| (s, p)))
            });
        let Some((square, mut piece)) = holder else {
            state.rule_runtime.crown_id = None;
            state.rule_runtime.crown_color = None;
            state.rule_runtime.crown_count = 0;
            return;
        };
        if matches!(
            piece.kind,
            PieceKind::Monster | PieceKind::Football | PieceKind::Scarecrow
        ) {
            return;
        }
        if piece.kind != PieceKind::Crown {
            let royal = self.is_royal_piece(state, piece);
            self.clear_piece_kind_runtime(state, piece.id);
            piece.kind = PieceKind::Crown;
            state.board.set(square, Some(piece));
            state.card_runtime.moved_piece_ids.insert(piece.id);
            if royal {
                state.card_runtime.royal_piece_ids.insert(piece.id);
            }
        }
        state.card_runtime.crown_pos = None;
        state.rule_runtime.crown_id = Some(piece.id);
        state.rule_runtime.crown_square = Some(square);
        if state.rule_runtime.crown_color != Some(piece.owner) {
            state.rule_runtime.crown_color = Some(piece.owner);
            state.rule_runtime.crown_count = 0;
            state.rule_runtime.crown_last_fullmove = state.turns_taken[1]
                + u32::from(!count && state.turn == Player::Black && state.actions_remaining == 0);
        }
        if count {
            let elapsed =
                state.turns_taken[1].saturating_sub(state.rule_runtime.crown_last_fullmove);
            state.rule_runtime.crown_count =
                (u32::from(state.rule_runtime.crown_count) + elapsed).min(10) as u8;
            state.rule_runtime.crown_last_fullmove = state.turns_taken[1];
            if state.rule_runtime.crown_count >= 10 && state.status == GameStatus::Ongoing {
                state.status = GameStatus::Won {
                    winner: piece.owner,
                    reason: WinReason::CardEffect(ID.into()),
                };
            }
        }
    }

    pub(in crate::engine) fn transfer_rule_crown(
        &self,
        state: &mut GameState,
        from: Square,
        victim: u32,
        captor: Option<u32>,
    ) {
        if rule_active(state, ID) && state.rule_runtime.crown_id == Some(victim) {
            state.rule_runtime.crown_id = captor;
            state.rule_runtime.crown_color = None;
            state.rule_runtime.crown_count = 0;
            state.rule_runtime.crown_square = Some(from);
            state.card_runtime.crown_pos = captor.is_none().then_some(from);
        }
    }
}

impl Engine {
    pub(in crate::engine) fn install_crown(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let mut center = rule_center();
        let open: Vec<_> = center
            .iter()
            .copied()
            .filter(|s| self.card_landing_open(state, *s))
            .collect();
        if !open.is_empty() {
            center = open;
        }
        let index = rule_random(state, events, center.len(), "rule.crown.spawn").unwrap();
        state.card_runtime.crown_pos = Some(center[index]);
        state.rule_runtime.crown_last_fullmove = state.turns_taken[1];
        self.settle_rule_crown(state, false);
        state.rule_runtime.crown_last_fullmove = state.turns_taken[1];
        Ok(())
    }
}

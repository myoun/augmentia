//! Ordered cross-card settlement. Keep call order stable: it affects events and RNG.
use super::*;

impl Engine {
    pub(in crate::engine) fn settle_card_turn_start(&self, state: &mut GameState, owner: Player) {
        let expired_walls: Vec<_> = state
            .card_runtime
            .blockade_walls
            .iter()
            .filter_map(|(square, (by, due))| {
                (*by == owner && state.turns_taken[owner.index()] >= *due).then_some(*square)
            })
            .collect();
        for square in expired_walls {
            state.card_runtime.remove_barricade(square);
        }
        let due: Vec<_> = state
            .card_runtime
            .scarecrow_reservations
            .iter()
            .filter_map(|(id, (by, turn))| {
                (*by == owner && state.turns_taken[owner.index()] >= *turn).then_some(*id)
            })
            .collect();
        for id in due {
            state.card_runtime.scarecrow_reservations.remove(&id);
            if let Some((square, mut piece)) = state.board.find_by_id(id) {
                self.clear_removed_piece_statuses(state, id);
                piece.kind = PieceKind::Scarecrow;
                state.board.set(square, Some(piece));
                state.card_runtime.moved_piece_ids.insert(id);
                state.card_runtime.piece_origins.insert(id, square);
            }
        }
        if state.card_runtime.bishop_infiltration_due[owner.index()]
            .is_some_and(|due| state.turns_taken[owner.index()] >= due)
        {
            state.card_runtime.bishop_infiltration_due[owner.index()] = None;
        }
        self.settle_stake(state, owner);
        self.settle_palaces(state);
        state
            .card_runtime
            .ice_king
            .retain(|_, (source, due)| *source != owner || state.turns_taken[owner.index()] < *due);
        self.settle_penta_check_turn_start(state, owner);
        self.settle_hope_turn_start(state, owner);
    }
}

impl Engine {
    pub(in crate::engine) fn apply_acquired_card_definition(
        &self,
        state: &mut GameState,
        id: &str,
        owner: Player,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            self.apply_card_definition(state, id, owner, events);
            return Ok(());
        }
        super::acquire(self, state, id, owner, events)
    }
}

impl Engine {
    pub(in crate::engine) fn settle_check_objectives(&self, state: &mut GameState) {
        if state.ruleset_id != crate::LATEST_RULESET_ID
            || (state.card_runtime.october.penta_check == [None; 2]
                && state.card_runtime.october.prophecy_remaining == [None; 2])
        {
            return;
        }
        let checked = [
            self.card_royal_checked(state, Player::White),
            self.card_royal_checked(state, Player::Black),
        ];
        if checked.contains(&true) {
            self.cancel_prophecies(state);
        }
        self.observe_penta_checks(state, checked);
    }
}

impl Engine {
    pub(in crate::engine) fn settle_delayed_turn_start(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID || state.status != GameStatus::Ongoing {
            return;
        }
        self.settle_rule_tickets(state, owner, events);
        self.settle_icbm(state, owner, events);
        self.settle_seal(state, owner, events);
        self.settle_ascension(state, owner, events);
        self.resolve_gomoku_victory(state);
    }
}

impl Engine {
    pub(in crate::engine) fn settle_delayed_turn_end(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID || state.status != GameStatus::Ongoing {
            return;
        }
        self.settle_greek_gift(state, actor, events);
        self.settle_taboo(state, actor, events);
        self.settle_collapse(state, actor, events);
        self.settle_othello(state, actor, events);
        self.resolve_gomoku_victory(state);
    }
}

//! Ordered cross-card settlement. Keep call order stable: it affects events and RNG.
use super::*;

impl Engine {
    pub(in crate::engine) fn settle_card_turn_start(&self, state: &mut GameState, owner: Player) {
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
        self.resolve_gomoku_victory(state);
    }
}

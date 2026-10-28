use super::*;
pub(super) const ID: &str = crate::BLACK_HOLE;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_black_hole(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn settle_rule_black_holes(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) {
        if !rule_active(state, ID) {
            return;
        }
        let victims: Vec<_> = state
            .board
            .iter()
            .filter(|(s, p)| {
                state.card_runtime.black_holes.contains(s)
                    || state
                        .card_runtime
                        .large_pieces
                        .get(&p.id)
                        .is_some_and(|large| {
                            large_piece_cells(large.anchor)
                                .iter()
                                .any(|s| state.card_runtime.black_holes.contains(s))
                        })
            })
            .collect();
        let memories =
            self.rule_capture_memories(state, &victims.iter().map(|(_, p)| *p).collect::<Vec<_>>());
        for (square, piece) in victims {
            self.remove_piece_for_card(state, square, piece, state.turn, ID, events);
        }
        self.settle_rule_capture_memories(state, &memories, events);
    }
}

impl Engine {
    pub(in crate::engine) fn install_black_hole(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        state.card_runtime.black_holes.extend(rule_center());
        self.settle_rule_black_holes(state, events);
        self.resolve_missing_king(state, events);
        Ok(())
    }
}

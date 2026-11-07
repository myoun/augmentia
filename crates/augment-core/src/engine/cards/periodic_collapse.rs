use super::*;
pub(super) const ID: &str = crate::PERIODIC_COLLAPSE;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_periodic_collapse(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn settle_periodic_collapse(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) {
        if !rule_active(state, ID) {
            return;
        }
        // Upstream skips the periodic apply when a card collapse already resolved
        // in the same turn boundary (main-RjrZ5fFP.js:213178), so one boundary can
        // never remove two rings. The card settles earlier in the boundary and
        // advances the shared depth, which is the signal used here.
        let card_collapsed = events.iter().any(|event| {
            matches!(event, Event::EffectApplied { command, .. }
                if command.starts_with("collapse:"))
        });
        if card_collapsed {
            state.rule_runtime.collapse_next = state
                .rule_runtime
                .collapse_next
                .map(|n| n.saturating_add(20));
            return;
        }
        while state
            .rule_runtime
            .collapse_next
            .is_some_and(|n| state.turns_taken[1] >= n)
            && state.status == GameStatus::Ongoing
        {
            let depth = state.rule_runtime.collapse_depth;
            if depth < 4 {
                let victims: Vec<_> = state
                    .board
                    .iter()
                    .filter(|(s, p)| {
                        s.file() == depth
                            || s.file() == 7 - depth
                            || s.rank() == depth
                            || s.rank() == 7 - depth
                            || state.card_runtime.large_anchor(p.id).is_some_and(|anchor| {
                                large_piece_cells(anchor).iter().any(|s| {
                                    s.file() == depth
                                        || s.file() == 7 - depth
                                        || s.rank() == depth
                                        || s.rank() == 7 - depth
                                })
                            })
                    })
                    .collect();
                let memories = self.rule_capture_memories(
                    state,
                    &victims.iter().map(|(_, p)| *p).collect::<Vec<_>>(),
                );
                for (square, piece) in victims {
                    self.remove_piece_for_card(state, square, piece, state.turn, ID, events);
                }
                self.settle_rule_capture_memories(state, &memories, events);
                state.rule_runtime.collapse_depth += 1;
                let d = state.rule_runtime.collapse_depth;
                let open = |s: Square| {
                    s.file() >= d && s.rank() >= d && s.file() < 8 - d && s.rank() < 8 - d
                };
                state.card_runtime.barricades.retain(|s| open(*s));
                state.card_runtime.blockade_walls.retain(|s, _| open(*s));
                state.card_runtime.rule_bombs.retain(|s| open(*s));
                state.rule_runtime.bomb_ignore.retain(|s, _| open(*s));
                state
                    .card_runtime
                    .pending_lobsters
                    .retain(|p| open(p.square));
                state
                    .card_runtime
                    .pending_scarecrows
                    .retain(|p| open(p.square));
                state.card_runtime.quantum_positions.retain(|_, s| open(*s));
                if state.card_runtime.football_pos.is_some_and(|s| !open(s)) {
                    state.card_runtime.football_pos = None;
                }
                if state.card_runtime.crown_pos.is_some_and(|s| !open(s)) {
                    state.card_runtime.crown_pos = None;
                }
                self.resolve_missing_king(state, events);
            }
            state.rule_runtime.collapse_next = state
                .rule_runtime
                .collapse_next
                .map(|n| n.saturating_add(20));
        }
    }
}

impl Engine {
    pub(in crate::engine) fn install_periodic_collapse(
        &self,
        state: &mut GameState,
        _events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        state.rule_runtime.collapse_next = Some((state.turns_taken[1] / 20 + 1) * 20);
        Ok(())
    }
}

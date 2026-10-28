use super::*;
pub(super) const ID: &str = crate::WINTER_KINGDOM;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_winter_kingdom(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn settle_rule_winter(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        if !rule_active(state, ID) || state.rule_runtime.winter_disabled {
            return;
        }
        let completed = state.turns_taken[1];
        let cycle = if actor == Player::White {
            completed + 1
        } else {
            completed
        };
        if cycle == 0 || cycle % 3 != 0 {
            return;
        }
        if actor == Player::White && state.rule_runtime.winter_preview_cycle == Some(cycle)
            || actor == Player::Black && state.rule_runtime.winter_cycle == cycle
        {
            return;
        }
        let entries = web_ordered_pieces(state);
        let enough = [Player::White, Player::Black].into_iter().all(|owner| {
            entries
                .iter()
                .filter(|(_, p)| {
                    p.owner == owner
                        && !matches!(p.kind, PieceKind::Scarecrow | PieceKind::Football)
                        && state.card_runtime.monster_rule != Some(p.id)
                })
                .count()
                > 4
        });
        if actor == Player::Black {
            state.rule_runtime.winter_frozen.clear();
        }
        if !enough {
            if actor == Player::Black {
                state.rule_runtime.winter_disabled = true;
                state.rule_runtime.winter_preview.clear();
                state.rule_runtime.winter_preview_cycle = None;
            }
            return;
        }
        if actor == Player::Black && state.rule_runtime.winter_preview_cycle == Some(cycle) {
            state.rule_runtime.winter_frozen = state
                .rule_runtime
                .winter_preview
                .iter()
                .copied()
                .filter(|id| {
                    state.board.find_by_id(*id).is_some_and(|(_, p)| {
                        !self.is_royal_piece(state, p) && p.kind != PieceKind::Scarecrow
                    })
                })
                .collect();
        } else {
            let mut selected = Vec::new();
            for owner in [Player::White, Player::Black] {
                let mut pool: Vec<_> = entries
                    .iter()
                    .filter(|(_, p)| {
                        p.owner == owner
                            && !self.is_royal_piece(state, *p)
                            && !matches!(p.kind, PieceKind::Scarecrow | PieceKind::Football)
                            && state.card_runtime.monster_rule != Some(p.id)
                    })
                    .map(|(_, p)| p.id)
                    .collect();
                for i in 0..pool.len().min(3) {
                    let index = rule_random(
                        state,
                        events,
                        pool.len(),
                        &format!("rule.winter.{cycle}.{owner}.{i}"),
                    )
                    .unwrap();
                    selected.push(pool.remove(index));
                }
            }
            if actor == Player::White {
                state.rule_runtime.winter_preview = selected;
                state.rule_runtime.winter_preview_cycle = Some(cycle);
                return;
            }
            state.rule_runtime.winter_frozen = selected.into_iter().collect();
        }
        state.rule_runtime.winter_preview.clear();
        state.rule_runtime.winter_preview_cycle = None;
        state.rule_runtime.winter_cycle = cycle;
    }
}

impl Engine {
    pub(in crate::engine) fn install_winter_kingdom(
        &self,
        state: &mut GameState,
        _events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        state.rule_runtime.winter_cycle = 0;
        state.rule_runtime.winter_disabled = false;
        Ok(())
    }
}

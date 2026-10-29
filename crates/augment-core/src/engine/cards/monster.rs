use super::*;
pub(super) const ID: &str = crate::MONSTER_RULE;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_monster(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn settle_rule_monster(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) {
        if !rule_active(state, ID) {
            return;
        }
        let halfmove = rule_halfmove(state);
        if halfmove == 0
            || halfmove % 3 != 0
            || state.rule_runtime.monster_last_halfmove == halfmove
        {
            return;
        }
        state.rule_runtime.monster_last_halfmove = halfmove;
        let Some(id) = state.card_runtime.monster_rule else {
            return;
        };
        let Some((from, monster)) = state.board.find_by_id(id) else {
            return;
        };
        let mut candidates = Vec::new();
        for (df, dr) in [
            (-1, 1),
            (0, 1),
            (1, 1),
            (-1, 0),
            (1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ] {
            let Some(to) = from.offset(df, dr) else {
                continue;
            };
            if self.rule_cell_collapsed(state, to)
                || state.card_runtime.black_holes.contains(&to)
                || state.card_runtime.barricades.contains(&to)
                || state.card_runtime.crown_pos == Some(to)
                || state.card_runtime.football_pos == Some(to)
                || state
                    .card_runtime
                    .rule_portals
                    .is_some_and(|pair| pair.contains(&to))
                || state
                    .card_runtime
                    .pending_lobsters
                    .iter()
                    .any(|p| p.square == to)
                || state
                    .card_runtime
                    .pending_scarecrows
                    .iter()
                    .any(|p| p.square == to)
            {
                continue;
            }
            if self.board_piece_at(state, to).is_none() {
                if self.card_landing_open(state, to) {
                    candidates.push(to);
                }
                continue;
            }
            let target = self.board_piece_at(state, to).unwrap();
            if matches!(
                target.kind,
                PieceKind::Guard
                    | PieceKind::Monster
                    | PieceKind::Scarecrow
                    | PieceKind::DarkWizard
                    | PieceKind::RevolvingDoor
            ) || state.rule_runtime.winter_frozen.contains(&target.id)
                || state.card_runtime.frozen.contains_key(&target.id)
                || state.card_runtime.protected_piece_ids.contains(&target.id)
                || state
                    .card_runtime
                    .outpost_protected_piece_ids
                    .contains(&target.id)
                || state
                    .card_runtime
                    .sacrifice_protection
                    .contains_key(&target.id)
                || state
                    .card_runtime
                    .coronation_protection
                    .contains_key(&target.id)
                || self.middle_encouragement_protects(state, target)
                || self.middle_submerged_protects(state, target)
                || state.card_runtime.high_ground_squares.contains(&to)
                    && !state.card_runtime.high_ground_squares.contains(&from)
            {
                continue;
            }
            candidates.push(to);
        }
        if candidates.is_empty() {
            return;
        }
        let i = rule_random(state, events, candidates.len(), "rule.monster.move").unwrap();
        let to = candidates[i];
        if let Some((anchor, victim)) = self.board_piece_entry_at(state, to) {
            if self.fracture_large_cell(state, victim, to) != Some(false) {
                self.remove_piece_for_card(state, anchor, victim, state.turn, ID, events);
            }
        }
        state.board.set(from, None);
        state.board.set(to, Some(monster));
        events.push(Event::EffectApplied {
            source_card_instance: 0,
            command: format!("rule-monster:{from}:{to}"),
        });
    }
}

impl Engine {
    pub(in crate::engine) fn install_monster(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let square = self.current_rule_spawn_cell(state, events, ID)?;
        let piece =
            Piece::new(Player::White, PieceKind::Monster).with_id(state.allocate_piece_id());
        state.board.set(square, Some(piece));
        state.card_runtime.monster_rule = Some(piece.id);
        state.card_runtime.moved_piece_ids.insert(piece.id);
        Ok(())
    }
}

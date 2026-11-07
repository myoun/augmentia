use super::*;
pub(super) const ID: &str = crate::FOOTBALL;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_football(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn rule_football_actions(
        &self,
        state: &GameState,
        actions: &mut Vec<Action>,
    ) {
        if !rule_active(state, ID) {
            return;
        }
        let Some(from) = state.card_runtime.football_pos else {
            return;
        };
        let corner = matches!(from.file(), 0 | 7) && matches!(from.rank(), 0 | 7);
        for (df, dr) in [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ] {
            let kicker = from
                .offset(-df, -dr)
                .and_then(|s| self.board_piece_at(state, s));
            let enabled = kicker.is_some_and(|p| {
                p.owner == state.turn
                    && !state.card_runtime.frozen.contains_key(&p.id)
                    && !state.rule_runtime.winter_frozen.contains(&p.id)
                    && !state.card_runtime.poison_stun_turns.contains_key(&p.id)
            });
            let mut cursor = from;
            while let Some(to) = cursor.offset(df, dr) {
                if self.rule_cell_collapsed(state, to)
                    || state.card_runtime.barricades.contains(&to)
                    || state.card_runtime.black_holes.contains(&to)
                    || state.card_runtime.crown_pos == Some(to)
                {
                    break;
                }
                let target = self.board_piece_at(state, to);
                let legal = match target {
                    None => enabled || corner && cursor == from,
                    Some(p) => {
                        enabled
                            && p.owner != state.turn
                            && !matches!(p.kind, PieceKind::Monster | PieceKind::Football)
                            && p.kind != PieceKind::Jester
                            && !state.card_runtime.protected_piece_ids.contains(&p.id)
                            && !state
                                .card_runtime
                                .outpost_protected_piece_ids
                                .contains(&p.id)
                            && !state.card_runtime.frozen.contains_key(&p.id)
                            && !state.rule_runtime.winter_frozen.contains(&p.id)
                            && !self.middle_encouragement_protects(state, p)
                            && !self.middle_submerged_protects(state, p)
                            && !state.card_runtime.sacrifice_protection.contains_key(&p.id)
                            && !state.card_runtime.coronation_protection.contains_key(&p.id)
                            && state.card_runtime.armistice_remaining_ply == 0
                    }
                };
                if legal {
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                }
                if target.is_some() || !enabled {
                    break;
                }
                cursor = to;
            }
        }
    }

    pub(in crate::engine) fn apply_rule_football(
        &self,
        state: &GameState,
        action: Action,
        movement: MoveAction,
    ) -> Result<Transition, EngineError> {
        let mut next = state.clone();
        let actor = state.turn;
        let mut events = vec![Event::ActionApplied { actor, action }];
        let mut impact = false;
        if let Some((anchor, victim)) = self.board_piece_entry_at(&next, movement.to) {
            if next.card_runtime.shielded_piece_ids.remove(&victim.id)
                || victim.kind == PieceKind::Guard
            {
                impact = true;
            } else if let Some(defeated) = self.fracture_large_cell(&mut next, victim, movement.to)
            {
                impact = true;
                if defeated {
                    self.remove_piece_for_card(&mut next, anchor, victim, actor, ID, &mut events);
                }
            } else if let Some(health) = next.card_runtime.health_mut(victim.id) {
                impact = true;
                if health.take_hit() {
                    self.remove_piece_for_card(&mut next, anchor, victim, actor, ID, &mut events);
                }
            } else {
                self.remove_piece_for_card(&mut next, anchor, victim, actor, ID, &mut events);
            }
        }
        if !impact {
            next.card_runtime.football_pos = Some(movement.to);
        }
        next.ply = next.ply.saturating_add(1);
        next.actions_remaining = next.actions_remaining.saturating_sub(1);
        if impact {
            next.actions_remaining = 0;
        }
        next.quiet_ply = 0;
        next.en_passant = None;
        events.push(Event::EffectApplied {
            source_card_instance: 0,
            command: format!("football:{}:{}", movement.from, movement.to),
        });
        self.current_rule_arrivals(&mut next, &mut events);
        self.resolve_missing_king(&mut next, &mut events);
        if next.status != GameStatus::Ongoing {
            self.push_game_ended(&next, &mut events);
        } else if next.actions_remaining == 0 {
            self.finish_turn(&mut next, actor, &mut events)?;
        } else {
            self.resolve_no_action_loss(&mut next, actor, &mut events);
        }
        Ok(Transition {
            state: next,
            events,
        })
    }
}

impl Engine {
    pub(in crate::engine) fn install_football(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let square = self.current_rule_spawn_cell(state, events, ID)?;
        state.card_runtime.football_pos = Some(square);
        Ok(())
    }
}

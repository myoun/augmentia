use super::*;
pub(super) const ID: &str = crate::RULE_TICKET;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
const EXCLUDED: &[&str] = &[
    crate::CHESS_960,
    crate::CHESS_344200,
    crate::CHESS_N_POW_30,
    crate::DIAGONAL_CHESS,
];
fn candidates(state: &GameState) -> Vec<&'static str> {
    if state
        .cards
        .iter()
        .any(|card| card.definition_id == crate::REVERSE_PAWNS)
        || state
            .applied_rules
            .iter()
            .any(|rule| rule == crate::REVERSE_PAWNS)
    {
        return Vec::new();
    }
    crate::latest_catalog()
        .iter()
        .filter(|card| {
            card.enabled
                && card.phase == "RULE"
                && card.activation == "ACTIVE"
                && card.supported()
                && !EXCLUDED.contains(&card.id.as_str())
                && !state.applied_rules.iter().any(|rule| rule == &card.id)
                && !state.rule_runtime.enabled.contains(&card.id)
                && !state
                    .card_runtime
                    .pending_rule_tickets
                    .iter()
                    .any(|ticket| ticket.rule_id == card.id)
        })
        .map(|card| card.id.as_str())
        .collect()
}
fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for rule_id in candidates(state) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Choice(rule_id.to_owned()),
        });
    }
}
fn eligibility(state: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        !candidates(state).is_empty(),
        "선택할 수 있는 규칙이 필요합니다.",
    )
}
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    source: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let ActionTarget::Choice(rule_id) = target else {
        return Err(EngineError::Invariant(
            "rule ticket requires a named rule".into(),
        ));
    };
    if !candidates(state).contains(&rule_id.as_str()) {
        return Err(EngineError::Invariant("invalid rule ticket choice".into()));
    }
    let mut probe = state.clone();
    probe.rule_runtime.ticket_random_record = Some(Vec::new());
    let _ = engine.install_rule(&mut probe, rule_id, &mut Vec::new());
    let plan = probe
        .rule_runtime
        .ticket_random_record
        .take()
        .unwrap_or_default();
    let mut random_values = Vec::with_capacity(plan.len());
    for (bound, predicted) in plan {
        let (roll, index) = state.rng_mut().web_choice_index(bound);
        let value = index.expect("recorded random request is nonempty");
        debug_assert_eq!(value, predicted);
        random_values.push(value);
        events.push(Event::RandomSample {
            purpose: format!("rule-ticket:{rule_id}:{}", random_values.len() - 1),
            roll_53: roll,
        });
    }
    state
        .card_runtime
        .pending_rule_tickets
        .push(PendingRuleTicket {
            owner: actor,
            source_card_instance: source,
            due_turn: state.turns_taken[actor.index()].saturating_add(1),
            rule_id: rule_id.clone(),
            random_values,
        });
    Ok(())
}
impl Engine {
    pub(in crate::engine) fn settle_rule_tickets(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let pending = std::mem::take(&mut state.card_runtime.pending_rule_tickets);
        for ticket in pending {
            if ticket.owner != owner || state.turns_taken[owner.index()] < ticket.due_turn {
                state.card_runtime.pending_rule_tickets.push(ticket);
                continue;
            }
            if !state
                .cards
                .iter()
                .any(|card| card.instance_id == ticket.source_card_instance && card.owner == owner)
            {
                continue;
            }
            let mut probe = state.clone();
            probe.rule_runtime.ticket_random_replay = Some((ticket.random_values, 0, false));
            let mut applied_events = Vec::new();
            let applied = self
                .install_rule(&mut probe, &ticket.rule_id, &mut applied_events)
                .is_ok();
            let replay = probe
                .rule_runtime
                .ticket_random_replay
                .take()
                .expect("replay is active");
            if applied && !replay.2 {
                *state = probe;
                events.extend(applied_events);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    fn choice(engine: &Engine, state: &GameState) -> Vec<String> {
        engine
            .legal_actions(state)
            .unwrap()
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::Choice(rule),
                    ..
                } => Some(rule),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn rule_ticket_offers_active_rules_and_excludes_layout_rules() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("d4", Player::White, PieceKind::Rook)]);
        state = engine
            .grant_card(&state, 1, crate::RULE_TICKET, Player::White)
            .unwrap()
            .state;
        let offered = choice(&engine, &state);
        assert!(!offered.is_empty(), "a ticket must offer rule choices");
        for excluded in [
            crate::CHESS_960,
            crate::CHESS_344200,
            crate::CHESS_N_POW_30,
            crate::DIAGONAL_CHESS,
        ] {
            assert!(
                !offered.iter().any(|rule| rule == excluded),
                "{excluded} is a fixed-layout rule and must not be offered"
            );
        }
        assert!(offered.iter().any(|rule| rule == crate::ACCELERATION));
    }

    #[test]
    fn rule_ticket_installs_on_the_next_owner_turn_not_immediately() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("d4", Player::White, PieceKind::Rook)]);
        use_at(
            &mut state,
            &engine,
            crate::RULE_TICKET,
            ActionTarget::Choice(crate::ACCELERATION.to_owned()),
        );
        assert!(
            !state
                .applied_rules
                .iter()
                .any(|rule| rule == crate::ACCELERATION),
            "the rule must not install on the turn the ticket is used"
        );
        assert_eq!(state.card_runtime.pending_rule_tickets.len(), 1);
        // The ticket settles at the owner's next turn start, so the immediate
        // opponent move is not enough.
        handoff(&engine, &mut state);
        assert!(
            !state
                .applied_rules
                .iter()
                .any(|rule| rule == crate::ACCELERATION)
        );
        handoff(&engine, &mut state);
        assert!(
            state
                .applied_rules
                .iter()
                .any(|rule| rule == crate::ACCELERATION),
            "the ticket must install at the owner's next turn start"
        );
        assert!(state.card_runtime.pending_rule_tickets.is_empty());
    }

    #[test]
    fn rule_ticket_removes_installed_rules_from_later_choice_lists() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("d4", Player::White, PieceKind::Rook)]);
        state = engine
            .grant_card(&state, 1, crate::RULE_TICKET, Player::White)
            .unwrap()
            .state;
        let picked = choice(&engine, &state)[0].clone();
        state = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Choice(picked.clone()),
                },
            )
            .unwrap()
            .state;
        assert_eq!(state.card_runtime.pending_rule_tickets.len(), 1);
        // A rule that is already queued must not be offered again.
        if let Ok(granted) = engine.grant_card(&state, 2, crate::RULE_TICKET, Player::White) {
            let after = choice(&engine, &granted.state);
            assert!(
                !after.iter().any(|rule| rule == &picked),
                "a pending rule must not be offered again"
            );
        }
    }

    #[test]
    fn rule_ticket_rejects_choices_outside_the_offered_pool() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(&[("d4", Player::White, PieceKind::Rook)]);
        let state = engine
            .grant_card(&state, 1, crate::RULE_TICKET, Player::White)
            .unwrap()
            .state;
        let rejected = engine.apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::Choice(crate::CHESS_960.to_owned()),
            },
        );
        assert!(rejected.is_err(), "an excluded rule must be rejected");
    }
}

use super::*;
pub(super) const ID: &str = crate::BLACK_BOX;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Upstream `V12_BOX_EXCLUDED`: cards that must never be triggered from a box.
/// `black-box` is here to break recursion; the rest either rewrite the board
/// layout or install persistent state that cannot be applied atomically.
const EXCLUDED: &[&str] = &[
    crate::BIG_BISHOP,
    crate::BLACK_BOX,
    crate::WHITE_BOX,
    // `shotgun-king` is DLC-only and absent from this catalog, so there is no
    // card id constant for it. Keep the upstream guard as a literal anyway.
    "shotgun-king",
    crate::HORDE,
    crate::LONDON_SYSTEM,
    crate::BIG_ROOK,
    crate::INITIATIVE,
];

/// Cards the v14+ contract additionally excludes from the trigger pool: they
/// open a follow-up selection or replay an already used card.
const DEFERRED: &[&str] = &[
    crate::JOKER,
    crate::RULE_TICKET,
    crate::PREMOVE,
    crate::SACRIFICE,
];

fn owned_ids(state: &GameState) -> BTreeSet<&str> {
    state
        .cards
        .iter()
        .map(|card| card.definition_id.as_str())
        .collect()
}

/// Every card a box may trigger, in catalog order. The caller shuffles it.
fn candidates(state: &GameState, actor: Player) -> Vec<&'static str> {
    let owned = owned_ids(state);
    let actor_cards: Vec<&str> = state
        .cards
        .iter()
        .filter(|card| card.owner == actor)
        .map(|card| card.definition_id.as_str())
        .collect();
    crate::latest_catalog()
        .iter()
        .filter(|card| {
            card.enabled
                && card.phase != "RULE"
                && card.activation == "ACTIVE"
                && card.supported()
                && !EXCLUDED.contains(&card.id.as_str())
                && !DEFERRED.contains(&card.id.as_str())
                && !owned.contains(card.id.as_str())
                && actor_cards
                    .iter()
                    .all(|owned_id| !crate::mutually_exclusive_cards(&card.id, owned_id))
        })
        .map(|card| card.id.as_str())
        .collect()
}

fn shuffle<T>(state: &mut GameState, values: &mut [T], purpose: &str, events: &mut Vec<Event>) {
    for index in (1..values.len()).rev() {
        let (roll, selected) = state.rng_mut().web_choice_index(index + 1);
        events.push(Event::RandomSample {
            purpose: purpose.to_owned(),
            roll_53: roll,
        });
        values.swap(index, selected.expect("shuffle prefix is non-empty"));
    }
}

/// The nested card runs against a probe that already carries it as the source
/// instance, so its own `play` can look itself up by `instance_id`. A failure
/// discards the probe, which keeps the outer transition atomic.
fn try_trigger(
    engine: &Engine,
    state: &GameState,
    actor: Player,
    source: u32,
    card_id: &str,
    events: &mut Vec<Event>,
) -> Option<GameState> {
    let mut probe = state.clone();
    match probe
        .cards
        .iter_mut()
        .find(|card| card.instance_id == source)
    {
        Some(entry) => {
            entry.definition_id = card_id.to_owned();
            entry.charges = 1;
        }
        None => probe.add_card(CardInstance {
            instance_id: source,
            definition_id: card_id.to_owned(),
            owner: actor,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        }),
    }
    let virtual_card = CardInstance {
        instance_id: source,
        definition_id: card_id.to_owned(),
        owner: actor,
        half_stars: 0,
        charges: 1,
        available_after_turn: None,
    };
    let mut generated = Vec::new();
    super::actions(engine, &probe, &virtual_card, &mut generated);
    let mut generated: Vec<_> = generated
        .into_iter()
        .filter(|action| matches!(action, Action::UseCard { instance_id, .. } if *instance_id == source))
        .collect();
    if generated.is_empty() {
        return None;
    }
    shuffle(&mut probe, &mut generated, "black-box:target", events);
    for action in generated {
        let Action::UseCard { target, .. } = action else {
            continue;
        };
        let mut attempt = probe.clone();
        let mut attempt_events = Vec::new();
        if super::play(
            engine,
            &mut attempt,
            card_id,
            actor,
            source,
            &target,
            &mut attempt_events,
        )
        .is_err()
        {
            continue;
        }
        // The nested card keeps its own turn policy: an END_TURN trigger must
        // still end the outer turn.
        if engine
            .card_definition(card_id)
            .is_some_and(|definition| definition.ends_turn)
        {
            attempt.request_turn_end();
        }
        // The probe temporarily re-labelled the source instance as the nested
        // card and gave it a charge. Restore the real box state, because the
        // outer engine already consumed the real charge before calling `play`.
        if let Some(entry) = attempt
            .cards
            .iter_mut()
            .find(|entry| entry.instance_id == source)
        {
            entry.definition_id = ID.to_owned();
            entry.charges = state
                .cards
                .iter()
                .find(|entry| entry.instance_id == source)
                .map_or(0, |entry| entry.charges);
        }
        events.extend(attempt_events);
        return Some(attempt);
    }
    None
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    source: u32,
    _: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let mut pool = candidates(state, actor);
    if pool.is_empty() {
        return Err(EngineError::Invariant(
            "발동할 수 있는 액티브 카드가 없습니다.".into(),
        ));
    }
    shuffle(state, &mut pool, "black-box", events);
    for card_id in pool {
        let Some(triggered) = try_trigger(engine, state, actor, source, card_id, events) else {
            continue;
        };
        *state = triggered;
        events.push(Event::EffectApplied {
            source_card_instance: source,
            command: format!("black-box:{card_id}"),
        });
        return Ok(());
    }
    Err(EngineError::Invariant(
        "발동 가능한 액티브 카드를 찾지 못했습니다.".into(),
    ))
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if state.turn != card.owner {
        return;
    }
    if candidates(state, card.owner).is_empty() {
        return;
    }
    out.push(Action::UseCard {
        instance_id: card.instance_id,
        target: ActionTarget::None,
    });
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        !candidates(state, owner).is_empty(),
        "발동할 수 있는 액티브 카드가 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    fn card(instance_id: u32) -> CardInstance {
        CardInstance {
            instance_id,
            definition_id: crate::BLACK_BOX.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        }
    }

    fn with_black_box(pieces: &[(&str, Player, PieceKind)]) -> (Engine, GameState) {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(pieces);
        state.add_card(card(1));
        (engine, state)
    }

    fn triggered(events: &[Event]) -> Option<String> {
        events.iter().find_map(|event| match event {
            Event::EffectApplied { command, .. } => {
                command.strip_prefix("black-box:").map(str::to_owned)
            }
            _ => None,
        })
    }

    #[test]
    fn black_box_is_reachable_through_the_engine_and_consumes_its_charge() {
        let (engine, state) = with_black_box(&[
            ("d4", Player::White, PieceKind::Rook),
            ("d8", Player::Black, PieceKind::Queen),
        ]);
        let offered = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .filter(|action| matches!(action, Action::UseCard { instance_id: 1, .. }))
            .collect::<Vec<_>>();
        assert_eq!(
            offered.len(),
            1,
            "the box offers exactly one targetless use"
        );
        let action = offered.into_iter().next().unwrap();
        let transition = engine.apply(&state, action).unwrap();
        assert_eq!(
            transition.state.cards[0].definition_id,
            crate::BLACK_BOX,
            "the source instance stays black-box after use"
        );
        assert_eq!(
            transition.state.cards[0].charges, 0,
            "the box must be consumed"
        );
        assert!(
            triggered(&transition.events).is_some(),
            "an end-to-end use records the reveal"
        );
    }

    #[test]
    fn black_box_is_not_offered_when_nothing_can_be_triggered() {
        let (engine, mut state) = with_black_box(&[]);
        let mut instance = 100;
        for id in candidates(&state, Player::White) {
            instance += 1;
            state.add_card(CardInstance {
                instance_id: instance,
                definition_id: id.to_owned(),
                owner: Player::White,
                half_stars: 0,
                charges: 0,
                available_after_turn: None,
            });
        }
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .iter()
                .any(|action| matches!(action, Action::UseCard { instance_id: 1, .. })),
            "an empty pool must not be offered"
        );
    }

    #[test]
    fn black_box_candidate_pool_is_active_non_rule_supported_and_excludes_the_guard_set() {
        let (_, state) = with_black_box(&[]);
        let pool = candidates(&state, Player::White);
        assert!(
            !pool.is_empty(),
            "the pool must not be empty on a standard draft"
        );
        for excluded in EXCLUDED.iter().chain(DEFERRED.iter()) {
            assert!(
                !pool.contains(excluded),
                "{excluded} must never be triggered from a box"
            );
        }
        for id in &pool {
            let entry = crate::latest_catalog()
                .iter()
                .find(|card| card.id == *id)
                .expect("candidate comes from the catalog");
            assert!(entry.supported(), "{id} must be an executable card");
            assert_eq!(entry.activation, "ACTIVE", "{id} must be active");
            assert_ne!(entry.phase, "RULE", "{id} must not install a rule");
        }
    }

    #[test]
    fn black_box_pool_respects_already_acquired_cards_and_exclusivity() {
        let (_, mut state) = with_black_box(&[]);
        let before = candidates(&state, Player::White);
        let probe = before[0];
        state.add_card(CardInstance {
            instance_id: 9,
            definition_id: probe.to_owned(),
            owner: Player::Black,
            half_stars: 0,
            charges: 0,
            available_after_turn: None,
        });
        let after = candidates(&state, Player::White);
        assert!(
            !after.contains(&probe),
            "a card already in either inventory must not be triggered again"
        );
        // The upstream pool also enforces mutual exclusion against the actor's own
        // cards; every remaining candidate must survive that filter.
        for id in &after {
            for owned in state
                .cards
                .iter()
                .filter(|card| card.owner == Player::White)
                .map(|card| card.definition_id.as_str())
            {
                assert!(
                    !crate::mutually_exclusive_cards(id, owned),
                    "{id} conflicts with the owned {owned}"
                );
            }
        }
    }

    #[test]
    fn black_box_triggers_one_random_candidate_and_records_the_reveal() {
        let (engine, base) = with_black_box(&[
            ("d4", Player::White, PieceKind::Rook),
            ("d8", Player::Black, PieceKind::Queen),
        ]);
        let mut revealed = BTreeSet::new();
        for seed in 0..48_u64 {
            let mut state = base.clone();
            *state.rng_mut() = crate::DeterministicRng::new(seed);
            let mut events = Vec::new();
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &ActionTarget::None,
                &mut events,
            )
            .unwrap();
            let card_id = triggered(&events).expect("the reveal must be recorded");
            assert!(
                candidates(&base, Player::White).contains(&card_id.as_str()),
                "{card_id} is not a legal box candidate"
            );
            assert_eq!(
                state
                    .cards
                    .iter()
                    .find(|entry| entry.instance_id == 1)
                    .unwrap()
                    .definition_id,
                crate::BLACK_BOX,
                "the source instance stays black-box"
            );
            assert!(
                !state
                    .cards
                    .iter()
                    .any(|entry| entry.definition_id == card_id && entry.instance_id != 1),
                "the triggered card must not join the inventory"
            );
            assert!(
                events
                    .iter()
                    .any(|event| matches!(event, Event::RandomSample { purpose, .. } if purpose == "black-box")),
                "the candidate draw must be recorded"
            );
            revealed.insert(card_id);
        }
        assert!(
            revealed.len() > 1,
            "different seeds must reveal different cards: {revealed:?}"
        );
    }

    #[test]
    fn black_box_never_reveals_an_excluded_or_unimplemented_card() {
        let (engine, base) = with_black_box(&[]);
        for seed in 0..400_u64 {
            let mut state = base.clone();
            *state.rng_mut() = crate::DeterministicRng::new(seed);
            let mut events = Vec::new();
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &ActionTarget::None,
                &mut events,
            )
            .unwrap();
            let card_id = triggered(&events).expect("the reveal must be recorded");
            assert!(
                !EXCLUDED.contains(&card_id.as_str()),
                "{card_id} is guarded"
            );
            assert!(
                !DEFERRED.contains(&card_id.as_str()),
                "{card_id} is deferred"
            );
            assert!(
                crate::latest_catalog()
                    .iter()
                    .find(|card| card.id == card_id)
                    .is_some_and(|entry| entry.supported()),
                "{card_id} must be executable"
            );
        }
    }

    #[test]
    fn black_box_leaves_state_untouched_when_no_candidate_can_apply() {
        let (engine, mut state) = with_black_box(&[]);
        // Own every box candidate: the pool is empty, so the trigger fails.
        let mut instance = 100;
        for id in candidates(&state, Player::White) {
            instance += 1;
            state.add_card(CardInstance {
                instance_id: instance,
                definition_id: id.to_owned(),
                owner: Player::White,
                half_stars: 0,
                charges: 0,
                available_after_turn: None,
            });
        }
        assert!(candidates(&state, Player::White).is_empty());
        let before = state.clone();
        let mut events = Vec::new();
        let error = play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::None,
            &mut events,
        );
        assert!(error.is_err(), "an empty pool must fail the trigger");
        assert_eq!(state, before, "a failed trigger must not corrupt state");
        assert!(events.is_empty(), "a failed trigger emits nothing");
    }

    #[test]
    fn black_box_nested_end_turn_card_still_ends_the_outer_turn() {
        let (engine, mut state) = with_black_box(&[]);
        // Force the only candidate to be an END_TURN card by pruning the catalog
        // through ownership: everything else is held by the opponent. `miracle`
        // is the reliable one: it takes no target and always succeeds.
        let keep = crate::MIRACLE;
        let mut instance = 100;
        for id in candidates(&state, Player::White) {
            if id == keep {
                continue;
            }
            instance += 1;
            state.add_card(CardInstance {
                instance_id: instance,
                definition_id: id.to_owned(),
                owner: Player::Black,
                half_stars: 0,
                charges: 0,
                available_after_turn: None,
            });
        }
        let pool = candidates(&state, Player::White);
        assert_eq!(
            pool,
            vec![keep],
            "pruning must leave only the END_TURN card"
        );
        assert!(
            engine
                .card_definition(keep)
                .is_some_and(|definition| definition.ends_turn),
            "{keep} must end the turn"
        );
        let mut events = Vec::new();
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::None,
            &mut events,
        )
        .unwrap();
        assert_eq!(triggered(&events).as_deref(), Some(keep));
        assert!(
            state.take_turn_end_request(),
            "an END_TURN trigger must request the outer turn end"
        );
    }
}

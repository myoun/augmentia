use super::*;
pub(super) const ID: &str = crate::CAPTURE_THE_FLAG;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    engine.install_capture_the_flag(state, events);
    Ok(())
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    // Upstream `v19DraftRequirementMet` allows ANY unconditionally.
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, eligibility);

impl Engine {
    pub(in crate::engine) fn install_capture_the_flag(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) {
        let mut flags = [Square::new(0, 0).unwrap(), Square::new(0, 7).unwrap()];
        for owner in [Player::White, Player::Black] {
            let (roll, file, replayed) = rule_sample(state, 8);
            flags[owner.index()] = Square::new(
                file.unwrap() as u8,
                if owner == Player::White { 0 } else { 7 },
            )
            .unwrap();
            if !replayed {
                events.push(Event::RandomSample {
                    purpose: format!("capture-the-flag:{owner}"),
                    roll_53: roll,
                });
            }
        }
        state.card_runtime.capture_the_flag = Some(crate::CaptureTheFlagState {
            flags,
            occupations: [None, None],
        });
    }
}

impl Engine {
    pub(in crate::engine) fn advance_capture_flags(
        &self,
        state: &mut GameState,
        actor: Player,
        completed: bool,
    ) {
        if state.status != GameStatus::Ongoing {
            return;
        }
        let Some(rule) = &mut state.card_runtime.capture_the_flag else {
            return;
        };
        let mut defeated = Vec::new();
        for owner in [Player::White, Player::Black] {
            let index = owner.index();
            let occupant = state
                .board
                .get(rule.flags[index])
                .filter(|piece| piece.owner != owner);
            let Some(piece) = occupant else {
                rule.occupations[index] = None;
                continue;
            };
            match rule.occupations[index] {
                Some(previous) if previous.piece_id == piece.id => {
                    if state.turns_taken[index] > previous.after_owner_turn {
                        defeated.push(owner);
                    }
                }
                _ => {
                    let before = state.turns_taken[index]
                        .saturating_sub(u32::from(completed && actor == owner));
                    rule.occupations[index] = Some(crate::FlagOccupation {
                        piece_id: piece.id,
                        after_owner_turn: before.saturating_add(u32::from(actor == owner)),
                    });
                }
            }
        }
        if defeated.len() == 2 {
            state.status = GameStatus::Draw {
                reason: DrawReason::CardEffect(crate::CAPTURE_THE_FLAG.to_owned()),
            };
        } else if let Some(owner) = defeated.first() {
            state.status = GameStatus::Won {
                winner: owner.opponent(),
                reason: WinReason::CardEffect(crate::CAPTURE_THE_FLAG.to_owned()),
            };
        }
    }
}

#[cfg(test)]
mod historical_contract_tests {
    use super::super::tests::board_objectives::*;
    use super::*;
    #[test]
    fn capture_the_flag_defender_has_one_turn_to_liberate() {
        let engine = Engine::bundled_latest().unwrap();
        let state = flag_test_state();
        let occupied = engine.apply(&state, flag_move("a2", "a1")).unwrap().state;
        assert_eq!(occupied.status, GameStatus::Ongoing);
        let lost = engine
            .apply(&occupied, flag_move("h1", "g1"))
            .unwrap()
            .state;
        assert_eq!(
            lost.status,
            GameStatus::Won {
                winner: Player::Black,
                reason: WinReason::CardEffect("capture-the-flag".to_owned()),
            }
        );
        let freed = engine
            .apply(&occupied, flag_move("b1", "a1"))
            .unwrap()
            .state;
        assert_eq!(freed.status, GameStatus::Ongoing);
        assert!(freed.card_runtime.capture_the_flag.unwrap().occupations[0].is_none());
        let fingerprint = occupied.fingerprint();
        assert!(engine.apply(&occupied, flag_move("a2", "a1")).is_err());
        assert_eq!(occupied.fingerprint(), fingerprint);
    }
    #[test]
    fn capture_the_flag_installs_seeded_home_flags() {
        let engine = Engine::bundled_latest().unwrap();
        let state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 915);
        let first = engine.apply_rule(&state, "capture-the-flag").unwrap();
        let second = engine.apply_rule(&state, "capture-the-flag").unwrap();
        assert_eq!(first.state.fingerprint(), second.state.fingerprint());
        assert_eq!(first.events, second.events);
        assert_ne!(state.fingerprint(), first.state.fingerprint());
        let flags = first.state.card_runtime.capture_the_flag.as_ref().unwrap();
        assert_eq!(flags.flags[0].rank(), 0);
        assert_eq!(flags.flags[1].rank(), 7);
        assert_eq!(flags.occupations, [None, None]);
        assert_eq!(
            first
                .events
                .iter()
                .filter(|e| matches!(e, Event::RandomSample { .. }))
                .count(),
            2
        );
        assert!(engine.apply_rule(&first.state, "capture-the-flag").is_err());
    }
    #[test]
    fn capture_the_flag_fingerprint_and_replay_restore_deadline() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = flag_test_state();
        state.actions_remaining = 2;
        let occupied = engine.apply(&state, flag_move("a2", "a1")).unwrap().state;
        let flags = occupied.card_runtime.capture_the_flag.as_ref().unwrap();
        assert!(flags.occupations[0].is_some());
        let mut changed = occupied.clone();
        changed
            .card_runtime
            .capture_the_flag
            .as_mut()
            .unwrap()
            .occupations[0]
            .as_mut()
            .unwrap()
            .after_owner_turn += 1;
        assert_ne!(changed.position_key(), occupied.position_key());
        assert_ne!(changed.fingerprint(), occupied.fingerprint());
        let mut changed_cell = occupied.clone();
        changed_cell
            .card_runtime
            .capture_the_flag
            .as_mut()
            .unwrap()
            .flags[0] = "c1".parse().unwrap();
        assert_ne!(changed_cell.fingerprint(), occupied.fingerprint());
        let ready = engine
            .grant_card(&occupied, 915, crate::REPLAY, Player::Black)
            .unwrap()
            .state;
        let restored = engine
            .apply(
                &ready,
                Action::UseCard {
                    instance_id: 915,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
        assert_eq!(
            restored.card_runtime.capture_the_flag,
            state.card_runtime.capture_the_flag
        );
        assert_eq!(restored.board, state.board);
    }
}

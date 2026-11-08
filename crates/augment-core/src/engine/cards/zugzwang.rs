use super::*;

pub(super) const ID: &str = crate::ZUGZWANG;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// A "king move" in the upstream sense: `isWorkerZugzwangKingMove`
/// (main-RjrZ5fFP.js:232470-232472) excludes castling and the swap-style arrivals,
/// so a stuck king that can only castle is not a zugzwang candidate.
fn is_plain_king_move(state: &GameState, king: Piece, movement: MoveAction) -> bool {
    let home_rank = if king.owner == Player::White { 0 } else { 7 };
    let castling = king.kind == PieceKind::King
        && movement.from.file() == 4
        && movement.from.rank() == home_rank
        && movement.to.rank() == home_rank
        && matches!(movement.to.file(), 2 | 6);
    // Swaps and castling-partner arrivals land on a friendly piece; a plain step never does.
    let onto_friendly = state
        .board
        .get(movement.to)
        .is_some_and(|target| target.owner == king.owner);
    !castling && !onto_friendly
}

/// Mirrors `workerZugzwangKingMoves` (main-RjrZ5fFP.js:232473-232481) by asking the
/// engine for the opponent's fully filtered legal moves on a probe of the same
/// position with the flag cleared. Returns the number of destinations that are a
/// legal arrival for one of the opponent's royals.
fn enemy_king_move_count(engine: &Engine, state: &GameState, enemy: Player) -> usize {
    let mut probe = state.clone();
    probe.turn = enemy;
    probe.actions_remaining = probe.actions_remaining.max(1);
    probe.card_runtime.zugzwang[enemy.index()] = false;
    let royals: Vec<(Square, Piece)> = probe
        .board
        .iter()
        .filter(|(_, piece)| piece.owner == enemy && engine.is_royal_piece(&probe, *piece))
        .collect();
    let Ok(actions) = engine.legal_actions(&probe) else {
        return 0;
    };
    actions
        .iter()
        .filter(|action| {
            let Action::Move(movement) = action else {
                return false;
            };
            let Some((_, king)) = royals.iter().find(|(square, _)| *square == movement.from) else {
                return false;
            };
            is_plain_king_move(&probe, *king, *movement)
        })
        .count()
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    if !matches!(target, ActionTarget::None) {
        return Err(EngineError::Invariant(
            "추크츠방은 대상을 선택하지 않습니다.".to_owned(),
        ));
    }
    let enemy = actor.opponent();
    if enemy_king_move_count(engine, state, enemy) == 0 {
        return Err(EngineError::Invariant(
            "상대 킹이 움직일 수 없어 추크츠방을 사용할 수 없습니다.".to_owned(),
        ));
    }
    // `zugzwang()` (main-RjrZ5fFP.js:348808-348814) only arms the opponent; the shared
    // engine enforces the flag in `generate_moves_for_piece` and expires it when that
    // opponent finishes their turn (`expireZugzwangAfterCompletedTurn`, L4740-4747).
    state.card_runtime.zugzwang[enemy.index()] = true;
    events.push(Event::EffectApplied {
        source_card_instance: instance_id,
        command: format!("zugzwang:{enemy:?}"),
    });
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if enemy_king_move_count(engine, state, state.turn.opponent()) > 0 {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

/// No board probe is possible here: the callback signature has no `Engine`, and a
/// royal's real mobility needs full move filtering. `play` re-validates and `actions`
/// gates the offer, so an editor grant is the only way to reach an unusable card.
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{handoff, movement, position};
    use super::super::tests::target_procedures::position as seated;
    use super::*;

    #[test]
    fn zugzwang_arms_only_the_opponent_and_never_the_actor() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("d6", Player::Black, PieceKind::Rook)]);
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        )
        .unwrap();
        assert!(state.card_runtime.zugzwang[Player::Black.index()]);
        assert!(!state.card_runtime.zugzwang[Player::White.index()]);
        // The card ends the turn (`turn_policy: END_TURN`), so the opponent is on move.
        handoff(&engine, &mut state);
        assert_eq!(state.turn, Player::Black);
        // With the flag armed the rook is frozen out and only the royal may move.
        let legal = engine.legal_actions(&state).unwrap();
        assert!(!legal.is_empty());
        assert!(legal.iter().all(|action| match action {
            Action::Move(m) => m.from == "h8".parse().unwrap(),
            _ => true,
        }));
        let moved = engine.apply(&state, movement("h8", "h7")).unwrap();
        assert!(!moved.state.card_runtime.zugzwang[Player::Black.index()]);
        assert_eq!(moved.state.turn, Player::White);
    }

    #[test]
    fn zugzwang_rejects_and_withholds_the_action_when_the_enemy_king_is_stuck() {
        let engine = Engine::bundled_latest().unwrap();
        // The engine does not pre-filter moves that walk into check (objectives are
        // settled separately), so physical confinement is the only true "cannot move":
        // the corner king is boxed in by its own pieces on a7, b7 and b8.
        let mut state = seated(
            &[
                ("a8", Player::Black, PieceKind::King),
                ("a7", Player::Black, PieceKind::Rook),
                ("b7", Player::Black, PieceKind::Rook),
                ("b8", Player::Black, PieceKind::Rook),
                ("h1", Player::White, PieceKind::King),
            ],
            Player::White,
        );
        let card = CardInstance {
            instance_id: 1,
            definition_id: ID.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        };
        let mut offered = Vec::new();
        actions(&engine, &state, &card, &mut offered);
        assert!(offered.is_empty());
        let error = play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        );
        assert!(error.is_err());
        assert!(!state.card_runtime.zugzwang[Player::Black.index()]);
    }

    #[test]
    fn zugzwang_offers_the_action_when_the_enemy_king_can_move() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(&[("d6", Player::Black, PieceKind::Rook)]);
        let card = CardInstance {
            instance_id: 1,
            definition_id: ID.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        };
        let mut offered = Vec::new();
        actions(&engine, &state, &card, &mut offered);
        assert_eq!(
            offered,
            vec![Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None
            }]
        );
    }

    #[test]
    fn zugzwang_requires_a_none_target() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        assert!(
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &ActionTarget::Square("a1".parse().unwrap()),
                &mut Vec::new(),
            )
            .is_err()
        );
        assert_eq!(
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &ActionTarget::None,
                &mut Vec::new(),
            )
            .map(|()| state.card_runtime.zugzwang),
            Ok([false, true])
        );
    }

    /// The catalog declares END_TURN for zugzwang, so using it must complete the actor's turn.
    #[test]
    fn zugzwang_ends_the_turn() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("d6", Player::Black, PieceKind::Rook)]);
        state.actions_remaining = 2;
        state = engine
            .grant_card(&state, 1, crate::ZUGZWANG, Player::White)
            .expect("zugzwang is supported")
            .state;
        let after = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .expect("the card is usable");
        assert!(
            after
                .events
                .iter()
                .any(|event| matches!(event, Event::TurnEnded { .. })),
            "using zugzwang must end the turn: {:?}",
            after.events
        );
        assert_eq!(after.state.turn, Player::Black);
    }
}

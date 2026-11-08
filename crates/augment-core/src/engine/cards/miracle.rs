use super::*;
pub(super) const ID: &str = crate::MIRACLE;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let opponent = actor.opponent();
    // Miracle (live JS:91144-91179): every enemy piece currently capturable by a
    // friendly bishop switches owners; every friendly bishop is then sacrificed.
    let bishops = state
        .board
        .iter()
        .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Bishop)
        .collect::<Vec<_>>();
    let mut capturable = Vec::new();
    for (bishop_square, bishop) in &bishops {
        let moves = {
            let mut candidates = Vec::new();
            engine.generate_slider_moves(
                state,
                *bishop_square,
                *bishop,
                &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
                &mut candidates,
            );
            candidates
        };
        for action in moves {
            if let Action::Move(move_action) = action {
                if let Some(target_piece) = state.board.get(move_action.to) {
                    if target_piece.owner == opponent
                        && !capturable
                            .iter()
                            .any(|(square, _): &(Square, Piece)| *square == move_action.to)
                    {
                        capturable.push((move_action.to, target_piece));
                    }
                }
            }
        }
    }
    for (square, piece) in capturable {
        let mut converted = piece;
        converted.owner = actor;
        state.board.set(square, Some(converted));
    }
    for (square, bishop) in bishops {
        engine.delete_piece_for_card(state, square, bishop, actor, MIRACLE, events);
    }
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    _state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let push_none = |actions: &mut Vec<Action>| {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        })
    };
    push_none(actions);
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    // Upstream `v19DraftRequirementMet` allows ANY unconditionally.
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    fn use_card(engine: &Engine, state: &mut GameState, actor: Player) {
        play(
            engine,
            state,
            actor,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        )
        .expect("miracle needs no target");
    }

    /// Every enemy piece a friendly bishop could capture switches sides, and every friendly
    /// bishop is then sacrificed (live JS 91144-91179).
    #[test]
    fn miracle_converts_capturable_enemies_and_sacrifices_bishops() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("c1", Player::White, PieceKind::Bishop),
            ("h6", Player::Black, PieceKind::Knight),
            ("b3", Player::Black, PieceKind::Rook),
        ]);
        let knight = state.board.get("h6".parse().unwrap()).unwrap();
        use_card(&engine, &mut state, Player::White);

        // The knight on the bishop's diagonal becomes allied.
        let converted = state.board.get("h6".parse().unwrap()).unwrap();
        assert_eq!(converted.id, knight.id, "identity is preserved");
        assert_eq!(converted.owner, Player::White);
        // The rook off the diagonal is untouched.
        assert_eq!(
            state.board.get("b3".parse().unwrap()).unwrap().owner,
            Player::Black
        );
        // The bishop is sacrificed.
        assert!(
            !state
                .board
                .iter()
                .any(|(_, piece)| piece.owner == Player::White && piece.kind == PieceKind::Bishop),
            "all friendly bishops are sacrificed"
        );
    }

    /// The catalog declares END_TURN for miracle.
    #[test]
    fn miracle_ends_the_turn() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("c1", Player::White, PieceKind::Bishop),
            ("h6", Player::Black, PieceKind::Knight),
        ]);
        state.actions_remaining = 2;
        state = engine
            .grant_card(&state, 1, crate::MIRACLE, Player::White)
            .expect("miracle is supported")
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
            "using miracle must end the turn: {:?}",
            after.events
        );
        assert_eq!(after.state.turn, Player::Black);
    }
}

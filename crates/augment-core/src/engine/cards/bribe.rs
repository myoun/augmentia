use super::*;
pub(super) const ID: &str = crate::BRIBE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// `simplePieceEffectSelectionContract` (main-RjrZ5fFP.js L198901-198918) for
/// `bribe`: own piece, `effectiveType === "knight"`. Unlike the 0915 handler this
/// has no royal filter, so only the effective type decides.
fn targets(state: &GameState, owner: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| piece.owner == owner && piece.kind == PieceKind::Knight)
        .collect()
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, piece) = piece_target(state, target)?;
    if piece.owner != actor || piece.kind != PieceKind::Knight {
        return Err(EngineError::Invariant(
            "bribe requires an own knight".into(),
        ));
    }
    // `applyBribeV5` (L198930-198939): `typeOverride = "amazon"`,
    // `bribedRemaining = 3`, `bribedOriginalType = "knight"`. The transform is not
    // a move, so `moved` is left alone; `transform_current_piece` still clears the
    // queens-gambit protection and stamps the fresh no-capture window, matching
    // `markV13FreshCardTransforms` (L198731-198761) inside `finishV4CardAction`.
    let piece = engine.transform_current_piece(state, square, PieceKind::Amazon, false)?;
    // The 3-owner-turn countdown is settled by `tick_end_statuses_after_turn`, which
    // reverts the piece to a knight once `due_turn` is reached.
    let due_turn = state.turns_taken[actor.index()].saturating_add(3);
    state
        .card_runtime
        .bribe_reverts
        .retain(|revert| revert.piece_id != piece.id);
    state.card_runtime.bribe_reverts.push(BribeRevert {
        piece_id: piece.id,
        due_turn,
    });
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("bribe:{}:{}", piece.id, square),
    });
    Ok(())
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, _) in targets(state, state.turn) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        });
    }
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        !targets(state, owner).is_empty(),
        "아군 나이트가 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::target_procedures::*;
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: ID.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        }
    }

    #[test]
    fn bribe_turns_one_knight_into_an_amazon_and_schedules_a_three_turn_revert() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Knight),
            ],
            Player::White,
        );
        let knight = state.board.get("c3".parse().unwrap()).unwrap();
        let mut events = Vec::new();
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt("c3".parse().unwrap()),
            &mut events,
        )
        .unwrap();
        let bribed = state.board.get("c3".parse().unwrap()).unwrap();
        assert_eq!(bribed.id, knight.id, "identity is preserved");
        assert_eq!(bribed.owner, Player::White);
        assert_eq!(bribed.kind, PieceKind::Amazon);
        assert_eq!(
            state.card_runtime.bribe_reverts,
            vec![BribeRevert {
                piece_id: knight.id,
                due_turn: 3,
            }]
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::EffectApplied { command, .. }
                if command == &format!("bribe:{}:c3", knight.id)
        )));
    }

    #[test]
    fn bribe_reverts_exactly_after_three_owner_turns() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Knight),
            ],
            Player::White,
        );
        let mut events = Vec::new();
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt("c3".parse().unwrap()),
            &mut events,
        )
        .unwrap();
        let id = state.board.get("c3".parse().unwrap()).unwrap().id;
        // The countdown only ticks on the owner's own turn boundary.
        for expected in [1_u32, 2] {
            state.turns_taken[Player::White.index()] = expected;
            engine.tick_end_statuses_after_turn(&mut state, Player::Black, &mut Vec::new());
            assert_eq!(
                state.board.get("c3".parse().unwrap()).unwrap().kind,
                PieceKind::Amazon
            );
        }
        state.turns_taken[Player::White.index()] = 3;
        let mut events = Vec::new();
        engine.tick_end_statuses_after_turn(&mut state, Player::White, &mut events);
        assert_eq!(
            state.board.get("c3".parse().unwrap()).unwrap().kind,
            PieceKind::Knight
        );
        assert!(state.card_runtime.bribe_reverts.is_empty());
        assert!(events.iter().any(|event| matches!(
            event,
            Event::EffectApplied { command, .. } if command == &format!("bribe-revert:{id}:c3")
        )));
    }

    #[test]
    fn bribe_candidates_are_own_knights_only() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Knight),
                ("f3", Player::White, PieceKind::Bishop),
                ("d5", Player::Black, PieceKind::Knight),
            ],
            Player::White,
        );
        let mut out = Vec::new();
        actions(&engine, &state, &card(), &mut out);
        assert_eq!(
            out,
            vec![Action::UseCard {
                instance_id: 1,
                target: ActionTarget::PieceAt("c3".parse().unwrap())
            }]
        );
        assert!(eligibility(&state, Player::White, AcquisitionContext::Setup).is_eligible());
        // Black's own knight at d5 satisfies Black; removing it leaves no candidate.
        assert!(eligibility(&state, Player::Black, AcquisitionContext::Setup).is_eligible());
        let mut no_black_knight = state.clone();
        no_black_knight.board.set("d5".parse().unwrap(), None);
        assert!(
            !eligibility(&no_black_knight, Player::Black, AcquisitionContext::Setup).is_eligible()
        );
    }

    #[test]
    fn bribe_rejects_non_knights_and_opponent_pieces_without_mutating_state() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("f3", Player::White, PieceKind::Bishop),
                ("d5", Player::Black, PieceKind::Knight),
            ],
            Player::White,
        );
        for square in ["f3", "d5"] {
            let before = state.clone();
            let mut events = Vec::new();
            assert!(
                play(
                    &engine,
                    &mut state,
                    Player::White,
                    1,
                    &ActionTarget::PieceAt(square.parse().unwrap()),
                    &mut events,
                )
                .is_err(),
                "{square} must not be a bribe target"
            );
            assert_eq!(state, before);
            assert!(events.is_empty());
        }
        let mut out = Vec::new();
        actions(&engine, &state, &card(), &mut out);
        assert!(out.is_empty());
        assert!(!eligibility(&state, Player::White, AcquisitionContext::Setup).is_eligible());
    }

    #[test]
    fn bribe_clears_queens_gambit_protection_like_a_current_transform() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("c3", Player::White, PieceKind::Knight),
            ],
            Player::White,
        );
        let knight = state.board.get("c3".parse().unwrap()).unwrap();
        state.card_runtime.protected_piece_ids.insert(knight.id);
        state
            .card_runtime
            .queens_gambit_protection
            .insert(knight.id);
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt("c3".parse().unwrap()),
            &mut Vec::new(),
        )
        .unwrap();
        assert!(
            !state
                .card_runtime
                .queens_gambit_protection
                .contains(&knight.id)
        );
        assert!(!state.card_runtime.protected_piece_ids.contains(&knight.id));
        // markV13FreshCardTransforms: the new card transform cannot capture immediately.
        assert!(
            state
                .card_runtime
                .fresh_no_capture_until
                .contains_key(&knight.id)
        );
    }
}

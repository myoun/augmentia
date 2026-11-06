use super::*;
pub(super) const ID: &str = crate::WITCH_TRIAL;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Source `targetTimedSpecV5` for `witch-trial` (main-RjrZ5fFP.js L198968-198972)
/// plus the draw predicate `hasEnemyNonRoyalPiece` (L315423-315431, called with
/// `excludeMerchant` + `excludeVip` at L314622). The royal predicate is
/// `v15KingAugmentRole` (L199811), which under the September 18 balance policy
/// also covers merchants, and `hasEnemyNonRoyalPiece` drops the Colossus body.
/// The 0915 handler also allowed `Vip`; the current contract does not.
fn targetable(state: &GameState, piece: Piece) -> bool {
    !royal(state, piece)
        && piece.kind != PieceKind::Merchant
        && !matches!(piece.kind, PieceKind::Colossus | PieceKind::Vip)
}

/// `targetTimedSelectionContract` (L199000-199005) must not expose hidden
/// targets: stealth pieces and camouflage-hidden non-royals are not offerable.
fn visible_to(state: &GameState, square: Square, piece: Piece) -> bool {
    if state.card_runtime.stealth_piece_ids.contains(&piece.id) {
        return false;
    }
    let light = (square.file() + square.rank()) % 2 == 1;
    !(state.card_runtime.camouflage_color
        && !royal(state, piece)
        && (piece.owner == Player::White) == light)
}

fn candidates(state: &GameState, actor: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(square, piece)| {
            piece.owner == actor.opponent()
                && targetable(state, *piece)
                && visible_to(state, *square, *piece)
        })
        .collect()
}

fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    if piece.owner != actor.opponent() || !targetable(state, piece) {
        return Err(EngineError::Invariant(
            "witch-trial requires a non-royal enemy piece".into(),
        ));
    }
    // `applyTargetTimedV5` (L199006-199026): `witchTrialBy = actor` and
    // `witchTrialRemaining = 3`. The 2-turn September 26 rebalance (L199022) does
    // not apply to this catalog hash (`usesSeptember26Rebalance`, L847-849).
    // Expiry and the capture shield are already settled by
    // `tick_middle_statuses_after_turn` (piece_effects.rs) and
    // `resolve_middle_after_move`, so no new lifecycle hook is needed.
    state.card_runtime.witch_trials.insert(
        piece.id,
        TimedPieceStatus {
            by: actor,
            remaining_turns: 3,
        },
    );
    Ok(())
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, _) in candidates(state, state.turn) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        });
    }
}

fn eligibility(state: &GameState, owner: Player, context: AcquisitionContext) -> CardEligibility {
    if context == AcquisitionContext::Draft {
        return CardEligibility::Eligible;
    }
    require(
        !candidates(state, owner).is_empty(),
        "의심할 상대 비왕실 기물이 필요합니다.",
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

    fn target_on(square: &str) -> ActionTarget {
        ActionTarget::PieceAt(square.parse().unwrap())
    }

    fn trial_of(state: &GameState, square: &str) -> Option<TimedPieceStatus> {
        let piece = state.board.get(square.parse().unwrap())?;
        state.card_runtime.witch_trials.get(&piece.id).copied()
    }

    #[test]
    fn witch_trial_stamps_three_owner_turns_on_a_non_royal_enemy() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d5", Player::Black, PieceKind::Rook),
            ],
            Player::White,
        );
        let mut events = Vec::new();
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &target_on("d5"),
            &mut events,
        )
        .unwrap();
        assert_eq!(
            trial_of(&state, "d5"),
            Some(TimedPieceStatus {
                by: Player::White,
                remaining_turns: 3,
            })
        );
    }

    #[test]
    fn witch_trial_excludes_royals_merchants_vips_and_colossus_but_not_big_rook() {
        let engine = Engine::bundled_latest().unwrap();
        for (kind, allowed) in [
            (PieceKind::King, false),
            (PieceKind::RoyalKnight, false),
            (PieceKind::DarkWizard, false),
            (PieceKind::Merchant, false),
            (PieceKind::Vip, false),
            (PieceKind::Colossus, false),
            (PieceKind::BigRook, true),
            (PieceKind::Rook, true),
            (PieceKind::Pawn, true),
        ] {
            let mut state = position(
                &[
                    ("a1", Player::White, PieceKind::King),
                    ("d5", Player::Black, kind),
                ],
                Player::White,
            );
            let mut out = Vec::new();
            actions(&engine, &state, &card(), &mut out);
            assert_eq!(
                !out.is_empty(),
                allowed,
                "{kind:?} should be {}a witch-trial target",
                if allowed { "" } else { "not " }
            );
            let mut events = Vec::new();
            let attempt = play(
                &engine,
                &mut state,
                Player::White,
                1,
                &target_on("d5"),
                &mut events,
            );
            assert_eq!(attempt.is_ok(), allowed, "{kind:?} play outcome");
            if allowed {
                assert_eq!(
                    trial_of(&state, "d5").map(|status| status.remaining_turns),
                    Some(3),
                    "{kind:?} should be marked"
                );
            } else {
                assert!(state.card_runtime.witch_trials.is_empty(), "{kind:?}");
            }
        }
    }

    #[test]
    fn witch_trial_only_targets_the_opponent_and_ignores_hidden_pieces() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("b2", Player::White, PieceKind::Rook),
                ("d5", Player::Black, PieceKind::Rook),
            ],
            Player::White,
        );
        let mut events = Vec::new();
        assert!(
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &target_on("b2"),
                &mut events
            )
            .is_err()
        );
        assert!(trial_of(&state, "b2").is_none());

        let hidden = state.board.get("d5".parse().unwrap()).unwrap();
        state.card_runtime.stealth_piece_ids.insert(hidden.id);
        let mut out = Vec::new();
        actions(&engine, &state, &card(), &mut out);
        assert!(out.is_empty(), "hidden targets must not be offered");
    }

    #[test]
    fn witch_trial_expires_after_three_owner_turns_and_removes_the_piece() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d5", Player::Black, PieceKind::Rook),
            ],
            Player::White,
        );
        let mut events = Vec::new();
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &target_on("d5"),
            &mut events,
        )
        .unwrap();
        let mut events = Vec::new();
        for remaining in [2, 1] {
            engine.tick_middle_statuses_after_turn(&mut state, Player::Black, &mut events);
            assert_eq!(
                trial_of(&state, "d5").map(|status| status.remaining_turns),
                Some(remaining)
            );
            assert!(state.board.get("d5".parse().unwrap()).is_some());
        }
        let doomed = state.board.get("d5".parse().unwrap()).unwrap();
        engine.tick_middle_statuses_after_turn(&mut state, Player::Black, &mut events);
        assert!(state.card_runtime.witch_trials.is_empty());
        assert!(state.board.get("d5".parse().unwrap()).is_none());
        assert!(events.iter().any(|event| matches!(
            event,
            Event::PieceRemoved { piece, cause, .. }
                if piece.id == doomed.id && cause == crate::WITCH_TRIAL
        )));
    }

    #[test]
    fn witch_trial_does_not_tick_on_the_casters_turn() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d5", Player::Black, PieceKind::Rook),
            ],
            Player::White,
        );
        let mut events = Vec::new();
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &target_on("d5"),
            &mut events,
        )
        .unwrap();
        for _ in 0..5 {
            engine.tick_middle_statuses_after_turn(&mut state, Player::White, &mut Vec::new());
        }
        assert_eq!(
            trial_of(&state, "d5").map(|status| status.remaining_turns),
            Some(3)
        );
    }

    #[test]
    fn witch_trial_capture_clears_the_trial_and_grants_a_shield() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d5", Player::Black, PieceKind::Rook),
                ("d2", Player::White, PieceKind::Pawn),
            ],
            Player::Black,
        );
        let rook = state.board.get("d5".parse().unwrap()).unwrap();
        state.card_runtime.witch_trials.insert(
            rook.id,
            TimedPieceStatus {
                by: Player::White,
                remaining_turns: 3,
            },
        );
        let next = engine.apply(&state, movement("d5", "d2")).unwrap().state;
        assert!(!next.card_runtime.witch_trials.contains_key(&rook.id));
        assert!(next.card_runtime.shielded_piece_ids.contains(&rook.id));
    }
}

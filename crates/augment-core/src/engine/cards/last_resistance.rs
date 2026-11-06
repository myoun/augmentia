use super::*;
pub(super) const ID: &str = crate::LAST_RESISTANCE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// `v4Pieces` (main-RjrZ5fFP.js L197906-197926) walks row 0..7 / col 0..7, which is
/// the same order as `web_ordered_pieces` here. `applyLastResistanceV5` (L199027-199037)
/// takes only `[0]`, so exactly one royal is marked, not every royal.
fn first_royal(engine: &Engine, state: &GameState, actor: Player) -> Option<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .find(|(_, piece)| piece.owner == actor && engine.is_royal_piece(state, *piece))
}

/// Engine-free royal predicate for eligibility. `findKingAugmentPiece` (L314759) and
/// `isRoyalKing` (L359754) treat a merchant as royal under the September 18 balance
/// policy, so merchants count as recipients.
fn royal_recipient(state: &GameState, piece: Piece) -> bool {
    royal(state, piece) || piece.kind == PieceKind::Merchant
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    // L199028 requires an explicit empty selection (`selectedSquares(action, 0)`).
    if !matches!(target, ActionTarget::None) {
        return Err(EngineError::Invariant(
            "last-resistance takes no target".into(),
        ));
    }
    let (_, piece) = first_royal(engine, state, actor).ok_or_else(|| {
        EngineError::Invariant("last-resistance requires an own royal piece".into())
    })?;
    // L199032: remember whether the piece was already permanently protected, so the
    // protection outlives the timed status. Core's existing encoding for that is
    // `remember_permanent_before_timed_protection`.
    state
        .card_runtime
        .remember_permanent_before_timed_protection(piece.id);
    // L199033-199035: 3 owner turns of `permanent-protection` + `last-resistance`.
    state.card_runtime.protected_piece_ids.insert(piece.id);
    state.card_runtime.last_resistance.insert(
        piece.id,
        TimedPieceStatus {
            by: piece.owner,
            remaining_turns: 3,
        },
    );
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if first_royal(engine, state, state.turn).is_some() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == owner && royal_recipient(state, piece)),
        "효과를 받을 아군 왕실 기물이 필요합니다.",
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
    fn last_resistance_protects_only_the_first_royal_in_web_order_for_three_turns() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h3", Player::White, PieceKind::RoyalKnight),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        let king = state.board.get("a1".parse().unwrap()).unwrap();
        let knight = state.board.get("h3".parse().unwrap()).unwrap();
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
        // a8..h8, a7..h7, ... so h3 (rank 3) precedes a1 before the king is reached.
        assert_eq!(
            state.card_runtime.last_resistance.get(&knight.id).copied(),
            Some(TimedPieceStatus {
                by: Player::White,
                remaining_turns: 3,
            })
        );
        assert!(state.card_runtime.protected_piece_ids.contains(&knight.id));
        assert!(!state.card_runtime.last_resistance.contains_key(&king.id));
        assert!(!state.card_runtime.protected_piece_ids.contains(&king.id));
    }

    #[test]
    fn last_resistance_expires_after_three_owner_turns_and_drops_protection() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        let king = state.board.get("a1".parse().unwrap()).unwrap();
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
        for remaining in [2, 1] {
            engine.tick_middle_statuses_after_turn(&mut state, Player::White, &mut Vec::new());
            assert_eq!(
                state
                    .card_runtime
                    .last_resistance
                    .get(&king.id)
                    .map(|status| status.remaining_turns),
                Some(remaining)
            );
            assert!(state.card_runtime.protected_piece_ids.contains(&king.id));
        }
        engine.tick_middle_statuses_after_turn(&mut state, Player::White, &mut Vec::new());
        assert!(!state.card_runtime.last_resistance.contains_key(&king.id));
        assert!(!state.card_runtime.protected_piece_ids.contains(&king.id));
    }

    #[test]
    fn last_resistance_keeps_pre_existing_permanent_protection_after_expiry() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        let king = state.board.get("a1".parse().unwrap()).unwrap();
        state.card_runtime.protected_piece_ids.insert(king.id);
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
        assert!(
            state
                .card_runtime
                .queens_gambit_previous_protection
                .contains(&king.id)
        );
        for _ in 0..3 {
            engine.tick_middle_statuses_after_turn(&mut state, Player::White, &mut Vec::new());
        }
        assert!(!state.card_runtime.last_resistance.contains_key(&king.id));
        assert!(
            state.card_runtime.protected_piece_ids.contains(&king.id),
            "permanent protection must survive the timed status"
        );
    }

    #[test]
    fn last_resistance_does_not_tick_on_the_opponent_turn() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        let king = state.board.get("a1".parse().unwrap()).unwrap();
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
        for _ in 0..5 {
            engine.tick_middle_statuses_after_turn(&mut state, Player::Black, &mut Vec::new());
        }
        assert_eq!(
            state
                .card_runtime
                .last_resistance
                .get(&king.id)
                .map(|status| status.remaining_turns),
            Some(3)
        );
    }

    #[test]
    fn last_resistance_counts_a_merchant_royal_and_rejects_without_one() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::Pawn),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        assert!(!eligibility(&state, Player::White, AcquisitionContext::Setup).is_eligible());
        let mut out = Vec::new();
        actions(&engine, &state, &card(), &mut out);
        assert!(out.is_empty());
        let before = state.clone();
        assert!(
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &ActionTarget::None,
                &mut Vec::new()
            )
            .is_err()
        );
        assert_eq!(state, before);

        state.board.set(
            "d4".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Merchant)),
        );
        let merchant = state.board.get("d4".parse().unwrap()).unwrap();
        assert!(eligibility(&state, Player::White, AcquisitionContext::Setup).is_eligible());
        out.clear();
        actions(&engine, &state, &card(), &mut out);
        assert_eq!(
            out,
            vec![Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None
            }]
        );
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        )
        .unwrap();
        assert!(
            state
                .card_runtime
                .last_resistance
                .contains_key(&merchant.id)
        );
    }

    #[test]
    fn last_resistance_rejects_a_piece_target() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        let before = state.clone();
        assert!(
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &ActionTarget::PieceAt("a1".parse().unwrap()),
                &mut Vec::new()
            )
            .is_err()
        );
        assert_eq!(state, before);
    }
}

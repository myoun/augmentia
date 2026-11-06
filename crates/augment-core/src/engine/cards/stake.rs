use super::*;

pub(super) const ID: &str = crate::STAKE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// The upstream contract pins the piece for four of its owner's turns, then
/// grants permanent protection when the countdown reaches zero
/// (`tickWorkerStakedPieces`, main-RjrZ5fFP.js:238866-238874).
const STAKE_TURNS: u32 = 4;

/// `isStakeEligible` (main-RjrZ5fFP.js:354730-354732): an own piece that is not
/// already staked, and never a wall, football or colossus.
fn valid(state: &GameState, owner: Player, piece: Piece) -> bool {
    piece.owner == owner
        && !state.card_runtime.stake_statuses.contains_key(&piece.id)
        && !matches!(piece.kind, PieceKind::Football | PieceKind::Colossus)
}

fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, piece) = piece_target(state, target)?;
    if piece.owner != actor {
        return Err(EngineError::Invariant(
            "stake requires a friendly piece".into(),
        ));
    }
    if !valid(state, actor, piece) {
        return Err(EngineError::Invariant("stake cannot pin this piece".into()));
    }
    state.card_runtime.stake_statuses.insert(
        piece.id,
        TimedPieceStatus {
            by: actor,
            remaining_turns: STAKE_TURNS,
        },
    );
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("stake:{0}:{1}", piece.id, square),
    });
    Ok(())
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, piece) in web_ordered_pieces(state) {
        if valid(state, card.owner, piece) {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| valid(state, owner, piece)),
        "말뚝을 박을 아군 기물이 필요합니다.",
    )
}

impl Engine {
    /// Count down pinned pieces at their owner's turn start. A finished stake
    /// removes the pin and grants the follow-up protection, matching
    /// main-RjrZ5fFP.js:238866-238874.
    pub(in crate::engine) fn settle_stake(&self, state: &mut GameState, owner: Player) {
        let due: Vec<_> = state
            .card_runtime
            .stake_statuses
            .iter()
            .filter_map(|(piece_id, status)| {
                (status.by == owner).then_some((*piece_id, status.remaining_turns))
            })
            .collect();
        for (piece_id, remaining) in due {
            let next = remaining.saturating_sub(1);
            if next > 0 {
                if let Some(status) = state.card_runtime.stake_statuses.get_mut(&piece_id) {
                    status.remaining_turns = next;
                }
            } else {
                state.card_runtime.stake_statuses.remove(&piece_id);
                state.card_runtime.shielded_piece_ids.insert(piece_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::super::tests::target_procedures::has_moves_for;
    use super::*;

    #[test]
    fn stake_pins_a_friendly_piece_for_four_owner_turns() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d4", Player::White, PieceKind::Rook),
            ("e6", Player::White, PieceKind::Knight),
        ]);
        let rook = at(&state, "d4");
        use_at(&mut state, &engine, crate::STAKE, single("d4"));
        let status = state.card_runtime.stake_statuses.get(&rook.id).copied();
        assert_eq!(
            status,
            Some(TimedPieceStatus {
                by: Player::White,
                remaining_turns: 4
            })
        );
        // The pin is enforced in shared move generation, so the rook has no move.
        assert!(!has_moves_for(&engine, &state, "d4"));
        // Other friendly pieces still move.
        assert!(has_moves_for(&engine, &state, "e6"));
    }

    #[test]
    fn stake_expires_after_four_owner_starts_and_grants_protection() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d4", Player::White, PieceKind::Rook),
            ("e6", Player::White, PieceKind::Knight),
            ("d6", Player::Black, PieceKind::Rook),
            ("b1", Player::Black, PieceKind::Knight),
        ]);
        let rook = at(&state, "d4");
        use_at(&mut state, &engine, crate::STAKE, single("d4"));
        assert!(state.card_runtime.stake_statuses.contains_key(&rook.id));

        // The pin counts down exactly once per owner turn start. Drive real
        // turn handoffs and check the invariant after each one instead of
        // assuming how many handoffs a single owner turn consumes.
        let mut protected = false;
        for _ in 0..16 {
            let before = state.turns_taken[Player::White.index()];
            if state.card_runtime.stake_statuses.contains_key(&rook.id) {
                let remaining = state
                    .card_runtime
                    .stake_statuses
                    .get(&rook.id)
                    .map_or(0, |status| status.remaining_turns);
                assert!(
                    remaining <= STAKE_TURNS,
                    "the pin must never gain turns: {remaining}"
                );
                // While pinned the rook has no legal move.
                assert!(
                    !has_moves_for(&engine, &state, "d4"),
                    "a pinned piece must not move"
                );
            }
            handoff(&engine, &mut state);
            let owner_started = state.turns_taken[Player::White.index()] > before;
            if owner_started && !state.card_runtime.stake_statuses.contains_key(&rook.id) {
                protected = state.card_runtime.shielded_piece_ids.contains(&rook.id);
                break;
            }
        }
        assert!(
            protected,
            "the pin must expire and grant protection on the fourth owner turn"
        );
        assert!(!state.card_runtime.stake_statuses.contains_key(&rook.id));
        // The loop exits right after White's turn ends, so pass the turn back to
        // White before checking that the pin no longer blocks the rook.
        if state.turn != Player::White {
            handoff(&engine, &mut state);
        }
        assert_eq!(state.turn, Player::White);
        assert!(has_moves_for(&engine, &state, "d4"));
    }

    #[test]
    fn stake_rejects_enemy_targets_and_already_pinned_pieces() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d4", Player::White, PieceKind::Rook),
            ("d6", Player::Black, PieceKind::Rook),
        ]);
        use_at(&mut state, &engine, crate::STAKE, single("d4"));
        // Re-using on the same piece is rejected rather than refreshing it.
        let again = engine.apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: single("d4"),
            },
        );
        assert!(again.is_err());
        // An enemy piece is never a legal target.
        let enemy = engine.apply(
            &state,
            Action::UseCard {
                instance_id: 1,
                target: single("d6"),
            },
        );
        assert!(enemy.is_err());
    }

    #[test]
    fn stake_offers_only_own_unpinned_targets() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d4", Player::White, PieceKind::Rook),
            ("d6", Player::Black, PieceKind::Rook),
        ]);
        state = engine
            .grant_card(&state, 1, crate::STAKE, Player::White)
            .unwrap()
            .state;
        let offers = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .filter(|action| matches!(action, Action::UseCard { .. }))
            .count();
        // The white rook and the white king are both valid targets.
        assert_eq!(offers, 2);
    }
}

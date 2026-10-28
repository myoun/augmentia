use super::*;
pub(super) const ID: &str = crate::POISONED_PAWN;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let mut candidates = engine.poison_targets(state, actor);
    if candidates.is_empty() {
        return Err(EngineError::Invariant(
            "독을 부여할 아군 폰이 없습니다.".into(),
        ));
    }
    for ordinal in 0..candidates.len().min(3) {
        // The active card draws once per pick, including a final singleton.
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        events.push(Event::RandomSample {
            purpose: format!("card.v19.poisoned-pawn.{ordinal}"),
            roll_53: roll,
        });
        let (_, piece) = candidates.remove(index.expect("nonempty poison candidates"));
        state.card_runtime.poisoned_piece_ids.insert(piece.id);
    }
    engine.resolve_gomoku_victory(state);
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    let mut push = |target| {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target,
        })
    };
    if !engine.poison_targets(state, actor).is_empty() {
        push(ActionTarget::None);
    }
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let rt = &state.card_runtime;
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    require(
        has_own(&|p| p.kind == Pawn && !rt.poisoned_piece_ids.contains(&p.id)),
        "독이 부여되지 않은 아군 폰이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

impl Engine {
    pub(in crate::engine) fn poison_targets(
        &self,
        state: &GameState,
        actor: Player,
    ) -> Vec<(Square, Piece)> {
        web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, p)| {
                p.owner == actor
                    && p.kind == PieceKind::Pawn
                    && !state.card_runtime.poisoned_piece_ids.contains(&p.id)
            })
            .collect()
    }
}

impl Engine {
    pub(in crate::engine) fn settle_poison_captures(
        &self,
        state: &mut GameState,
        attacker_id: u32,
        poisoned_before: &BTreeSet<u32>,
        events: &[Event],
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID || poisoned_before.is_empty() {
            return;
        }
        let Some((_, attacker)) = state.board.find_by_id(attacker_id) else {
            return;
        };
        let start = events
            .iter()
            .rposition(|e| matches!(e, Event::ActionApplied { .. }))
            .unwrap_or(0);
        let poisoned = events[start..].iter().any(|event| {
            let (actor, piece) = match event {
                Event::PieceCaptured { actor, piece, .. } => (*actor, piece),
                Event::PieceRemoved {
                    actor,
                    piece,
                    cause,
                    ..
                } if cause == "colossus-sector" => (*actor, piece),
                _ => return false,
            };
            actor == attacker.owner
                && piece.owner != attacker.owner
                && poisoned_before.contains(&piece.id)
        });
        if poisoned {
            // Capture turn is ticked immediately on handoff, leaving two future
            // own turns. Poison blocks action, but does not grant frozen immunity.
            let remaining = state
                .card_runtime
                .poison_stun_turns
                .entry(attacker.id)
                .or_default();
            *remaining = (*remaining).max(3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::target_procedures::*;
    use super::*;
    #[test]
    fn poison_stuns_two_future_own_turns_but_does_not_protect_the_capturer() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("h1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("a1", Player::White, PieceKind::Rook),
                ("a5", Player::Black, PieceKind::Pawn),
                ("b5", Player::Black, PieceKind::Rook),
            ],
            Player::White,
        );
        let pawn = state.board.get("a5".parse().unwrap()).unwrap();
        let rook = state.board.get("a1".parse().unwrap()).unwrap();
        state.card_runtime.poisoned_piece_ids.insert(pawn.id);
        state = engine.apply(&state, movement("a1", "a5")).unwrap().state;
        assert_eq!(state.card_runtime.poison_stun_turns.get(&rook.id), Some(&2));
        assert!(!state.card_runtime.frozen.contains_key(&rook.id));
        assert!(has_move(&engine, &state, "b5", "a5"));
        for (from, to, remaining) in [
            ("h8", "g8", 2),
            ("h1", "h2", 1),
            ("g8", "h8", 1),
            ("h2", "h3", 0),
            ("h8", "g8", 0),
        ] {
            state = engine.apply(&state, movement(from, to)).unwrap().state;
            assert_eq!(
                state
                    .card_runtime
                    .poison_stun_turns
                    .get(&rook.id)
                    .copied()
                    .unwrap_or(0),
                remaining
            );
            if state.turn == Player::White && remaining > 0 {
                assert!(!has_moves_for(&engine, &state, "a5"));
            }
        }
        assert!(has_move(&engine, &state, "a5", "a6"));
    }
}

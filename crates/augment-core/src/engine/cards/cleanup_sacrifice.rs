use super::*;
pub(super) const ID: &str = crate::CLEANUP_SACRIFICE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// JS `isCleanupSacrificeExcludedPiece`: royal-like pieces and installations
/// (wall/football/black-hole) can neither be sacrificed nor be the random victim.
fn excluded(engine: &Engine, state: &GameState, piece: Piece) -> bool {
    engine.is_royal_piece(state, piece)
        || matches!(piece.kind, PieceKind::Barrier | PieceKind::Football)
}

/// Candidates are enumerated in web `forEachSquare` order so the seeded draw
/// matches the upstream `v8RandomPick(candidates, randomValues)` index.
fn candidates(
    engine: &Engine,
    state: &GameState,
    owner: Player,
    kind: PieceKind,
) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| {
            piece.owner == owner.opponent()
                && piece.kind == kind
                && !excluded(engine, state, *piece)
        })
        .collect()
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, selected) = piece_target(state, target)?;
    if selected.owner != actor || excluded(engine, state, selected) {
        return Err(EngineError::Invariant(
            "cleanup sacrifice requires a non-royal own piece".into(),
        ));
    }
    let entries = candidates(engine, state, actor, selected.kind);
    let (roll, index) = state.rng_mut().web_choice_index(entries.len());
    events.push(Event::RandomSample {
        purpose: "cleanup-sacrifice".to_owned(),
        roll_53: roll,
    });
    let (victim_square, victim) = index
        .and_then(|index| entries.get(index).copied())
        .ok_or_else(|| EngineError::Invariant("cleanup sacrifice has no same-kind enemy".into()))?;
    // The enemy is removed first, then the sacrificed own piece, matching the
    // 0915 handler and the upstream two-step `removeCleanupVictim` +
    // `removeSacrificedPiece` order. Both count as captures for the other side.
    engine.remove_piece_for_card(state, victim_square, victim, actor, ID, events);
    engine.break_initiative_by_capture(state, victim.owner, actor);
    engine.remove_piece_for_card(state, square, selected, actor.opponent(), ID, events);
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    let actor = state.turn;
    for (square, piece) in state.board.iter() {
        if piece.owner == actor
            && !excluded(engine, state, piece)
            && !candidates(engine, state, actor, piece.kind).is_empty()
        {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    let excluded = |piece: Piece| {
        royal(state, piece)
            || matches!(
                piece.kind,
                PieceKind::Merchant | PieceKind::Barrier | PieceKind::Football
            )
    };
    let playable = state.board.iter().any(|(_, piece)| {
        piece.owner == owner
            && !excluded(piece)
            && state.board.iter().any(|(_, enemy)| {
                enemy.owner == owner.opponent() && enemy.kind == piece.kind && !excluded(enemy)
            })
    });
    require(
        playable,
        "같은 종류의 비왕실 상대 기물이 있는 아군 기물이 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    fn removals(events: &[Event]) -> Vec<Piece> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::PieceRemoved { piece, cause, .. } if cause == crate::CLEANUP_SACRIFICE => {
                    Some(*piece)
                }
                _ => None,
            })
            .collect()
    }

    fn play_events(
        engine: &Engine,
        state: &mut GameState,
        target: ActionTarget,
    ) -> Result<Vec<Event>, EngineError> {
        let mut events = Vec::new();
        let actor = state.turn;
        super::play(engine, state, actor, 1, &target, &mut events).map(|()| events)
    }

    fn sample_count(events: &[Event]) -> usize {
        events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    Event::RandomSample { purpose, .. } if purpose == "cleanup-sacrifice"
                )
            })
            .count()
    }

    #[test]
    fn cleanup_sacrifice_removes_the_selected_piece_and_a_seeded_same_kind_enemy() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("b8", Player::Black, PieceKind::Rook),
            ("g8", Player::Black, PieceKind::Rook),
        ]);
        let selected = at(&state, "b1");
        let events = play_events(&engine, &mut state, single("b1")).unwrap();

        assert!(state.board.get("b1".parse().unwrap()).is_none());
        assert_eq!(
            state
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == Player::Black && piece.kind == PieceKind::Rook)
                .count(),
            1,
            "exactly one black rook is removed"
        );
        let removed = removals(&events);
        assert_eq!(removed.len(), 2);
        assert_eq!(removed[0].owner, Player::Black, "enemy removed first");
        assert_eq!(removed[1].id, selected.id, "selected piece is sacrificed");
        assert_eq!(sample_count(&events), 1);
        assert_eq!(state.turn, Player::White, "turn policy preserves the turn");
    }

    #[test]
    fn cleanup_sacrifice_requires_a_same_kind_enemy_and_rejects_illegal_targets() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("a8", Player::Black, PieceKind::Knight),
        ]);
        state.add_card(CardInstance {
            instance_id: 1,
            definition_id: crate::CLEANUP_SACRIFICE.to_owned(),
            owner: Player::White,
            half_stars: 3,
            charges: 1,
            available_after_turn: None,
        });
        assert!(
            play_events(&engine, &mut state, single("b1")).is_err(),
            "no same-kind enemy means no legal play"
        );
        assert!(
            play_events(&engine, &mut state, single("a8")).is_err(),
            "an enemy piece is not a valid sacrifice"
        );
        let mut out = Vec::new();
        super::actions(&engine, &state, &state.cards[0].clone(), &mut out);
        assert!(out.is_empty(), "no legal card actions without a match");

        // A non-royal enemy of the same kind makes the card playable again.
        state.board.set(
            "d8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Rook).with_id(300)),
        );
        let mut out = Vec::new();
        super::actions(&engine, &state, &state.cards[0].clone(), &mut out);
        assert_eq!(
            out,
            vec![Action::UseCard {
                instance_id: 1,
                target: single("b1")
            }]
        );
        let events = play_events(&engine, &mut state, single("b1")).unwrap();
        assert!(state.board.get("b1".parse().unwrap()).is_none());
        assert!(state.board.get("d8".parse().unwrap()).is_none());
        assert!(state.board.get("a8".parse().unwrap()).is_some());
        assert_eq!(removals(&events).len(), 2);
    }

    #[test]
    fn cleanup_sacrifice_excludes_royal_and_installation_pieces_on_both_sides() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("b8", Player::Black, PieceKind::Rook),
            ("c1", Player::White, PieceKind::Barrier),
            ("c8", Player::Black, PieceKind::Football),
        ]);
        // A royal-flagged black rook cannot be picked as the random victim.
        let royal_rook = at(&state, "b8");
        state.card_runtime.royal_piece_ids.insert(royal_rook.id);
        state.add_card(CardInstance {
            instance_id: 1,
            definition_id: crate::CLEANUP_SACRIFICE.to_owned(),
            owner: Player::White,
            half_stars: 3,
            charges: 1,
            available_after_turn: None,
        });

        let mut out = Vec::new();
        super::actions(&engine, &state, &state.cards[0].clone(), &mut out);
        assert!(
            out.iter().all(|action| !matches!(
                action,
                Action::UseCard { target: ActionTarget::PieceAt(square), .. }
                    if *square == "c1".parse().unwrap()
            )),
            "installations are not valid sacrifices"
        );
        assert!(
            out.is_empty(),
            "a royal enemy of the same kind is not a candidate"
        );
        assert!(play_events(&engine, &mut state, single("b1")).is_err());
        assert!(play_events(&engine, &mut state, single("c1")).is_err());

        // A non-royal enemy rook restores the play; royal pieces stay untouched.
        let mut playable = state.clone();
        playable.board.set(
            "d8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Rook).with_id(301)),
        );
        let events = play_events(&engine, &mut playable, single("b1")).unwrap();
        assert!(playable.board.get("d8".parse().unwrap()).is_none());
        assert!(playable.board.get("b8".parse().unwrap()).is_some());
        assert!(playable.board.get("c8".parse().unwrap()).is_some());
        assert_eq!(removals(&events).len(), 2);
    }

    #[test]
    fn cleanup_sacrifice_draw_is_seeded_and_reproducible() {
        let engine = Engine::bundled_latest().unwrap();
        let mut left = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("b8", Player::Black, PieceKind::Rook),
            ("g8", Player::Black, PieceKind::Rook),
        ]);
        let mut right = left.clone();
        let left_events = play_events(&engine, &mut left, single("b1")).unwrap();
        let right_events = play_events(&engine, &mut right, single("b1")).unwrap();
        assert_eq!(left, right);
        assert_eq!(left_events, right_events);
        assert_eq!(sample_count(&left_events), 1);
    }

    #[test]
    fn cleanup_sacrifice_eligibility_requires_a_matching_non_royal_enemy() {
        let state = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("b8", Player::Black, PieceKind::Rook),
        ]);
        assert!(super::eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible());
        let knights_only = position(&[("b1", Player::White, PieceKind::Knight)]);
        assert!(
            super::eligibility(&knights_only, Player::White, AcquisitionContext::Draft)
                .reason()
                .is_some()
        );
        let mut royal_enemy = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("b8", Player::Black, PieceKind::Rook),
        ]);
        let enemy = at(&royal_enemy, "b8");
        royal_enemy.card_runtime.royal_piece_ids.insert(enemy.id);
        assert!(
            super::eligibility(&royal_enemy, Player::White, AcquisitionContext::Draft)
                .reason()
                .is_some()
        );
    }
}

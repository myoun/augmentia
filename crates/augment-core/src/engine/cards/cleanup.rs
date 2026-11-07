use super::*;
pub(super) const ID: &str = crate::CLEANUP;

/// Most pieces one cast may remove, from `INTERNAL_CLEANUP_LIMIT` (main bundle 2921)
/// and the validator at 48468.
const MAX_TARGETS: usize = 3;

pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let squares = squares_target(target)?;
    if squares.len() > MAX_TARGETS {
        return Err(EngineError::Invariant(
            "cleanup removes at most three pieces".to_owned(),
        ));
    }
    // Every selection must name a distinct allied piece the card may remove.
    let mut seen = BTreeSet::new();
    let mut chosen = Vec::new();
    for square in squares {
        if !seen.insert(square) {
            return Err(EngineError::Invariant(
                "cleanup targets must be distinct".to_owned(),
            ));
        }
        let Some(piece) = state.board.get(square) else {
            return Err(EngineError::Invariant(
                "cleanup target square is empty".to_owned(),
            ));
        };
        if !removable(state, actor, piece) {
            return Err(EngineError::Invariant(
                "cleanup target is not a removable allied piece".to_owned(),
            ));
        }
        chosen.push((square, piece));
    }
    for (square, piece) in chosen {
        engine.delete_piece_for_card(state, square, piece, actor.opponent(), CLEANUP, events);
    }
    Ok(())
}

/// Whether Cleanup may remove this piece.
///
/// Mirrors `cleanupPieceTypeAllowed` (main bundle 10993) plus the royal, large-body
/// and identity guards in `v16CleanupTargets` (48466-48475).
fn removable(state: &GameState, actor: Player, piece: Piece) -> bool {
    piece.owner == actor
        && !royal(state, piece)
        && !matches!(
            piece.kind,
            PieceKind::Monster | PieceKind::Scarecrow | PieceKind::Football
        )
        && !state.card_runtime.is_large(piece.id)
        && !state
            .card_runtime
            .large_missing_cells
            .contains_key(&piece.id)
}

/// Every combination of one to three removable allied pieces is offered.
///
/// Upstream accepts any 1..=3 distinct squares holding eligible plain pieces, so a
/// generator that emits only one square per piece hides most of the action space.
pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    let candidates = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| removable(state, actor, *piece))
        .map(|(square, _)| square)
        .collect::<Vec<_>>();

    for i in 0..candidates.len() {
        actions.push(use_card(card, vec![candidates[i]]));
        for j in i + 1..candidates.len() {
            actions.push(use_card(card, vec![candidates[i], candidates[j]]));
            for k in j + 1..candidates.len() {
                actions.push(use_card(
                    card,
                    vec![candidates[i], candidates[j], candidates[k]],
                ));
            }
        }
    }
}

fn use_card(card: &CardInstance, squares: Vec<Square>) -> Action {
    Action::UseCard {
        instance_id: card.instance_id,
        target: ActionTarget::Squares(squares),
    }
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    require(
        has_non_royal(state, player),
        "제거할 아군 기물이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::CLEANUP.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        }
    }

    fn offered(state: &GameState) -> Vec<Vec<Square>> {
        let engine = Engine::bundled_latest().unwrap();
        let mut out = Vec::new();
        actions(&engine, state, &card(), &mut out);
        out.into_iter()
            .map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::Squares(squares),
                    ..
                } => squares,
                other => panic!("unexpected action {other:?}"),
            })
            .collect()
    }

    #[test]
    fn cleanup_offers_every_combination_of_up_to_three_pieces() {
        let state = position(&[
            ("b2", Player::White, PieceKind::Rook),
            ("c2", Player::White, PieceKind::Bishop),
            ("d2", Player::White, PieceKind::Knight),
        ]);
        let offered = offered(&state);
        // Three candidates: 3 singles + 3 pairs + 1 triple.
        assert_eq!(offered.len(), 7);
        assert_eq!(offered.iter().filter(|s| s.len() == 1).count(), 3);
        assert_eq!(offered.iter().filter(|s| s.len() == 2).count(), 3);
        assert_eq!(offered.iter().filter(|s| s.len() == 3).count(), 1);
    }

    #[test]
    fn cleanup_skips_royals_large_bodies_and_excluded_types() {
        let mut state = position(&[
            ("b2", Player::White, PieceKind::Rook),
            ("c2", Player::White, PieceKind::Football),
            ("d2", Player::White, PieceKind::Scarecrow),
            ("e2", Player::White, PieceKind::Knight),
        ]);
        // A multi-cell body is not a cleanup target even though it is a plain piece.
        let big = state.board.get("e2".parse().unwrap()).unwrap();
        state
            .card_runtime
            .insert_large_body(big.id, "e2".parse().unwrap(), PieceHealth::new(2));
        let offered = offered(&state);
        // Only the rook remains: the king, football, scarecrow and large body are skipped.
        assert_eq!(
            offered.len(),
            1,
            "only one removable piece remains: {offered:?}"
        );
        assert_eq!(offered[0], vec!["b2".parse().unwrap()]);
    }

    #[test]
    fn cleanup_removes_only_the_selected_pieces() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b2", Player::White, PieceKind::Rook),
            ("c2", Player::White, PieceKind::Bishop),
            ("d2", Player::White, PieceKind::Knight),
        ]);
        let rook = state.board.get("b2".parse().unwrap()).unwrap();
        let bishop = state.board.get("c2".parse().unwrap()).unwrap();
        let knight = state.board.get("d2".parse().unwrap()).unwrap();
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::Squares(vec!["b2".parse().unwrap(), "c2".parse().unwrap()]),
            &mut Vec::new(),
        )
        .unwrap();
        assert!(state.board.find_by_id(rook.id).is_none());
        assert!(state.board.find_by_id(bishop.id).is_none());
        assert!(
            state.board.find_by_id(knight.id).is_some(),
            "unselected survives"
        );
    }

    #[test]
    fn cleanup_rejects_too_many_duplicate_or_foreign_targets() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b2", Player::White, PieceKind::Rook),
            ("c2", Player::White, PieceKind::Bishop),
            ("d2", Player::White, PieceKind::Knight),
            ("e2", Player::White, PieceKind::Amazon),
            ("h7", Player::Black, PieceKind::Rook),
        ]);
        for target in [
            ActionTarget::Squares(vec![
                "b2".parse().unwrap(),
                "c2".parse().unwrap(),
                "d2".parse().unwrap(),
                "e2".parse().unwrap(),
            ]),
            ActionTarget::Squares(vec!["b2".parse().unwrap(), "b2".parse().unwrap()]),
            ActionTarget::Squares(vec!["h7".parse().unwrap()]),
        ] {
            let before = state.clone();
            assert!(
                play(
                    &engine,
                    &mut state,
                    Player::White,
                    1,
                    &target,
                    &mut Vec::new()
                )
                .is_err(),
                "{target:?} must be rejected"
            );
            assert_eq!(state, before, "{target:?} mutated state");
        }
    }
}

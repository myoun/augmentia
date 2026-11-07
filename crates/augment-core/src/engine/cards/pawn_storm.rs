use super::*;
pub(super) const ID: &str = crate::PAWN_STORM;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let direction = if actor == Player::White { 1 } else { -1 };
    let mut entries = squares_target(target)?
        .into_iter()
        .filter_map(|square| state.board.get(square).map(|piece| (square, piece)))
        .collect::<Vec<_>>();
    entries.sort_by_key(|(square, _)| {
        if actor == Player::White {
            7 - square.rank()
        } else {
            square.rank()
        }
    });
    for (from, mut piece) in entries {
        if let Some(to) = from.offset(0, direction)
            && state.board.get(to).is_none()
        {
            state.board.set(from, None);
            if to.rank() == if actor == Player::White { 7 } else { 0 } {
                piece.kind = PieceKind::Queen;
            }
            state.board.set(to, Some(piece));
            state.card_runtime.moved_piece_ids.insert(piece.id);
            engine.mark_fresh_no_capture(state, piece);
        }
    }
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    let direction = if actor == Player::White { 1 } else { -1 };
    let candidates = state
        .board
        .iter()
        .filter(|(square, piece)| {
            piece.owner == actor
                && piece.kind == PieceKind::Pawn
                && square
                    .offset(0, direction)
                    .is_some_and(|to| state.board.get(to).is_none())
        })
        .map(|(square, _)| square)
        .collect::<Vec<_>>();
    // Every non-empty subset of the eligible pawns is a valid selection: upstream applies
    // the chosen pawns individually and only ignores duplicates (main bundle 25582-25606),
    // so any subset advances and no particular size is required. Emitting only the
    // singletons and the full set would hide every middle-sized choice.
    for mask in 1_usize..(1_usize << candidates.len()) {
        let chosen = candidates
            .iter()
            .enumerate()
            .filter(|(index, _)| mask & (1 << index) != 0)
            .map(|(_, square)| *square)
            .collect::<Vec<_>>();
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(chosen),
        });
    }
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    require(
        has_kind(state, player, PieceKind::Pawn),
        "아군 폰이 필요합니다.",
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
            definition_id: crate::PAWN_STORM.to_owned(),
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

    /// Three advancing pawns must yield every non-empty subset, not just one and all.
    #[test]
    fn pawn_storm_enumerates_every_non_empty_subset() {
        let state = position(&[
            ("b2", Player::White, PieceKind::Pawn),
            ("c2", Player::White, PieceKind::Pawn),
            ("d2", Player::White, PieceKind::Pawn),
        ]);
        let offered = offered(&state);
        // 2^3 - 1 = 7 subsets.
        assert_eq!(offered.len(), 7, "every non-empty subset: {offered:?}");
        for size in 1..=3 {
            let count = offered
                .iter()
                .filter(|squares| squares.len() == size)
                .count();
            let expected = match size {
                1 => 3,
                2 => 3,
                _ => 1,
            };
            assert_eq!(count, expected, "subsets of size {size}");
        }
    }

    /// A pawn that cannot advance is not a candidate, so it never appears in a subset.
    #[test]
    fn pawn_storm_skips_blocked_pawns() {
        let state = position(&[
            ("b2", Player::White, PieceKind::Pawn),
            ("c2", Player::White, PieceKind::Pawn),
            ("c3", Player::Black, PieceKind::Rook),
        ]);
        let offered = offered(&state);
        assert_eq!(offered.len(), 1, "only b2 can advance: {offered:?}");
        assert_eq!(offered[0], vec!["b2".parse().unwrap()]);
    }
}

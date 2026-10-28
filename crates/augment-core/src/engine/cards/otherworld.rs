use super::*;
pub(super) const ID: &str = crate::OTHERWORLD;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let id = ID;
    let pawns: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
        .collect();
    // applyOpeningPassiveAcquisition.choose: no draw for zero/one target.
    // A previously offered target may disappear before a draft settles.
    if pawns.is_empty() {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant(
            "이세계에 사용할 아군 폰이 없습니다.".into(),
        ));
    }
    let index = if pawns.len() == 1 {
        0
    } else {
        let (roll, index) = state.rng_mut().web_choice_index(pawns.len());
        events.push(Event::RandomSample {
            purpose: "acquire.opening.otherworld.target".into(),
            roll_53: roll,
        });
        index.expect("nonempty pawn candidates")
    };
    let (square, pawn) = pawns[index];
    let origin = state
        .card_runtime
        .piece_origins
        .get(&pawn.id)
        .copied()
        .unwrap_or(square);
    engine.delete_piece_for_card(state, square, pawn, owner, id, events);
    state
        .card_runtime
        .pending_otherworld
        .push(PendingOtherworld {
            piece_id: pawn.id,
            owner,
            square,
            origin,
            due_ply: state.ply.saturating_add(28),
        });
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let count = |owner, kind| {
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .count()
    };
    let own = |kind| count(player, kind);
    require(own(Pawn) > 0, "아군 폰이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::acquisition::*;
    use super::super::tests::target_procedures;
    use super::*;
    #[test]
    fn otherworld_return_is_not_a_reserved_spawn_cell_for_checker_or_false_start() {
        let engine = Engine::bundled_latest().unwrap();
        let before = target_procedures::position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("a3", Player::White, PieceKind::Pawn),
            ],
            Player::White,
        );
        let mut before = engine
            .grant_card(&before, 1, OTHERWORLD, Player::White)
            .unwrap()
            .state;
        let checker = engine
            .grant_card(&before, 2, CHECKER, Player::White)
            .unwrap()
            .state;
        assert_eq!(
            checker
                .board
                .iter()
                .filter(|(_, p)| p.kind == PieceKind::Checker)
                .count(),
            4
        );
        before.board.set(
            "a2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn).with_id(100)),
        );
        let shifted = engine
            .grant_card(&before, 2, crate::FALSE_START, Player::White)
            .unwrap()
            .state;
        assert_eq!(
            shifted.board.get("a4".parse().unwrap()).unwrap().kind,
            PieceKind::Pawn
        );
    }
    #[test]
    fn otherworld_removes_now_skips_single_target_rng_and_returns_at_28_moves() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d5", Player::White, PieceKind::Pawn),
        ]);
        let pawn = before.board.get("d5".parse().unwrap()).unwrap();
        before
            .card_runtime
            .piece_origins
            .insert(pawn.id, "e2".parse().unwrap());
        let transition = engine
            .grant_card(&before, 1, OTHERWORLD, Player::White)
            .unwrap();
        let mut state = transition.state;
        assert!(state.board.get("d5".parse().unwrap()).is_none());
        assert_eq!(state.rng_state(), before.rng_state());
        assert_eq!(state.card_runtime.pending_otherworld[0].due_ply, 28);
        state.ply = 27;
        engine.resolve_after_counted_move(&mut state, &mut Vec::new());
        assert!(state.board.get("d5".parse().unwrap()).is_none());
        state.ply = 28;
        engine.resolve_after_counted_move(&mut state, &mut Vec::new());
        let wizard = state.board.get("d5".parse().unwrap()).unwrap();
        assert_eq!((wizard.id, wizard.kind), (pawn.id, PieceKind::Wizard));
        assert_eq!(
            state.card_runtime.piece_origins[&wizard.id],
            "e2".parse().unwrap()
        );
        let mut two = before.clone();
        two.board.set(
            "c4".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn).with_id(100)),
        );
        let result = engine
            .grant_card(&two, 1, OTHERWORLD, Player::White)
            .unwrap();
        assert_eq!(
            result
                .events
                .iter()
                .filter(|e| matches!(e, Event::RandomSample { .. }))
                .count(),
            1
        );
        assert_eq!(
            result,
            engine
                .grant_card(&two, 1, OTHERWORLD, Player::White)
                .unwrap()
        );
    }
}

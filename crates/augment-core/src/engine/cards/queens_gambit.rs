use super::*;
pub(super) const ID: &str = crate::QUEENS_GAMBIT;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let queens: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Queen && !royal(state, *p))
        .collect();
    if queens.is_empty() {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant("no queen for queens gambit".into()));
    }
    let choose = |state: &mut GameState, events: &mut Vec<Event>, count: usize, part: &str| {
        if count == 1 {
            0
        } else {
            let (roll, index) = state.rng_mut().web_choice_index(count);
            events.push(Event::RandomSample {
                purpose: format!("acquire.opening.queens-gambit.{part}"),
                roll_53: roll,
            });
            index.unwrap()
        }
    };
    let (square, queen) = queens[choose(state, events, queens.len(), "target")];
    let separated: Vec<_> = (0_u8..8)
        .filter(|f| f.abs_diff(square.file()) > 1)
        .collect();
    let occupied: BTreeSet<_> = state
        .board
        .iter()
        .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
        .map(|(s, _)| s.file())
        .collect();
    let with_pawns: Vec<_> = separated
        .iter()
        .copied()
        .filter(|f| occupied.contains(f))
        .collect();
    let candidates = if with_pawns.is_empty() {
        separated
    } else {
        with_pawns
    };
    let random_file = candidates[choose(state, events, candidates.len(), "file")];
    engine.delete_piece_for_card(state, square, queen, owner, ID, events);
    for (_, pawn) in state.board.iter().filter(|(s, p)| {
        p.owner == owner
            && p.kind == PieceKind::Pawn
            && (s.file() == square.file() || s.file() == random_file)
    }) {
        apply_protection(&mut state.card_runtime, pawn.id);
    }
    Ok(())
}

pub(super) fn apply_protection(runtime: &mut CardRuntimeState, piece_id: u32) {
    if runtime.protected_piece_ids.contains(&piece_id)
        && !runtime.queens_gambit_protection.contains(&piece_id)
        && !runtime.sacrifice_protection.contains_key(&piece_id)
        && !runtime.last_resistance.contains_key(&piece_id)
        && !runtime.coronation_protection.contains_key(&piece_id)
    {
        runtime.queens_gambit_previous_protection.insert(piece_id);
    }
    runtime.protected_piece_ids.insert(piece_id);
    runtime.queens_gambit_protection.insert(piece_id);
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Queen && !royal(state, p)),
        "비왕실 아군 퀸이 필요합니다.",
    )
}

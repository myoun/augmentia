use super::*;
pub(super) const ID: &str = crate::EMERGENCY_EVACUATION;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn advance(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    from: Square,
    moved: &mut Vec<u32>,
) -> bool {
    let Some(piece) = state.board.get(from) else {
        return false;
    };
    let direction = if actor == Player::White { -1 } else { 1 };
    let Some(to) = from.offset(0, direction) else {
        return false;
    };
    if piece.owner != actor
        || moved.contains(&piece.id)
        || engine.piece_has_ability(state, piece, PieceKind::Slime)
        || matches!(
            piece.kind,
            PieceKind::Football | PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop
        )
        || state.card_runtime.large_pieces.contains_key(&piece.id)
        || (state.card_runtime.bunkers.contains_key(&piece.id)
            && engine.is_royal_piece(state, piece))
        || engine.board_piece_at(state, to).is_some()
        || state.card_runtime.barricades.contains(&to)
        || state.card_runtime.crown_pos == Some(to)
        || engine.rule_cell_collapsed(state, to)
        || state
            .card_runtime
            .pending_lobsters
            .iter()
            .any(|p| p.square == to)
        || state
            .card_runtime
            .pending_scarecrows
            .iter()
            .any(|p| p.square == to)
        || state
            .card_runtime
            .pending_portals
            .iter()
            .any(|p| p.squares.contains(&to))
        || state.card_runtime.portals.is_some_and(|p| p.contains(&to))
        || state
            .card_runtime
            .rule_portals
            .is_some_and(|p| p.contains(&to))
        || engine.fianchetto_destination_blocked(state, piece, from, to)
        || engine.majesty_destination_blocked(state, piece, to)
    {
        return false;
    }
    for bond in &state.card_runtime.twin_bonds {
        let partner = if bond.first_id == piece.id {
            Some(bond.second_id)
        } else if bond.second_id == piece.id {
            Some(bond.first_id)
        } else {
            None
        };
        if let Some((partner_from, partner)) = partner.and_then(|id| state.board.find_by_id(id))
            && (engine.fianchetto_destination_blocked(state, partner, partner_from, to)
                || engine.majesty_destination_blocked(state, partner, to))
        {
            return false;
        }
    }
    state.board.set(from, None);
    state.board.set(to, Some(piece));
    moved.push(piece.id);
    true
}

pub(in crate::engine) fn valid_selection(
    engine: &Engine,
    state: &GameState,
    squares: &[Square],
) -> bool {
    if squares.is_empty() || squares.len() > 3 {
        return false;
    }
    let mut probe = state.clone();
    let mut moved = Vec::new();
    for square in squares {
        advance(engine, &mut probe, state.turn, *square, &mut moved);
    }
    !moved.is_empty()
}
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let squares = squares_target(target)?;
    let en_passant_victim = state
        .en_passant
        .and_then(|ep| state.board.get(ep.captured).map(|p| (ep.captured, p.id)));
    let mut moved = Vec::new();
    for square in squares {
        advance(engine, state, actor, square, &mut moved);
    }
    if moved.is_empty() {
        return Err(EngineError::Invariant(
            "evacuation has no movable target".into(),
        ));
    }
    state
        .card_runtime
        .moved_piece_ids
        .extend(moved.iter().copied());
    if en_passant_victim
        .is_some_and(|(square, id)| state.board.get(square).is_none_or(|p| p.id != id))
    {
        state.en_passant = None;
    }
    engine.settle_rule_black_holes(state, events);
    engine.resolve_missing_king(state, events);
    state.card_runtime.evacuation_no_capture.retain(|id| {
        state
            .board
            .find_by_id(*id)
            .is_some_and(|(_, p)| p.owner != actor)
    });
    state.card_runtime.evacuation_no_capture.extend(
        moved
            .into_iter()
            .filter(|id| state.board.find_by_id(*id).is_some()),
    );
    Ok(())
}
fn enumerate(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    card: &CardInstance,
    moved: &mut Vec<u32>,
    path: &mut Vec<Square>,
    out: &mut Vec<Action>,
) {
    if !path.is_empty() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(path.clone()),
        });
    }
    if path.len() == 3 {
        return;
    }
    let positions: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| p.owner == actor && !moved.contains(&p.id))
        .map(|(s, _)| s)
        .collect();
    for from in positions {
        let before = state.board.clone();
        let count = moved.len();
        if advance(engine, state, actor, from, moved) {
            path.push(from);
            enumerate(engine, state, actor, card, moved, path, out);
            path.pop();
        }
        moved.truncate(count);
        state.board = before;
    }
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    let mut probe = state.clone();
    enumerate(
        engine,
        &mut probe,
        card.owner,
        card,
        &mut Vec::new(),
        &mut Vec::new(),
        out,
    );
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    static ENGINE: std::sync::OnceLock<Engine> = std::sync::OnceLock::new();
    let engine = ENGINE.get_or_init(|| Engine::bundled_latest().expect("bundled latest ruleset"));
    let mut probe = state.clone();
    let squares: Vec<_> = state
        .board
        .iter()
        .filter_map(|(s, p)| (p.owner == owner).then_some(s))
        .collect();
    require(
        squares
            .into_iter()
            .any(|s| advance(engine, &mut probe, owner, s, &mut Vec::new())),
        "후퇴할 아군 기물이 필요합니다.",
    )
}

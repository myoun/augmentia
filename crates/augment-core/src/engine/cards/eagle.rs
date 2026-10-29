use super::*;
pub(super) const ID: &str = crate::EAGLE;
impl Engine {
    pub(in crate::engine) fn current_eagle_candidates(
        &self,
        state: &GameState,
        origin: Square,
        owner: Player,
    ) -> Vec<Square> {
        let mut cells: Vec<_> = (0..64)
            .filter_map(Square::from_index)
            .filter(|s| {
                self.card_landing_open(state, *s)
                    && !self.d4_restricts(state, owner, false, *s)
                    && !self.rule_cell_collapsed(state, *s)
                    && !self.is_square_attacked_by(state, *s, owner.opponent())
            })
            .collect();
        let distance = |s: Square| {
            s.file()
                .abs_diff(origin.file())
                .max(s.rank().abs_diff(origin.rank()))
        };
        if let Some(min) = cells.iter().map(|s| distance(*s)).min() {
            cells.retain(|s| distance(*s) == min);
        }
        cells.sort_by_key(|s| s.to_string());
        cells
    }
}
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    if *target != ActionTarget::None {
        return Err(EngineError::Invariant("eagle takes no selection".into()));
    }
    let knights: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| p.owner == actor && p.kind == PieceKind::Knight)
        .collect();
    if knights.is_empty() {
        return Err(EngineError::Invariant("no knights to exchange".into()));
    }
    for (square, _) in &knights {
        engine.transform_current_piece(state, *square, PieceKind::Alibaba, true)?;
    }
    for (origin, _) in knights {
        let cells = engine.current_eagle_candidates(state, origin, actor);
        if cells.is_empty() {
            continue;
        }
        let index = rule_random(state, events, cells.len(), "card.eagle.placement").unwrap();
        let square = cells[index];
        let piece = Piece::new(actor, PieceKind::Alibaba).with_id(state.allocate_piece_id());
        state.board.set(square, Some(piece));
        engine.register_spawned_piece(state, square, piece, true);
        state
            .rule_runtime
            .piece_fresh_until
            .insert(piece.id, rule_halfmove(state) + 2);
    }
    engine.resolve_gomoku_victory(state);
    Ok(())
}
pub(super) fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if state
        .board
        .iter()
        .any(|(_, p)| p.owner == card.owner && p.kind == PieceKind::Knight)
    {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
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
            .any(|(_, p)| p.owner == owner && p.kind == PieceKind::Knight),
        "아군 나이트가 필요합니다.",
    )
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

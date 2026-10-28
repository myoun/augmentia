use super::*;
pub(super) const ID: &str = crate::CHESS_960;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_chess_960(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

pub(super) fn back_rank(mut index: usize) -> [PieceKind; 8] {
    let mut layout = [None; 8];
    let dark = [0, 2, 4, 6];
    let light = [1, 3, 5, 7];
    layout[dark[index % 4]] = Some(PieceKind::Bishop);
    index /= 4;
    layout[light[index % 4]] = Some(PieceKind::Bishop);
    index /= 4;
    let open: Vec<_> = (0..8).filter(|i| layout[*i].is_none()).collect();
    layout[open[index % 6]] = Some(PieceKind::Queen);
    index /= 6;
    let open: Vec<_> = (0..8).filter(|i| layout[*i].is_none()).collect();
    let pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ];
    let (a, b) = pairs[index % 10];
    layout[open[a]] = Some(PieceKind::Knight);
    layout[open[b]] = Some(PieceKind::Knight);
    let open: Vec<_> = (0..8).filter(|i| layout[*i].is_none()).collect();
    for (i, k) in open
        .into_iter()
        .zip([PieceKind::Rook, PieceKind::King, PieceKind::Rook])
    {
        layout[i] = Some(k);
    }
    layout.map(Option::unwrap)
}
impl Engine {
    pub(in crate::engine) fn install_chess_960(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let layout = back_rank(rule_random(state, events, 960, "rule.chess-960.layout").unwrap());
        let before = state.board.clone();
        let mut placements = Vec::new();
        for owner in [Player::White, Player::Black] {
            let rank = if owner == Player::White { 0 } else { 7 };
            let entries: Vec<_> = web_ordered_pieces(state)
                .into_iter()
                .filter(|(s, p)| p.owner == owner && s.rank() == rank && p.kind != PieceKind::Pawn)
                .collect();
            if entries.len() != 8 {
                return Err(EngineError::Invariant(
                    "960 체스는 각 홈 랭크에 표준 기물 8개가 필요합니다.".into(),
                ));
            }
            let mut used = BTreeSet::new();
            for (file, kind) in layout.into_iter().enumerate() {
                let Some((_, piece)) = entries
                    .iter()
                    .find(|(_, p)| p.kind == kind && !used.contains(&p.id))
                else {
                    return Err(EngineError::Invariant(
                        "960 체스의 기물 구성이 맞지 않습니다.".into(),
                    ));
                };
                used.insert(piece.id);
                placements.push((Square::new(file as u8, rank).unwrap(), *piece));
            }
        }
        for (square, piece) in before.iter() {
            if placements.iter().any(|(_, p)| p.id == piece.id) {
                state.board.set(square, None);
            }
        }
        for (square, piece) in placements {
            state.board.set(square, Some(piece));
            state.card_runtime.piece_origins.insert(piece.id, square);
        }
        state.en_passant = None;
        Ok(())
    }
}

use super::*;
pub(super) const ID: &str = crate::WANTED;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn ranged(state: &GameState, piece: Piece, square: Square) -> bool {
    let kind = if piece.kind == PieceKind::Trickster {
        state
            .card_runtime
            .trickster_move_kinds
            .get(&piece.id)
            .copied()
            .unwrap_or(PieceKind::Queen)
    } else {
        piece.kind
    };
    match kind {
        PieceKind::Brutus
        | PieceKind::BigBishop
        | PieceKind::Chancellor
        | PieceKind::Archbishop
        | PieceKind::Rook
        | PieceKind::Bishop
        | PieceKind::Queen
        | PieceKind::Bear
        | PieceKind::Amazon
        | PieceKind::Cardinal
        | PieceKind::Hook
        | PieceKind::Windmill
        | PieceKind::BigRook
        | PieceKind::Jester
        | PieceKind::Idol
        | PieceKind::Grappler
        | PieceKind::RevolvingDoor => true,
        PieceKind::Clockwork => (-1..=1).any(|df| {
            (-1..=1).any(|dr| {
                (df != 0 || dr != 0)
                    && square
                        .offset(df, dr)
                        .and_then(|s| state.board.get(s))
                        .is_some_and(|p| p.owner == piece.owner)
            })
        }),
        PieceKind::Princess => !state
            .board
            .iter()
            .any(|(_, p)| p.owner == piece.owner && p.kind == PieceKind::Queen),
        PieceKind::MagicGirl => state.card_runtime.magic_girl_surge[piece.owner.index()],
        PieceKind::Berserker => {
            state
                .board
                .iter()
                .filter(|(_, p)| p.owner == piece.owner)
                .count()
                <= 9
        }
        _ => false,
    }
}
fn targets(state: &GameState, owner: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(s, p)| {
            p.owner == owner.opponent()
                && !state.card_runtime.wanted_piece_ids.contains(&p.id)
                && ranged(state, *p, *s)
        })
        .collect()
}
pub(super) fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let candidates = targets(state, owner);
    if candidates.is_empty() {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant("no wanted target".into()));
    }
    let index = if candidates.len() == 1 {
        0
    } else {
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        events.push(Event::RandomSample {
            purpose: "acquire.opening.wanted.target".into(),
            roll_53: roll,
        });
        index.unwrap()
    };
    let piece = candidates[index].1;
    state.card_runtime.wanted_piece_ids.insert(piece.id);
    state.card_runtime.submerged_piece_ids.insert(piece.id);
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    require(
        !targets(state, owner).is_empty(),
        "수배할 상대 원거리 기물이 필요합니다.",
    )
}

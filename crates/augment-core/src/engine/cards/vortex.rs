use super::*;
pub(super) const ID: &str = crate::VORTEX;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn candidates(engine: &Engine, state: &GameState, owner: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| {
            p.owner == owner.opponent()
                && p.kind != PieceKind::Pawn
                && !matches!(
                    p.kind,
                    PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop
                )
                && !state.card_runtime.large_pieces.contains_key(&p.id)
                && !engine.is_royal_piece(state, *p)
                && !engine.piece_has_ability(state, *p, PieceKind::Slime)
        })
        .collect()
}
fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let entries = candidates(engine, state, actor);
    if entries.len() < 2 {
        return Err(EngineError::Invariant("vortex requires two targets".into()));
    }
    let mut pieces: Vec<_> = entries.iter().map(|(_, p)| *p).collect();
    for i in (1..pieces.len()).rev() {
        let (roll, index) = state.rng_mut().web_choice_index(i + 1);
        events.push(Event::RandomSample {
            purpose: "vortex".into(),
            roll_53: roll,
        });
        pieces.swap(i, index.expect("nonempty shuffle"));
    }
    for ((square, _), piece) in entries.into_iter().zip(pieces) {
        state.board.set(square, Some(piece));
    }
    Ok(())
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if candidates(engine, state, card.owner).len() >= 2 {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    let count = state
        .board
        .iter()
        .filter(|(_, p)| {
            p.owner == owner.opponent()
                && p.kind != PieceKind::Pawn
                && !matches!(
                    p.kind,
                    PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop
                )
                && !state.card_runtime.large_pieces.contains_key(&p.id)
                && !royal(state, *p)
                && p.kind != PieceKind::Merchant
                && p.kind != PieceKind::Slime
                && !(p.kind == PieceKind::Trickster
                    && state.card_runtime.trickster_move_kinds.get(&p.id)
                        == Some(&PieceKind::Slime))
        })
        .count();
    require(count >= 2, "섞을 상대 기물이 두 개 이상 필요합니다.")
}

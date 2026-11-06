use super::*;
pub(super) const ID: &str = crate::TWINS;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn play(
    _: &Engine,
    state: &mut GameState,
    _: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let [first, second] = two_square_target(target)?;
    let first = state
        .board
        .get(first)
        .ok_or_else(|| EngineError::Invariant("first twin disappeared".into()))?;
    let second = state
        .board
        .get(second)
        .ok_or_else(|| EngineError::Invariant("second twin disappeared".into()))?;
    state.card_runtime.twin_bonds.push(TwinBond {
        first_id: first.id,
        second_id: second.id,
    });
    Ok(())
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    let candidates: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| {
            p.owner == card.owner
                && !engine.piece_has_ability(state, *p, PieceKind::Slime)
                && !matches!(
                    p.kind,
                    PieceKind::Colossus
                        | PieceKind::BigRook
                        | PieceKind::BigBishop
                        | PieceKind::Football
                )
                && !state
                    .card_runtime
                    .twin_bonds
                    .iter()
                    .any(|bond| bond.first_id == p.id || bond.second_id == p.id)
        })
        .map(|(square, _)| square)
        .collect();
    for i in 0..candidates.len() {
        for j in i + 1..candidates.len() {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![candidates[i], candidates[j]]),
            });
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![candidates[j], candidates[i]]),
            });
        }
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    let pieces: Vec<_> = state
        .board
        .iter()
        .filter_map(|(_, p)| {
            (p.owner == owner && !royal(state, p) && p.kind != PieceKind::Merchant)
                .then_some(p.kind)
        })
        .collect();
    require(
        pieces
            .iter()
            .enumerate()
            .any(|(i, kind)| pieces[i + 1..].contains(kind)),
        "같은 종류의 아군 비왕실 기물 두 개가 필요합니다.",
    )
}

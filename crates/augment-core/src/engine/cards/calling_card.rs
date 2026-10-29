use super::*;
pub(super) const ID: &str = crate::CALLING_CARD;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
fn candidates(state: &GameState, owner: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| {
            p.owner == owner.opponent()
                && p.kind != PieceKind::Pawn
                && !royal(state, *p)
                && !matches!(
                    p.kind,
                    PieceKind::Monster
                        | PieceKind::Football
                        | PieceKind::Scarecrow
                        | PieceKind::Guard
                        | PieceKind::RevolvingDoor
                        | PieceKind::Jester
                )
        })
        .collect()
}
pub(super) fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let targets = candidates(state, owner);
    if targets.is_empty() {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant("no calling-card target".into()));
    }
    let index = if targets.len() == 1 {
        0
    } else {
        let (roll, index) = state.rng_mut().web_choice_index(targets.len());
        events.push(Event::RandomSample {
            purpose: "acquire.opening.calling-card.target".into(),
            roll_53: roll,
        });
        index.unwrap()
    };
    state
        .card_runtime
        .calling_cards
        .insert(targets[index].1.id, owner);
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    require(
        !candidates(state, owner).is_empty(),
        "예고장을 보낼 상대 기물이 필요합니다.",
    )
}

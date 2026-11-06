use super::*;
pub(super) const ID: &str = crate::ICE_SHEET;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn targets(state: &GameState, actor: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(square, p)| {
            p.owner == actor.opponent()
                && (matches!(
                    p.kind,
                    PieceKind::MagicGirl | PieceKind::Berserker | PieceKind::Trickster
                ) || super::wanted::ranged(state, *p, *square))
        })
        .collect()
}
fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    for (_, piece) in targets(state, actor) {
        state.card_runtime.ice_sheet.insert(
            piece.id,
            TimedPieceStatus {
                by: actor,
                remaining_turns: 3,
            },
        );
    }
    Ok(())
}
fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if !targets(state, state.turn).is_empty() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

use super::*;
pub(super) const ID: &str = crate::DICE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let kinds = [
        PieceKind::Pawn,
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
        PieceKind::King,
    ];
    let (roll, index) = state.rng_mut().web_choice_index(kinds.len());
    events.push(Event::RandomSample {
        purpose: "dice".into(),
        roll_53: roll,
    });
    state.card_runtime.dice_locks[actor.opponent().index()] = Some(DiceLockStatus {
        kind: kinds[index.expect("six-sided die")],
        remaining_turns: 2,
    });
    Ok(())
}
fn actions(_: &Engine, _: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    out.push(Action::UseCard {
        instance_id: card.instance_id,
        target: ActionTarget::None,
    });
}
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

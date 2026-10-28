use super::*;
pub(super) const ID: &str = crate::FRENZY;
pub(super) fn play(
    _engine: &Engine,
    state: &mut GameState,
    _actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, pawn) = piece_target(state, target)?;
    state.card_runtime.frenzy_pawns.insert(pawn.id);
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let actor = state.turn;
    for (square, piece) in state.board.iter() {
        if piece.owner == actor && piece.kind == PieceKind::Pawn {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let count = |owner, kind| {
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .count()
    };
    let own = |kind| count(player, kind);
    require(own(Pawn) > 0, "아군 폰이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

use super::*;
pub(super) const ID: &str = crate::CHAMELEON_MUTATION;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn play(
    _: &Engine,
    state: &mut GameState,
    _: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    for square in squares_target(target)? {
        let piece = state
            .board
            .get(square)
            .ok_or_else(|| EngineError::Invariant("missing chameleon target".into()))?;
        state.card_runtime.chameleon_piece_ids.insert(piece.id);
    }
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    let squares = web_ordered_pieces(state)
        .into_iter()
        .filter_map(|(square, piece)| {
            (piece.owner == card.owner
                && !engine.is_royal_piece(state, piece)
                && !matches!(
                    piece.kind,
                    PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop
                ))
            .then_some(square)
        })
        .collect::<Vec<_>>();
    for i in 0..squares.len() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(vec![squares[i]]),
        });
        for j in i + 1..squares.len() {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![squares[i], squares[j]]),
            });
            for k in j + 1..squares.len() {
                out.push(Action::UseCard {
                    instance_id: card.instance_id,
                    target: ActionTarget::Squares(vec![squares[i], squares[j], squares[k]]),
                });
            }
        }
    }
}
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

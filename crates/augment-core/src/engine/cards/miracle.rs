use super::*;
pub(super) const ID: &str = crate::MIRACLE;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let opponent = actor.opponent();
    // Miracle (live JS:91144-91179): every enemy piece currently capturable by a
    // friendly bishop switches owners; every friendly bishop is then sacrificed.
    let bishops = state
        .board
        .iter()
        .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Bishop)
        .collect::<Vec<_>>();
    let mut capturable = Vec::new();
    for (bishop_square, bishop) in &bishops {
        let moves = {
            let mut candidates = Vec::new();
            engine.generate_slider_moves(
                state,
                *bishop_square,
                *bishop,
                &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
                &mut candidates,
            );
            candidates
        };
        for action in moves {
            if let Action::Move(move_action) = action {
                if let Some(target_piece) = state.board.get(move_action.to) {
                    if target_piece.owner == opponent
                        && !capturable
                            .iter()
                            .any(|(square, _): &(Square, Piece)| *square == move_action.to)
                    {
                        capturable.push((move_action.to, target_piece));
                    }
                }
            }
        }
    }
    for (square, piece) in capturable {
        let mut converted = piece;
        converted.owner = actor;
        state.board.set(square, Some(converted));
    }
    for (square, bishop) in bishops {
        engine.delete_piece_for_card(state, square, bishop, actor, MIRACLE, events);
    }
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    _state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    let push_none = |actions: &mut Vec<Action>| {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        })
    };
    push_none(actions);
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    // Upstream `v19DraftRequirementMet` allows ANY unconditionally.
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

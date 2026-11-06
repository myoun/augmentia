use super::*;

pub(super) fn targets(
    engine: &Engine,
    state: &GameState,
    actor: Player,
    id: &str,
) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(square, piece)| {
            if piece.owner != actor.opponent()
                || state.card_runtime.stealth_piece_ids.contains(&piece.id)
            {
                return false;
            }
            let light = (square.file() + square.rank()) % 2 == 1;
            if state.card_runtime.camouflage_color
                && !engine.is_royal_piece(state, *piece)
                && (piece.owner == Player::White) == light
            {
                return false;
            }
            match id {
                crate::DISARM => !engine.is_royal_piece(state, *piece),
                crate::INERTIA | crate::SEVERANCE => super::wanted::ranged(state, *piece, *square),
                _ => false,
            }
        })
        .collect()
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    out: &mut Vec<Action>,
) {
    for (square, _) in targets(engine, state, state.turn, &card.definition_id) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        });
    }
}

pub(super) fn eligibility(state: &GameState, owner: Player, id: &str) -> CardEligibility {
    // Draft eligibility is defined by the catalog draw requirement; hidden targets
    // are checked again against the player's view when the card is used.
    let available = state.board.iter().any(|(square, piece)| {
        piece.owner == owner.opponent()
            && match id {
                crate::DISARM => !royal(state, piece) && piece.kind != PieceKind::Merchant,
                _ => {
                    !royal(state, piece)
                        && piece.kind != PieceKind::Merchant
                        && super::wanted::ranged(state, piece, square)
                }
            }
    });
    require(available, "조건에 맞는 상대 기물이 필요합니다.")
}

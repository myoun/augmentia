use super::*;
pub(super) const ID: &str = crate::GREEK_GIFT;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let id = ID;
    let (square, piece) = piece_target(state, target)?;
    engine.delete_piece_for_card(state, square, piece, actor, id, events);
    state.card_runtime.october.greek_gift_pending[actor.index()] = true;
    engine.resolve_gomoku_victory(state);
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    for (square, piece) in state.board.iter() {
        if !engine.is_royal_piece(state, piece)
            && (piece.owner == state.turn && piece.kind == PieceKind::Bishop)
        {
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
    context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    if context == AcquisitionContext::Draft {
        return CardEligibility::Eligible;
    }
    require(
        has_own(&|p| p.kind == Bishop && !royal(state, p)),
        "희생할 비왕실 아군 비숍이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

impl Engine {
    pub(in crate::engine) fn settle_greek_gift(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        if std::mem::take(&mut state.card_runtime.october.greek_gift_pending[actor.index()]) {
            let kings: Vec<_> = state
                .board
                .iter()
                .filter(|(_, p)| p.owner == actor.opponent() && self.is_royal_piece(state, *p))
                .flat_map(|(s, p)| {
                    if state.card_runtime.large_pieces.contains_key(&p.id) {
                        large_piece_cells(s)
                    } else {
                        vec![s]
                    }
                })
                .collect();
            let victims: Vec<_> = state
                .board
                .iter()
                .filter(|(s, p)| {
                    p.owner == actor.opponent()
                        && p.kind == PieceKind::Pawn
                        && !self.is_royal_piece(state, *p)
                        && kings.iter().any(|king| {
                            s.file()
                                .abs_diff(king.file())
                                .max(s.rank().abs_diff(king.rank()))
                                == 1
                        })
                })
                .collect();
            for (s, p) in victims {
                self.remove_piece_for_card(state, s, p, actor, crate::GREEK_GIFT, events);
            }
            self.resolve_missing_king(state, events);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;
    #[test]
    fn greek_gift_sacrifice_is_noncapture_and_waits_for_actual_turn_end() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("c1", Player::White, PieceKind::Bishop),
            ("g7", Player::Black, PieceKind::Pawn),
            ("h7", Player::Black, PieceKind::Pawn),
            ("g8", Player::Black, PieceKind::Pawn),
            ("b7", Player::Black, PieceKind::Pawn),
            ("g6", Player::White, PieceKind::Pawn),
            ("f8", Player::Black, PieceKind::Knight),
        ]);
        let bishop = at(&s, "c1");
        let protected = at(&s, "g7");
        let royal = at(&s, "g8");
        s.card_runtime.recurrence_piece_ids.insert(bishop.id);
        s.card_runtime.shielded_piece_ids.insert(protected.id);
        s.card_runtime.royal_piece_ids.insert(royal.id);
        use_at(&mut s, &engine, crate::GREEK_GIFT, single("c1"));
        assert!(s.board.get("c1".parse().unwrap()).is_none());
        assert!(s.card_runtime.pending_recurrences.is_empty());
        s.actions_remaining = 2;
        s = engine.apply(&s, movement("a1", "a2")).unwrap().state;
        assert!(s.card_runtime.october.greek_gift_pending[0]);
        assert_eq!(at(&s, "g7"), protected);
        s = engine.apply(&s, movement("a2", "a3")).unwrap().state;
        for q in ["g7", "h7"] {
            assert!(s.board.get(q.parse().unwrap()).is_none());
        }
        for q in ["g8", "b7", "g6"] {
            assert!(s.board.get(q.parse().unwrap()).is_some());
        }
        assert!(!s.card_runtime.october.greek_gift_pending[0]);
    }
}

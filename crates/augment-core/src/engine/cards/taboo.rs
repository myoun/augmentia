use super::*;
pub(super) const ID: &str = crate::TABOO;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let [square, landing] = two_square_target(target)?;
    let piece = state
        .board
        .get(square)
        .ok_or_else(|| EngineError::Invariant("missing taboo queen".into()))?;
    state.card_runtime.effects.pending_taboo.push(PendingTaboo {
        owner: actor,
        piece_id: piece.id,
        square: landing,
    });
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
            && (piece.owner == state.turn && piece.kind == PieceKind::Queen)
        {
            for index in 0..64 {
                let landing = Square::from_index(index).unwrap();
                if engine.card_landing_open(state, landing) {
                    actions.push(Action::UseCard {
                        instance_id: card.instance_id,
                        target: ActionTarget::Squares(vec![square, landing]),
                    });
                }
            }
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
        has_own(&|p| p.kind == Queen && !royal(state, p)),
        "비왕실 아군 퀸이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

impl Engine {
    pub(in crate::engine) fn settle_taboo(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        let pending = std::mem::take(&mut state.card_runtime.effects.pending_taboo);
        for entry in pending {
            if entry.owner != actor {
                state.card_runtime.effects.pending_taboo.push(entry);
                continue;
            }
            if state.status != GameStatus::Ongoing {
                continue;
            }
            let Some((source, piece)) = state.board.find_by_id(entry.piece_id) else {
                continue;
            };
            if piece.owner != actor {
                continue;
            }
            // Follow the identity even after a move or transformation this turn.
            self.delete_piece_for_card(state, source, piece, actor, crate::TABOO, events);
            self.resolve_missing_king(state, events);
            if state.status != GameStatus::Ongoing {
                continue;
            }
            let mut seen = BTreeSet::new();
            for square in [source, entry.square] {
                if !seen.insert(square) {
                    continue;
                }
                self.crush_card_landing(state, square, actor, crate::TABOO, events);
                // Source removes movable wall/football/monster installations.
                state.card_runtime.remove_barricade(square);
                if self.card_landing_open(state, square) {
                    let monster =
                        Piece::new(actor, PieceKind::Monster).with_id(state.allocate_piece_id());
                    state.board.set(square, Some(monster));
                    state.card_runtime.moved_piece_ids.insert(monster.id);
                    state.card_runtime.piece_origins.insert(monster.id, square);
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("taboo-monster:{}:{}", monster.id, square),
                    });
                }
            }
            self.resolve_missing_king(state, events);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingTaboo {
    pub owner: Player,
    pub piece_id: u32,
    pub square: Square,
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;
    #[test]
    fn taboo_follows_moving_queen_crushes_landing_and_spawns_neutral_monsters() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
        let queen = at(&s, "d1");
        use_at(&mut s, &engine, crate::TABOO, pair("d1", "e5"));
        assert_eq!(at(&s, "d1"), queen);
        assert!(s.board.get("e5".parse().unwrap()).is_none());
        s.board.set(
            "e5".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Pawn).with_id(100)),
        );
        s.card_runtime.recurrence_piece_ids.insert(queen.id);
        s = engine.apply(&s, movement("d1", "d4")).unwrap().state;
        assert!(s.board.find_by_id(queen.id).is_none());
        assert!(s.card_runtime.pending_recurrences.is_empty());
        for q in ["d4", "e5"] {
            assert_eq!(at(&s, q).kind, PieceKind::Monster);
        }
        assert!(s.card_runtime.effects.pending_taboo.is_empty());
        s.turn = Player::White;
        s.card_runtime.royal_command[0] = Some(crate::RoyalCommandStatus {
            starts_at_turn: 0,
            expires_at_turn: 100,
        });
        assert!(
            !has_move(&engine, &s, "d4", "c4"),
            "neutral monsters do not inherit the summoner's grants"
        );
    }
    #[test]
    fn taboo_deduplicates_landings_and_cancels_when_identity_is_gone_or_enemy() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
        use_at(&mut s, &engine, crate::TABOO, pair("d1", "d4"));
        let out = engine.apply(&s, movement("d1", "d4")).unwrap();
        assert_eq!(
            out.state
                .board
                .iter()
                .filter(|(_, p)| p.kind == PieceKind::Monster)
                .count(),
            1
        );
        for convert in [false, true] {
            let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
            use_at(&mut s, &engine, crate::TABOO, pair("d1", "d4"));
            let mut queen = at(&s, "d1");
            queen.owner = Player::Black;
            s.board.set("d1".parse().unwrap(), convert.then_some(queen));
            handoff(&engine, &mut s);
            assert!(!s.board.iter().any(|(_, p)| p.kind == PieceKind::Monster));
            assert!(s.card_runtime.effects.pending_taboo.is_empty());
        }
    }
}

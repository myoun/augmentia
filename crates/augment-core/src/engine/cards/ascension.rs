use super::*;
pub(super) const ID: &str = crate::ASCENSION;

/// The October 27 balance pass moved Ascension to END and lengthened the flight
/// from three owner starts to seven (main bundle 49208).
const RETURN_TURNS: u32 = 7;

pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let id = ID;
    let [square, landing] = two_square_target(target)?;
    let piece = state
        .board
        .get(square)
        .ok_or_else(|| EngineError::Invariant("missing ascension queen".into()))?;
    let mut saved = CardRuntimeState::default();
    turn_effects::restore_piece_runtime(&mut saved, &state.card_runtime, &[piece.id]);
    engine.delete_piece_for_card(state, square, piece, actor, id, events);
    state
        .card_runtime
        .effects
        .pending_ascensions
        .push(PendingAscension {
            piece,
            square: landing,
            due_turn: state.turns_taken[actor.index()].saturating_add(RETURN_TURNS),
            runtime: Box::new(saved),
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
                // The current contract no longer requires the landing square to be
                // empty: the return clears whatever occupies it, so only the hard
                // restrictions (d4, collapsed cells, pending spawns) apply
                // (ascensionLandingAllowed, main bundle 49123-49125).
                if !engine.d4_restricts(state, piece.owner, false, landing)
                    && !engine.collapsed_card_landing(state, landing)
                    && !state
                        .card_runtime
                        .pending_lobsters
                        .iter()
                        .any(|pending| pending.square == landing)
                    && !state
                        .card_runtime
                        .pending_scarecrows
                        .iter()
                        .any(|pending| pending.square == landing)
                    && !state
                        .card_runtime
                        .pending_portals
                        .iter()
                        .any(|pending| pending.squares.contains(&landing))
                {
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
    pub(in crate::engine) fn settle_ascension(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let pending = std::mem::take(&mut state.card_runtime.effects.pending_ascensions);
        for entry in pending {
            if entry.piece.owner != owner || state.turns_taken[owner.index()] < entry.due_turn {
                state.card_runtime.effects.pending_ascensions.push(entry);
                continue;
            }
            if self.d4_restricts(state, entry.piece.owner, false, entry.square) {
                continue;
            }
            // Landing clears alternate quantum occupancy without killing its real piece.
            state
                .card_runtime
                .quantum_positions
                .retain(|_, s| *s != entry.square);
            self.crush_card_landing(state, entry.square, owner, crate::ASCENSION, events);
            state.card_runtime.remove_barricade(entry.square);
            if !self.collapsed_card_landing(state, entry.square)
                && self.board_piece_at(state, entry.square).is_none()
            {
                state.board.set(entry.square, Some(entry.piece));
                turn_effects::restore_piece_runtime(
                    &mut state.card_runtime,
                    &entry.runtime,
                    &[entry.piece.id],
                );
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("ascension-return:{}:{}", entry.piece.id, entry.square),
                });
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingAscension {
    pub piece: Piece,
    pub square: Square,
    pub due_turn: u32,
    /// Only the suspended identity's runtime; no nested replays or triggers.
    pub runtime: Box<CardRuntimeState>,
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;
    #[test]
    fn ascension_respects_d4_at_selection_and_again_on_return() {
        let engine = Engine::bundled_latest().unwrap();
        let base = position(&[("d1", Player::White, PieceKind::Queen)]);
        let blocked = engine
            .acquire_for_setup(&base, crate::D4, Player::Black)
            .unwrap()
            .state;
        let blocked = engine
            .grant_card(&blocked, 2, crate::ASCENSION, Player::White)
            .unwrap()
            .state;
        let action = Action::UseCard {
            instance_id: 2,
            target: pair("d1", "d5"),
        };
        assert!(!engine.legal_actions(&blocked).unwrap().contains(&action));
        assert!(engine.apply(&blocked, action).is_err());

        let mut pending = base;
        let queen = at(&pending, "d1");
        use_at(&mut pending, &engine, crate::ASCENSION, pair("d1", "d5"));
        pending = engine
            .acquire_for_setup(&pending, crate::D4, Player::Black)
            .unwrap()
            .state;
        let blocker = Piece::new(Player::Black, PieceKind::Pawn).with_id(100);
        pending.board.set("d5".parse().unwrap(), Some(blocker));
        pending.turns_taken[Player::White.index()] = 7;
        engine.settle_ascension(&mut pending, Player::White, &mut Vec::new());
        assert_eq!(pending.board.get("d5".parse().unwrap()), Some(blocker));
        assert!(pending.board.find_by_id(queen.id).is_none());
        assert!(pending.card_runtime.effects.pending_ascensions.is_empty());
    }
    #[test]
    fn ascension_returns_on_third_owner_start_with_frozen_metadata_and_original_identity() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
        let mut queen = at(&s, "d1");
        queen.capture_count = 4;
        s.board.set("d1".parse().unwrap(), Some(queen));
        s.card_runtime.shielded_piece_ids.insert(queen.id);
        s.card_runtime.frozen.insert(
            queen.id,
            FrozenStatus {
                source: Player::Black,
                remaining_turns: 2,
            },
        );
        s.card_runtime.recurrence_piece_ids.insert(queen.id);
        use_at(&mut s, &engine, crate::ASCENSION, pair("d1", "e5"));
        assert!(s.board.find_by_id(queen.id).is_none());
        assert!(!s.card_runtime.shielded_piece_ids.contains(&queen.id));
        let pending = &s.card_runtime.effects.pending_ascensions[0];
        assert!(pending.runtime.effects.pending_ascensions.is_empty());
        assert_eq!(pending.due_turn, 7);
        // Place an occupant on the landing square before the return is due: the
        // current contract allows a non-empty landing and clears it on return.
        s.board.set(
            "e5".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Pawn).with_id(100)),
        );
        s.card_runtime.shielded_piece_ids.insert(100);
        for _ in 0..14 {
            handoff(&engine, &mut s);
            if s.card_runtime.effects.pending_ascensions.is_empty() {
                break;
            }
            assert!(s.board.find_by_id(queen.id).is_none());
        }
        assert_eq!(at(&s, "e5"), queen);
        assert!(s.card_runtime.recurrence_piece_ids.contains(&queen.id));
        assert!(s.card_runtime.effects.pending_ascensions.is_empty());
        assert!(s.card_runtime.shielded_piece_ids.contains(&queen.id));
        assert_eq!(s.card_runtime.frozen[&queen.id].remaining_turns, 2);
        assert!(!s.card_runtime.moved_piece_ids.contains(&queen.id));
        assert!(
            !s.card_runtime
                .fresh_no_capture_until
                .contains_key(&queen.id)
        );
    }
    #[test]
    fn ascension_extra_actions_do_not_advance_return_and_collapsed_landing_consumes_reservation() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
        use_at(&mut s, &engine, crate::ASCENSION, pair("d1", "a5"));
        s.actions_remaining = 2;
        s = engine.apply(&s, movement("a1", "b2")).unwrap().state;
        assert_eq!(s.turn, Player::White);
        assert_eq!(s.turns_taken[0], 0);
        assert_eq!(s.card_runtime.effects.pending_ascensions[0].due_turn, 7);
        let black = at(&s, "h8");
        s.board.set("h8".parse().unwrap(), None);
        s.board.set("g7".parse().unwrap(), Some(black));
        s.card_runtime.collapsed_files_ranks = true;
        // Seven owner starts, matching the current flight length.
        for _ in 0..14 {
            handoff(&engine, &mut s);
        }
        assert!(s.card_runtime.effects.pending_ascensions.is_empty());
        assert!(s.board.get("a5".parse().unwrap()).is_none());
    }
}

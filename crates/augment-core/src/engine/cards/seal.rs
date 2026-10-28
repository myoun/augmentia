use super::*;
pub(super) const ID: &str = crate::SEAL;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    state.card_runtime.october.pending_seals.push(PendingSeal {
        by: actor,
        piece_id: piece.id,
        due_turn: state.turns_taken[actor.index()].saturating_add(1),
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
            && (piece.owner != state.turn
                && !matches!(
                    piece.kind,
                    PieceKind::King | PieceKind::Monster | PieceKind::Football | PieceKind::Crown
                ))
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
    let has_enemy = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player.opponent() && predicate(p))
    };
    if context == AcquisitionContext::Draft {
        return CardEligibility::Eligible;
    }
    require(
        has_enemy(&|p| !royal(state, p) && !matches!(p.kind, King | Monster | Football | Crown)),
        "속박할 상대 비왕실 기물이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

impl Engine {
    pub(in crate::engine) fn release_seal_after_move(
        &self,
        state: &mut GameState,
        moving: Piece,
        before: &crate::Board,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            return;
        }
        let relocated = before
            .find_by_id(moving.id)
            .zip(state.board.find_by_id(moving.id))
            .is_some_and(|((old, _), (new, _))| old != new);
        if relocated {
            let release: Vec<_> = state
                .board
                .iter()
                .filter(|(_, p)| p.id != moving.id && p.owner == moving.owner)
                .map(|(_, p)| p.id)
                .collect();
            for id in release {
                state.card_runtime.october.bound_piece_ids.remove(&id);
            }
        }
    }
}

impl Engine {
    pub(in crate::engine) fn settle_seal(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let seals = std::mem::take(&mut state.card_runtime.october.pending_seals);
        for seal in seals {
            if seal.by != owner || state.turns_taken[owner.index()] < seal.due_turn {
                state.card_runtime.october.pending_seals.push(seal);
            } else if let Some((_, piece)) = state.board.find_by_id(seal.piece_id)
                && piece.owner == owner.opponent()
                && !self.is_royal_piece(state, piece)
                && piece.kind != PieceKind::King
            {
                state.card_runtime.october.bound_piece_ids.insert(piece.id);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("seal-bound:{}", piece.id),
                });
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingSeal {
    pub by: Player,
    pub piece_id: u32,
    pub due_turn: u32,
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;
    #[test]
    fn seal_tracks_identity_at_next_owner_start_and_ally_move_releases_without_immunity() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d5", Player::Black, PieceKind::Rook),
            ("c3", Player::White, PieceKind::Bishop),
        ]);
        let id = at(&state, "d5").id;
        use_at(&mut state, &engine, crate::SEAL, single("d5"));
        assert!(state.card_runtime.october.bound_piece_ids.is_empty());
        state = engine.apply(&state, movement("a1", "a2")).unwrap().state;
        assert!(has_move(&engine, &state, "d5", "d4"));
        state = engine.apply(&state, movement("d5", "d4")).unwrap().state;
        assert!(state.card_runtime.october.bound_piece_ids.contains(&id));
        assert!(state.card_runtime.october.pending_seals.is_empty());
        assert!(
            has_move(&engine, &state, "c3", "d4"),
            "bound grants no capture protection"
        );
        state = engine.apply(&state, movement("a2", "a3")).unwrap().state;
        assert!(!has_move(&engine, &state, "d4", "d3"));
        state = engine.apply(&state, movement("h8", "h7")).unwrap().state;
        assert!(!state.card_runtime.october.bound_piece_ids.contains(&id));
    }
    #[test]
    fn seal_rechecks_dead_converted_and_royal_targets_without_hitting_replacements() {
        let engine = Engine::bundled_latest().unwrap();
        for variant in 0..3 {
            let mut s = position(&[("d5", Player::Black, PieceKind::Rook)]);
            let id = at(&s, "d5").id;
            use_at(&mut s, &engine, crate::SEAL, single("d5"));
            let mut p = at(&s, "d5");
            match variant {
                0 => p.id = 100,
                1 => p.owner = Player::White,
                _ => {
                    s.card_runtime.royal_piece_ids.insert(id);
                }
            }
            s.board.set("d5".parse().unwrap(), Some(p));
            s.turns_taken[0] = 1;
            engine.settle_delayed_turn_start(&mut s, Player::White, &mut Vec::new());
            assert!(s.card_runtime.october.bound_piece_ids.is_empty());
            assert!(s.card_runtime.october.pending_seals.is_empty());
        }
    }
    #[test]
    fn bound_checker_has_no_forced_jump_and_bound_rook_cannot_be_swapped_or_castled() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("d4", Player::White, PieceKind::Checker),
            ("e5", Player::Black, PieceKind::Pawn),
            ("c2", Player::White, PieceKind::Dragon),
            ("h1", Player::White, PieceKind::Rook),
        ]);
        for q in ["d4", "h1"] {
            s.card_runtime.october.bound_piece_ids.insert(at(&s, q).id);
        }
        assert!(!has_move(&engine, &s, "d4", "f6"));
        assert!(!has_move(&engine, &s, "c2", "h1"));
        assert!(has_move(&engine, &s, "a1", "a2"));
        let white_king = at(&s, "a1");
        s.board.set("a1".parse().unwrap(), None);
        s.board.set("e1".parse().unwrap(), Some(white_king));
        assert!(!has_move(&engine, &s, "e1", "g1"));
    }
}

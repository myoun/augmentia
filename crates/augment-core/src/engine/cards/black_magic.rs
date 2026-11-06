use super::*;
pub(super) const ID: &str = crate::BLACK_MAGIC;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// The current contract summons one allied monster beside the owner's king.
/// Upstream `v19ApplyAlliedBlackMagic` (L202704-202732) walks the own royals,
/// collects open adjacent cells, sorts them by web row then column, and spawns
/// on the first one. Web rows are top-down, so that order is descending core
/// rank and ascending file.
fn candidates(engine: &Engine, state: &GameState, actor: Player) -> Vec<Square> {
    let mut squares = engine.black_magic_spawn_candidates(state, actor);
    squares.retain(|square| engine.card_landing_open(state, *square));
    squares.sort_by_key(|square| (std::cmp::Reverse(square.rank()), square.file()));
    squares
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let Some(square) = candidates(engine, state, actor).into_iter().next() else {
        return Err(EngineError::Invariant(
            "black magic has no open square beside the king".to_owned(),
        ));
    };
    let monster = Piece::new(actor, PieceKind::Monster).with_id(state.allocate_piece_id());
    state.board.set(square, Some(monster));
    // The summoned monster is already moved and belongs to the summoner, which
    // is what separates it from the inert rule monster (`monster_rule`).
    engine.register_spawned_piece(state, square, monster, true);
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("black-magic-monster:{}:{}", monster.id, square),
    });
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    // Only the owner's own turn can place a monster, and PRESERVE_TURN keeps it.
    if card.owner == state.turn && !candidates(engine, state, card.owner).is_empty() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    let has_ring = state.board.iter().any(|(square, piece)| {
        piece.owner == owner
            && royal(state, piece)
            && [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
            .iter()
            .any(|(df, dr)| {
                square.offset(*df, *dr).is_some_and(|cell| {
                    state.board.get(cell).is_none()
                        && !state.card_runtime.barricades.contains(&cell)
                        && !state.card_runtime.black_holes.contains(&cell)
                        && state.card_runtime.crown_pos != Some(cell)
                })
            })
    });
    require(has_ring, "킹 옆에 괴물을 소환할 빈칸이 필요합니다.")
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, handoff, movement, position};
    use super::*;

    fn spawned_monsters(state: &GameState, actor: Player) -> Vec<(Square, Piece)> {
        let mut monsters = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Monster)
            .collect::<Vec<_>>();
        monsters.sort_by_key(|(square, _)| *square);
        monsters
    }

    fn use_card(engine: &Engine, state: &mut GameState, actor: Player) -> Result<(), EngineError> {
        play(
            engine,
            state,
            actor,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        )
    }

    #[test]
    fn black_magic_summons_one_allied_monster_beside_the_king() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        let king = at(&state, "a1");
        let king_square = state.board.find_by_id(king.id).unwrap().0;
        use_card(&engine, &mut state, Player::White).unwrap();
        // Web order sorts by row first, so a2 precedes b1.
        assert_eq!(
            state.board.get("a2".parse().unwrap()).unwrap().kind,
            PieceKind::Monster
        );
        let monsters = spawned_monsters(&state, Player::White);
        assert_eq!(monsters.len(), 1);
        let (square, monster) = monsters[0];
        assert_eq!(square, "a2".parse::<Square>().unwrap());
        assert!(
            square.file().abs_diff(king_square.file()) <= 1
                && square.rank().abs_diff(king_square.rank()) <= 1,
            "the monster must spawn adjacent to the king"
        );
        // Distinguished from the neutral rule monster.
        assert!(
            state.card_runtime.monster_rule.is_none(),
            "black magic must not claim the neutral rule-monster slot"
        );
        assert!(state.card_runtime.moved_piece_ids.contains(&monster.id));
        assert!(state.card_runtime.piece_origins.contains_key(&monster.id));
        assert_eq!(state.turn, Player::White, "PRESERVE_TURN keeps the turn");
    }

    #[test]
    fn black_magic_uses_the_first_web_ordered_open_cell_and_never_overwrites() {
        let engine = Engine::bundled_latest().unwrap();
        // Kill every neighbor except the far corner pair.
        let mut state = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("a2", Player::White, PieceKind::Rook),
        ]);
        use_card(&engine, &mut state, Player::White).unwrap();
        assert_eq!(
            state.board.get("b2".parse().unwrap()).unwrap().kind,
            PieceKind::Monster
        );
        // Both rooks survived: the card never captures or overwrites allies.
        assert_eq!(
            state.board.get("b1".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        assert_eq!(
            state.board.get("a2".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
    }

    #[test]
    fn black_magic_rejects_a_blocked_ring_without_mutating_state() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("a2", Player::White, PieceKind::Rook),
            ("b1", Player::White, PieceKind::Rook),
            ("b2", Player::White, PieceKind::Rook),
        ]);
        let before = state.clone();
        assert!(use_card(&engine, &mut state, Player::White).is_err());
        assert_eq!(state, before);
        assert!(!eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible());
        // Free one cell and the card becomes usable again.
        state.board.set("b2".parse().unwrap(), None);
        assert!(eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible());
        use_card(&engine, &mut state, Player::White).unwrap();
        assert_eq!(
            state.board.get("b2".parse().unwrap()).unwrap().kind,
            PieceKind::Monster
        );
    }

    #[test]
    fn black_magic_monster_stays_with_its_summoner_and_outlives_the_turn() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        use_card(&engine, &mut state, Player::White).unwrap();
        let monster_id = spawned_monsters(&state, Player::White)[0].1.id;
        handoff(&engine, &mut state);
        assert_eq!(state.turn, Player::Black);
        let (_, monster) = state.board.find_by_id(monster_id).unwrap();
        assert_eq!(
            monster.owner,
            Player::White,
            "allied monster keeps its owner"
        );
        assert_ne!(
            state.card_runtime.monster_rule,
            Some(monster_id),
            "allied monsters are not the rule monster"
        );
        let _ = movement("a1", "a2");
        let _ = at(&state, "a2");
    }

    #[test]
    fn black_magic_requires_an_own_royal_to_anchor_the_spawn() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        // Remove the white king; keep the black one.
        state.board.set("a1".parse().unwrap(), None);
        assert!(!eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible());
        let mut out = Vec::new();
        actions(
            &engine,
            &state,
            &CardInstance {
                instance_id: 1,
                definition_id: crate::BLACK_MAGIC.to_owned(),
                owner: Player::White,
                half_stars: 0,
                charges: 1,
                available_after_turn: None,
            },
            &mut out,
        );
        assert!(out.is_empty());
        assert!(use_card(&engine, &mut state, Player::White).is_err());
    }
}

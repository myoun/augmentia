use super::*;
pub(super) const ID: &str = crate::MUTATION;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

/// Mutation is a PASSIVE: acquiring it arms the owner's promotion table so that a
/// last-rank pawn may also promote into a monster. The promotion *choice* itself
/// already exists in `Engine::push_card_pawn_move`, which appends
/// `PieceKind::Monster` whenever `card_runtime.mutation[owner]` is armed, so this
/// file only has to arm the flag and provide the extra-monster summon.
fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.mutation[owner.index()] = true;
    Ok(())
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    if let Some(ineligible) = enabled(state.card_runtime.mutation[owner.index()]) {
        return ineligible;
    }
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == owner && piece.kind == PieceKind::Pawn),
        "괴물로 변이시킬 아군 폰이 필요합니다.",
    )
}

/// Upstream `mutationSummonCandidates` (L4894-4908): the eight neighbours of the
/// promotion square, empty cells first, then enemies, then (when every neighbour
/// is an ally) the allies themselves. Black holes are never candidates.
fn summon_candidates(
    engine: &Engine,
    state: &GameState,
    actor: Player,
    square: Square,
) -> Vec<Square> {
    let ring = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    .into_iter()
    .filter_map(|(file_delta, rank_delta)| square.offset(file_delta, rank_delta))
    .filter(|cell| !state.card_runtime.black_holes.contains(cell))
    .collect::<Vec<_>>();
    let empty = ring
        .iter()
        .copied()
        .filter(|cell| engine.card_landing_open(state, *cell))
        .collect::<Vec<_>>();
    if !empty.is_empty() {
        return empty;
    }
    let occupied = ring
        .iter()
        .copied()
        .filter(|cell| state.board.get(*cell).is_some())
        .collect::<Vec<_>>();
    let enemies = occupied
        .iter()
        .copied()
        .filter(|cell| {
            state
                .board
                .get(*cell)
                .is_some_and(|piece| piece.owner != actor)
        })
        .collect::<Vec<_>>();
    if enemies.is_empty() {
        occupied
    } else {
        enemies
    }
}

impl Engine {
    /// Summon the second monster promised by the current mutation contract.
    ///
    /// Called after a pawn move that promoted into `PieceKind::Monster`; the
    /// promotion square is the anchor. Returns whether a monster was placed.
    /// The first monster (the promotion itself) is produced by the normal move
    /// application, so this function only adds the extra body.
    pub(in crate::engine) fn summon_mutation_extra_monster(
        &self,
        state: &mut GameState,
        actor: Player,
        promotion_square: Square,
        events: &mut Vec<Event>,
    ) -> bool {
        if !state.card_runtime.mutation[actor.index()] {
            return false;
        }
        let candidates = summon_candidates(self, state, actor, promotion_square);
        if candidates.is_empty() {
            return false;
        }
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        events.push(Event::RandomSample {
            purpose: "move.mutation.extra-monster".to_owned(),
            roll_53: roll,
        });
        let Some(index) = index else {
            return false;
        };
        let square = candidates[index];
        if let Some((anchor, victim)) = self.board_piece_entry_at(state, square) {
            if self.fracture_large_cell(state, victim, square) != Some(false) {
                self.remove_piece_for_card(state, anchor, victim, actor, ID, events);
            }
        }
        let monster = Piece::new(actor, PieceKind::Monster).with_id(state.allocate_piece_id());
        state.board.set(square, Some(monster));
        // A mutation monster is an allied monster: it belongs to the promoting
        // player, unlike the neutral rule monster tracked by `monster_rule`.
        self.register_spawned_piece(state, square, monster, true);
        events.push(Event::EffectApplied {
            source_card_instance: 0,
            command: format!("mutation-extra-monster:{}:{}", monster.id, square),
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, movement, position};
    use super::*;

    fn state_with_pawn() -> GameState {
        position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a7", Player::White, PieceKind::Pawn),
        ])
    }

    #[test]
    fn mutation_arms_the_pawn_promotion_table() {
        let engine = Engine::bundled_latest().unwrap();
        let base = state_with_pawn();
        let monster_promotion = Action::Move(MoveAction::new(
            "a7".parse().unwrap(),
            "a8".parse().unwrap(),
            Some(PieceKind::Monster),
        ));
        assert!(
            !engine
                .legal_actions(&base)
                .unwrap()
                .contains(&monster_promotion),
            "without the card the pawn cannot promote into a monster"
        );
        let mut armed = base.clone();
        acquire(&engine, &mut armed, Player::White, &mut Vec::new()).unwrap();
        assert!(armed.card_runtime.mutation[Player::White.index()]);
        assert!(
            engine
                .legal_actions(&armed)
                .unwrap()
                .contains(&monster_promotion),
            "the armed passive adds the monster promotion choice"
        );
        // The standard choices stay available.
        for kind in [
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
        ] {
            assert!(
                engine
                    .legal_actions(&armed)
                    .unwrap()
                    .contains(&Action::Move(MoveAction::new(
                        "a7".parse().unwrap(),
                        "a8".parse().unwrap(),
                        Some(kind)
                    )))
            );
        }
    }

    #[test]
    fn mutation_monster_promotion_keeps_the_pawn_owner() {
        let engine = Engine::bundled_latest().unwrap();
        let mut armed = state_with_pawn();
        acquire(&engine, &mut armed, Player::White, &mut Vec::new()).unwrap();
        let promoted = engine
            .apply(
                &armed,
                Action::Move(MoveAction::new(
                    "a7".parse().unwrap(),
                    "a8".parse().unwrap(),
                    Some(PieceKind::Monster),
                )),
            )
            .unwrap()
            .state;
        let monster = at(&promoted, "a8");
        assert_eq!(monster.owner, Player::White);
        assert_eq!(monster.kind, PieceKind::Monster);
        assert!(
            promoted.card_runtime.monster_rule.is_none(),
            "an allied promotion monster is not the neutral rule monster"
        );
    }

    #[test]
    fn mutation_extra_monster_uses_the_first_empty_ring_cell_and_is_deterministic() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("d8", Player::White, PieceKind::Monster)]);
        acquire(&engine, &mut state, Player::White, &mut Vec::new()).unwrap();
        let mut events = Vec::new();
        assert!(engine.summon_mutation_extra_monster(
            &mut state,
            Player::White,
            "d8".parse().unwrap(),
            &mut events,
        ));
        let spawned = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == Player::White && piece.kind == PieceKind::Monster)
            .collect::<Vec<_>>();
        assert_eq!(spawned.len(), 2, "the promoted monster plus the extra one");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::RandomSample { purpose, .. } if purpose == "move.mutation.extra-monster"
        )));
        // Same input, same destination.
        let mut again = position(&[("d8", Player::White, PieceKind::Monster)]);
        acquire(&engine, &mut again, Player::White, &mut Vec::new()).unwrap();
        engine.summon_mutation_extra_monster(
            &mut again,
            Player::White,
            "d8".parse().unwrap(),
            &mut Vec::new(),
        );
        assert_eq!(state, again);
    }

    #[test]
    fn mutation_extra_monster_prefers_an_enemy_when_the_ring_is_full() {
        let engine = Engine::bundled_latest().unwrap();
        // d4 has a complete eight-cell ring, unlike a back-rank promotion square.
        let mut state = position(&[("d4", Player::White, PieceKind::Monster)]);
        acquire(&engine, &mut state, Player::White, &mut Vec::new()).unwrap();
        let center: Square = "d4".parse().unwrap();
        // Fill every ring cell; exactly one is an enemy.
        let mut enemy_square = None;
        for (file_delta, rank_delta) in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ] {
            let cell = center.offset(file_delta, rank_delta).unwrap();
            let owner = if enemy_square.is_none() {
                enemy_square = Some(cell);
                Player::Black
            } else {
                Player::White
            };
            let id = state.allocate_piece_id();
            state
                .board
                .set(cell, Some(Piece::new(owner, PieceKind::Pawn).with_id(id)));
        }
        let enemy_square = enemy_square.unwrap();
        let enemy = state.board.get(enemy_square).unwrap();
        engine.summon_mutation_extra_monster(&mut state, Player::White, center, &mut Vec::new());
        assert!(
            state.board.find_by_id(enemy.id).is_none(),
            "a full ring captures the enemy neighbour instead of an ally"
        );
        assert_eq!(
            state.board.get(enemy_square).unwrap().kind,
            PieceKind::Monster
        );
        // Allied neighbours survive; no ally was sacrificed.
        assert_eq!(
            state
                .board
                .iter()
                .filter(|(square, piece)| {
                    *square != enemy_square
                        && *square != center
                        && piece.owner == Player::White
                        && piece.kind == PieceKind::Pawn
                })
                .count(),
            7
        );
    }

    #[test]
    fn mutation_is_ineligible_without_a_pawn_and_when_already_armed() {
        let engine = Engine::bundled_latest().unwrap();
        let no_pawn = position(&[]);
        assert!(!eligibility(&no_pawn, Player::White, AcquisitionContext::Draft).is_eligible());
        let mut armed = state_with_pawn();
        acquire(&engine, &mut armed, Player::White, &mut Vec::new()).unwrap();
        assert!(
            !eligibility(&armed, Player::White, AcquisitionContext::Draft).is_eligible(),
            "an already armed mutation cannot be acquired twice"
        );
        // The extra summon is a no-op while the passive is not armed.
        let mut unarmed = position(&[("d8", Player::White, PieceKind::Monster)]);
        assert!(!engine.summon_mutation_extra_monster(
            &mut unarmed,
            Player::White,
            "d8".parse().unwrap(),
            &mut Vec::new(),
        ));
        let _ = movement("a1", "a2");
    }
}

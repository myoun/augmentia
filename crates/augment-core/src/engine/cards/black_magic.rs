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
    state.card_runtime.allied_monster_ids.insert(monster.id);
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

impl Engine {
    /// Advance every allied Black Magic monster one random queen-step.
    ///
    /// Upstream collects the `blackMagicMonster` attributes and moves them when
    /// `monsterMoveDue` fires (main-RjrZ5fFP.js:210935-210960, cadence at 820-822),
    /// picking one of eight queen directions uniformly. An allied monster only
    /// captures enemies of ordinary kinds (main-RjrZ5fFP.js:4822-4833).
    pub(in crate::engine) fn settle_allied_monsters(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) {
        if state.card_runtime.allied_monster_ids.is_empty() {
            return;
        }
        // The shared monster cadence is one move every third half-move.
        let halfmove = rule_halfmove(state);
        if halfmove == 0
            || halfmove % 3 != 0
            || state.rule_runtime.monster_last_halfmove == halfmove
        {
            return;
        }
        state.rule_runtime.monster_last_halfmove = halfmove;

        // Move in stable piece-id order so the RNG stream is reproducible.
        let mut ids: Vec<u32> = state
            .card_runtime
            .allied_monster_ids
            .iter()
            .copied()
            .filter(|id| state.board.find_by_id(*id).is_some())
            .collect();
        ids.sort_unstable();
        for piece_id in ids {
            let Some((from, monster)) = state.board.find_by_id(piece_id) else {
                state.card_runtime.allied_monster_ids.remove(&piece_id);
                continue;
            };
            if state.card_runtime.frozen.contains_key(&piece_id)
                || state.rule_runtime.winter_frozen.contains(&piece_id)
                || state.card_runtime.poison_stun_turns.contains_key(&piece_id)
            {
                continue;
            }
            let mut candidates = Vec::new();
            for (df, dr) in [
                (-1, 1),
                (0, 1),
                (1, 1),
                (-1, 0),
                (1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
            ] {
                let Some(to) = from.offset(df, dr) else {
                    continue;
                };
                if self.collapsed_card_landing(state, to)
                    || state.card_runtime.barricades.contains(&to)
                    || state.card_runtime.black_holes.contains(&to)
                    || state.card_runtime.crown_pos == Some(to)
                    || self
                        .card_runtime_portal_cells(state)
                        .is_some_and(|cells| cells.contains(&to))
                    || state
                        .card_runtime
                        .pending_lobsters
                        .iter()
                        .any(|pending| pending.square == to)
                    || state
                        .card_runtime
                        .pending_scarecrows
                        .iter()
                        .any(|pending| pending.square == to)
                {
                    continue;
                }
                match self.board_piece_at(state, to) {
                    None => {
                        if self.card_landing_open(state, to) {
                            candidates.push(to);
                        }
                    }
                    Some(target) => {
                        // Allied monsters capture ordinary enemy pieces only.
                        let capturable = target.owner == monster.owner.opponent()
                            && !self.is_royal_piece(state, target)
                            && !matches!(
                                target.kind,
                                PieceKind::Monster
                                    | PieceKind::Scarecrow
                                    | PieceKind::Football
                                    | PieceKind::DarkWizard
                                    | PieceKind::Guard
                            )
                            && !state.card_runtime.shielded_piece_ids.contains(&target.id)
                            && !state.card_runtime.protected_piece_ids.contains(&target.id)
                            && !state
                                .card_runtime
                                .outpost_protected_piece_ids
                                .contains(&target.id);
                        if capturable {
                            candidates.push(to);
                        }
                    }
                }
            }
            if candidates.is_empty() {
                continue;
            }
            let index = rule_random(
                state,
                events,
                candidates.len(),
                &format!("black-magic-monster.{piece_id}"),
            )
            .unwrap_or(0);
            let to = candidates[index];
            if let Some((anchor, victim)) = self.board_piece_entry_at(state, to) {
                self.remove_piece_for_card(state, anchor, victim, monster.owner, ID, events);
            }
            state.board.set(from, None);
            state.board.set(to, Some(monster));
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("allied-monster:{piece_id}:{from}->{to}"),
            });
        }
    }

    fn card_runtime_portal_cells(&self, state: &GameState) -> Option<Vec<Square>> {
        let mut cells = Vec::new();
        if let Some(pair) = state.card_runtime.portals {
            cells.extend(pair);
        }
        if let Some(pair) = state.card_runtime.rule_portals {
            cells.extend(pair);
        }
        (!cells.is_empty()).then_some(cells)
    }
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

    /// An allied monster is registered as such and is therefore allowed to
    /// generate moves, unlike the inert neutral rule monster.
    #[test]
    fn an_allied_monster_is_marked_and_can_move() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        use_card(&engine, &mut state, Player::White).unwrap();
        let (square, monster) = spawned_monsters(&state, Player::White)[0];
        assert!(
            state.card_runtime.allied_monster_ids.contains(&monster.id),
            "the summoned monster must be registered as allied"
        );
        assert!(
            state.card_runtime.effects.bound_piece_ids.is_empty(),
            "an allied monster must not be bound"
        );
        // The monster is the summoner's piece, so only its owner can move it, and
        // only when it is that player's turn.
        state.turn = Player::White;
        let moves = engine.generate_piece_move_actions(&state, monster.id);
        assert!(
            !moves.is_empty(),
            "an allied monster must expose at least one move from {square}"
        );
    }

    /// The neutral rule monster stays immobile; only allied monsters move.
    #[test]
    fn the_neutral_rule_monster_stays_immobile() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        let monster = Piece::new(Player::White, PieceKind::Monster).with_id(900);
        state.board.set("d5".parse().unwrap(), Some(monster));
        state.card_runtime.monster_rule = Some(monster.id);
        state.turn = Player::White;
        assert!(
            engine
                .generate_piece_move_actions(&state, monster.id)
                .is_empty(),
            "the neutral rule monster must not have legal moves"
        );
    }

    /// The settlement advances an allied monster on the shared three-half-move
    /// cadence, and leaves it alone on the intervening turns.
    #[test]
    fn allied_monsters_advance_on_the_shared_monster_cadence() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        use_card(&engine, &mut state, Player::White).unwrap();
        let (before_square, monster) = spawned_monsters(&state, Player::White)[0];

        // The cadence is `rule_halfmove` non-zero and divisible by three. Halfmove
        // counts black's completed turns, so advance the counter directly instead
        // of playing out turns just to reach a due boundary.
        state.rule_runtime.monster_last_halfmove = 0;
        state.actions_remaining = 1;
        state.turns_taken[Player::Black.index()] = 3;
        state.turn = Player::White;
        assert_eq!(rule_halfmove(&state) % 3, 0);
        let mut events = Vec::new();
        engine.settle_allied_monsters(&mut state, &mut events);
        let after = state
            .board
            .find_by_id(monster.id)
            .map(|(square, _)| square)
            .expect("the monster must survive");
        assert_ne!(
            after, before_square,
            "the allied monster must step on a due half-move"
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, Event::EffectApplied { command, .. }
                    if command.starts_with("allied-monster:"))),
            "the move must be reported"
        );
    }
}

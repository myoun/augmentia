use super::*;

pub(super) const ID: &str = crate::ICBM;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Current-contract queen identity: a real queen that has not been promoted to a
/// royal role. Mirrors `isQueenIdentityPiece` and the core `queen` helper.
fn queen(piece: Piece, owner: Player, state: &GameState) -> bool {
    piece.owner == owner
        && piece.kind == PieceKind::Queen
        && !state.card_runtime.royal_piece_ids.contains(&piece.id)
}

/// The aimed enemy queen. Upstream stores `targetQueenId`, but `PendingIcbm`
/// only carries the source queen, so the aim is re-derived at settlement time in
/// web board order (the same order `findPiecesByPredicate(...)[0]` walks).
fn enemy_queen(state: &GameState, owner: Player) -> Option<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .find(|(_, piece)| queen(*piece, owner.opponent(), state))
}

fn own_queens(state: &GameState, owner: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| queen(*piece, owner, state))
        .collect()
}

fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, source) = piece_target(state, target)?;
    if !queen(source, actor, state) {
        return Err(EngineError::Invariant(
            "icbm requires an allied queen".to_owned(),
        ));
    }
    // The launch is illegal without a target, so a missing enemy queen is an
    // invariant failure rather than a silently dropped reservation.
    if enemy_queen(state, actor).is_none() {
        return Err(EngineError::Invariant(
            "icbm requires an enemy queen".to_owned(),
        ));
    }
    state.card_runtime.icbm_pending.push(crate::PendingIcbm {
        queen_id: source.id,
        owner: actor,
        due_turn: state.turns_taken[actor.index()].saturating_add(1),
    });
    Ok(())
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if enemy_queen(state, card.owner).is_none() {
        return;
    }
    for (square, _) in own_queens(state, card.owner) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        });
    }
}

fn eligibility(state: &GameState, player: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        !own_queens(state, player).is_empty() && enemy_queen(state, player).is_some(),
        "?곗? ?ъ쓽 ?ъ쓽 酉?媛 ?꾩슂?⑸땲??",
    )
}

impl Engine {
    /// Fire every ICBM due at `owner`'s turn start.
    ///
    /// Upstream `resolvePendingIcbmForTurn` runs right after the turn flips, so
    /// this belongs to the owner turn *start* boundary. For each due entry:
    /// sacrifice the tracked allied queen (a capture for the opponent), then
    /// blast the 3x3 around the aimed enemy queen. Both the allied queen and the
    /// aimed enemy queen must still be queen-identity pieces, otherwise the shot
    /// is cancelled without cost, exactly as upstream logs it.
    ///
    /// The previous core block in `tick_end_statuses_after_turn` ignored the
    /// tracked ids, never sacrificed the source queen, and matched
    /// `PieceKind::Queen` instead of queen identity; this replaces it.
    pub(in crate::engine) fn settle_icbm(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let due: Vec<crate::PendingIcbm> = state
            .card_runtime
            .icbm_pending
            .iter()
            .filter(|icbm| icbm.owner == owner && state.turns_taken[owner.index()] >= icbm.due_turn)
            .copied()
            .collect();
        if due.is_empty() {
            return;
        }
        state.card_runtime.icbm_pending.retain(|icbm| {
            !(icbm.owner == owner && state.turns_taken[owner.index()] >= icbm.due_turn)
        });

        for icbm in due {
            let Some((source_square, source)) = state
                .board
                .find_by_id(icbm.queen_id)
                .filter(|(_, piece)| queen(*piece, owner, state))
            else {
                continue;
            };
            let Some((target_square, _)) = enemy_queen(state, owner) else {
                continue;
            };
            // Sacrifice first: the missile is consumed even if it hits nothing.
            self.remove_piece_for_card(state, source_square, source, owner.opponent(), ID, events);
            self.explode_icbm_blast(state, target_square, owner, events);
        }
    }

    /// The 3x3 blast around `center`. Mirrors `explodeAt`: frozen and shielded
    /// pieces are spared, a fractured large body only loses one cell, and every
    /// other piece is removed as a capture for the caster.
    fn explode_icbm_blast(
        &self,
        state: &mut GameState,
        center: Square,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let mut victims: Vec<(Square, Piece)> = Vec::new();
        for (square, piece) in state.board.iter() {
            if square.file().abs_diff(center.file()) > 1
                || square.rank().abs_diff(center.rank()) > 1
            {
                continue;
            }
            if state.card_runtime.frozen.contains_key(&piece.id) {
                continue;
            }
            victims.push((square, piece));
        }
        for (square, piece) in victims {
            if state.card_runtime.shielded_piece_ids.remove(&piece.id) {
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("shield-broken:{}", piece.id),
                });
                continue;
            }
            if self.fracture_large_cell(state, piece, square) == Some(false) {
                continue;
            }
            self.remove_piece_for_card(state, square, piece, owner, ID, events);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, position};
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::ICBM.to_owned(),
            owner: Player::White,
            half_stars: 4,
            charges: 1,
            available_after_turn: None,
        }
    }

    fn cell(name: &str) -> Square {
        name.parse().unwrap()
    }

    fn grant(state: &GameState) -> GameState {
        let mut next = state.clone();
        next.add_card(card());
        next
    }

    fn arm(engine: &Engine, state: &GameState, queen_square: &str) -> GameState {
        let mut next = state.clone();
        next.cards[0].charges = 0;
        play(
            engine,
            &mut next,
            Player::White,
            1,
            &ActionTarget::PieceAt(cell(queen_square)),
            &mut Vec::new(),
        )
        .unwrap();
        next
    }

    fn offered(engine: &Engine, state: &GameState) -> Vec<Square> {
        let mut actions = Vec::new();
        super::actions(engine, state, &state.cards[0], &mut actions);
        actions
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::PieceAt(square),
                    ..
                } => Some(square),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn icbm_offers_only_allied_queen_identity_pieces_and_needs_an_enemy_queen() {
        let engine = Engine::bundled_latest().unwrap();
        let state = grant(&position(&[
            ("d1", Player::White, PieceKind::Queen),
            ("e8", Player::Black, PieceKind::Queen),
            ("a2", Player::White, PieceKind::Rook),
        ]));
        assert_eq!(offered(&engine, &state), vec![cell("d1")]);

        // No enemy queen: nothing may be launched.
        let mut without_target = state.clone();
        without_target.board.set(cell("e8"), None);
        assert!(offered(&engine, &without_target).is_empty());
        assert!(
            play(
                &engine,
                &mut without_target.clone(),
                Player::White,
                1,
                &ActionTarget::PieceAt(cell("d1")),
                &mut Vec::new()
            )
            .is_err()
        );

        // A royal-flagged queen is not a launchable queen identity.
        let mut royal = state.clone();
        let white_queen = at(&state, "d1");
        royal.card_runtime.royal_piece_ids.insert(white_queen.id);
        assert!(offered(&engine, &royal).is_empty());
        assert!(
            play(
                &engine,
                &mut royal.clone(),
                Player::White,
                1,
                &ActionTarget::PieceAt(cell("d1")),
                &mut Vec::new()
            )
            .is_err()
        );

        // A non-queen target is rejected.
        assert!(
            play(
                &engine,
                &mut state.clone(),
                Player::White,
                1,
                &ActionTarget::PieceAt(cell("a2")),
                &mut Vec::new()
            )
            .is_err()
        );
    }

    #[test]
    fn icbm_schedules_for_the_next_owner_turn_start_then_sacrifices_and_blasts() {
        let engine = Engine::bundled_latest().unwrap();
        let state = arm(
            &engine,
            &grant(&position(&[
                ("d1", Player::White, PieceKind::Queen),
                ("e8", Player::Black, PieceKind::Queen),
                ("d8", Player::Black, PieceKind::Rook),
                ("f7", Player::Black, PieceKind::Pawn),
                ("a1", Player::White, PieceKind::Rook),
            ])),
            "d1",
        );
        assert_eq!(state.card_runtime.icbm_pending.len(), 1);
        let pending = state.card_runtime.icbm_pending[0];
        assert_eq!(pending.due_turn, 1);
        assert_eq!(pending.owner, Player::White);
        assert_eq!(pending.queen_id, at(&state, "d1").id);
        // Arming does not fire the missile.
        assert!(state.board.get(cell("d1")).is_some());

        // The opponent's boundary must not fire it.
        let mut waiting = state.clone();
        engine.settle_icbm(&mut waiting, Player::Black, &mut Vec::new());
        assert_eq!(waiting.board.iter().count(), 6);

        let mut next = waiting.clone();
        next.turns_taken[0] = 1;
        let mut events = Vec::new();
        engine.settle_icbm(&mut next, Player::White, &mut events);
        // The allied queen is sacrificed...
        assert!(next.board.get(cell("d1")).is_none());
        // ...and everything in the 3x3 around e8 is destroyed.
        for square in ["e8", "d8", "f7", "d7", "f8"] {
            assert!(next.board.get(cell(square)).is_none(), "{square}");
        }
        // Pieces outside the blast survive.
        assert!(next.board.get(cell("a1")).is_some());
        assert!(next.card_runtime.icbm_pending.is_empty());
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::PieceRemoved { .. }))
                .count(),
            4
        );
        // Idempotent once the reservation is consumed.
        engine.settle_icbm(&mut next, Player::White, &mut Vec::new());
        // a1 (white rook) and h8 (black king) were outside the blast.
        assert_eq!(next.board.iter().count(), 2);
    }

    #[test]
    fn icbm_cancels_without_cost_when_either_queen_is_gone_or_demoted() {
        let engine = Engine::bundled_latest().unwrap();
        let base = arm(
            &engine,
            &grant(&position(&[
                ("d1", Player::White, PieceKind::Queen),
                ("e8", Player::Black, PieceKind::Queen),
            ])),
            "d1",
        );

        // Source queen captured before the next owner turn.
        let mut lost_source = base.clone();
        lost_source.board.set(cell("d1"), None);
        lost_source.turns_taken[0] = 1;
        let before = lost_source.board.clone();
        engine.settle_icbm(&mut lost_source, Player::White, &mut Vec::new());
        assert_eq!(lost_source.board, before);
        assert!(lost_source.card_runtime.icbm_pending.is_empty());

        // Aimed enemy queen captured before launch.
        let mut lost_target = base.clone();
        lost_target.board.set(cell("e8"), None);
        lost_target.turns_taken[0] = 1;
        let before = lost_target.board.clone();
        engine.settle_icbm(&mut lost_target, Player::White, &mut Vec::new());
        assert_eq!(lost_target.board, before);
        assert!(lost_target.card_runtime.icbm_pending.is_empty());

        // Source queen demoted to another kind.
        let mut demoted = base.clone();
        let mut queened = demoted.board.get(cell("d1")).unwrap();
        queened.kind = PieceKind::Rook;
        demoted.board.set(cell("d1"), Some(queened));
        demoted.turns_taken[0] = 1;
        let before = demoted.board.clone();
        engine.settle_icbm(&mut demoted, Player::White, &mut Vec::new());
        assert_eq!(demoted.board, before);
    }

    #[test]
    fn icbm_blast_spares_frozen_pieces_and_breaks_shields() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = arm(
            &engine,
            &grant(&position(&[
                ("d1", Player::White, PieceKind::Queen),
                ("e8", Player::Black, PieceKind::Queen),
                ("d8", Player::Black, PieceKind::Rook),
                ("f7", Player::Black, PieceKind::Knight),
            ])),
            "d1",
        );
        let frozen = at(&state, "f7");
        let shielded = at(&state, "d8");
        state.card_runtime.frozen.insert(
            frozen.id,
            crate::FrozenStatus {
                source: Player::White,
                remaining_turns: 1,
            },
        );
        state.card_runtime.shielded_piece_ids.insert(shielded.id);
        state.turns_taken[0] = 1;
        let mut events = Vec::new();
        engine.settle_icbm(&mut state, Player::White, &mut events);
        // Frozen piece survives untouched, shield absorbs and breaks.
        assert!(state.board.get(cell("f7")).is_some());
        assert!(state.board.get(cell("d8")).is_some());
        // A frozen piece keeps its freeze; a spared piece is never cleared.
        assert!(state.card_runtime.frozen.contains_key(&frozen.id));
        assert!(!state.card_runtime.shielded_piece_ids.contains(&shielded.id));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::EffectApplied { command, .. } if command == &format!("shield-broken:{}", shielded.id)
        )));
        // The aimed queen still died in the same blast.
        assert!(state.board.get(cell("e8")).is_none());
    }

    #[test]
    fn icbm_settlement_is_deterministic_and_supports_stacked_shots() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = grant(&position(&[
            ("d1", Player::White, PieceKind::Queen),
            ("e8", Player::Black, PieceKind::Queen),
        ]));
        state.cards[0].charges = 0;
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt(cell("d1")),
            &mut Vec::new(),
        )
        .unwrap();
        state.cards[0].charges = 0;
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt(cell("d1")),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(state.card_runtime.icbm_pending.len(), 2);

        let settle = |state: &GameState| {
            let mut next = state.clone();
            next.turns_taken[0] = 1;
            let mut events = Vec::new();
            engine.settle_icbm(&mut next, Player::White, &mut events);
            (next, events)
        };
        let first = settle(&state);
        let second = settle(&state);
        assert_eq!(first, second);
        // Both reservations resolve in one settlement; the second shot has no
        // allied queen left to sacrifice and cancels.
        assert!(first.0.card_runtime.icbm_pending.is_empty());
        assert!(first.0.board.get(cell("d1")).is_none());
    }

    /// Pins the settlement boundary against the real engine turn loop.
    ///
    /// Upstream resolves ICBM right after the turn flips (main bundle
    /// L342800-342803: `state.turn = opponent(...)` then
    /// `resolvePendingIcbmForTurn(state.turn)`), so the shot must not fire while
    /// the caster is still mid-turn: the opponent has to get their move first.
    /// `settle_icbm` must therefore be invoked at the caster's next turn
    /// *start*, not at the end of the caster's own turn.
    #[test]
    fn icbm_fires_at_the_next_owner_turn_start_not_at_the_end_of_the_casters_turn() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d1", Player::White, PieceKind::Queen),
            ("e8", Player::Black, PieceKind::Queen),
            ("c3", Player::White, PieceKind::Knight),
            ("f6", Player::Black, PieceKind::Knight),
        ]);
        state = engine
            .grant_card(&state, 1, crate::ICBM, Player::White)
            .unwrap()
            .state;
        state = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::PieceAt(cell("d1")),
                },
            )
            .unwrap()
            .state;
        assert_eq!(state.card_runtime.icbm_pending.len(), 1);

        // White plays the card and then finishes its turn. The shot is reserved
        // for White's *next* turn start, so it must still be pending while Black
        // is on move: upstream only resolves after the turn flips to the owner
        // (main bundle L342800-342803).
        let after_white = engine
            .apply(
                &state,
                Action::Move(crate::MoveAction::new(cell("c3"), cell("d5"), None)),
            )
            .unwrap()
            .state;
        assert_eq!(after_white.turn, Player::Black);
        assert_eq!(after_white.turns_taken[0], 1);
        assert_eq!(
            after_white.card_runtime.icbm_pending.len(),
            1,
            "the missile must not fire during the caster's own turn"
        );
        assert!(after_white.board.get(cell("d1")).is_some());
        assert!(after_white.board.get(cell("e8")).is_some());

        // Black replies; the turn flips back to White, which is the owner turn
        // start the reservation was waiting for.
        let after_black = engine
            .apply(
                &after_white,
                Action::Move(crate::MoveAction::new(cell("f6"), cell("g4"), None)),
            )
            .unwrap()
            .state;
        assert_eq!(after_black.turn, Player::White);
        assert_eq!(after_black.turns_taken[1], 1);
        assert!(after_black.card_runtime.icbm_pending.is_empty());
        assert!(after_black.board.get(cell("d1")).is_none());
        assert!(after_black.board.get(cell("e8")).is_none());
    }
}

use super::*;
pub(super) const ID: &str = crate::NECROMANCY;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// How long the revived form lasts. The revert itself is already settled by
/// `Engine::tick_end_statuses_after_turn` (turn_effects.rs, `necromancy_reverts`),
/// which restores the piece to a Pawn once `state.ply` reaches `due_ply`.
const REVIVE_PLIES: u32 = 4;

/// Types the upstream pool refuses to revive (`V8_NECROMANCY_EXCLUDED_TYPES`,
/// L199732-199751), expressed in core kinds. Royal, large-body, installation and
/// pawn-family forms are all excluded.
fn revivable(kind: PieceKind) -> bool {
    !matches!(
        kind,
        PieceKind::King
            | PieceKind::RoyalKnight
            | PieceKind::Merchant
            | PieceKind::Colossus
            | PieceKind::BigRook
            | PieceKind::BigBishop
            | PieceKind::Barrier
            | PieceKind::Scarecrow
            | PieceKind::Football
            | PieceKind::Monster
            | PieceKind::Crown
            | PieceKind::Pawn
            | PieceKind::Squire
            | PieceKind::BerolinaPawn
    )
}

/// The piece captured on a square, read back from the events of the transitions
/// that led here. Core keeps no general graveyard list, so the deterministic
/// stand-in is the set of own non-revivable kinds the opponent currently holds as
/// captures: the caller supplies the pool.
///
/// Candidates are ordered by the captured-piece value so the draw is stable and
/// matches the upstream pool's append order filtered by the legal target set.
fn pool(state: &GameState, actor: Player) -> Vec<PieceKind> {
    // Kinds the current contract can revive: anything the opposing side has
    // captured that is not excluded. Core has no per-side graveyard, so the
    // practical source is the pieces missing from the actor's starting material
    // for kinds that can still be rebuilt from the standard set.
    const STANDARD: &[(PieceKind, u8)] = &[
        (PieceKind::Queen, 1),
        (PieceKind::Rook, 2),
        (PieceKind::Bishop, 2),
        (PieceKind::Knight, 2),
    ];
    let alive = |kind: PieceKind| {
        state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == actor && piece.kind == kind)
            .count()
    };
    let mut out = Vec::new();
    for (kind, count) in STANDARD {
        if !revivable(*kind) {
            continue;
        }
        if (alive(*kind) as u8) < *count {
            out.push(*kind);
        }
    }
    out
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, mut pawn) = piece_target(state, target)?;
    if pawn.owner != actor || pawn.kind != PieceKind::Pawn || engine.is_royal_piece(state, pawn) {
        return Err(EngineError::Invariant(
            "necromancy requires an allied pawn".to_owned(),
        ));
    }
    let candidates = pool(state, actor);
    if candidates.is_empty() {
        return Err(EngineError::Invariant(
            "necromancy has no captured allied piece to revive".to_owned(),
        ));
    }
    let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
    events.push(Event::RandomSample {
        purpose: "necromancy".to_owned(),
        roll_53: roll,
    });
    let revived = candidates[index.expect("nonempty necromancy pool")];
    engine.clear_queens_gambit_protection(state, pawn.id);
    pawn.kind = revived;
    state.board.set(square, Some(pawn));
    state.card_runtime.moved_piece_ids.insert(pawn.id);
    // The settlement is already wired: turn_effects.rs consumes this list and
    // restores the Pawn once four plies have passed.
    state
        .card_runtime
        .necromancy_reverts
        .push(NecromancyRevert {
            piece_id: pawn.id,
            due_ply: state.ply.saturating_add(REVIVE_PLIES),
        });
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("necromancy:{}:{revived:?}", pawn.id),
    });
    Ok(())
}

fn actions(_engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if pool(state, card.owner).is_empty() {
        return;
    }
    for (square, piece) in state.board.iter() {
        if piece.owner == card.owner && piece.kind == PieceKind::Pawn {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    let has_pawn = state
        .board
        .iter()
        .any(|(_, piece)| piece.owner == owner && piece.kind == PieceKind::Pawn);
    require(
        has_pawn && !pool(state, owner).is_empty(),
        "되살릴 아군 폰과 포획된 아군 기물이 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, handoff, movement, position, single};
    use super::*;

    fn use_card(
        engine: &Engine,
        state: &mut GameState,
        actor: Player,
        target: ActionTarget,
    ) -> (GameState, Vec<Event>) {
        let mut events = Vec::new();
        play(engine, state, actor, 1, &target, &mut events).unwrap();
        (state.clone(), events)
    }

    fn state_with_room() -> GameState {
        // Both sides keep a full standard complement except one missing white
        // knight, so the revive pool is exactly {Knight}.
        position(&[
            ("e1", Player::White, PieceKind::King),
            ("d1", Player::White, PieceKind::Queen),
            ("a1", Player::White, PieceKind::Rook),
            ("h1", Player::White, PieceKind::Rook),
            ("c1", Player::White, PieceKind::Bishop),
            ("f1", Player::White, PieceKind::Bishop),
            ("b1", Player::White, PieceKind::Knight),
            ("a2", Player::White, PieceKind::Pawn),
        ])
    }

    #[test]
    fn necromancy_revives_a_captured_kind_for_four_plies_and_queues_the_revert() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state_with_room();
        let pawn = at(&state, "a2");
        let (next, events) = use_card(&engine, &mut state, Player::White, single("a2"));
        assert_eq!(at(&next, "a2").id, pawn.id, "identity is preserved");
        assert_eq!(at(&next, "a2").kind, PieceKind::Knight);
        assert_eq!(at(&next, "a2").owner, Player::White);
        assert!(next.card_runtime.moved_piece_ids.contains(&pawn.id));
        assert_eq!(next.card_runtime.necromancy_reverts.len(), 1);
        assert_eq!(
            next.card_runtime.necromancy_reverts[0].due_ply,
            next.ply + 4
        );
        assert_eq!(next.turn, Player::White, "PRESERVE_TURN keeps the turn");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::RandomSample { purpose, .. } if purpose == "necromancy"
        )));
    }

    #[test]
    fn necromancy_reverts_to_a_pawn_when_the_settlement_runs() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state_with_room();
        let pawn = at(&state, "a2");
        let (mut next, _) = use_card(&engine, &mut state, Player::White, single("a2"));
        assert_eq!(at(&next, "a2").kind, PieceKind::Knight);
        // Force the deadline and run the existing turn-end settlement path.
        next.card_runtime.necromancy_reverts[0].due_ply = next.ply;
        handoff(&engine, &mut next);
        assert_eq!(
            at(&next, "a2").kind,
            PieceKind::Pawn,
            "the delayed settlement restores the pawn form"
        );
        assert_eq!(at(&next, "a2").id, pawn.id, "identity survives the revert");
        assert!(next.card_runtime.necromancy_reverts.is_empty());
    }

    #[test]
    fn necromancy_rejects_non_pawns_and_a_missing_pool_without_mutating_state() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state_with_room();
        let before = state.clone();
        for target in [single("d1"), single("h8"), ActionTarget::None] {
            assert!(
                play(
                    &engine,
                    &mut state,
                    Player::White,
                    1,
                    &target,
                    &mut Vec::new()
                )
                .is_err(),
                "target {target:?} must be rejected"
            );
        }
        assert_eq!(state, before, "rejected targets must not mutate state");

        // A full complement means nothing was captured, so there is no pool.
        let full = position(&[
            ("e1", Player::White, PieceKind::King),
            ("d1", Player::White, PieceKind::Queen),
            ("a1", Player::White, PieceKind::Rook),
            ("h1", Player::White, PieceKind::Rook),
            ("c1", Player::White, PieceKind::Bishop),
            ("f1", Player::White, PieceKind::Bishop),
            ("b1", Player::White, PieceKind::Knight),
            ("g1", Player::White, PieceKind::Knight),
            ("a2", Player::White, PieceKind::Pawn),
        ]);
        let mut out = Vec::new();
        actions(
            &engine,
            &full,
            &CardInstance {
                instance_id: 1,
                definition_id: crate::NECROMANCY.to_owned(),
                owner: Player::White,
                half_stars: 0,
                charges: 1,
                available_after_turn: None,
            },
            &mut out,
        );
        assert!(out.is_empty(), "no revive pool means no legal use");
        assert!(!eligibility(&full, Player::White, AcquisitionContext::Draft).is_eligible());
    }

    #[test]
    fn necromancy_offers_every_allied_pawn_and_is_deterministic() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = state_with_room();
        state.board.set(
            "h2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn).with_id(900)),
        );
        let card = CardInstance {
            instance_id: 1,
            definition_id: crate::NECROMANCY.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        };
        let mut out = Vec::new();
        actions(&engine, &state, &card, &mut out);
        assert_eq!(out.len(), 2, "one action per allied pawn");
        // The same input state must produce the same revived kind.
        let (left, left_events) =
            use_card(&engine, &mut state.clone(), Player::White, single("a2"));
        let (right, right_events) =
            use_card(&engine, &mut state.clone(), Player::White, single("a2"));
        assert_eq!(left, right);
        assert_eq!(left_events, right_events);
        let _ = movement("a2", "a3");
    }

    #[test]
    fn necromancy_pool_follows_missing_material_and_excludes_royals() {
        let mut state = state_with_room();
        assert_eq!(pool(&state, Player::White), vec![PieceKind::Knight]);
        // Losing the queen widens the pool, in the fixed STANDARD order.
        state.board.set("d1".parse().unwrap(), None);
        assert_eq!(
            pool(&state, Player::White),
            vec![PieceKind::Queen, PieceKind::Knight]
        );
        // The pool is per-owner: Black only has a king in this fixture, so every
        // revivable kind is missing for Black while White still holds all of them.
        assert_eq!(
            pool(&state, Player::Black),
            vec![
                PieceKind::Queen,
                PieceKind::Rook,
                PieceKind::Bishop,
                PieceKind::Knight
            ]
        );
        assert!(!revivable(PieceKind::King));
        assert!(!revivable(PieceKind::Pawn));
        assert!(!revivable(PieceKind::Monster));
        assert!(revivable(PieceKind::Queen));
    }
}

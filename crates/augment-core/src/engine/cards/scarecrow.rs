use super::*;
pub(super) const ID: &str = crate::SCARECROW;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Whether a piece may be sacrificed to place the scarecrow.
///
/// The bundle refuses a royal sacrifice for this snapshot: it guards on `usesNovember01Cards`, which
/// is true here because `REFORM_CATALOG_HASH` is this catalog's hash, and then rejects the king and
/// editor-royal targets. Without that guard a player could delete their own king.
fn sacrifice_allowed(engine: &Engine, state: &GameState, owner: Player, square: Square) -> bool {
    let Some(piece) = state.board.get(square) else {
        return false;
    };
    piece.owner == owner
        && !matches!(piece.kind, PieceKind::Football | PieceKind::King)
        && !engine.d4_restricts(state, owner, false, square)
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, victim) = piece_target(state, target)?;
    // A king is never a legal sacrifice; `actions` filters it, and this rejects a hand-built action.
    if !sacrifice_allowed(engine, state, actor, square) {
        return Err(EngineError::Invariant(
            "허수아비의 제물로는 왕이나 축구공을 쓸 수 없습니다.".into(),
        ));
    }
    engine.delete_piece_for_card(state, square, victim, actor, ID, events);
    let scarecrow = Piece::new(actor, PieceKind::Scarecrow).with_id(state.allocate_piece_id());
    state.board.set(square, Some(scarecrow));
    state.card_runtime.moved_piece_ids.insert(scarecrow.id);
    state
        .card_runtime
        .piece_origins
        .insert(scarecrow.id, square);
    // The waiting scarecrow is reserved, not protected. The bundle marks it `scarecrowReserved`
    // (which blocks capture) and only adds `protected` under the October 7 balance, which does not
    // apply to this snapshot. Reserving it here would make it invulnerable for a reason the source
    // does not give.
    state.card_runtime.scarecrow_reservations.insert(
        scarecrow.id,
        (actor, state.turns_taken[actor.index()].saturating_add(3)),
    );
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, _) in web_ordered_pieces(state) {
        if sacrifice_allowed(engine, state, card.owner, square) {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    static ENGINE: std::sync::OnceLock<Engine> = std::sync::OnceLock::new();
    let engine = ENGINE.get_or_init(|| Engine::bundled_latest().expect("bundled latest ruleset"));
    require(
        (0..64).filter_map(Square::from_index).any(|s| {
            engine.card_landing_open(state, s) && !engine.d4_restricts(state, owner, false, s)
        }),
        "허수아비를 설치할 빈칸이 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, position};
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::SCARECROW.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        }
    }

    fn offered(state: &GameState) -> Vec<Action> {
        let engine = Engine::bundled_latest().unwrap();
        let mut out = Vec::new();
        actions(&engine, state, &card(), &mut out);
        out
    }

    fn targets(state: &GameState) -> Vec<String> {
        offered(state)
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::PieceAt(square),
                    ..
                } => Some(square.to_string()),
                _ => None,
            })
            .collect()
    }

    /// The king must never be offered as a sacrifice.
    ///
    /// The bundle refuses a royal target for this snapshot, guarding on `usesNovember01Cards`, which
    /// holds here because the catalog hash equals `REFORM_CATALOG_HASH`. Without the guard a player
    /// could delete their own king and lose deliberately.
    #[test]
    fn the_king_is_not_offered_as_a_sacrifice() {
        let state = position(&[("b2", Player::White, PieceKind::Rook)]);
        let offered = targets(&state);
        assert!(
            !offered.contains(&"a1".to_owned()),
            "the white king must not be a target: {offered:?}"
        );
        assert!(
            offered.contains(&"b2".to_owned()),
            "an ordinary piece must be a target: {offered:?}"
        );
    }

    /// A hand-built action targeting the king is rejected, not just hidden from the enumeration.
    #[test]
    fn playing_on_the_king_is_rejected() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("b2", Player::White, PieceKind::Rook)]);
        let before = state.clone();
        let result = play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt("a1".parse().unwrap()),
            &mut Vec::new(),
        );
        assert!(result.is_err(), "sacrificing the king must fail");
        assert_eq!(state, before, "a rejected play must not mutate state");
    }

    /// The waiting scarecrow is reserved, not protected.
    ///
    /// The bundle marks it `scarecrowReserved`, which blocks capture, and only adds `protected` under
    /// the October 7 balance, which does not apply here. Marking it protected would make it
    /// invulnerable for a reason the source does not give.
    #[test]
    fn the_waiting_scarecrow_is_reserved_but_not_protected() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("b2", Player::White, PieceKind::Rook)]);
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt("b2".parse().unwrap()),
            &mut Vec::new(),
        )
        .expect("the sacrifice is legal");

        let scarecrow = at(&state, "b2");
        assert_eq!(scarecrow.kind, PieceKind::Scarecrow);
        assert!(
            state
                .card_runtime
                .scarecrow_reservations
                .contains_key(&scarecrow.id),
            "the scarecrow must be reserved while it waits"
        );
        assert!(
            !state
                .card_runtime
                .protected_piece_ids
                .contains(&scarecrow.id),
            "the waiting scarecrow must not be protected in this snapshot"
        );
    }

    /// The reservation lasts three owner turns, matching `remainingOwnTurns: 3`.
    #[test]
    fn the_reservation_lasts_three_owner_turns() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("b2", Player::White, PieceKind::Rook)]);
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt("b2".parse().unwrap()),
            &mut Vec::new(),
        )
        .expect("the sacrifice is legal");

        let scarecrow = at(&state, "b2");
        let (owner, due) = state.card_runtime.scarecrow_reservations[&scarecrow.id];
        assert_eq!(owner, Player::White);
        assert_eq!(due, state.turns_taken[Player::White.index()] + 3);
    }

    /// A football is not a legal sacrifice, matching the pre-existing exclusion.
    #[test]
    fn a_football_is_not_offered() {
        let state = position(&[
            ("b2", Player::White, PieceKind::Football),
            ("c2", Player::White, PieceKind::Rook),
        ]);
        let offered = targets(&state);
        assert!(!offered.contains(&"b2".to_owned()), "{offered:?}");
        assert!(offered.contains(&"c2".to_owned()), "{offered:?}");
    }
}

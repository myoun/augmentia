use super::*;
pub(super) const ID: &str = crate::CANCELING;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Home square of the enemy castling king: e1 for white, e8 for black.
fn home_square(enemy: Player) -> Square {
    Square::new(4, if enemy == Player::White { 0 } else { 7 }).expect("king home square")
}

/// Source `applyCancelingV4` (main-RjrZ5fFP.js L198522-198566): the card only
/// affects a royal that still carries the `castled` flag and is away from home.
/// `v15KingAugmentRole` (L199811) is the royal predicate used there; under the
/// September 18 balance policy that includes merchants, so this mirrors
/// `Engine::is_royal_piece` rather than the merchant-excluding `royal` helper.
fn castled_royal(engine: &Engine, state: &GameState, enemy: Player) -> Option<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .find(|(square, piece)| {
            piece.owner == enemy
                && engine.is_royal_piece(state, *piece)
                && state.card_runtime.castled_piece_ids.contains(&piece.id)
                && *square != home_square(enemy)
        })
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let enemy = actor.opponent();
    let (from, king) = castled_royal(engine, state, enemy).ok_or_else(|| {
        EngineError::Invariant("canceling requires a castled enemy royal away from home".into())
    })?;
    let destination = home_square(enemy);
    // L198529-198543: the occupant of the home square is cleared straight off the
    // board (`position.board[..] = null` with no capture entry), so it is removed
    // without capture credit or capture reactions.
    if let Some(occupant) = state.board.get(destination) {
        engine.delete_piece_for_card(state, destination, occupant, actor, ID, events);
    }
    // L198553-198559: on a collapsed home square the returned king dies instead
    // of landing (v9RemovePieceIds), which the missing-king check turns into defeat.
    if engine.collapsed_card_landing(state, destination) {
        engine.delete_piece_for_card(state, from, king, actor, ID, events);
        return Ok(());
    }
    state.board.set(from, None);
    state.board.set(destination, Some(king));
    state.card_runtime.moved_piece_ids.insert(king.id);
    // L198552: `attributes.castled = false`.
    state.card_runtime.castled_piece_ids.remove(&king.id);
    // L198546-198548: the enemy loses both castling rights and the en passant target.
    state.en_passant = None;
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("canceling:{}:{}->{}", king.id, from, destination),
    });
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if castled_royal(engine, state, state.turn.opponent()).is_some() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

fn eligibility(state: &GameState, owner: Player, context: AcquisitionContext) -> CardEligibility {
    // `isCardDraftDrawableForColor` (L314463-314467): canceling is a future-target
    // card, so it is always drawable in the draft and the target is re-checked on use.
    if context == AcquisitionContext::Draft {
        return CardEligibility::Eligible;
    }
    require(
        web_ordered_pieces(state)
            .into_iter()
            .any(|(square, piece)| {
                piece.owner == owner.opponent()
                    && (royal(state, piece) || piece.kind == PieceKind::Merchant)
                    && state.card_runtime.castled_piece_ids.contains(&piece.id)
                    && square != home_square(owner.opponent())
            }),
        "캐슬링해서 원래 자리를 비운 상대 왕실 기물이 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::target_procedures::*;
    use super::*;

    fn castled_move(state: &mut GameState, king_square: &str) {
        let king = state.board.get(king_square.parse().unwrap()).unwrap();
        let target = "g8".parse().unwrap();
        state.board.set(king_square.parse().unwrap(), None);
        state.board.set(target, Some(king));
        state.card_runtime.moved_piece_ids.insert(king.id);
        state.card_runtime.castled_piece_ids.insert(king.id);
    }

    #[test]
    fn canceling_returns_castled_enemy_king_and_clears_its_home_square_occupant() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("g8", Player::Black, PieceKind::King),
                ("e8", Player::Black, PieceKind::Rook),
            ],
            Player::White,
        );
        castled_move(&mut state, "g8");
        let king = state.board.get("g8".parse().unwrap()).unwrap();
        let occupant = state.board.get("e8".parse().unwrap()).unwrap();
        let mut events = Vec::new();
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::None,
            &mut events,
        )
        .unwrap();
        assert_eq!(state.board.get("g8".parse().unwrap()), None);
        assert_eq!(
            state
                .board
                .get("e8".parse().unwrap())
                .map(|p| (p.id, p.kind)),
            Some((king.id, PieceKind::King))
        );
        // The occupant is cleared off the board without capture credit or reactions.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Event::PieceCaptured { .. }))
        );
        assert!(events.iter().any(|e| matches!(
            e,
            Event::PieceRemoved { piece, cause, .. } if piece.id == occupant.id && cause == ID
        )));
        assert!(!state.card_runtime.castled_piece_ids.contains(&king.id));
        assert!(state.card_runtime.moved_piece_ids.contains(&king.id));
        assert_eq!(state.en_passant, None);
    }

    #[test]
    fn canceling_enumerates_only_a_castled_royal_away_from_home() {
        let engine = Engine::bundled_latest().unwrap();
        let card = CardInstance {
            instance_id: 1,
            definition_id: ID.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        };
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("g8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        let mut out = Vec::new();
        actions(&engine, &state, &card, &mut out);
        assert!(
            out.is_empty(),
            "an uncastled king is not a canceling target"
        );
        assert!(!eligibility(&state, Player::White, AcquisitionContext::Setup).is_eligible());
        assert!(eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible());

        castled_move(&mut state, "g8");
        out.clear();
        actions(&engine, &state, &card, &mut out);
        assert_eq!(
            out,
            vec![Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None
            }]
        );
        assert!(eligibility(&state, Player::White, AcquisitionContext::Setup).is_eligible());

        // A king that is back home is no longer a target even with the flag set.
        let king = state.board.get("g8".parse().unwrap()).unwrap();
        state.board.set("g8".parse().unwrap(), None);
        state.board.set("e8".parse().unwrap(), Some(king));
        out.clear();
        actions(&engine, &state, &card, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn canceling_rejects_without_a_castled_king_and_preserves_state() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("g8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        let before = state.clone();
        let mut events = Vec::new();
        assert!(
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &ActionTarget::None,
                &mut events
            )
            .is_err()
        );
        assert_eq!(state, before);
        assert!(events.is_empty());
    }

    #[test]
    fn canceling_kills_the_returning_king_on_a_collapsed_home_square() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("g8", Player::Black, PieceKind::King),
            ],
            Player::White,
        );
        castled_move(&mut state, "g8");
        state.card_runtime.collapsed_files_ranks = true;
        let mut events = Vec::new();
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::None,
            &mut events,
        )
        .unwrap();
        assert_eq!(state.board.get("e8".parse().unwrap()), None);
        assert_eq!(state.board.get("g8".parse().unwrap()), None);
        assert!(events.iter().any(|e| matches!(
            e,
            Event::PieceRemoved { cause, .. } if cause == ID
        )));
    }
}

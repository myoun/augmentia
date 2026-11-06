use super::*;
pub(super) const ID: &str = crate::EXILE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Enemy pieces that can be sent home. Upstream `exileSelectionContract`: the
/// piece needs a recorded origin that is empty and different from its current
/// square, and neither royals nor slime-shaped pieces can be relocated.
fn candidates(engine: &Engine, state: &GameState, actor: Player) -> Vec<(Square, Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter_map(|(square, piece)| {
            let origin = state.card_runtime.piece_origins.get(&piece.id).copied()?;
            (piece.owner == actor.opponent()
                && !engine.is_royal_piece(state, piece)
                && !engine.piece_has_ability(state, piece, PieceKind::Slime)
                && origin != square
                && state.board.get(origin).is_none())
            .then_some((square, origin, piece))
        })
        .collect()
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, piece) = piece_target(state, target)?;
    let Some(origin) = state.card_runtime.piece_origins.get(&piece.id).copied() else {
        return Err(EngineError::Invariant(
            "exile target has no origin".to_owned(),
        ));
    };
    if piece.owner != actor.opponent()
        || engine.is_royal_piece(state, piece)
        || engine.piece_has_ability(state, piece, PieceKind::Slime)
        || origin == square
        || state.board.get(origin).is_some()
    {
        return Err(EngineError::Invariant("invalid exile target".to_owned()));
    }
    state.board.set(square, None);
    state.board.set(origin, Some(piece));
    if let Some(en_passant) = state.en_passant
        && (en_passant.target == square || en_passant.target == origin)
    {
        state.en_passant = None;
    }
    // Reform exile grants protection until the caster's turn ends. The timed
    // protection map is exactly that window: entries keyed by `by` tick down at
    // that player's turn end (`tick_middle_statuses_after_turn`) and disappear
    // at zero, and the map is consulted by capture generation and resolution.
    state.card_runtime.vigilance_protection.insert(
        piece.id,
        TimedPieceStatus {
            by: actor,
            remaining_turns: 1,
        },
    );
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("exile:{}:{}->{}", piece.id, square, origin),
    });
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, _, _) in candidates(engine, state, card.owner) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        });
    }
}

fn eligibility(state: &GameState, owner: Player, context: AcquisitionContext) -> CardEligibility {
    if context == AcquisitionContext::Draft {
        return CardEligibility::Eligible;
    }
    let available = state.board.iter().any(|(square, piece)| {
        state
            .card_runtime
            .piece_origins
            .get(&piece.id)
            .is_some_and(|origin| {
                piece.owner == owner.opponent()
                    && !royal(state, piece)
                    && !matches!(piece.kind, PieceKind::Slime | PieceKind::Merchant)
                    && *origin != square
                    && state.board.get(*origin).is_none()
            })
    });
    require(available, "유배할 상대 기물이 필요합니다.")
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    #[test]
    fn exile_returns_the_enemy_piece_home_and_protects_it_until_the_caster_turn_ends() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("d4", Player::Black, PieceKind::Knight),
        ]);
        let knight = at(&s, "d4");
        s.card_runtime
            .piece_origins
            .insert(knight.id, "d7".parse().unwrap());
        use_at(&mut s, &engine, crate::EXILE, single("d4"));
        assert!(s.board.get("d4".parse().unwrap()).is_none());
        assert_eq!(at(&s, "d7").id, knight.id);
        assert_eq!(
            s.card_runtime
                .vigilance_protection
                .get(&knight.id)
                .map(|status| (status.by, status.remaining_turns)),
            Some((Player::White, 1))
        );
        // The protection expires exactly at the caster's own turn end.
        engine.tick_middle_statuses_after_turn(&mut s, Player::White, &mut Vec::new());
        assert!(!s.card_runtime.vigilance_protection.contains_key(&knight.id));
    }

    #[test]
    fn exile_is_unavailable_when_the_origin_is_occupied_or_the_piece_is_home() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("d4", Player::Black, PieceKind::Knight),
            ("d7", Player::Black, PieceKind::Pawn),
            ("h6", Player::Black, PieceKind::Bishop),
        ]);
        let knight = at(&s, "d4");
        s.card_runtime
            .piece_origins
            .insert(knight.id, "d7".parse().unwrap());
        let bishop = at(&s, "h6");
        s.card_runtime
            .piece_origins
            .insert(bishop.id, "h6".parse().unwrap());
        s = engine
            .grant_card(&s, 1, crate::EXILE, Player::White)
            .unwrap()
            .state;
        let targets: Vec<ActionTarget> = engine
            .legal_actions(&s)
            .unwrap()
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    target: target @ ActionTarget::PieceAt(_),
                    ..
                } => Some(target),
                _ => None,
            })
            .collect();
        assert!(targets.is_empty());
        let before = s.clone();
        assert!(
            engine
                .apply(
                    &s,
                    Action::UseCard {
                        instance_id: 1,
                        target: single("d4"),
                    },
                )
                .is_err()
        );
        assert_eq!(s, before);
    }
}

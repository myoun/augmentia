use super::*;
pub(super) const ID: &str = crate::TRAITOR;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Upstream `v8TraitorPieces` for the current balance: only opposing pieces that
/// have already moved are eligible, and wall/football/colossus bodies never are.
fn moved_enemies(state: &GameState) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| {
            piece.owner == state.turn.opponent()
                && state.card_runtime.moved_piece_ids.contains(&piece.id)
                && !matches!(
                    piece.kind,
                    PieceKind::Football | PieceKind::Colossus | PieceKind::Monster
                )
        })
        .collect()
}

/// Non-royal candidates that can be converted, weakest first by the same
/// material table the other END cards use.
fn weakest(state: &GameState) -> Vec<(Square, Piece)> {
    let enemies: Vec<_> = moved_enemies(state)
        .into_iter()
        .filter(|(_, piece)| !royal(state, *piece))
        .collect();
    let pawns: Vec<_> = enemies
        .iter()
        .copied()
        .filter(|(_, piece)| piece.kind == PieceKind::Pawn)
        .collect();
    if !pawns.is_empty() {
        return pawns;
    }
    let minimum = enemies
        .iter()
        .map(|(_, piece)| super::revelation::combat_value(piece.kind))
        .min()
        .unwrap_or(0);
    enemies
        .into_iter()
        .filter(|(_, piece)| super::revelation::combat_value(piece.kind) == minimum)
        .collect()
}

fn royal_target(state: &GameState) -> Option<(Square, Piece)> {
    moved_enemies(state)
        .into_iter()
        .find(|(_, piece)| royal(state, *piece))
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let pool = weakest(state);
    if pool.is_empty() {
        let Some((square, piece)) = royal_target(state) else {
            return Err(EngineError::Invariant("no traitor target".to_owned()));
        };
        convert_royal(engine, state, actor, square, piece, events);
        return Ok(());
    }
    let index = if pool.len() == 1 {
        0
    } else {
        let (roll, index) = state.rng_mut().web_choice_index(pool.len());
        events.push(Event::RandomSample {
            purpose: "traitor".to_owned(),
            roll_53: roll,
        });
        index.unwrap_or(0)
    };
    let (square, mut piece) = pool[index];
    piece.owner = actor;
    state.board.set(square, Some(piece));
    // Conversions do not count as movement in the current profile, so the
    // defector keeps its capture-ready status and identity.
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("traitor:{}:{}", piece.id, square),
    });
    Ok(())
}

/// `v19ConvertRoyal`: a defecting royal keeps its identity but changes owner. A
/// base king becomes a queen, and the defector's former side loses the game the
/// instant the royal changes hands, unless democracy keeps that side alive.
fn convert_royal(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    square: Square,
    piece: Piece,
    events: &mut Vec<Event>,
) {
    let former = piece.owner;
    let mut converted = piece;
    converted.owner = actor;
    if converted.kind == PieceKind::King {
        converted.kind = PieceKind::Queen;
    }
    state.board.set(square, Some(converted));
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("traitor-royal:{}:{}", converted.id, square),
    });
    if !engine.is_royal_piece(state, converted) {
        state.card_runtime.royal_piece_ids.insert(converted.id);
    }
    if !state.card_runtime.democracy[former.index()] {
        state.status = GameStatus::Won {
            winner: actor,
            reason: WinReason::CardEffect(ID.to_owned()),
        };
    }
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if !weakest(state).is_empty() || royal_target(state).is_some() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

fn eligibility(state: &GameState, _: Player, context: AcquisitionContext) -> CardEligibility {
    if context == AcquisitionContext::Draft {
        return CardEligibility::Eligible;
    }
    require(
        !weakest(state).is_empty() || royal_target(state).is_some(),
        "전향시킬 상대 기물이 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    #[test]
    fn traitor_converts_the_weakest_moved_enemy_piece() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("d4", Player::Black, PieceKind::Pawn),
            ("h6", Player::Black, PieceKind::Rook),
            ("b2", Player::Black, PieceKind::Pawn),
        ]);
        let moved_pawn = at(&s, "d4");
        let unmoved_pawn = at(&s, "b2");
        let rook = at(&s, "h6");
        s.card_runtime.moved_piece_ids.insert(moved_pawn.id);
        s.card_runtime.moved_piece_ids.insert(rook.id);
        use_at(&mut s, &engine, crate::TRAITOR, ActionTarget::None);
        // Only moved pieces are candidates, and pawns outrank the rook.
        assert_eq!(at(&s, "d4").owner, Player::White);
        assert_eq!(at(&s, "d4").id, moved_pawn.id);
        assert_eq!(at(&s, "d4").kind, PieceKind::Pawn);
        assert_eq!(at(&s, "h6").owner, Player::Black);
        assert_eq!(at(&s, "b2").owner, Player::Black);
        assert_eq!(at(&s, "b2").id, unmoved_pawn.id);
        // A defected pawn keeps its identity and may capture immediately.
        assert!(has_move(&engine, &s, "d4", "e5") || has_move(&engine, &s, "d4", "e4"));
    }

    #[test]
    fn traitor_falls_back_to_the_moved_enemy_royal_and_ends_the_game() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("h8", Player::Black, PieceKind::King)]);
        let king = at(&s, "h8");
        s.card_runtime.moved_piece_ids.insert(king.id);
        use_at(&mut s, &engine, crate::TRAITOR, ActionTarget::None);
        let converted = at(&s, "h8");
        assert_eq!(converted.owner, Player::White);
        assert_eq!(converted.id, king.id);
        assert_eq!(converted.kind, PieceKind::Queen);
        assert_eq!(
            s.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect(crate::TRAITOR.to_owned()),
            }
        );
    }

    #[test]
    fn traitor_is_unavailable_and_state_stable_without_moved_enemy_pieces() {
        let engine = Engine::bundled_latest().unwrap();
        let s = position(&[("d4", Player::Black, PieceKind::Pawn)]);
        let s = engine
            .grant_card(&s, 1, crate::TRAITOR, Player::White)
            .unwrap()
            .state;
        assert!(
            !engine
                .legal_actions(&s)
                .unwrap()
                .iter()
                .any(|action| matches!(action, Action::UseCard { .. }))
        );
        let before = s.clone();
        assert!(
            engine
                .apply(
                    &s,
                    Action::UseCard {
                        instance_id: 1,
                        target: ActionTarget::None,
                    },
                )
                .is_err()
        );
        assert_eq!(s, before);
    }
}

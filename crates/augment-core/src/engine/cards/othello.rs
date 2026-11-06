use super::*;
pub(super) const ID: &str = crate::OTHELLO;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

const QUEEN_DIRECTIONS: [(i8, i8); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

/// The othello card arms the owner's *next* turn end. The card instance is
/// already spent (`charges == 0`) when `play` runs, so its `available_after_turn`
/// slot is free to carry the owner's completed-turn index, exactly like collapse.
fn armed(card: &CardInstance) -> bool {
    card.charges == 0 && card.available_after_turn.is_some()
}

fn armed_for(state: &GameState, owner: Player) -> bool {
    state
        .cards
        .iter()
        .any(|card| card.definition_id == ID && card.owner == owner && armed(card))
}

fn convertible(piece: Piece) -> bool {
    !matches!(
        piece.kind,
        PieceKind::Football | PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop
    )
}

fn is_bracketed(state: &GameState, square: Square, actor: Player) -> bool {
    QUEEN_DIRECTIONS.iter().any(|(df, dr)| {
        let friendly = |file: i8, rank: i8| {
            square
                .offset(file, rank)
                .and_then(|near| state.board.get(near))
                .is_some_and(|near| near.owner == actor)
        };
        friendly(*df, *dr) && friendly(-*df, -*dr)
    })
}

/// Upstream `v19ResolveOthelloAll` candidates in web square order: every enemy
/// piece (except the immovable bodies) bracketed by two friendly pieces.
fn candidates(state: &GameState, actor: Player) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(square, piece)| {
            piece.owner == actor.opponent()
                && convertible(*piece)
                && is_bracketed(state, *square, actor)
        })
        .collect()
}

fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    instance_id: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    if candidates(state, actor).is_empty() {
        return Err(EngineError::Invariant(
            "othello has no bracketed enemy piece".to_owned(),
        ));
    }
    if armed_for(state, actor) {
        return Err(EngineError::Invariant(
            "othello is already scheduled for this owner".to_owned(),
        ));
    }
    let card = state
        .cards
        .iter_mut()
        .find(|card| card.instance_id == instance_id && card.owner == actor)
        .ok_or_else(|| EngineError::Invariant("othello card instance is missing".to_owned()))?;
    card.available_after_turn = Some(state.turns_taken[actor.index()]);
    Ok(())
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if !armed_for(state, card.owner) && !candidates(state, card.owner).is_empty() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

fn eligibility(state: &GameState, owner: Player, context: AcquisitionContext) -> CardEligibility {
    if context == AcquisitionContext::Draft {
        return CardEligibility::Eligible;
    }
    // Draft eligibility is decided by the catalog draw requirement; the concrete
    // bracketing is evaluated when the card is used or settled.
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == owner.opponent() && convertible(piece)),
        "전향시킬 상대 기물이 필요합니다.",
    )
}

impl Engine {
    /// Turn-end settlement for othello: convert every enemy piece bracketed by
    /// two friendly pieces, then disarm the owner's pending card.
    ///
    /// The Lead must call this from the current-profile turn end (the same place
    /// `settle_delayed_turn_end` runs), after the owner's own moves and before
    /// the next player takes over.
    pub(in crate::engine) fn settle_othello(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID || state.status != GameStatus::Ongoing {
            return;
        }
        let due: Vec<u32> = state
            .cards
            .iter()
            .filter(|card| {
                card.definition_id == ID
                    && card.owner == actor
                    && armed(card)
                    && card
                        .available_after_turn
                        .is_some_and(|due| state.turns_taken[actor.index()] >= due)
            })
            .map(|card| card.instance_id)
            .collect();
        if due.is_empty() {
            return;
        }
        for instance_id in due {
            if let Some(card) = state
                .cards
                .iter_mut()
                .find(|card| card.instance_id == instance_id)
            {
                card.available_after_turn = None;
            }
        }
        self.resolve_othello_all(state, actor, events);
    }
    fn resolve_othello_all(&self, state: &mut GameState, actor: Player, events: &mut Vec<Event>) {
        for (square, mut piece) in candidates(state, actor) {
            if self.is_royal_piece(state, piece) || piece.kind == PieceKind::Vip {
                // `v19ConvertRoyal`: the defecting royal keeps its identity; base
                // kings become queens and the former side loses unless democracy
                // keeps it alive.
                let former = piece.owner;
                piece.owner = actor;
                if piece.kind == PieceKind::King {
                    piece.kind = PieceKind::Queen;
                }
                state.board.set(square, Some(piece));
                if !self.is_royal_piece(state, piece) {
                    state.card_runtime.royal_piece_ids.insert(piece.id);
                }
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("othello-royal:{}:{}", piece.id, square),
                });
                if !state.card_runtime.democracy[former.index()] {
                    state.status = GameStatus::Won {
                        winner: actor,
                        reason: WinReason::CardEffect(ID.to_owned()),
                    };
                    return;
                }
                continue;
            }
            piece.owner = actor;
            state.board.set(square, Some(piece));
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("othello:{}:{}", piece.id, square),
            });
        }
        self.resolve_missing_king(state, events);
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    #[test]
    fn othello_converts_every_bracketed_enemy_piece_at_the_owner_turn_end() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("c4", Player::White, PieceKind::Rook),
            ("d4", Player::Black, PieceKind::Knight),
            ("e4", Player::White, PieceKind::Rook),
            ("d5", Player::Black, PieceKind::Pawn),
            ("c5", Player::White, PieceKind::Bishop),
            ("e5", Player::White, PieceKind::Bishop),
        ]);
        use_at(&mut s, &engine, crate::OTHELLO, ActionTarget::None);
        // Nothing converts until the owner's turn actually ends.
        assert_eq!(at(&s, "d4").owner, Player::Black);
        assert!(
            s.cards
                .iter()
                .any(|card| card.definition_id == crate::OTHELLO && armed(card))
        );
        // The real turn-end lifecycle (settle_delayed_turn_end) is what fires it.
        handoff(&engine, &mut s);
        assert_eq!(at(&s, "d4").owner, Player::White);
        // Vertically bracketed too: c5/d5/e5.
        assert_eq!(at(&s, "d5").owner, Player::White);
        assert!(
            !s.cards
                .iter()
                .any(|card| card.definition_id == crate::OTHELLO && armed(card))
        );
    }

    #[test]
    fn othello_ignores_pieces_that_are_not_between_two_friendly_pieces() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("c4", Player::White, PieceKind::Rook),
            ("d4", Player::Black, PieceKind::Knight),
            ("e4", Player::White, PieceKind::Rook),
            ("d6", Player::Black, PieceKind::Rook),
        ]);
        use_at(&mut s, &engine, crate::OTHELLO, ActionTarget::None);
        engine.settle_othello(&mut s, Player::White, &mut Vec::new());
        // Bracketed d4 converts, the unbracketed d6 rook stays.
        assert_eq!(at(&s, "d4").owner, Player::White);
        assert_eq!(at(&s, "d6").owner, Player::Black);

        // A position with no bracketed enemy piece cannot play the card at all.
        let base = position(&[
            ("d4", Player::Black, PieceKind::Knight),
            ("d5", Player::White, PieceKind::Rook),
        ]);
        let none = engine
            .grant_card(&base, 2, crate::OTHELLO, Player::White)
            .unwrap()
            .state;
        assert!(
            !engine
                .legal_actions(&none)
                .unwrap()
                .iter()
                .any(|action| matches!(action, Action::UseCard { .. }))
        );
        assert!(
            engine
                .apply(
                    &none,
                    Action::UseCard {
                        instance_id: 2,
                        target: ActionTarget::None,
                    },
                )
                .is_err()
        );
        assert!(none.card_runtime.vip_pending.is_empty());
    }

    #[test]
    fn othello_converting_the_enemy_royal_ends_the_game() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("d8", Player::White, PieceKind::Rook),
            ("e8", Player::Black, PieceKind::King),
            ("f8", Player::White, PieceKind::Rook),
        ]);
        use_at(&mut s, &engine, crate::OTHELLO, ActionTarget::None);
        engine.settle_othello(&mut s, Player::White, &mut Vec::new());
        let converted = at(&s, "e8");
        assert_eq!(converted.owner, Player::White);
        assert_eq!(converted.kind, PieceKind::Queen);
        assert_eq!(
            s.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect(crate::OTHELLO.to_owned()),
            }
        );
    }
}

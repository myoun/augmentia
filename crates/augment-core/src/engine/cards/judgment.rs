use super::*;
pub(super) const ID: &str = crate::JUDGMENT;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Upstream `isLargeFootprintPiece`: wall-sized bodies are not judged.
fn large(state: &GameState, piece: Piece) -> bool {
    matches!(
        piece.kind,
        PieceKind::BigRook | PieceKind::BigBishop | PieceKind::Colossus
    ) || state.card_runtime.piece_status.contains_key(&piece.id)
}

/// Every judged piece carrying the highest capture count, if that count is at
/// least the upstream threshold of two captures.
fn candidates(state: &GameState) -> Vec<(Square, Piece)> {
    let judged: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| piece.capture_count >= 2 && !large(state, *piece))
        .collect();
    let Some(maximum) = judged.iter().map(|(_, piece)| piece.capture_count).max() else {
        return Vec::new();
    };
    judged
        .into_iter()
        .filter(|(_, piece)| piece.capture_count == maximum)
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
    if !candidates(state)
        .iter()
        .any(|(candidate_square, candidate)| {
            *candidate_square == square && candidate.id == piece.id
        })
    {
        return Err(EngineError::Invariant("invalid judgment target".to_owned()));
    }
    // The current contract removes the judged piece from the game instead of
    // exiling it until the next draft (0915 behavior).
    engine.delete_piece_for_card(state, square, piece, actor, ID, events);
    if engine.is_royal_piece(state, piece) {
        // `targetWasKing`: losing the judged royal is an environmental defeat.
        state.status = GameStatus::Won {
            winner: if piece.owner == actor {
                actor.opponent()
            } else {
                actor
            },
            reason: WinReason::CardEffect(ID.to_owned()),
        };
    }
    Ok(())
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, _) in candidates(state) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        });
    }
}

fn eligibility(state: &GameState, _: Player, context: AcquisitionContext) -> CardEligibility {
    if context == AcquisitionContext::Draft {
        return CardEligibility::Eligible;
    }
    require(
        !candidates(state).is_empty(),
        "2개 이상 잡은 기물이 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    fn judged(s: &mut GameState, square: &str, captures: u16) {
        let mut piece = at(s, square);
        piece.capture_count = captures;
        s.board.set(square.parse().unwrap(), Some(piece));
    }

    #[test]
    fn judgment_targets_every_piece_tied_for_the_highest_capture_count() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("b8", Player::Black, PieceKind::Rook),
            ("g1", Player::White, PieceKind::Bishop),
        ]);
        judged(&mut s, "b1", 3);
        judged(&mut s, "b8", 3);
        judged(&mut s, "g1", 2);
        s = engine
            .grant_card(&s, 1, crate::JUDGMENT, Player::White)
            .unwrap()
            .state;
        let targets: Vec<Square> = engine
            .legal_actions(&s)
            .unwrap()
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::PieceAt(square),
                } => Some(square),
                _ => None,
            })
            .collect();
        assert_eq!(targets, vec!["b1".parse().unwrap(), "b8".parse().unwrap()]);
        let after = engine
            .apply(
                &s,
                Action::UseCard {
                    instance_id: 1,
                    target: single("b8"),
                },
            )
            .unwrap()
            .state;
        assert!(after.board.get("b8".parse().unwrap()).is_none());
        assert!(after.board.get("b1".parse().unwrap()).is_some());
        assert_eq!(after.status, GameStatus::Ongoing);
    }

    #[test]
    fn judgment_is_unavailable_below_two_captures_and_rejects_large_bodies() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("b8", Player::Black, PieceKind::Rook),
            ("d4", Player::Black, PieceKind::BigRook),
        ]);
        judged(&mut s, "b1", 2);
        judged(&mut s, "b8", 1);
        judged(&mut s, "d4", 5);
        s = engine
            .grant_card(&s, 1, crate::JUDGMENT, Player::White)
            .unwrap()
            .state;
        let targets: Vec<Square> = engine
            .legal_actions(&s)
            .unwrap()
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::PieceAt(square),
                    ..
                } => Some(square),
                _ => None,
            })
            .collect();
        assert_eq!(targets, vec!["b1".parse().unwrap()]);
    }

    #[test]
    fn judging_a_royal_is_an_environmental_defeat() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("h8", Player::Black, PieceKind::King)]);
        judged(&mut s, "h8", 2);
        s = engine
            .grant_card(&s, 1, crate::JUDGMENT, Player::White)
            .unwrap()
            .state;
        let after = engine
            .apply(
                &s,
                Action::UseCard {
                    instance_id: 1,
                    target: single("h8"),
                },
            )
            .unwrap()
            .state;
        assert!(after.board.get("h8".parse().unwrap()).is_none());
        assert_eq!(
            after.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect(crate::JUDGMENT.to_owned()),
            }
        );
    }

    /// A lethal large-body hit is a real capture, so it must raise the attacker's
    /// capture tally. Upstream counts every capture in `applyV3MoveHistory`
    /// (main-RjrZ5fFP.js:226376-226391), not only captures that leave the attacker
    /// standing on the victim's square, so a lethal hit on an HP body counts too.
    #[test]
    fn a_lethal_large_body_hit_counts_as_a_capture() {
        let engine = Engine::bundled_latest().unwrap();
        // A Colossus is an HP body that reaches the damage path under the current
        // ruleset; big rooks and big bishops resolve earlier and never get here.
        let mut s = position(&[
            ("e1", Player::White, PieceKind::King),
            ("a4", Player::White, PieceKind::Rook),
            ("d4", Player::Black, PieceKind::Colossus),
        ]);
        let defender = at(&s, "d4");
        s.card_runtime.insert_large_body(
            defender.id,
            "d4".parse().unwrap(),
            PieceHealth::damaged(1, 3),
        );
        let attacker = at(&s, "a4");
        assert_eq!(attacker.capture_count, 0);

        let after = engine.apply(&s, movement("a4", "d4")).unwrap().state;
        assert!(
            after.board.find_by_id(defender.id).is_none(),
            "the lethal hit must remove the large body"
        );
        let attacker = after
            .board
            .find_by_id(attacker.id)
            .map(|(_, piece)| piece)
            .expect("the attacking rook must survive");
        assert_eq!(
            attacker.capture_count, 1,
            "a lethal large-body hit must be tallied on the attacker"
        );
    }

    /// Judgment reads that tally directly, so the same attacker becomes a legal
    /// target once it reaches the two-capture threshold.
    #[test]
    fn judgment_reads_the_tally_from_a_lethal_large_body_hit() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("e1", Player::White, PieceKind::King),
            ("a4", Player::White, PieceKind::Rook),
            ("d4", Player::Black, PieceKind::Colossus),
        ]);
        let defender = at(&s, "d4");
        s.card_runtime.insert_large_body(
            defender.id,
            "d4".parse().unwrap(),
            PieceHealth::damaged(1, 3),
        );
        let attacker = at(&s, "a4");
        let mut after = engine.apply(&s, movement("a4", "d4")).unwrap().state;
        // Judgment needs two captures; the hit supplied one, so add a second to
        // reach the threshold and confirm the card offers this exact attacker.
        if let Some((square, mut piece)) = after.board.find_by_id(attacker.id) {
            piece.capture_count = 2;
            after.board.set(square, Some(piece));
        }
        assert!(
            candidates(&after)
                .iter()
                .any(|(_, piece)| piece.id == attacker.id),
            "a piece that captured via a lethal large-body hit must be judgeable"
        );
    }
}

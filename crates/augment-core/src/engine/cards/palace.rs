use super::*;
pub(super) const ID: &str = crate::PALACE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn enemy_royal(engine: &Engine, state: &GameState, actor: Player) -> Option<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .find(|(_, piece)| piece.owner == actor.opponent() && engine.is_royal_piece(state, *piece))
}

/// 3x3 cells around `center`, clipped to the board, expressed as the inclusive
/// min/max square pair persisted in `card_runtime.palaces`.
pub(super) fn palace_bounds(center: Square) -> [Square; 2] {
    let min_file = center.file().saturating_sub(1);
    let max_file = u8::min(7, center.file().saturating_add(1));
    let min_rank = center.rank().saturating_sub(1);
    let max_rank = u8::min(7, center.rank().saturating_add(1));
    [
        Square::new(min_file, min_rank).expect("clipped palace corner"),
        Square::new(max_file, max_rank).expect("clipped palace corner"),
    ]
}
pub(super) fn within(state: &GameState, owner: Player, square: Square) -> bool {
    state
        .card_runtime
        .palaces
        .get(&owner)
        .is_some_and(|bounds| {
            square.file() >= bounds[0].file()
                && square.file() <= bounds[1].file()
                && square.rank() >= bounds[0].rank()
                && square.rank() <= bounds[1].rank()
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
    let Some((square, king)) = enemy_royal(engine, state, actor) else {
        return Err(EngineError::Invariant(
            "palace requires an enemy royal".to_owned(),
        ));
    };
    let bounds = palace_bounds(square);
    state.card_runtime.palaces.insert(king.owner, bounds);
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("palace:{}:{}:{}", king.id, bounds[0], bounds[1]),
    });
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if enemy_royal(engine, state, card.owner).is_some() {
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
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == owner.opponent() && royal(state, piece)),
        "상대 왕실 기물이 필요합니다.",
    )
}

impl Engine {
    /// Current-profile palace rule: the enemy royal that was inside the palace
    /// when it was designated cannot leave it by moving (`isWorkerPalaceMoveBlocked`).
    /// The caller must invoke this from the move filter for the piece's own moves.
    pub(in crate::engine) fn palace_blocks_move(
        &self,
        state: &GameState,
        piece: Piece,
        from: Square,
        to: Square,
    ) -> bool {
        if state.ruleset_id != crate::LATEST_RULESET_ID
            || !state.card_runtime.palaces.contains_key(&piece.owner)
            || !self.is_royal_piece(state, piece)
            || !within(state, piece.owner, from)
        {
            return false;
        }
        !within(state, piece.owner, to)
    }

    /// `v9ResolvePalaces`: drop a palace once its royal is gone or has stepped
    /// outside the designated cells. Call once per turn boundary.
    pub(in crate::engine) fn settle_palaces(&self, state: &mut GameState) {
        if state.ruleset_id != crate::LATEST_RULESET_ID || state.card_runtime.palaces.is_empty() {
            return;
        }
        let stale: Vec<Player> = state
            .card_runtime
            .palaces
            .keys()
            .copied()
            .filter(|owner| {
                web_ordered_pieces(state)
                    .into_iter()
                    .find_map(|(square, piece)| {
                        (piece.owner == *owner && self.is_royal_piece(state, piece))
                            .then_some(within(state, *owner, square))
                    })
                    != Some(true)
            })
            .collect();
        for owner in stale {
            state.card_runtime.palaces.remove(&owner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    #[test]
    fn palace_marks_the_clipped_3x3_around_the_enemy_royal() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("h8", Player::Black, PieceKind::King)]);
        assert!(s.card_runtime.palaces.is_empty());
        use_at(&mut s, &engine, crate::PALACE, ActionTarget::None);
        let bounds = s.card_runtime.palaces.get(&Player::Black).copied().unwrap();
        assert_eq!(bounds[0], "g7".parse().unwrap());
        assert_eq!(bounds[1], "h8".parse().unwrap());
        // The royal is pinned inside; everything else may still cross.
        let king = at(&s, "h8");
        assert!(engine.palace_blocks_move(&s, king, "h8".parse().unwrap(), "h6".parse().unwrap()));
        assert!(engine.palace_blocks_move(&s, king, "g7".parse().unwrap(), "f6".parse().unwrap()));
        assert!(!engine.palace_blocks_move(&s, king, "h8".parse().unwrap(), "h7".parse().unwrap()));
        let mut pawn = Piece::new(Player::Black, PieceKind::Pawn);
        pawn.id = 999;
        assert!(!engine.palace_blocks_move(&s, pawn, "g7".parse().unwrap(), "g6".parse().unwrap()));

        // The rule is reachable through real legal-move generation: the enemy
        // king cannot step out of the palace, but in-palace steps stay available.
        s.turn = Player::Black;
        assert!(!has_move(&engine, &s, "h8", "h6"));
        assert!(!has_move(&engine, &s, "h8", "g6"));
        assert!(has_move(&engine, &s, "h8", "h7") || has_move(&engine, &s, "h8", "g7"));
    }

    #[test]
    fn palace_settlement_drops_the_designation_when_its_royal_is_gone() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("h8", Player::Black, PieceKind::King)]);
        use_at(&mut s, &engine, crate::PALACE, ActionTarget::None);
        assert_eq!(s.card_runtime.palaces.len(), 1);
        engine.settle_palaces(&mut s);
        assert_eq!(s.card_runtime.palaces.len(), 1);
        s.board.set("h8".parse().unwrap(), None);
        engine.settle_palaces(&mut s);
        assert!(s.card_runtime.palaces.is_empty());
    }
}

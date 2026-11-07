use super::*;
pub(super) const ID: &str = crate::INITIATIVE;
pub(super) fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let target = owner.opponent();
    state.card_runtime.initiative[target.index()] = Some(InitiativeStatus {
        by: owner,
        start_turn: state.turns_taken[target.index()],
        limit: if state.ruleset_id == crate::LATEST_RULESET_ID {
            8
        } else {
            10
        },
    });
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    // OPPONENT_PIECE: upstream requires at least one opposing piece (bundle 55815-55816).
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == player.opponent()),
        "상대 기물이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn settle_initiative_checks(&self, state: &mut GameState) {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            return;
        }
        for target in [Player::White, Player::Black] {
            let Some(status) = state.card_runtime.initiative[target.index()] else {
                continue;
            };
            let king_squares: Vec<_> = state
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == target && self.is_royal_piece(state, *piece))
                .map(|(square, _)| square)
                .collect();
            let mut probe = state.clone();
            // Disable castling in an attack query to avoid recursively checking
            // castling safety. Keep the real target identity and its protections.
            probe.card_runtime.moved_piece_ids.extend(
                state
                    .board
                    .iter()
                    .filter(|(_, piece)| self.is_royal_piece(state, *piece))
                    .map(|(_, piece)| piece.id),
            );
            let checked = probe.board.iter().filter(|(_, piece)| piece.owner == status.by).any(|(from, piece)| {
                let mut actions = Vec::new();
                self.generate_moves_for_piece(&probe, from, piece, &mut actions);
                actions.iter().any(|action| matches!(action, Action::Move(movement) if king_squares.contains(&movement.to)))
            });
            if checked {
                state.card_runtime.initiative[target.index()] = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::interactions::*;
    use super::*;
    #[test]
    fn initiative_lasts_eight_owner_turns_and_releases_on_capture_or_real_check() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::Black, PieceKind::Rook),
                ("d2", Player::White, PieceKind::Pawn),
            ],
            Player::Black,
        );
        let state = engine
            .grant_card(&state, 1, INITIATIVE, Player::White)
            .unwrap()
            .state;
        assert_eq!(state.card_runtime.initiative[1].unwrap().limit, 8);
        assert!(!has_move(&engine, &state, "d4", "d2"));
        let mut expired = state.clone();
        expired.turns_taken[1] = 8;
        assert!(has_move(&engine, &expired, "d4", "d2"));
        let checking = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("g2", Player::White, PieceKind::Rook),
            ],
            Player::White,
        );
        let checking = engine
            .grant_card(&checking, 1, INITIATIVE, Player::White)
            .unwrap()
            .state;
        assert!(checking.card_runtime.initiative[1].is_some());
        let checked = engine.apply(&checking, movement("g2", "h2")).unwrap().state;
        assert!(checked.card_runtime.initiative[1].is_none());
        let mut protected = checking.clone();
        let king = protected.board.get("h8".parse().unwrap()).unwrap();
        protected.card_runtime.shielded_piece_ids.insert(king.id);
        // A consumable shield is a reaction and does not block the direct attack query.
        let checked = engine
            .apply(&protected, movement("g2", "h2"))
            .unwrap()
            .state;
        assert!(checked.card_runtime.initiative[1].is_none());
        let mut frozen = checking;
        frozen.card_runtime.frozen.insert(
            king.id,
            FrozenStatus {
                source: Player::White,
                remaining_turns: 3,
            },
        );
        let frozen = engine.apply(&frozen, movement("g2", "h2")).unwrap().state;
        assert!(frozen.card_runtime.initiative[1].is_some());
    }
}

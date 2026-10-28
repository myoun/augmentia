use super::*;
pub(super) const ID: &str = crate::HIGHLANDER;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    _owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    engine.resolve_highlander_victory(state);
    Ok(())
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn resolve_highlander_victory(&self, state: &mut GameState) {
        if state.status != GameStatus::Ongoing {
            return;
        }
        for owner in [Player::White, Player::Black] {
            if !state.cards.iter().any(|card| {
                card.owner == owner
                    && card.definition_id == crate::HIGHLANDER
                    && card.available_after_turn.is_none()
            }) {
                continue;
            }
            let mut types = BTreeSet::new();
            let eligible = state
                .board
                .iter()
                .filter(|(_, piece)| {
                    piece.owner == owner
                        && (state.ruleset_id != crate::LATEST_RULESET_ID
                            || piece.kind != PieceKind::Monster)
                })
                .all(|(_, piece)| types.insert(piece.kind));
            let latest_eligible = state.ruleset_id != crate::LATEST_RULESET_ID
                || (types.len() >= 4 && types.contains(&PieceKind::King));
            if eligible && !types.is_empty() && latest_eligible {
                state.status = GameStatus::Won {
                    winner: owner,
                    reason: WinReason::CardEffect(crate::HIGHLANDER.to_owned()),
                };
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::interactions::*;
    use super::*;
    #[test]
    fn latest_highlander_requires_four_distinct_types_and_a_literal_king() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = base();
        for (square, kind) in [("b2", PieceKind::Knight), ("c2", PieceKind::Bishop)] {
            state.board.set(
                square.parse().unwrap(),
                Some(Piece::new(Player::White, kind).with_id(10 + square.as_bytes()[0] as u32)),
            );
        }
        let state = engine
            .grant_card(&state, 1, crate::HIGHLANDER, Player::White)
            .unwrap()
            .state;
        assert_eq!(state.status, GameStatus::Ongoing);
        let mut four = state.clone();
        four.board.set(
            "d2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook).with_id(200)),
        );
        engine.resolve_highlander_victory(&mut four);
        assert!(matches!(
            four.status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));
        let mut duplicate = state.clone();
        duplicate.board.set(
            "d2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Knight).with_id(200)),
        );
        engine.resolve_highlander_victory(&mut duplicate);
        assert_eq!(duplicate.status, GameStatus::Ongoing);
        let mut no_king = four;
        no_king.status = GameStatus::Ongoing;
        let mut king = no_king.board.get("a1".parse().unwrap()).unwrap();
        king.kind = PieceKind::RoyalKnight;
        no_king.board.set("a1".parse().unwrap(), Some(king));
        engine.resolve_highlander_victory(&mut no_king);
        assert_eq!(no_king.status, GameStatus::Ongoing);
    }
}

#[cfg(test)]
mod historical_contract_tests {
    use super::*;

    #[test]
    fn highlander_unique_types_win_but_duplicate_pawns_do_not() {
        let engine = Engine::bundled_20260818().unwrap();
        for (duplicate_pawn, wins) in [(false, true), (true, false)] {
            let mut board = crate::Board::empty();
            for (square, owner, kind) in [
                ("a1", Player::White, PieceKind::King),
                ("b2", Player::White, PieceKind::Pawn),
                (
                    "c2",
                    Player::White,
                    if duplicate_pawn {
                        PieceKind::Pawn
                    } else {
                        PieceKind::Bishop
                    },
                ),
                ("h8", Player::Black, PieceKind::King),
            ] {
                board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
            }
            let state = GameState::from_board(
                crate::BUNDLED_RULESET_ID,
                Mode::Standard,
                board,
                Player::White,
                915,
            );
            let next = engine
                .grant_card(&state, 1, "highlander", Player::White)
                .unwrap()
                .state;
            assert_eq!(
                matches!(
                    next.status,
                    GameStatus::Won {
                        winner: Player::White,
                        ..
                    }
                ),
                wins
            );
        }
    }
}

use super::*;
pub(super) const ID: &str = crate::GOMOKU;
pub(super) fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.gomoku[owner.index()] = true;
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let rt = &state.card_runtime;
    let i = player.index();
    require(!rt.gomoku[i], "이미 오목이 적용 중입니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn settle_gomoku_lines(&self, state: &mut GameState) {
        let winners: Vec<_> = [Player::White, Player::Black]
            .into_iter()
            .filter(|owner| {
                state.card_runtime.gomoku[owner.index()]
                    && (0..8).any(|file| {
                        let mut count = 0;
                        (0..8).any(|rank| {
                            count = if state
                                .board
                                .get(Square::new(file, rank).unwrap())
                                .is_some_and(|p| p.owner == *owner)
                            {
                                count + 1
                            } else {
                                0
                            };
                            count >= 5
                        })
                    })
            })
            .collect();
        match winners.as_slice() {
            [winner] => {
                state.status = GameStatus::Won {
                    winner: *winner,
                    reason: WinReason::CardEffect(crate::GOMOKU.into()),
                }
            }
            [_, _] => {
                state.status = GameStatus::Draw {
                    reason: DrawReason::CardEffect(crate::GOMOKU.into()),
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::acquisition::*;
    use super::*;
    #[test]
    fn gomoku_is_checked_on_acquisition_and_simultaneous_lines_draw() {
        let engine = Engine::bundled_latest().unwrap();
        let mut before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
        ]);
        for rank in 1..5 {
            let piece = Piece::new(Player::White, PieceKind::Pawn).with_id(100 + rank as u32);
            before.board.set(Square::new(0, rank).unwrap(), Some(piece));
        }
        let won = engine
            .grant_card(&before, 1, crate::GOMOKU, Player::White)
            .unwrap();
        assert_eq!(
            won.state.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect(crate::GOMOKU.into())
            }
        );
        let mut both = before;
        for rank in 3..7 {
            both.board.set(
                Square::new(7, rank).unwrap(),
                Some(Piece::new(Player::Black, PieceKind::Pawn).with_id(200 + rank as u32)),
            );
        }
        both.card_runtime.gomoku = [true; 2];
        engine.resolve_gomoku_victory(&mut both);
        assert_eq!(
            both.status,
            GameStatus::Draw {
                reason: DrawReason::CardEffect(crate::GOMOKU.into())
            }
        );
    }
}

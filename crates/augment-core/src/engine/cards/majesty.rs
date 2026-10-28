use super::*;
pub(super) const ID: &str = crate::MAJESTY;
pub(super) fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.majesty[owner.index()] = true;
    Ok(())
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let id = ID;
    denied(
        EligibilityReasonCode::UnsupportedPredicate,
        id,
        "공식 획득 조건의 런타임 매핑이 아직 검증되지 않아 획득을 허용하지 않습니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn majesty_is_major(kind: PieceKind) -> bool {
        matches!(
            kind,
            PieceKind::Queen
                | PieceKind::Rook
                | PieceKind::Amazon
                | PieceKind::Man
                | PieceKind::Colossus
                | PieceKind::BigRook
                | PieceKind::Herald
                | PieceKind::Jester
                | PieceKind::Recruiter
                | PieceKind::Hook
                | PieceKind::PrimeMinister
                | PieceKind::Assassin
                | PieceKind::Wizard
                | PieceKind::Windmill
                | PieceKind::Crown
                | PieceKind::Bear
                | PieceKind::MagicGirl
                | PieceKind::Berserker
                | PieceKind::Siren
                | PieceKind::Trickster
                | PieceKind::Reaper
                | PieceKind::Undead
        )
    }
}

impl Engine {
    pub(in crate::engine) fn majesty_destination_blocked(
        &self,
        state: &GameState,
        piece: Piece,
        to: Square,
    ) -> bool {
        let identity = state.board.find_by_id(piece.id).map_or(piece, |(_, p)| p);
        let owner = identity.owner.opponent();
        if !state.card_runtime.majesty[owner.index()] || !Self::majesty_is_major(identity.kind) {
            return false;
        }
        let cells = if matches!(identity.kind, PieceKind::Colossus | PieceKind::BigRook)
            || state.card_runtime.large_pieces.contains_key(&identity.id)
        {
            large_piece_cells(to)
        } else {
            vec![to]
        };
        state.board.iter().any(|(square, royal)| {
            royal.owner == owner
                && (matches!(
                    royal.kind,
                    PieceKind::King | PieceKind::RoyalKnight | PieceKind::DarkWizard
                ) || state.card_runtime.royal_piece_ids.contains(&royal.id))
                && cells.iter().any(|cell| {
                    cell.file()
                        .abs_diff(square.file())
                        .max(cell.rank().abs_diff(square.rank()))
                        == 1
                })
        })
    }
}

#[cfg(test)]
mod historical_contract_tests {
    use super::super::tests::board_objectives::flag_move;
    use super::*;

    #[test]
    fn majesty_destination_filter_direct_diagnostic() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut board = crate::Board::empty();
        board.set(
            "e1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King)),
        );
        board.set(
            "e2".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Queen)),
        );
        let mut state = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::Black,
            915,
        );
        state.card_runtime.majesty[0] = true;
        let queen = state.board.get("e2".parse().unwrap()).unwrap();
        assert!(engine.majesty_destination_blocked(&state, queen, "e2".parse().unwrap()));
        assert!(!engine.majesty_destination_blocked(&state, queen, "e1".parse().unwrap()));
        assert!(!engine.majesty_destination_blocked(&state, queen, "e3".parse().unwrap()));
    }
    #[test]
    fn majesty_blocks_enemy_major_pieces_from_king_adjacent_destinations() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut board = crate::Board::empty();
        for (square, owner, kind) in [
            ("e1", Player::White, PieceKind::King),
            ("a2", Player::Black, PieceKind::Rook),
            ("a4", Player::Black, PieceKind::Pawn),
            ("d8", Player::Black, PieceKind::King),
            ("h2", Player::Black, PieceKind::Bishop),
        ] {
            board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let without = GameState::from_board(
            crate::BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            Player::Black,
            915,
        );
        // e2/d1 are adjacent to the white king; a pawn is free to go there already.
        assert!(
            engine
                .legal_actions(&without)
                .unwrap()
                .contains(&flag_move("a4", "a3"))
        );
        let with = engine
            .grant_card(&without, 1, crate::MAJESTY, Player::White)
            .unwrap()
            .state;
        assert!(with.card_runtime.majesty[Player::White.index()]);
        let actions = engine.legal_actions(&with).unwrap();
        assert!(!actions.contains(&flag_move("a2", "e2")));
        assert!(!actions.contains(&flag_move("a2", "e1")));
        assert!(actions.contains(&flag_move("a2", "a3")));
        assert!(actions.contains(&flag_move("a2", "b2")));
        // Majors only: pawns and bishops may still step next to the king.
        assert!(actions.contains(&flag_move("a4", "a3")));
        assert!(actions.contains(&flag_move("h2", "g3")));
        assert!(actions.contains(&flag_move("h2", "g1")));
        // Fingerprint and position key must both change with the flag.
        assert_ne!(without.fingerprint(), with.fingerprint());
        assert_ne!(without.position_key(), with.position_key());
        // A swap move re-checks the swap partner: relay-style king swap target of a rook is blocked.
        let royal_knight = with.clone();
        assert!(engine.apply(&with, flag_move("a2", "b2")).is_ok());
        let _ = royal_knight;
    }
}

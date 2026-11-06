use super::*;

pub(super) const ID: &str = crate::LEAP;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

/// The card is a standing permission for its owner's pawns, so it only flips the
/// owner flag the shared pawn generator consults.
fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.pawn_leap[owner.index()] = true;
    Ok(())
}

/// `isCardDrawableForColor` requires `count("pawn") > 0` (main-RjrZ5fFP.js:314537).
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    if let Some(already) = enabled(state.card_runtime.pawn_leap[owner.index()]) {
        return already;
    }
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == owner && piece.kind == PieceKind::Pawn),
        "도약할 아군 폰이 필요합니다.",
    )
}

impl Engine {
    /// Leap destinations for one pawn, from `addPawnDirectionalMoves`
    /// (main-RjrZ5fFP.js:345675-345691, diagonal form at L345735-345741).
    ///
    /// The contract is "jump over an enemy piece and advance, never capturing on the
    /// way": the immediate square must hold an *enemy* piece and the square beyond it
    /// must be empty. The current September-26 rebalance removed the old
    /// "only over an enemy pawn" clause, so any enemy piece may be vaulted.
    ///
    /// Both the straight and the two diagonal lines are covered. A promotion that
    /// happens to land on the last rank is ordinary promotion handling and belongs to
    /// the caller.
    #[allow(dead_code)] // Wired by the Lead at the pawn-move generator.
    pub(in crate::engine) fn leap_destinations(
        &self,
        state: &GameState,
        piece: Piece,
        from: Square,
    ) -> Vec<Square> {
        if piece.kind != PieceKind::Pawn || !state.card_runtime.pawn_leap[piece.owner.index()] {
            return Vec::new();
        }
        let direction = if piece.owner == Player::White { 1 } else { -1 };
        let mut destinations = Vec::new();
        for file_delta in [0_i8, -1, 1] {
            let Some(hurdle) = from.offset(file_delta, direction) else {
                continue;
            };
            let Some(vaulted) = self.movement_occupant(state, hurdle) else {
                continue;
            };
            if vaulted.owner == piece.owner {
                continue;
            }
            let Some(landing) = from.offset(file_delta * 2, direction * 2) else {
                continue;
            };
            if self.card_landing_open(state, landing) {
                destinations.push(landing);
            }
        }
        destinations
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, position};
    use super::super::tests::target_procedures::has_move;
    use super::*;

    fn armed_engine_state(engine: &Engine, extra: &[(&str, Player, PieceKind)]) -> GameState {
        let mut state = position(extra);
        acquire(engine, &mut state, Player::White, &mut Vec::new()).unwrap();
        assert!(state.card_runtime.pawn_leap[Player::White.index()]);
        assert!(!state.card_runtime.pawn_leap[Player::Black.index()]);
        state
    }

    #[test]
    fn leap_vaults_an_enemy_piece_straight_ahead_without_capturing() {
        let engine = Engine::bundled_latest().unwrap();
        let state = armed_engine_state(
            &engine,
            &[
                ("b2", Player::White, PieceKind::Pawn),
                ("b3", Player::Black, PieceKind::Knight),
            ],
        );
        let pawn = at(&state, "b2");
        let destinations = engine.leap_destinations(&state, pawn, "b2".parse().unwrap());
        assert!(destinations.contains(&"b4".parse().unwrap()));
        // The vault never captures, so the hurdle survives and b3 is not a destination.
        assert!(!destinations.contains(&"b3".parse().unwrap()));
        // Ordinary single steps still follow normal pawn rules: b3 is occupied.
        assert!(!has_move(&engine, &state, "b2", "b3"));
        assert!(state.board.get("b3".parse().unwrap()).is_some());
    }

    #[test]
    fn leap_covers_both_forward_diagonals() {
        let engine = Engine::bundled_latest().unwrap();
        let state = armed_engine_state(
            &engine,
            &[
                ("d4", Player::White, PieceKind::Pawn),
                ("c5", Player::Black, PieceKind::Rook),
                ("e5", Player::Black, PieceKind::Bishop),
                ("d5", Player::Black, PieceKind::Pawn),
            ],
        );
        let pawn = at(&state, "d4");
        let mut destinations = engine.leap_destinations(&state, pawn, "d4".parse().unwrap());
        destinations.sort();
        assert_eq!(
            destinations,
            vec![
                "b6".parse().unwrap(),
                "d6".parse().unwrap(),
                "f6".parse().unwrap()
            ]
        );
    }

    #[test]
    fn leap_requires_an_enemy_hurdle_and_an_empty_landing_square() {
        let engine = Engine::bundled_latest().unwrap();
        // A friendly hurdle does not count.
        let friendly = armed_engine_state(
            &engine,
            &[
                ("b2", Player::White, PieceKind::Pawn),
                ("b3", Player::White, PieceKind::Knight),
            ],
        );
        assert!(
            engine
                .leap_destinations(&friendly, at(&friendly, "b2"), "b2".parse().unwrap())
                .is_empty()
        );
        // An occupied landing square blocks the vault.
        let blocked = armed_engine_state(
            &engine,
            &[
                ("b2", Player::White, PieceKind::Pawn),
                ("b3", Player::Black, PieceKind::Knight),
                ("b4", Player::Black, PieceKind::Rook),
            ],
        );
        assert!(
            engine
                .leap_destinations(&blocked, at(&blocked, "b2"), "b2".parse().unwrap())
                .is_empty()
        );
        // No hurdle at all: nothing beyond a normal step.
        let empty = armed_engine_state(&engine, &[("b2", Player::White, PieceKind::Pawn)]);
        assert!(
            engine
                .leap_destinations(&empty, at(&empty, "b2"), "b2".parse().unwrap())
                .is_empty()
        );
    }

    #[test]
    fn leap_only_applies_to_pawns_and_only_while_armed() {
        let engine = Engine::bundled_latest().unwrap();
        let state = armed_engine_state(
            &engine,
            &[
                ("b2", Player::White, PieceKind::Knight),
                ("b3", Player::Black, PieceKind::Knight),
            ],
        );
        assert!(
            engine
                .leap_destinations(&state, at(&state, "b2"), "b2".parse().unwrap())
                .is_empty()
        );
        let unarmed = position(&[
            ("b2", Player::White, PieceKind::Pawn),
            ("b3", Player::Black, PieceKind::Knight),
        ]);
        assert!(
            engine
                .leap_destinations(&unarmed, at(&unarmed, "b2"), "b2".parse().unwrap())
                .is_empty()
        );
    }

    #[test]
    fn leap_works_for_black_pawns_in_the_other_direction() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b7", Player::Black, PieceKind::Pawn),
            ("b6", Player::White, PieceKind::Knight),
        ]);
        acquire(&engine, &mut state, Player::Black, &mut Vec::new()).unwrap();
        let pawn = at(&state, "b7");
        assert_eq!(
            engine.leap_destinations(&state, pawn, "b7".parse().unwrap()),
            vec!["b5".parse().unwrap()]
        );
        // The white player never gained the permission.
        assert!(state.card_runtime.pawn_leap[Player::Black.index()]);
        assert!(!state.card_runtime.pawn_leap[Player::White.index()]);
    }
}

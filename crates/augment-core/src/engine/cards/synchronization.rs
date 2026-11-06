use super::*;

pub(super) const ID: &str = crate::SYNCHRONIZATION;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, None, eligibility);

/// Synchronization has no expiring state: holding the card *is* being armed.
/// Upstream models it as `state.synchronization[color] = true` on acquisition
/// (`applySeptember26SimpleCard`, main-RjrZ5fFP.js:1306-1313) and never clears it,
/// and the current engine already keys standing card permissions off `state.cards`
/// (see `d4.rs`). Reading membership here keeps this card out of shared state.
fn armed(state: &GameState, owner: Player) -> bool {
    state
        .cards
        .iter()
        .any(|card| card.owner == owner && card.definition_id == ID)
}

/// `synchronizedSquareColor` (main-RjrZ5fFP.js:1340-1350): the shared square colour
/// of every remaining piece of `owner`, or `None` when they are mixed. A colour with
/// no pieces left has no parity and therefore no restriction.
pub(in crate::engine) fn synchronization_parity(state: &GameState, owner: Player) -> Option<bool> {
    let mut parity: Option<bool> = None;
    for (square, piece) in state.board.iter() {
        if piece.owner != owner {
            continue;
        }
        let shade = (square.file() + square.rank()) % 2 == 1;
        match parity {
            None => parity = Some(shade),
            Some(current) if current == shade => {}
            Some(_) => return None,
        }
    }
    parity
}

/// `expansionDestinationAllowed` (main-RjrZ5fFP.js:1351-1364): with the card armed
/// for `mover.opponent()`, the mover may not land on a square whose colour differs
/// from the armed player's synchronized colour.
///
/// Return `true` to *forbid* the arrival.
pub(in crate::engine) fn synchronization_forbids_arrival(
    state: &GameState,
    mover: Player,
    destination: Square,
) -> bool {
    let owner = mover.opponent();
    if !armed(state, owner) {
        return false;
    }
    let Some(parity) = synchronization_parity(state, owner) else {
        return false;
    };
    let destination_shade = (destination.file() + destination.rank()) % 2 == 1;
    destination_shade != parity
}

/// Draw requirement is `ANY`; the effect is always installable. Re-acquiring an
/// already armed copy is still rejected so the shared global-uniqueness rule holds.
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    enabled(armed(state, owner)).unwrap_or(CardEligibility::Eligible)
}

impl Engine {
    /// Convenience wrapper for the shared move filter, mirroring the d4/majesty
    /// style hooks. Forbids an arrival when the opponent's sync bond is active.
    #[allow(dead_code)] // Wired by the Lead at the card-state move filter.
    pub(in crate::engine) fn synchronization_blocks_arrival(
        &self,
        state: &GameState,
        piece: Piece,
        destination: Square,
    ) -> bool {
        synchronization_forbids_arrival(state, piece.owner, destination)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{movement, position};
    use super::super::tests::target_procedures::{has_move, position as seated};
    use super::*;

    /// Install the card straight into `state.cards`; the registry-based grant path
    /// is unavailable while the card sits in the profile's `pending` list.
    fn armed_state(mut state: GameState, owner: Player) -> GameState {
        state.add_card(CardInstance {
            instance_id: 1,
            definition_id: ID.to_owned(),
            owner,
            half_stars: 0,
            charges: 0,
            available_after_turn: None,
        });
        assert!(armed(&state, owner));
        state
    }

    #[test]
    fn synchronization_parity_is_none_once_friendly_pieces_share_no_colour() {
        // Only the king on a1: (0+0) is even, so the colour is synchronized.
        let only_king = position(&[]);
        assert_eq!(
            synchronization_parity(&only_king, Player::White),
            Some(false)
        );
        // Adding a knight on b1 (odd) breaks the bond.
        let mixed = position(&[("b1", Player::White, PieceKind::Knight)]);
        assert_eq!(synchronization_parity(&mixed, Player::White), None);
        // All-even again.
        let even = position(&[
            ("b1", Player::White, PieceKind::Knight),
            ("d1", Player::White, PieceKind::Rook),
        ]);
        assert_eq!(synchronization_parity(&even, Player::White), Some(true));
        // A piece set that happens to be all odd.
        let odd = position(&[
            ("c1", Player::White, PieceKind::Knight),
            ("a3", Player::White, PieceKind::Rook),
        ]);
        assert_eq!(synchronization_parity(&odd, Player::White), Some(false));
        // No pieces at all on a colour: no parity, no restriction.
        let lone = seated(&[("h8", Player::Black, PieceKind::King)], Player::White);
        assert_eq!(synchronization_parity(&lone, Player::White), None);
    }

    #[test]
    fn synchronization_forbids_enemy_arrivals_on_the_opposite_shade() {
        let engine = Engine::bundled_latest().unwrap();
        // White is synchronized on the dark shade (c3 is (2+2) even).
        let mut state = armed_state(
            position(&[
                ("c3", Player::White, PieceKind::Knight),
                ("d6", Player::Black, PieceKind::Rook),
            ]),
            Player::White,
        );
        assert_eq!(synchronization_parity(&state, Player::White), Some(false));
        // d6 is (3+5) even, in step with White; e6 is (4+5) odd, forbidden.
        assert!(!synchronization_forbids_arrival(
            &state,
            Player::Black,
            "d7".parse().unwrap()
        ));
        assert!(synchronization_forbids_arrival(
            &state,
            Player::Black,
            "e6".parse().unwrap()
        ));
        // White's own arrivals are never restricted by its own card.
        assert!(!synchronization_forbids_arrival(
            &state,
            Player::White,
            "e6".parse().unwrap()
        ));
        // Without the card the same arrival is allowed.
        let plain = position(&[
            ("c3", Player::White, PieceKind::Knight),
            ("d6", Player::Black, PieceKind::Rook),
        ]);
        assert!(!synchronization_forbids_arrival(
            &plain,
            Player::Black,
            "e6".parse().unwrap()
        ));
        state.turn = Player::Black;
        // The hook agrees with the exported predicate on the live position.
        let rook = state.board.get("d6".parse().unwrap()).unwrap();
        assert!(engine.synchronization_blocks_arrival(&state, rook, "e6".parse().unwrap()));
        assert!(!engine.synchronization_blocks_arrival(&state, rook, "d7".parse().unwrap()));
        // Black is on move, so the filtered legal set never reaches a forbidden shade.
        assert!(has_move(&engine, &state, "d6", "d7"));
        assert!(!has_move(&engine, &state, "d6", "e6"));
        let _ = movement("d6", "e6");
    }

    #[test]
    fn synchronization_is_inert_while_friendly_pieces_are_mixed_or_absent() {
        let mixed = armed_state(
            position(&[
                ("b1", Player::White, PieceKind::Knight),
                ("d1", Player::White, PieceKind::Rook),
                ("d6", Player::Black, PieceKind::Rook),
            ]),
            Player::White,
        );
        assert_eq!(synchronization_parity(&mixed, Player::White), None);
        assert!(!synchronization_forbids_arrival(
            &mixed,
            Player::Black,
            "e6".parse().unwrap()
        ));
        // An empty side cannot synchronize either.
        let empty = armed_state(seated(&[], Player::White), Player::White);
        assert_eq!(synchronization_parity(&empty, Player::White), None);
        assert!(!synchronization_forbids_arrival(
            &empty,
            Player::Black,
            "e6".parse().unwrap()
        ));
        // The card only ever protects the side that owns it.
        assert!(!armed(&mixed, Player::Black));
    }

    #[test]
    fn synchronization_eligibility_rejects_a_duplicate_grant() {
        let state = armed_state(position(&[]), Player::White);
        assert!(!eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible());
        assert!(!eligibility(&state, Player::White, AcquisitionContext::Setup).is_eligible());
        assert!(eligibility(&state, Player::Black, AcquisitionContext::Draft).is_eligible());
    }
}

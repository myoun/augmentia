use super::*;
pub(super) const ID: &str = crate::IQP;
pub(super) fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.october.iqp_pending[owner.index()] = true;
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
    pub(in crate::engine) fn settle_iqp_acquisitions(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID
            || !matches!(state.phase, GamePhase::Playing)
        {
            return;
        }
        for owner in [Player::White, Player::Black] {
            if !std::mem::take(&mut state.card_runtime.october.iqp_pending[owner.index()]) {
                continue;
            }
            let targets: Vec<_> = state
                .board
                .iter()
                .filter(|(square, piece)| {
                    piece.owner == owner.opponent()
                        && piece.kind == PieceKind::Pawn
                        && matches!(square.file(), 2 | 4)
                })
                .collect();
            for (square, piece) in targets {
                // v8RemoveCurrentPieces(... rememberCapture:false): direct removal,
                // without crediting a capturing piece or scheduling capture reactions.
                self.delete_piece_for_card(state, square, piece, owner, crate::IQP, events);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn iqp_removes_current_enemy_c_e_pawns_after_draft_without_capture_reactions() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 1);
        // Move an original d-pawn into the c-file: current location is authoritative.
        let d = state.board.get("d7".parse().unwrap()).unwrap();
        state.board.set("d7".parse().unwrap(), None);
        state.board.set("c5".parse().unwrap(), Some(d));
        let c = state.board.get("c7".parse().unwrap()).unwrap();
        state.card_runtime.shielded_piece_ids.insert(c.id);
        state.card_runtime.recurrence_piece_ids.insert(c.id);
        state.phase = GamePhase::Draft(DraftState {
            phase: DraftPhase::Opening,
            offers: vec![crate::IQP.into()],
            resume_turn: Player::White,
        });
        let picked = engine
            .apply(&state, Action::DraftPick { offer_index: 0 })
            .unwrap();
        assert!(picked.state.board.get("c7".parse().unwrap()).is_some());
        assert!(picked.state.card_runtime.october.iqp_pending[0]);
        let mut next_state = picked.state;
        let GamePhase::Draft(offer) = &mut next_state.phase else {
            panic!("expected second opening draft")
        };
        offer.offers = vec![crate::RADICAL_CHARGE.into()];
        let next = engine
            .apply(&next_state, Action::DraftPick { offer_index: 0 })
            .unwrap();
        assert!(next.state.board.get("c7".parse().unwrap()).is_none());
        assert!(next.state.board.get("e7".parse().unwrap()).is_none());
        assert!(next.state.board.get("c5".parse().unwrap()).is_none());
        assert!(next.state.board.get("c2".parse().unwrap()).is_some());
        assert!(next.state.card_runtime.pending_recurrences.is_empty());
        assert_eq!(next.state.card_runtime.october.iqp_pending, [false; 2]);
    }
}

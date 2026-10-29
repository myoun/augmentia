use super::*;
pub(super) const ID: &str = crate::MERCHANT_GUILD;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let king = web_ordered_pieces(state)
        .into_iter()
        .find(|(_, p)| p.owner == owner && p.kind == PieceKind::King);
    let Some((from, mut piece)) = king else {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant("no merchant king".into()));
    };
    let Some(to) = from.offset(0, if owner == Player::White { 2 } else { -2 }) else {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant(
            "merchant destination off board".into(),
        ));
    };
    if let Some(target) = state.board.get(to) {
        if target.owner != owner {
            if matches!(state.phase, GamePhase::Draft(_)) {
                return Ok(());
            }
            return Err(EngineError::Invariant(
                "merchant destination blocked".into(),
            ));
        }
        engine.delete_piece_for_card(state, to, target, owner, ID, events);
    }
    state.board.set(from, None);
    engine.clear_queens_gambit_protection(state, piece.id);
    piece.kind = PieceKind::Merchant;
    state.board.set(to, Some(piece));
    state.card_runtime.royal_piece_ids.insert(piece.id);
    state.card_runtime.moved_piece_ids.insert(piece.id);
    state.card_runtime.merchant_gold.insert(piece.id, 1);
    Ok(())
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    let king = state
        .board
        .iter()
        .find(|(_, p)| p.owner == owner && p.kind == PieceKind::King);
    require(
        king.and_then(|(s, _)| s.offset(0, if owner == Player::White { 2 } else { -2 }))
            .is_some_and(|to| state.board.get(to).is_none_or(|p| p.owner == owner)),
        "아군 킹과 전방 두 칸의 배치 공간이 필요합니다.",
    )
}
impl Engine {
    pub(in crate::engine) fn current_merchant_price(
        &self,
        state: &GameState,
        target: Piece,
    ) -> u16 {
        if self.is_royal_piece(state, target) || target.kind == PieceKind::Vip {
            20
        } else {
            super::revelation::combat_value(target.kind).max(2)
        }
    }
    pub(in crate::engine) fn current_merchant_target(
        &self,
        state: &GameState,
        merchant: Piece,
        target: Piece,
    ) -> bool {
        target.owner != merchant.owner
            && !matches!(
                target.kind,
                PieceKind::Merchant
                    | PieceKind::Monster
                    | PieceKind::Football
                    | PieceKind::Scarecrow
            )
            && !state.card_runtime.shielded_piece_ids.contains(&target.id)
            && (!state.card_runtime.protected_piece_ids.contains(&target.id)
                || state
                    .card_runtime
                    .queens_gambit_protection
                    .contains(&target.id)
                    && !state
                        .card_runtime
                        .queens_gambit_previous_protection
                        .contains(&target.id)
                    && !state
                        .card_runtime
                        .sacrifice_protection
                        .contains_key(&target.id)
                    && !state.card_runtime.last_resistance.contains_key(&target.id)
                    && !state
                        .card_runtime
                        .coronation_protection
                        .contains_key(&target.id))
            && !self.middle_submerged_protects(state, target)
            && !state
                .card_runtime
                .outpost_protected_piece_ids
                .contains(&target.id)
            && self.capture_status_allows_ignoring_fresh(state, merchant)
            && self.current_merchant_price(state, target)
                <= state
                    .card_runtime
                    .merchant_gold
                    .get(&merchant.id)
                    .copied()
                    .unwrap_or(0)
    }
}

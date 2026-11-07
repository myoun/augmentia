use super::*;
pub(super) const ID: &str = crate::UNDERGROUND_BUNKER;

/// Durability granted to the king, from the catalog text ("5만큼의 HP") and the
/// internal balance revision (`INTERNAL_BUNKER_HP`, main bundle 2922).
const BUNKER_HP: u8 = 5;

/// Underground Bunker gives the owner's king five hits of durability and takes away
/// its ability to move.
///
/// Upstream marks the king, sets its hp and maxHp to the bunker value, and treats it
/// as immobile from then on (`isWorkerUndergroundBunkerKing`, main bundle 28805-28807,
/// applied at 25755-25760).
///
/// The card is disabled upstream in the 2026-11-02 snapshot, so this behavior is
/// retained for the historical rulesets and is not in the current profile.
pub(super) fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let Some((square, king)) = state
        .board
        .iter()
        .find(|(_, piece)| piece.owner == owner && royal(state, *piece))
    else {
        return Err(EngineError::Invariant(
            "underground bunker needs an allied royal piece".to_owned(),
        ));
    };
    state
        .card_runtime
        .grant_health(king.id, PieceHealth::new(BUNKER_HP), true);
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("underground-bunker:{}:{square}:{BUNKER_HP}", king.id),
    });
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    // A king that already has a bunker cannot receive another one.
    let available = state
        .board
        .iter()
        .any(|(_, piece)| piece.owner == owner && royal(state, piece))
        && !state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == owner && state.card_runtime.is_immobile(piece.id));
    require(available, "지하벙커를 적용할 아군 킹이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    fn bunkered(engine: &Engine) -> (GameState, Piece) {
        let mut state = position(&[("a2", Player::White, PieceKind::Pawn)]);
        let king = state.board.get("a1".parse().unwrap()).unwrap();
        acquire(engine, &mut state, Player::White, &mut Vec::new()).unwrap();
        (state, king)
    }

    #[test]
    fn bunker_gives_the_king_five_hits_without_a_footprint() {
        let engine = Engine::bundled_latest().unwrap();
        let (state, king) = bunkered(&engine);
        assert_eq!(state.card_runtime.health(king.id).unwrap().hp, BUNKER_HP);
        assert_eq!(
            state.card_runtime.health(king.id).unwrap().max_hp,
            BUNKER_HP
        );
        assert!(state.card_runtime.is_immobile(king.id));
        // Health is independent of size, which is the point of the split.
        assert!(!state.card_runtime.is_large(king.id));
        assert_eq!(state.card_runtime.large_anchor(king.id), None);
    }

    #[test]
    fn a_bunkered_king_survives_hits_until_the_last_one() {
        let engine = Engine::bundled_latest().unwrap();
        let (mut state, king) = bunkered(&engine);
        for expected in (1..BUNKER_HP).rev() {
            let destroyed = state
                .card_runtime
                .health_mut(king.id)
                .map(PieceHealth::take_hit)
                .unwrap();
            assert!(!destroyed, "the king must survive above zero");
            assert_eq!(state.card_runtime.health(king.id).unwrap().hp, expected);
        }
        assert!(
            state
                .card_runtime
                .health_mut(king.id)
                .map(PieceHealth::take_hit)
                .unwrap(),
            "the last hit destroys the king"
        );
    }

    #[test]
    fn bunker_is_rejected_without_a_king_or_when_already_applied() {
        let engine = Engine::bundled_latest().unwrap();
        // No white royal on the board.
        let mut stripped = position(&[("a2", Player::White, PieceKind::Pawn)]);
        stripped.board.set("a1".parse().unwrap(), None);
        assert!(!eligibility(&stripped, Player::White, AcquisitionContext::Draft).is_eligible());

        // A bunkered king cannot be bunkered again.
        let (once, _) = bunkered(&engine);
        assert!(!eligibility(&once, Player::White, AcquisitionContext::Draft).is_eligible());
    }

    #[test]
    fn the_bunkered_king_has_no_legal_moves() {
        let engine = Engine::bundled_latest().unwrap();
        let (state, king) = bunkered(&engine);
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .iter()
                .any(|action| matches!(action, Action::Move(movement) if movement.from == state.board.find_by_id(king.id).unwrap().0)),
            "a bunkered king must not move"
        );
    }
}

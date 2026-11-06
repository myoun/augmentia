use super::*;
pub(super) const ID: &str = crate::VIP;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Own or enemy pawns can be chosen. Anything already promoted is not a pawn and
/// therefore cannot be nominated again.
fn candidates(state: &GameState) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| piece.kind == PieceKind::Pawn)
        .collect()
}

fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, pawn) = piece_target(state, target)?;
    if pawn.kind != PieceKind::Pawn
        || state
            .card_runtime
            .vip_pending
            .iter()
            .any(|pending| pending.piece_id == pawn.id)
    {
        return Err(EngineError::Invariant("invalid vip target".to_owned()));
    }
    // Upstream `applyVipV5`: three of the pawn owner's own turns, one extra ply
    // when the caster nominates an enemy pawn. `state.ply` is the bundle's
    // `currentPly`, and `tick_end_statuses_after_turn` performs the conversion,
    // so no new state field or lifecycle hook is needed.
    let due_ply = state
        .ply
        .saturating_add(if pawn.owner == actor { 6 } else { 7 });
    state.card_runtime.vip_pending.push(PendingVip {
        piece_id: pawn.id,
        owner: pawn.owner,
        due_ply,
    });
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("vip:{}:{due_ply}", pawn.id),
    });
    Ok(())
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, _) in candidates(state) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        });
    }
}

fn eligibility(state: &GameState, _: Player, context: AcquisitionContext) -> CardEligibility {
    if context == AcquisitionContext::Draft {
        return CardEligibility::Eligible;
    }
    require(!candidates(state).is_empty(), "보드에 폰이 필요합니다.")
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    fn tick_vip(engine: &Engine, s: &mut GameState) {
        engine.tick_end_statuses_after_turn(s, s.turn.opponent(), &mut Vec::new());
    }

    #[test]
    fn vip_nominates_an_own_pawn_due_three_owner_turns_later() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("c2", Player::White, PieceKind::Pawn)]);
        let pawn = at(&s, "c2");
        use_at(&mut s, &engine, crate::VIP, single("c2"));
        assert_eq!(s.card_runtime.vip_pending.len(), 1);
        let pending = s.card_runtime.vip_pending[0];
        assert_eq!(pending.piece_id, pawn.id);
        assert_eq!(pending.owner, Player::White);
        assert_eq!(pending.due_ply, 6);
        // The pawn stays a pawn until the due ply passes.
        tick_vip(&engine, &mut s);
        assert_eq!(at(&s, "c2").kind, PieceKind::Pawn);
        s.ply = pending.due_ply;
        tick_vip(&engine, &mut s);
        assert_eq!(at(&s, "c2").kind, PieceKind::Vip);
        assert!(s.card_runtime.vip_pending.is_empty());
    }

    #[test]
    fn vip_can_also_nominate_an_enemy_pawn_one_ply_later() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("g7", Player::Black, PieceKind::Pawn)]);
        use_at(&mut s, &engine, crate::VIP, single("g7"));
        let pending = s.card_runtime.vip_pending[0];
        assert_eq!(pending.owner, Player::Black);
        assert_eq!(pending.due_ply, 7);
        s.ply = pending.due_ply;
        tick_vip(&engine, &mut s);
        let converted = at(&s, "g7");
        assert_eq!(converted.kind, PieceKind::Vip);
        assert_eq!(converted.owner, Player::Black);
    }

    #[test]
    fn vip_rejects_non_pawn_targets_and_keeps_state_stable() {
        let engine = Engine::bundled_latest().unwrap();
        let mut s = position(&[("d1", Player::White, PieceKind::Queen)]);
        s = engine
            .grant_card(&s, 1, crate::VIP, Player::White)
            .unwrap()
            .state;
        assert!(
            !engine
                .legal_actions(&s)
                .unwrap()
                .iter()
                .any(|action| matches!(action, Action::UseCard { .. }))
        );
        let before = s.clone();
        assert!(
            engine
                .apply(
                    &s,
                    Action::UseCard {
                        instance_id: 1,
                        target: single("d1"),
                    },
                )
                .is_err()
        );
        assert_eq!(s.board, before.board);
        assert!(s.card_runtime.vip_pending.is_empty());
    }
}

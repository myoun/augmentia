use super::*;
pub(super) const ID: &str = crate::RECURRENCE;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    _target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    engine.apply_recurrence(state, actor, events)?;
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    if !engine.recurrence_targets(state, state.turn).is_empty() {
        actions.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let rt = &state.card_runtime;
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    require(
        context == AcquisitionContext::Draft
            || has_own(&|p| {
                !matches!(p.kind, Pawn | Monster | Crown | Football)
                    && !royal(state, p)
                    && !rt.recurrence_piece_ids.contains(&p.id)
            }),
        "회귀를 부여할 비왕실 비폰 기물이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

impl Engine {
    pub(in crate::engine) fn recurrence_targets(
        &self,
        state: &GameState,
        owner: Player,
    ) -> Vec<(Square, Piece)> {
        web_ordered_pieces(state)
            .into_iter()
            .filter(|(_, p)| {
                p.owner == owner
                    && !matches!(
                        p.kind,
                        PieceKind::Pawn
                            | PieceKind::Monster
                            | PieceKind::Crown
                            | PieceKind::Football
                    )
                    && !self.is_royal_piece(state, *p)
                    && !state.card_runtime.recurrence_piece_ids.contains(&p.id)
            })
            .collect()
    }
}

impl Engine {
    pub(in crate::engine) fn apply_recurrence(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let candidates = self.recurrence_targets(state, owner);
        if candidates.is_empty() {
            return Err(EngineError::Invariant(
                "회귀를 부여할 기물이 없습니다.".into(),
            ));
        }
        // Active recurrence requests a draw even when there is only one candidate.
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        events.push(Event::RandomSample {
            purpose: "card.internal.recurrence.piece".into(),
            roll_53: roll,
        });
        state
            .card_runtime
            .recurrence_piece_ids
            .insert(candidates[index.unwrap()].1.id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::acquisition::*;
    use super::*;
    #[test]
    fn recurrence_is_active_does_not_fire_on_grant_and_draws_even_for_one_target() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(&[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("c3", Player::White, PieceKind::Knight),
            ("b2", Player::White, PieceKind::Pawn),
        ]);
        let granted = engine
            .grant_card(&before, 1, crate::RECURRENCE, Player::White)
            .unwrap();
        assert_eq!(granted.state.rng_state(), before.rng_state());
        assert!(granted.state.card_runtime.recurrence_piece_ids.is_empty());
        assert_eq!(granted.state.cards[0].charges, 1);
        let used = engine
            .apply(
                &granted.state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .unwrap();
        let knight = used.state.board.get("c3".parse().unwrap()).unwrap();
        assert_eq!(
            used.state.card_runtime.recurrence_piece_ids,
            [knight.id].into_iter().collect()
        );
        assert_eq!(
            used.events
                .iter()
                .filter(|e| matches!(e, Event::RandomSample { .. }))
                .count(),
            1
        );
        assert_eq!(used.state.ply, 0);
        assert_eq!(used.state.cards[0].charges, 0);
    }
}

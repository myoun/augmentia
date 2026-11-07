use super::*;
pub(super) const ID: &str = crate::OMEGA_CHESSTRON;

/// Owner turns between the sacrifice and the summon (`remaining: 7`, main bundle
/// 24378).
const SUMMON_DELAY_TURNS: u32 = 7;

/// Omega Chesstron sacrifices every non-royal own piece, then summons the body
/// seven owner turns later.
///
/// Upstream records one `pendingOmegaSummons` entry and clears the sacrificed
/// pieces immediately (main bundle 24374-24380), so the card is a single
/// delayed summon rather than one summon per sacrificed piece.
pub(super) fn acquire(
    _: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let sacrificed: Vec<Square> = state
        .board
        .iter()
        .filter(|(_, piece)| piece.owner == owner && !royal(state, *piece))
        .map(|(square, _)| square)
        .collect();
    for square in sacrificed {
        state.board.set(square, None);
    }
    state
        .card_runtime
        .pending_omega_summons
        .retain(|entry| entry.owner != owner);
    state
        .card_runtime
        .pending_omega_summons
        .push(crate::state::PendingOmegaSummon {
            owner,
            due_turn: state.turns_taken[owner.index()].saturating_add(SUMMON_DELAY_TURNS),
        });
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("omega-sacrifice:{owner}"),
    });
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    let pending = state
        .card_runtime
        .pending_omega_summons
        .iter()
        .any(|entry| entry.owner == owner);
    if let Some(ineligible) = enabled(pending) {
        return ineligible;
    }
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    /// Summon every due Omega Chesstron body on an open 3x3 anchor.
    pub(in crate::engine) fn settle_omega_summons(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let due = state
            .card_runtime
            .pending_omega_summons
            .iter()
            .filter(|entry| {
                entry.owner == owner && state.turns_taken[owner.index()] >= entry.due_turn
            })
            .count();
        if due == 0 {
            return;
        }
        state.card_runtime.pending_omega_summons.retain(|entry| {
            !(entry.owner == owner && state.turns_taken[owner.index()] >= entry.due_turn)
        });
        for _ in 0..due {
            let Some(anchor) = self.omega_anchor(state) else {
                continue;
            };
            let body =
                Piece::new(owner, PieceKind::OmegaChesstron).with_id(state.allocate_piece_id());
            state.board.set(anchor, Some(body));
            state
                .card_runtime
                .insert_large_body(body.id, anchor, PieceHealth::new(3));
            self.register_spawned_piece(state, anchor, body, true);
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("omega-summon:{}:{}", body.id, anchor),
            });
        }
    }

    /// The first open 3x3 anchor, searched from the top rank down in file order.
    fn omega_anchor(&self, state: &GameState) -> Option<Square> {
        for rank in (2..8).rev() {
            for file in 0..6 {
                let Some(anchor) = Square::new(file, rank) else {
                    continue;
                };
                let open = (0..3).all(|offset_file| {
                    (0..3).all(|offset_rank| {
                        anchor
                            .offset(offset_file, -offset_rank)
                            .is_some_and(|cell| self.card_landing_open(state, cell))
                    })
                });
                if open {
                    return Some(anchor);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    #[test]
    fn omega_sacrifices_non_royals_and_schedules_one_summon() {
        let engine = Engine::bundled_latest().unwrap();
        let before = position(&[
            ("d4", Player::White, PieceKind::Rook),
            ("e4", Player::White, PieceKind::Knight),
        ]);
        let after = engine
            .grant_card(&before, 1, crate::OMEGA_CHESSTRON, Player::White)
            .unwrap()
            .state;
        // Every non-royal is gone; only kings remain.
        assert_eq!(
            after
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == Player::White)
                .count(),
            1
        );
        assert_eq!(after.card_runtime.pending_omega_summons.len(), 1);
        assert_eq!(
            after.card_runtime.pending_omega_summons[0].due_turn,
            SUMMON_DELAY_TURNS
        );
    }

    #[test]
    fn the_body_is_summoned_after_seven_owner_turns() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("d4", Player::White, PieceKind::Rook)]);
        state = engine
            .grant_card(&state, 1, crate::OMEGA_CHESSTRON, Player::White)
            .unwrap()
            .state;
        // Not yet due at six turns.
        state.turns_taken[Player::White.index()] = 6;
        let mut events = Vec::new();
        engine.settle_omega_summons(&mut state, Player::White, &mut events);
        assert!(
            !state
                .board
                .iter()
                .any(|(_, piece)| piece.kind == PieceKind::OmegaChesstron)
        );
        // Due at seven.
        state.turns_taken[Player::White.index()] = 7;
        engine.settle_omega_summons(&mut state, Player::White, &mut events);
        assert!(
            state
                .board
                .iter()
                .any(|(_, piece)| piece.kind == PieceKind::OmegaChesstron),
            "the body must appear on the seventh owner turn"
        );
        assert!(state.card_runtime.pending_omega_summons.is_empty());
    }
}

use super::*;
pub(super) const ID: &str = crate::BOMBARDMENT;

/// Owner turns between the targeting and the collapse. The November expansion
/// shortened it from three to two (`remaining: usesNovemberExpansion ? 2 : 3`,
/// main bundle 49160).
const COLLAPSE_DELAY_TURNS: u32 = 2;

pub(super) fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let file = square_target(target)?.file();
    state
        .rule_runtime
        .pending_bombardments
        .push(crate::state::PendingBombardment {
            by: actor,
            file,
            due_turn: state.turns_taken[actor.index()].saturating_add(COLLAPSE_DELAY_TURNS),
        });
    events.push(Event::EffectApplied {
        source_card_instance: 0,
        command: format!("bombardment:{actor}:{file}"),
    });
    Ok(())
}

pub(super) fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if card.owner != state.turn {
        return;
    }
    for file in 0..8 {
        if let Some(square) = Square::new(file, 0) {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}

pub(super) fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

impl Engine {
    /// Collapse every due bombarded file: all pieces standing on it are removed
    /// and any installation or reservation there is cleared.
    ///
    /// Mirrors `applyBombardmentFile` (main bundle 58154) driven by the pending
    /// entry's remaining-turn countdown.
    pub(in crate::engine) fn settle_bombardments(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let due = state
            .rule_runtime
            .pending_bombardments
            .iter()
            .filter(|entry| entry.by == owner && state.turns_taken[owner.index()] >= entry.due_turn)
            .map(|entry| entry.file)
            .collect::<Vec<_>>();
        if due.is_empty() {
            return;
        }
        state.rule_runtime.pending_bombardments.retain(|entry| {
            !(entry.by == owner && state.turns_taken[owner.index()] >= entry.due_turn)
        });
        for file in due {
            let cells: Vec<Square> = (0..8).filter_map(|rank| Square::new(file, rank)).collect();
            let victims: Vec<_> = state
                .board
                .iter()
                .filter(|(square, piece)| {
                    cells.contains(square)
                        || state
                            .card_runtime
                            .large_anchor(piece.id)
                            .is_some_and(|anchor| {
                                large_piece_cells(anchor)
                                    .iter()
                                    .any(|cell| cells.contains(cell))
                            })
                })
                .collect();
            let memories = self
                .rule_capture_memories(state, &victims.iter().map(|(_, p)| *p).collect::<Vec<_>>());
            for (square, piece) in victims {
                self.remove_piece_for_card(state, square, piece, owner, ID, events);
            }
            self.settle_rule_capture_memories(state, &memories, events);
            for cell in cells {
                state.card_runtime.remove_barricade(cell);
                state.card_runtime.black_holes.remove(&cell);
                state.card_runtime.rule_bombs.remove(&cell);
                state.card_runtime.highway_cells.remove(&cell);
                state
                    .card_runtime
                    .quantum_positions
                    .retain(|_, s| *s != cell);
                state.rule_runtime.bomb_ignore.remove(&cell);
            }
            state
                .card_runtime
                .pending_lobsters
                .retain(|pending| pending.square.file() != file);
            state
                .card_runtime
                .pending_scarecrows
                .retain(|pending| pending.square.file() != file);
            events.push(Event::EffectApplied {
                source_card_instance: 0,
                command: format!("bombardment-collapse:{file}"),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::position;
    use super::*;

    #[test]
    fn bombardment_offers_one_target_per_file_and_records_the_delay() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("d4", Player::White, PieceKind::Rook)]);
        state = engine
            .grant_card(&state, 1, crate::BOMBARDMENT, Player::White)
            .unwrap()
            .state;
        let actions: Vec<Action> = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .filter(|action| matches!(action, Action::UseCard { .. }))
            .collect();
        assert_eq!(actions.len(), 8, "one target per file");
        let after = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::PieceAt("c1".parse().unwrap()),
                },
            )
            .unwrap()
            .state;
        let pending = after.rule_runtime.pending_bombardments[0];
        assert_eq!(pending.file, 2, "the c file");
        assert_eq!(pending.due_turn, 2, "two owner turns after the cast");
    }

    #[test]
    fn the_bombarded_file_collapses_after_two_owner_turns() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("c2", Player::Black, PieceKind::Rook),
            ("d2", Player::Black, PieceKind::Rook),
        ]);
        state
            .rule_runtime
            .pending_bombardments
            .push(crate::state::PendingBombardment {
                by: Player::White,
                file: 2,
                due_turn: 1,
            });
        state.turns_taken[Player::White.index()] = 1;
        let mut events = Vec::new();
        engine.settle_bombardments(&mut state, Player::White, &mut events);
        assert!(
            state.board.get("c2".parse().unwrap()).is_none(),
            "the bombarded file is cleared"
        );
        assert!(
            state.board.get("d2".parse().unwrap()).is_some(),
            "neighbouring files survive"
        );
        assert!(state.rule_runtime.pending_bombardments.is_empty());
    }
}

use super::*;
pub(super) const ID: &str = crate::FILE_CONVEYOR;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let square = square_target(target)?;
    state
        .card_runtime
        .file_conveyors
        .retain(|(file, _, _)| *file != square.file());
    state.card_runtime.file_conveyors.push((
        square.file(),
        actor,
        state.turns_taken[actor.index()].saturating_add(2),
    ));
    Ok(())
}
fn actions(_: &Engine, _: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for i in 0..64 {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Square(Square::from_index(i).unwrap()),
        });
    }
}
fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}
impl Engine {
    pub(in crate::engine) fn settle_file_conveyors(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            return;
        }
        let files: Vec<_> = state
            .card_runtime
            .file_conveyors
            .iter()
            .copied()
            .filter(|(_, owner, due)| *owner == actor && state.turns_taken[owner.index()] >= *due)
            .collect();
        for (file, owner, _) in files {
            if state.status != GameStatus::Ongoing {
                break;
            }
            let ranks: Vec<_> = if owner == Player::White {
                (0..8).collect()
            } else {
                (0..8).rev().collect()
            };
            let ring: Vec<_> = ranks
                .into_iter()
                .map(|rank| Square::new(file, rank).unwrap())
                .collect();
            self.rotate_conveyor(state, &ring, true, events);
            self.resolve_missing_king(state, events);
        }
    }
}

use super::*;

fn candidates(engine: &Engine, state: &GameState, actor: Player, id: &str) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| match id {
            crate::EVASION => {
                p.owner == actor
                    && p.kind != PieceKind::Football
                    && !state.card_runtime.evasion_piece_ids.contains(&p.id)
            }
            crate::ROYAL_SHIELD => {
                p.owner == actor && !state.card_runtime.shielded_piece_ids.contains(&p.id)
            }
            crate::FANATICAL_RITUAL => {
                p.owner == actor.opponent()
                    && !engine.is_royal_piece(state, *p)
                    && p.kind != PieceKind::Colossus
            }
            _ => false,
        })
        .collect()
}

pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    if *target != ActionTarget::None {
        return Err(EngineError::Invariant(
            "random status takes no target".into(),
        ));
    }
    let id = state
        .cards
        .iter()
        .find(|c| c.instance_id == instance_id)
        .map(|c| c.definition_id.clone())
        .ok_or_else(|| EngineError::Invariant("missing random status card".into()))?;
    let pool = candidates(engine, state, actor, &id);
    if pool.is_empty() {
        return Err(EngineError::Invariant("no random status candidate".into()));
    }
    let (roll, index) = state.rng_mut().web_choice_index(pool.len());
    events.push(Event::RandomSample {
        purpose: format!("card.v19.{id}.target"),
        roll_53: roll,
    });
    let (square, piece) = pool[index.expect("nonempty candidates")];
    match id.as_str() {
        crate::EVASION => {
            state.card_runtime.evasion_piece_ids.insert(piece.id);
        }
        crate::ROYAL_SHIELD => {
            state.card_runtime.shielded_piece_ids.insert(piece.id);
        }
        crate::FANATICAL_RITUAL => {
            engine.transform_current_piece(state, square, PieceKind::Fanatic, true)?;
        }
        _ => return Err(EngineError::Invariant("unknown random status".into())),
    }
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    out: &mut Vec<Action>,
) {
    if !candidates(engine, state, state.turn, &card.definition_id).is_empty() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

pub(super) fn eligibility(_: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    CardEligibility::Eligible
}

use super::transform_support::{self, TransformSource, TransformSpec};
use super::*;
pub(super) const ID: &str = crate::REAPER;
const SPEC: TransformSpec = TransformSpec {
    source: TransformSource::Kind(PieceKind::Queen),
    kind: PieceKind::Reaper,
    allow_royal: true,
};
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, piece) = piece_target(state, target)?;
    if !SPEC.accepts(state, actor, piece) {
        return Err(EngineError::Invariant(
            "invalid piece transformation target".into(),
        ));
    }
    engine.transform_current_piece(state, square, SPEC.kind, true)?;
    Ok(())
}
pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    out: &mut Vec<Action>,
) {
    transform_support::actions(engine, state, card, SPEC, out);
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    transform_support::eligibility(state, owner, SPEC)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

impl Engine {
    pub(in crate::engine) fn resolve_current_reaper_deaths(
        &self,
        state: &mut GameState,
        deaths: &[(Square, Piece)],
        captured_by: Player,
        events: &mut Vec<Event>,
    ) {
        if state.status != GameStatus::Ongoing {
            return;
        }
        let dead: BTreeSet<_> = deaths.iter().map(|(_, p)| p.id).collect();
        let mut winners = BTreeSet::new();
        for (death, victim) in deaths {
            if captured_by != victim.owner.opponent() {
                continue;
            }
            let mut witnesses: Vec<_> = state
                .board
                .iter()
                .filter(|(s, p)| {
                    p.owner == victim.owner
                        && self.piece_has_ability(state, *p, PieceKind::Reaper)
                        && !dead.contains(&p.id)
                        && s.file()
                            .abs_diff(death.file())
                            .max(s.rank().abs_diff(death.rank()))
                            == 1
                })
                .collect();
            witnesses.sort_by_key(|(s, p)| (s.to_string(), p.id));
            for (_, reaper) in witnesses {
                let souls = state
                    .card_runtime
                    .reaper_souls
                    .entry(reaper.id)
                    .or_default();
                *souls = souls.saturating_add(1).min(2);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("reaper-soul:{}:{}/2", reaper.id, *souls),
                });
                if *souls < 2 || !winners.insert(reaper.owner) {
                    continue;
                }
                let target = state
                    .board
                    .iter()
                    .filter(|(_, p)| {
                        p.owner == reaper.owner.opponent() && self.is_royal_piece(state, *p)
                    })
                    .min_by_key(|(s, _)| s.to_string());
                if let Some((square, king)) = target {
                    state.board.set(square, None);
                    self.clear_removed_piece_statuses(state, king.id);
                    if let Some((from, p)) = state.board.find_by_id(reaper.id) {
                        state.board.set(from, None);
                        state.board.set(square, Some(p));
                        state.card_runtime.moved_piece_ids.insert(p.id);
                    }
                    events.push(Event::PieceRemoved {
                        actor: reaper.owner,
                        piece: king,
                        at: square,
                        cause: "reaper-execution".into(),
                    });
                }
            }
        }
        if winners.len() == 2 {
            state.status = GameStatus::Draw {
                reason: DrawReason::CardEffect("simultaneous-special-victory".into()),
            };
        } else if let Some(winner) = winners.into_iter().next() {
            state.turn = winner;
            state.status = GameStatus::Won {
                winner,
                reason: WinReason::CardEffect("reaper-two-souls".into()),
            };
        }
    }
}

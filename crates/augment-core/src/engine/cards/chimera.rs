use super::*;
pub(super) const ID: &str = crate::CHIMERA;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn valid(state: &GameState, owner: Player, piece: Piece) -> bool {
    piece.owner == owner
        && crate::latest::latest_major_kind(piece.kind)
        && !royal(state, piece)
        && piece.kind != PieceKind::Merchant
}
fn play(
    _: &Engine,
    state: &mut GameState,
    _: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (_, piece) = piece_target(state, target)?;
    state.card_runtime.chimera_piece_ids.insert(piece.id);
    state.card_runtime.moved_piece_ids.insert(piece.id);
    Ok(())
}
fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, piece) in web_ordered_pieces(state) {
        if valid(state, card.owner, piece) {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| valid(state, owner, piece)),
        "아군 비왕실 메이저 기물이 필요합니다.",
    )
}

impl Engine {
    pub(in crate::engine) fn resolve_current_chimera(
        &self,
        state: &mut GameState,
        piece_id: u32,
        events: &mut Vec<Event>,
    ) {
        let Some((square, mut piece)) = state.board.find_by_id(piece_id) else {
            return;
        };
        let pool: Vec<_> = super::chess_n_pow_30::TRICKSTER_POOL
            .iter()
            .copied()
            .filter(|kind| crate::latest::latest_major_kind(*kind))
            .collect();
        let (roll, index) = state.rng_mut().web_choice_index(pool.len());
        events.push(Event::RandomSample {
            purpose: "chimera".into(),
            roll_53: roll,
        });
        let kind = pool[index.expect("current chimera pool is nonempty")];
        state.card_runtime.shielded_piece_ids.remove(&piece_id);
        state.card_runtime.windmill_modes.remove(&piece_id);
        state.card_runtime.log_statuses.remove(&piece_id);
        state.card_runtime.wizard_mana.remove(&piece_id);
        state.card_runtime.reaper_souls.remove(&piece_id);
        state.card_runtime.bear_retaliations.remove(&piece_id);
        state
            .card_runtime
            .herald_jump_locked_until
            .remove(&piece_id);
        state.card_runtime.piece_status.remove(&piece_id);
        state.card_runtime.large_missing_cells.remove(&piece_id);
        state.card_runtime.trickster_move_kinds.remove(&piece_id);
        piece.kind = kind;
        state.board.set(square, Some(piece));
        match kind {
            PieceKind::Windmill => {
                state
                    .card_runtime
                    .windmill_modes
                    .insert(piece_id, WindmillMode::Bishop);
            }
            PieceKind::Reaper => {
                state.card_runtime.reaper_souls.insert(piece_id, 0);
            }
            PieceKind::Bear | PieceKind::Hedgehog => {
                state.card_runtime.bear_retaliations.insert(piece_id, 2);
            }
            _ => {}
        }
    }
}

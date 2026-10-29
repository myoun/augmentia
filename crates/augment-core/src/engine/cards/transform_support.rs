//! Shared selection and identity-preserving transformation for current piece cards.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum TransformSource {
    Kind(PieceKind),
    Minor,
}

#[derive(Clone, Copy)]
pub(super) struct TransformSpec {
    pub source: TransformSource,
    pub kind: PieceKind,
    pub allow_royal: bool,
}

impl TransformSpec {
    pub(super) fn accepts(self, state: &GameState, owner: Player, piece: Piece) -> bool {
        piece.owner == owner
            && piece.kind != PieceKind::Monster
            && (self.allow_royal || !royal(state, piece))
            && match self.source {
                TransformSource::Kind(kind) => piece.kind == kind,
                TransformSource::Minor => crate::latest::latest_minor_kind(piece.kind),
            }
    }
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    spec: TransformSpec,
    out: &mut Vec<Action>,
) {
    for (square, _) in state
        .board
        .iter()
        .filter(|(_, p)| spec.accepts(state, card.owner, *p))
    {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        });
    }
}

pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    spec: TransformSpec,
) -> CardEligibility {
    require(
        state
            .board
            .iter()
            .any(|(_, p)| spec.accepts(state, owner, p)),
        "변환할 아군 기물이 필요합니다.",
    )
}

impl Engine {
    pub(in crate::engine) fn transform_current_piece(
        &self,
        state: &mut GameState,
        square: Square,
        kind: PieceKind,
        mark_moved: bool,
    ) -> Result<Piece, EngineError> {
        let mut piece = state
            .board
            .get(square)
            .ok_or_else(|| EngineError::Invariant("missing transform target".into()))?;
        let changed = piece.kind != kind;
        if changed {
            self.clear_piece_kind_runtime(state, piece.id);
            state.card_runtime.trickster_move_kinds.remove(&piece.id);
            state.card_runtime.idol_used_turn.remove(&piece.id);
            state.card_runtime.idol_rest_piece_ids.remove(&piece.id);
            state.rule_runtime.beacon_due.remove(&piece.id);
        }
        piece.kind = kind;
        state.board.set(square, Some(piece));
        if mark_moved {
            state.card_runtime.moved_piece_ids.insert(piece.id);
        }
        if changed {
            self.clear_queens_gambit_protection(state, piece.id);
            self.mark_fresh_no_capture(state, piece);
            state
                .rule_runtime
                .piece_fresh_until
                .insert(piece.id, rule_halfmove(state) + 2);
        }
        match kind {
            PieceKind::Wizard => {
                state.card_runtime.wizard_mana.insert(piece.id, 0);
            }
            PieceKind::Reaper => {
                state.card_runtime.reaper_souls.insert(piece.id, 0);
            }
            PieceKind::Windmill => {
                state
                    .card_runtime
                    .windmill_modes
                    .insert(piece.id, WindmillMode::Bishop);
            }
            PieceKind::Beacon => {
                state.rule_runtime.beacon_due.remove(&piece.id);
            }
            _ => {}
        }
        self.resolve_gomoku_victory(state);
        Ok(piece)
    }
}

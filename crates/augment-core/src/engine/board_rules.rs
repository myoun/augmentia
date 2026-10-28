//! Shared board-rule settlement.
use super::*;
use crate::*;
impl Engine {
    pub(super) fn resolve_rule_after_move(
        &self,
        state: &mut GameState,
        piece: Piece,
        movement: MoveAction,
        captured: Option<(Square, Piece)>,
        events: &mut Vec<Event>,
    ) {
        let actor = piece.owner;

        if state.card_runtime.macho_chess
            && state.status == GameStatus::Ongoing
            && let Some((square, royal)) = state.board.find_by_id(piece.id)
            && self.is_royal_piece(state, royal)
            && square.rank() == if actor == Player::White { 7 } else { 0 }
        {
            state.status = GameStatus::Won {
                winner: actor,
                reason: WinReason::CardEffect(MACHO_CHESS.to_owned()),
            };
            return;
        }

        // Black Hole check
        if state.card_runtime.black_holes.contains(&movement.to) {
            if let Some((sq, p)) = state.board.find_by_id(piece.id) {
                self.remove_piece_for_card(state, sq, p, actor, BLACK_HOLE, events);
            }
        }

        // Bombs check
        if state.card_runtime.rule_bombs.remove(&movement.to) {
            let victims = state
                .board
                .iter()
                .filter(|(sq, _)| {
                    sq.file() == movement.to.file() || sq.rank() == movement.to.rank()
                })
                .collect::<Vec<_>>();
            for (sq, victim) in victims {
                self.remove_piece_for_card(state, sq, victim, actor, RULE_BOMBS, events);
            }
        }

        // Transcendence: Pawn -> Knight -> Rook -> Queen upon direct capture
        if captured.is_some() && state.card_runtime.transcendence {
            if let Some((sq, mut p)) = state.board.find_by_id(piece.id) {
                self.clear_queens_gambit_protection(state, p.id);
                p.kind = match p.kind {
                    PieceKind::Pawn => PieceKind::Knight,
                    PieceKind::Knight => PieceKind::Rook,
                    PieceKind::Rook => PieceKind::Queen,
                    other => other,
                };
                state.board.set(sq, Some(p));
            }
        }

        // Crown check
        if let Some(crown_sq) = state.card_runtime.crown_pos {
            if movement.to == crown_sq {
                state.card_runtime.crown_pos = None;
                state.card_runtime.crown_holder = Some((actor, 1));
            }
        } else if let Some((holder, count)) = state.card_runtime.crown_holder {
            if holder == actor {
                let new_count = count.saturating_add(1);
                if new_count >= 10 {
                    state.status = GameStatus::Won {
                        winner: actor,
                        reason: WinReason::CardEffect("crown".to_owned()),
                    };
                } else {
                    state.card_runtime.crown_holder = Some((holder, new_count));
                }
            }
        }

        // Camouflage Color
        if state.card_runtime.camouflage_color {
            let is_light = (movement.to.file() + movement.to.rank()) % 2 != 0;
            if (actor == Player::White && is_light) || (actor == Player::Black && !is_light) {
                state.card_runtime.stealth_piece_ids.insert(piece.id);
            } else {
                state.card_runtime.stealth_piece_ids.remove(&piece.id);
            }
        }
    }
}

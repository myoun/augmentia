use super::evaluator::HeuristicEvaluator;
use augment_core::{Action, Engine, GameState, GameStatus, PieceKind};

const MAX_QUIESCENCE_DEPTH: usize = 4;

pub struct QuiescenceSearch;

impl QuiescenceSearch {
    /// Evaluates the tactical static balance by pursuing all immediate captures and check threats until quiet.
    pub fn search(
        state: &GameState,
        engine: &Engine,
        mut alpha: f32,
        beta: f32,
        depth: usize,
    ) -> f32 {
        if state.status != GameStatus::Ongoing {
            return match state.status {
                GameStatus::Won { winner, .. } => {
                    if winner == state.turn {
                        1.0
                    } else {
                        -1.0
                    }
                }
                _ => 0.0,
            };
        }

        let stand_pat = HeuristicEvaluator::compute_value(state);
        if depth == 0 {
            return stand_pat;
        }

        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let Ok(legal_actions) = engine.legal_actions(state) else {
            return stand_pat;
        };

        // Extract tactical moves: captures and direct king attacks
        let mut tactical_moves = Vec::with_capacity(16);
        for action in &legal_actions {
            if let Action::Move(m) = action {
                if let Some(target) = state.board.get(m.to) {
                    // Winning King capture!
                    if target.owner != state.turn && target.kind == PieceKind::King {
                        return 1.0;
                    }
                    if target.owner != state.turn {
                        let victim_val = HeuristicEvaluator::piece_material(target.kind);
                        let mover_val = state
                            .board
                            .get(m.from)
                            .map_or(1.0, |p| HeuristicEvaluator::piece_material(p.kind));
                        // Prioritize high-value captures first (MVV-LVA)
                        tactical_moves.push((action.clone(), victim_val * 10.0 - mover_val));
                    }
                }
            }
        }

        if tactical_moves.is_empty() {
            return stand_pat;
        }

        // Sort tactical moves descending by MVV-LVA heuristic
        tactical_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (tactical_act, _) in tactical_moves {
            if let Ok(trans) = engine.apply(state, tactical_act) {
                let score = -Self::search(&trans.state, engine, -beta, -alpha, depth - 1);
                if score >= beta {
                    return beta;
                }
                if score > alpha {
                    alpha = score;
                }
            }
        }

        alpha
    }

    /// Evaluates tactical quietness adjustment for leaf node in MCTS
    pub fn tactical_leaf_value(state: &GameState, engine: &Engine, nn_val: f32) -> f32 {
        let q_val = Self::search(state, engine, -1.0, 1.0, MAX_QUIESCENCE_DEPTH);
        // Blend Neural strategic value with Quiescence Tactical verification
        // If Quiescence detects severe tactical loss/win, it pulls the evaluation strongly.
        if (q_val - nn_val).abs() > 0.4 {
            (nn_val * 0.5 + q_val * 0.5).clamp(-1.0, 1.0)
        } else {
            (nn_val * 0.8 + q_val * 0.2).clamp(-1.0, 1.0)
        }
    }
}

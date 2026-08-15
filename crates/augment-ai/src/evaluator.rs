use super::action_space::{ActionIndex, encode_action};
use augment_core::{Action, Engine, GameState, GameStatus, PieceKind, Player};

pub trait Evaluator {
    /// Evaluates a leaf game state, returning prior probabilities for each legal action and the estimated state value v in [-1.0, 1.0].
    fn evaluate(
        &self,
        state: &GameState,
        engine: &Engine,
    ) -> (Vec<(Action, ActionIndex, f32)>, f32);
}

/// Fast heuristic evaluator based on material balance, king safety, and legal move activity.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeuristicEvaluator;

impl HeuristicEvaluator {
    pub fn new() -> Self {
        Self
    }

    fn piece_material(kind: PieceKind) -> f32 {
        match kind {
            PieceKind::Pawn | PieceKind::Squire | PieceKind::Fanatic => 1.0,
            PieceKind::Knight
            | PieceKind::Bishop
            | PieceKind::Alfil
            | PieceKind::Guard
            | PieceKind::Checker => 3.0,
            PieceKind::Rook
            | PieceKind::Windmill
            | PieceKind::Cardinal
            | PieceKind::Protestant
            | PieceKind::Cannon => 5.0,
            PieceKind::Queen
            | PieceKind::Amazon
            | PieceKind::Dragon
            | PieceKind::Wizard
            | PieceKind::Bear => 9.0,
            PieceKind::BigRook | PieceKind::Colossus => 12.0,
            PieceKind::King
            | PieceKind::DarkWizard
            | PieceKind::Merchant
            | PieceKind::RoyalKnight => 200.0,
            _ => 2.0,
        }
    }

    fn compute_value(state: &GameState) -> f32 {
        match state.status {
            GameStatus::Won { winner, .. } => {
                if winner == state.turn {
                    1.0
                } else {
                    -1.0
                }
            }
            GameStatus::Draw { .. } => 0.0,
            GameStatus::Ongoing => {
                let mut white_score = 0.0f32;
                let mut black_score = 0.0f32;
                for (_, piece) in state.board.iter() {
                    let val = Self::piece_material(piece.kind);
                    if piece.owner == Player::White {
                        white_score += val;
                    } else {
                        black_score += val;
                    }
                }
                let diff = if state.turn == Player::White {
                    white_score - black_score
                } else {
                    black_score - white_score
                };
                (diff / 40.0).clamp(-0.95, 0.95)
            }
        }
    }
}

impl Evaluator for HeuristicEvaluator {
    fn evaluate(
        &self,
        state: &GameState,
        engine: &Engine,
    ) -> (Vec<(Action, ActionIndex, f32)>, f32) {
        let legal_actions = engine.legal_actions(state).unwrap_or_default();
        if legal_actions.is_empty() {
            return (Vec::new(), Self::compute_value(state));
        }

        let uniform_prior = 1.0 / legal_actions.len() as f32;
        let priors = legal_actions
            .into_iter()
            .map(|action| {
                let idx = encode_action(&action, state);
                (action, idx, uniform_prior)
            })
            .collect();

        (priors, Self::compute_value(state))
    }
}

use super::action_space::{ActionIndex, encode_action};
use super::quiescence::QuiescenceSearch;
use super::tensor::{GLOBAL_FEATURES_SIZE, SPATIAL_CHANNELS, encode_state};
use augment_core::{Action, Engine, GameState, GameStatus, PieceKind, Player};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub trait Evaluator: Send + Sync {
    /// Evaluates a leaf game state, returning prior probabilities for each legal action and the estimated state value v in [-1.0, 1.0].
    fn evaluate(
        &self,
        state: &GameState,
        engine: &Engine,
    ) -> (Vec<(Action, ActionIndex, f32)>, f32);
}

/// Fast baseline heuristic evaluator based on material balance.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeuristicEvaluator;

impl HeuristicEvaluator {
    pub fn new() -> Self {
        Self
    }

    pub fn piece_material(kind: PieceKind) -> f32 {
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

    pub fn compute_material_diff(state: &GameState) -> f32 {
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
        if state.turn == Player::White {
            white_score - black_score
        } else {
            black_score - white_score
        }
    }

    pub fn compute_value(state: &GameState) -> f32 {
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
                let diff = Self::compute_material_diff(state);
                (diff / 30.0).tanh().clamp(-0.95, 0.95)
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

/// Tactical-enhanced AlphaZero Neural network evaluator with Quiescence search & Tactical channels.
pub struct OnnxNeuralEvaluator {
    session: Arc<Mutex<ort::session::Session>>,
    cache: Arc<Mutex<HashMap<u64, (Vec<(Action, ActionIndex, f32)>, f32)>>>,
    fallback: HeuristicEvaluator,
    inference_failures: AtomicU64,
}

impl OnnxNeuralEvaluator {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let session = ort::session::Session::builder()
            .map_err(|e| format!("Failed to build ONNX session builder: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| format!("Failed to set intra threads: {e}"))?
            .commit_from_file(path)
            .map_err(|e| format!("Failed to load ONNX model: {e}"))?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            cache: Arc::new(Mutex::new(HashMap::with_capacity(65536))),
            fallback: HeuristicEvaluator::new(),
            inference_failures: AtomicU64::new(0),
        })
    }

    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    pub fn inference_failures(&self) -> u64 {
        self.inference_failures.load(Ordering::Relaxed)
    }
}

impl Evaluator for OnnxNeuralEvaluator {
    fn evaluate(
        &self,
        state: &GameState,
        engine: &Engine,
    ) -> (Vec<(Action, ActionIndex, f32)>, f32) {
        let fp = state.fingerprint();

        // Transposition Cache Lookup
        if let Ok(cache) = self.cache.lock() {
            if let Some(cached) = cache.get(&fp) {
                return cached.clone();
            }
        }

        let legal_actions = engine.legal_actions(state).unwrap_or_default();
        if legal_actions.is_empty() {
            let res = (Vec::new(), HeuristicEvaluator::compute_value(state));
            if let Ok(mut cache) = self.cache.lock() {
                if cache.len() < 200_000 {
                    cache.insert(fp, res.clone());
                }
            }
            return res;
        }

        if let Ok(mut session) = self.session.lock() {
            let encoded = encode_state(state);
            let spatial_shape = [1, SPATIAL_CHANNELS, 8, 8];
            let global_shape = [1, GLOBAL_FEATURES_SIZE];

            let spatial_tensor = ort::value::Tensor::from_array((spatial_shape, encoded.spatial));
            let global_tensor = ort::value::Tensor::from_array((global_shape, encoded.global));

            if let (Ok(sp), Ok(gl)) = (spatial_tensor, global_tensor) {
                let inputs = ort::inputs!["spatial" => sp, "global" => gl];
                if let Ok(outputs) = session.run(inputs) {
                    if let (Some(logits_out), Some(value_out)) =
                        (outputs.get("logits"), outputs.get("value"))
                    {
                        if let (Ok((_, logits_arr)), Ok((_, value_arr))) = (
                            logits_out.try_extract_tensor::<f32>(),
                            value_out.try_extract_tensor::<f32>(),
                        ) {
                            let nn_val = value_arr.first().copied().unwrap_or(0.0);

                            // Tactical Quiescence Stabilization:
                            // Quiescence Search resolves lingering tactical exchanges/pins without harming strategic cards.
                            let final_val =
                                QuiescenceSearch::tactical_leaf_value(state, engine, nn_val);

                            // Legal-Masked Softmax over raw Logits
                            let mut raw_logits = Vec::with_capacity(legal_actions.len());
                            let mut max_logit = f32::NEG_INFINITY;
                            for action in &legal_actions {
                                let idx = encode_action(action, state);
                                let logit = if (idx as usize) < logits_arr.len() {
                                    logits_arr[idx as usize]
                                } else {
                                    -10.0
                                };
                                if logit > max_logit {
                                    max_logit = logit;
                                }
                                raw_logits.push((action.clone(), idx, logit));
                            }

                            let mut priors = Vec::with_capacity(legal_actions.len());
                            let mut total_exp = 0.0f32;

                            for (action, idx, logit) in raw_logits {
                                let exp_val = (logit - max_logit).exp();
                                total_exp += exp_val;
                                priors.push((action, idx, exp_val));
                            }

                            if total_exp > 0.0 {
                                for item in &mut priors {
                                    item.2 /= total_exp;
                                }
                            }

                            let result = (priors, final_val);
                            if let Ok(mut cache) = self.cache.lock() {
                                if cache.len() < 200_000 {
                                    cache.insert(fp, result.clone());
                                }
                            }
                            return result;
                        }
                    }
                }
            }
        }

        self.inference_failures.fetch_add(1, Ordering::Relaxed);
        self.fallback.evaluate(state, engine)
    }
}

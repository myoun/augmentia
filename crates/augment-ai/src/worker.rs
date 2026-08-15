use super::action_space::{ActionIndex, ActionMask, generate_action_mask};
use super::evaluator::Evaluator;
use super::mcts::{MctsConfig, MctsTree};
use super::tensor::{EncodedState, encode_state};
use augment_core::{Engine, GameStatus, Mode, Player};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfPlayStep {
    pub state_encoded: EncodedState,
    pub action_mask: ActionMask,
    pub policy_target: Vec<(ActionIndex, f32)>,
    pub player: Player,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfPlayGameResult {
    pub steps: Vec<SelfPlayStep>,
    pub winner: Option<Player>,
    pub total_plies: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingSample {
    pub spatial: Vec<f32>,
    pub global: Vec<f32>,
    pub mask: Vec<usize>,
    pub policy: Vec<(ActionIndex, f32)>,
    pub value: f32,
}

impl SelfPlayGameResult {
    pub fn to_training_samples(&self) -> Vec<TrainingSample> {
        self.steps
            .iter()
            .map(|step| {
                let val = match self.winner {
                    Some(w) => {
                        if w == step.player {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    None => 0.0,
                };
                let mask_indices = (0..super::action_space::ACTION_SPACE_SIZE)
                    .filter(|&i| step.action_mask.is_set(i))
                    .collect();
                TrainingSample {
                    spatial: step.state_encoded.spatial.clone(),
                    global: step.state_encoded.global.clone(),
                    mask: mask_indices,
                    policy: step.policy_target.clone(),
                    value: val,
                }
            })
            .collect()
    }
}

pub struct SelfPlayWorker;

impl SelfPlayWorker {
    pub fn play_game(
        engine: &Engine,
        evaluator: &dyn Evaluator,
        config: MctsConfig,
        seed: u64,
        max_plies: u32,
    ) -> SelfPlayGameResult {
        let mut state = engine.new_game(Mode::Standard, seed).unwrap();
        let mut tree = MctsTree::new(config);
        let mut steps = Vec::new();

        while state.status == GameStatus::Ongoing && state.ply < max_plies {
            let mask = generate_action_mask(&state, engine);
            let encoded = encode_state(&state);
            let current_player = state.turn;

            let search_result = tree.search(&state, engine, evaluator);
            let Some((chosen_action, _chosen_idx, visits)) = search_result else {
                break;
            };

            let total_visits: u32 = visits.iter().map(|(_, _, v)| *v).sum();
            let policy_target: Vec<(ActionIndex, f32)> = visits
                .into_iter()
                .map(|(_, idx, v)| (idx, v as f32 / total_visits.max(1) as f32))
                .collect();

            steps.push(SelfPlayStep {
                state_encoded: encoded,
                action_mask: mask,
                policy_target,
                player: current_player,
            });

            if let Ok(transition) = engine.apply(&state, chosen_action) {
                state = transition.state;
            } else {
                break;
            }
        }

        let winner = match state.status {
            GameStatus::Won { winner, .. } => Some(winner),
            _ => None,
        };

        SelfPlayGameResult {
            steps,
            winner,
            total_plies: state.ply,
        }
    }
}

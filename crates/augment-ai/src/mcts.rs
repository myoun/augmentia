use super::action_space::ActionIndex;
use super::evaluator::Evaluator;
use augment_core::{Action, Engine, GameState, GameStatus, PieceKind, Player};
use rand::distributions::WeightedIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution as RandDistrDistribution, Gamma};

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub c_puct: f32,
    pub num_simulations: usize,
    /// Applied only when sampling a played self-play action from root visits.
    /// It does not affect PUCT selection.
    pub temperature: f32,
}

pub type SearchResult = (Action, ActionIndex, Vec<(Action, ActionIndex, u32)>);

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_puct: 2.0,
            num_simulations: 200,
            temperature: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RootNoiseConfig {
    pub alpha: f32,
    pub epsilon: f32,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct RootActionDiagnostics {
    pub action: Action,
    pub action_index: ActionIndex,
    pub raw_prior: f32,
    pub root_noise: Option<f32>,
    pub effective_prior: f32,
    pub visits: u32,
    pub first_visit_simulation: Option<usize>,
    /// Q converted to the root player's perspective.
    pub q_root: f32,
}

#[derive(Clone, Debug)]
pub struct SearchDiagnostics {
    pub root_raw_value: f32,
    pub root_legal_actions: usize,
    pub root_visited_action_count: usize,
    pub root_visit_entropy: f32,
    pub top1_visit_fraction: f32,
    pub effective_branching_factor: f32,
    pub low_prior_visit_rate: f32,
    pub max_search_depth: usize,
    pub root_actions: Vec<RootActionDiagnostics>,
}

#[derive(Clone, Debug)]
pub struct MctsNode {
    pub state: GameState,
    pub parent: Option<usize>,
    pub action_from_parent: Option<Action>,
    pub action_index: ActionIndex,
    pub player: Player,
    pub visit_count: u32,
    pub value_sum: f32,
    /// The PUCT prior. Root children use the noisy effective prior during self-play.
    pub prior_prob: f32,
    /// The neural-network prior before root Dirichlet mixing.
    pub raw_prior_prob: f32,
    /// Root Dirichlet draw, only present on root children when noise is enabled.
    pub root_noise: Option<f32>,
    pub first_visit_simulation: Option<usize>,
    pub depth: usize,
    pub children: Vec<usize>,
    pub is_terminal: bool,
}

impl MctsNode {
    pub fn new(
        state: GameState,
        parent: Option<usize>,
        action_from_parent: Option<Action>,
        action_index: ActionIndex,
        prior_prob: f32,
    ) -> Self {
        let is_terminal = state.status != GameStatus::Ongoing;
        let player = state.turn;
        Self {
            state,
            parent,
            action_from_parent,
            action_index,
            player,
            visit_count: 0,
            value_sum: 0.0,
            prior_prob,
            raw_prior_prob: prior_prob,
            root_noise: None,
            first_visit_simulation: None,
            depth: 0,
            children: Vec::new(),
            is_terminal,
        }
    }

    #[inline]
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}

pub struct MctsTree {
    pub nodes: Vec<MctsNode>,
    pub config: MctsConfig,
    pub last_diagnostics: Option<SearchDiagnostics>,
}

impl MctsTree {
    pub fn new(config: MctsConfig) -> Self {
        Self {
            nodes: Vec::with_capacity(1024),
            config,
            last_diagnostics: None,
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.last_diagnostics = None;
    }

    /// Samples a played self-play action from a completed root visit distribution.
    /// PUCT search itself is already complete when this runs.
    pub fn sample_action_from_visits(
        visits: &[(Action, ActionIndex, u32)],
        temperature: f32,
        seed: u64,
    ) -> Option<(Action, ActionIndex, u32)> {
        if visits.is_empty() {
            return None;
        }

        if temperature <= 1e-6 {
            return visits
                .iter()
                .max_by_key(|(_, _, count)| *count)
                .cloned();
        }

        let exponent = 1.0f64 / temperature as f64;
        let weights: Vec<f64> = visits
            .iter()
            .map(|(_, _, count)| (*count as f64).powf(exponent))
            .collect();

        let distribution = WeightedIndex::new(&weights).ok()?;
        let mut rng = StdRng::seed_from_u64(seed);
        visits.get(rng.sample(distribution)).cloned()
    }

    /// Fast O(1) Check: checks whether the opponent can capture the mover's King immediately.
    #[inline]
    fn leads_to_immediate_king_loss(state: &GameState, engine: &Engine, mover: Player) -> bool {
        if state.status != GameStatus::Ongoing || state.turn == mover {
            return false;
        }
        if let Ok(opp_actions) = engine.legal_actions(state) {
            for opp_act in &opp_actions {
                if let Action::Move(m) = opp_act {
                    if let Some(target_p) = state.board.get(m.to) {
                        if target_p.owner == mover && target_p.kind == PieceKind::King {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn sample_dirichlet(action_count: usize, config: RootNoiseConfig) -> Vec<f32> {
        if action_count == 0 || config.alpha <= 0.0 {
            return vec![0.0; action_count];
        }

        let gamma = Gamma::new(config.alpha as f64, 1.0).expect("valid Dirichlet alpha");
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut samples = Vec::with_capacity(action_count);
        let mut total = 0.0f64;
        for _ in 0..action_count {
            let draw = gamma.sample(&mut rng);
            total += draw;
            samples.push(draw);
        }

        if total <= f64::EPSILON {
            return vec![1.0 / action_count as f32; action_count];
        }

        samples
            .into_iter()
            .map(|draw| (draw / total) as f32)
            .collect()
    }

    pub fn search(
        &mut self,
        root_state: &GameState,
        engine: &Engine,
        evaluator: &dyn Evaluator,
    ) -> Option<SearchResult> {
        self.search_with_root_noise(root_state, engine, evaluator, None)
    }

    pub fn search_with_root_noise(
        &mut self,
        root_state: &GameState,
        engine: &Engine,
        evaluator: &dyn Evaluator,
        root_noise: Option<RootNoiseConfig>,
    ) -> Option<SearchResult> {
        self.clear();
        self.nodes
            .push(MctsNode::new(root_state.clone(), None, None, 0, 1.0));

        let (root_priors, root_value) = evaluator.evaluate(root_state, engine);
        if root_priors.is_empty() {
            return None;
        }
        let root_player = root_state.turn;
        let noise_values = root_noise.map(|cfg| Self::sample_dirichlet(root_priors.len(), cfg));
        let noise_epsilon = root_noise.map_or(0.0, |cfg| cfg.epsilon.clamp(0.0, 1.0));

        // Immediate king capture is terminally proven and may bypass MCTS.
        for (action, act_idx, raw_prior) in &root_priors {
            if let Action::Move(m) = action {
                if let Some(target_p) = root_state.board.get(m.to) {
                    if target_p.owner != root_player && target_p.kind == PieceKind::King {
                        let root_actions: Vec<RootActionDiagnostics> = root_priors
                            .iter()
                            .enumerate()
                            .map(|(i, (candidate_action, candidate_index, candidate_prior))| {
                                let eta = noise_values.as_ref().map(|draws| draws[i]);
                                let effective_prior = eta.map_or(*candidate_prior, |noise| {
                                    (1.0 - noise_epsilon) * *candidate_prior + noise_epsilon * noise
                                });
                                let visits = if *candidate_index == *act_idx { 1000 } else { 0 };
                                RootActionDiagnostics {
                                    action: candidate_action.clone(),
                                    action_index: *candidate_index,
                                    raw_prior: *candidate_prior,
                                    root_noise: eta,
                                    effective_prior,
                                    visits,
                                    first_visit_simulation: if visits > 0 { Some(0) } else { None },
                                    q_root: if visits > 0 { 1.0 } else { 0.0 },
                                }
                            })
                            .collect();
                        self.last_diagnostics = Some(SearchDiagnostics {
                            root_raw_value: root_value,
                            root_legal_actions: root_actions.len(),
                            root_visited_action_count: 1,
                            root_visit_entropy: 0.0,
                            top1_visit_fraction: 1.0,
                            effective_branching_factor: 1.0,
                            low_prior_visit_rate: if *raw_prior < 1e-4 { 1.0 } else { 0.0 },
                            max_search_depth: 1,
                            root_actions,
                        });
                        return Some((
                            action.clone(),
                            *act_idx,
                            vec![(action.clone(), *act_idx, 1000)],
                        ));
                    }
                }
            }
        }

        // Expand root with either raw NN priors (evaluation) or P' (self-play).
        for (position, (action, act_idx, raw_prior)) in root_priors.into_iter().enumerate() {
            if let Ok(transition) = engine.apply(root_state, action.clone()) {
                let eta = noise_values.as_ref().map(|draws| draws[position]);
                let effective_prior = match eta {
                    Some(noise) => (1.0 - noise_epsilon) * raw_prior + noise_epsilon * noise,
                    None => raw_prior,
                };
                let mut child = MctsNode::new(
                    transition.state,
                    Some(0),
                    Some(action),
                    act_idx,
                    effective_prior,
                );
                child.raw_prior_prob = raw_prior;
                child.root_noise = eta;
                child.depth = 1;
                let child_idx = self.nodes.len();
                self.nodes.push(child);
                self.nodes[0].children.push(child_idx);
            }
        }
        self.nodes[0].visit_count = 1;
        self.nodes[0].value_sum = root_value;

        for simulation in 1..=self.config.num_simulations {
            let mut current_idx = 0usize;
            let mut selected_root_child = None;

            // Selection.
            while !self.nodes[current_idx].children.is_empty()
                && !self.nodes[current_idx].is_terminal
            {
                let parent_idx = current_idx;
                current_idx = self.select_child(parent_idx);
                if parent_idx == 0 {
                    selected_root_child = Some(current_idx);
                }
            }

            if let Some(root_child_idx) = selected_root_child {
                if self.nodes[root_child_idx].first_visit_simulation.is_none() {
                    self.nodes[root_child_idx].first_visit_simulation = Some(simulation);
                }
            }

            // Expansion and evaluation.
            let is_terminal = self.nodes[current_idx].is_terminal;
            let leaf_player = self.nodes[current_idx].player;
            let leaf_value = if is_terminal {
                match self.nodes[current_idx].state.status {
                    GameStatus::Won { winner, .. } => {
                        if winner == leaf_player { 1.0 } else { -1.0 }
                    }
                    _ => 0.0,
                }
            } else {
                let parent_player = self.nodes[current_idx]
                    .parent
                    .map_or(leaf_player, |parent| self.nodes[parent].player);
                let node_state = self.nodes[current_idx].state.clone();

                if Self::leads_to_immediate_king_loss(&node_state, engine, parent_player) {
                    if leaf_player == parent_player { -1.0 } else { 1.0 }
                } else {
                    let (priors, val) = evaluator.evaluate(&node_state, engine);
                    let child_depth = self.nodes[current_idx].depth + 1;
                    for (action, act_idx, prior) in priors {
                        if let Ok(transition) = engine.apply(&node_state, action.clone()) {
                            let mut child = MctsNode::new(
                                transition.state,
                                Some(current_idx),
                                Some(action),
                                act_idx,
                                prior,
                            );
                            child.depth = child_depth;
                            let child_idx = self.nodes.len();
                            self.nodes.push(child);
                            self.nodes[current_idx].children.push(child_idx);
                        }
                    }
                    val
                }
            };

            // Backpropagation flips only when the player perspective changes.
            let mut backprop_idx = Some(current_idx);
            while let Some(idx) = backprop_idx {
                self.nodes[idx].visit_count += 1;
                let sign = if self.nodes[idx].player == leaf_player { 1.0 } else { -1.0 };
                self.nodes[idx].value_sum += leaf_value * sign;
                backprop_idx = self.nodes[idx].parent;
            }
        }

        let root_children = self.nodes[0].children.clone();
        if root_children.is_empty() {
            return None;
        }

        let mut visit_distribution = Vec::with_capacity(root_children.len());
        let mut diagnostics_actions = Vec::with_capacity(root_children.len());
        let mut total_visits = 0u32;
        let mut max_visits = 0u32;
        let mut visited_actions = 0usize;
        let mut low_prior_visits = 0u32;
        let mut max_depth = 0usize;

        for &child_idx in &root_children {
            let child = &self.nodes[child_idx];
            let action = child.action_from_parent.clone().expect("root child action");
            let visits = child.visit_count;
            total_visits += visits;
            max_visits = max_visits.max(visits);
            if visits > 0 {
                visited_actions += 1;
            }
            if child.raw_prior_prob < 1e-4 {
                low_prior_visits += visits;
            }
            max_depth = max_depth.max(child.depth);
            let q_root = if child.player == root_player {
                child.q_value()
            } else {
                -child.q_value()
            };

            diagnostics_actions.push(RootActionDiagnostics {
                action: action.clone(),
                action_index: child.action_index,
                raw_prior: child.raw_prior_prob,
                root_noise: child.root_noise,
                effective_prior: child.prior_prob,
                visits,
                first_visit_simulation: child.first_visit_simulation,
                q_root,
            });

            let is_suicide = Self::leads_to_immediate_king_loss(&child.state, engine, root_player);
            visit_distribution.push((action, child.action_index, visits, is_suicide));
        }

        for node in &self.nodes {
            max_depth = max_depth.max(node.depth);
        }

        let mut root_visit_entropy = 0.0f32;
        if total_visits > 0 {
            for action in &diagnostics_actions {
                if action.visits > 0 {
                    let p = action.visits as f32 / total_visits as f32;
                    root_visit_entropy -= p * p.ln();
                }
            }
        }
        let top1_visit_fraction = if total_visits > 0 {
            max_visits as f32 / total_visits as f32
        } else {
            0.0
        };

        self.last_diagnostics = Some(SearchDiagnostics {
            root_raw_value: root_value,
            root_legal_actions: diagnostics_actions.len(),
            root_visited_action_count: visited_actions,
            root_visit_entropy,
            top1_visit_fraction,
            effective_branching_factor: root_visit_entropy.exp(),
            low_prior_visit_rate: if total_visits > 0 {
                low_prior_visits as f32 / total_visits as f32
            } else {
                0.0
            },
            max_search_depth: max_depth,
            root_actions: diagnostics_actions,
        });

        let has_safe = visit_distribution.iter().any(|(_, _, _, is_suicide)| !*is_suicide);
        let best_entry = if has_safe {
            visit_distribution
                .iter()
                .filter(|(_, _, _, is_suicide)| !*is_suicide)
                .max_by_key(|(_, _, visits, _)| *visits)?
        } else {
            visit_distribution.iter().max_by_key(|(_, _, visits, _)| *visits)?
        };

        let best_action = best_entry.0.clone();
        let best_index = best_entry.1;
        let final_visits = visit_distribution
            .into_iter()
            .map(|(action, index, visits, _)| (action, index, visits))
            .collect();

        Some((best_action, best_index, final_visits))
    }

    fn select_child(&self, node_idx: usize) -> usize {
        let parent = &self.nodes[node_idx];
        let sqrt_parent = (parent.visit_count.max(1) as f32).sqrt();
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = parent.children[0];

        for &child_idx in &parent.children {
            let child = &self.nodes[child_idx];
            let q = if child.visit_count == 0 {
                0.0
            } else if parent.player == child.player {
                child.q_value()
            } else {
                -child.q_value()
            };
            let u = self.config.c_puct * child.prior_prob * sqrt_parent
                / (1.0 + child.visit_count as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }
        best_idx
    }
}

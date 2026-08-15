use super::action_space::ActionIndex;
use super::evaluator::Evaluator;
use augment_core::{Action, Engine, GameState, GameStatus, Player};

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub c_puct: f32,
    pub num_simulations: usize,
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

#[derive(Clone, Debug)]
pub struct MctsNode {
    pub state: GameState,
    pub parent: Option<usize>,
    pub action_from_parent: Option<Action>,
    pub action_index: ActionIndex,
    pub player: Player,
    pub visit_count: u32,
    pub value_sum: f32,
    pub prior_prob: f32,
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
}

impl MctsTree {
    pub fn new(config: MctsConfig) -> Self {
        Self {
            nodes: Vec::with_capacity(1024),
            config,
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    pub fn search(
        &mut self,
        root_state: &GameState,
        engine: &Engine,
        evaluator: &dyn Evaluator,
    ) -> Option<SearchResult> {
        self.clear();
        let root = MctsNode::new(root_state.clone(), None, None, 0, 1.0);
        self.nodes.push(root);

        let (priors, root_value) = evaluator.evaluate(root_state, engine);
        if priors.is_empty() {
            return None;
        }

        // Expand root
        let root_idx = 0;
        for (action, act_idx, prior) in priors {
            if let Ok(transition) = engine.apply(root_state, action.clone()) {
                let child = MctsNode::new(
                    transition.state,
                    Some(root_idx),
                    Some(action),
                    act_idx,
                    prior,
                );
                let child_idx = self.nodes.len();
                self.nodes.push(child);
                self.nodes[root_idx].children.push(child_idx);
            }
        }
        self.nodes[root_idx].visit_count = 1;
        self.nodes[root_idx].value_sum = root_value;

        // Simulation loop
        for _ in 0..self.config.num_simulations {
            let mut current_idx = root_idx;

            // 1. Selection
            while !self.nodes[current_idx].children.is_empty()
                && !self.nodes[current_idx].is_terminal
            {
                current_idx = self.select_child(current_idx);
            }

            // 2. Expansion & Evaluation
            let node = &self.nodes[current_idx];
            let (leaf_value, needs_expansion) = if node.is_terminal {
                let val = match node.state.status {
                    GameStatus::Won { winner, .. } => {
                        if winner == node.player {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    _ => 0.0,
                };
                (val, false)
            } else {
                let (priors, val) = evaluator.evaluate(&node.state, engine);
                if !priors.is_empty() && node.visit_count > 0 {
                    (val, true)
                } else {
                    (val, false)
                }
            };

            if needs_expansion {
                let node_state = self.nodes[current_idx].state.clone();
                let (priors, _) = evaluator.evaluate(&node_state, engine);
                for (action, act_idx, prior) in priors {
                    if let Ok(transition) = engine.apply(&node_state, action.clone()) {
                        let child = MctsNode::new(
                            transition.state,
                            Some(current_idx),
                            Some(action),
                            act_idx,
                            prior,
                        );
                        let child_idx = self.nodes.len();
                        self.nodes.push(child);
                        self.nodes[current_idx].children.push(child_idx);
                    }
                }
            }

            // 3. Backpropagation
            let leaf_player = self.nodes[current_idx].player;
            let mut backprop_idx = Some(current_idx);
            while let Some(idx) = backprop_idx {
                self.nodes[idx].visit_count += 1;
                let node_player = self.nodes[idx].player;
                let sign = if node_player == leaf_player {
                    1.0
                } else {
                    -1.0
                };
                self.nodes[idx].value_sum += leaf_value * sign;
                backprop_idx = self.nodes[idx].parent;
            }
        }

        // Collect root visits
        let root = &self.nodes[root_idx];
        if root.children.is_empty() {
            return None;
        }

        let mut visit_distribution: Vec<(Action, ActionIndex, u32)> = Vec::new();
        for &child_idx in &root.children {
            let child = &self.nodes[child_idx];
            if let Some(act) = &child.action_from_parent {
                visit_distribution.push((act.clone(), child.action_index, child.visit_count));
            }
        }

        let best = visit_distribution
            .iter()
            .max_by_key(|(_, _, visits)| *visits)?;
        Some((best.0.clone(), best.1, visit_distribution))
    }

    fn select_child(&self, node_idx: usize) -> usize {
        let parent = &self.nodes[node_idx];
        let total_parent_visits = parent.visit_count.max(1) as f32;
        let sqrt_parent = total_parent_visits.sqrt();

        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = parent.children[0];

        for &child_idx in &parent.children {
            let child = &self.nodes[child_idx];
            // Q is from parent's perspective
            let q = if child.visit_count == 0 {
                0.0
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

use crate::{EvaluationV3, LegalActionSetV3, V3OnnxRuntime};
use augment_core::{Action, Engine, GameState, GameStatus, Player};
use rand::distributions::WeightedIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution as RandDistribution, Gamma};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct MctsV3Config {
    pub c_puct: f32,
    pub num_simulations: usize,
    pub leaf_batch_size: usize,
    pub root_min_visits: u32,
    pub fpu_reduction: f32,
    pub virtual_loss: f32,
}

impl Default for MctsV3Config {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            num_simulations: 400,
            leaf_batch_size: 16,
            root_min_visits: 1,
            fpu_reduction: 0.25,
            virtual_loss: 1.0,
        }
    }
}

impl MctsV3Config {
    fn validate(self) -> Result<Self, MctsV3Error> {
        if !self.c_puct.is_finite() || self.c_puct <= 0.0 {
            return search_error("c_puct must be finite and positive");
        }
        if self.num_simulations == 0 || self.leaf_batch_size == 0 {
            return search_error("simulation count and leaf batch size must be positive");
        }
        if !self.fpu_reduction.is_finite()
            || self.fpu_reduction < 0.0
            || !self.virtual_loss.is_finite()
            || self.virtual_loss < 0.0
        {
            return search_error("FPU reduction and virtual loss must be finite and non-negative");
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RootNoiseConfigV3 {
    pub alpha: f32,
    pub epsilon: f32,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct RootActionDiagnosticsV3 {
    pub action: Action,
    pub action_index: usize,
    pub raw_prior: f32,
    pub root_noise: Option<f32>,
    pub effective_prior: f32,
    pub visits: u32,
    pub first_visit_simulation: Option<usize>,
    pub q_root: f32,
}

#[derive(Clone, Debug)]
pub struct SearchDiagnosticsV3 {
    pub root_raw_value: f32,
    pub root_legal_actions: usize,
    pub root_visited_action_count: usize,
    pub root_visit_coverage: f32,
    pub root_visit_entropy: f32,
    pub top1_visit_fraction: f32,
    pub effective_branching_factor: f32,
    pub low_prior_visit_rate: f32,
    pub max_search_depth: usize,
    pub completed_simulations: usize,
    pub evaluator_batches: usize,
    pub evaluated_leaves: usize,
    pub maximum_leaf_batch: usize,
    pub tree_reused: bool,
    pub transposition_candidate_duplicates: usize,
    pub root_min_visits_satisfied: bool,
    pub root_actions: Vec<RootActionDiagnosticsV3>,
}

#[derive(Clone, Debug)]
pub struct SearchResultV3 {
    pub action: Action,
    pub action_index: usize,
    pub visits: Vec<(Action, usize, u32)>,
}

#[derive(Debug)]
pub enum MctsV3Error {
    InvalidConfig(String),
    Evaluation(String),
    Engine(String),
    Invariant(String),
    TerminalRoot,
}

impl fmt::Display for MctsV3Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid V3 MCTS config: {message}"),
            Self::Evaluation(message) => write!(formatter, "V3 evaluator failed: {message}"),
            Self::Engine(message) => write!(formatter, "V3 MCTS engine error: {message}"),
            Self::Invariant(message) => write!(formatter, "V3 MCTS invariant failed: {message}"),
            Self::TerminalRoot => write!(formatter, "cannot search a terminal root"),
        }
    }
}

impl std::error::Error for MctsV3Error {}

pub trait BatchEvaluatorV3: Send + Sync {
    fn evaluate_batch_v3(
        &self,
        states: &[GameState],
        engine: &Engine,
    ) -> Result<Vec<EvaluationV3>, MctsV3Error>;
}

impl BatchEvaluatorV3 for V3OnnxRuntime {
    fn evaluate_batch_v3(
        &self,
        states: &[GameState],
        engine: &Engine,
    ) -> Result<Vec<EvaluationV3>, MctsV3Error> {
        V3OnnxRuntime::evaluate_batch(self, states, engine)
            .map_err(|error| MctsV3Error::Evaluation(error.to_string()))
    }
}

#[derive(Clone, Debug)]
struct MctsNodeV3 {
    state: Option<GameState>,
    parent: Option<usize>,
    action_from_parent: Option<Action>,
    action_index: usize,
    visit_count: u32,
    value_sum: f32,
    network_value: Option<f32>,
    virtual_visits: u32,
    prior: f32,
    raw_prior: f32,
    root_noise: Option<f32>,
    first_visit_simulation: Option<usize>,
    depth: usize,
    children: Vec<usize>,
    expanded: bool,
    terminal: bool,
}

impl MctsNodeV3 {
    fn root(state: GameState) -> Self {
        let terminal = state.status != GameStatus::Ongoing;
        Self {
            state: Some(state),
            parent: None,
            action_from_parent: None,
            action_index: 0,
            visit_count: 0,
            value_sum: 0.0,
            network_value: None,
            virtual_visits: 0,
            prior: 1.0,
            raw_prior: 1.0,
            root_noise: None,
            first_visit_simulation: None,
            depth: 0,
            children: Vec::new(),
            expanded: false,
            terminal,
        }
    }

    fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }

    fn state(&self) -> Result<&GameState, MctsV3Error> {
        self.state
            .as_ref()
            .ok_or_else(|| MctsV3Error::Invariant("node state was not materialized".to_owned()))
    }
}

#[derive(Clone, Debug)]
struct SelectedLeafV3 {
    node: usize,
    path: Vec<usize>,
}

pub struct MctsV3Tree {
    nodes: Vec<MctsNodeV3>,
    config: MctsV3Config,
    root: usize,
    pub last_diagnostics: Option<SearchDiagnosticsV3>,
}

impl MctsV3Tree {
    pub fn new(config: MctsV3Config) -> Result<Self, MctsV3Error> {
        Ok(Self {
            nodes: Vec::with_capacity(4096),
            config: config.validate()?,
            root: 0,
            last_diagnostics: None,
        })
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root = 0;
        self.last_diagnostics = None;
    }

    pub fn search(
        &mut self,
        root_state: &GameState,
        engine: &Engine,
        evaluator: &dyn BatchEvaluatorV3,
    ) -> Result<SearchResultV3, MctsV3Error> {
        self.search_with_root_noise(root_state, engine, evaluator, None)
    }

    pub fn search_with_root_noise(
        &mut self,
        root_state: &GameState,
        engine: &Engine,
        evaluator: &dyn BatchEvaluatorV3,
        root_noise: Option<RootNoiseConfigV3>,
    ) -> Result<SearchResultV3, MctsV3Error> {
        self.search_internal(root_state, engine, evaluator, root_noise, None)
    }

    pub fn search_for_duration(
        &mut self,
        root_state: &GameState,
        engine: &Engine,
        evaluator: &dyn BatchEvaluatorV3,
        duration: Duration,
    ) -> Result<SearchResultV3, MctsV3Error> {
        if duration.is_zero() {
            return search_error("wall-clock search duration must be positive");
        }
        self.search_internal(
            root_state,
            engine,
            evaluator,
            None,
            Some(Instant::now() + duration),
        )
    }

    fn search_internal(
        &mut self,
        root_state: &GameState,
        engine: &Engine,
        evaluator: &dyn BatchEvaluatorV3,
        root_noise: Option<RootNoiseConfigV3>,
        deadline: Option<Instant>,
    ) -> Result<SearchResultV3, MctsV3Error> {
        if root_state.status != GameStatus::Ongoing {
            return Err(MctsV3Error::TerminalRoot);
        }
        let tree_reused = self.prepare_root(root_state);
        let mut evaluator_batches = 0usize;
        let mut evaluated_leaves = 0usize;
        let mut maximum_leaf_batch = 0usize;
        let root_raw_value = if self.nodes[self.root].expanded {
            self.nodes[self.root].network_value.ok_or_else(|| {
                MctsV3Error::Invariant("expanded root has no network value".to_owned())
            })?
        } else {
            let root = self.nodes[self.root].state()?.clone();
            let mut evaluations =
                evaluator.evaluate_batch_v3(std::slice::from_ref(&root), engine)?;
            evaluator_batches += 1;
            evaluated_leaves += 1;
            maximum_leaf_batch = 1;
            let evaluation = evaluations
                .pop()
                .ok_or_else(|| MctsV3Error::Evaluation("root evaluation is missing".to_owned()))?;
            let value = evaluation.value;
            self.expand_node(self.root, evaluation, engine)?;
            value
        };
        self.apply_root_noise(root_noise)?;

        let mut completed = 0usize;
        while completed < self.config.num_simulations {
            if completed > 0 && deadline.is_some_and(|limit| Instant::now() >= limit) {
                break;
            }
            let requested = self
                .config
                .leaf_batch_size
                .min(self.config.num_simulations - completed);
            let mut selected = Vec::with_capacity(requested);
            let mut selected_nodes = HashSet::with_capacity(requested);
            for _ in 0..requested {
                let leaf = self.select_leaf(completed + selected.len() + 1, engine)?;
                if !selected_nodes.insert(leaf.node) {
                    self.release_virtual(&leaf.path)?;
                    break;
                }
                selected.push(leaf);
            }
            if selected.is_empty() {
                return invariant("leaf batching made no progress");
            }

            let pending: Vec<_> = selected
                .iter()
                .filter(|leaf| !self.nodes[leaf.node].terminal)
                .map(|leaf| self.nodes[leaf.node].state().cloned())
                .collect::<Result<Vec<_>, _>>()?;
            let evaluations = if pending.is_empty() {
                Vec::new()
            } else {
                evaluator_batches += 1;
                evaluated_leaves += pending.len();
                maximum_leaf_batch = maximum_leaf_batch.max(pending.len());
                match evaluator.evaluate_batch_v3(&pending, engine) {
                    Ok(evaluations) => evaluations,
                    Err(error) => {
                        for leaf in &selected {
                            self.release_virtual(&leaf.path)?;
                        }
                        return Err(error);
                    }
                }
            };
            if evaluations.len() != pending.len() {
                for leaf in &selected {
                    self.release_virtual(&leaf.path)?;
                }
                return Err(MctsV3Error::Evaluation(format!(
                    "expected {} leaf evaluations, got {}",
                    pending.len(),
                    evaluations.len()
                )));
            }
            let mut evaluations = evaluations.into_iter();
            for leaf in selected {
                self.release_virtual(&leaf.path)?;
                let leaf_player = self.nodes[leaf.node].state()?.turn;
                let value = if self.nodes[leaf.node].terminal {
                    terminal_value(self.nodes[leaf.node].state()?)?
                } else {
                    let evaluation = evaluations.next().ok_or_else(|| {
                        MctsV3Error::Evaluation("leaf evaluation is missing".to_owned())
                    })?;
                    let value = evaluation.value;
                    self.expand_node(leaf.node, evaluation, engine)?;
                    value
                };
                self.backup(&leaf.path, leaf_player, value)?;
                completed += 1;
            }
        }
        self.build_result(
            root_raw_value,
            completed,
            evaluator_batches,
            evaluated_leaves,
            maximum_leaf_batch,
            tree_reused,
        )
    }

    pub fn sample_action_from_visits(
        visits: &[(Action, usize, u32)],
        temperature: f32,
        seed: u64,
    ) -> Option<(Action, usize, u32)> {
        if visits.is_empty() || !temperature.is_finite() || temperature < 0.0 {
            return None;
        }
        if temperature <= 1e-6 {
            return visits
                .iter()
                .max_by_key(|(_, index, count)| (*count, usize::MAX - *index))
                .cloned();
        }
        let exponent = 1.0f64 / temperature as f64;
        let weights: Vec<_> = visits
            .iter()
            .map(|(_, _, count)| (*count as f64).powf(exponent))
            .collect();
        let distribution = WeightedIndex::new(weights).ok()?;
        let mut rng = StdRng::seed_from_u64(seed);
        visits.get(rng.sample(distribution)).cloned()
    }

    fn prepare_root(&mut self, state: &GameState) -> bool {
        if self.nodes.is_empty() {
            self.nodes.push(MctsNodeV3::root(state.clone()));
            self.root = 0;
            return false;
        }
        if self.nodes[self.root].state.as_ref() == Some(state) {
            return true;
        }
        let reusable = self.nodes[self.root]
            .children
            .iter()
            .copied()
            .find(|index| self.nodes[*index].state.as_ref() == Some(state));
        if let Some(index) = reusable {
            self.compact_subtree(index);
            true
        } else {
            self.clear();
            self.nodes.push(MctsNodeV3::root(state.clone()));
            false
        }
    }

    fn compact_subtree(&mut self, old_root: usize) {
        let mut queue = VecDeque::from([old_root]);
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        while let Some(index) = queue.pop_front() {
            if seen.insert(index) {
                order.push(index);
                queue.extend(self.nodes[index].children.iter().copied());
            }
        }
        let remap: HashMap<_, _> = order
            .iter()
            .enumerate()
            .map(|(new, old)| (*old, new))
            .collect();
        let mut nodes: Vec<_> = order.iter().map(|old| self.nodes[*old].clone()).collect();
        let mut depths = vec![0usize; nodes.len()];
        for (index, node) in nodes.iter_mut().enumerate() {
            node.parent = node.parent.and_then(|old| remap.get(&old).copied());
            node.children = node
                .children
                .iter()
                .filter_map(|old| remap.get(old).copied())
                .collect();
            let depth = if index == 0 {
                0
            } else {
                node.parent.map_or(0, |parent| depths[parent] + 1)
            };
            node.depth = depth;
            depths[index] = depth;
            node.virtual_visits = 0;
        }
        nodes[0].parent = None;
        self.nodes = nodes;
        self.root = 0;
    }

    fn expand_node(
        &mut self,
        node_index: usize,
        evaluation: EvaluationV3,
        engine: &Engine,
    ) -> Result<(), MctsV3Error> {
        if self.nodes[node_index].expanded {
            return invariant("attempted to expand one node twice");
        }
        if !evaluation.value.is_finite() || !(-1.0001..=1.0001).contains(&evaluation.value) {
            return Err(MctsV3Error::Evaluation(format!(
                "invalid value {}",
                evaluation.value
            )));
        }
        let network_value = evaluation.value;
        if evaluation.priors.is_empty() {
            return Err(MctsV3Error::Evaluation(
                "ongoing state has no policy priors".to_owned(),
            ));
        }
        let total: f32 = evaluation.priors.iter().map(|item| item.2).sum();
        if !total.is_finite()
            || (total - 1.0).abs() > 1e-4
            || evaluation
                .priors
                .iter()
                .any(|item| !item.2.is_finite() || item.2 < 0.0)
        {
            return Err(MctsV3Error::Evaluation(
                "policy priors must be finite, non-negative, and normalized".to_owned(),
            ));
        }
        let state = self.nodes[node_index].state()?.clone();
        let legal = LegalActionSetV3::from_state(&state, engine)
            .map_err(|error| MctsV3Error::Engine(error.to_string()))?;
        if legal.len() != evaluation.priors.len() {
            return Err(MctsV3Error::Evaluation(format!(
                "policy has {} actions but engine has {}",
                evaluation.priors.len(),
                legal.len()
            )));
        }
        let depth = self.nodes[node_index].depth + 1;
        let mut validated = Vec::with_capacity(evaluation.priors.len());
        for (position, (action, action_index, prior)) in evaluation.priors.into_iter().enumerate() {
            if action_index != position {
                return Err(MctsV3Error::Evaluation(format!(
                    "V3 local action index {action_index} must equal policy position {position}"
                )));
            }
            let expected = legal
                .decode_action(position, &state)
                .map_err(|error| MctsV3Error::Engine(error.to_string()))?;
            if action != expected {
                return Err(MctsV3Error::Evaluation(format!(
                    "policy action {position} differs from the engine legal action"
                )));
            }
            validated.push((action, action_index, prior));
        }
        let mut children = Vec::with_capacity(validated.len());
        for (action, action_index, prior) in validated {
            let child_index = self.nodes.len();
            self.nodes.push(MctsNodeV3 {
                terminal: false,
                state: None,
                parent: Some(node_index),
                action_from_parent: Some(action),
                action_index,
                visit_count: 0,
                value_sum: 0.0,
                network_value: None,
                virtual_visits: 0,
                prior,
                raw_prior: prior,
                root_noise: None,
                first_visit_simulation: None,
                depth,
                children: Vec::new(),
                expanded: false,
            });
            children.push(child_index);
        }
        self.nodes[node_index].children = children;
        self.nodes[node_index].network_value = Some(network_value);
        self.nodes[node_index].expanded = true;
        Ok(())
    }

    fn apply_root_noise(&mut self, config: Option<RootNoiseConfigV3>) -> Result<(), MctsV3Error> {
        let children = self.nodes[self.root].children.clone();
        let noise = match config {
            None => None,
            Some(config) => Some(sample_dirichlet(children.len(), config)?),
        };
        for (position, child_index) in children.into_iter().enumerate() {
            let child = &mut self.nodes[child_index];
            match (config, noise.as_ref()) {
                (Some(config), Some(noise)) => {
                    let epsilon = config.epsilon.clamp(0.0, 1.0);
                    child.root_noise = Some(noise[position]);
                    child.prior = (1.0 - epsilon) * child.raw_prior + epsilon * noise[position];
                }
                _ => {
                    child.root_noise = None;
                    child.prior = child.raw_prior;
                }
            }
        }
        Ok(())
    }

    fn select_leaf(
        &mut self,
        simulation: usize,
        engine: &Engine,
    ) -> Result<SelectedLeafV3, MctsV3Error> {
        let mut current = self.root;
        let mut path = vec![current];
        while self.nodes[current].expanded
            && !self.nodes[current].children.is_empty()
            && !self.nodes[current].terminal
        {
            current = self.select_child(current)?;
            self.materialize_node(current, engine)?;
            path.push(current);
        }
        if path.len() > 1 && self.nodes[path[1]].first_visit_simulation.is_none() {
            self.nodes[path[1]].first_visit_simulation = Some(simulation);
        }
        for index in &path {
            self.nodes[*index].virtual_visits += 1;
        }
        Ok(SelectedLeafV3 {
            node: current,
            path,
        })
    }

    fn materialize_node(&mut self, node_index: usize, engine: &Engine) -> Result<(), MctsV3Error> {
        if self.nodes[node_index].state.is_some() {
            return Ok(());
        }
        let parent = self.nodes[node_index]
            .parent
            .ok_or_else(|| MctsV3Error::Invariant("unmaterialized root".to_owned()))?;
        let parent_state = self.nodes[parent].state()?.clone();
        let action = self.nodes[node_index]
            .action_from_parent
            .clone()
            .ok_or_else(|| MctsV3Error::Invariant("child has no incoming action".to_owned()))?;
        let transition = engine
            .apply(&parent_state, action)
            .map_err(|error| MctsV3Error::Engine(error.to_string()))?;
        self.nodes[node_index].terminal = transition.state.status != GameStatus::Ongoing;
        self.nodes[node_index].state = Some(transition.state);
        Ok(())
    }

    fn select_child(&self, node_index: usize) -> Result<usize, MctsV3Error> {
        let parent = &self.nodes[node_index];
        if node_index == self.root && self.config.root_min_visits > 0 {
            if let Some(index) = parent
                .children
                .iter()
                .copied()
                .filter(|child| {
                    self.nodes[*child].visit_count + self.nodes[*child].virtual_visits
                        < self.config.root_min_visits
                })
                .max_by(|left, right| {
                    self.nodes[*left]
                        .prior
                        .total_cmp(&self.nodes[*right].prior)
                        .then_with(|| {
                            self.nodes[*right]
                                .action_index
                                .cmp(&self.nodes[*left].action_index)
                        })
                })
            {
                return Ok(index);
            }
        }
        let parent_visits = parent.visit_count + parent.virtual_visits;
        let sqrt_parent = parent_visits.max(1) as f32;
        let sqrt_parent = sqrt_parent.sqrt();
        let parent_q = parent.q_value();
        let parent_player = parent.state()?.turn;
        parent
            .children
            .iter()
            .copied()
            .max_by(|left, right| {
                let score = |index: usize| {
                    let child = &self.nodes[index];
                    let effective_visits = child.visit_count + child.virtual_visits;
                    let q = if child.visit_count == 0 {
                        parent_q - self.config.fpu_reduction
                    } else if child
                        .state
                        .as_ref()
                        .map_or(parent_player, |state| state.turn)
                        == parent_player
                    {
                        child.q_value()
                    } else {
                        -child.q_value()
                    };
                    let virtual_penalty = self.config.virtual_loss * child.virtual_visits as f32
                        / effective_visits.max(1) as f32;
                    let exploration = self.config.c_puct * child.prior * sqrt_parent
                        / (1.0 + effective_visits as f32);
                    q - virtual_penalty + exploration
                };
                score(*left).total_cmp(&score(*right)).then_with(|| {
                    self.nodes[*right]
                        .action_index
                        .cmp(&self.nodes[*left].action_index)
                })
            })
            .ok_or_else(|| MctsV3Error::Invariant("expanded node has no child".to_owned()))
    }

    fn release_virtual(&mut self, path: &[usize]) -> Result<(), MctsV3Error> {
        for index in path {
            let node = &mut self.nodes[*index];
            node.virtual_visits = node
                .virtual_visits
                .checked_sub(1)
                .ok_or_else(|| MctsV3Error::Invariant("virtual visit underflow".to_owned()))?;
        }
        Ok(())
    }

    fn backup(
        &mut self,
        path: &[usize],
        leaf_player: Player,
        value: f32,
    ) -> Result<(), MctsV3Error> {
        if !value.is_finite() {
            return invariant("cannot back up a non-finite value");
        }
        for index in path.iter().rev() {
            let node = &mut self.nodes[*index];
            node.visit_count += 1;
            let signed = if node.state()?.turn == leaf_player {
                value
            } else {
                -value
            };
            node.value_sum += signed;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_result(
        &mut self,
        root_raw_value: f32,
        completed_simulations: usize,
        evaluator_batches: usize,
        evaluated_leaves: usize,
        maximum_leaf_batch: usize,
        tree_reused: bool,
    ) -> Result<SearchResultV3, MctsV3Error> {
        let root = &self.nodes[self.root];
        if root.children.is_empty() {
            return invariant("searched root has no children");
        }
        let root_player = root.state()?.turn;
        let mut visits = Vec::with_capacity(root.children.len());
        let mut actions = Vec::with_capacity(root.children.len());
        let mut total_visits = 0u32;
        let mut max_visits = 0u32;
        let mut visited = 0usize;
        let mut low_prior_visits = 0u32;
        let mut max_depth = 0usize;
        for child_index in root.children.iter().copied() {
            let child = &self.nodes[child_index];
            let action = child
                .action_from_parent
                .clone()
                .ok_or_else(|| MctsV3Error::Invariant("root child has no action".to_owned()))?;
            total_visits += child.visit_count;
            max_visits = max_visits.max(child.visit_count);
            visited += usize::from(child.visit_count > 0);
            if child.raw_prior < 1e-4 {
                low_prior_visits += child.visit_count;
            }
            max_depth = max_depth.max(child.depth);
            let q_root =
                if child.state.as_ref().map_or(root_player, |state| state.turn) == root_player {
                    child.q_value()
                } else {
                    -child.q_value()
                };
            visits.push((action.clone(), child.action_index, child.visit_count));
            actions.push(RootActionDiagnosticsV3 {
                action,
                action_index: child.action_index,
                raw_prior: child.raw_prior,
                root_noise: child.root_noise,
                effective_prior: child.prior,
                visits: child.visit_count,
                first_visit_simulation: child.first_visit_simulation,
                q_root,
            });
        }
        for node in &self.nodes {
            max_depth = max_depth.max(node.depth);
        }
        let mut fingerprints = HashSet::with_capacity(self.nodes.len());
        let transposition_candidate_duplicates = self
            .nodes
            .iter()
            .filter_map(|node| node.state.as_ref())
            .filter(|state| !fingerprints.insert(state.fingerprint()))
            .count();
        let mut entropy = 0.0f32;
        if total_visits > 0 {
            for action in &actions {
                if action.visits > 0 {
                    let probability = action.visits as f32 / total_visits as f32;
                    entropy -= probability * probability.ln();
                }
            }
        }
        let best = visits
            .iter()
            .max_by(|left, right| left.2.cmp(&right.2).then_with(|| right.1.cmp(&left.1)))
            .cloned()
            .ok_or_else(|| MctsV3Error::Invariant("visit distribution is empty".to_owned()))?;
        self.last_diagnostics = Some(SearchDiagnosticsV3 {
            root_raw_value,
            root_legal_actions: actions.len(),
            root_visited_action_count: visited,
            root_visit_coverage: visited as f32 / actions.len() as f32,
            root_visit_entropy: entropy,
            top1_visit_fraction: if total_visits == 0 {
                0.0
            } else {
                max_visits as f32 / total_visits as f32
            },
            effective_branching_factor: entropy.exp(),
            low_prior_visit_rate: if total_visits == 0 {
                0.0
            } else {
                low_prior_visits as f32 / total_visits as f32
            },
            max_search_depth: max_depth,
            completed_simulations,
            evaluator_batches,
            evaluated_leaves,
            maximum_leaf_batch,
            tree_reused,
            transposition_candidate_duplicates,
            root_min_visits_satisfied: actions
                .iter()
                .all(|action| action.visits >= self.config.root_min_visits),
            root_actions: actions,
        });
        Ok(SearchResultV3 {
            action: best.0,
            action_index: best.1,
            visits,
        })
    }
}

fn terminal_value(state: &GameState) -> Result<f32, MctsV3Error> {
    match state.status {
        GameStatus::Won { winner, .. } => Ok(if winner == state.turn { 1.0 } else { -1.0 }),
        GameStatus::Draw { .. } => Ok(0.0),
        GameStatus::Ongoing => invariant("requested terminal value for an ongoing state"),
    }
}

fn sample_dirichlet(count: usize, config: RootNoiseConfigV3) -> Result<Vec<f32>, MctsV3Error> {
    if count == 0 || !config.alpha.is_finite() || config.alpha <= 0.0 {
        return search_error("Dirichlet alpha and action count must be positive");
    }
    if !config.epsilon.is_finite() || !(0.0..=1.0).contains(&config.epsilon) {
        return search_error("root noise epsilon must be in [0, 1]");
    }
    let gamma = Gamma::new(config.alpha as f64, 1.0)
        .map_err(|error| MctsV3Error::InvalidConfig(error.to_string()))?;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut values = Vec::with_capacity(count);
    let mut total = 0.0f64;
    for _ in 0..count {
        let value = gamma.sample(&mut rng);
        values.push(value);
        total += value;
    }
    if !total.is_finite() || total <= f64::EPSILON {
        return search_error("Dirichlet sample has no finite mass");
    }
    Ok(values
        .into_iter()
        .map(|value| (value / total) as f32)
        .collect())
}

fn search_error<T>(message: impl Into<String>) -> Result<T, MctsV3Error> {
    Err(MctsV3Error::InvalidConfig(message.into()))
}

fn invariant<T>(message: impl Into<String>) -> Result<T, MctsV3Error> {
    Err(MctsV3Error::Invariant(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LegalActionSetV3;
    use augment_core::{BUNDLED_RULESET_ID, Board, GamePhase, Mode, Piece, PieceKind, SACRIFICE};
    use std::sync::Mutex;

    struct RecordingEvaluator {
        batch_sizes: Mutex<Vec<usize>>,
        suppress_king_capture_prior: bool,
        prefer_sacrifice: bool,
    }

    impl RecordingEvaluator {
        fn new(suppress_king_capture_prior: bool) -> Self {
            Self {
                batch_sizes: Mutex::new(Vec::new()),
                suppress_king_capture_prior,
                prefer_sacrifice: false,
            }
        }

        fn sacrifice() -> Self {
            Self {
                batch_sizes: Mutex::new(Vec::new()),
                suppress_king_capture_prior: false,
                prefer_sacrifice: true,
            }
        }

        fn largest_batch(&self) -> usize {
            self.batch_sizes
                .lock()
                .unwrap()
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
        }
    }

    impl BatchEvaluatorV3 for RecordingEvaluator {
        fn evaluate_batch_v3(
            &self,
            states: &[GameState],
            engine: &Engine,
        ) -> Result<Vec<EvaluationV3>, MctsV3Error> {
            self.batch_sizes.lock().unwrap().push(states.len());
            states
                .iter()
                .map(|state| {
                    let legal = LegalActionSetV3::from_state(state, engine)
                        .map_err(|error| MctsV3Error::Evaluation(error.to_string()))?;
                    let mut weights = Vec::with_capacity(legal.len());
                    for index in 0..legal.len() {
                        let action = legal
                            .decode_action(index, state)
                            .map_err(|error| MctsV3Error::Evaluation(error.to_string()))?;
                        let king_capture = matches!(
                            &action,
                            Action::Move(movement)
                                if state.board.get(movement.to).is_some_and(|piece| {
                                    piece.owner != state.turn && piece.kind == PieceKind::King
                                })
                        );
                        let sacrifice = matches!(action, Action::UseCard { .. });
                        let suppressed = (self.suppress_king_capture_prior && king_capture)
                            || (self.prefer_sacrifice && sacrifice);
                        weights.push((action, index, if suppressed { 1e-12 } else { 1.0 }));
                    }
                    let total: f32 = weights.iter().map(|item| item.2).sum();
                    Ok(EvaluationV3 {
                        priors: weights
                            .into_iter()
                            .map(|(action, index, weight)| (action, index, weight / total))
                            .collect(),
                        value: if self.prefer_sacrifice {
                            let white_value = if !state.card_runtime.sacrifice_protection.is_empty()
                            {
                                1.0
                            } else if state.ply > 0 {
                                -1.0
                            } else {
                                0.0
                            };
                            if state.turn == Player::White {
                                white_value
                            } else {
                                -white_value
                            }
                        } else {
                            0.0
                        },
                    })
                })
                .collect()
        }
    }

    fn config(simulations: usize, batch: usize) -> MctsV3Config {
        MctsV3Config {
            num_simulations: simulations,
            leaf_batch_size: batch,
            ..MctsV3Config::default()
        }
    }

    fn tactical_state(shield_king: bool) -> GameState {
        let mut board = Board::empty();
        board.set(
            "h1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King).with_id(1)),
        );
        board.set(
            "a1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook).with_id(2)),
        );
        board.set(
            "a8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King).with_id(3)),
        );
        let mut state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 7);
        state.phase = GamePhase::Playing;
        if shield_king {
            state.card_runtime.shielded_piece_ids.insert(3);
        }
        state
    }

    #[test]
    fn leaf_evaluations_are_batched_and_every_root_action_is_visited() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 11);
        state.phase = GamePhase::Playing;
        let evaluator = RecordingEvaluator::new(false);
        let legal_count = engine.legal_actions(&state).unwrap().len();
        let mut tree = MctsV3Tree::new(config(legal_count + 16, 16)).unwrap();
        tree.search(&state, &engine, &evaluator).unwrap();
        let diagnostics = tree.last_diagnostics.as_ref().unwrap();
        assert_eq!(diagnostics.root_visit_coverage, 1.0);
        assert!(diagnostics.root_min_visits_satisfied);
        assert!(evaluator.largest_batch() > 1);
        assert!(diagnostics.maximum_leaf_batch > 1);
    }

    #[test]
    fn tiny_prior_forced_win_is_not_starved() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = tactical_state(false);
        let evaluator = RecordingEvaluator::new(true);
        let legal_count = engine.legal_actions(&state).unwrap().len();
        let mut tree = MctsV3Tree::new(config(legal_count + 64, 8)).unwrap();
        let result = tree.search(&state, &engine, &evaluator).unwrap();
        let transition = engine.apply(&state, result.action).unwrap();
        assert!(matches!(
            transition.state.status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));
        let winning = tree
            .last_diagnostics
            .as_ref()
            .unwrap()
            .root_actions
            .iter()
            .find(|action| action.raw_prior < 1e-8)
            .unwrap();
        assert!(winning.visits > 0);
        assert_eq!(
            tree.last_diagnostics.as_ref().unwrap().root_visit_coverage,
            1.0
        );
    }

    #[test]
    fn low_prior_sacrifice_card_can_win_the_search() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut board = Board::empty();
        board.set(
            "h1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King).with_id(1)),
        );
        board.set(
            "a2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn).with_id(2)),
        );
        board.set(
            "b2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Queen).with_id(3)),
        );
        board.set(
            "g8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King).with_id(4)),
        );
        let state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 31);
        let state = engine
            .grant_card(&state, 77, SACRIFICE, Player::White)
            .unwrap()
            .state;
        let legal_count = engine.legal_actions(&state).unwrap().len();
        let evaluator = RecordingEvaluator::sacrifice();
        let mut tree = MctsV3Tree::new(config(legal_count + 96, 8)).unwrap();
        let result = tree.search(&state, &engine, &evaluator).unwrap();
        assert!(matches!(result.action, Action::UseCard { .. }));
        let next = engine.apply(&state, result.action).unwrap().state;
        assert!(!next.card_runtime.sacrifice_protection.is_empty());
        assert!(
            tree.last_diagnostics
                .as_ref()
                .unwrap()
                .root_actions
                .iter()
                .any(|action| action.raw_prior < 1e-8 && action.visits > 0)
        );
    }

    #[test]
    fn shielded_king_move_is_not_fabricated_as_a_terminal_win() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = tactical_state(true);
        let legal = engine.legal_actions(&state).unwrap();
        let capture = legal
            .iter()
            .find(|action| matches!(action, Action::Move(movement) if movement.to == "a8".parse().unwrap()))
            .unwrap();
        let transition = engine.apply(&state, capture.clone()).unwrap();
        assert_eq!(transition.state.status, GameStatus::Ongoing);

        let evaluator = RecordingEvaluator::new(false);
        let mut tree = MctsV3Tree::new(config(32, 8)).unwrap();
        let result = tree.search(&state, &engine, &evaluator).unwrap();
        assert_eq!(result.visits.iter().map(|item| item.2).sum::<u32>(), 32);
        assert!(result.visits.iter().all(|item| item.2 < 1_000));
    }

    #[test]
    fn backup_flips_only_when_the_player_to_move_changes() {
        let mut white = tactical_state(false);
        white.turn = Player::White;
        let mut still_white = white.clone();
        still_white.ply += 1;
        let mut black = still_white.clone();
        black.turn = Player::Black;
        let mut tree = MctsV3Tree::new(config(1, 1)).unwrap();
        tree.nodes = vec![
            MctsNodeV3::root(white),
            MctsNodeV3::root(still_white),
            MctsNodeV3::root(black),
        ];
        tree.backup(&[0, 1, 2], Player::Black, 0.75).unwrap();
        assert_eq!(tree.nodes[0].value_sum, -0.75);
        assert_eq!(tree.nodes[1].value_sum, -0.75);
        assert_eq!(tree.nodes[2].value_sum, 0.75);
    }

    #[test]
    fn next_position_reuses_and_compacts_the_selected_subtree() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 19);
        state.phase = GamePhase::Playing;
        let evaluator = RecordingEvaluator::new(false);
        let mut tree = MctsV3Tree::new(config(48, 8)).unwrap();
        let first = tree.search(&state, &engine, &evaluator).unwrap();
        let next = engine.apply(&state, first.action).unwrap().state;
        tree.search(&next, &engine, &evaluator).unwrap();
        assert!(tree.last_diagnostics.as_ref().unwrap().tree_reused);
        assert_eq!(tree.nodes[0].state.as_ref(), Some(&next));
        assert_eq!(tree.nodes[0].depth, 0);
    }

    #[test]
    fn root_noise_and_temperature_are_explicit_and_deterministic() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 23);
        state.phase = GamePhase::Playing;
        let evaluator = RecordingEvaluator::new(false);
        let noise = RootNoiseConfigV3 {
            alpha: 0.3,
            epsilon: 0.25,
            seed: 99,
        };
        let mut left = MctsV3Tree::new(config(24, 8)).unwrap();
        let mut right = MctsV3Tree::new(config(24, 8)).unwrap();
        let left_result = left
            .search_with_root_noise(&state, &engine, &evaluator, Some(noise))
            .unwrap();
        let right_result = right
            .search_with_root_noise(&state, &engine, &evaluator, Some(noise))
            .unwrap();
        let left_priors: Vec<_> = left
            .last_diagnostics
            .as_ref()
            .unwrap()
            .root_actions
            .iter()
            .map(|item| item.effective_prior)
            .collect();
        let right_priors: Vec<_> = right
            .last_diagnostics
            .as_ref()
            .unwrap()
            .root_actions
            .iter()
            .map(|item| item.effective_prior)
            .collect();
        assert_eq!(left_priors, right_priors);
        assert!((left_priors.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(
            MctsV3Tree::sample_action_from_visits(&left_result.visits, 0.8, 123),
            MctsV3Tree::sample_action_from_visits(&right_result.visits, 0.8, 123)
        );
    }

    #[test]
    fn wall_clock_search_completes_at_least_one_simulation_and_honors_the_cap() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = tactical_state(false);
        let evaluator = RecordingEvaluator::new(false);
        let mut tree = MctsV3Tree::new(config(64, 8)).unwrap();
        tree.search_for_duration(&state, &engine, &evaluator, Duration::from_nanos(1))
            .unwrap();
        let completed = tree
            .last_diagnostics
            .as_ref()
            .unwrap()
            .completed_simulations;
        assert!((1..=64).contains(&completed));
    }
}

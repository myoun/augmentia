use crate::{
    ActionDescriptorV3, EvaluationV3, LegalActionSetV3, PieceAbilityV3, V3OnnxRuntime,
    encode_observation_v3,
};
use augment_core::{
    Action, ActionTarget, CHAMELEON_MUTATION, Engine, GamePhase, GameState, GameStatus, Player,
    SearchChanceContext, Square,
};
use rand::distributions::WeightedIndex;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution as RandDistribution, Gamma};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
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
    pub draft_future_offer_samples: usize,
    pub search_seed: u64,
    pub reuse_descendant_depth: usize,
    pub progressive_widening: bool,
    pub widening_k0: usize,
    pub widening_c: f32,
    pub widening_alpha: f32,
    pub proposal_top_policy: usize,
    pub proposal_family_per_group: usize,
    pub proposal_tail: usize,
    pub proposal_seed: u64,
    pub gumbel_root: bool,
    pub gumbel_candidates: usize,
    pub gumbel_seed: u64,
    pub gumbel_q_scale: f32,
    pub gumbel_planning_budget: usize,
    pub virtual_chameleon_hierarchy: bool,
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
            draft_future_offer_samples: 8,
            search_seed: 0x5633_5f53_4541_5243,
            reuse_descendant_depth: 4,
            progressive_widening: false,
            widening_k0: 32,
            widening_c: 4.0,
            widening_alpha: 0.5,
            proposal_top_policy: 24,
            proposal_family_per_group: 2,
            proposal_tail: 8,
            proposal_seed: 0x5052_4f50_4f53_414c,
            gumbel_root: false,
            gumbel_candidates: 64,
            gumbel_seed: 0x4755_4d42_454c_5631,
            gumbel_q_scale: 1.0,
            gumbel_planning_budget: 300,
            virtual_chameleon_hierarchy: false,
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
        if !(1..=16).contains(&self.draft_future_offer_samples) {
            return search_error("draft future-offer samples must be in 1..=16");
        }
        if self.reuse_descendant_depth > 16 {
            return search_error("descendant reuse depth must not exceed 16");
        }
        if self.widening_k0 == 0
            || !self.widening_c.is_finite()
            || self.widening_c < 0.0
            || !self.widening_alpha.is_finite()
            || !(0.0..=1.0).contains(&self.widening_alpha)
            || self.proposal_top_policy == 0
            || self.proposal_family_per_group == 0
            || self.proposal_tail == 0
        {
            return search_error("progressive widening and proposal configuration is invalid");
        }
        if self.gumbel_candidates == 0
            || self.gumbel_planning_budget == 0
            || !self.gumbel_q_scale.is_finite()
            || self.gumbel_q_scale <= 0.0
        {
            return search_error("Gumbel root configuration is invalid");
        }
        if self.gumbel_root && self.virtual_chameleon_hierarchy {
            return search_error(
                "Gumbel root and virtual action hierarchy require a separate combined semantics",
            );
        }
        Ok(self)
    }

    fn effective_gumbel_budget(self) -> usize {
        self.num_simulations.min(self.gumbel_planning_budget)
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
    pub opened: bool,
    pub proposal_rank: usize,
    pub proposal_probability: f32,
    pub opened_at_simulation: Option<usize>,
    pub action_family: String,
}

#[derive(Clone, Debug)]
pub struct SearchDiagnosticsV3 {
    pub root_raw_value: f32,
    pub root_legal_actions: usize,
    pub root_visited_action_count: usize,
    pub root_opened_action_count: usize,
    pub root_action_family_count: usize,
    pub root_opened_action_family_count: usize,
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
    pub tree_reuse_depth: Option<usize>,
    pub transposition_candidate_duplicates: usize,
    pub root_min_visits_satisfied: bool,
    pub sampled_chance_actions: usize,
    pub sampled_chance_outcomes: usize,
    pub search_semantics_id: &'static str,
    pub progressive_widening_enabled: bool,
    pub proposal_tail_visit_rate: f32,
    pub gumbel_root_enabled: bool,
    pub gumbel_initial_candidates: usize,
    pub gumbel_halving_rounds: usize,
    pub virtual_hierarchy_enabled: bool,
    pub virtual_node_count: usize,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MctsNodeRoleV3 {
    State,
    ChanceAction,
    VirtualPrefix,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ActionFamilyKeyV3 {
    Move,
    PieceAbility(PieceAbilityV3),
    Card(String),
    Draft,
    Choice(String),
}

#[derive(Clone, Debug)]
struct PendingActionV3 {
    action: Action,
    action_index: usize,
    raw_prior: f32,
    selection_mass: f32,
    root_noise: Option<f32>,
    proposal_rank: usize,
    proposal_probability: f32,
    tail_proposal: bool,
}

#[derive(Clone, Debug)]
struct RootGumbelStateV3 {
    active: Vec<usize>,
    gumbels: HashMap<usize, f32>,
    round_start_visits: HashMap<usize, u32>,
    round_quota: u32,
    call_start_root_visits: u32,
    initial_candidates: usize,
    completed_rounds: usize,
}

#[derive(Clone, Debug)]
struct MctsNodeV3 {
    role: MctsNodeRoleV3,
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
    selection_mass: f32,
    root_noise: Option<f32>,
    first_visit_simulation: Option<usize>,
    depth: usize,
    children: Vec<usize>,
    expanded: bool,
    terminal: bool,
    value_player: Player,
    information_key: Option<[u8; 32]>,
    chance_sampled: bool,
    chance_event_depth: u32,
    pending_actions: VecDeque<PendingActionV3>,
    total_legal_actions: usize,
    initial_open_count: usize,
    proposal_rank: usize,
    proposal_probability: f32,
    tail_proposal: bool,
    opened_at_simulation: Option<usize>,
    virtual_card_instance: Option<u32>,
    virtual_prefix: Vec<Square>,
}

impl MctsNodeV3 {
    fn root(state: GameState) -> Self {
        let terminal = state.status != GameStatus::Ongoing;
        let value_player = state.turn;
        Self {
            role: MctsNodeRoleV3::State,
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
            selection_mass: 1.0,
            root_noise: None,
            first_visit_simulation: None,
            depth: 0,
            children: Vec::new(),
            expanded: false,
            terminal,
            value_player,
            information_key: None,
            chance_sampled: false,
            chance_event_depth: 0,
            pending_actions: VecDeque::new(),
            total_legal_actions: 0,
            initial_open_count: 0,
            proposal_rank: 0,
            proposal_probability: 1.0,
            tail_proposal: false,
            opened_at_simulation: Some(0),
            virtual_card_instance: None,
            virtual_prefix: Vec::new(),
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
    root_gumbel: Option<RootGumbelStateV3>,
}

impl MctsV3Tree {
    pub fn new(config: MctsV3Config) -> Result<Self, MctsV3Error> {
        Ok(Self {
            nodes: Vec::with_capacity(4096),
            config: config.validate()?,
            root: 0,
            last_diagnostics: None,
            root_gumbel: None,
        })
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn set_search_budget(
        &mut self,
        num_simulations: usize,
        leaf_batch_size: usize,
    ) -> Result<(), MctsV3Error> {
        let mut config = self.config;
        config.num_simulations = num_simulations;
        config.leaf_batch_size = leaf_batch_size;
        self.config = config.validate()?;
        Ok(())
    }

    pub fn set_search_seed(&mut self, search_seed: u64) {
        if self.config.search_seed != search_seed {
            self.config.search_seed = search_seed;
            self.clear();
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root = 0;
        self.last_diagnostics = None;
        self.root_gumbel = None;
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
        let tree_reuse_depth = self.prepare_root(root_state);
        let tree_reused = tree_reuse_depth.is_some();
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
        let gumbel_enabled = self.config.gumbel_root
            && !matches!(self.nodes[self.root].state()?.phase, GamePhase::Draft(_));
        if gumbel_enabled && root_noise.is_some_and(|noise| noise.epsilon > 0.0) {
            return search_error("Gumbel root and Dirichlet noise cannot be enabled together");
        }
        self.apply_root_noise(root_noise)?;
        if gumbel_enabled {
            self.initialize_root_gumbel()?;
        } else {
            self.root_gumbel = None;
        }

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
                if self.gumbel_batch_boundary_pending() {
                    break;
                }
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
            tree_reuse_depth,
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

    fn prepare_root(&mut self, state: &GameState) -> Option<usize> {
        if self.nodes.is_empty() {
            self.nodes.push(MctsNodeV3::root(state.clone()));
            self.root = 0;
            return None;
        }
        if self.nodes[self.root].state.as_ref() == Some(state) {
            return Some(0);
        }

        let root_depth = self.nodes[self.root].depth;
        let mut queue = VecDeque::from([self.root]);
        let mut reusable = None;
        while let Some(index) = queue.pop_front() {
            let relative_depth = self.nodes[index].depth.saturating_sub(root_depth);
            if relative_depth >= self.config.reuse_descendant_depth {
                continue;
            }
            for child in self.nodes[index].children.iter().copied() {
                let node = &self.nodes[child];
                if !node.chance_sampled {
                    if node.role == MctsNodeRoleV3::State {
                        if node.state.as_ref().is_some_and(|candidate| {
                            candidate.fingerprint() == state.fingerprint() && candidate == state
                        }) {
                            reusable = Some((child, relative_depth + 1));
                            break;
                        }
                        queue.push_back(child);
                    } else if node.role == MctsNodeRoleV3::VirtualPrefix {
                        queue.push_back(child);
                    }
                }
            }
            if reusable.is_some() {
                break;
            }
        }
        if let Some((index, depth)) = reusable {
            if self.config.gumbel_root && depth > 0 {
                self.clear();
                self.nodes.push(MctsNodeV3::root(state.clone()));
                return None;
            }
            self.compact_subtree(index);
            Some(depth)
        } else {
            self.clear();
            self.nodes.push(MctsNodeV3::root(state.clone()));
            None
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
        let information_key = search_information_key_v1(&state, &legal)?;
        if legal.len() != evaluation.priors.len() {
            return Err(MctsV3Error::Evaluation(format!(
                "policy has {} actions but engine has {}",
                evaluation.priors.len(),
                legal.len()
            )));
        }
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
        let gumbel_here = node_index == self.root
            && self.config.gumbel_root
            && !matches!(state.phase, GamePhase::Draft(_));
        let (proposal, mandatory_initial) = if self.config.progressive_widening || gumbel_here {
            build_candidate_proposal(validated, &legal, &information_key, self.config)?
        } else {
            (
                validated
                    .into_iter()
                    .enumerate()
                    .map(
                        |(proposal_rank, (action, action_index, prior))| PendingActionV3 {
                            action,
                            action_index,
                            raw_prior: prior,
                            selection_mass: prior,
                            root_noise: None,
                            proposal_rank,
                            proposal_probability: 1.0,
                            tail_proposal: false,
                        },
                    )
                    .collect(),
                legal.len(),
            )
        };
        let initial_open_count = if gumbel_here {
            self.config
                .gumbel_candidates
                .max(mandatory_initial)
                .min(self.config.effective_gumbel_budget())
                .min(proposal.len())
        } else if self.config.progressive_widening {
            self.config
                .widening_k0
                .max(mandatory_initial)
                .min(proposal.len())
        } else {
            proposal.len()
        };
        self.nodes[node_index].children.clear();
        self.nodes[node_index].pending_actions = proposal.into();
        self.nodes[node_index].total_legal_actions = legal.len();
        self.nodes[node_index].initial_open_count = initial_open_count;
        self.nodes[node_index].network_value = Some(network_value);
        self.nodes[node_index].information_key = Some(information_key);
        self.nodes[node_index].expanded = true;
        for _ in 0..initial_open_count {
            self.open_next_action(node_index, Some(0))?;
        }
        self.renormalize_open_priors(node_index)?;
        Ok(())
    }

    fn open_next_action(
        &mut self,
        node_index: usize,
        simulation: Option<usize>,
    ) -> Result<Option<usize>, MctsV3Error> {
        let Some(candidate) = self.nodes[node_index].pending_actions.pop_front() else {
            return Ok(None);
        };
        if self.config.virtual_chameleon_hierarchy
            && chameleon_virtual_path(self.nodes[node_index].state()?, &candidate.action).is_some()
        {
            return self
                .insert_chameleon_action(node_index, candidate, simulation)
                .map(Some);
        }
        let child_index = self.nodes.len();
        let depth = self.nodes[node_index].depth + 1;
        let value_player = self.nodes[node_index].value_player;
        let chance_sampled = self.nodes[node_index].chance_sampled;
        let chance_event_depth = self.nodes[node_index].chance_event_depth;
        self.nodes.push(MctsNodeV3 {
            role: MctsNodeRoleV3::State,
            terminal: false,
            state: None,
            parent: Some(node_index),
            action_from_parent: Some(candidate.action),
            action_index: candidate.action_index,
            visit_count: 0,
            value_sum: 0.0,
            network_value: None,
            virtual_visits: 0,
            prior: candidate.selection_mass,
            raw_prior: candidate.raw_prior,
            selection_mass: candidate.selection_mass,
            root_noise: candidate.root_noise,
            first_visit_simulation: None,
            depth,
            children: Vec::new(),
            expanded: false,
            value_player,
            information_key: None,
            chance_sampled,
            chance_event_depth,
            pending_actions: VecDeque::new(),
            total_legal_actions: 0,
            initial_open_count: 0,
            proposal_rank: candidate.proposal_rank,
            proposal_probability: candidate.proposal_probability,
            tail_proposal: candidate.tail_proposal,
            opened_at_simulation: simulation,
            virtual_card_instance: None,
            virtual_prefix: Vec::new(),
        });
        self.nodes[node_index].children.push(child_index);
        Ok(Some(child_index))
    }

    fn insert_chameleon_action(
        &mut self,
        state_node: usize,
        candidate: PendingActionV3,
        simulation: Option<usize>,
    ) -> Result<usize, MctsV3Error> {
        let (instance_id, squares) = chameleon_virtual_path(
            self.nodes[state_node].state()?,
            &candidate.action,
        )
        .ok_or_else(|| {
            MctsV3Error::Invariant("non-chameleon action reached virtual insertion".to_owned())
        })?;
        if squares.is_empty() {
            return invariant("chameleon virtual action has an empty target");
        }
        let mut current = self.nodes[state_node]
            .children
            .iter()
            .copied()
            .find(|child| {
                self.nodes[*child].role == MctsNodeRoleV3::VirtualPrefix
                    && self.nodes[*child].virtual_card_instance == Some(instance_id)
                    && self.nodes[*child].virtual_prefix.is_empty()
            })
            .unwrap_or_else(|| {
                let node = self.new_virtual_prefix_node(
                    state_node,
                    instance_id,
                    Vec::new(),
                    candidate.action_index,
                    candidate.proposal_rank,
                );
                self.nodes[state_node].children.push(node);
                node
            });
        for prefix_len in 1..=squares.len() {
            let prefix = squares[..prefix_len].to_vec();
            let next = self.nodes[current]
                .children
                .iter()
                .copied()
                .find(|child| {
                    self.nodes[*child].role == MctsNodeRoleV3::VirtualPrefix
                        && self.nodes[*child].virtual_card_instance == Some(instance_id)
                        && self.nodes[*child].virtual_prefix == prefix
                })
                .unwrap_or_else(|| {
                    let node = self.new_virtual_prefix_node(
                        current,
                        instance_id,
                        prefix,
                        candidate.action_index,
                        candidate.proposal_rank,
                    );
                    self.nodes[current].children.push(node);
                    node
                });
            current = next;
        }

        let child_index = self.nodes.len();
        let value_player = self.nodes[state_node].value_player;
        let chance_sampled = self.nodes[state_node].chance_sampled;
        let chance_event_depth = self.nodes[state_node].chance_event_depth;
        self.nodes.push(MctsNodeV3 {
            role: MctsNodeRoleV3::State,
            terminal: false,
            state: None,
            parent: Some(current),
            action_from_parent: Some(candidate.action),
            action_index: candidate.action_index,
            visit_count: 0,
            value_sum: 0.0,
            network_value: None,
            virtual_visits: 0,
            prior: candidate.selection_mass,
            raw_prior: candidate.raw_prior,
            selection_mass: candidate.selection_mass,
            root_noise: candidate.root_noise,
            first_visit_simulation: None,
            depth: self.nodes[state_node].depth + 1,
            children: Vec::new(),
            expanded: false,
            value_player,
            information_key: None,
            chance_sampled,
            chance_event_depth,
            pending_actions: VecDeque::new(),
            total_legal_actions: 0,
            initial_open_count: 0,
            proposal_rank: candidate.proposal_rank,
            proposal_probability: candidate.proposal_probability,
            tail_proposal: candidate.tail_proposal,
            opened_at_simulation: simulation,
            virtual_card_instance: None,
            virtual_prefix: Vec::new(),
        });
        self.nodes[current].children.push(child_index);
        Ok(child_index)
    }

    fn new_virtual_prefix_node(
        &mut self,
        parent: usize,
        instance_id: u32,
        prefix: Vec<Square>,
        action_index: usize,
        proposal_rank: usize,
    ) -> usize {
        let node_index = self.nodes.len();
        self.nodes.push(MctsNodeV3 {
            role: MctsNodeRoleV3::VirtualPrefix,
            terminal: false,
            state: None,
            parent: Some(parent),
            action_from_parent: None,
            action_index,
            visit_count: 0,
            value_sum: 0.0,
            network_value: None,
            virtual_visits: 0,
            prior: 0.0,
            raw_prior: 0.0,
            selection_mass: 0.0,
            root_noise: None,
            first_visit_simulation: None,
            depth: self.nodes[parent].depth,
            children: Vec::new(),
            expanded: true,
            value_player: self.nodes[parent].value_player,
            information_key: None,
            chance_sampled: self.nodes[parent].chance_sampled,
            chance_event_depth: self.nodes[parent].chance_event_depth,
            pending_actions: VecDeque::new(),
            total_legal_actions: 0,
            initial_open_count: 0,
            proposal_rank,
            proposal_probability: 1.0,
            tail_proposal: false,
            opened_at_simulation: Some(0),
            virtual_card_instance: Some(instance_id),
            virtual_prefix: prefix,
        });
        node_index
    }

    fn renormalize_open_priors(&mut self, node_index: usize) -> Result<(), MctsV3Error> {
        if self.nodes[node_index].role == MctsNodeRoleV3::ChanceAction {
            return Ok(());
        }
        let children = self.nodes[node_index].children.clone();
        for child in &children {
            if self.nodes[*child].role == MctsNodeRoleV3::VirtualPrefix {
                self.refresh_virtual_mass(*child)?;
            }
        }
        let total = children
            .iter()
            .map(|child| self.nodes[*child].selection_mass)
            .sum::<f32>();
        if !total.is_finite() || total <= f32::EPSILON {
            return invariant("opened action prior mass is not finite and positive");
        }
        for child in children {
            self.nodes[child].prior = self.nodes[child].selection_mass / total;
        }
        Ok(())
    }

    fn refresh_virtual_mass(&mut self, node_index: usize) -> Result<(), MctsV3Error> {
        if self.nodes[node_index].role != MctsNodeRoleV3::VirtualPrefix {
            return invariant("virtual mass refresh received a non-virtual node");
        }
        let children = self.nodes[node_index].children.clone();
        for child in &children {
            if self.nodes[*child].role == MctsNodeRoleV3::VirtualPrefix {
                self.refresh_virtual_mass(*child)?;
            }
        }
        let raw = children
            .iter()
            .map(|child| self.nodes[*child].raw_prior)
            .sum::<f32>();
        let selection = children
            .iter()
            .map(|child| self.nodes[*child].selection_mass)
            .sum::<f32>();
        if !raw.is_finite()
            || !selection.is_finite()
            || raw <= f32::EPSILON
            || selection <= f32::EPSILON
        {
            return invariant("virtual prefix has no finite positive policy mass");
        }
        self.nodes[node_index].raw_prior = raw;
        self.nodes[node_index].selection_mass = selection;
        for child in children {
            self.nodes[child].prior = self.nodes[child].selection_mass / selection;
        }
        Ok(())
    }

    fn widen_node(&mut self, node_index: usize, simulation: usize) -> Result<(), MctsV3Error> {
        if !self.config.progressive_widening
            || self.nodes[node_index].role != MctsNodeRoleV3::State
            || self.nodes[node_index].pending_actions.is_empty()
            || (node_index == self.root && self.root_gumbel.is_some())
        {
            return Ok(());
        }
        let visits = self.nodes[node_index].visit_count + self.nodes[node_index].virtual_visits;
        let additional = (self.config.widening_c * (visits as f32).powf(self.config.widening_alpha))
            .floor() as usize;
        let target = self.nodes[node_index]
            .initial_open_count
            .saturating_add(additional)
            .min(self.nodes[node_index].total_legal_actions);
        let mut changed = false;
        let mut opened_actions = self.completed_action_children(node_index).len();
        while opened_actions < target {
            if self
                .open_next_action(node_index, Some(simulation))?
                .is_none()
            {
                break;
            }
            opened_actions += 1;
            changed = true;
        }
        if changed {
            self.renormalize_open_priors(node_index)?;
        }
        Ok(())
    }

    fn apply_root_noise(&mut self, config: Option<RootNoiseConfigV3>) -> Result<(), MctsV3Error> {
        let children = self.completed_action_children(self.root);
        let action_count = self.nodes[self.root].total_legal_actions;
        let noise = match config {
            None => None,
            Some(config) => Some(sample_dirichlet(action_count, config)?),
        };
        for child_index in children {
            let child = &mut self.nodes[child_index];
            match (config, noise.as_ref()) {
                (Some(config), Some(noise)) => {
                    let epsilon = config.epsilon.clamp(0.0, 1.0);
                    let sampled = noise[child.action_index];
                    child.root_noise = Some(sampled);
                    child.selection_mass = (1.0 - epsilon) * child.raw_prior + epsilon * sampled;
                }
                _ => {
                    child.root_noise = None;
                    child.selection_mass = child.raw_prior;
                }
            }
        }
        for candidate in &mut self.nodes[self.root].pending_actions {
            match (config, noise.as_ref()) {
                (Some(config), Some(noise)) => {
                    let epsilon = config.epsilon.clamp(0.0, 1.0);
                    let sampled = noise[candidate.action_index];
                    candidate.root_noise = Some(sampled);
                    candidate.selection_mass =
                        (1.0 - epsilon) * candidate.raw_prior + epsilon * sampled;
                }
                _ => {
                    candidate.root_noise = None;
                    candidate.selection_mass = candidate.raw_prior;
                }
            }
        }
        self.renormalize_open_priors(self.root)
    }

    fn completed_action_children(&self, state_node: usize) -> Vec<usize> {
        fn collect(tree: &MctsV3Tree, node: usize, output: &mut Vec<usize>) {
            for child in tree.nodes[node].children.iter().copied() {
                if tree.nodes[child].action_from_parent.is_some() {
                    output.push(child);
                } else if tree.nodes[child].role == MctsNodeRoleV3::VirtualPrefix {
                    collect(tree, child, output);
                }
            }
        }
        let mut output = Vec::new();
        collect(self, state_node, &mut output);
        output
    }

    fn initialize_root_gumbel(&mut self) -> Result<(), MctsV3Error> {
        let active = self.nodes[self.root].children.clone();
        let planning_budget = self.config.effective_gumbel_budget();
        if active.is_empty() || active.len() > planning_budget {
            return invariant("Gumbel root candidate count must fit the simulation budget");
        }
        let information_key = self.nodes[self.root].information_key.ok_or_else(|| {
            MctsV3Error::Invariant("Gumbel root has no information key".to_owned())
        })?;
        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&information_key[16..24]);
        let seed = self.config.gumbel_seed ^ u64::from_le_bytes(prefix);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut gumbels = HashMap::with_capacity(active.len());
        let mut round_start_visits = HashMap::with_capacity(active.len());
        for child in &active {
            let uniform = rng.gen_range(f64::EPSILON..(1.0 - f64::EPSILON));
            gumbels.insert(*child, (-(-uniform.ln()).ln()) as f32);
            round_start_visits.insert(*child, self.nodes[*child].visit_count);
        }
        let round_quota = gumbel_round_quota(planning_budget, active.len());
        self.root_gumbel = Some(RootGumbelStateV3 {
            initial_candidates: active.len(),
            active,
            gumbels,
            round_start_visits,
            round_quota,
            call_start_root_visits: self.nodes[self.root].visit_count,
            completed_rounds: 0,
        });
        Ok(())
    }

    fn gumbel_batch_boundary_pending(&self) -> bool {
        let Some(state) = &self.root_gumbel else {
            return false;
        };
        let allocated = state.active.iter().all(|child| {
            let start = state.round_start_visits[child];
            self.nodes[*child]
                .visit_count
                .saturating_add(self.nodes[*child].virtual_visits)
                .saturating_sub(start)
                >= state.round_quota
        });
        let committed = state.active.iter().all(|child| {
            self.nodes[*child]
                .visit_count
                .saturating_sub(state.round_start_visits[child])
                >= state.round_quota
        });
        allocated && !committed
    }

    fn select_root_gumbel_child(&mut self) -> Result<usize, MctsV3Error> {
        let mut state = self
            .root_gumbel
            .take()
            .ok_or_else(|| MctsV3Error::Invariant("Gumbel root state is missing".to_owned()))?;
        loop {
            let round_complete = state.active.iter().all(|child| {
                self.nodes[*child]
                    .visit_count
                    .saturating_sub(state.round_start_visits[child])
                    >= state.round_quota
            });
            if !round_complete || state.active.len() == 1 {
                break;
            }
            let root_player = self.nodes[self.root].value_player;
            let root_value = self.nodes[self.root].network_value.unwrap_or(0.0);
            state.active.sort_by(|left, right| {
                let rank = |child: usize| {
                    let node = &self.nodes[child];
                    let q = if node.visit_count == 0 {
                        root_value
                    } else if node.value_player == root_player {
                        node.q_value()
                    } else {
                        -node.q_value()
                    };
                    state.gumbels[&child]
                        + node.raw_prior.max(f32::MIN_POSITIVE).ln()
                        + self.config.gumbel_q_scale * q
                };
                rank(*right).total_cmp(&rank(*left)).then_with(|| {
                    self.nodes[*left]
                        .action_index
                        .cmp(&self.nodes[*right].action_index)
                })
            });
            state.active.truncate(state.active.len().div_ceil(2));
            state.completed_rounds += 1;
            state.round_start_visits = state
                .active
                .iter()
                .map(|child| (*child, self.nodes[*child].visit_count))
                .collect();
            let used = self.nodes[self.root]
                .visit_count
                .saturating_sub(state.call_start_root_visits) as usize;
            let remaining = self.config.effective_gumbel_budget().saturating_sub(used);
            state.round_quota = gumbel_round_quota(remaining, state.active.len());
        }
        let selected = state
            .active
            .iter()
            .copied()
            .min_by_key(|child| {
                let start = state.round_start_visits[child];
                (
                    self.nodes[*child]
                        .visit_count
                        .saturating_add(self.nodes[*child].virtual_visits)
                        .saturating_sub(start),
                    self.nodes[*child].action_index,
                )
            })
            .ok_or_else(|| {
                MctsV3Error::Invariant("Gumbel root has no active candidate".to_owned())
            })?;
        self.root_gumbel = Some(state);
        Ok(selected)
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
            self.widen_node(current, simulation)?;
            current = if current == self.root && self.root_gumbel.is_some() {
                self.select_root_gumbel_child()?
            } else {
                self.select_child(current)?
            };
            if self.nodes[current].role == MctsNodeRoleV3::State
                && self.nodes[current].state.is_none()
            {
                self.materialize_node(current, engine)?;
            }
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
        let mut state_parent = parent;
        while self.nodes[state_parent].state.is_none() {
            state_parent = self.nodes[state_parent].parent.ok_or_else(|| {
                MctsV3Error::Invariant("virtual action path has no state ancestor".to_owned())
            })?;
        }
        let parent_state = self.nodes[state_parent].state()?.clone();
        let action = self.nodes[node_index]
            .action_from_parent
            .clone()
            .ok_or_else(|| MctsV3Error::Invariant("child has no incoming action".to_owned()))?;
        let information_key = self.nodes[state_parent].information_key.ok_or_else(|| {
            MctsV3Error::Invariant("expanded parent has no information key".to_owned())
        })?;
        let chance_event_depth = self.nodes[state_parent].chance_event_depth;
        let sample_transition = |sample_index: usize| {
            let seed = derive_chance_lane_seed(
                self.config.search_seed,
                &information_key,
                sample_index as u64,
                chance_event_depth,
            );
            let mut context = SearchChanceContext::new(seed);
            engine
                .apply_for_search(&parent_state, action.clone(), &mut context)
                .map_err(|error| MctsV3Error::Engine(error.to_string()))
        };
        let first = sample_transition(0)?;
        if first.sampled_chance && self.config.draft_future_offer_samples > 1 {
            let sample_count = self.config.draft_future_offer_samples;
            let mut transitions = Vec::with_capacity(sample_count);
            transitions.push(first.transition);
            for sample_index in 1..sample_count {
                let sampled = sample_transition(sample_index)?;
                if !sampled.sampled_chance {
                    return invariant(
                        "chance transition changed classification between sample lanes",
                    );
                }
                transitions.push(sampled.transition);
            }

            let parent_player = self.nodes[state_parent].value_player;
            let parent_depth = self.nodes[state_parent].depth;
            self.nodes[node_index].role = MctsNodeRoleV3::ChanceAction;
            self.nodes[node_index].state = None;
            self.nodes[node_index].value_player = parent_player;
            self.nodes[node_index].depth = parent_depth;
            self.nodes[node_index].expanded = true;
            self.nodes[node_index].terminal = false;
            self.nodes[node_index].chance_sampled = true;
            let probability = 1.0 / sample_count as f32;
            let mut outcomes = Vec::with_capacity(sample_count);
            for (sample_index, transition) in transitions.into_iter().enumerate() {
                let value_player = transition.state.turn;
                let terminal = transition.state.status != GameStatus::Ongoing;
                let child_index = self.nodes.len();
                self.nodes.push(MctsNodeV3 {
                    role: MctsNodeRoleV3::State,
                    terminal,
                    state: Some(transition.state),
                    parent: Some(node_index),
                    action_from_parent: None,
                    action_index: sample_index,
                    visit_count: 0,
                    value_sum: 0.0,
                    network_value: None,
                    virtual_visits: 0,
                    prior: probability,
                    raw_prior: probability,
                    selection_mass: probability,
                    root_noise: None,
                    first_visit_simulation: None,
                    depth: parent_depth + 1,
                    children: Vec::new(),
                    expanded: false,
                    value_player,
                    information_key: None,
                    chance_sampled: true,
                    chance_event_depth: chance_event_depth.saturating_add(1),
                    pending_actions: VecDeque::new(),
                    total_legal_actions: 0,
                    initial_open_count: 0,
                    proposal_rank: sample_index,
                    proposal_probability: 1.0,
                    tail_proposal: false,
                    opened_at_simulation: Some(0),
                    virtual_card_instance: None,
                    virtual_prefix: Vec::new(),
                });
                outcomes.push(child_index);
            }
            self.nodes[node_index].children = outcomes;
            return Ok(());
        }

        let transition = first.transition;
        if first.sampled_chance {
            self.nodes[node_index].chance_sampled = true;
            self.nodes[node_index].chance_event_depth = chance_event_depth.saturating_add(1);
        }
        self.nodes[node_index].terminal = transition.state.status != GameStatus::Ongoing;
        self.nodes[node_index].value_player = transition.state.turn;
        self.nodes[node_index].state = Some(transition.state);
        Ok(())
    }

    fn select_child(&self, node_index: usize) -> Result<usize, MctsV3Error> {
        let parent = &self.nodes[node_index];
        if parent.role == MctsNodeRoleV3::ChanceAction {
            return parent
                .children
                .iter()
                .copied()
                .min_by_key(|child| {
                    (
                        self.nodes[*child].visit_count + self.nodes[*child].virtual_visits,
                        self.nodes[*child].action_index,
                    )
                })
                .ok_or_else(|| MctsV3Error::Invariant("chance action has no outcomes".to_owned()));
        }
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
        let parent_player = parent.value_player;
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
                    } else if child.value_player == parent_player {
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
            let signed = if node.value_player == leaf_player {
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
        tree_reuse_depth: Option<usize>,
    ) -> Result<SearchResultV3, MctsV3Error> {
        let action_children = self.completed_action_children(self.root);
        let root = &self.nodes[self.root];
        if action_children.is_empty() {
            return invariant("searched root has no children");
        }
        let root_player = root.state()?.turn;
        let root_state = root.state()?;
        let mut visits = Vec::with_capacity(root.total_legal_actions);
        let mut actions = Vec::with_capacity(root.total_legal_actions);
        let mut total_visits = 0u32;
        let mut max_visits = 0u32;
        let mut visited = 0usize;
        let mut low_prior_visits = 0u32;
        let mut tail_visits = 0u32;
        let mut max_depth = 0usize;
        let root_open_mass = root
            .children
            .iter()
            .map(|child| self.nodes[*child].selection_mass)
            .sum::<f32>();
        for child_index in action_children {
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
            if child.tail_proposal {
                tail_visits += child.visit_count;
            }
            max_depth = max_depth.max(child.depth);
            let q_root = if child.value_player == root_player {
                child.q_value()
            } else {
                -child.q_value()
            };
            let action_family = diagnostic_action_family(root_state, &action);
            visits.push((action.clone(), child.action_index, child.visit_count));
            actions.push(RootActionDiagnosticsV3 {
                action,
                action_index: child.action_index,
                raw_prior: child.raw_prior,
                root_noise: child.root_noise,
                effective_prior: child.selection_mass / root_open_mass,
                visits: child.visit_count,
                first_visit_simulation: child.first_visit_simulation,
                q_root,
                opened: true,
                proposal_rank: child.proposal_rank,
                proposal_probability: child.proposal_probability,
                opened_at_simulation: child.opened_at_simulation,
                action_family,
            });
        }
        for candidate in &root.pending_actions {
            visits.push((candidate.action.clone(), candidate.action_index, 0));
            actions.push(RootActionDiagnosticsV3 {
                action: candidate.action.clone(),
                action_index: candidate.action_index,
                raw_prior: candidate.raw_prior,
                root_noise: candidate.root_noise,
                effective_prior: 0.0,
                visits: 0,
                first_visit_simulation: None,
                q_root: 0.0,
                opened: false,
                proposal_rank: candidate.proposal_rank,
                proposal_probability: candidate.proposal_probability,
                opened_at_simulation: None,
                action_family: diagnostic_action_family(root_state, &candidate.action),
            });
        }
        visits.sort_by_key(|item| item.1);
        actions.sort_by_key(|item| item.action_index);
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
        let root_action_family_count = actions
            .iter()
            .map(|action| action.action_family.as_str())
            .collect::<HashSet<_>>()
            .len();
        let root_opened_action_family_count = actions
            .iter()
            .filter(|action| action.opened)
            .map(|action| action.action_family.as_str())
            .collect::<HashSet<_>>()
            .len();
        let sampled_chance_actions = self
            .nodes
            .iter()
            .filter(|node| node.role == MctsNodeRoleV3::ChanceAction)
            .count();
        let sampled_chance_outcomes = self
            .nodes
            .iter()
            .filter(|node| node.role == MctsNodeRoleV3::ChanceAction)
            .map(|node| node.children.len())
            .sum();
        let virtual_node_count = self
            .nodes
            .iter()
            .filter(|node| node.role == MctsNodeRoleV3::VirtualPrefix)
            .count();
        let (gumbel_root_enabled, gumbel_initial_candidates, gumbel_halving_rounds) =
            self.root_gumbel.as_ref().map_or((false, 0, 0), |state| {
                (true, state.initial_candidates, state.completed_rounds)
            });
        self.last_diagnostics = Some(SearchDiagnosticsV3 {
            root_raw_value,
            root_legal_actions: root.total_legal_actions,
            root_visited_action_count: visited,
            root_opened_action_count: actions.iter().filter(|action| action.opened).count(),
            root_action_family_count,
            root_opened_action_family_count,
            root_visit_coverage: visited as f32 / root.total_legal_actions as f32,
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
            tree_reuse_depth,
            transposition_candidate_duplicates,
            root_min_visits_satisfied: actions
                .iter()
                .filter(|action| action.opened)
                .all(|action| action.visits >= self.config.root_min_visits),
            sampled_chance_actions,
            sampled_chance_outcomes,
            search_semantics_id: if self.config.virtual_chameleon_hierarchy
                && self.config.progressive_widening
            {
                "augmentia.mcts-v3.fair-chance.virtual-chameleon.progressive-widening.v1"
            } else if self.config.virtual_chameleon_hierarchy {
                "augmentia.mcts-v3.fair-chance.virtual-chameleon.v1"
            } else if gumbel_root_enabled && self.config.progressive_widening {
                "augmentia.mcts-v3.fair-chance.gumbel.progressive-widening.v1"
            } else if gumbel_root_enabled {
                "augmentia.mcts-v3.fair-chance.gumbel.v1"
            } else if self.config.progressive_widening {
                "augmentia.mcts-v3.fair-chance.progressive-widening.v1"
            } else {
                "augmentia.mcts-v3.fair-chance.v1"
            },
            progressive_widening_enabled: self.config.progressive_widening,
            proposal_tail_visit_rate: if total_visits == 0 {
                0.0
            } else {
                tail_visits as f32 / total_visits as f32
            },
            gumbel_root_enabled,
            gumbel_initial_candidates,
            gumbel_halving_rounds,
            virtual_hierarchy_enabled: self.config.virtual_chameleon_hierarchy,
            virtual_node_count,
            root_actions: actions,
        });
        Ok(SearchResultV3 {
            action: best.0,
            action_index: best.1,
            visits,
        })
    }
}

fn build_candidate_proposal(
    validated: Vec<(Action, usize, f32)>,
    legal: &LegalActionSetV3,
    information_key: &[u8; 32],
    config: MctsV3Config,
) -> Result<(Vec<PendingActionV3>, usize), MctsV3Error> {
    if validated.len() != legal.len() {
        return invariant("candidate proposal input differs from legal action count");
    }
    let mut order = Vec::with_capacity(validated.len());
    let mut selected = HashSet::with_capacity(validated.len());
    let mut deterministic = vec![false; validated.len()];
    let mut stochastic_probability = vec![1.0f32; validated.len()];

    let mut by_policy = (0..validated.len()).collect::<Vec<_>>();
    by_policy.sort_by(|left, right| {
        validated[*right]
            .2
            .total_cmp(&validated[*left].2)
            .then_with(|| left.cmp(right))
    });
    for index in by_policy.into_iter().take(config.proposal_top_policy) {
        if selected.insert(index) {
            deterministic[index] = true;
            order.push(index);
        }
    }

    let mut families: BTreeMap<ActionFamilyKeyV3, Vec<usize>> = BTreeMap::new();
    for (index, descriptor) in legal.actions.iter().enumerate() {
        families
            .entry(action_family_key(descriptor))
            .or_default()
            .push(index);
    }
    for indices in families.values_mut() {
        indices.sort_by(|left, right| {
            validated[*right]
                .2
                .total_cmp(&validated[*left].2)
                .then_with(|| left.cmp(right))
        });
        for index in indices
            .iter()
            .copied()
            .take(config.proposal_family_per_group)
        {
            if selected.insert(index) {
                deterministic[index] = true;
                order.push(index);
            }
        }
    }

    let mut tail = (0..validated.len())
        .filter(|index| !selected.contains(index))
        .collect::<Vec<_>>();
    let mut rng =
        StdRng::seed_from_u64(derive_proposal_seed(config.proposal_seed, information_key));
    tail.shuffle(&mut rng);
    let tail_take = config.proposal_tail.min(tail.len());
    let tail_probability = if tail.is_empty() {
        1.0
    } else {
        tail_take as f32 / tail.len() as f32
    };
    for index in &tail {
        stochastic_probability[*index] = tail_probability;
    }
    order.extend(tail);
    let mandatory_initial = deterministic.iter().filter(|value| **value).count() + tail_take;

    if order.len() != validated.len() || selected.len() > order.len() {
        return invariant("candidate proposal omitted or duplicated an action");
    }
    let mut items = validated.into_iter().map(Some).collect::<Vec<_>>();
    let mut proposal = Vec::with_capacity(items.len());
    for (proposal_rank, index) in order.into_iter().enumerate() {
        let (action, action_index, prior) = items[index].take().ok_or_else(|| {
            MctsV3Error::Invariant("candidate proposal duplicated an action".to_owned())
        })?;
        proposal.push(PendingActionV3 {
            action,
            action_index,
            raw_prior: prior,
            selection_mass: prior,
            root_noise: None,
            proposal_rank,
            proposal_probability: if deterministic[index] {
                1.0
            } else {
                stochastic_probability[index]
            },
            tail_proposal: !deterministic[index],
        });
    }
    if items.iter().any(Option::is_some) {
        return invariant("candidate proposal did not consume every action");
    }
    Ok((proposal, mandatory_initial.min(legal.len())))
}

fn action_family_key(descriptor: &ActionDescriptorV3) -> ActionFamilyKeyV3 {
    match descriptor {
        ActionDescriptorV3::Move { .. } => ActionFamilyKeyV3::Move,
        ActionDescriptorV3::PieceAbility { ability, .. } => {
            ActionFamilyKeyV3::PieceAbility(*ability)
        }
        ActionDescriptorV3::Card { definition_id, .. } => {
            ActionFamilyKeyV3::Card(definition_id.clone())
        }
        ActionDescriptorV3::DraftPick { .. } => ActionFamilyKeyV3::Draft,
        ActionDescriptorV3::ResolveChoice { handler_id, .. } => {
            ActionFamilyKeyV3::Choice(handler_id.clone())
        }
    }
}

fn chameleon_virtual_path(state: &GameState, action: &Action) -> Option<(u32, Vec<Square>)> {
    let Action::UseCard {
        instance_id,
        target: ActionTarget::Squares(squares),
    } = action
    else {
        return None;
    };
    state
        .cards
        .iter()
        .find(|card| card.instance_id == *instance_id)
        .filter(|card| card.definition_id == CHAMELEON_MUTATION)
        .map(|_| (*instance_id, squares.clone()))
}

fn diagnostic_action_family(state: &GameState, action: &Action) -> String {
    match action {
        Action::Move(_) => "move".to_owned(),
        Action::UsePieceAbility { ability, .. } => format!("ability:{ability:?}"),
        Action::UseCard { instance_id, .. } => state
            .cards
            .iter()
            .find(|card| card.instance_id == *instance_id)
            .map_or_else(
                || "card:<missing>".to_owned(),
                |card| format!("card:{}", card.definition_id),
            ),
        Action::DraftPick { .. } => "draft_pick".to_owned(),
        Action::ResolveChoice { .. } => match &state.phase {
            GamePhase::AwaitingChoice(choice) => format!("choice:{}", choice.handler_id),
            _ => "choice:<stale>".to_owned(),
        },
    }
}

fn derive_proposal_seed(proposal_seed: u64, key: &[u8; 32]) -> u64 {
    let mut suffix = [0u8; 8];
    suffix.copy_from_slice(&key[8..16]);
    let mut value = proposal_seed ^ u64::from_le_bytes(suffix) ^ 0x5052_4f50_4f53_5631;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn gumbel_round_quota(remaining_budget: usize, active_candidates: usize) -> u32 {
    if active_candidates == 0 || remaining_budget == 0 {
        return 1;
    }
    if active_candidates == 1 {
        return remaining_budget.min(u32::MAX as usize) as u32;
    }
    let rounds = (usize::BITS - (active_candidates - 1).leading_zeros()) as usize;
    (remaining_budget / active_candidates.saturating_mul(rounds).max(1))
        .max(1)
        .min(u32::MAX as usize) as u32
}

fn search_information_key_v1(
    state: &GameState,
    legal: &LegalActionSetV3,
) -> Result<[u8; 32], MctsV3Error> {
    let observation = encode_observation_v3(state);
    let payload = serde_json::to_vec(&(
        "augmentia.search-information.v1",
        state.turn.index() as u8,
        observation,
        legal,
    ))
    .map_err(|error| MctsV3Error::Invariant(format!("information key encoding failed: {error}")))?;
    let digest = Sha256::digest(payload);
    let mut key = [0u8; 32];
    key.copy_from_slice(&digest);
    Ok(key)
}

fn derive_chance_lane_seed(
    search_seed: u64,
    key: &[u8; 32],
    sample_lane: u64,
    chance_event_depth: u32,
) -> u64 {
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&key[..8]);
    let mut value = search_seed
        ^ u64::from_le_bytes(prefix)
        ^ sample_lane.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ u64::from(chance_event_depth).wrapping_mul(0xd1b5_4a32_d192_ed03)
        ^ 0x4452_4146_545f_5631;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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
    use augment_core::{
        BUNDLED_RULESET_ID, Board, CHAMELEON_MUTATION, GamePhase, Mode, Piece, PieceKind, SACRIFICE,
    };
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

    fn wide_card_state(engine: &Engine) -> GameState {
        let mut board = Board::empty();
        board.set(
            "h1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King).with_id(1)),
        );
        board.set(
            "h8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King).with_id(2)),
        );
        for (offset, square) in [
            "a1", "b1", "c1", "d1", "e1", "f1", "g1", "a2", "b2", "c2", "d2", "e2", "f2", "g2",
        ]
        .into_iter()
        .enumerate()
        {
            board.set(
                square.parse().unwrap(),
                Some(Piece::new(Player::White, PieceKind::Pawn).with_id(10 + offset as u32)),
            );
        }
        let state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 91);
        engine
            .grant_card(&state, 700, CHAMELEON_MUTATION, Player::White)
            .unwrap()
            .state
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
    fn exact_materialized_grandchild_is_reused_with_reported_depth() {
        let engine = Engine::bundled_20260818().unwrap();
        let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 19);
        state.phase = GamePhase::Playing;
        let evaluator = RecordingEvaluator::new(false);
        let mut tree = MctsV3Tree::new(MctsV3Config {
            num_simulations: 192,
            leaf_batch_size: 8,
            reuse_descendant_depth: 4,
            ..MctsV3Config::default()
        })
        .unwrap();
        tree.search(&state, &engine, &evaluator).unwrap();
        let target = tree
            .nodes
            .iter()
            .find(|node| {
                node.depth >= 2
                    && !node.chance_sampled
                    && node
                        .state
                        .as_ref()
                        .is_some_and(|candidate| candidate.status == GameStatus::Ongoing)
            })
            .and_then(|node| node.state.clone())
            .expect("search must materialize a deterministic grandchild");

        let depth = tree.prepare_root(&target);
        assert!(depth.is_some_and(|depth| depth >= 2));
        assert_eq!(tree.nodes[tree.root].state.as_ref(), Some(&target));
        assert_eq!(tree.nodes[tree.root].depth, 0);
    }

    #[test]
    fn sampled_chance_descendant_is_never_persistently_reused() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = engine.new_game(Mode::Standard, 404).unwrap();
        let evaluator = RecordingEvaluator::new(false);
        let mut tree = MctsV3Tree::new(MctsV3Config {
            num_simulations: 24,
            leaf_batch_size: 8,
            draft_future_offer_samples: 4,
            ..MctsV3Config::default()
        })
        .unwrap();
        tree.search(&state, &engine, &evaluator).unwrap();
        let action = tree.nodes[tree.root].children[0];
        let outcome = tree.nodes[action].children[0];
        let sampled = tree.nodes[outcome].state().unwrap().clone();
        assert!(tree.nodes[outcome].chance_sampled);

        assert_eq!(tree.prepare_root(&sampled), None);
        assert_eq!(tree.nodes[tree.root].state.as_ref(), Some(&sampled));
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

    #[test]
    fn draft_search_samples_future_offers_without_changing_actual_rng() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = engine.new_game(Mode::Standard, 2026).unwrap();
        let rng_before = state.rng_state();
        let evaluator = RecordingEvaluator::new(false);
        let mut tree = MctsV3Tree::new(MctsV3Config {
            num_simulations: 72,
            leaf_batch_size: 8,
            draft_future_offer_samples: 8,
            search_seed: 77,
            ..MctsV3Config::default()
        })
        .unwrap();

        tree.search(&state, &engine, &evaluator).unwrap();
        let diagnostics = tree.last_diagnostics.as_ref().unwrap();
        assert_eq!(state.rng_state(), rng_before);
        assert_eq!(diagnostics.sampled_chance_actions, 3);
        assert_eq!(diagnostics.sampled_chance_outcomes, 24);

        let mut distinct_offers = HashSet::new();
        for action_node in tree.nodes[tree.root].children.iter().copied() {
            assert_eq!(tree.nodes[action_node].role, MctsNodeRoleV3::ChanceAction);
            for outcome in tree.nodes[action_node].children.iter().copied() {
                let GamePhase::Draft(draft) = &tree.nodes[outcome].state().unwrap().phase else {
                    panic!("white Draft selection must expose a sampled black offer");
                };
                distinct_offers.insert(draft.offers.clone());
            }
        }
        assert!(distinct_offers.len() > 1);
    }

    #[test]
    fn draft_search_seed_reproduces_sampled_tree_and_visits() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = engine.new_game(Mode::Standard, 303).unwrap();
        let evaluator = RecordingEvaluator::new(false);
        let config = MctsV3Config {
            num_simulations: 48,
            leaf_batch_size: 8,
            draft_future_offer_samples: 4,
            search_seed: 991,
            ..MctsV3Config::default()
        };
        let mut left = MctsV3Tree::new(config).unwrap();
        let mut right = MctsV3Tree::new(config).unwrap();
        let left_result = left.search(&state, &engine, &evaluator).unwrap();
        let right_result = right.search(&state, &engine, &evaluator).unwrap();

        assert_eq!(left_result.visits, right_result.visits);
        let sampled = |tree: &MctsV3Tree| {
            tree.nodes[tree.root]
                .children
                .iter()
                .flat_map(|action| tree.nodes[*action].children.iter())
                .map(|outcome| tree.nodes[*outcome].state().unwrap().fingerprint())
                .collect::<Vec<_>>()
        };
        assert_eq!(sampled(&left), sampled(&right));
    }

    #[test]
    fn progressive_widening_opens_a_reproducible_prefix_and_preserves_full_result_shape() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = wide_card_state(&engine);
        let evaluator = RecordingEvaluator::new(false);
        let legal_count = LegalActionSetV3::from_state(&state, &engine).unwrap().len();
        assert!(legal_count > 400);
        let config = MctsV3Config {
            num_simulations: 24,
            leaf_batch_size: 4,
            progressive_widening: true,
            widening_k0: 8,
            widening_c: 1.0,
            widening_alpha: 0.5,
            proposal_top_policy: 4,
            proposal_family_per_group: 2,
            proposal_tail: 2,
            proposal_seed: 1234,
            ..MctsV3Config::default()
        };
        let mut left = MctsV3Tree::new(config).unwrap();
        let mut right = MctsV3Tree::new(config).unwrap();
        let left_result = left.search(&state, &engine, &evaluator).unwrap();
        let right_result = right.search(&state, &engine, &evaluator).unwrap();
        let diagnostics = left.last_diagnostics.as_ref().unwrap();

        assert_eq!(left_result.visits.len(), legal_count);
        assert_eq!(left_result.visits, right_result.visits);
        assert!(diagnostics.progressive_widening_enabled);
        assert!(diagnostics.root_opened_action_count >= 8);
        assert!(diagnostics.root_opened_action_count < legal_count);
        assert_eq!(diagnostics.root_legal_actions, legal_count);
        assert!(
            diagnostics
                .root_actions
                .iter()
                .any(|action| { action.opened && matches!(action.action, Action::UseCard { .. }) })
        );
        assert!(
            diagnostics
                .root_actions
                .iter()
                .any(|action| !action.opened && action.visits == 0)
        );
        assert!(diagnostics.root_actions.iter().all(|action| {
            action.proposal_probability > 0.0 && action.proposal_probability <= 1.0
        }));
        let open_prior = diagnostics
            .root_actions
            .iter()
            .filter(|action| action.opened)
            .map(|action| action.effective_prior)
            .sum::<f32>();
        assert!((open_prior - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gumbel_root_halves_a_fixed_candidate_set_with_exact_budget() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = wide_card_state(&engine);
        let evaluator = RecordingEvaluator::new(false);
        let legal_count = LegalActionSetV3::from_state(&state, &engine).unwrap().len();
        let config = MctsV3Config {
            num_simulations: 128,
            leaf_batch_size: 8,
            gumbel_root: true,
            gumbel_candidates: 16,
            gumbel_seed: 55,
            proposal_top_policy: 4,
            proposal_family_per_group: 1,
            proposal_tail: 2,
            ..MctsV3Config::default()
        };
        let mut left = MctsV3Tree::new(config).unwrap();
        let mut right = MctsV3Tree::new(config).unwrap();
        let left_result = left.search(&state, &engine, &evaluator).unwrap();
        let right_result = right.search(&state, &engine, &evaluator).unwrap();
        let diagnostics = left.last_diagnostics.as_ref().unwrap();

        assert_eq!(left_result.visits, right_result.visits);
        assert_eq!(
            left_result.visits.iter().map(|item| item.2).sum::<u32>(),
            128
        );
        assert_eq!(left_result.visits.len(), legal_count);
        assert!(diagnostics.gumbel_root_enabled);
        assert_eq!(diagnostics.gumbel_initial_candidates, 16);
        assert!(diagnostics.gumbel_halving_rounds >= 3);
        assert_eq!(diagnostics.root_opened_action_count, 16);
        assert!(
            diagnostics
                .root_actions
                .iter()
                .filter(|action| !action.opened)
                .all(|action| action.visits == 0)
        );
    }

    #[test]
    fn gumbel_root_rejects_dirichlet_but_is_disabled_for_draft() {
        let engine = Engine::bundled_20260818().unwrap();
        let evaluator = RecordingEvaluator::new(false);
        let state = tactical_state(false);
        let mut tree = MctsV3Tree::new(MctsV3Config {
            num_simulations: 32,
            gumbel_root: true,
            gumbel_candidates: 8,
            proposal_top_policy: 4,
            proposal_family_per_group: 1,
            proposal_tail: 1,
            ..MctsV3Config::default()
        })
        .unwrap();
        let error = tree
            .search_with_root_noise(
                &state,
                &engine,
                &evaluator,
                Some(RootNoiseConfigV3 {
                    alpha: 0.3,
                    epsilon: 0.25,
                    seed: 1,
                }),
            )
            .unwrap_err();
        assert!(error.to_string().contains("cannot be enabled together"));

        let draft = engine.new_game(Mode::Standard, 12).unwrap();
        let mut draft_tree = MctsV3Tree::new(MctsV3Config {
            num_simulations: 24,
            gumbel_root: true,
            gumbel_candidates: 8,
            ..MctsV3Config::default()
        })
        .unwrap();
        draft_tree
            .search_with_root_noise(
                &draft,
                &engine,
                &evaluator,
                Some(RootNoiseConfigV3 {
                    alpha: 0.3,
                    epsilon: 0.25,
                    seed: 1,
                }),
            )
            .unwrap();
        assert!(
            !draft_tree
                .last_diagnostics
                .as_ref()
                .unwrap()
                .gumbel_root_enabled
        );
    }

    #[test]
    fn chameleon_virtual_hierarchy_has_one_canonical_path_per_engine_action() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = wide_card_state(&engine);
        let evaluator = RecordingEvaluator::new(false);
        let legal = engine.legal_actions(&state).unwrap();
        let mut tree = MctsV3Tree::new(MctsV3Config {
            num_simulations: 96,
            leaf_batch_size: 8,
            virtual_chameleon_hierarchy: true,
            ..MctsV3Config::default()
        })
        .unwrap();
        let result = tree
            .search_with_root_noise(
                &state,
                &engine,
                &evaluator,
                Some(RootNoiseConfigV3 {
                    alpha: 0.3,
                    epsilon: 0.25,
                    seed: 88,
                }),
            )
            .unwrap();
        let diagnostics = tree.last_diagnostics.as_ref().unwrap();
        assert!(diagnostics.virtual_hierarchy_enabled);
        assert!(diagnostics.virtual_node_count > 0);
        assert_eq!(result.visits.len(), legal.len());

        let action_nodes = tree.completed_action_children(tree.root);
        assert_eq!(action_nodes.len(), legal.len());
        let mut tree_actions = action_nodes
            .iter()
            .map(|node| tree.nodes[*node].action_from_parent.clone().unwrap())
            .collect::<Vec<_>>();
        tree_actions.sort();
        assert_eq!(tree_actions, legal);

        fn collect_virtual(tree: &MctsV3Tree, node: usize, output: &mut Vec<usize>) {
            for child in tree.nodes[node].children.iter().copied() {
                if tree.nodes[child].role == MctsNodeRoleV3::VirtualPrefix {
                    output.push(child);
                    collect_virtual(tree, child, output);
                }
            }
        }
        let mut root_virtual = Vec::new();
        collect_virtual(&tree, tree.root, &mut root_virtual);
        let mut prefixes = HashSet::new();
        for node_index in root_virtual {
            let node = &tree.nodes[node_index];
            assert!(node.state.is_none());
            assert!(node.action_from_parent.is_none());
            assert!(prefixes.insert((node.virtual_card_instance, node.virtual_prefix.clone())));
            let child_prior = node
                .children
                .iter()
                .map(|child| tree.nodes[*child].prior)
                .sum::<f32>();
            assert!((child_prior - 1.0).abs() < 1e-5);
            assert_eq!(node.value_player, state.turn);
        }
        let full_prior = diagnostics
            .root_actions
            .iter()
            .map(|action| action.effective_prior)
            .sum::<f32>();
        assert!((full_prior - 1.0).abs() < 1e-5);
        assert!(
            diagnostics
                .root_actions
                .iter()
                .all(|action| action.root_noise.is_some())
        );
    }

    #[test]
    fn widening_counts_completed_virtual_actions_instead_of_prefix_nodes() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = wide_card_state(&engine);
        let evaluator = RecordingEvaluator::new(false);
        let legal_count = engine.legal_actions(&state).unwrap().len();
        let mut tree = MctsV3Tree::new(MctsV3Config {
            num_simulations: 16,
            leaf_batch_size: 4,
            progressive_widening: true,
            virtual_chameleon_hierarchy: true,
            widening_k0: 8,
            widening_c: 1.0,
            proposal_top_policy: 4,
            proposal_family_per_group: 1,
            proposal_tail: 2,
            ..MctsV3Config::default()
        })
        .unwrap();
        tree.search(&state, &engine, &evaluator).unwrap();
        let diagnostics = tree.last_diagnostics.as_ref().unwrap();
        assert!(diagnostics.root_opened_action_count >= 8);
        assert!(diagnostics.root_opened_action_count < legal_count);
        assert_eq!(
            tree.completed_action_children(tree.root).len(),
            diagnostics.root_opened_action_count
        );
    }
}

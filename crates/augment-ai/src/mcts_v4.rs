//! Engine-exact MCTS over the canonical V4 factorized action trie.
//!
//! A virtual prefix never owns a `GameState`. The engine is called only after a
//! `complete:action` leaf has resolved to one member of `LegalActionSetV3`.

use crate::{
    ActionDescriptorV3, FactorizedTrieV4, LegalActionSetV3, build_factorized_trie_v4,
    encode_observation_v3, factorized_action_path_v4,
};
use augment_core::{Action, Engine, GamePhase, GameState, GameStatus, Player, SearchChanceContext};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

pub const SEARCH_SEMANTICS_V4_ID: &str =
    "augmentia.mcts-v4.factorized-gumbel-widening-public-chance.v2";
pub const ROOT_SELECTION_V4_ID: &str = "augmentia.gumbel-sequential-halving.v1";
pub const CHANCE_SAMPLING_V4_ID: &str = "augmentia.public-information-multisample-chance.v1";

#[derive(Clone, Copy, Debug)]
pub struct MctsV4Config {
    pub num_simulations: usize,
    pub c_puct: f32,
    pub fpu_reduction: f32,
    pub widening_k0: usize,
    pub widening_c: f32,
    pub widening_alpha: f32,
    pub gumbel_candidates: usize,
    /// Number of independent public-information samples for a transition that consumes RNG.
    pub chance_outcome_samples: usize,
    /// Seed for hidden-chance search lanes. This never consumes the live game RNG.
    pub search_seed: u64,
    pub gumbel_seed: u64,
    pub gumbel_q_scale: f32,
}

impl Default for MctsV4Config {
    fn default() -> Self {
        Self {
            num_simulations: 400,
            c_puct: 1.5,
            fpu_reduction: 0.25,
            widening_k0: 2,
            widening_c: 2.0,
            widening_alpha: 0.5,
            gumbel_candidates: 16,
            chance_outcome_samples: 8,
            search_seed: 0x5345_4152_4348_5634,
            gumbel_seed: 0x4755_4d42_454c_5634,
            gumbel_q_scale: 1.0,
        }
    }
}

impl MctsV4Config {
    fn validate(self) -> Result<Self, MctsV4Error> {
        if self.num_simulations == 0 || self.num_simulations > u32::MAX as usize {
            return Err(MctsV4Error::InvalidConfig(
                "num_simulations must be in 1..=u32::MAX".into(),
            ));
        }
        if !self.c_puct.is_finite()
            || self.c_puct <= 0.0
            || !self.fpu_reduction.is_finite()
            || self.fpu_reduction < 0.0
        {
            return Err(MctsV4Error::InvalidConfig(
                "PUCT parameters are invalid".into(),
            ));
        }
        if self.widening_k0 == 0
            || !self.widening_c.is_finite()
            || self.widening_c < 0.0
            || !self.widening_alpha.is_finite()
            || !(0.0..=1.0).contains(&self.widening_alpha)
        {
            return Err(MctsV4Error::InvalidConfig(
                "progressive-widening parameters are invalid".into(),
            ));
        }
        if self.gumbel_candidates == 0
            || self.chance_outcome_samples == 0
            || self.chance_outcome_samples > 64
            || !self.gumbel_q_scale.is_finite()
            || self.gumbel_q_scale <= 0.0
        {
            return Err(MctsV4Error::InvalidConfig(
                "Gumbel or chance-sampling parameters are invalid".into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct FactorizedEvaluationV4 {
    /// Value from the point of view of `state.turn`.
    pub value: f32,
    /// One unnormalized logit per complete descriptor, in legal-set order.
    /// Prefix priors are exact subtree sums of the resulting distribution.
    pub action_logits: Vec<f32>,
}

/// Evaluate only information available through `ObservationV3` and the exact legal set.
///
/// Implementations receive `GameState` so they can call the authoritative encoder, but must not
/// inspect the live RNG state or any field excluded from `ObservationV3`. Production runtimes must
/// be regression-tested with public-identical states that carry different live RNG states.
pub trait FactorizedEvaluatorV4: Send + Sync {
    fn evaluate_v4(
        &self,
        state: &GameState,
        legal_actions: &LegalActionSetV3,
        engine: &Engine,
    ) -> Result<FactorizedEvaluationV4, MctsV4Error>;

    /// Optional value-only preparation for internal search nodes. The legal set
    /// remains exact. The owned scorer must retain only public-information data.
    /// Returning `None` preserves the complete-action evaluator path.
    fn prepare_prefix_evaluation_v4(
        &self,
        _state: &GameState,
        _legal_actions: &LegalActionSetV3,
        _engine: &Engine,
    ) -> Result<Option<PreparedPrefixEvaluationV4>, MctsV4Error> {
        Ok(None)
    }
}

/// An encoded public state and a scorer for conditional, exact legal trie rows.
pub struct PreparedPrefixEvaluationV4 {
    pub value: f32,
    pub scorer: Box<dyn PrefixScorerV4>,
}

pub trait PrefixScorerV4: Send + Sync {
    /// Return one finite logit per child, in the supplied canonical order.
    /// `children` contains every legal child of `prefix`, never a widened subset.
    fn score_prefix_v4(
        &self,
        prefix: &[String],
        children: &[String],
    ) -> Result<Vec<f32>, MctsV4Error>;
}

/// Small reproducible evaluator useful for smoke tests and deterministic tools.
#[derive(Clone, Copy, Debug)]
pub struct DeterministicFactorizedEvaluatorV4 {
    pub seed: u64,
    pub value: f32,
}

impl FactorizedEvaluatorV4 for DeterministicFactorizedEvaluatorV4 {
    fn evaluate_v4(
        &self,
        _state: &GameState,
        legal_actions: &LegalActionSetV3,
        _engine: &Engine,
    ) -> Result<FactorizedEvaluationV4, MctsV4Error> {
        let mut logits = Vec::with_capacity(legal_actions.len());
        for descriptor in &legal_actions.actions {
            let encoded = serde_json::to_vec(descriptor)
                .map_err(|error| MctsV4Error::Evaluation(error.to_string()))?;
            let mut hash = Sha256::new();
            hash.update(self.seed.to_le_bytes());
            hash.update(encoded);
            let digest = hash.finalize();
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&digest[..4]);
            logits.push((u32::from_le_bytes(bytes) as f64 / u32::MAX as f64) as f32);
        }
        Ok(FactorizedEvaluationV4 {
            value: self.value,
            action_logits: logits,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RootActionVisitV4 {
    pub action: Action,
    pub descriptor: ActionDescriptorV3,
    pub action_index: usize,
    pub prior: f32,
    pub visits: u32,
    pub q_root: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResultV4 {
    pub action: Action,
    pub descriptor: ActionDescriptorV3,
    pub action_index: usize,
    pub visits: Vec<RootActionVisitV4>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchDiagnosticsV4 {
    pub root_raw_value: f32,
    pub completed_simulations: usize,
    pub evaluator_calls: usize,
    /// Conditional rows scored, including eager complete-policy evaluations.
    /// Lazy single-child rows need no scorer call.
    pub scored_prefix_rows: usize,
    pub engine_apply_count: usize,
    pub prefix_engine_apply_count: usize,
    pub sampled_chance_actions: usize,
    pub sampled_chance_outcomes: usize,
    pub root_gumbel_candidates: usize,
    pub root_gumbel_halving_rounds: usize,
    pub maximum_depth: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MctsV4Error {
    InvalidConfig(String),
    Evaluation(String),
    Engine(String),
    FactorizedPolicy(String),
    Invariant(String),
    TerminalRoot,
}

impl fmt::Display for MctsV4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid V4 MCTS config: {message}"),
            Self::Evaluation(message) => write!(formatter, "V4 evaluator failed: {message}"),
            Self::Engine(message) => write!(formatter, "V4 MCTS engine error: {message}"),
            Self::FactorizedPolicy(message) => {
                write!(formatter, "V4 factorized policy error: {message}")
            }
            Self::Invariant(message) => write!(formatter, "V4 MCTS invariant failed: {message}"),
            Self::TerminalRoot => write!(formatter, "cannot search a terminal root"),
        }
    }
}

impl std::error::Error for MctsV4Error {}

#[derive(Clone, Debug)]
struct PrefixNodeV4 {
    token: Option<String>,
    parent: Option<usize>,
    children: Vec<usize>,
    action_index: Option<usize>,
    prior: f32,
    visits: u32,
    value_sum: f32,
    opened: bool,
    scored: bool,
}

impl PrefixNodeV4 {
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

#[derive(Clone, Debug)]
struct SearchTrieV4 {
    // Retaining the canonical trie makes this representation explicitly checked
    // against the public FactorizedTrieV4 contract. `arena` only adds MCTS stats.
    _canonical: FactorizedTrieV4,
    arena: Vec<PrefixNodeV4>,
    action_leaves: Vec<usize>,
}

#[derive(Default)]
struct BuildPrefixV4 {
    children: BTreeMap<String, BuildPrefixV4>,
    action_index: Option<usize>,
    mass: f32,
}

impl SearchTrieV4 {
    fn new(actions: &[ActionDescriptorV3], priors: &[f32]) -> Result<Self, MctsV4Error> {
        let canonical = build_factorized_trie_v4(actions)
            .map_err(|error| MctsV4Error::FactorizedPolicy(error.to_string()))?;
        let mut build = BuildPrefixV4::default();
        for (action_index, (action, prior)) in actions.iter().zip(priors).enumerate() {
            let path = factorized_action_path_v4(action)
                .map_err(|error| MctsV4Error::FactorizedPolicy(error.to_string()))?;
            build.mass += *prior;
            let mut current = &mut build;
            for token in path {
                current = current.children.entry(token).or_default();
                current.mass += *prior;
            }
            if current.action_index.replace(action_index).is_some() {
                return Err(MctsV4Error::Invariant(
                    "canonical leaf was not unique".into(),
                ));
            }
        }
        let mut trie = Self {
            _canonical: canonical,
            arena: Vec::new(),
            action_leaves: vec![usize::MAX; actions.len()],
        };
        trie.flatten(build, None, None, 1.0);
        if trie.action_leaves.contains(&usize::MAX) {
            return Err(MctsV4Error::Invariant(
                "an action has no canonical leaf".into(),
            ));
        }
        Ok(trie)
    }

    fn flatten(
        &mut self,
        node: BuildPrefixV4,
        parent: Option<usize>,
        token: Option<String>,
        parent_mass: f32,
    ) -> usize {
        let index = self.arena.len();
        let prior = if parent.is_none() {
            1.0
        } else if parent_mass > 0.0 {
            node.mass / parent_mass
        } else {
            0.0
        };
        self.arena.push(PrefixNodeV4 {
            token,
            parent,
            children: Vec::new(),
            action_index: node.action_index,
            prior,
            visits: 0,
            value_sum: 0.0,
            opened: parent.is_none(),
            scored: true,
        });
        if let Some(action_index) = node.action_index {
            self.action_leaves[action_index] = index;
        }
        let mass = node.mass;
        for (child_token, child) in node.children {
            let child_index = self.flatten(child, Some(index), Some(child_token), mass);
            self.arena[index].children.push(child_index);
        }
        index
    }

    fn path_to_action(&self, action_index: usize) -> Result<Vec<usize>, MctsV4Error> {
        let mut current = *self
            .action_leaves
            .get(action_index)
            .ok_or_else(|| MctsV4Error::Invariant("root action index is out of range".into()))?;
        let mut reversed = Vec::new();
        loop {
            reversed.push(current);
            match self.arena[current].parent {
                Some(parent) => current = parent,
                None => break,
            }
        }
        reversed.reverse();
        Ok(reversed)
    }
}

struct StateNodeV4 {
    state: GameState,
    visits: u32,
    value_sum: f32,
    expanded: bool,
    legal: Option<LegalActionSetV3>,
    information_key: Option<[u8; 32]>,
    chance_event_depth: u32,
    action_priors: Vec<f32>,
    trie: Option<SearchTrieV4>,
    prefix_scorer: Option<Box<dyn PrefixScorerV4>>,
    /// One deterministic child or multiple equally weighted chance samples per complete action.
    action_children: Vec<Vec<usize>>,
}

impl StateNodeV4 {
    fn new(state: GameState) -> Self {
        Self::with_chance_depth(state, 0)
    }

    fn with_chance_depth(state: GameState, chance_event_depth: u32) -> Self {
        Self {
            state,
            visits: 0,
            value_sum: 0.0,
            expanded: false,
            legal: None,
            information_key: None,
            chance_event_depth,
            action_priors: Vec::new(),
            trie: None,
            prefix_scorer: None,
            action_children: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PathEntryV4 {
    State(usize),
    Prefix { state: usize, prefix: usize },
}

#[derive(Clone, Debug)]
struct RootGumbelV4 {
    active: Vec<usize>,
    gumbels: Vec<f32>,
    round_start: Vec<u32>,
    quota: u32,
    issued: usize,
    completed_rounds: usize,
}

pub struct MctsV4Tree {
    config: MctsV4Config,
    states: Vec<StateNodeV4>,
    root_gumbel: Option<RootGumbelV4>,
    pub last_diagnostics: Option<SearchDiagnosticsV4>,
}

impl MctsV4Tree {
    pub fn new(config: MctsV4Config) -> Result<Self, MctsV4Error> {
        Ok(Self {
            config: config.validate()?,
            states: Vec::new(),
            root_gumbel: None,
            last_diagnostics: None,
        })
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.root_gumbel = None;
        self.last_diagnostics = None;
    }

    pub fn set_search_seed(&mut self, seed: u64) {
        self.config.search_seed = seed;
    }

    pub fn set_gumbel_seed(&mut self, seed: u64) {
        self.config.gumbel_seed = seed;
    }

    pub fn search(
        &mut self,
        root_state: &GameState,
        engine: &Engine,
        evaluator: &dyn FactorizedEvaluatorV4,
    ) -> Result<SearchResultV4, MctsV4Error> {
        if root_state.status != GameStatus::Ongoing {
            return Err(MctsV4Error::TerminalRoot);
        }
        // V4 intentionally starts a fresh sequential-halving tournament. Reusing
        // root counts would change both its budget and its elimination schedule.
        self.clear();
        self.states.push(StateNodeV4::new(root_state.clone()));
        let mut diagnostics = SearchDiagnosticsV4::default();
        diagnostics.root_raw_value = self.expand_state(0, engine, evaluator, &mut diagnostics)?;
        self.initialize_gumbel()?;
        diagnostics.root_gumbel_candidates = self
            .root_gumbel
            .as_ref()
            .map_or(0, |state| state.active.len());

        for _ in 0..self.config.num_simulations {
            let mut path = vec![PathEntryV4::State(0)];
            let mut state_index = 0usize;
            let mut depth = 0usize;
            loop {
                if self.states[state_index].state.status != GameStatus::Ongoing {
                    let value = terminal_value(&self.states[state_index].state)?;
                    let player = self.states[state_index].state.turn;
                    self.backup(&path, player, value)?;
                    break;
                }
                if !self.states[state_index].expanded {
                    let value =
                        self.expand_state(state_index, engine, evaluator, &mut diagnostics)?;
                    let player = self.states[state_index].state.turn;
                    self.backup(&path, player, value)?;
                    break;
                }

                let (action_index, prefix_path) = if state_index == 0 {
                    let action = self.select_root_gumbel_action()?;
                    let trie = self.states[0].trie.as_mut().ok_or_else(|| {
                        MctsV4Error::Invariant("expanded root has no trie".into())
                    })?;
                    let prefix_path = trie.path_to_action(action)?;
                    for prefix in &prefix_path {
                        trie.arena[*prefix].opened = true;
                    }
                    (action, prefix_path)
                } else {
                    self.select_internal_action(state_index, &mut diagnostics)?
                };
                for prefix in prefix_path {
                    path.push(PathEntryV4::Prefix {
                        state: state_index,
                        prefix,
                    });
                }

                // This assertion is the boundary: no engine call can occur while
                // selection is at an incomplete virtual prefix.
                let leaf = self.states[state_index]
                    .trie
                    .as_ref()
                    .and_then(|trie| trie.action_leaves.get(action_index).copied())
                    .ok_or_else(|| {
                        MctsV4Error::Invariant("selected action has no complete leaf".into())
                    })?;
                if self.states[state_index]
                    .trie
                    .as_ref()
                    .and_then(|trie| trie.arena[leaf].action_index)
                    != Some(action_index)
                {
                    return Err(MctsV4Error::Invariant(
                        "engine apply attempted before a complete leaf".into(),
                    ));
                }

                let next = self.select_action_outcome(
                    state_index,
                    action_index,
                    engine,
                    &mut diagnostics,
                )?;
                state_index = next;
                depth += 1;
                diagnostics.maximum_depth = diagnostics.maximum_depth.max(depth);
                path.push(PathEntryV4::State(state_index));
            }
            diagnostics.completed_simulations += 1;
        }
        diagnostics.root_gumbel_halving_rounds = self
            .root_gumbel
            .as_ref()
            .map_or(0, |state| state.completed_rounds);
        let root_trie = self.states[0]
            .trie
            .as_ref()
            .ok_or_else(|| MctsV4Error::Invariant("expanded root has no trie".into()))?;
        let complete_visits = root_trie
            .action_leaves
            .iter()
            .try_fold(0u32, |total, leaf| {
                total.checked_add(root_trie.arena[*leaf].visits)
            })
            .ok_or_else(|| {
                MctsV4Error::Invariant("complete-action visit total overflowed".into())
            })?;
        if complete_visits != diagnostics.completed_simulations as u32 {
            return Err(MctsV4Error::Invariant(format!(
                "root complete visits {complete_visits} do not equal simulation budget {}",
                diagnostics.completed_simulations,
            )));
        }
        let result = self.build_result(root_state)?;
        self.last_diagnostics = Some(diagnostics);
        Ok(result)
    }

    fn select_action_outcome(
        &mut self,
        state_index: usize,
        action_index: usize,
        engine: &Engine,
        diagnostics: &mut SearchDiagnosticsV4,
    ) -> Result<usize, MctsV4Error> {
        if self.states[state_index]
            .action_children
            .get(action_index)
            .is_none()
        {
            return Err(MctsV4Error::Invariant(
                "selected action index is outside the expanded state".into(),
            ));
        }
        if self.states[state_index].action_children[action_index].is_empty() {
            let parent_state = self.states[state_index].state.clone();
            let action = self.states[state_index]
                .legal
                .as_ref()
                .ok_or_else(|| MctsV4Error::Invariant("expanded state has no legal set".into()))?
                .decode_action(action_index, &parent_state)
                .map_err(|error| MctsV4Error::Engine(error.to_string()))?;
            let information_key = self.states[state_index].information_key.ok_or_else(|| {
                MctsV4Error::Invariant("expanded state has no public information key".into())
            })?;
            let chance_event_depth = self.states[state_index].chance_event_depth;
            let sample_transition = |sample_lane: usize| {
                let seed = derive_chance_lane_seed(
                    self.config.search_seed,
                    &information_key,
                    sample_lane as u64,
                    chance_event_depth,
                );
                let mut chance = SearchChanceContext::new(seed);
                engine
                    .apply_for_search(&parent_state, action.clone(), &mut chance)
                    .map_err(|error| MctsV4Error::Engine(error.to_string()))
            };

            let first = sample_transition(0)?;
            diagnostics.engine_apply_count += 1;
            let sampled_chance = first.sampled_chance;
            let sample_count = if sampled_chance {
                self.config.chance_outcome_samples
            } else {
                1
            };
            let mut transitions = Vec::with_capacity(sample_count);
            transitions.push(first.transition);
            for sample_lane in 1..sample_count {
                let sampled = sample_transition(sample_lane)?;
                diagnostics.engine_apply_count += 1;
                if !sampled.sampled_chance {
                    return Err(MctsV4Error::Invariant(
                        "chance transition changed classification between sample lanes".into(),
                    ));
                }
                transitions.push(sampled.transition);
            }
            if sampled_chance {
                diagnostics.sampled_chance_actions += 1;
                diagnostics.sampled_chance_outcomes += transitions.len();
            }

            let child_chance_depth = chance_event_depth + u32::from(sampled_chance);
            let mut children = Vec::with_capacity(transitions.len());
            for transition in transitions {
                let child = self.states.len();
                self.states.push(StateNodeV4::with_chance_depth(
                    transition.state,
                    child_chance_depth,
                ));
                children.push(child);
            }
            self.states[state_index].action_children[action_index] = children;
        }

        self.states[state_index].action_children[action_index]
            .iter()
            .copied()
            .min_by_key(|child| (self.states[*child].visits, *child))
            .ok_or_else(|| MctsV4Error::Invariant("action has no sampled outcome".into()))
    }

    fn expand_state(
        &mut self,
        state_index: usize,
        engine: &Engine,
        evaluator: &dyn FactorizedEvaluatorV4,
        diagnostics: &mut SearchDiagnosticsV4,
    ) -> Result<f32, MctsV4Error> {
        let legal = LegalActionSetV3::from_state(&self.states[state_index].state, engine)
            .map_err(|error| MctsV4Error::Engine(error.to_string()))?;
        if legal.is_empty() {
            return Err(MctsV4Error::Invariant(
                "ongoing state has no legal actions".into(),
            ));
        }
        // Root Gumbel needs all complete-action priors. Internal leaf expansion
        // needs only the value; policy rows wait until selection reaches them.
        let prepared = if state_index == 0 {
            None
        } else {
            evaluator.prepare_prefix_evaluation_v4(
                &self.states[state_index].state,
                &legal,
                engine,
            )?
        };
        let (value, priors, prefix_scorer) = if let Some(prepared) = prepared {
            (
                prepared.value,
                vec![0.0; legal.len()],
                Some(prepared.scorer),
            )
        } else {
            let evaluation =
                evaluator.evaluate_v4(&self.states[state_index].state, &legal, engine)?;
            let priors = normalized_policy(&evaluation.action_logits, legal.len())?;
            (evaluation.value, priors, None)
        };
        diagnostics.evaluator_calls += 1;
        if !value.is_finite() {
            return Err(MctsV4Error::Evaluation("value is not finite".into()));
        }
        let mut trie = SearchTrieV4::new(&legal.actions, &priors)?;
        if prefix_scorer.is_some() {
            for prefix in &mut trie.arena {
                prefix.scored = false;
            }
        } else {
            diagnostics.scored_prefix_rows += trie
                .arena
                .iter()
                .filter(|prefix| !prefix.children.is_empty())
                .count();
        }
        let information_key = search_information_key_v1(&self.states[state_index].state, &legal)?;
        // Draft offers are already public to the actor. They are not subject to
        // proposal-style hiding, even when an internal Draft prefix is wide.
        if matches!(self.states[state_index].state.phase, GamePhase::Draft(_)) {
            for prefix in &mut trie.arena {
                prefix.opened = true;
            }
        }
        let action_count = legal.len();
        let node = &mut self.states[state_index];
        node.legal = Some(legal);
        node.information_key = Some(information_key);
        node.action_priors = priors;
        node.trie = Some(trie);
        node.prefix_scorer = prefix_scorer;
        node.action_children = vec![Vec::new(); action_count];
        node.expanded = true;
        Ok(value)
    }

    fn select_internal_action(
        &mut self,
        state_index: usize,
        diagnostics: &mut SearchDiagnosticsV4,
    ) -> Result<(usize, Vec<usize>), MctsV4Error> {
        let mut prefix = 0usize;
        let mut path = vec![0usize];
        loop {
            let action = self.states[state_index]
                .trie
                .as_ref()
                .and_then(|trie| trie.arena[prefix].action_index);
            if let Some(action_index) = action {
                return Ok((action_index, path));
            }
            self.score_prefix(state_index, prefix, diagnostics)?;
            self.widen_prefix(state_index, prefix)?;
            let trie = self.states[state_index]
                .trie
                .as_ref()
                .ok_or_else(|| MctsV4Error::Invariant("expanded state has no trie".into()))?;
            let parent = &trie.arena[prefix];
            let parent_visits = parent.visits.max(1) as f32;
            let parent_q = parent.q();
            let selected = parent
                .children
                .iter()
                .copied()
                .filter(|child| trie.arena[*child].opened)
                .max_by(|left, right| {
                    let score = |child: usize| {
                        let candidate = &trie.arena[child];
                        let q = if candidate.visits == 0 {
                            parent_q - self.config.fpu_reduction
                        } else {
                            candidate.q()
                        };
                        q + self.config.c_puct * candidate.prior * parent_visits.sqrt()
                            / (1.0 + candidate.visits as f32)
                    };
                    score(*left)
                        .total_cmp(&score(*right))
                        .then_with(|| trie.arena[*right].token.cmp(&trie.arena[*left].token))
                })
                .ok_or_else(|| {
                    MctsV4Error::Invariant("progressive widening opened no prefix".into())
                })?;
            prefix = selected;
            path.push(prefix);
        }
    }

    fn score_prefix(
        &mut self,
        state_index: usize,
        prefix: usize,
        diagnostics: &mut SearchDiagnosticsV4,
    ) -> Result<(), MctsV4Error> {
        let node = &mut self.states[state_index];
        let trie = node
            .trie
            .as_mut()
            .ok_or_else(|| MctsV4Error::Invariant("expanded state has no trie".into()))?;
        if trie.arena[prefix].scored {
            return Ok(());
        }
        let children = trie.arena[prefix].children.clone();
        let priors =
            if children.len() == 1 {
                // A forced edge has conditional probability one, independent of logits.
                vec![1.0]
            } else {
                let mut tokens = Vec::new();
                let mut current = prefix;
                while let Some(parent) = trie.arena[current].parent {
                    tokens.push(trie.arena[current].token.clone().ok_or_else(|| {
                        MctsV4Error::Invariant("non-root prefix has no token".into())
                    })?);
                    current = parent;
                }
                tokens.reverse();
                let child_tokens = children
                    .iter()
                    .map(|child| {
                        trie.arena[*child].token.clone().ok_or_else(|| {
                            MctsV4Error::Invariant("child prefix has no token".into())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let scorer = node.prefix_scorer.as_ref().ok_or_else(|| {
                    MctsV4Error::Invariant("unscored prefix has no scorer".into())
                })?;
                let logits = scorer.score_prefix_v4(&tokens, &child_tokens)?;
                let priors = normalized_policy(&logits, children.len())?;
                diagnostics.scored_prefix_rows += 1;
                priors
            };
        for (child, prior) in children.into_iter().zip(priors) {
            trie.arena[child].prior = prior;
        }
        trie.arena[prefix].scored = true;
        Ok(())
    }

    fn widen_prefix(&mut self, state_index: usize, prefix: usize) -> Result<(), MctsV4Error> {
        let trie = self.states[state_index]
            .trie
            .as_mut()
            .ok_or_else(|| MctsV4Error::Invariant("expanded state has no trie".into()))?;
        let node = &trie.arena[prefix];
        let limit = (self.config.widening_k0 as f32
            + self.config.widening_c * (node.visits as f32).powf(self.config.widening_alpha))
        .floor() as usize;
        let limit = limit.max(1).min(node.children.len());
        let mut candidates = node.children.clone();
        candidates.sort_by(|left, right| {
            trie.arena[*right]
                .prior
                .total_cmp(&trie.arena[*left].prior)
                .then_with(|| trie.arena[*left].token.cmp(&trie.arena[*right].token))
        });
        for child in candidates.into_iter().take(limit) {
            trie.arena[child].opened = true;
        }
        Ok(())
    }

    fn initialize_gumbel(&mut self) -> Result<(), MctsV4Error> {
        let priors = &self.states[0].action_priors;
        let candidate_count = priors
            .len()
            .min(self.config.gumbel_candidates)
            .min(self.config.num_simulations);
        if candidate_count == 0 {
            return Err(MctsV4Error::Invariant(
                "root Gumbel has no candidate".into(),
            ));
        }
        let information_key = self.states[0].information_key.ok_or_else(|| {
            MctsV4Error::Invariant("expanded root has no public information key".into())
        })?;
        let mut rng = StdRng::seed_from_u64(derive_gumbel_seed(
            self.config.gumbel_seed,
            &information_key,
        ));
        let gumbels = (0..priors.len())
            .map(|_| {
                let uniform = rng.gen_range(f32::EPSILON..1.0);
                -(-uniform.ln()).ln()
            })
            .collect::<Vec<_>>();
        let mut ranked = (0..priors.len()).collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            root_raw_score(priors[*right], gumbels[*right])
                .total_cmp(&root_raw_score(priors[*left], gumbels[*left]))
                .then_with(|| left.cmp(right))
        });
        ranked.truncate(candidate_count);
        let rounds = ceil_log2(candidate_count).max(1);
        let quota = (self.config.num_simulations / (candidate_count * rounds)).max(1) as u32;
        self.root_gumbel = Some(RootGumbelV4 {
            active: ranked,
            gumbels,
            round_start: vec![0; priors.len()],
            quota,
            issued: 0,
            completed_rounds: 0,
        });
        Ok(())
    }

    fn select_root_gumbel_action(&mut self) -> Result<usize, MctsV4Error> {
        self.advance_gumbel_round()?;
        let state = self
            .root_gumbel
            .as_ref()
            .ok_or_else(|| MctsV4Error::Invariant("root Gumbel was not initialized".into()))?;
        let trie = self.states[0]
            .trie
            .as_ref()
            .ok_or_else(|| MctsV4Error::Invariant("expanded root has no trie".into()))?;
        let selected = state
            .active
            .iter()
            .copied()
            .min_by_key(|action| {
                let visits = trie.arena[trie.action_leaves[*action]].visits;
                (visits.saturating_sub(state.round_start[*action]), *action)
            })
            .ok_or_else(|| MctsV4Error::Invariant("root Gumbel has no active action".into()))?;
        self.root_gumbel.as_mut().unwrap().issued += 1;
        Ok(selected)
    }

    fn advance_gumbel_round(&mut self) -> Result<(), MctsV4Error> {
        let should_advance = {
            let state = self
                .root_gumbel
                .as_ref()
                .ok_or_else(|| MctsV4Error::Invariant("root Gumbel was not initialized".into()))?;
            if state.active.len() <= 1 {
                false
            } else {
                let trie = self.states[0].trie.as_ref().unwrap();
                state.active.iter().all(|action| {
                    trie.arena[trie.action_leaves[*action]]
                        .visits
                        .saturating_sub(state.round_start[*action])
                        >= state.quota
                })
            }
        };
        if !should_advance {
            return Ok(());
        }

        let remaining = self
            .config
            .num_simulations
            .saturating_sub(self.root_gumbel.as_ref().unwrap().issued);
        if remaining == 0 {
            return Ok(());
        }
        let priors = self.states[0].action_priors.clone();
        let trie = self.states[0].trie.as_ref().unwrap();
        let mut state = self.root_gumbel.take().unwrap();
        state.active.sort_by(|left, right| {
            let score = |action: usize| {
                let leaf = &trie.arena[trie.action_leaves[action]];
                root_raw_score(priors[action], state.gumbels[action])
                    + self.config.gumbel_q_scale * leaf.q()
            };
            score(*right)
                .total_cmp(&score(*left))
                .then_with(|| left.cmp(right))
        });
        let survivors = state.active.len().div_ceil(2);
        state.active.truncate(survivors);
        state.completed_rounds += 1;
        for action in &state.active {
            state.round_start[*action] = trie.arena[trie.action_leaves[*action]].visits;
        }
        let rounds = ceil_log2(state.active.len()).max(1);
        state.quota = (remaining / (state.active.len() * rounds)).max(1) as u32;
        self.root_gumbel = Some(state);
        Ok(())
    }

    fn backup(
        &mut self,
        path: &[PathEntryV4],
        leaf_player: Player,
        value: f32,
    ) -> Result<(), MctsV4Error> {
        if !value.is_finite() {
            return Err(MctsV4Error::Invariant(
                "cannot back up a non-finite value".into(),
            ));
        }
        for entry in path.iter().rev() {
            match *entry {
                PathEntryV4::State(index) => {
                    let signed =
                        perspective_value(self.states[index].state.turn, leaf_player, value);
                    self.states[index].visits += 1;
                    self.states[index].value_sum += signed;
                }
                PathEntryV4::Prefix { state, prefix } => {
                    let signed =
                        perspective_value(self.states[state].state.turn, leaf_player, value);
                    let trie = self.states[state].trie.as_mut().ok_or_else(|| {
                        MctsV4Error::Invariant("backup prefix has no trie".into())
                    })?;
                    trie.arena[prefix].visits += 1;
                    trie.arena[prefix].value_sum += signed;
                }
            }
        }
        Ok(())
    }

    fn build_result(&self, root_state: &GameState) -> Result<SearchResultV4, MctsV4Error> {
        let root = &self.states[0];
        let legal = root
            .legal
            .as_ref()
            .ok_or_else(|| MctsV4Error::Invariant("root has no legal set".into()))?;
        let trie = root
            .trie
            .as_ref()
            .ok_or_else(|| MctsV4Error::Invariant("root has no trie".into()))?;
        let mut visits = Vec::with_capacity(legal.len());
        for action_index in 0..legal.len() {
            let leaf = &trie.arena[trie.action_leaves[action_index]];
            visits.push(RootActionVisitV4 {
                action: legal
                    .decode_action(action_index, root_state)
                    .map_err(|error| MctsV4Error::Engine(error.to_string()))?,
                descriptor: legal.actions[action_index].clone(),
                action_index,
                prior: root.action_priors[action_index],
                visits: leaf.visits,
                q_root: leaf.q(),
            });
        }
        let active = &self
            .root_gumbel
            .as_ref()
            .ok_or_else(|| MctsV4Error::Invariant("root Gumbel is missing".into()))?
            .active;
        let selected = active
            .iter()
            .copied()
            .max_by(|left, right| {
                let score = |action: usize| {
                    let leaf = &trie.arena[trie.action_leaves[action]];
                    (
                        root_raw_score(
                            root.action_priors[action],
                            self.root_gumbel.as_ref().unwrap().gumbels[action],
                        ) + self.config.gumbel_q_scale * leaf.q(),
                        leaf.visits,
                    )
                };
                let left_score = score(*left);
                let right_score = score(*right);
                left_score
                    .0
                    .total_cmp(&right_score.0)
                    .then_with(|| left_score.1.cmp(&right_score.1))
                    .then_with(|| right.cmp(left))
            })
            .ok_or_else(|| MctsV4Error::Invariant("root Gumbel has no survivor".into()))?;
        Ok(SearchResultV4 {
            action: visits[selected].action.clone(),
            descriptor: visits[selected].descriptor.clone(),
            action_index: selected,
            visits,
        })
    }
}

fn normalized_policy(logits: &[f32], expected: usize) -> Result<Vec<f32>, MctsV4Error> {
    if logits.len() != expected {
        return Err(MctsV4Error::Evaluation(format!(
            "expected {expected} action logits, got {}",
            logits.len()
        )));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(MctsV4Error::Evaluation(
            "action logits contain a non-finite value".into(),
        ));
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights = logits
        .iter()
        .map(|logit| (*logit - maximum).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(MctsV4Error::Evaluation(
            "action logits cannot be normalized".into(),
        ));
    }
    Ok(weights.into_iter().map(|weight| weight / total).collect())
}

fn root_raw_score(prior: f32, gumbel: f32) -> f32 {
    prior.max(f32::MIN_POSITIVE).ln() + gumbel
}

fn search_information_key_v1(
    state: &GameState,
    legal: &LegalActionSetV3,
) -> Result<[u8; 32], MctsV4Error> {
    let observation = encode_observation_v3(state);
    let payload = serde_json::to_vec(&(
        "augmentia.search-information.v1",
        state.turn.index() as u8,
        observation,
        legal,
    ))
    .map_err(|error| MctsV4Error::Invariant(format!("information key encoding failed: {error}")))?;
    let digest = Sha256::digest(payload);
    let mut key = [0u8; 32];
    key.copy_from_slice(&digest);
    Ok(key)
}

fn derive_gumbel_seed(gumbel_seed: u64, information_key: &[u8; 32]) -> u64 {
    let mut suffix = [0u8; 8];
    suffix.copy_from_slice(&information_key[16..24]);
    let mut value = gumbel_seed ^ u64::from_le_bytes(suffix) ^ 0x4755_4d42_454c_5634;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn derive_chance_lane_seed(
    search_seed: u64,
    information_key: &[u8; 32],
    sample_lane: u64,
    chance_event_depth: u32,
) -> u64 {
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&information_key[..8]);
    let mut value = search_seed
        ^ u64::from_le_bytes(prefix)
        ^ sample_lane.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ u64::from(chance_event_depth).wrapping_mul(0xd1b5_4a32_d192_ed03)
        ^ 0x4348_414e_4345_5634;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

fn perspective_value(owner: Player, leaf_player: Player, value: f32) -> f32 {
    if owner == leaf_player { value } else { -value }
}

fn terminal_value(state: &GameState) -> Result<f32, MctsV4Error> {
    match state.status {
        GameStatus::Won { winner, .. } => Ok(if winner == state.turn { 1.0 } else { -1.0 }),
        GameStatus::Draw { .. } => Ok(0.0),
        GameStatus::Ongoing => Err(MctsV4Error::Invariant(
            "requested terminal value for an ongoing state".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use augment_core::{BUNDLED_RULESET_ID, Board, DraftState, GamePhase, Mode, Piece, PieceKind};
    use std::collections::BTreeSet;

    fn tactical_state() -> GameState {
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
        state
    }

    fn config(simulations: usize, seed: u64) -> MctsV4Config {
        MctsV4Config {
            num_simulations: simulations,
            gumbel_candidates: 8,
            gumbel_seed: seed,
            widening_k0: 1,
            widening_c: 1.0,
            ..MctsV4Config::default()
        }
    }

    fn evaluator() -> DeterministicFactorizedEvaluatorV4 {
        DeterministicFactorizedEvaluatorV4 {
            seed: 19,
            value: 0.125,
        }
    }

    #[test]
    fn every_canonical_leaf_maps_to_one_complete_legal_action() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = tactical_state();
        let legal = LegalActionSetV3::from_state(&state, &engine).unwrap();
        let priors = vec![1.0 / legal.len() as f32; legal.len()];
        let trie = SearchTrieV4::new(&legal.actions, &priors).unwrap();
        assert_eq!(trie.action_leaves.len(), legal.len());
        for action_index in 0..legal.len() {
            let path = trie.path_to_action(action_index).unwrap();
            let leaf = &trie.arena[*path.last().unwrap()];
            assert_eq!(leaf.token.as_deref(), Some("complete:action"));
            assert_eq!(leaf.action_index, Some(action_index));
            let decoded = legal.decode_action(action_index, &state).unwrap();
            assert!(engine.legal_actions(&state).unwrap().contains(&decoded));
        }
    }

    #[test]
    fn engine_apply_is_guarded_by_complete_leaf() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = tactical_state();
        let mut tree = MctsV4Tree::new(config(31, 3)).unwrap();
        tree.search(&state, &engine, &evaluator()).unwrap();
        let diagnostics = tree.last_diagnostics.as_ref().unwrap();
        assert!(diagnostics.engine_apply_count > 0);
        assert_eq!(diagnostics.prefix_engine_apply_count, 0);
        for node in &tree.states {
            if let Some(trie) = &node.trie {
                for (action, leaf) in trie.action_leaves.iter().enumerate() {
                    assert_eq!(trie.arena[*leaf].action_index, Some(action));
                }
            }
        }
    }

    fn assert_prefix_conservation(trie: &SearchTrieV4, prefix: usize) {
        let node = &trie.arena[prefix];
        if !node.children.is_empty() {
            let total = node
                .children
                .iter()
                .map(|child| trie.arena[*child].visits)
                .sum::<u32>();
            assert_eq!(node.visits, total, "prefix {:?}", node.token);
            for child in &node.children {
                assert_prefix_conservation(trie, *child);
            }
        }
    }

    #[test]
    fn factorized_visits_are_conserved_at_every_visited_prefix() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = tactical_state();
        let mut tree = MctsV4Tree::new(config(97, 5)).unwrap();
        let result = tree.search(&state, &engine, &evaluator()).unwrap();
        assert_eq!(
            result.visits.iter().map(|visit| visit.visits).sum::<u32>(),
            97
        );
        let root_trie = tree.states[0].trie.as_ref().unwrap();
        assert_eq!(root_trie.arena[0].visits, 97);
        assert_eq!(tree.states[0].visits, 97);
        assert_prefix_conservation(root_trie, 0);
    }

    #[test]
    fn backup_flips_only_when_player_to_move_changes() {
        let white = tactical_state();
        let mut still_white = white.clone();
        still_white.ply += 1;
        let mut black = still_white.clone();
        black.turn = Player::Black;
        let mut tree = MctsV4Tree::new(config(1, 1)).unwrap();
        tree.states = vec![
            StateNodeV4::new(white),
            StateNodeV4::new(still_white),
            StateNodeV4::new(black),
        ];
        tree.backup(
            &[
                PathEntryV4::State(0),
                PathEntryV4::State(1),
                PathEntryV4::State(2),
            ],
            Player::Black,
            0.75,
        )
        .unwrap();
        assert_eq!(tree.states[0].value_sum, -0.75);
        assert_eq!(tree.states[1].value_sum, -0.75);
        assert_eq!(tree.states[2].value_sum, 0.75);
    }

    #[test]
    fn search_random_domains_depend_on_public_key_lane_and_depth() {
        let first = [0u8; 32];
        let mut second = first;
        second[0] = 1;
        second[16] = 1;
        assert_ne!(
            derive_chance_lane_seed(7, &first, 0, 0),
            derive_chance_lane_seed(7, &second, 0, 0),
        );
        assert_ne!(
            derive_chance_lane_seed(7, &first, 0, 0),
            derive_chance_lane_seed(7, &first, 1, 0),
        );
        assert_ne!(
            derive_chance_lane_seed(7, &first, 0, 0),
            derive_chance_lane_seed(7, &first, 0, 1),
        );
        assert_ne!(
            derive_gumbel_seed(7, &first),
            derive_gumbel_seed(7, &second),
        );
    }

    #[test]
    fn public_information_key_excludes_live_rng() {
        let engine = Engine::bundled_20260818().unwrap();
        let first = GameState::from_board(
            BUNDLED_RULESET_ID,
            Mode::Standard,
            Board::standard(),
            Player::White,
            11,
        );
        let second = GameState::from_board(
            BUNDLED_RULESET_ID,
            Mode::Standard,
            Board::standard(),
            Player::White,
            99,
        );
        assert_ne!(first.rng_state(), second.rng_state());
        let first_legal = LegalActionSetV3::from_state(&first, &engine).unwrap();
        let second_legal = LegalActionSetV3::from_state(&second, &engine).unwrap();
        assert_eq!(first_legal, second_legal);
        assert_eq!(
            search_information_key_v1(&first, &first_legal).unwrap(),
            search_information_key_v1(&second, &second_legal).unwrap(),
        );
    }

    #[test]
    fn chance_search_is_live_rng_independent_and_samples_multiple_outcomes() {
        let engine = Engine::bundled_20260818().unwrap();
        let template = engine.new_game(Mode::Standard, 2026).unwrap();
        let GamePhase::Draft(template_draft) = template.phase else {
            panic!("new standard game must begin with a Draft")
        };
        let make_state = |seed| {
            let mut state = GameState::from_board(
                BUNDLED_RULESET_ID,
                Mode::Standard,
                Board::standard(),
                Player::White,
                seed,
            );
            state.phase = GamePhase::Draft(DraftState {
                phase: template_draft.phase,
                offers: template_draft.offers.clone(),
                resume_turn: template_draft.resume_turn,
            });
            state
        };
        let left_state = make_state(11);
        let right_state = make_state(99);
        let config = MctsV4Config {
            num_simulations: 48,
            gumbel_candidates: 3,
            chance_outcome_samples: 4,
            search_seed: 0x1234,
            gumbel_seed: 0x5678,
            ..MctsV4Config::default()
        };
        let mut left = MctsV4Tree::new(config).unwrap();
        let mut right = MctsV4Tree::new(config).unwrap();
        let left_result = left.search(&left_state, &engine, &evaluator()).unwrap();
        let right_result = right.search(&right_state, &engine, &evaluator()).unwrap();

        assert_eq!(left_result, right_result);
        assert_eq!(left.last_diagnostics, right.last_diagnostics);
        let diagnostics = left.last_diagnostics.as_ref().unwrap();
        assert!(diagnostics.sampled_chance_actions >= 3);
        assert_eq!(
            diagnostics.sampled_chance_outcomes,
            diagnostics.sampled_chance_actions * config.chance_outcome_samples,
        );
        assert_eq!(left_state.rng_state(), 11);
        assert_eq!(right_state.rng_state(), 99);
        for outcomes in &left.states[0].action_children {
            assert_eq!(outcomes.len(), config.chance_outcome_samples);
            let visits = outcomes
                .iter()
                .map(|child| left.states[*child].visits)
                .collect::<Vec<_>>();
            let minimum = visits.iter().copied().min().unwrap();
            let maximum = visits.iter().copied().max().unwrap();
            assert!(
                maximum - minimum <= 1,
                "chance lanes were not visited evenly"
            );
        }

        let sampled_offers = |tree: &MctsV4Tree| {
            tree.states[0]
                .action_children
                .iter()
                .flatten()
                .filter_map(|child| match &tree.states[*child].state.phase {
                    GamePhase::Draft(draft) => Some(draft.offers.clone()),
                    _ => None,
                })
                .collect::<BTreeSet<_>>()
        };
        let left_offers = sampled_offers(&left);
        assert_eq!(left_offers, sampled_offers(&right));
        assert!(left_offers.len() > 1);
    }

    #[test]
    fn simulation_budget_is_exact_and_seed_is_deterministic() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = tactical_state();
        let mut first = MctsV4Tree::new(config(113, 0xfeed)).unwrap();
        let mut second = MctsV4Tree::new(config(113, 0xfeed)).unwrap();
        let first_result = first.search(&state, &engine, &evaluator()).unwrap();
        let second_result = second.search(&state, &engine, &evaluator()).unwrap();
        assert_eq!(
            first
                .last_diagnostics
                .as_ref()
                .unwrap()
                .completed_simulations,
            113
        );
        assert_eq!(first_result, second_result);
        assert_eq!(first.last_diagnostics, second.last_diagnostics);
        assert!(
            first
                .last_diagnostics
                .as_ref()
                .unwrap()
                .root_gumbel_halving_rounds
                > 0
        );
    }

    struct CountingPrefixScorer {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        invalid: bool,
    }

    impl PrefixScorerV4 for CountingPrefixScorer {
        fn score_prefix_v4(
            &self,
            _prefix: &[String],
            children: &[String],
        ) -> Result<Vec<f32>, MctsV4Error> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(vec![
                if self.invalid { f32::NAN } else { 0.0 };
                children.len()
            ])
        }
    }

    #[test]
    fn lazy_rows_are_validated_cached_and_forced_edges_need_no_score() {
        let engine = Engine::bundled_20260818().unwrap();
        let state = tactical_state();
        let legal = LegalActionSetV3::from_state(&state, &engine).unwrap();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut trie = SearchTrieV4::new(&legal.actions, &vec![0.0; legal.len()]).unwrap();
        for row in &mut trie.arena {
            row.scored = false;
        }
        let forced = trie
            .arena
            .iter()
            .position(|row| row.children.len() == 1)
            .unwrap();
        let branch = trie
            .arena
            .iter()
            .position(|row| row.children.len() > 1)
            .unwrap();
        let mut node = StateNodeV4::new(state);
        node.trie = Some(trie);
        node.prefix_scorer = Some(Box::new(CountingPrefixScorer {
            calls: calls.clone(),
            invalid: false,
        }));
        let mut tree = MctsV4Tree::new(config(1, 4)).unwrap();
        tree.states.push(node);
        let mut diagnostics = SearchDiagnosticsV4::default();
        tree.score_prefix(0, forced, &mut diagnostics).unwrap();
        tree.score_prefix(0, forced, &mut diagnostics).unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
        let trie = tree.states[0].trie.as_ref().unwrap();
        assert_eq!(trie.arena[trie.arena[forced].children[0]].prior, 1.0);
        tree.states[0].prefix_scorer = Some(Box::new(CountingPrefixScorer {
            calls: calls.clone(),
            invalid: true,
        }));
        assert!(tree.score_prefix(0, branch, &mut diagnostics).is_err());
        assert!(!tree.states[0].trie.as_ref().unwrap().arena[branch].scored);
        tree.states[0].prefix_scorer = Some(Box::new(CountingPrefixScorer {
            calls: calls.clone(),
            invalid: false,
        }));
        tree.score_prefix(0, branch, &mut diagnostics).unwrap();
        tree.score_prefix(0, branch, &mut diagnostics).unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(diagnostics.scored_prefix_rows, 1);
        let trie = tree.states[0].trie.as_ref().unwrap();
        let sum: f32 = trie.arena[branch]
            .children
            .iter()
            .map(|child| trie.arena[*child].prior)
            .sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lazy_onnx_scores_only_visited_rows_and_matches_eager_conditionals() {
        use crate::V4OnnxRuntime;
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            eprintln!("skipping lazy ONNX parity because ORT_DYLIB_PATH is not configured");
            return;
        }
        let metadata_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../train/v3/fixtures/v4_bundle/v4-model.v4-bundle.metadata.json");
        let runtime =
            V4OnnxRuntime::from_bundle(metadata_path, BUNDLED_RULESET_ID, &"0".repeat(64)).unwrap();
        let engine = Engine::bundled_20260818().unwrap();
        let state = GameState::from_board(
            BUNDLED_RULESET_ID,
            Mode::Standard,
            Board::standard(),
            Player::White,
            11,
        );
        let legal = LegalActionSetV3::from_state(&state, &engine).unwrap();
        runtime.evaluate_factorized(&state, &legal).unwrap();
        let runs = runtime.session_runs();
        let prepared = runtime
            .prepare_prefix_evaluation_v4(&state, &legal, &engine)
            .unwrap()
            .unwrap();
        assert_eq!(
            runtime.session_runs(),
            runs,
            "warm preparation only reuses encoding"
        );
        let decision = crate::runtime_v4::build_factorized_decision_v4(&legal.actions).unwrap();
        let row = decision
            .rows
            .iter()
            .find(|row| row.children.len() > 1)
            .unwrap();
        prepared
            .scorer
            .score_prefix_v4(&row.prefix, &row.children)
            .unwrap();
        assert_eq!(
            runtime.session_runs(),
            runs + 1,
            "lazy scorer shares the runtime counter"
        );
        let config = MctsV4Config {
            num_simulations: 48,
            ..MctsV4Config::default()
        };
        let mut lazy = MctsV4Tree::new(config).unwrap();
        let result = lazy.search(&state, &engine, &runtime).unwrap();
        let mut scored = 0;
        let mut all_rows = 0;
        let mut unvisited_branch_rows = 0;
        for (index, node) in lazy.states.iter().enumerate() {
            let Some(trie) = &node.trie else { continue };
            assert_prefix_conservation(trie, 0);
            let eager = runtime
                .evaluate_factorized(&node.state, node.legal.as_ref().unwrap())
                .unwrap();
            let priors = normalized_policy(&eager.action_logits, node.action_priors.len()).unwrap();
            let eager_trie =
                SearchTrieV4::new(&node.legal.as_ref().unwrap().actions, &priors).unwrap();
            for (prefix, row) in trie.arena.iter().enumerate() {
                if row.children.is_empty() {
                    continue;
                }
                all_rows += 1;
                if index == 0 {
                    scored += 1;
                    continue;
                }
                assert_eq!(
                    row.scored,
                    row.visits > 0,
                    "prefix {prefix} of state {index}"
                );
                if row.children.len() > 1 {
                    if row.scored {
                        scored += 1;
                    } else {
                        unvisited_branch_rows += 1;
                    }
                }
                if row.scored {
                    // Direct conditional softmax and complete-action subtree
                    // sums differ in f32 rounding. This is numeric parity,
                    // not a general promise of bit-identical search trees.
                    for child in &row.children {
                        assert!(
                            (trie.arena[*child].prior - eager_trie.arena[*child].prior).abs()
                                < 2e-5
                        );
                    }
                }
            }
        }
        let diagnostics = lazy.last_diagnostics.as_ref().unwrap();
        assert_eq!(diagnostics.scored_prefix_rows, scored);
        assert!(scored < all_rows);
        assert!(unvisited_branch_rows > 0);
        assert_eq!(
            result.visits.iter().map(|visit| visit.visits).sum::<u32>(),
            48
        );
        struct Eager<'a>(&'a V4OnnxRuntime);
        impl FactorizedEvaluatorV4 for Eager<'_> {
            fn evaluate_v4(
                &self,
                state: &GameState,
                legal: &LegalActionSetV3,
                engine: &Engine,
            ) -> Result<FactorizedEvaluationV4, MctsV4Error> {
                self.0.evaluate_v4(state, legal, engine)
            }
        }
        let mut eager = MctsV4Tree::new(config).unwrap();
        let eager_result = eager.search(&state, &engine, &Eager(&runtime)).unwrap();
        assert_eq!(result.action_index, eager_result.action_index);
        assert_eq!(
            result.visits.iter().map(|v| v.visits).collect::<Vec<_>>(),
            eager_result
                .visits
                .iter()
                .map(|v| v.visits)
                .collect::<Vec<_>>()
        );
        eprintln!("lazy rows {scored}, eager rows {all_rows}");
    }
}

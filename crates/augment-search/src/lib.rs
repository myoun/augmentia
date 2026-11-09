//! Model-free diagnostic baseline. Engine-enumerated action subset, approximate material evaluation.
//! Depth counts completed player turns by default; actions within a turn share a depth.
//! Chance is sampled, not a forced-win proof.
pub mod arena;
pub mod features;
pub mod policy;
pub mod tactics;
pub mod teacher;

use augment_core::{
    Action, Engine, GameState, GameStatus, LegalActionBatch, PieceKind, Player, SearchChanceContext,
};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const WIN: f64 = 100_000.0;
// Abort rather than score a truncated action chain as a completed turn.
const MAX_ACTION_PATH: u16 = 64;

/// Cards whose generator accepts targets it does not enumerate.
///
/// `cleanup` and `hypocrisy` were here because they emitted one target instead of every
/// valid combination. Both now enumerate fully, and their card tests assert it, so the
/// list is empty. A card belongs here only after a test shows a valid target is missing
/// from the enumeration; being listed makes search results weaker, not stronger.
const DYNAMIC_TARGET_CARDS: &[&str] = &[];

// Keep the large engine transition off each recursive search frame.
#[inline(never)]
fn apply_search_transition(
    actions: &LegalActionBatch<'_>,
    action: &Action,
    chance: &mut SearchChanceContext,
) -> Result<Box<augment_core::SearchTransition>, augment_core::EngineError> {
    actions.apply_for_search(action, chance).map(Box::new)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthUnit {
    #[default]
    Turns,
    Actions,
}

fn remaining_depth(unit: DepthUnit, depth: u8, events: &[augment_core::Event]) -> u8 {
    let consumed = match unit {
        DepthUnit::Actions => 1,
        DepthUnit::Turns => events
            .iter()
            .filter(|event| matches!(event, augment_core::Event::TurnEnded { .. }))
            .count(),
    };
    depth.saturating_sub(consumed.min(u8::MAX as usize) as u8)
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub time_limit: Duration,
    /// Remaining player turns (not white+black pairs), or actions in legacy mode.
    pub max_depth: u8,
    pub depth_unit: DepthUnit,
    pub max_nodes: u64,
    pub chance_samples: usize,
    /// Independent of the live state's private RNG.
    pub search_seed: u64,
    /// Which leaf evaluation to use. The default keeps the material baseline so a change in
    /// playing strength can be attributed rather than assumed.
    pub evaluation: Evaluation,
    /// Optional learned policy used only to order the root, never to prune it.
    ///
    /// A prior that could prune would make a result depend on the policy being right, and a wrong
    /// prior would hide a win. Ordering keeps every candidate scored, so the result stays exact for
    /// the searched depth while a good prior reaches it sooner.
    pub policy: Option<std::sync::Arc<crate::policy::LinearPolicy>>,
}

/// Leaf evaluation variants.
///
/// `Material` counts piece values only and is the historical baseline. `Positional` adds the terms
/// that decide many short games but that material cannot see: king safety, unspent card resources,
/// mobility, and progress of the endgame conditions. Both are legal information only; neither reads
/// hidden state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub enum Evaluation {
    #[default]
    Material,
    Positional,
}

impl Evaluation {
    /// Stable label used in reports, so an artifact says which evaluation produced it.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Material => "material-v1",
            Self::Positional => "positional-v1",
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            time_limit: Duration::from_millis(250),
            max_depth: 4,
            depth_unit: DepthUnit::Turns,
            max_nodes: 100_000,
            chance_samples: 4,
            search_seed: 19,
            evaluation: Evaluation::default(),
            policy: None,
        }
    }
}
#[derive(Default, Debug, Clone, Serialize)]
pub struct SearchStats {
    pub nodes: u64,
    pub legal_calls: u64,
    pub apply_calls: u64,
    pub chance_samples: u64,
    pub evaluations: u64,
    /// Cards whose action generator is known to accept targets it does not enumerate,
    /// so a missing target must not be read as "that move is unavailable".
    ///
    /// Empty as of the 2026-11-02 snapshot: cleanup and hypocrisy now enumerate every
    /// valid combination. Entries are added only when a generator is re-verified as
    /// incomplete, and every entry narrows what a search result may claim.
    pub known_omitted_target_cards: Vec<String>,
    pub cutoffs: u64,
    /// Maximum action edges on a visited path, including unfinished iterations.
    pub max_action_path: u16,
    pub legal_ms: f64,
    /// Includes engine validation, cloning, events and hidden-RNG probe.
    pub apply_ms: f64,
    pub evaluation_ms: f64,
}
#[derive(Debug, Serialize)]
pub struct SearchReport {
    /// Debug representation only; consumers execute selected_action, not this text.
    pub action: Option<String>,
    #[serde(skip)]
    pub selected_action: Option<Action>,
    /// At completed_depth=0 this evaluates the root, NOT the fallback action.
    pub score: f64,
    /// Even full root coverage is only over Engine::legal_actions, not every accepted target.
    pub action_space: String,
    pub completed_depth: u8,
    pub depth_unit: DepthUnit,
    pub root_actions: usize,
    /// Enumerated root coverage in the deepest FULLY completed iteration, never a partial iteration.
    pub scored_root_actions: usize,
    pub stop_reason: String,
    pub elapsed_ms: f64,
    pub budget_overrun_ms: f64,
    pub stats: SearchStats,
    /// Everything needed to reproduce this decision, recorded with it.
    ///
    /// A record that names the chosen action but not the position, budget or generation settings
    /// cannot be replayed or checked against a policy, so those travel together.
    pub provenance: Provenance,
}

/// The inputs a decision depended on, recorded so the decision can be reproduced.
#[derive(Clone, Debug, Serialize)]
pub struct Provenance {
    /// Which position was searched. Two records with the same fingerprint are the same decision.
    pub state_fingerprint: u64,
    /// Full ruleset identity: the ruleset plus the catalog and card profile in force.
    pub engine: EngineIdentity,
    /// The budget and search settings that produced the result.
    pub config: SearchProvenance,
}

impl Provenance {
    /// Whether two records describe the same decision, i.e. the same position under the same rules.
    pub fn same_decision(&self, other: &Self) -> bool {
        self.state_fingerprint == other.state_fingerprint
            && self.engine.ruleset_id == other.engine.ruleset_id
            && self.engine.catalog_hash == other.engine.catalog_hash
    }
}

/// Identity of the engine state that decided legality and card behavior.
#[derive(Clone, Debug, Serialize)]
pub struct EngineIdentity {
    pub ruleset_id: String,
    pub catalog_hash: String,
    pub card_profile_hash: String,
}

/// The settings that shape a decision, recorded verbatim.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct SearchProvenance {
    pub time_limit_ms: u128,
    pub max_depth: u8,
    pub depth_unit: DepthUnit,
    pub max_nodes: u64,
    pub chance_samples: usize,
    pub search_seed: u64,
    /// Serialized as its stable label, so an artifact says `positional-v1` rather than `Positional`.
    #[serde(serialize_with = "serialize_evaluation")]
    pub evaluation: Evaluation,
}

fn serialize_evaluation<S: serde::Serializer>(
    evaluation: &Evaluation,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(evaluation.label())
}

impl From<&SearchConfig> for SearchProvenance {
    fn from(config: &SearchConfig) -> Self {
        Self {
            time_limit_ms: config.time_limit.as_millis(),
            max_depth: config.max_depth,
            depth_unit: config.depth_unit,
            max_nodes: config.max_nodes,
            chance_samples: config.chance_samples,
            search_seed: config.search_seed,
            evaluation: config.evaluation,
        }
    }
}
#[derive(Debug)]
pub enum SearchError {
    InvalidConfig,
    Engine(String),
    EmptyOngoingState,
}
impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SearchError {}

/// Cooperative deadline: one legal_actions/apply call cannot be preempted.
/// If depth 1 cannot finish, return a legal fallback with completed_depth=0.
/// Errors are surfaced, never reclassified as draws or losing positions.
pub fn search(
    engine: &Engine,
    state: &GameState,
    config: SearchConfig,
) -> Result<SearchReport, SearchError> {
    search_with_cancel(engine, state, config, None)
}

/// Cancel cooperatively between engine calls. A single transition cannot be preempted.
pub fn search_with_cancel(
    engine: &Engine,
    state: &GameState,
    config: SearchConfig,
    cancellation: Option<&AtomicBool>,
) -> Result<SearchReport, SearchError> {
    if config.max_depth == 0
        || config.max_nodes == 0
        || config.chance_samples == 0
        || config.chance_samples > 64
    {
        return Err(SearchError::InvalidConfig);
    }
    let start = Instant::now();
    // Capture the root identity before searching: the record must name the position it decided,
    // and the position is not recoverable from the action alone.
    let provenance = Provenance {
        state_fingerprint: state.fingerprint(),
        engine: {
            let (ruleset_id, catalog_hash, profile_hash) = augment_core::latest_identity();
            EngineIdentity {
                ruleset_id: ruleset_id.to_owned(),
                catalog_hash: catalog_hash.to_owned(),
                card_profile_hash: format!("{profile_hash:016x}"),
            }
        },
        config: SearchProvenance::from(&config),
    };
    let time_limit = config.time_limit;
    let depth_unit_recorded = config.depth_unit;
    let search_seed = config.search_seed;
    let max_depth = config.max_depth;
    let mut ctx = Context {
        engine,
        config,
        start,
        root: state.turn,
        stats: SearchStats::default(),
        active_edges: 0,
        cancellation,
    };
    let actions = ctx.legal(state)?;
    if state.status == GameStatus::Ongoing && actions.is_empty() {
        return Err(SearchError::EmptyOngoingState);
    }
    let mut best = actions.first().cloned();
    let mut score = ctx.evaluate(state);
    let mut completed_depth = 0;
    let mut stop = if actions.is_empty() {
        "terminal"
    } else {
        "depth_limit"
    };
    for depth in 1..=max_depth {
        if actions.is_empty() {
            break;
        }
        let mut order: Vec<usize> = (0..actions.len()).collect();
        // Previous iteration's best goes first, then a learned prior if one is configured. The
        // prior only reorders: every candidate is still scored, so a wrong prior costs time, not
        // correctness.
        let prior: Option<Vec<f32>> = ctx.config.policy.as_ref().map(|policy| {
            let position = crate::features::position_features(state, state.turn);
            actions
                .iter()
                .map(|action| {
                    let action_features = crate::features::action_features(state, action);
                    policy.score(position.as_slice(), action_features.as_slice())
                })
                .collect()
        });
        order.sort_by(|left, right| {
            let incumbent_left = Some(&actions[*left]) != best.as_ref();
            let incumbent_right = Some(&actions[*right]) != best.as_ref();
            if incumbent_left != incumbent_right {
                return incumbent_left.cmp(&incumbent_right);
            }
            if let Some(prior) = &prior {
                let by_prior = prior[*right]
                    .partial_cmp(&prior[*left])
                    .unwrap_or(std::cmp::Ordering::Equal);
                if by_prior != std::cmp::Ordering::Equal {
                    return by_prior;
                }
            }
            left.cmp(right)
        });
        let mut iteration_best = None;
        let mut iteration_score = f64::NEG_INFINITY;
        let mut aborted = false;
        for index in order {
            match ctx.edge(
                &actions,
                &actions[index],
                depth,
                // A fail-low bound cannot replace the incumbent. Chance edges still
                // use full child windows, so the selected score is exact for this sample tree.
                iteration_score,
                f64::INFINITY,
                mix(search_seed ^ index as u64),
            ) {
                Ok(value) => {
                    if value > iteration_score {
                        iteration_score = value;
                        iteration_best = Some(actions[index].clone());
                    }
                }
                Err(Failure::Stop(reason)) => {
                    stop = reason;
                    aborted = true;
                    break;
                }
                Err(Failure::Error(error)) => return Err(error),
            }
        }
        if aborted {
            break;
        }
        best = iteration_best;
        score = iteration_score;
        completed_depth = depth;
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(SearchReport {
        action: best.as_ref().map(|a| format!("{a:?}")),
        selected_action: best,
        score,
        action_space: "engine-enumerated-subset".into(),
        completed_depth,
        depth_unit: depth_unit_recorded,
        root_actions: actions.len(),
        scored_root_actions: if completed_depth > 0 {
            actions.len()
        } else {
            0
        },
        stop_reason: stop.into(),
        elapsed_ms,
        budget_overrun_ms: (elapsed_ms - time_limit.as_secs_f64() * 1000.0).max(0.0),
        stats: ctx.stats,
        provenance,
    })
}

enum Failure {
    Stop(&'static str),
    Error(SearchError),
}
impl From<SearchError> for Failure {
    fn from(e: SearchError) -> Self {
        Self::Error(e)
    }
}
struct Context<'a> {
    engine: &'a Engine,
    config: SearchConfig,
    start: Instant,
    root: Player,
    stats: SearchStats,
    active_edges: u16,
    cancellation: Option<&'a AtomicBool>,
}
impl Context<'_> {
    fn check(&self) -> Result<(), Failure> {
        if self
            .cancellation
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Err(Failure::Stop("cancelled"));
        }
        if self.start.elapsed() >= self.config.time_limit {
            return Err(Failure::Stop("time_limit"));
        }
        if self.stats.nodes >= self.config.max_nodes {
            return Err(Failure::Stop("node_limit"));
        }
        Ok(())
    }
    fn legal<'a>(&mut self, state: &'a GameState) -> Result<LegalActionBatch<'a>, SearchError>
    where
        Self: 'a,
    {
        let start = Instant::now();
        for card in &state.cards {
            if card.owner == state.turn
                && card.charges > 0
                && DYNAMIC_TARGET_CARDS.contains(&card.definition_id.as_str())
                && !self
                    .stats
                    .known_omitted_target_cards
                    .contains(&card.definition_id)
            {
                self.stats
                    .known_omitted_target_cards
                    .push(card.definition_id.clone());
            }
        }
        let result = self.engine.legal_action_batch(state);
        self.stats.legal_calls += 1;
        self.stats.legal_ms += start.elapsed().as_secs_f64() * 1000.0;
        result.map_err(|e| SearchError::Engine(e.to_string()))
    }
    fn evaluate(&mut self, state: &GameState) -> f64 {
        let start = Instant::now();
        let value = evaluate(state, self.root, self.config.evaluation);
        self.stats.evaluations += 1;
        self.stats.evaluation_ms += start.elapsed().as_secs_f64() * 1000.0;
        value
    }
    fn edge(
        &mut self,
        actions: &LegalActionBatch<'_>,
        action: &Action,
        depth: u8,
        alpha: f64,
        beta: f64,
        path: u64,
    ) -> Result<f64, Failure> {
        self.check()?;
        if self.active_edges >= MAX_ACTION_PATH {
            return Err(Failure::Stop("action_path_limit"));
        }
        self.active_edges += 1;
        self.stats.max_action_path = self.stats.max_action_path.max(self.active_edges);
        let result = self.edge_inner(actions, action, depth, alpha, beta, path);
        self.active_edges -= 1;
        result
    }
    fn edge_inner(
        &mut self,
        actions: &LegalActionBatch<'_>,
        action: &Action,
        depth: u8,
        alpha: f64,
        beta: f64,
        path: u64,
    ) -> Result<f64, Failure> {
        self.check()?;
        let mut total = 0.0;
        for sample in 0..self.config.chance_samples {
            self.check()?;
            self.stats.nodes += 1;
            let mut chance = SearchChanceContext::new(mix(
                path ^ (sample as u64).wrapping_mul(0x9e3779b97f4a7c15)
            ));
            let start = Instant::now();
            let transition = apply_search_transition(actions, action, &mut chance);
            self.stats.apply_calls += 1;
            self.stats.apply_ms += start.elapsed().as_secs_f64() * 1000.0;
            let transition = transition.map_err(|e| SearchError::Engine(e.to_string()))?;
            // A single engine call can overrun. Do not claim the iteration completed then.
            if self.start.elapsed() >= self.config.time_limit {
                return Err(Failure::Stop("time_limit"));
            }
            if transition.sampled_chance {
                self.stats.chance_samples += 1;
            }
            // Never average alpha-beta bounds as though they were exact values.
            let (a, b) = if transition.sampled_chance {
                (f64::NEG_INFINITY, f64::INFINITY)
            } else {
                (alpha, beta)
            };
            let value = self.node(
                &transition.transition.state,
                remaining_depth(self.config.depth_unit, depth, &transition.transition.events),
                a,
                b,
                mix(path ^ sample as u64),
            )?;
            if !transition.sampled_chance {
                return Ok(value);
            }
            total += value;
        }
        Ok(total / self.config.chance_samples as f64)
    }
    fn node(
        &mut self,
        state: &GameState,
        depth: u8,
        mut alpha: f64,
        mut beta: f64,
        path: u64,
    ) -> Result<f64, Failure> {
        // The edge already charged this node. Permit a final leaf at the exact node budget.
        if depth == 0 || state.status != GameStatus::Ongoing {
            return Ok(self.evaluate(state));
        }
        self.check()?;
        let actions = self.legal(state)?;
        if actions.is_empty() {
            return Err(SearchError::EmptyOngoingState.into());
        }
        let maximizing = state.turn == self.root;
        let mut best = if maximizing {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        for (index, action) in actions.iter().enumerate() {
            let value = self.edge(
                &actions,
                action,
                depth,
                alpha,
                beta,
                mix(path ^ index as u64),
            )?;
            if maximizing {
                best = best.max(value);
                alpha = alpha.max(best);
            } else {
                best = best.min(value);
                beta = beta.min(best);
            }
            if alpha >= beta {
                self.stats.cutoffs += 1;
                break;
            }
        }
        Ok(best)
    }
}
/// Deliberately weak, transparent baseline. Does not value card synergies or alternate objectives.
pub fn material_score(state: &GameState, root: Player) -> f64 {
    match state.status {
        GameStatus::Won { winner, .. } => {
            if winner == root {
                WIN
            } else {
                -WIN
            }
        }
        GameStatus::Draw { .. } => 0.0,
        GameStatus::Ongoing => state
            .board
            .iter()
            .map(|(_, piece)| {
                let value = match piece.kind {
                    PieceKind::Monster if state.ruleset_id == augment_core::LATEST_RULESET_ID => {
                        0.0
                    }
                    PieceKind::Pawn => 1.0,
                    PieceKind::Knight | PieceKind::Bishop => 3.0,
                    PieceKind::Rook => 5.0,
                    PieceKind::Queen => 9.0,
                    PieceKind::Archbishop => 7.5,
                    PieceKind::Chancellor => 8.5,
                    PieceKind::King => 100.0,
                    _ => 3.0, // Unknown strategic value is not unknown legality.
                };
                if piece.owner == root { value } else { -value }
            })
            .sum(),
    }
}

/// Weights for the positional terms, kept in one place so the trade-offs are reviewable.
mod weights {
    /// A king that can be captured soon is worth far less than a safe one, but the term must stay
    /// bounded: it is a preference, not a win proof.
    pub const KING_ATTACKER: f64 = 0.6;
    /// An unspent card is a resource. Active charges are worth more than installed passives.
    pub const CARD_CHARGE: f64 = 1.2;
    pub const CARD_PASSIVE: f64 = 0.4;
    /// Rough mobility value per reachable action, capped so a wide position cannot dominate.
    pub const MOBILITY: f64 = 0.02;
    pub const MOBILITY_CAP: f64 = 3.0;
}

/// Evaluation that adds positional terms on top of the material baseline.
///
/// The terms answer what material cannot, and every one of them is O(pieces) so it can run at every
/// leaf:
///   - king safety: enemy pieces near our king, ours near theirs,
///   - card resources: charges and installed passives still available,
///   - mobility: a piece-placement approximation of activity.
///
/// An earlier version enumerated legal actions for real mobility. That made evaluation 38x more
/// expensive than material and cost three plies of depth, which is a bad trade: depth is worth more
/// than an exact activity count. This approximation is derived from the position alone.
///
/// Purely a heuristic over public state. It reports no probability and proves nothing.
pub fn positional_score(state: &GameState, root: Player) -> f64 {
    let base = material_score(state, root);
    if !matches!(state.status, GameStatus::Ongoing) {
        return base;
    }
    let opponent = root.opponent();
    let mut score = base;

    // King safety: count enemy pieces in the 5x5 neighbourhood around our king, and ours around
    // theirs. Attacking pieces near the enemy king are worth more than distant ones.
    let ring_attacks = |target: Player, by: Player| -> f64 {
        let Some((king_square, _)) = state
            .board
            .iter()
            .find(|(_, piece)| piece.owner == target && piece.kind == PieceKind::King)
        else {
            return 0.0;
        };
        state
            .board
            .iter()
            .filter(|(square, piece)| {
                piece.owner == by
                    && piece.kind != PieceKind::King
                    && square.file().abs_diff(king_square.file()) <= 2
                    && square.rank().abs_diff(king_square.rank()) <= 2
            })
            .count() as f64
    };
    score -= weights::KING_ATTACKER * ring_attacks(root, opponent);
    score += weights::KING_ATTACKER * ring_attacks(opponent, root);

    // Card resources: unspent charges are tempo the side can still convert.
    for card in &state.cards {
        let value = if card.charges > 0 {
            f64::from(card.charges) * weights::CARD_CHARGE
        } else {
            weights::CARD_PASSIVE
        };
        if card.owner == root {
            score += value;
        } else {
            score -= value;
        }
    }

    // Activity: count the empty squares each side's pieces could structurally reach, ignoring
    // blockers. This distinguishes a rook on an open file from one boxed in without running the
    // move generator.
    let (mut ours, mut theirs) = (0.0, 0.0);
    for (square, piece) in state.board.iter() {
        let reach = reachable_squares(piece.kind, &square);
        if piece.owner == root {
            ours += reach;
        } else {
            theirs += reach;
        }
    }
    let activity =
        ((ours - theirs) * weights::MOBILITY).clamp(-weights::MOBILITY_CAP, weights::MOBILITY_CAP);
    score += activity;

    score
}

/// How many squares a piece kind can structurally reach from `square`, ignoring occupancy.
///
/// This is a shape measure, not a legality test: sliding pieces get their full ray length, leapers a
/// constant. It exists to be cheap enough for every leaf.
fn reachable_squares(kind: PieceKind, square: &augment_core::Square) -> f64 {
    let file = f64::from(square.file().min(7));
    let rank = f64::from(square.rank().min(7));
    let horizontal = file.min(7.0 - file);
    let vertical = rank.min(7.0 - rank);
    let diagonal = horizontal.min(vertical);
    let rays = horizontal + vertical + 2.0 * diagonal;
    match kind {
        PieceKind::Rook => rays,
        PieceKind::Bishop => 2.0 * diagonal,
        PieceKind::Queen => rays + 2.0 * diagonal,
        PieceKind::Knight => 3.0,
        PieceKind::Archbishop => 2.0 * diagonal + 3.0,
        PieceKind::Chancellor => rays + 3.0,
        PieceKind::Pawn => 2.0,
        PieceKind::King => 3.0,
        // Monsters and other card pieces are not given an activity value; their worth is in their
        // ability, which this heuristic cannot see.
        _ => 0.0,
    }
}

/// Evaluate a leaf with the configured variant.
pub fn evaluate(state: &GameState, root: Player, evaluation: Evaluation) -> f64 {
    match evaluation {
        Evaluation::Material => material_score(state, root),
        Evaluation::Positional => positional_score(state, root),
    }
}
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use augment_core::{Board, Mode, Piece, Square};
    fn fixture() -> (Engine, GameState) {
        let engine = Engine::bundled_latest().unwrap();
        let mut board = Board::empty();
        for (sq, owner, kind) in [
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("h1", Player::White, PieceKind::Rook),
        ] {
            board.set(sq.parse::<Square>().unwrap(), Some(Piece::new(owner, kind)));
        }
        let state = GameState::from_board(
            engine.ruleset().id.clone(),
            Mode::Standard,
            board,
            Player::White,
            1,
        );
        (engine, state)
    }
    #[test]
    fn search_finds_compound_transform_then_knight_capture_without_mutating_root() {
        let engine = Engine::bundled_latest().unwrap();
        for (id, base) in [
            (augment_core::ARCHBISHOP, PieceKind::Bishop),
            (augment_core::CHANCELLOR, PieceKind::Rook),
        ] {
            let mut board = Board::empty();
            for (square, owner, kind) in [
                ("a1", Player::White, PieceKind::King),
                ("f5", Player::Black, PieceKind::King),
                ("b1", Player::White, PieceKind::Knight),
                ("d4", Player::White, base),
            ] {
                board.set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
            }
            let mut state = GameState::from_board(
                augment_core::LATEST_RULESET_ID,
                Mode::Standard,
                board,
                Player::White,
                1,
            );
            state.completed_drafts = [true; 3];
            let state = engine
                .grant_card(&state, 1, id, Player::White)
                .unwrap()
                .state;
            let fingerprint = state.fingerprint();
            let result = search(&engine, &state, config(2)).unwrap();
            assert_eq!(
                result.selected_action,
                Some(Action::UseCard {
                    instance_id: 1,
                    target: augment_core::ActionTarget::Squares(vec![
                        "b1".parse().unwrap(),
                        "d4".parse().unwrap()
                    ])
                })
            );
            assert_eq!(result.score, WIN);
            assert_eq!(state.fingerprint(), fingerprint);
        }
    }
    fn config(depth: u8) -> SearchConfig {
        SearchConfig {
            time_limit: Duration::from_secs(30),
            max_depth: depth,
            depth_unit: DepthUnit::Actions,
            ..Default::default()
        }
    }
    #[test]
    fn chooses_immediate_win_without_mutating_state() {
        let (engine, state) = fixture();
        let before = state.fingerprint();
        let result = search(&engine, &state, config(1)).unwrap();
        let next = engine
            .apply(&state, result.selected_action.unwrap())
            .unwrap()
            .state;
        assert!(matches!(
            next.status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));
        assert_eq!(result.completed_depth, 1);
        assert_eq!(state.fingerprint(), before);
    }
    #[test]
    fn zero_time_returns_legal_explicitly_unsearched_fallback() {
        let (engine, state) = fixture();
        let result = search(
            &engine,
            &state,
            SearchConfig {
                time_limit: Duration::ZERO,
                ..config(3)
            },
        )
        .unwrap();
        assert_eq!(result.completed_depth, 0);
        assert_eq!(result.scored_root_actions, 0);
        assert_eq!(result.stop_reason, "time_limit");
        assert!(
            engine
                .legal_actions(&state)
                .unwrap()
                .contains(&result.selected_action.unwrap())
        );
    }
    #[test]
    fn cooperative_cancellation_is_not_a_completed_iteration() {
        let (engine, state) = fixture();
        let cancel = AtomicBool::new(true);
        let result = search_with_cancel(&engine, &state, config(3), Some(&cancel)).unwrap();
        assert_eq!(result.completed_depth, 0);
        assert_eq!(result.stop_reason, "cancelled");
        assert_eq!(result.stats.nodes, 0);
        assert_eq!(result.score, material_score(&state, state.turn));
    }
    #[test]
    fn partial_iteration_does_not_replace_completed_result() {
        let (engine, state) = fixture();
        let count = engine.legal_actions(&state).unwrap().len() as u64;
        let shallow = search(&engine, &state, config(1)).unwrap();
        let interrupted = search(
            &engine,
            &state,
            SearchConfig {
                max_nodes: count + 1,
                ..config(3)
            },
        )
        .unwrap();
        assert_eq!(interrupted.completed_depth, 1);
        assert_eq!(interrupted.selected_action, shallow.selected_action);
        assert_eq!(interrupted.stop_reason, "node_limit");
    }
    #[test]
    fn same_player_continuation_is_maximized_not_negated() {
        let (engine, mut state) = fixture();
        state.actions_remaining = 2;
        // Move along the h-file: the same player must still be allowed to capture next.
        let action = Action::Move(augment_core::MoveAction::new(
            "h1".parse().unwrap(),
            "h2".parse().unwrap(),
            None,
        ));
        let child = engine.apply(&state, action.clone()).unwrap().state;
        assert_eq!(child.turn, Player::White);
        let mut ctx = Context {
            engine: &engine,
            config: config(2),
            start: Instant::now(),
            root: Player::White,
            stats: SearchStats::default(),
            active_edges: 0,
            cancellation: None,
        };
        assert_eq!(
            ctx.edge(
                &engine.legal_action_batch(&state).unwrap(),
                &action,
                2,
                f64::NEG_INFINITY,
                f64::INFINITY,
                19
            )
            .unwrap_or_else(|_| panic!("search failed")),
            WIN
        );
    }
    #[test]
    fn opponent_node_minimizes_root_value() {
        let (engine, mut state) = fixture();
        state.turn = Player::Black;
        state.board.set("h1".parse().unwrap(), None);
        state.board.set(
            "a8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Rook)),
        );
        let mut ctx = Context {
            engine: &engine,
            config: config(1),
            start: Instant::now(),
            root: Player::White,
            stats: SearchStats::default(),
            active_edges: 0,
            cancellation: None,
        };
        let score = ctx
            .node(&state, 1, f64::NEG_INFINITY, f64::INFINITY, 19)
            .unwrap_or_else(|_| panic!("search failed"));
        assert_eq!(score, -WIN);
    }
    #[test]
    fn terminal_and_empty_ongoing_are_distinct() {
        let (engine, state) = fixture();
        let winning = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .find(|a| {
                matches!(
                    engine.apply(&state, a.clone()).unwrap().state.status,
                    GameStatus::Won { .. }
                )
            })
            .unwrap();
        let terminal = engine.apply(&state, winning).unwrap().state;
        let report = search(&engine, &terminal, config(2)).unwrap();
        assert!(report.selected_action.is_none());
        assert_eq!(report.stop_reason, "terminal");
        assert_eq!(report.score, material_score(&terminal, terminal.turn));
        let mut empty = state.clone();
        empty.board = Board::empty();
        assert!(matches!(
            search(&engine, &empty, config(1)),
            Err(SearchError::EmptyOngoingState)
        ));
    }
    #[test]
    fn engine_errors_are_not_draws() {
        let (engine, mut state) = fixture();
        state.ruleset_id = "unsupported".into();
        assert!(matches!(
            search(&engine, &state, config(1)),
            Err(SearchError::Engine(_))
        ));
    }
    #[test]
    fn chance_is_multisampled_without_changing_live_rng() {
        let engine = Engine::bundled_latest().unwrap();
        let initial = engine.new_game(Mode::Standard, 701).unwrap();
        // Reuse real draft offers, but a no-rule initial board so two constructors differ only in RNG.
        let mut state = GameState::standard(engine.ruleset().id.clone(), Mode::Standard, 701);
        state.phase = initial.phase;
        let rng_before = state.rng_state();
        let action = Action::DraftPick { offer_index: 0 };
        let mut ctx = Context {
            engine: &engine,
            config: config(1),
            start: Instant::now(),
            root: state.turn,
            stats: SearchStats::default(),
            active_edges: 0,
            cancellation: None,
        };
        let actual = ctx
            .edge(
                &engine.legal_action_batch(&state).unwrap(),
                &action,
                1,
                999.0,
                1000.0,
                19,
            )
            .unwrap_or_else(|_| panic!("search failed"));
        let mut expected = 0.0;
        for sample in 0..ctx.config.chance_samples {
            let mut chance = SearchChanceContext::new(mix(
                19 ^ (sample as u64).wrapping_mul(0x9e3779b97f4a7c15)
            ));
            let child = engine
                .apply_for_search(&state, action.clone(), &mut chance)
                .unwrap();
            assert!(child.sampled_chance);
            expected += material_score(&child.transition.state, state.turn);
        }
        assert_eq!(actual, expected / ctx.config.chance_samples as f64);
        assert_eq!(ctx.stats.chance_samples, 4);
        assert_eq!(state.rng_state(), rng_before);
        let mut different = GameState::standard(engine.ruleset().id.clone(), Mode::Standard, 702);
        different.phase = state.phase.clone();
        assert_eq!(state.position_key(), different.position_key());
        assert_ne!(state.rng_state(), different.rng_state());
        let left = search(&engine, &state, config(2)).unwrap();
        let right = search(&engine, &different, config(2)).unwrap();
        assert_eq!(left.selected_action, right.selected_action);
        assert_eq!(left.score, right.score);
        assert_eq!(left.stats.nodes, right.stats.nodes);
    }
    fn unpruned(engine: &Engine, state: &GameState, depth: u8, root: Player) -> f64 {
        if depth == 0 || state.status != GameStatus::Ongoing {
            return material_score(state, root);
        }
        let values = engine
            .legal_actions(state)
            .unwrap()
            .into_iter()
            .map(|action| {
                let child = engine.apply(state, action).unwrap().state;
                unpruned(engine, &child, depth - 1, root)
            });
        if state.turn == root {
            values.fold(f64::NEG_INFINITY, f64::max)
        } else {
            values.fold(f64::INFINITY, f64::min)
        }
    }
    #[test]
    fn alpha_beta_matches_exhaustive_actor_aware_reference() {
        let (engine, original) = fixture();
        for actor in [Player::White, Player::Black] {
            for budget in [1, 2] {
                let mut state = original.clone();
                state.turn = actor;
                state.actions_remaining = budget;
                let mut ctx = Context {
                    engine: &engine,
                    config: config(3),
                    start: Instant::now(),
                    root: Player::White,
                    stats: SearchStats::default(),
                    active_edges: 0,
                    cancellation: None,
                };
                let actual = ctx
                    .node(&state, 3, f64::NEG_INFINITY, f64::INFINITY, 19)
                    .unwrap_or_else(|_| panic!("search failed"));
                assert_eq!(actual, unpruned(&engine, &state, 3, Player::White));
                let report = search(&engine, &state, config(3)).unwrap();
                assert_eq!(report.completed_depth, 3);
                assert_eq!(report.score, unpruned(&engine, &state, 3, actor));
                let child = engine
                    .apply(&state, report.selected_action.unwrap())
                    .unwrap()
                    .state;
                assert_eq!(report.score, unpruned(&engine, &child, 2, actor));
            }
        }
    }
    #[test]
    fn chance_average_uses_exact_children_even_under_finite_window() {
        let engine = Engine::bundled_latest().unwrap();
        let mut board = Board::empty();
        for (sq, owner, kind) in [
            ("h1", Player::White, PieceKind::King),
            ("a1", Player::White, PieceKind::Rook),
            ("h8", Player::Black, PieceKind::King),
            ("a8", Player::Black, PieceKind::Rook),
            ("h2", Player::Black, PieceKind::Rook),
        ] {
            board.set(sq.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let state = GameState::from_board(
            engine.ruleset().id.clone(),
            Mode::Standard,
            board,
            Player::White,
            1,
        );
        let state = engine
            .grant_card(&state, 113, augment_core::CLEANUP_SACRIFICE, Player::White)
            .unwrap()
            .state;
        let action = Action::UseCard {
            instance_id: 113,
            target: augment_core::ActionTarget::PieceAt("a1".parse().unwrap()),
        };
        for (depth_unit, edge_depth) in [(DepthUnit::Actions, 2), (DepthUnit::Turns, 1)] {
            let mut ctx = Context {
                engine: &engine,
                config: SearchConfig {
                    depth_unit,
                    chance_samples: 16,
                    ..config(2)
                },
                start: Instant::now(),
                root: Player::White,
                stats: SearchStats::default(),
                active_edges: 0,
                cancellation: None,
            };
            let mut values = Vec::new();
            for sample in 0..ctx.config.chance_samples {
                let mut chance = SearchChanceContext::new(mix(
                    19 ^ (sample as u64).wrapping_mul(0x9e3779b97f4a7c15)
                ));
                let child = engine
                    .apply_for_search(&state, action.clone(), &mut chance)
                    .unwrap();
                assert!(child.sampled_chance);
                values.push(unpruned(&engine, &child.transition.state, 1, Player::White));
            }
            assert!(
                values.iter().any(|value| *value != values[0]),
                "fixture must have different-valued outcomes"
            );
            let expected = values.iter().sum::<f64>() / values.len() as f64;
            let actual = ctx
                .edge(
                    &engine.legal_action_batch(&state).unwrap(),
                    &action,
                    edge_depth,
                    -1000.0,
                    -999.0,
                    19,
                )
                .unwrap_or_else(|_| panic!("search failed"));
            assert_eq!(actual, expected);
            assert_eq!(ctx.stats.chance_samples, 16);
        }
    }
    #[test]
    fn interrupted_chance_average_is_not_a_value() {
        let engine = Engine::bundled_latest().unwrap();
        let state = engine.new_game(Mode::Standard, 701).unwrap();
        let mut ctx = Context {
            engine: &engine,
            config: SearchConfig {
                max_nodes: 2,
                ..config(1)
            },
            start: Instant::now(),
            root: state.turn,
            stats: SearchStats::default(),
            active_edges: 0,
            cancellation: None,
        };
        let value = ctx.edge(
            &engine.legal_action_batch(&state).unwrap(),
            &Action::DraftPick { offer_index: 0 },
            1,
            f64::NEG_INFINITY,
            f64::INFINITY,
            19,
        );
        assert!(matches!(value, Err(Failure::Stop("node_limit"))));
        assert_eq!(ctx.stats.nodes, 2);
        assert_eq!(ctx.stats.chance_samples, 2);
    }
    /// The omission counter exists to narrow a claim when a generator is incomplete.
    /// For the current snapshot nothing is incomplete, so the counter must stay empty
    /// and the report must not warn about a card it can enumerate.
    #[test]
    fn current_snapshot_reports_no_omitted_target_cards() {
        let engine = Engine::bundled_latest().unwrap();
        let mut board = Board::empty();
        for (sq, owner, kind) in [
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("h1", Player::White, PieceKind::Rook),
            ("b2", Player::White, PieceKind::Bishop),
        ] {
            board.set(sq.parse::<Square>().unwrap(), Some(Piece::new(owner, kind)));
        }
        let state = GameState::from_board(
            engine.ruleset().id.clone(),
            Mode::Standard,
            board,
            Player::White,
            1,
        );
        let state = engine
            .grant_card(&state, 31, augment_core::CLEANUP, Player::White)
            .unwrap()
            .state;
        // Both multi-target forms are enumerated now, so a search over this state
        // must not report a known omission. Target order follows the generator's web
        // ordering, so compare as a set rather than a sequence.
        let two_target_forms: Vec<Vec<Square>> = engine
            .legal_actions(&state)
            .unwrap()
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    instance_id: 31,
                    target: augment_core::ActionTarget::Squares(squares),
                } if squares.len() == 2 => Some(squares),
                _ => None,
            })
            .collect();
        assert_eq!(
            two_target_forms.len(),
            1,
            "the only two-target form is h1 plus b2: {two_target_forms:?}"
        );
        let pair: std::collections::BTreeSet<Square> =
            two_target_forms[0].iter().copied().collect();
        assert_eq!(
            pair,
            [
                "h1".parse::<Square>().unwrap(),
                "b2".parse::<Square>().unwrap()
            ]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
        );
        let report = search(&engine, &state, config(1)).unwrap();
        assert!(
            report.stats.known_omitted_target_cards.is_empty(),
            "no card should be reported as omitting targets: {:?}",
            report.stats.known_omitted_target_cards
        );
    }
    #[test]
    fn forced_move_and_promotion_choice_are_searched() {
        let (engine, mut forced) = fixture();
        let rook = forced.board.get("h1".parse().unwrap()).unwrap();
        forced.phase = augment_core::GamePhase::ForcedMove {
            piece_id: rook.id,
            source: augment_core::ForcedMoveSource::Effect("search-fixture".into()),
            continuation: augment_core::ForcedMoveContinuation::ResumeTurn,
        };
        let result = search(&engine, &forced, config(1)).unwrap();
        assert_eq!(result.score, WIN);
        engine
            .apply(&forced, result.selected_action.unwrap())
            .unwrap();
        let mut board = Board::empty();
        for (sq, owner, kind) in [
            ("a1", Player::White, PieceKind::King),
            ("d4", Player::White, PieceKind::Pawn),
            ("e5", Player::Black, PieceKind::Pawn),
            ("h8", Player::Black, PieceKind::King),
        ] {
            board.set(sq.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        let mut state = GameState::from_board(
            engine.ruleset().id.clone(),
            Mode::Standard,
            board,
            Player::White,
            1,
        );
        let mut pawn = state.board.get("d4".parse().unwrap()).unwrap();
        pawn.capture_count = 1;
        state.board.set("d4".parse().unwrap(), Some(pawn));
        state.card_runtime.field_promotion[0] = true;
        let pending = engine
            .apply(
                &state,
                Action::Move(augment_core::MoveAction::new(
                    "d4".parse().unwrap(),
                    "e5".parse().unwrap(),
                    None,
                )),
            )
            .unwrap()
            .state;
        assert!(matches!(
            pending.phase,
            augment_core::GamePhase::AwaitingChoice(_)
        ));
        let report = search(&engine, &pending, config(1)).unwrap();
        assert!(matches!(
            report.selected_action,
            Some(Action::ResolveChoice {
                option_index: 0,
                ..
            })
        ));
        assert_eq!(
            engine
                .apply(&pending, report.selected_action.unwrap())
                .unwrap()
                .state
                .turn,
            Player::Black
        );
    }
    #[test]
    fn exact_node_budget_finishes_and_draw_is_terminal() {
        let (engine, state) = fixture();
        let count = engine.legal_actions(&state).unwrap().len() as u64;
        let report = search(
            &engine,
            &state,
            SearchConfig {
                max_nodes: count,
                ..config(1)
            },
        )
        .unwrap();
        assert_eq!(report.completed_depth, 1);
        assert_eq!(report.stats.nodes, count);
        assert_eq!(report.stop_reason, "depth_limit");
        let mut draw = state;
        draw.status = GameStatus::Draw {
            reason: augment_core::DrawReason::EqualStarsOnRepetition,
        };
        let report = search(&engine, &draw, config(1)).unwrap();
        assert!(report.selected_action.is_none());
        assert_eq!(report.score, 0.0);
        assert_eq!(report.stop_reason, "terminal");
    }
    #[test]
    fn default_depth_unit_is_turns() {
        assert_eq!(SearchConfig::default().depth_unit, DepthUnit::Turns);
    }

    #[test]
    fn only_completed_turn_events_consume_turn_depth() {
        use augment_core::Event;
        let temporary_actor_changes = [
            Event::TurnStarted {
                player: Player::Black,
                actions_remaining: 1,
            },
            Event::TurnEnding {
                player: Player::White,
            },
        ];
        assert_eq!(
            remaining_depth(DepthUnit::Turns, 3, &temporary_actor_changes),
            3
        );
        assert_eq!(
            remaining_depth(DepthUnit::Actions, 3, &temporary_actor_changes),
            2
        );
        let completed = [
            Event::TurnEnded {
                player: Player::White,
            },
            Event::TurnEnded {
                player: Player::Black,
            },
        ];
        assert_eq!(remaining_depth(DepthUnit::Turns, 3, &completed), 1);
        assert_eq!(remaining_depth(DepthUnit::Turns, 1, &completed), 0);
    }

    fn two_action_fixture() -> (Engine, GameState) {
        let (engine, mut state) = fixture();
        state.board.set("a1".parse().unwrap(), None);
        state.board.set("h1".parse().unwrap(), None);
        for (square, owner, kind) in [
            ("b1", Player::White, PieceKind::King),
            ("a2", Player::White, PieceKind::Rook),
            ("a3", Player::Black, PieceKind::Queen),
        ] {
            state
                .board
                .set(square.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        state.actions_remaining = 2;
        (engine, state)
    }

    #[test]
    fn depth_one_turn_sees_win_after_second_action() {
        let (engine, state) = two_action_fixture();
        let before = state.fingerprint();
        let action_report = search(&engine, &state, config(1)).unwrap();
        assert!(action_report.score < WIN);
        let turn_report = search(
            &engine,
            &state,
            SearchConfig {
                depth_unit: DepthUnit::Turns,
                ..config(1)
            },
        )
        .unwrap();
        assert_eq!(turn_report.completed_depth, 1);
        assert_eq!(turn_report.score, WIN);
        assert!(turn_report.stats.max_action_path >= 2);
        let first = engine
            .apply(&state, turn_report.selected_action.unwrap())
            .unwrap();
        assert_eq!(first.state.turn, Player::White);
        assert_eq!(first.state.actions_remaining, 1);
        assert!(
            engine
                .legal_actions(&first.state)
                .unwrap()
                .into_iter()
                .any(|action| {
                    matches!(
                        engine.apply(&first.state, action).unwrap().state.status,
                        GameStatus::Won {
                            winner: Player::White,
                            ..
                        }
                    )
                })
        );
        assert_eq!(before, state.fingerprint());
    }

    fn unpruned_turns(engine: &Engine, state: &GameState, depth: u8, root: Player) -> f64 {
        if state.status != GameStatus::Ongoing || depth == 0 {
            return material_score(state, root);
        }
        let maximizing = state.turn == root;
        engine
            .legal_actions(state)
            .unwrap()
            .into_iter()
            .map(|action| {
                let child = engine.apply(state, action).unwrap().state;
                // Independent reference: completed player counters, not event dispatch.
                let turns = child
                    .turns_taken
                    .iter()
                    .zip(state.turns_taken.iter())
                    .map(|(after, before)| after.saturating_sub(*before) as u64)
                    .sum::<u64>();
                unpruned_turns(
                    engine,
                    &child,
                    depth.saturating_sub(turns.min(255) as u8),
                    root,
                )
            })
            .reduce(|a, b| if maximizing { a.max(b) } else { a.min(b) })
            .unwrap()
    }

    #[test]
    fn turn_depth_matches_unpruned_reference_and_standard_chess_actions() {
        let (engine, mut state) = fixture();
        state.board.set("h1".parse().unwrap(), None);
        for depth in 1..=2 {
            let turns = search(
                &engine,
                &state,
                SearchConfig {
                    depth_unit: DepthUnit::Turns,
                    ..config(depth)
                },
            )
            .unwrap();
            let actions = search(&engine, &state, config(depth)).unwrap();
            assert_eq!(
                turns.score,
                unpruned_turns(&engine, &state, depth, state.turn)
            );
            assert_eq!(turns.score, actions.score);
            assert_eq!(turns.selected_action, actions.selected_action);
            assert_eq!(turns.stats.nodes, actions.stats.nodes);
            assert_eq!(turns.stats.max_action_path, actions.stats.max_action_path);
        }
        state.actions_remaining = 2;
        let turns = search(
            &engine,
            &state,
            SearchConfig {
                depth_unit: DepthUnit::Turns,
                ..config(1)
            },
        )
        .unwrap();
        assert_eq!(turns.score, unpruned_turns(&engine, &state, 1, state.turn));
        assert_eq!(turns.stats.max_action_path, 2);
    }

    #[test]
    fn unfinished_turn_never_claims_completed_depth_and_path_limit_unwinds() {
        let (engine, mut state) = fixture();
        state.board.set("h1".parse().unwrap(), None);
        state.actions_remaining = 2;
        let report = search(
            &engine,
            &state,
            SearchConfig {
                depth_unit: DepthUnit::Turns,
                max_nodes: 1,
                ..config(1)
            },
        )
        .unwrap();
        assert_eq!(report.completed_depth, 0);
        assert_eq!(report.scored_root_actions, 0);
        assert_eq!(report.stop_reason, "node_limit");
        let batch = engine.legal_action_batch(&state).unwrap();
        let mut ctx = Context {
            engine: &engine,
            config: SearchConfig {
                depth_unit: DepthUnit::Turns,
                ..config(1)
            },
            start: Instant::now(),
            root: state.turn,
            stats: SearchStats::default(),
            active_edges: MAX_ACTION_PATH - 1,
            cancellation: None,
        };
        assert!(matches!(
            ctx.edge(&batch, &batch[0], 1, f64::NEG_INFINITY, f64::INFINITY, 19),
            Err(Failure::Stop("action_path_limit"))
        ));
        assert_eq!(ctx.active_edges, MAX_ACTION_PATH - 1);
    }

    #[test]
    fn long_same_turn_chain_stops_without_stack_overflow_or_fake_completion() {
        let (engine, mut state) = fixture();
        state.board.set("h1".parse().unwrap(), None);
        state.actions_remaining = MAX_ACTION_PATH + 1;
        let report = search(
            &engine,
            &state,
            SearchConfig {
                depth_unit: DepthUnit::Turns,
                ..config(1)
            },
        )
        .unwrap();
        assert_eq!(report.stop_reason, "action_path_limit");
        assert_eq!(report.completed_depth, 0);
        assert_eq!(report.scored_root_actions, 0);
        assert_eq!(report.stats.max_action_path, MAX_ACTION_PATH);
        assert!(report.selected_action.is_some());
    }

    #[test]
    fn invalid_budgets_fail_closed() {
        let (engine, state) = fixture();
        assert!(matches!(
            search(
                &engine,
                &state,
                SearchConfig {
                    chance_samples: 0,
                    ..config(1)
                }
            ),
            Err(SearchError::InvalidConfig)
        ));
        assert!(matches!(
            search(
                &engine,
                &state,
                SearchConfig {
                    max_nodes: 0,
                    ..config(1)
                }
            ),
            Err(SearchError::InvalidConfig)
        ));
    }

    /// A recorded decision must be reproducible from the record alone.
    ///
    /// This is what P2-1 asks for: the provenance names the position, the engine identity and the
    /// budget, and running the same configuration on the same position must produce the same action.
    /// Without the position in the record the action could not be replayed at all.
    #[test]
    fn a_recorded_decision_reproduces_from_its_provenance() {
        let (engine, state) = fixture();
        let settings = config(4);
        let first = search(&engine, &state, settings.clone()).unwrap();

        // The provenance names the position actually searched.
        assert_eq!(first.provenance.state_fingerprint, state.fingerprint());
        assert_eq!(first.provenance.config.search_seed, settings.search_seed);
        assert_eq!(
            first.provenance.config.time_limit_ms,
            settings.time_limit.as_millis()
        );
        assert_eq!(
            first.provenance.engine.catalog_hash,
            augment_core::latest_identity().1
        );

        // Re-running the recorded configuration on the recorded position gives the same action.
        let second = search(&engine, &state, settings.clone()).unwrap();
        assert_eq!(
            first.action, second.action,
            "a recorded decision must replay identically"
        );
        assert_eq!(first.score, second.score);
        assert!(first.provenance.same_decision(&second.provenance));
    }

    /// Two different positions must not look like the same decision.
    #[test]
    fn provenance_distinguishes_positions() {
        let (engine, state) = fixture();
        let mut other = state.clone();
        other.board.set(
            augment_core::Square::new(4, 3).unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn)),
        );
        let first = search(&engine, &state, config(2)).unwrap();
        let second = search(&engine, &other, config(2)).unwrap();
        assert_ne!(
            first.provenance.state_fingerprint,
            second.provenance.state_fingerprint
        );
        assert!(!first.provenance.same_decision(&second.provenance));
    }

    /// The recorded evaluation must be the one that ran, not a default.
    #[test]
    fn provenance_records_the_evaluation_that_ran() {
        let (engine, state) = fixture();
        let material = search(
            &engine,
            &state,
            SearchConfig {
                evaluation: Evaluation::Material,
                ..config(1)
            },
        )
        .unwrap();
        let positional = search(
            &engine,
            &state,
            SearchConfig {
                evaluation: Evaluation::Positional,
                ..config(1)
            },
        )
        .unwrap();
        assert_eq!(material.provenance.config.evaluation, Evaluation::Material);
        assert_eq!(
            positional.provenance.config.evaluation,
            Evaluation::Positional
        );
        // The serialized form is the stable label, so an artifact is readable without this code.
        let text = serde_json::to_string(&positional.provenance).unwrap();
        assert!(text.contains("positional-v1"), "{text}");
    }
}

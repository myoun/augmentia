//! Model-free diagnostic baseline. Engine-enumerated action subset, approximate material evaluation.
//! Depth counts completed player turns by default; actions within a turn share a depth.
//! Chance is sampled, not a forced-win proof.
pub mod tactics;

use augment_core::{
    Action, Engine, GameState, GameStatus, LegalActionBatch, PieceKind, Player, SearchChanceContext,
};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const WIN: f64 = 100_000.0;
// Abort rather than score a truncated action chain as a completed turn.
const MAX_ACTION_PATH: u16 = 64;

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

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    pub time_limit: Duration,
    /// Remaining player turns (not white+black pairs), or actions in legacy mode.
    pub max_depth: u8,
    pub depth_unit: DepthUnit,
    pub max_nodes: u64,
    pub chance_samples: usize,
    /// Independent of the live state's private RNG.
    pub search_seed: u64,
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
    /// Detected dynamic target cards whose generator omits valid combinations.
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
    for depth in 1..=config.max_depth {
        if actions.is_empty() {
            break;
        }
        let mut order: Vec<usize> = (0..actions.len()).collect();
        // Previous iteration's best goes first. No action is pruned by a learned prior.
        order.sort_by_key(|i| (Some(&actions[*i]) != best.as_ref(), *i));
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
                mix(config.search_seed ^ index as u64),
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
        depth_unit: config.depth_unit,
        root_actions: actions.len(),
        scored_root_actions: if completed_depth > 0 {
            actions.len()
        } else {
            0
        },
        stop_reason: stop.into(),
        elapsed_ms,
        budget_overrun_ms: (elapsed_ms - config.time_limit.as_secs_f64() * 1000.0).max(0.0),
        stats: ctx.stats,
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
                && [augment_core::CLEANUP, augment_core::HYPOCRISY]
                    .contains(&card.definition_id.as_str())
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
        let value = material_score(state, self.root);
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
            let transition = actions.apply_for_search(action, &mut chance);
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
        let engine = Engine::bundled_20260915().unwrap();
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
        let engine = Engine::bundled_20260818().unwrap();
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
        let engine = Engine::bundled_20260915().unwrap();
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
        let engine = Engine::bundled_20260818().unwrap();
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
    #[test]
    fn reports_dynamic_targets_as_omitted_instead_of_claiming_full_coverage() {
        let (engine, state) = fixture();
        let state = engine
            .grant_card(&state, 31, augment_core::CLEANUP, Player::White)
            .unwrap()
            .state;
        let target = Action::UseCard {
            instance_id: 31,
            target: augment_core::ActionTarget::Squares(vec![
                "a1".parse().unwrap(),
                "h1".parse().unwrap(),
            ]),
        };
        assert!(!engine.legal_actions(&state).unwrap().contains(&target));
        assert!(engine.apply(&state, target).is_ok());
        let report = search(&engine, &state, config(1)).unwrap();
        assert_eq!(report.action_space, "engine-enumerated-subset");
        assert!(
            report
                .stats
                .known_omitted_target_cards
                .contains(&augment_core::CLEANUP.to_owned())
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
}

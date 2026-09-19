//! Fixed-budget V4 factorized-search baseline benchmark.
//!
//! Runs one sealed simulation budget over a fixed fixture with either the
//! checked-in split ONNX bundle or the deterministic hash evaluator, then
//! prints one JSON report. A sealed baseline records this exact command,
//! engine commit, bundle metadata hash, seeds, simulation count, chance sample
//! count, and runtime options; changing any of them starts a new baseline.

use augment_ai::{
    CHANCE_SAMPLING_V4_ID, DeterministicFactorizedEvaluatorV4, FactorizedEvaluatorV4,
    LegalActionSetV3, MctsV4Config, MctsV4Tree, ROOT_SELECTION_V4_ID, SEARCH_SEMANTICS_V4_ID,
    V3ExecutionProvider, V3RuntimeOptions, V4OnnxRuntime,
};
use augment_core::{
    Board, CHAMELEON_MUTATION, Engine, GamePhase, GameState, Mode, Piece, PieceKind, Player,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

const SYNTHETIC_RULESET_MANIFEST_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Serialize)]
struct V4BenchmarkReport {
    evaluator: String,
    fixture: String,
    ruleset_id: String,
    ruleset_manifest_sha256: String,
    simulations: usize,
    iterations: usize,
    chance_outcome_samples: usize,
    gumbel_candidates: usize,
    search_seed: u64,
    gumbel_seed: u64,
    root_legal_actions: usize,
    mean_search_ms: f64,
    median_search_ms: f64,
    simulations_per_second: f64,
    mean_evaluator_calls: f64,
    mean_scored_prefix_rows: f64,
    mean_engine_apply_count: f64,
    mean_prefix_engine_apply_count: f64,
    maximum_sampled_chance_outcomes: usize,
    root_gumbel_halving_rounds: usize,
    maximum_depth: usize,
    root_visit_coverage: f64,
    top_action_index: usize,
    top_action_prior: f32,
    visit_distribution_sha256: String,
    repeat_top_action_equal: bool,
    repeat_total_visit_delta: u32,
    search_semantics_id: &'static str,
    root_selection_id: &'static str,
    chance_sampling_id: &'static str,
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let evaluator = optional_string(&args, "--evaluator", "onnx");
    let fixture = optional_string(&args, "--fixture", "standard");
    let simulations = optional_usize(&args, "--simulations", 200);
    let iterations = optional_usize(&args, "--iterations", 8);
    let chance_outcome_samples = optional_usize(&args, "--chance-outcome-samples", 8);
    let gumbel_candidates = optional_usize(&args, "--gumbel-candidates", 16);
    let search_seed = optional_u64(&args, "--search-seed", 0x5345_4152_4348_5634);
    let gumbel_seed = optional_u64(&args, "--gumbel-seed", 0x4755_4d42_454c_5634);
    if simulations == 0 || iterations == 0 {
        panic!("--simulations and --iterations must be positive");
    }
    let engine = selected_engine(&args);
    let ruleset_id = engine.ruleset().id.clone();
    let ruleset_hash = optional_string(&args, "--ruleset-hash", SYNTHETIC_RULESET_MANIFEST_SHA256);
    let state = benchmark_state(&engine, &fixture);
    let legal = LegalActionSetV3::from_state(&state, &engine).expect("legal action set");
    let config = MctsV4Config {
        num_simulations: simulations,
        chance_outcome_samples,
        gumbel_candidates: gumbel_candidates.clamp(1, legal.len()),
        search_seed,
        gumbel_seed,
        ..MctsV4Config::default()
    };
    let report = match evaluator.as_str() {
        "deterministic" => {
            let seed = optional_u64(&args, "--deterministic-seed", 0x5634_4245_4e43_4831);
            let runtime = DeterministicFactorizedEvaluatorV4 { seed, value: 0.05 };
            run_report_with(
                &engine,
                &state,
                &legal,
                &fixture,
                &evaluator,
                config,
                iterations,
                &runtime,
                &ruleset_hash,
            )
        }
        "onnx" => {
            let bundle = required(&args, "--bundle");
            let intra_threads = optional_usize(&args, "--intra-threads", 1);
            let runtime = V4OnnxRuntime::from_bundle_with_options(
                bundle,
                &ruleset_id,
                &ruleset_hash,
                V3RuntimeOptions {
                    execution_provider: V3ExecutionProvider::Cpu,
                    intra_threads,
                },
            )
            .expect("load checked-in V4 bundle");
            run_report_with(
                &engine,
                &state,
                &legal,
                &fixture,
                &evaluator,
                config,
                iterations,
                &runtime,
                &ruleset_hash,
            )
        }
        other => panic!("--evaluator must be onnx or deterministic, got {other:?}"),
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

#[allow(clippy::too_many_arguments)]
fn run_report_with<E: FactorizedEvaluatorV4>(
    engine: &Engine,
    state: &GameState,
    legal: &LegalActionSetV3,
    fixture: &str,
    evaluator: &str,
    config: MctsV4Config,
    iterations: usize,
    runtime: &E,
    ruleset_manifest_sha256: &str,
) -> V4BenchmarkReport {
    let mut elapsed = Vec::with_capacity(iterations);
    let mut evaluator_calls = 0usize;
    let mut scored_prefix_rows = 0usize;
    let mut engine_applies = 0usize;
    let mut prefix_engine_applies = 0usize;
    let mut maximum_sampled_chance_outcomes = 0usize;
    let mut halving_rounds = 0usize;
    let mut maximum_depth = 0usize;
    let mut coverage_sum = 0.0f64;
    let mut repeat_top_action_equal = true;
    let mut repeat_total_visit_delta = 0u32;
    for _ in 0..iterations {
        let mut tree = MctsV4Tree::new(config).expect("valid V4 config");
        let started = Instant::now();
        let result = tree.search(state, engine, runtime).expect("V4 search");
        elapsed.push(started.elapsed().as_secs_f64() * 1_000.0);
        let diagnostics = tree.last_diagnostics.as_ref().unwrap();
        evaluator_calls += diagnostics.evaluator_calls;
        scored_prefix_rows += diagnostics.scored_prefix_rows;
        engine_applies += diagnostics.engine_apply_count;
        prefix_engine_applies += diagnostics.prefix_engine_apply_count;
        maximum_sampled_chance_outcomes =
            maximum_sampled_chance_outcomes.max(diagnostics.sampled_chance_outcomes);
        halving_rounds = halving_rounds.max(diagnostics.root_gumbel_halving_rounds);
        maximum_depth = maximum_depth.max(diagnostics.maximum_depth);
        let visited = result
            .visits
            .iter()
            .filter(|visit| visit.visits > 0)
            .count();
        coverage_sum += visited as f64 / result.visits.len().max(1) as f64;
        // One fresh-tree repeat on the shared runtime also proves that a warm
        // encoder cache never changes the selected action or visit counts.
        let mut repeat_tree = MctsV4Tree::new(config).expect("valid V4 config");
        let repeat_result = repeat_tree
            .search(state, engine, runtime)
            .expect("V4 repeat");
        repeat_top_action_equal &= repeat_result.action_index == result.action_index;
        repeat_total_visit_delta += repeat_result
            .visits
            .iter()
            .zip(result.visits.iter())
            .map(|(left, right)| left.visits.abs_diff(right.visits))
            .sum::<u32>();
    }
    elapsed.sort_by(|left, right| left.total_cmp(right));
    let mean = elapsed.iter().sum::<f64>() / iterations as f64;
    let median = elapsed[iterations / 2];
    let mut tree = MctsV4Tree::new(config).expect("valid V4 config");
    let reference = tree
        .search(state, engine, runtime)
        .expect("V4 reference result");
    let mut hash = Sha256::new();
    for visit in &reference.visits {
        hash.update(visit.action_index.to_le_bytes());
        hash.update(visit.visits.to_le_bytes());
        hash.update(visit.prior.to_le_bytes());
    }
    V4BenchmarkReport {
        evaluator: evaluator.to_owned(),
        fixture: fixture.to_owned(),
        ruleset_id: engine.ruleset().id.clone(),
        ruleset_manifest_sha256: ruleset_manifest_sha256.to_owned(),
        simulations: config.num_simulations,
        iterations,
        chance_outcome_samples: config.chance_outcome_samples,
        gumbel_candidates: config.gumbel_candidates,
        search_seed: config.search_seed,
        gumbel_seed: config.gumbel_seed,
        root_legal_actions: legal.len(),
        mean_search_ms: mean,
        median_search_ms: median,
        simulations_per_second: config.num_simulations as f64 * iterations as f64
            / (elapsed.iter().sum::<f64>() / 1_000.0),
        mean_evaluator_calls: evaluator_calls as f64 / iterations as f64,
        mean_scored_prefix_rows: scored_prefix_rows as f64 / iterations as f64,
        mean_engine_apply_count: engine_applies as f64 / iterations as f64,
        mean_prefix_engine_apply_count: prefix_engine_applies as f64 / iterations as f64,
        maximum_sampled_chance_outcomes,
        root_gumbel_halving_rounds: halving_rounds,
        maximum_depth,
        root_visit_coverage: coverage_sum / iterations as f64,
        top_action_index: reference.action_index,
        top_action_prior: reference.visits[reference.action_index].prior,
        visit_distribution_sha256: format!("{:x}", hash.finalize()),
        repeat_top_action_equal,
        repeat_total_visit_delta,
        search_semantics_id: SEARCH_SEMANTICS_V4_ID,
        root_selection_id: ROOT_SELECTION_V4_ID,
        chance_sampling_id: CHANCE_SAMPLING_V4_ID,
    }
}

fn benchmark_state(engine: &Engine, fixture: &str) -> GameState {
    match fixture {
        "standard" => {
            let mut state =
                GameState::standard(engine.ruleset().id.clone(), Mode::Standard, 2026082102);
            state.phase = GamePhase::Playing;
            state
        }
        "draft" => engine
            .new_game(Mode::Standard, 2026082102)
            .expect("start standard game in opening draft"),
        "wide_chameleon" => {
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
            let state = GameState::from_board(
                engine.ruleset().id.clone(),
                Mode::Standard,
                board,
                Player::White,
                2026082102,
            );
            engine
                .grant_card(&state, 700, CHAMELEON_MUTATION, Player::White)
                .expect("grant chameleon fixture card")
                .state
        }
        other => panic!("--fixture must be standard, draft, or wide_chameleon, got {other:?}"),
    }
}

fn selected_engine(args: &[String]) -> Engine {
    let requested = optional_string(args, "--ruleset", "20260818");
    match requested.as_str() {
        "20260818" | "augment_chess_20260818" => {
            Engine::bundled_20260818().expect("load 20260818 engine")
        }
        "20260915" | "augment_chess_20260915" => {
            Engine::bundled_20260915().expect("load 20260915 engine")
        }
        other => {
            panic!("--ruleset must be 20260818 or 20260915 (or the full ruleset id), got {other:?}")
        }
    }
}

fn required(args: &[String], name: &str) -> PathBuf {
    let index = args
        .iter()
        .position(|argument| argument == name)
        .unwrap_or_else(|| panic!("{name} is required"));
    PathBuf::from(
        args.get(index + 1)
            .unwrap_or_else(|| panic!("{name} needs a value")),
    )
}

fn optional_string(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|argument| argument == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .unwrap_or_else(|| default.to_owned())
}

fn optional_usize(args: &[String], name: &str, default: usize) -> usize {
    optional_string(args, name, &default.to_string())
        .parse()
        .unwrap_or_else(|error| panic!("{name} must be a usize: {error}"))
}

fn optional_u64(args: &[String], name: &str, default: u64) -> u64 {
    optional_string(args, name, &default.to_string())
        .parse()
        .unwrap_or_else(|error| panic!("{name} must be a u64: {error}"))
}

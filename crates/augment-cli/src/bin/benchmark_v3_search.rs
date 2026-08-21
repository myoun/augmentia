use augment_ai::{MctsV3Config, MctsV3Tree, V3OnnxRuntime};
use augment_core::{BUNDLED_RULESET_ID, Engine, GamePhase, GameState, Mode};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Serialize)]
struct BatchSearchReport {
    leaf_batch_size: usize,
    iterations: usize,
    simulations_per_search: usize,
    simulations_per_second: f64,
    mean_search_ms: f64,
    mean_evaluator_batches: f64,
    maximum_leaf_batch: usize,
    root_visit_coverage: f32,
    maximum_transposition_candidates: usize,
}

#[derive(Serialize)]
struct SearchBenchmarkReport {
    session_instances: usize,
    session_runs: u64,
    batches: Vec<BatchSearchReport>,
    subtree_reuse_verified: bool,
    transposition_enabled: bool,
    quiescence_enabled: bool,
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let model = required_path(&args, "--model");
    let metadata = required_path(&args, "--metadata");
    let ruleset_hash = required(&args, "--ruleset-hash");
    let simulations = optional_usize(&args, "--simulations", 200);
    let iterations = optional_usize(&args, "--iterations", 10);
    let engine = Engine::bundled_20260807().expect("load bundled engine");
    let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 2026082102);
    state.phase = GamePhase::Playing;
    let runtime = V3OnnxRuntime::from_files(model, metadata, BUNDLED_RULESET_ID, &ruleset_hash)
        .expect("load strict V3 runtime");

    let mut batches = Vec::new();
    for leaf_batch_size in [1, 8, 16] {
        let config = MctsV3Config {
            num_simulations: simulations,
            leaf_batch_size,
            ..MctsV3Config::default()
        };
        let mut elapsed = 0.0f64;
        let mut evaluator_batches = 0usize;
        let mut maximum_leaf_batch = 0usize;
        let mut coverage = 0.0f32;
        let mut maximum_transposition_candidates = 0usize;
        for _ in 0..iterations {
            let mut tree = MctsV3Tree::new(config).expect("valid search config");
            let started = Instant::now();
            tree.search(&state, &engine, &runtime).expect("V3 search");
            elapsed += started.elapsed().as_secs_f64();
            let diagnostics = tree.last_diagnostics.as_ref().unwrap();
            evaluator_batches += diagnostics.evaluator_batches;
            maximum_leaf_batch = maximum_leaf_batch.max(diagnostics.maximum_leaf_batch);
            coverage += diagnostics.root_visit_coverage;
            maximum_transposition_candidates = maximum_transposition_candidates
                .max(diagnostics.transposition_candidate_duplicates);
        }
        batches.push(BatchSearchReport {
            leaf_batch_size,
            iterations,
            simulations_per_search: simulations,
            simulations_per_second: simulations as f64 * iterations as f64 / elapsed,
            mean_search_ms: elapsed * 1_000.0 / iterations as f64,
            mean_evaluator_batches: evaluator_batches as f64 / iterations as f64,
            maximum_leaf_batch,
            root_visit_coverage: coverage / iterations as f32,
            maximum_transposition_candidates,
        });
    }

    let reuse_config = MctsV3Config {
        num_simulations: simulations,
        leaf_batch_size: 16,
        ..MctsV3Config::default()
    };
    let mut reuse_tree = MctsV3Tree::new(reuse_config).unwrap();
    let first = reuse_tree.search(&state, &engine, &runtime).unwrap();
    let next = engine.apply(&state, first.action).unwrap().state;
    reuse_tree.search(&next, &engine, &runtime).unwrap();
    let subtree_reuse_verified = reuse_tree.last_diagnostics.as_ref().unwrap().tree_reused;

    println!(
        "{}",
        serde_json::to_string_pretty(&SearchBenchmarkReport {
            session_instances: 1,
            session_runs: runtime.session_runs(),
            batches,
            subtree_reuse_verified,
            transposition_enabled: false,
            quiescence_enabled: false,
        })
        .unwrap()
    );
}

fn required(args: &[String], flag: &str) -> String {
    args.iter()
        .position(|item| item == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .unwrap_or_else(|| panic!("{flag} is required"))
}

fn required_path(args: &[String], flag: &str) -> PathBuf {
    required(args, flag).into()
}

fn optional_usize(args: &[String], flag: &str, default: usize) -> usize {
    args.iter()
        .position(|item| item == flag)
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse().expect("numeric argument"))
        .unwrap_or(default)
}

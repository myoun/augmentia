use augment_ai::{MctsV3Config, MctsV3Tree, V3ExecutionProvider, V3OnnxRuntime, V3RuntimeOptions};
use augment_core::{
    BUNDLED_RULESET_ID, Board, CHAMELEON_MUTATION, Engine, GamePhase, GameState, Mode, Piece,
    PieceKind, Player,
};
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
    mean_root_legal_actions: f64,
    mean_root_opened_actions: f64,
    mean_family_coverage: f64,
    mean_tail_visit_rate: f64,
    maximum_sampled_chance_outcomes: usize,
    maximum_reuse_depth: usize,
    search_semantics_id: String,
}

#[derive(Serialize)]
struct SearchBenchmarkReport {
    execution_provider: &'static str,
    intra_threads: usize,
    session_instances: usize,
    session_runs: u64,
    batches: Vec<BatchSearchReport>,
    subtree_reuse_verified: bool,
    transposition_enabled: bool,
    quiescence_enabled: bool,
    candidate_scoring: CandidateScoringReport,
}

#[derive(Serialize)]
struct CandidateScoringReport {
    legal_actions: usize,
    candidate_actions: usize,
    iterations: usize,
    mean_full_ms: f64,
    mean_candidate_ms: f64,
    maximum_probability_delta: f32,
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let model = required_path(&args, "--model");
    let metadata = required_path(&args, "--metadata");
    let ruleset_hash = required(&args, "--ruleset-hash");
    let simulations = optional_usize(&args, "--simulations", 200);
    let iterations = optional_usize(&args, "--iterations", 10);
    let leaf_batch_sizes = optional_usize_value(&args, "--leaf-batch-size")
        .map_or_else(|| vec![1, 8, 16], |value| vec![value]);
    let execution_provider = match optional_string(&args, "--execution-provider", "cpu").as_str() {
        "cpu" => V3ExecutionProvider::Cpu,
        "cuda" => V3ExecutionProvider::Cuda,
        value => panic!("--execution-provider must be cpu or cuda, got {value:?}"),
    };
    let intra_threads = optional_usize(&args, "--intra-threads", 1);
    let requested_candidates = optional_usize(&args, "--candidate-actions", 32);
    let search_variant = optional_string(&args, "--search", "fair_puct");
    let engine = Engine::bundled_20260807().expect("load bundled engine");
    let fixture = optional_string(&args, "--fixture", "standard");
    let state = benchmark_state(&engine, &fixture);
    let runtime = V3OnnxRuntime::from_files_with_options(
        model,
        metadata,
        BUNDLED_RULESET_ID,
        &ruleset_hash,
        V3RuntimeOptions {
            execution_provider,
            intra_threads,
        },
    )
    .expect("load strict V3 runtime");
    let legal_count = engine.legal_actions(&state).unwrap().len();
    let candidate_count = requested_candidates.clamp(1, legal_count);
    let candidate_indices = (0..candidate_count).collect::<Vec<_>>();
    let mut full_elapsed = 0.0;
    let mut candidate_elapsed = 0.0;
    let mut full = None;
    let mut candidate = None;
    for _ in 0..iterations {
        let started = Instant::now();
        full = Some(
            runtime
                .evaluate(&state, &engine)
                .expect("full V3 evaluation"),
        );
        full_elapsed += started.elapsed().as_secs_f64();
        let started = Instant::now();
        candidate = Some(
            runtime
                .evaluate_candidates(&state, &engine, &candidate_indices)
                .expect("candidate V3 evaluation"),
        );
        candidate_elapsed += started.elapsed().as_secs_f64();
    }
    let full = full.unwrap();
    let candidate = candidate.unwrap();
    let full_subset_total = full
        .priors
        .iter()
        .filter(|(_, index, _)| *index < candidate_count)
        .map(|item| item.2)
        .sum::<f32>();
    let candidate_probabilities = softmax(
        &candidate
            .logits
            .iter()
            .map(|item| item.2)
            .collect::<Vec<_>>(),
    );
    let maximum_probability_delta = candidate
        .logits
        .iter()
        .zip(candidate_probabilities)
        .map(|((_, index, _), probability)| {
            let expected = full.priors[*index].2 / full_subset_total;
            (expected - probability).abs()
        })
        .fold(0.0f32, f32::max);
    let candidate_scoring = CandidateScoringReport {
        legal_actions: legal_count,
        candidate_actions: candidate_count,
        iterations,
        mean_full_ms: full_elapsed * 1_000.0 / iterations as f64,
        mean_candidate_ms: candidate_elapsed * 1_000.0 / iterations as f64,
        maximum_probability_delta,
    };

    let mut batches = Vec::new();
    for leaf_batch_size in leaf_batch_sizes {
        let config = benchmark_search_config(simulations, leaf_batch_size, &search_variant);
        let mut elapsed = 0.0f64;
        let mut evaluator_batches = 0usize;
        let mut maximum_leaf_batch = 0usize;
        let mut coverage = 0.0f32;
        let mut maximum_transposition_candidates = 0usize;
        let mut root_legal_actions = 0usize;
        let mut root_opened_actions = 0usize;
        let mut family_coverage = 0.0f64;
        let mut tail_visit_rate = 0.0f64;
        let mut maximum_sampled_chance_outcomes = 0usize;
        let mut maximum_reuse_depth = 0usize;
        let mut search_semantics_id = String::new();
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
            root_legal_actions += diagnostics.root_legal_actions;
            root_opened_actions += diagnostics.root_opened_action_count;
            family_coverage += diagnostics.root_opened_action_family_count as f64
                / diagnostics.root_action_family_count.max(1) as f64;
            tail_visit_rate += diagnostics.proposal_tail_visit_rate as f64;
            maximum_sampled_chance_outcomes =
                maximum_sampled_chance_outcomes.max(diagnostics.sampled_chance_outcomes);
            maximum_reuse_depth =
                maximum_reuse_depth.max(diagnostics.tree_reuse_depth.unwrap_or(0));
            search_semantics_id = diagnostics.search_semantics_id.to_owned();
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
            mean_root_legal_actions: root_legal_actions as f64 / iterations as f64,
            mean_root_opened_actions: root_opened_actions as f64 / iterations as f64,
            mean_family_coverage: family_coverage / iterations as f64,
            mean_tail_visit_rate: tail_visit_rate / iterations as f64,
            maximum_sampled_chance_outcomes,
            maximum_reuse_depth,
            search_semantics_id,
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
            execution_provider: execution_provider.as_str(),
            intra_threads,
            session_instances: 1,
            session_runs: runtime.session_runs(),
            batches,
            subtree_reuse_verified,
            transposition_enabled: false,
            quiescence_enabled: false,
            candidate_scoring,
        })
        .unwrap()
    );
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = logits
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect::<Vec<_>>();
    let total = values.iter().sum::<f32>();
    for value in &mut values {
        *value /= total;
    }
    values
}

fn benchmark_state(engine: &Engine, fixture: &str) -> GameState {
    match fixture {
        "standard" => {
            let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 2026082102);
            state.phase = GamePhase::Playing;
            state
        }
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
                BUNDLED_RULESET_ID,
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
        other => panic!("--fixture must be standard or wide_chameleon, got {other:?}"),
    }
}

fn benchmark_search_config(
    simulations: usize,
    leaf_batch_size: usize,
    variant: &str,
) -> MctsV3Config {
    let mut config = MctsV3Config {
        num_simulations: simulations,
        leaf_batch_size,
        reuse_descendant_depth: 0,
        gumbel_planning_budget: simulations,
        ..MctsV3Config::default()
    };
    match variant {
        "fair_puct" => {}
        "reuse" => config.reuse_descendant_depth = 4,
        "widening" => config.progressive_widening = true,
        "gumbel" => {
            config.gumbel_root = true;
            config.gumbel_candidates = if simulations <= 150 { 32 } else { 64 };
        }
        "virtual_chameleon" => config.virtual_chameleon_hierarchy = true,
        "widening_virtual_chameleon" => {
            config.progressive_widening = true;
            config.virtual_chameleon_hierarchy = true;
        }
        other => panic!("unsupported --search variant {other:?}"),
    }
    config
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

fn optional_usize_value(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].parse().expect("numeric argument"))
}

fn optional_string(args: &[String], flag: &str, default: &str) -> String {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| default.to_owned())
}

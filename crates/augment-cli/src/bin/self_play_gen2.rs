use augment_ai::action_space::generate_action_mask;
use augment_ai::tensor::encode_state;
use augment_ai::{
    ACTION_V2_SCHEMA_ID, MctsConfig, MctsTree, OBSERVATION_V2_SCHEMA_ID, OnnxNeuralEvaluator,
    RootNoiseConfig, SELF_PLAY_V2_RECORD_SCHEMA_ID, SearchDiagnostics, V2_BASELINE_MODEL_ID,
};
use augment_core::{Action, BUNDLED_RULESET_ID, Engine, GameState, GameStatus, Mode, PieceKind};
use rayon::prelude::*;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Serialize)]
struct RootActionTrace {
    action_index: u16,
    action_category: String,
    is_capture: bool,
    is_king_capture_threat: bool,
    raw_prior: f32,
    root_noise: f32,
    effective_prior: f32,
    visit_count: u32,
    first_visit_simulation: Option<usize>,
    q_root: f32,
}

#[derive(Serialize)]
struct RootSearchTelemetry {
    root_raw_value: f32,
    legal_action_count: usize,
    visited_action_count: usize,
    root_visit_entropy: f32,
    top1_visit_fraction: f32,
    effective_branching_factor: f32,
    low_prior_visit_rate: f32,
    max_search_depth: usize,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    noise_seed: u64,
    actions: Vec<RootActionTrace>,
}

#[derive(Serialize)]
struct SelfPlaySample {
    schema_version: &'static str,
    ruleset_id: &'static str,
    observation_schema_id: &'static str,
    action_schema_id: &'static str,
    model_id: &'static str,
    game_id: usize,
    ply: usize,
    seed: u64,
    spatial: Vec<f32>,
    global: Vec<f32>,
    mask: Vec<usize>,
    policy: Vec<(u16, f32)>,
    value: f32,
    outcome_is_terminal: bool,
    outcome_source: &'static str,
    root_search: RootSearchTelemetry,
}

fn temperature_for_ply(ply: usize) -> f32 {
    match ply {
        0..=20 => 0.8,
        21..=40 => 0.4,
        _ => 0.1,
    }
}

fn to_root_telemetry(
    diagnostics: SearchDiagnostics,
    noise: RootNoiseConfig,
    root_state: &GameState,
    _engine: &Engine,
) -> RootSearchTelemetry {
    let actions = diagnostics
        .root_actions
        .into_iter()
        .map(|stat| {
            let action_category = match &stat.action {
                Action::Move(_) => "move",
                Action::UseCard { .. } => "card",
                Action::UsePieceAbility { .. } => "ability",
                Action::DraftPick { .. } => "draft",
                Action::ResolveChoice { .. } => "choice",
            }
            .to_string();
            let is_capture = matches!(
                &stat.action,
                Action::Move(m) if root_state.board.get(m.to).is_some()
            );
            let is_king_capture_threat = matches!(
                &stat.action,
                Action::Move(m) if root_state.board.get(m.to).is_some_and(|piece| {
                    piece.owner != root_state.turn && piece.kind == PieceKind::King
                })
            );
            RootActionTrace {
                action_index: stat.action_index,
                action_category,
                is_capture,
                is_king_capture_threat,
                raw_prior: stat.raw_prior,
                root_noise: stat.root_noise.unwrap_or(0.0),
                effective_prior: stat.effective_prior,
                visit_count: stat.visits,
                first_visit_simulation: stat.first_visit_simulation,
                q_root: stat.q_root,
            }
        })
        .collect();

    RootSearchTelemetry {
        root_raw_value: diagnostics.root_raw_value,
        legal_action_count: diagnostics.root_legal_actions,
        visited_action_count: diagnostics.root_visited_action_count,
        root_visit_entropy: diagnostics.root_visit_entropy,
        top1_visit_fraction: diagnostics.top1_visit_fraction,
        effective_branching_factor: diagnostics.effective_branching_factor,
        low_prior_visit_rate: diagnostics.low_prior_visit_rate,
        max_search_depth: diagnostics.max_search_depth,
        dirichlet_alpha: noise.alpha,
        dirichlet_epsilon: noise.epsilon,
        noise_seed: noise.seed,
        actions,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: usize = args
        .iter()
        .position(|arg| arg == "--games")
        .and_then(|pos| args.get(pos + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(100);
    let sims: usize = args
        .iter()
        .position(|arg| arg == "--sims")
        .and_then(|pos| args.get(pos + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(400);
    let threads: usize = args
        .iter()
        .position(|arg| arg == "--threads")
        .and_then(|pos| args.get(pos + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let max_plies: usize = args
        .iter()
        .position(|arg| arg == "--max-plies")
        .and_then(|pos| args.get(pos + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(65);
    let terminal_only = !args.iter().any(|arg| arg == "--include-truncated");
    let append_output = args.iter().any(|arg| arg == "--append");
    let start_game_id: usize = args
        .iter()
        .position(|arg| arg == "--start-game-id")
        .and_then(|pos| args.get(pos + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let out_path = args
        .iter()
        .position(|arg| arg == "--out")
        .and_then(|pos| args.get(pos + 1))
        .cloned()
        .unwrap_or_else(|| "replay-buffers/v2/selfplay-gen2-audit.jsonl".to_string());

    let log_path = "runs/v2/selfplay-gen2.log";
    if let Some(parent) = Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).expect("Failed to create Gen2 output directory");
    }
    if let Some(parent) = Path::new(log_path).parent() {
        std::fs::create_dir_all(parent).expect("Failed to create Gen2 log directory");
    }
    let log = Arc::new(Mutex::new(
        File::create(log_path).expect("Failed to create Gen2 log"),
    ));
    {
        let mut log_file = log.lock().unwrap();
        writeln!(log_file, "AUGMENTIA GEN2 SELF-PLAY COVERAGE AUDIT").unwrap();
        writeln!(log_file, "parent=augment_net_v2_canonical_baseline").unwrap();
        writeln!(
            log_file,
            "games={} sims={} threads={} max_plies={}",
            games, sims, threads, max_plies
        )
        .unwrap();
        writeln!(log_file, "root_noise=Dirichlet(alpha=0.3, epsilon=0.25)").unwrap();
        writeln!(
            log_file,
            "temperature_schedule=ply<=20:0.8, ply<=40:0.4, later:0.1"
        )
        .unwrap();
        writeln!(log_file, "out={}", out_path).unwrap();
        writeln!(log_file, "terminal_only={}", terminal_only).unwrap();
        writeln!(log_file, "append_output={}", append_output).unwrap();
        writeln!(log_file, "start_game_id={}", start_game_id).unwrap();
        log_file.flush().unwrap();
    }

    let output = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .append(append_output)
            .truncate(!append_output)
            .open(&out_path)
            .expect("Failed to open Gen2 output"),
    ));
    let completed_games = Arc::new(AtomicUsize::new(0));
    let total_samples = Arc::new(AtomicUsize::new(0));
    let t0 = Instant::now();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to create Gen2 worker pool");

    pool.install(|| {
        let next_game = Arc::new(AtomicUsize::new(start_game_id));
        let end_game_id = start_game_id.saturating_add(games);
        (0..threads).into_par_iter().for_each(|_worker_id| {
            let engine = Engine::bundled_20260807().expect("Failed to load engine");
            let evaluator = OnnxNeuralEvaluator::from_file(
                "models/baseline/augment_net_v2_canonical_baseline.onnx",
            )
            .expect("Failed to load frozen baseline ONNX");

            loop {
                let game_id = next_game.fetch_add(1, Ordering::Relaxed);
                if game_id >= end_game_id {
                    break;
                }
                evaluator.clear_cache();
                let seed = 20260817u64.wrapping_add((game_id as u64).wrapping_mul(10007));
                let mut state = engine
                    .new_game_transition(Mode::Standard, seed)
                    .expect("Failed to initialize game")
                    .state;
                let config = MctsConfig {
                    c_puct: 1.5,
                    num_simulations: sims,
                    temperature: 0.0,
                };

                #[allow(clippy::type_complexity)]
                let mut game_history: Vec<(
                    GameState,
                    Vec<(u16, f32)>,
                    RootSearchTelemetry,
                )> = Vec::new();
                let mut ply = 0usize;

                while state.status == GameStatus::Ongoing && ply < max_plies {
                    ply += 1;
                    let noise = RootNoiseConfig {
                        alpha: 0.3,
                        epsilon: 0.25,
                        seed: seed.wrapping_mul(1_000_003).wrapping_add(ply as u64),
                    };
                    let mut tree = MctsTree::new(config);
                    let search =
                        tree.search_with_root_noise(&state, &engine, &evaluator, Some(noise));
                    let Some((_, _, visits)) = search else {
                        break;
                    };
                    let diagnostics = tree
                        .last_diagnostics
                        .clone()
                        .expect("search diagnostics must exist after a normal search");
                    let total_visits: u32 = visits.iter().map(|(_, _, count)| *count).sum();
                    let policy_target: Vec<(u16, f32)> = visits
                        .iter()
                        .filter(|(_, _, count)| *count > 0)
                        .map(|(_, index, count)| {
                            (*index, *count as f32 / total_visits.max(1) as f32)
                        })
                        .collect();
                    let played_temperature = temperature_for_ply(ply);
                    let sampled = MctsTree::sample_action_from_visits(
                        &visits,
                        played_temperature,
                        seed.wrapping_add((ply as u64).wrapping_mul(7_919)),
                    );
                    let Some((played_action, _, _)) = sampled else {
                        break;
                    };

                    game_history.push((
                        state.clone(),
                        policy_target,
                        to_root_telemetry(diagnostics, noise, &state, &engine),
                    ));
                    if let Ok(transition) = engine.apply(&state, played_action) {
                        state = transition.state;
                    } else {
                        break;
                    }
                }

                let outcome_is_terminal = state.status != GameStatus::Ongoing;
                if terminal_only && !outcome_is_terminal {
                    let done = completed_games.fetch_add(1, Ordering::Relaxed) + 1;
                    let mut log_file = log.lock().unwrap();
                    writeln!(
                        log_file,
                        "[{}/{}] discarded_truncated game_id={} plies={}",
                        done, games, game_id, ply
                    )
                    .unwrap();
                    log_file.flush().unwrap();
                    continue;
                }

                let winner = match state.status {
                    GameStatus::Won { winner, .. } => Some(winner),
                    _ => None,
                };
                let outcome_source = match state.status {
                    GameStatus::Won { .. } => "terminal_win",
                    GameStatus::Draw { .. } => "terminal_draw",
                    GameStatus::Ongoing => "truncated_assumed_draw",
                };
                let mut lines = Vec::with_capacity(game_history.len());
                for (step_idx, (position, policy, telemetry)) in
                    game_history.into_iter().enumerate()
                {
                    let value = match winner {
                        Some(player) if player == position.turn => 1.0,
                        Some(_) => -1.0,
                        None => 0.0,
                    };
                    let encoded = encode_state(&position);
                    let mask = generate_action_mask(&position, &engine)
                        .to_bool_vec()
                        .into_iter()
                        .enumerate()
                        .filter(|(_, enabled)| *enabled)
                        .map(|(index, _)| index)
                        .collect();
                    let sample = SelfPlaySample {
                        schema_version: SELF_PLAY_V2_RECORD_SCHEMA_ID,
                        ruleset_id: BUNDLED_RULESET_ID,
                        observation_schema_id: OBSERVATION_V2_SCHEMA_ID,
                        action_schema_id: ACTION_V2_SCHEMA_ID,
                        model_id: V2_BASELINE_MODEL_ID,
                        game_id,
                        ply: step_idx + 1,
                        seed,
                        spatial: encoded.spatial,
                        global: encoded.global,
                        mask,
                        policy,
                        value,
                        outcome_is_terminal,
                        outcome_source,
                        root_search: telemetry,
                    };
                    lines.push(serde_json::to_string(&sample).unwrap());
                }

                let sample_count = lines.len();
                {
                    let mut out = output.lock().unwrap();
                    for line in lines {
                        writeln!(out, "{}", line).unwrap();
                    }
                    out.flush().unwrap();
                }
                total_samples.fetch_add(sample_count, Ordering::Relaxed);
                let done = completed_games.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 5 == 0 || done == games {
                    let elapsed = t0.elapsed().as_secs_f32();
                    let sample_count = total_samples.load(Ordering::Relaxed);
                    let mut log_file = log.lock().unwrap();
                    writeln!(
                        log_file,
                        "[{}/{}] samples={} games_per_sec={:.3} elapsed_sec={:.1}",
                        done,
                        games,
                        sample_count,
                        done as f32 / elapsed.max(0.001),
                        elapsed,
                    )
                    .unwrap();
                    log_file.flush().unwrap();
                }
            }
        });
    });

    let mut log_file = log.lock().unwrap();
    writeln!(
        log_file,
        "COMPLETE games={} samples={} elapsed_sec={:.1}",
        completed_games.load(Ordering::Relaxed),
        total_samples.load(Ordering::Relaxed),
        t0.elapsed().as_secs_f32(),
    )
    .unwrap();
    log_file.flush().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_sample_always_carries_schema_provenance() {
        let sample = SelfPlaySample {
            schema_version: SELF_PLAY_V2_RECORD_SCHEMA_ID,
            ruleset_id: BUNDLED_RULESET_ID,
            observation_schema_id: OBSERVATION_V2_SCHEMA_ID,
            action_schema_id: ACTION_V2_SCHEMA_ID,
            model_id: V2_BASELINE_MODEL_ID,
            game_id: 0,
            ply: 1,
            seed: 7,
            spatial: Vec::new(),
            global: Vec::new(),
            mask: Vec::new(),
            policy: Vec::new(),
            value: 0.0,
            outcome_is_terminal: true,
            outcome_source: "terminal_draw",
            root_search: RootSearchTelemetry {
                root_raw_value: 0.0,
                legal_action_count: 0,
                visited_action_count: 0,
                root_visit_entropy: 0.0,
                top1_visit_fraction: 0.0,
                effective_branching_factor: 0.0,
                low_prior_visit_rate: 0.0,
                max_search_depth: 0,
                dirichlet_alpha: 0.0,
                dirichlet_epsilon: 0.0,
                noise_seed: 0,
                actions: Vec::new(),
            },
        };

        let value = serde_json::to_value(sample).unwrap();
        assert_eq!(value["ruleset_id"], BUNDLED_RULESET_ID);
        assert_eq!(value["observation_schema_id"], OBSERVATION_V2_SCHEMA_ID);
        assert_eq!(value["action_schema_id"], ACTION_V2_SCHEMA_ID);
        assert_eq!(value["model_id"], V2_BASELINE_MODEL_ID);
    }
}

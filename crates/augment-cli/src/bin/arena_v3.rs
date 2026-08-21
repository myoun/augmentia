use augment_ai::{
    MctsConfig, MctsTree, MctsV3Config, MctsV3Tree, OnnxNeuralEvaluator, V3OnnxRuntime,
};
use augment_core::{Action, BUNDLED_RULESET_ID, Engine, GameStatus, Mode, Player};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum BudgetKind {
    Simulations,
    WallClock,
}

#[derive(Clone, Copy)]
struct ArenaBudget {
    kind: BudgetKind,
    simulations: usize,
    move_time_ms: u64,
    max_simulations: usize,
    leaf_batch_size: usize,
}

enum Opponent {
    V3(Box<V3OnnxRuntime>),
    V2(OnnxNeuralEvaluator),
}

#[derive(Serialize)]
struct SearchUsage {
    searches: usize,
    simulations: u64,
    elapsed_ms: f64,
}

#[derive(Serialize)]
struct ArenaGame {
    pair_index: usize,
    seed: u64,
    candidate_color: &'static str,
    outcome: &'static str,
    terminal: bool,
    termination: String,
    decisions: usize,
    elapsed_ms: f64,
    candidate: SearchUsage,
    opponent: SearchUsage,
    candidate_cards: Vec<String>,
    opponent_cards: Vec<String>,
}

#[derive(Serialize)]
struct ArenaReport {
    report_schema_id: &'static str,
    ruleset_id: &'static str,
    ruleset_manifest_sha256: String,
    candidate_sha256: String,
    opponent_kind: &'static str,
    opponent_sha256: String,
    budget_kind: BudgetKind,
    simulations: usize,
    move_time_ms: u64,
    max_simulations: usize,
    leaf_batch_size: usize,
    pairs: usize,
    base_seed: u64,
    max_decisions: usize,
    games: Vec<ArenaGame>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("V3 arena failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let candidate_model = required_path(&args, "--candidate-model")?;
    let candidate_metadata = required_path(&args, "--candidate-metadata")?;
    let opponent_kind = required(&args, "--opponent-kind")?;
    let opponent_model = required_path(&args, "--opponent-model")?;
    let ruleset_hash = required(&args, "--ruleset-hash")?;
    let budget_kind = match required(&args, "--budget-kind")?.as_str() {
        "simulations" => BudgetKind::Simulations,
        "wall_clock" => BudgetKind::WallClock,
        other => return Err(format!("unsupported budget kind {other}").into()),
    };
    let budget = ArenaBudget {
        kind: budget_kind,
        simulations: numeric(&args, "--simulations", 400usize)?,
        move_time_ms: numeric(&args, "--move-time-ms", 1000u64)?,
        max_simulations: numeric(&args, "--max-simulations", 100_000usize)?,
        leaf_batch_size: numeric(&args, "--leaf-batch-size", 16usize)?,
    };
    let pairs = numeric(&args, "--pairs", 100usize)?;
    let base_seed = numeric(&args, "--base-seed", 2026082101u64)?;
    let max_decisions = numeric(&args, "--max-decisions", 512usize)?;
    if pairs == 0
        || max_decisions == 0
        || budget.simulations == 0
        || budget.move_time_ms == 0
        || budget.max_simulations == 0
        || budget.leaf_batch_size == 0
    {
        return Err("arena counts and budgets must be positive".into());
    }
    let candidate = V3OnnxRuntime::from_files(
        &candidate_model,
        &candidate_metadata,
        BUNDLED_RULESET_ID,
        &ruleset_hash,
    )?;
    let opponent_metadata = optional_path(&args, "--opponent-metadata");
    let opponent = match opponent_kind.as_str() {
        "v3" => Opponent::V3(Box::new(V3OnnxRuntime::from_files(
            &opponent_model,
            opponent_metadata.ok_or("--opponent-metadata is required for V3")?,
            BUNDLED_RULESET_ID,
            &ruleset_hash,
        )?)),
        "v2" => Opponent::V2(OnnxNeuralEvaluator::from_file(&opponent_model)?),
        other => return Err(format!("unsupported opponent kind {other}").into()),
    };
    let engine = Engine::bundled_20260807()?;
    let mut games = Vec::with_capacity(pairs * 2);
    for pair_index in 0..pairs {
        let seed = base_seed.wrapping_add(pair_index as u64);
        games.push(play_game(
            &engine,
            &candidate,
            &opponent,
            budget,
            pair_index,
            seed,
            Player::White,
            max_decisions,
        ));
        games.push(play_game(
            &engine,
            &candidate,
            &opponent,
            budget,
            pair_index,
            seed,
            Player::Black,
            max_decisions,
        ));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&ArenaReport {
            report_schema_id: "augmentia.arena.v3",
            ruleset_id: BUNDLED_RULESET_ID,
            ruleset_manifest_sha256: ruleset_hash,
            candidate_sha256: sha256_file(&candidate_model)?,
            opponent_kind: if matches!(opponent, Opponent::V3(_)) {
                "v3"
            } else {
                "v2"
            },
            opponent_sha256: sha256_file(&opponent_model)?,
            budget_kind,
            simulations: budget.simulations,
            move_time_ms: budget.move_time_ms,
            max_simulations: budget.max_simulations,
            leaf_batch_size: budget.leaf_batch_size,
            pairs,
            base_seed,
            max_decisions,
            games,
        })?
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn play_game(
    engine: &Engine,
    candidate: &V3OnnxRuntime,
    opponent: &Opponent,
    budget: ArenaBudget,
    pair_index: usize,
    seed: u64,
    candidate_seat: Player,
    max_decisions: usize,
) -> ArenaGame {
    let started = Instant::now();
    let mut state = match engine.new_game(Mode::Standard, seed) {
        Ok(state) => state,
        Err(error) => {
            return failed_game(pair_index, seed, candidate_seat, error.to_string());
        }
    };
    let mut candidate_searches = 0usize;
    let mut candidate_simulations = 0u64;
    let mut candidate_elapsed = 0.0;
    let mut opponent_searches = 0usize;
    let mut opponent_simulations = 0u64;
    let mut opponent_elapsed = 0.0;
    let mut candidate_cards = BTreeSet::new();
    let mut opponent_cards = BTreeSet::new();
    let mut decisions = 0usize;
    let mut search_error = None;

    while state.status == GameStatus::Ongoing && decisions < max_decisions {
        record_cards(
            &state,
            candidate_seat,
            &mut candidate_cards,
            &mut opponent_cards,
        );
        let candidate_turn = state.turn == candidate_seat;
        let search_started = Instant::now();
        let selected = if candidate_turn {
            search_v3(&state, engine, candidate, budget)
        } else {
            search_opponent(&state, engine, opponent, budget)
        };
        let elapsed = search_started.elapsed().as_secs_f64() * 1000.0;
        let (action, simulations) = match selected {
            Ok(result) => result,
            Err(error) => {
                search_error = Some(error);
                break;
            }
        };
        if candidate_turn {
            candidate_searches += 1;
            candidate_simulations += simulations as u64;
            candidate_elapsed += elapsed;
        } else {
            opponent_searches += 1;
            opponent_simulations += simulations as u64;
            opponent_elapsed += elapsed;
        }
        match engine.apply(&state, action) {
            Ok(transition) => state = transition.state,
            Err(error) => {
                search_error = Some(format!("engine apply failed: {error}"));
                break;
            }
        }
        decisions += 1;
    }
    record_cards(
        &state,
        candidate_seat,
        &mut candidate_cards,
        &mut opponent_cards,
    );
    let (outcome, terminal, termination) = if let Some(error) = search_error {
        ("error", false, error)
    } else {
        match &state.status {
            GameStatus::Won { winner, reason } => (
                if *winner == candidate_seat {
                    "candidate_win"
                } else {
                    "opponent_win"
                },
                true,
                format!("{reason:?}"),
            ),
            GameStatus::Draw { reason } => ("draw", true, format!("{reason:?}")),
            GameStatus::Ongoing => ("limit", false, "max_decisions".to_owned()),
        }
    };
    ArenaGame {
        pair_index,
        seed,
        candidate_color: player_name(candidate_seat),
        outcome,
        terminal,
        termination,
        decisions,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        candidate: SearchUsage {
            searches: candidate_searches,
            simulations: candidate_simulations,
            elapsed_ms: candidate_elapsed,
        },
        opponent: SearchUsage {
            searches: opponent_searches,
            simulations: opponent_simulations,
            elapsed_ms: opponent_elapsed,
        },
        candidate_cards: candidate_cards.into_iter().collect(),
        opponent_cards: opponent_cards.into_iter().collect(),
    }
}

fn search_v3(
    state: &augment_core::GameState,
    engine: &Engine,
    runtime: &V3OnnxRuntime,
    budget: ArenaBudget,
) -> Result<(Action, usize), String> {
    let cap = match budget.kind {
        BudgetKind::Simulations => budget.simulations,
        BudgetKind::WallClock => budget.max_simulations,
    };
    let mut tree = MctsV3Tree::new(MctsV3Config {
        num_simulations: cap,
        leaf_batch_size: budget.leaf_batch_size,
        ..MctsV3Config::default()
    })
    .map_err(|error| error.to_string())?;
    let result = match budget.kind {
        BudgetKind::Simulations => tree.search(state, engine, runtime),
        BudgetKind::WallClock => tree.search_for_duration(
            state,
            engine,
            runtime,
            Duration::from_millis(budget.move_time_ms),
        ),
    }
    .map_err(|error| error.to_string())?;
    let simulations = tree
        .last_diagnostics
        .as_ref()
        .map_or(0, |diagnostics| diagnostics.completed_simulations);
    Ok((result.action, simulations))
}

fn search_opponent(
    state: &augment_core::GameState,
    engine: &Engine,
    opponent: &Opponent,
    budget: ArenaBudget,
) -> Result<(Action, usize), String> {
    match opponent {
        Opponent::V3(runtime) => search_v3(state, engine, runtime, budget),
        Opponent::V2(evaluator) => {
            evaluator.clear_cache();
            let failures = evaluator.inference_failures();
            let cap = match budget.kind {
                BudgetKind::Simulations => budget.simulations,
                BudgetKind::WallClock => budget.max_simulations,
            };
            let mut tree = MctsTree::new(MctsConfig {
                num_simulations: cap,
                temperature: 0.0,
                ..MctsConfig::default()
            });
            let result = match budget.kind {
                BudgetKind::Simulations => tree.search(state, engine, evaluator),
                BudgetKind::WallClock => tree.search_for_duration(
                    state,
                    engine,
                    evaluator,
                    Duration::from_millis(budget.move_time_ms),
                ),
            }
            .ok_or_else(|| "V2 search returned no action".to_owned())?;
            if evaluator.inference_failures() != failures {
                return Err("V2 ONNX inference fell back to a heuristic".to_owned());
            }
            let simulations = result.2.iter().map(|item| item.2 as usize).sum();
            Ok((result.0, simulations))
        }
    }
}

fn record_cards(
    state: &augment_core::GameState,
    candidate_seat: Player,
    candidate: &mut BTreeSet<String>,
    opponent: &mut BTreeSet<String>,
) {
    for card in &state.cards {
        if card.owner == candidate_seat {
            candidate.insert(card.definition_id.clone());
        } else {
            opponent.insert(card.definition_id.clone());
        }
    }
}

fn failed_game(pair_index: usize, seed: u64, candidate_seat: Player, error: String) -> ArenaGame {
    ArenaGame {
        pair_index,
        seed,
        candidate_color: player_name(candidate_seat),
        outcome: "error",
        terminal: false,
        termination: error,
        decisions: 0,
        elapsed_ms: 0.0,
        candidate: SearchUsage {
            searches: 0,
            simulations: 0,
            elapsed_ms: 0.0,
        },
        opponent: SearchUsage {
            searches: 0,
            simulations: 0,
            elapsed_ms: 0.0,
        },
        candidate_cards: Vec::new(),
        opponent_cards: Vec::new(),
    }
}

fn player_name(player: Player) -> &'static str {
    match player {
        Player::White => "white",
        Player::Black => "black",
    }
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn required(args: &[String], flag: &str) -> Result<String, String> {
    args.iter()
        .position(|item| item == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .ok_or_else(|| format!("{flag} is required"))
}

fn required_path(args: &[String], flag: &str) -> Result<PathBuf, String> {
    required(args, flag).map(PathBuf::from)
}

fn optional_path(args: &[String], flag: &str) -> Option<PathBuf> {
    args.iter()
        .position(|item| item == flag)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
}

fn numeric<T>(args: &[String], flag: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    args.iter()
        .position(|item| item == flag)
        .and_then(|index| args.get(index + 1))
        .map_or(Ok(default), |value| {
            value.parse().map_err(|error| format!("{flag}: {error}"))
        })
}

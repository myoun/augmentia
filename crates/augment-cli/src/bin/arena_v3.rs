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

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum SearchVariant {
    FairPuct,
    Reuse,
    Widening,
    Gumbel,
    VirtualChameleon,
    WideningVirtualChameleon,
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
    candidate_budget: ArenaBudgetReport,
    opponent_budget: ArenaBudgetReport,
    candidate_search_variant: SearchVariant,
    opponent_search_variant: SearchVariant,
    pairs: usize,
    base_seed: u64,
    max_decisions: usize,
    games: Vec<ArenaGame>,
}

#[derive(Serialize)]
struct ArenaBudgetReport {
    simulations: usize,
    move_time_ms: u64,
    max_simulations: usize,
    leaf_batch_size: usize,
}

impl From<ArenaBudget> for ArenaBudgetReport {
    fn from(value: ArenaBudget) -> Self {
        Self {
            simulations: value.simulations,
            move_time_ms: value.move_time_ms,
            max_simulations: value.max_simulations,
            leaf_batch_size: value.leaf_batch_size,
        }
    }
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
    let base_budget = ArenaBudget {
        kind: budget_kind,
        simulations: numeric(&args, "--simulations", 400usize)?,
        move_time_ms: numeric(&args, "--move-time-ms", 1000u64)?,
        max_simulations: numeric(&args, "--max-simulations", 100_000usize)?,
        leaf_batch_size: numeric(&args, "--leaf-batch-size", 16usize)?,
    };
    let candidate_budget = ArenaBudget {
        kind: budget_kind,
        simulations: numeric(&args, "--candidate-simulations", base_budget.simulations)?,
        move_time_ms: numeric(&args, "--candidate-move-time-ms", base_budget.move_time_ms)?,
        max_simulations: numeric(
            &args,
            "--candidate-max-simulations",
            base_budget.max_simulations,
        )?,
        leaf_batch_size: numeric(
            &args,
            "--candidate-leaf-batch-size",
            base_budget.leaf_batch_size,
        )?,
    };
    let opponent_budget = ArenaBudget {
        kind: budget_kind,
        simulations: numeric(&args, "--opponent-simulations", base_budget.simulations)?,
        move_time_ms: numeric(&args, "--opponent-move-time-ms", base_budget.move_time_ms)?,
        max_simulations: numeric(
            &args,
            "--opponent-max-simulations",
            base_budget.max_simulations,
        )?,
        leaf_batch_size: numeric(
            &args,
            "--opponent-leaf-batch-size",
            base_budget.leaf_batch_size,
        )?,
    };
    let candidate_search_variant =
        search_variant(&args, "--candidate-search", SearchVariant::FairPuct)?;
    let opponent_search_variant =
        search_variant(&args, "--opponent-search", SearchVariant::FairPuct)?;
    let pairs = numeric(&args, "--pairs", 100usize)?;
    let base_seed = numeric(&args, "--base-seed", 2026082101u64)?;
    let max_decisions = numeric(&args, "--max-decisions", 512usize)?;
    if pairs == 0
        || max_decisions == 0
        || candidate_budget.simulations == 0
        || candidate_budget.move_time_ms == 0
        || candidate_budget.max_simulations == 0
        || candidate_budget.leaf_batch_size == 0
        || opponent_budget.simulations == 0
        || opponent_budget.move_time_ms == 0
        || opponent_budget.max_simulations == 0
        || opponent_budget.leaf_batch_size == 0
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
            candidate_budget,
            opponent_budget,
            candidate_search_variant,
            opponent_search_variant,
            pair_index,
            seed,
            Player::White,
            max_decisions,
        ));
        games.push(play_game(
            &engine,
            &candidate,
            &opponent,
            candidate_budget,
            opponent_budget,
            candidate_search_variant,
            opponent_search_variant,
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
            simulations: base_budget.simulations,
            move_time_ms: base_budget.move_time_ms,
            max_simulations: base_budget.max_simulations,
            leaf_batch_size: base_budget.leaf_batch_size,
            candidate_budget: candidate_budget.into(),
            opponent_budget: opponent_budget.into(),
            candidate_search_variant,
            opponent_search_variant,
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
    candidate_budget: ArenaBudget,
    opponent_budget: ArenaBudget,
    candidate_search_variant: SearchVariant,
    opponent_search_variant: SearchVariant,
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
    let mut candidate_tree = match MctsV3Tree::new(arena_tree_config(
        candidate_budget,
        arena_search_seed(seed),
        candidate_search_variant,
    )) {
        Ok(tree) => tree,
        Err(error) => return failed_game(pair_index, seed, candidate_seat, error.to_string()),
    };
    let mut opponent_tree = if matches!(opponent, Opponent::V3(_)) {
        match MctsV3Tree::new(arena_tree_config(
            opponent_budget,
            arena_search_seed(seed),
            opponent_search_variant,
        )) {
            Ok(tree) => Some(tree),
            Err(error) => return failed_game(pair_index, seed, candidate_seat, error.to_string()),
        }
    } else {
        None
    };

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
            search_v3(
                &mut candidate_tree,
                &state,
                engine,
                candidate,
                candidate_budget,
            )
        } else {
            search_opponent(
                opponent_tree.as_mut(),
                &state,
                engine,
                opponent,
                opponent_budget,
            )
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
    tree: &mut MctsV3Tree,
    state: &augment_core::GameState,
    engine: &Engine,
    runtime: &V3OnnxRuntime,
    budget: ArenaBudget,
) -> Result<(Action, usize), String> {
    let cap = match budget.kind {
        BudgetKind::Simulations => budget.simulations,
        BudgetKind::WallClock => budget.max_simulations,
    };
    tree.set_search_budget(cap, budget.leaf_batch_size)
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
    v3_tree: Option<&mut MctsV3Tree>,
    state: &augment_core::GameState,
    engine: &Engine,
    opponent: &Opponent,
    budget: ArenaBudget,
) -> Result<(Action, usize), String> {
    match opponent {
        Opponent::V3(runtime) => search_v3(
            v3_tree.ok_or_else(|| "V3 opponent tree is missing".to_owned())?,
            state,
            engine,
            runtime,
            budget,
        ),
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

fn arena_search_seed(seed: u64) -> u64 {
    seed ^ 0x4152_454e_415f_5631
}

fn arena_tree_config(
    budget: ArenaBudget,
    search_seed: u64,
    variant: SearchVariant,
) -> MctsV3Config {
    let cap = match budget.kind {
        BudgetKind::Simulations => budget.simulations,
        BudgetKind::WallClock => budget.max_simulations,
    };
    let mut config = MctsV3Config {
        num_simulations: cap,
        leaf_batch_size: budget.leaf_batch_size,
        search_seed,
        proposal_seed: search_seed ^ 0x5052_4f50_4f53_414c,
        gumbel_seed: search_seed ^ 0x4755_4d42_454c_5f31,
        gumbel_planning_budget: budget.simulations,
        reuse_descendant_depth: 0,
        ..MctsV3Config::default()
    };
    match variant {
        SearchVariant::FairPuct => {}
        SearchVariant::Reuse => config.reuse_descendant_depth = 4,
        SearchVariant::Widening => config.progressive_widening = true,
        SearchVariant::Gumbel => {
            config.gumbel_root = true;
            config.gumbel_candidates = if cap <= 150 { 32 } else { 64 };
        }
        SearchVariant::VirtualChameleon => config.virtual_chameleon_hierarchy = true,
        SearchVariant::WideningVirtualChameleon => {
            config.progressive_widening = true;
            config.virtual_chameleon_hierarchy = true;
        }
    }
    config
}

fn search_variant(
    args: &[String],
    flag: &str,
    default: SearchVariant,
) -> Result<SearchVariant, Box<dyn std::error::Error>> {
    let Some(value) = args
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].as_str())
    else {
        return Ok(default);
    };
    match value {
        "fair_puct" => Ok(SearchVariant::FairPuct),
        "reuse" => Ok(SearchVariant::Reuse),
        "widening" => Ok(SearchVariant::Widening),
        "gumbel" => Ok(SearchVariant::Gumbel),
        "virtual_chameleon" => Ok(SearchVariant::VirtualChameleon),
        "widening_virtual_chameleon" => Ok(SearchVariant::WideningVirtualChameleon),
        other => Err(format!("unsupported search variant {other:?}").into()),
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

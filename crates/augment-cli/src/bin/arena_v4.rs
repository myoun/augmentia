use augment_ai::data_v3::{
    AbsolutePlayerV3, AgentKindV3, GameResultV3, GeneratorProvenanceV3, ModelArtifactV3,
    PlayerAgentV3, RulesetRefV3, TerminationV3,
};
use augment_ai::{
    CHANCE_SAMPLING_V4_ID, ROOT_SELECTION_V4_ID, SEARCH_SEMANTICS_V4_ID, SelfPlayAgentV4,
    SelfPlayGameConfigV4, V3ExecutionProvider, V3RuntimeOptions, V4OnnxRuntime,
    generate_self_play_game_v4,
};
use augment_core::{Engine, Mode, Player};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const RULESET_ID: &str = "augment_chess_20260915";
const RULESET_MANIFEST_SHA256: &str =
    "7b2736f8e9f727db818d8c8f4aaf7742a23a717e2f31d749ea293a3dfc051b13";
const SOURCE_SEAL_TAG: &str = "ai/v4-training-base-20260919";
const SOURCE_SEAL_COMMIT: &str = "f4e64c35f5adcb7f361ed845fb55eb4d123b370b";

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum ArenaOutcome {
    CandidateWin,
    OpponentWin,
    Draw,
    Truncated,
}

#[derive(Debug, Serialize)]
struct ArenaGame {
    pair_index: usize,
    seed: u64,
    candidate_color: &'static str,
    outcome: ArenaOutcome,
    terminal: bool,
    decisions: usize,
    elapsed_seconds: f64,
    relative_path: String,
    trajectory_sha256: String,
    termination: TerminationV3,
}

#[derive(Debug, Serialize)]
struct ArenaSummary {
    candidate_wins: usize,
    opponent_wins: usize,
    draws: usize,
    truncated: usize,
    complete_pairs: usize,
    pair_score_mean: Option<f64>,
    pair_score_standard_error: Option<f64>,
    pair_score_ci95_low: Option<f64>,
    pair_score_ci95_high: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ArenaReport {
    report_schema_id: &'static str,
    run_id: String,
    source_seal_tag: &'static str,
    source_seal_commit: &'static str,
    engine_commit: String,
    dirty_worktree: bool,
    ruleset_id: &'static str,
    ruleset_manifest_sha256: &'static str,
    candidate_bundle_path: String,
    candidate_bundle_sha256: String,
    opponent_bundle_path: String,
    opponent_bundle_sha256: String,
    pairs: usize,
    base_seed: u64,
    simulations: usize,
    chance_outcome_samples: usize,
    temperature: f32,
    temperature_decisions: usize,
    max_decisions: usize,
    execution_provider: String,
    intra_threads: usize,
    search_semantics_id: &'static str,
    root_selection_id: &'static str,
    chance_sampling_id: &'static str,
    summary: ArenaSummary,
    games: Vec<ArenaGame>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("V4 arena failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }

    let run_id = required(&args, "--run-id")?;
    let output_dir = required_path(&args, "--output-dir")?;
    let candidate_bundle = required_path(&args, "--candidate-bundle")?;
    let opponent_bundle = required_path(&args, "--opponent-bundle")?;
    let pairs = numeric(&args, "--pairs", 16usize)?;
    let base_seed = numeric(&args, "--base-seed", 2026092701u64)?;
    let simulations = numeric(&args, "--simulations", 100usize)?;
    let chance_outcome_samples = numeric(&args, "--chance-outcome-samples", 8usize)?;
    let max_decisions = numeric(&args, "--max-decisions", 512usize)?;
    let intra_threads = numeric(&args, "--intra-threads", 1usize)?;
    let execution_provider = parse_execution_provider(&args)?;

    if pairs == 0 || simulations == 0 || max_decisions == 0 || intra_threads == 0 {
        return Err("arena counts and budgets must be positive".into());
    }
    if !(1..=64).contains(&chance_outcome_samples) {
        return Err("--chance-outcome-samples must be in 1..=64".into());
    }
    refuse_overwrite(&output_dir)?;

    let repo_root = git_path(["rev-parse", "--show-toplevel"])
        .ok_or("arena_v4 must run inside the Augmentia Git worktree")?;
    let engine_commit =
        git_output_in(&repo_root, ["rev-parse", "HEAD"]).ok_or("failed to read Git HEAD")?;
    let dirty_worktree = git_output_in(&repo_root, ["status", "--porcelain"])
        .is_some_and(|output| !output.is_empty());
    if dirty_worktree {
        return Err("refusing official V4 arena from a dirty worktree".into());
    }
    let sealed_commit = git_output_in(&repo_root, ["rev-list", "-n", "1", SOURCE_SEAL_TAG])
        .ok_or("source seal tag is missing")?;
    if sealed_commit != SOURCE_SEAL_COMMIT {
        return Err("source seal tag resolves to an unexpected commit".into());
    }

    let options = V3RuntimeOptions {
        execution_provider,
        intra_threads,
    };
    let candidate = V4OnnxRuntime::from_bundle_with_options(
        &candidate_bundle,
        RULESET_ID,
        RULESET_MANIFEST_SHA256,
        options,
    )?;
    let opponent = V4OnnxRuntime::from_bundle_with_options(
        &opponent_bundle,
        RULESET_ID,
        RULESET_MANIFEST_SHA256,
        options,
    )?;
    let candidate_sha = sha256_file(&candidate_bundle)?;
    let opponent_sha = sha256_file(&opponent_bundle)?;

    std::fs::create_dir_all(output_dir.join("games"))?;
    let engine = Engine::bundled_20260915()?;
    if engine.ruleset().id != RULESET_ID {
        return Err("bundled engine ruleset identity mismatch".into());
    }

    let ruleset = RulesetRefV3 {
        id: RULESET_ID.to_owned(),
        manifest_sha256: RULESET_MANIFEST_SHA256.to_owned(),
    };
    let generator = GeneratorProvenanceV3 {
        run_id: run_id.clone(),
        engine_commit: engine_commit.clone(),
        engine_api_version: 1,
        collector_id: "augment-cli/arena_v4:production".to_owned(),
        dirty_worktree,
    };
    let config_id = format!(
        "mcts-v4:sims={simulations}:chance={chance_outcome_samples}:temperature=0:root={ROOT_SELECTION_V4_ID}:search={SEARCH_SEMANTICS_V4_ID}"
    );
    let game_config = |seed| SelfPlayGameConfigV4 {
        seed,
        max_decisions,
        simulations,
        chance_outcome_samples,
        temperature: 0.0,
        temperature_decisions: 0,
    };

    let candidate_model = ModelArtifactV3 {
        model_id: "candidate-v4-bundle".to_owned(),
        sha256: candidate_sha.clone(),
    };
    let opponent_model = ModelArtifactV3 {
        model_id: "opponent-v4-bundle".to_owned(),
        sha256: opponent_sha.clone(),
    };

    let mut games = Vec::with_capacity(pairs * 2);
    for pair_index in 0..pairs {
        let seed = base_seed.wrapping_add(pair_index as u64);
        games.push(play_one(
            &engine,
            &candidate,
            &opponent,
            &candidate_model,
            &opponent_model,
            &ruleset,
            &generator,
            &config_id,
            &run_id,
            &output_dir,
            pair_index,
            seed,
            Player::White,
            game_config(seed),
        )?);
        games.push(play_one(
            &engine,
            &candidate,
            &opponent,
            &candidate_model,
            &opponent_model,
            &ruleset,
            &generator,
            &config_id,
            &run_id,
            &output_dir,
            pair_index,
            seed,
            Player::Black,
            game_config(seed),
        )?);
        eprintln!("completed pair {}/{}", pair_index + 1, pairs);
    }

    let summary = summarize(&games, pairs);
    let report = ArenaReport {
        report_schema_id: "augmentia.arena.v4.v1",
        run_id,
        source_seal_tag: SOURCE_SEAL_TAG,
        source_seal_commit: SOURCE_SEAL_COMMIT,
        engine_commit,
        dirty_worktree,
        ruleset_id: RULESET_ID,
        ruleset_manifest_sha256: RULESET_MANIFEST_SHA256,
        candidate_bundle_path: candidate_bundle.display().to_string(),
        candidate_bundle_sha256: candidate_sha,
        opponent_bundle_path: opponent_bundle.display().to_string(),
        opponent_bundle_sha256: opponent_sha,
        pairs,
        base_seed,
        simulations,
        chance_outcome_samples,
        temperature: 0.0,
        temperature_decisions: 0,
        max_decisions,
        execution_provider: execution_provider.as_str().to_owned(),
        intra_threads,
        search_semantics_id: SEARCH_SEMANTICS_V4_ID,
        root_selection_id: ROOT_SELECTION_V4_ID,
        chance_sampling_id: CHANCE_SAMPLING_V4_ID,
        summary,
        games,
    };
    std::fs::write(
        output_dir.join("arena.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report.summary)?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn play_one(
    engine: &Engine,
    candidate: &V4OnnxRuntime,
    opponent: &V4OnnxRuntime,
    candidate_model: &ModelArtifactV3,
    opponent_model: &ModelArtifactV3,
    ruleset: &RulesetRefV3,
    generator: &GeneratorProvenanceV3,
    config_id: &str,
    run_id: &str,
    output_dir: &Path,
    pair_index: usize,
    seed: u64,
    candidate_seat: Player,
    config: SelfPlayGameConfigV4,
) -> Result<ArenaGame, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let candidate_agent = SelfPlayAgentV4 {
        evaluator: candidate,
        provenance: agent_provenance(candidate_seat, "candidate-v4", candidate_model, config_id),
    };
    let opponent_seat = match candidate_seat {
        Player::White => Player::Black,
        Player::Black => Player::White,
    };
    let opponent_agent = SelfPlayAgentV4 {
        evaluator: opponent,
        provenance: agent_provenance(opponent_seat, "opponent-v4", opponent_model, config_id),
    };
    let agents = match candidate_seat {
        Player::White => [&candidate_agent, &opponent_agent],
        Player::Black => [&opponent_agent, &candidate_agent],
    };
    let color = match candidate_seat {
        Player::White => "white",
        Player::Black => "black",
    };
    let game_id = format!("{run_id}-pair-{pair_index:04}-{color}");
    let record = generate_self_play_game_v4(
        engine,
        Mode::Standard,
        game_id.clone(),
        ruleset.clone(),
        generator.clone(),
        agents,
        config,
    )?;
    let (outcome, terminal) = outcome_for_candidate(&record.termination, candidate_seat);
    let file_name = format!("pair-{pair_index:04}-candidate-{color}.json");
    let relative_path = format!("games/{file_name}");
    std::fs::write(
        output_dir.join(&relative_path),
        serde_json::to_vec_pretty(&record)?,
    )?;
    Ok(ArenaGame {
        pair_index,
        seed,
        candidate_color: color,
        outcome,
        terminal,
        decisions: record.steps.len(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        relative_path,
        trajectory_sha256: record.trajectory_sha256,
        termination: record.termination,
    })
}

fn agent_provenance(
    seat: Player,
    agent_id: &str,
    model: &ModelArtifactV3,
    config_id: &str,
) -> PlayerAgentV3 {
    PlayerAgentV3 {
        seat: AbsolutePlayerV3::from_player(seat),
        agent_kind: AgentKindV3::Mcts,
        agent_id: agent_id.to_owned(),
        model: Some(model.clone()),
        config_id: Some(config_id.to_owned()),
    }
}

fn outcome_for_candidate(
    termination: &TerminationV3,
    candidate_seat: Player,
) -> (ArenaOutcome, bool) {
    match termination {
        TerminationV3::EngineTerminal { result, .. } => {
            let outcome = match result {
                GameResultV3::Draw => ArenaOutcome::Draw,
                GameResultV3::WhiteWin if candidate_seat == Player::White => {
                    ArenaOutcome::CandidateWin
                }
                GameResultV3::BlackWin if candidate_seat == Player::Black => {
                    ArenaOutcome::CandidateWin
                }
                GameResultV3::WhiteWin | GameResultV3::BlackWin => ArenaOutcome::OpponentWin,
            };
            (outcome, true)
        }
        _ => (ArenaOutcome::Truncated, false),
    }
}

fn summarize(games: &[ArenaGame], pairs: usize) -> ArenaSummary {
    let mut candidate_wins = 0usize;
    let mut opponent_wins = 0usize;
    let mut draws = 0usize;
    let mut truncated = 0usize;
    for game in games {
        match game.outcome {
            ArenaOutcome::CandidateWin => candidate_wins += 1,
            ArenaOutcome::OpponentWin => opponent_wins += 1,
            ArenaOutcome::Draw => draws += 1,
            ArenaOutcome::Truncated => truncated += 1,
        }
    }

    let mut pair_scores = Vec::new();
    for pair_index in 0..pairs {
        let pair = games
            .iter()
            .filter(|game| game.pair_index == pair_index)
            .collect::<Vec<_>>();
        if pair.len() != 2 || pair.iter().any(|game| !game.terminal) {
            continue;
        }
        let score = pair
            .iter()
            .map(|game| game_score(game.outcome))
            .sum::<f64>()
            / 2.0;
        pair_scores.push(score);
    }
    let complete_pairs = pair_scores.len();
    let (mean, se, low, high) = mean_ci95(&pair_scores);

    ArenaSummary {
        candidate_wins,
        opponent_wins,
        draws,
        truncated,
        complete_pairs,
        pair_score_mean: mean,
        pair_score_standard_error: se,
        pair_score_ci95_low: low,
        pair_score_ci95_high: high,
    }
}

fn game_score(outcome: ArenaOutcome) -> f64 {
    match outcome {
        ArenaOutcome::CandidateWin => 1.0,
        ArenaOutcome::Draw => 0.5,
        ArenaOutcome::OpponentWin => 0.0,
        ArenaOutcome::Truncated => f64::NAN,
    }
}

fn mean_ci95(values: &[f64]) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    if values.is_empty() {
        return (None, None, None, None);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() == 1 {
        return (Some(mean), None, None, None);
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let se = (variance / n).sqrt();
    (
        Some(mean),
        Some(se),
        Some((mean - 1.96 * se).max(0.0)),
        Some((mean + 1.96 * se).min(1.0)),
    )
}

fn parse_execution_provider(args: &[String]) -> Result<V3ExecutionProvider, String> {
    match optional_string(args, "--execution-provider", "cpu").as_str() {
        "cpu" => Ok(V3ExecutionProvider::Cpu),
        "cuda" => Ok(V3ExecutionProvider::Cuda),
        value => Err(format!(
            "--execution-provider must be cpu or cuda, got {value:?}"
        )),
    }
}

fn refuse_overwrite(output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if output_dir.exists() {
        if !output_dir.is_dir() {
            return Err("output path exists and is not a directory".into());
        }
        if std::fs::read_dir(output_dir)?.next().transpose()?.is_some() {
            return Err("refusing to overwrite non-empty arena output directory".into());
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}

fn required(args: &[String], name: &str) -> Result<String, String> {
    args.iter()
        .position(|item| item == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .ok_or_else(|| format!("{name} is required"))
}

fn required_path(args: &[String], name: &str) -> Result<PathBuf, String> {
    required(args, name).map(PathBuf::from)
}

fn optional_string(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|item| item == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .unwrap_or_else(|| default.to_owned())
}

fn numeric<T>(args: &[String], name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    args.iter()
        .position(|item| item == name)
        .and_then(|index| args.get(index + 1))
        .map_or(Ok(default), |value| {
            value.parse().map_err(|error| format!("{name}: {error}"))
        })
}

fn git_path<const N: usize>(args: [&str; N]) -> Option<PathBuf> {
    git_output(args).map(PathBuf::from)
}

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn git_output_in<const N: usize>(repo_root: &Path, args: [&str; N]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(repo_root)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn print_help() {
    println!(
        "arena_v4 --run-id ID --output-dir DIR --candidate-bundle META --opponent-bundle META \
         [--pairs 16] [--base-seed 2026092701] [--simulations 100] \
         [--chance-outcome-samples 8] [--max-decisions 512] \
         [--execution-provider cpu|cuda] [--intra-threads 1]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_interval_tracks_pair_scores() {
        let (mean, se, low, high) = mean_ci95(&[1.0, 0.5, 0.0, 0.5]);
        assert_eq!(mean, Some(0.5));
        assert!(se.unwrap() > 0.0);
        assert!(low.unwrap() < 0.5);
        assert!(high.unwrap() > 0.5);
    }
}

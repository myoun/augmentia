use augment_ai::data_v3::{
    AbsolutePlayerV3, AgentKindV3, GeneratorProvenanceV3, ModelArtifactV3, PlayerAgentV3,
    RulesetRefV3, TerminationV3,
};
use augment_ai::{
    ACTION_V3_SCHEMA_ID, CHANCE_SAMPLING_V4_ID, DeterministicFactorizedEvaluatorV4,
    FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID, FACTORIZED_POLICY_V4_SCHEMA_ID, FactorizedEvaluatorV4,
    GAME_RECORD_V4_SCHEMA_ID, OBSERVATION_V3_SCHEMA_ID, ROOT_SELECTION_V4_ID,
    SEARCH_SEMANTICS_V4_ID, SelfPlayAgentV4, SelfPlayGameConfigV4, V3ExecutionProvider,
    V3RuntimeOptions, V4OnnxRuntime, generate_self_play_game_v4,
};
use augment_core::{Engine, Mode};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

const RULESET_ID: &str = "augment_chess_20260915";
const SOURCE_SEAL_TAG: &str = "ai/v4-training-base-20260919";
const SOURCE_SEAL_COMMIT: &str = "f4e64c35f5adcb7f361ed845fb55eb4d123b370b";
const RUN_MANIFEST_SCHEMA_ID: &str = "augmentia.self-play-v4-run.v1";

#[derive(Clone, Debug, Serialize)]
struct EvaluatorProvenanceV4 {
    kind: String,
    agent_id: String,
    model_sha256: Option<String>,
    checkpoint_sha256: Option<String>,
    bundle_metadata_path: Option<String>,
    bundle_metadata_sha256: Option<String>,
    state_encoder_sha256: Option<String>,
    factorized_scorer_sha256: Option<String>,
    bundle_code_commit: Option<String>,
    bootstrap_seed: Option<u64>,
    bootstrap_value: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
struct GameArtifactV4 {
    game_index: usize,
    game_id: String,
    seed_hex: String,
    relative_path: String,
    file_sha256: String,
    trajectory_sha256: String,
    decisions: usize,
    termination: TerminationV3,
}

#[derive(Clone, Debug, Serialize)]
struct RunManifestV4 {
    manifest_schema_id: String,
    run_id: String,
    run_kind: String,
    source_seal_tag: String,
    source_seal_commit: String,
    engine_commit: String,
    dirty_worktree: bool,
    ruleset: RulesetRefV3,
    evaluator: EvaluatorProvenanceV4,
    mode_id: String,
    game_index_start: usize,
    games: usize,
    base_seed: u64,
    seed_derivation: String,
    simulations: usize,
    chance_outcome_samples: usize,
    temperature: f32,
    temperature_decisions: usize,
    max_decisions: usize,
    workers: usize,
    execution_provider: String,
    intra_threads: usize,
    search_semantics_id: String,
    root_selection_id: String,
    chance_sampling_id: String,
    replay_schema_id: String,
    factorized_policy_schema_id: String,
    factorized_target_schema_id: String,
    observation_schema_id: String,
    action_schema_id: String,
    games_artifacts: Vec<GameArtifactV4>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("V4 self-play failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(());
    }

    let run_id = required(&args, "--run-id")?;
    validate_run_id(&run_id)?;
    let output_dir = required_path(&args, "--output-dir")?;
    let ruleset_arg = optional_string(&args, "--ruleset", "20260915");
    if ruleset_arg != "20260915" && ruleset_arg != RULESET_ID {
        return Err(format!(
            "production V4 self-play currently supports only 20260915/{RULESET_ID}, got {ruleset_arg:?}"
        )
        .into());
    }

    let games = numeric(&args, "--games", 32usize)?;
    let game_index_start = numeric(&args, "--game-index-start", 0usize)?;
    let base_seed = numeric(&args, "--base-seed", 2026091501u64)?;
    let simulations = numeric(&args, "--simulations", 200usize)?;
    let chance_outcome_samples = numeric(&args, "--chance-outcome-samples", 8usize)?;
    let temperature = numeric(&args, "--temperature", 1.0f32)?;
    let temperature_decisions = numeric(&args, "--temperature-decisions", 30usize)?;
    let max_decisions = numeric(&args, "--max-decisions", 512usize)?;
    let workers = numeric(&args, "--workers", 1usize)?;
    let intra_threads = numeric(&args, "--intra-threads", 1usize)?;
    let execution_provider = parse_execution_provider(&args)?;
    let allow_dirty_smoke = flag(&args, "--allow-dirty-smoke");

    if games == 0 {
        return Err("--games must be positive".into());
    }
    if simulations == 0 {
        return Err("--simulations must be positive".into());
    }
    if !(1..=64).contains(&chance_outcome_samples) {
        return Err("--chance-outcome-samples must be in 1..=64".into());
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("--temperature must be finite and non-negative".into());
    }
    if max_decisions == 0 {
        return Err("--max-decisions must be positive".into());
    }
    if workers == 0 {
        return Err("--workers must be positive".into());
    }
    if intra_threads == 0 {
        return Err("--intra-threads must be positive".into());
    }
    game_index_start
        .checked_add(games)
        .ok_or("--game-index-start + --games overflowed")?;

    let repo_root = git_path(["rev-parse", "--show-toplevel"])
        .ok_or("self_play_v4 must run inside the Augmentia Git worktree")?;
    let engine_commit =
        git_output_in(&repo_root, ["rev-parse", "HEAD"]).ok_or("failed to read Git HEAD")?;
    let dirty_worktree = git_output_in(&repo_root, ["status", "--porcelain"])
        .is_some_and(|output| !output.is_empty());
    let sealed_commit = git_output_in(&repo_root, ["rev-list", "-n", "1", SOURCE_SEAL_TAG])
        .ok_or("source seal tag ai/v4-training-base-20260919 is missing")?;
    if sealed_commit != SOURCE_SEAL_COMMIT {
        return Err(format!(
            "source seal tag {SOURCE_SEAL_TAG} resolves to {sealed_commit}, expected {SOURCE_SEAL_COMMIT}"
        )
        .into());
    }
    if dirty_worktree && !allow_dirty_smoke {
        return Err(
            "refusing production self-play from a dirty worktree; commit/stash changes or use --allow-dirty-smoke for non-promotable smoke artifacts"
                .into(),
        );
    }
    if allow_dirty_smoke && !run_id.to_ascii_lowercase().contains("smoke") {
        return Err("--allow-dirty-smoke requires --run-id to contain 'smoke'".into());
    }

    refuse_overwrite(&output_dir)?;
    std::fs::create_dir_all(output_dir.join("games"))?;

    let ruleset_path = format!("rulesets/{RULESET_ID}/manifest.json");
    let sealed_ruleset = git_blob(&repo_root, SOURCE_SEAL_COMMIT, &ruleset_path)?;
    let head_ruleset = git_blob(&repo_root, "HEAD", &ruleset_path)?;
    if head_ruleset != sealed_ruleset {
        return Err("HEAD ruleset manifest content differs from source seal".into());
    }
    let ruleset_hash = sealed_ruleset_manifest_sha256(&repo_root)?;
    let engine = Arc::new(Engine::bundled_20260915()?);
    if engine.ruleset().id != RULESET_ID {
        return Err(format!(
            "bundled engine ruleset mismatch: expected {RULESET_ID}, got {}",
            engine.ruleset().id
        )
        .into());
    }
    let ruleset = RulesetRefV3 {
        id: RULESET_ID.to_owned(),
        manifest_sha256: ruleset_hash.clone(),
    };

    let evaluator_kind = optional_string(&args, "--evaluator", "onnx");
    let (evaluator, evaluator_provenance, model_artifact): (
        Arc<dyn FactorizedEvaluatorV4 + Send + Sync>,
        EvaluatorProvenanceV4,
        Option<ModelArtifactV3>,
    ) = match evaluator_kind.as_str() {
        "bootstrap" => {
            let seed = numeric(&args, "--bootstrap-seed", 0x5634_424f_4f54_0001u64)?;
            let value = numeric(&args, "--bootstrap-value", 0.0f32)?;
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err("--bootstrap-value must be finite and in [-1, 1]".into());
            }
            let agent_id = format!("bootstrap-deterministic-v4:{seed:016x}");
            (
                Arc::new(DeterministicFactorizedEvaluatorV4 { seed, value }),
                EvaluatorProvenanceV4 {
                    kind: "bootstrap".to_owned(),
                    agent_id,
                    model_sha256: None,
                    checkpoint_sha256: None,
                    bundle_metadata_path: None,
                    bundle_metadata_sha256: None,
                    state_encoder_sha256: None,
                    factorized_scorer_sha256: None,
                    bundle_code_commit: None,
                    bootstrap_seed: Some(seed),
                    bootstrap_value: Some(value),
                },
                None,
            )
        }
        "onnx" => {
            let bundle = required_path(&args, "--bundle")?;
            let bundle_info = read_bundle_provenance(&bundle)?;
            let runtime = V4OnnxRuntime::from_bundle_with_options(
                &bundle,
                RULESET_ID,
                &ruleset_hash,
                V3RuntimeOptions {
                    execution_provider,
                    intra_threads,
                },
            )?;
            let agent_id = format!(
                "v4-bundle:{}",
                bundle
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("metadata")
            );
            let model = ModelArtifactV3 {
                model_id: agent_id.clone(),
                sha256: bundle_info.bundle_metadata_sha256.clone(),
            };
            (
                Arc::new(runtime),
                EvaluatorProvenanceV4 {
                    kind: "onnx".to_owned(),
                    agent_id,
                    model_sha256: Some(bundle_info.bundle_metadata_sha256.clone()),
                    checkpoint_sha256: bundle_info.checkpoint_sha256,
                    bundle_metadata_path: Some(bundle.display().to_string()),
                    bundle_metadata_sha256: Some(bundle_info.bundle_metadata_sha256),
                    state_encoder_sha256: Some(bundle_info.state_encoder_sha256),
                    factorized_scorer_sha256: Some(bundle_info.factorized_scorer_sha256),
                    bundle_code_commit: bundle_info.code_commit,
                    bootstrap_seed: None,
                    bootstrap_value: None,
                },
                Some(model),
            )
        }
        other => {
            return Err(format!("--evaluator must be 'onnx' or 'bootstrap', got {other:?}").into());
        }
    };

    let run_kind = if dirty_worktree {
        "smoke"
    } else {
        "production"
    };
    let generator = GeneratorProvenanceV3 {
        run_id: run_id.clone(),
        engine_commit: engine_commit.clone(),
        engine_api_version: 1,
        collector_id: format!("augment-cli/self_play_v4:{run_kind}"),
        dirty_worktree,
    };
    let search_config_id = format!(
        "mcts-v4:sims={simulations}:chance={chance_outcome_samples}:root={ROOT_SELECTION_V4_ID}:search={SEARCH_SEMANTICS_V4_ID}"
    );
    let mode = Mode::Standard;
    let game_started = Instant::now();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("v4-selfplay-{index}"))
        .build()?;

    let artifacts = pool.install(|| {
        (game_index_start..game_index_start + games)
            .into_par_iter()
            .map(|game_index| {
                generate_and_write_game(
                    game_index,
                    &run_id,
                    &output_dir,
                    &engine,
                    &mode,
                    &ruleset,
                    &generator,
                    &evaluator,
                    model_artifact.as_ref(),
                    &evaluator_provenance.agent_id,
                    &search_config_id,
                    base_seed,
                    simulations,
                    chance_outcome_samples,
                    temperature,
                    temperature_decisions,
                    max_decisions,
                )
            })
            .collect::<Result<Vec<_>, String>>()
    })?;

    let mut artifacts = artifacts;
    artifacts.sort_by_key(|artifact| artifact.game_index);

    let manifest = RunManifestV4 {
        manifest_schema_id: RUN_MANIFEST_SCHEMA_ID.to_owned(),
        run_id: run_id.clone(),
        run_kind: run_kind.to_owned(),
        source_seal_tag: SOURCE_SEAL_TAG.to_owned(),
        source_seal_commit: SOURCE_SEAL_COMMIT.to_owned(),
        engine_commit,
        dirty_worktree,
        ruleset,
        evaluator: evaluator_provenance,
        mode_id: "standard".to_owned(),
        game_index_start,
        games,
        base_seed,
        seed_derivation: "base_seed + game_index (wrapping u64)".to_owned(),
        simulations,
        chance_outcome_samples,
        temperature,
        temperature_decisions,
        max_decisions,
        workers,
        execution_provider: execution_provider.as_str().to_owned(),
        intra_threads,
        search_semantics_id: SEARCH_SEMANTICS_V4_ID.to_owned(),
        root_selection_id: ROOT_SELECTION_V4_ID.to_owned(),
        chance_sampling_id: CHANCE_SAMPLING_V4_ID.to_owned(),
        replay_schema_id: GAME_RECORD_V4_SCHEMA_ID.to_owned(),
        factorized_policy_schema_id: FACTORIZED_POLICY_V4_SCHEMA_ID.to_owned(),
        factorized_target_schema_id: FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID.to_owned(),
        observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
        action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
        games_artifacts: artifacts,
    };
    let manifest_path = output_dir.join("manifest.json");
    atomic_json(&manifest_path, &manifest)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "manifest": manifest_path,
            "run_id": run_id,
            "run_kind": run_kind,
            "ruleset_id": RULESET_ID,
            "ruleset_manifest_sha256": ruleset_hash,
            "games": games,
            "game_index_start": game_index_start,
            "simulations": simulations,
            "chance_outcome_samples": chance_outcome_samples,
            "workers": workers,
            "execution_provider": execution_provider.as_str(),
            "elapsed_seconds": game_started.elapsed().as_secs_f64(),
        }))?
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_and_write_game(
    game_index: usize,
    run_id: &str,
    output_dir: &Path,
    engine: &Engine,
    mode: &Mode,
    ruleset: &RulesetRefV3,
    generator: &GeneratorProvenanceV3,
    evaluator: &Arc<dyn FactorizedEvaluatorV4 + Send + Sync>,
    model_artifact: Option<&ModelArtifactV3>,
    agent_id: &str,
    search_config_id: &str,
    base_seed: u64,
    simulations: usize,
    chance_outcome_samples: usize,
    temperature: f32,
    temperature_decisions: usize,
    max_decisions: usize,
) -> Result<GameArtifactV4, String> {
    let seed = base_seed.wrapping_add(game_index as u64);
    let game_id = format!("{run_id}-game-{game_index:08}");
    let white = SelfPlayAgentV4 {
        evaluator: evaluator.as_ref(),
        provenance: player_agent(
            AbsolutePlayerV3::White,
            agent_id,
            model_artifact,
            search_config_id,
        ),
    };
    let black = SelfPlayAgentV4 {
        evaluator: evaluator.as_ref(),
        provenance: player_agent(
            AbsolutePlayerV3::Black,
            agent_id,
            model_artifact,
            search_config_id,
        ),
    };
    let started = Instant::now();
    let game = generate_self_play_game_v4(
        engine,
        mode.clone(),
        game_id.clone(),
        ruleset.clone(),
        generator.clone(),
        [&white, &black],
        SelfPlayGameConfigV4 {
            seed,
            max_decisions,
            simulations,
            chance_outcome_samples,
            temperature,
            temperature_decisions,
        },
    )
    .map_err(|error| error.to_string())?;
    game.validate().map_err(|error| error.to_string())?;

    let relative_path = format!("games/{game_id}.json");
    let path = output_dir.join(Path::new(&relative_path));
    let bytes = serialize_pretty_line(&game).map_err(|error| error.to_string())?;
    atomic_bytes(&path, &bytes).map_err(|error| error.to_string())?;
    let file_sha256 = sha256_bytes(&bytes);

    eprintln!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "event": "selfplay_v4_game_complete",
            "game_index": game_index,
            "game_id": game_id,
            "seed_hex": format!("{seed:016x}"),
            "decisions": game.steps.len(),
            "trajectory_sha256": game.trajectory_sha256,
            "file_sha256": file_sha256,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
        }))
        .map_err(|error| error.to_string())?
    );

    Ok(GameArtifactV4 {
        game_index,
        game_id,
        seed_hex: format!("{seed:016x}"),
        relative_path,
        file_sha256,
        trajectory_sha256: game.trajectory_sha256,
        decisions: game.steps.len(),
        termination: game.termination,
    })
}

fn player_agent(
    seat: AbsolutePlayerV3,
    agent_id: &str,
    model: Option<&ModelArtifactV3>,
    config_id: &str,
) -> PlayerAgentV3 {
    PlayerAgentV3 {
        seat,
        agent_kind: AgentKindV3::Mcts,
        agent_id: agent_id.to_owned(),
        model: model.cloned(),
        config_id: Some(config_id.to_owned()),
    }
}

struct BundleProvenance {
    bundle_metadata_sha256: String,
    checkpoint_sha256: Option<String>,
    state_encoder_sha256: String,
    factorized_scorer_sha256: String,
    code_commit: Option<String>,
}

fn read_bundle_provenance(path: &Path) -> Result<BundleProvenance, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let payload: Value = serde_json::from_slice(&bytes)?;
    let bundle_ruleset = json_string(&payload, &["ruleset_id"])?;
    if bundle_ruleset != RULESET_ID {
        return Err(format!(
            "bundle ruleset {bundle_ruleset:?} does not match required {RULESET_ID}"
        )
        .into());
    }
    Ok(BundleProvenance {
        bundle_metadata_sha256: sha256_bytes(&bytes),
        checkpoint_sha256: payload
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        state_encoder_sha256: json_string(&payload, &["state_encoder", "sha256"])?,
        factorized_scorer_sha256: json_string(&payload, &["factorized_scorer", "sha256"])?,
        code_commit: payload
            .get("code_commit")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    })
}

fn json_string(payload: &Value, path: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let mut current = payload;
    for key in path {
        current = current
            .get(*key)
            .ok_or_else(|| format!("bundle metadata is missing {}", path.join(".")))?;
    }
    current
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("bundle metadata {} must be a string", path.join(".")).into())
}

fn refuse_overwrite(output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if output_dir.exists() {
        if !output_dir.is_dir() {
            return Err(format!("output path {} is not a directory", output_dir.display()).into());
        }
        let mut entries = std::fs::read_dir(output_dir)?;
        if entries.next().transpose()?.is_some() {
            return Err(format!(
                "refusing to overwrite non-empty output directory {}",
                output_dir.display()
            )
            .into());
        }
    }
    Ok(())
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

fn serialize_pretty_line(value: &impl Serialize) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = serialize_pretty_line(value)?;
    atomic_bytes(path, &bytes)?;
    Ok(())
}

fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    let temporary = path.with_extension(format!(
        "{}.partial-{}",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("tmp"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    if let Err(error) = (|| -> Result<(), std::io::Error> {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)?;
        Ok(())
    })() {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

fn git_blob(
    repo_root: &Path,
    revision: &str,
    relative_path: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let spec = format!("{revision}:{relative_path}");
    let output = Command::new("git")
        .current_dir(repo_root)
        .args(["show", &spec])
        .output()?;
    if !output.status.success() {
        return Err(format!("failed to read Git blob {spec}").into());
    }
    Ok(output.stdout)
}

fn sealed_ruleset_manifest_sha256(repo_root: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = git_blob(
        repo_root,
        SOURCE_SEAL_COMMIT,
        "artifacts/v4-training-base-20260919/seal.json",
    )?;
    let seal: Value = serde_json::from_slice(&bytes)?;
    if seal.get("tag").and_then(Value::as_str) != Some(SOURCE_SEAL_TAG)
        || seal.get("commit").and_then(Value::as_str) != Some(SOURCE_SEAL_COMMIT)
        || seal
            .get("ruleset")
            .and_then(|value| value.get("id"))
            .and_then(Value::as_str)
            != Some(RULESET_ID)
    {
        return Err("source seal identity is inconsistent".into());
    }
    let hash = seal
        .get("ruleset")
        .and_then(|value| value.get("manifest_sha256"))
        .and_then(Value::as_str)
        .ok_or("source seal ruleset manifest SHA-256 is missing")?;
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("source seal ruleset manifest SHA-256 is invalid".into());
    }
    Ok(hash.to_owned())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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

fn required(args: &[String], flag_name: &str) -> Result<String, String> {
    args.iter()
        .position(|item| item == flag_name)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .ok_or_else(|| format!("{flag_name} is required"))
}

fn required_path(args: &[String], flag_name: &str) -> Result<PathBuf, String> {
    required(args, flag_name).map(PathBuf::from)
}

fn optional_string(args: &[String], flag_name: &str, default: &str) -> String {
    args.windows(2)
        .find(|pair| pair[0] == flag_name)
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| default.to_owned())
}

fn numeric<T>(args: &[String], flag_name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    args.iter()
        .position(|item| item == flag_name)
        .and_then(|index| args.get(index + 1))
        .map_or(Ok(default), |value| {
            value
                .parse()
                .map_err(|error| format!("{flag_name}: {error}"))
        })
}

fn flag(args: &[String], flag_name: &str) -> bool {
    args.iter().any(|item| item == flag_name)
}

fn validate_run_id(run_id: &str) -> Result<(), String> {
    if run_id.is_empty()
        || run_id.len() > 96
        || !run_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        return Err(
            "--run-id must be 1..=96 ASCII alphanumeric characters plus '-', '_' or '.'".to_owned(),
        );
    }
    Ok(())
}

fn print_help() {
    println!(
        "self_play_v4 --run-id ID --output-dir DIR [options]\n\n\
         Required for ONNX evaluator:\n  --bundle PATH\n\n\
         Evaluator:\n  --evaluator onnx|bootstrap       default: onnx\n  --bootstrap-seed U64             explicit deterministic bootstrap seed\n  --bootstrap-value F32            default: 0.0\n  --execution-provider cpu|cuda    default: cpu\n  --intra-threads N                default: 1\n\n\
         Self-play:\n  --ruleset 20260915               only production ruleset currently allowed\n  --games N                        default: 32\n  --game-index-start N             default: 0\n  --base-seed U64                  default: 2026091501\n  --simulations N                  default: 200\n  --chance-outcome-samples N       default: 8\n  --temperature F32                default: 1.0\n  --temperature-decisions N        default: 30\n  --max-decisions N                default: 512\n  --workers N                      default: 1\n\n\
         Safety:\n  --allow-dirty-smoke              allow dirty worktree only for run IDs containing 'smoke'\n"
    );
}

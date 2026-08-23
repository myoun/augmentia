use augment_ai::data_v3::{
    AbsolutePlayerV3, AgentKindV3, CompressionV3, DATASET_MANIFEST_V3_SCHEMA_ID, DataRecordV3,
    DatasetManifestV3, DatasetSplitV3, DatasetTotalsV3, GeneratorProvenanceV3, ModelArtifactV3,
    PlayerAgentV3, RulesetRefV3, SPLIT_POLICY_V3_ALGORITHM, SplitPolicyV3, TerminationV3,
};
use augment_ai::{
    ACTION_V3_SCHEMA_ID, OBSERVATION_V3_SCHEMA_ID, SelfPlayAgentV3, SelfPlayGameConfigV3,
    V3OnnxRuntime, generate_self_play_game_v3, verify_manifest_replays, write_jsonl_zstd_shard,
};
use augment_core::{BUNDLED_RULESET_ID, Engine, Mode};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if let Err(error) = run() {
        eprintln!("V3 self-play failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let model_a = required_path(&args, "--model-a")?;
    let metadata_a = required_path(&args, "--metadata-a")?;
    let model_b = optional_path(&args, "--model-b").unwrap_or_else(|| model_a.clone());
    let metadata_b = optional_path(&args, "--metadata-b").unwrap_or_else(|| metadata_a.clone());
    let output_dir = required_path(&args, "--output-dir")?;
    let ruleset_hash = required(&args, "--ruleset-hash")?;
    let run_id = required(&args, "--run-id")?;
    let created_at_utc = required(&args, "--created-at-utc")?;
    let games = numeric(&args, "--games", 32usize)?;
    let base_seed = numeric(&args, "--base-seed", 2026082101u64)?;
    let simulations = numeric(&args, "--simulations", 400usize)?;
    let leaf_batch_size = numeric(&args, "--leaf-batch-size", 16usize)?;
    let max_decisions = numeric(&args, "--max-decisions", 512usize)?;
    let temperature_decisions = numeric(&args, "--temperature-decisions", 30usize)?;
    let sampling_temperature = numeric(&args, "--temperature", 1.0f32)?;
    let noise_alpha = numeric(&args, "--noise-alpha", 0.3f32)?;
    let noise_epsilon = numeric(&args, "--noise-epsilon", 0.25f32)?;
    let train_bp = numeric(&args, "--train-bp", 9000u16)?;
    let validation_bp = numeric(&args, "--validation-bp", 500u16)?;
    let holdout_bp = numeric(&args, "--holdout-bp", 500u16)?;
    let compression_level = numeric(&args, "--compression-level", 3i32)?;
    if games == 0 {
        return Err("--games must be positive".into());
    }
    if output_dir.join("manifest.json").exists() {
        return Err(format!(
            "refusing to overwrite {}",
            output_dir.join("manifest.json").display()
        )
        .into());
    }
    std::fs::create_dir_all(&output_dir)?;

    let split_policy = SplitPolicyV3 {
        algorithm: SPLIT_POLICY_V3_ALGORITHM.to_owned(),
        salt: format!("selfplay-v3:{run_id}"),
        train_basis_points: train_bp,
        validation_basis_points: validation_bp,
        holdout_basis_points: holdout_bp,
    };
    let engine = Engine::bundled_20260807()?;
    let runtime_a =
        V3OnnxRuntime::from_files(&model_a, &metadata_a, BUNDLED_RULESET_ID, &ruleset_hash)?;
    let runtime_b =
        V3OnnxRuntime::from_files(&model_b, &metadata_b, BUNDLED_RULESET_ID, &ruleset_hash)?;
    let artifact_a = model_artifact(&model_a, &metadata_a, "agent-a")?;
    let artifact_b = model_artifact(&model_b, &metadata_b, "agent-b")?;
    let engine_commit = git_output(["rev-parse", "HEAD"]).unwrap_or_else(|| "0".repeat(40));
    let dirty_worktree =
        git_output(["status", "--porcelain"]).is_some_and(|output| !output.is_empty());
    let generator = GeneratorProvenanceV3 {
        run_id: run_id.clone(),
        engine_commit,
        engine_api_version: 1,
        collector_id: "augment-cli/self_play_v3".to_owned(),
        dirty_worktree,
    };
    let ruleset = RulesetRefV3 {
        id: BUNDLED_RULESET_ID.to_owned(),
        manifest_sha256: ruleset_hash,
    };
    let search_config_id = format!(
        "mcts-v3:sims={simulations}:batch={leaf_batch_size}:alpha={noise_alpha}:epsilon={noise_epsilon}"
    );
    let mut split_records: BTreeMap<DatasetSplitV3, Vec<DataRecordV3>> = BTreeMap::new();
    for game_index in 0..games {
        let swap = game_index % 2 == 1;
        let (white_runtime, white_artifact, black_runtime, black_artifact) = if swap {
            (&runtime_b, &artifact_b, &runtime_a, &artifact_a)
        } else {
            (&runtime_a, &artifact_a, &runtime_b, &artifact_b)
        };
        let white = SelfPlayAgentV3 {
            evaluator: white_runtime,
            provenance: player_agent(AbsolutePlayerV3::White, white_artifact, &search_config_id),
        };
        let black = SelfPlayAgentV3 {
            evaluator: black_runtime,
            provenance: player_agent(AbsolutePlayerV3::Black, black_artifact, &search_config_id),
        };
        let game = generate_self_play_game_v3(
            &engine,
            Mode::Standard,
            format!("{run_id}-game-{game_index:08}"),
            ruleset.clone(),
            generator.clone(),
            [&white, &black],
            SelfPlayGameConfigV3 {
                seed: base_seed.wrapping_add(game_index as u64),
                max_decisions,
                simulations,
                leaf_batch_size,
                root_noise_alpha: noise_alpha,
                root_noise_epsilon: noise_epsilon,
                sampling_temperature,
                temperature_decisions,
            },
        )?;
        if let TerminationV3::Error {
            stage,
            code,
            detail,
            ..
        } = &game.termination
        {
            return Err(format!(
                "self-play game {game_index} stopped at {stage:?}/{code}: {}",
                detail.as_deref().unwrap_or("no error detail")
            )
            .into());
        }
        let split = split_policy.assign(&game.split_group_key())?;
        split_records
            .entry(split)
            .or_default()
            .push(DataRecordV3::Game(Box::new(game)));
    }

    let mut shards = Vec::new();
    for (split, records) in split_records {
        let split_name = match split {
            DatasetSplitV3::Train => "train",
            DatasetSplitV3::Validation => "validation",
            DatasetSplitV3::Holdout => "holdout",
        };
        let relative = format!("shards/{split_name}-00000.games.jsonl.zst");
        let summary = write_jsonl_zstd_shard(
            output_dir.join(Path::new(&relative)),
            &records,
            compression_level,
        )?;
        shards.push(summary.to_manifest_entry(relative, split));
    }
    let mut totals = DatasetTotalsV3 {
        shards: shards.len() as u64,
        ..DatasetTotalsV3::default()
    };
    for shard in &shards {
        totals.records += shard.records;
        totals.games += shard.games;
        totals.decisions += shard.decisions;
        totals.game_steps += shard.game_steps;
        totals.policy_eligible_decisions += shard.policy_eligible_decisions;
        totals.value_eligible_decisions += shard.value_eligible_decisions;
    }
    let manifest = DatasetManifestV3 {
        manifest_schema_id: DATASET_MANIFEST_V3_SCHEMA_ID.to_owned(),
        dataset_id: format!("selfplay-v3:{run_id}"),
        created_at_utc,
        observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
        action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
        split_policy,
        compression: CompressionV3 {
            codec: "zstd".to_owned(),
            level: compression_level,
        },
        shards,
        totals,
    };
    let manifest_path = output_dir.join("manifest.json");
    atomic_json(&manifest_path, &manifest)?;
    verify_manifest_replays(&output_dir, &manifest, &engine)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "manifest": manifest_path,
            "dataset_id": manifest.dataset_id,
            "games": manifest.totals.games,
            "game_steps": manifest.totals.game_steps,
            "policy_eligible_decisions": manifest.totals.policy_eligible_decisions,
            "value_eligible_decisions": manifest.totals.value_eligible_decisions,
            "model_a": artifact_a,
            "model_b": artifact_b,
            "runtime_runs_a": runtime_a.session_runs(),
            "runtime_runs_b": runtime_b.session_runs(),
        }))?
    );
    Ok(())
}

fn player_agent(
    seat: AbsolutePlayerV3,
    artifact: &ModelArtifactV3,
    config_id: &str,
) -> PlayerAgentV3 {
    PlayerAgentV3 {
        seat,
        agent_kind: AgentKindV3::Mcts,
        agent_id: artifact.model_id.clone(),
        model: Some(artifact.clone()),
        config_id: Some(config_id.to_owned()),
    }
}

fn model_artifact(
    model: &Path,
    metadata: &Path,
    fallback_id: &str,
) -> Result<ModelArtifactV3, Box<dyn std::error::Error>> {
    let payload: serde_json::Value = serde_json::from_slice(&std::fs::read(metadata)?)?;
    let sha256 = sha256_file(model)?;
    if payload.get("onnx_sha256").and_then(|value| value.as_str()) != Some(sha256.as_str()) {
        return Err(format!("metadata hash does not match {}", model.display()).into());
    }
    let model_id = payload
        .get("model_id")
        .and_then(|value| value.as_str())
        .or_else(|| model.file_stem().and_then(|value| value.to_str()))
        .unwrap_or(fallback_id)
        .to_owned();
    Ok(ModelArtifactV3 { model_id, sha256 })
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    let temporary = path.with_extension(format!("json.partial-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary, path)?;
    Ok(())
}

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
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

use augment_ai::data_v3::{
    AbsolutePlayerV3, AuxiliaryTargetV3, AuxiliaryTargetValueV3, CompressionV3,
    DATASET_MANIFEST_V3_SCHEMA_ID, DECISION_RECORD_V3_SCHEMA_ID, DataRecordV3, DatasetManifestV3,
    DatasetSplitV3, DatasetTotalsV3, DecisionPolicySupervisionV3, DecisionPolicyTargetV3,
    DecisionProvenanceV3, DecisionRecordV3, DecisionSourceKindV3, DecisionTargetsV3,
    DecisionValueTargetV3, GeneratorProvenanceV3, PolicyTrainingScopeV3, RulesetRefV3,
    SPLIT_POLICY_V3_ALGORITHM, SourceDomainV3, SplitPolicyV3, fingerprint_hex,
};
use augment_ai::{
    ACTION_V3_SCHEMA_ID, ForcedWinConfigV3, ForcedWinErrorV3, LegalActionSetV3,
    OBSERVATION_V3_SCHEMA_ID, encode_observation_v3, solve_forced_win_v3, validate_manifest,
    verify_manifest_replays, write_jsonl_zstd_shard,
};
use augment_core::{BUNDLED_RULESET_ID, Engine, GameStatus, Mode};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Debug)]
struct CandidatePosition {
    game_id: String,
    seed_hex: String,
    mode_id: String,
    decision_index: u32,
    played_action_prefix: Vec<u32>,
    expected_state_fingerprint: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("V3 solver M1 failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let selfplay_manifest = required_path(&args, "--selfplay-manifest")?;
    let output_dir = required_path(&args, "--output-dir")?;
    let ruleset_hash = required(&args, "--ruleset-hash")?;
    let run_id = required(&args, "--run-id")?;
    let created_at_utc = required(&args, "--created-at-utc")?;
    let max_depth = numeric(&args, "--max-depth", 1u8)?;
    let max_nodes_per_position = numeric(&args, "--max-nodes-per-position", 2_000_000u64)?;
    let compression_level = numeric(&args, "--compression-level", 3i32)?;
    if output_dir.join("manifest.json").exists() {
        return Err(format!(
            "refusing to overwrite existing dataset {}",
            output_dir.display()
        )
        .into());
    }
    std::fs::create_dir_all(output_dir.join("shards"))?;

    let engine_commit = git_output(["rev-parse", "HEAD"]).unwrap_or_else(|| "0".repeat(40));
    let dirty_worktree =
        git_output(["status", "--porcelain"]).is_some_and(|output| !output.is_empty());
    let _generator = GeneratorProvenanceV3 {
        run_id: run_id.clone(),
        engine_commit,
        engine_api_version: 1,
        collector_id: "augment-cli/solve_v3_tactics_m1".to_owned(),
        dirty_worktree,
    };
    let _ruleset = RulesetRefV3 {
        id: BUNDLED_RULESET_ID.to_owned(),
        manifest_sha256: ruleset_hash.clone(),
    };
    let split_policy = SplitPolicyV3 {
        algorithm: SPLIT_POLICY_V3_ALGORITHM.to_owned(),
        salt: format!("solver-v3:{run_id}"),
        train_basis_points: 9_000,
        validation_basis_points: 500,
        holdout_basis_points: 500,
    };

    let engine = Engine::bundled_20260807()?;
    let manifest_bytes = std::fs::read(&selfplay_manifest)?;
    let dataset_manifest: DatasetManifestV3 = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest(&dataset_manifest)?;
    let selfplay_root = selfplay_manifest.parent().unwrap_or_else(|| Path::new("."));
    verify_manifest_replays(selfplay_root, &dataset_manifest, &engine)?;
    let mut candidates: Vec<CandidatePosition> = Vec::new();
    for shard in &dataset_manifest.shards {
        let records = read_records(&selfplay_manifest, &shard.path)?;
        for record in records {
            if let DataRecordV3::Game(game) = record {
                let mut played_action_prefix = Vec::with_capacity(game.steps.len());
                for step in &game.steps {
                    candidates.push(CandidatePosition {
                        game_id: game.game_id.clone(),
                        seed_hex: game.seed_hex.clone(),
                        mode_id: game.mode_id.clone(),
                        decision_index: step.decision_index,
                        played_action_prefix: played_action_prefix.clone(),
                        expected_state_fingerprint: step.state_fingerprint.clone(),
                    });
                    played_action_prefix.push(step.played_action_index);
                }
            }
        }
    }
    eprintln!(
        "{}",
        serde_json::json!({"event": "candidates_loaded", "count": candidates.len()})
    );

    let mut proven_records: Vec<DataRecordV3> = Vec::new();
    let mut seen_positions: BTreeSet<String> = BTreeSet::new();
    let solved = AtomicU64::new(0);
    let refuted = AtomicU64::new(0);
    let started = Instant::now();

    let results: Vec<Result<Option<DecisionRecordV3>, ForcedWinErrorV3>> = candidates
        .par_iter()
        .map_init(
            || Engine::bundled_20260807().expect("engine init"),
            |engine, candidate| {
                solve_candidate(
                    engine,
                    candidate,
                    &ruleset_hash,
                    max_depth,
                    max_nodes_per_position,
                )
            },
        )
        .collect();

    let refuted_log = output_dir.join("refuted.jsonl");
    let mut budget_exceeded = 0u64;
    for (candidate, result) in candidates.iter().zip(results) {
        match result {
            Ok(Some(record)) => {
                if seen_positions.insert(record.state_fingerprint.clone()) {
                    solved.fetch_add(1, Ordering::Relaxed);
                    proven_records.push(DataRecordV3::Decision(Box::new(record)));
                } else {
                    refuted.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(ForcedWinErrorV3::NodeBudgetExceeded { .. }) => {
                budget_exceeded += 1;
                append_refuted(
                    &refuted_log,
                    &candidate.game_id,
                    candidate.decision_index,
                    "budget",
                )?;
            }
            Ok(None) => {
                refuted.fetch_add(1, Ordering::Relaxed);
                append_refuted(
                    &refuted_log,
                    &candidate.game_id,
                    candidate.decision_index,
                    "",
                )?;
            }
            Err(error) => return Err(error.into()),
        }
    }

    if proven_records.is_empty() {
        return Err("no forced-win positions were proven; nothing to write".into());
    }
    let mut split_records: BTreeMap<DatasetSplitV3, Vec<DataRecordV3>> = BTreeMap::new();
    for record in proven_records {
        let split = split_policy.assign(&record.split_group_key())?;
        split_records.entry(split).or_default().push(record);
    }
    let mut shards = Vec::new();
    for (split, records) in &split_records {
        let split_name = match split {
            DatasetSplitV3::Train => "train",
            DatasetSplitV3::Validation => "validation",
            DatasetSplitV3::Holdout => "holdout",
        };
        let relative = format!("shards/{split_name}/decision-00000.jsonl.zst");
        let summary = write_jsonl_zstd_shard(
            output_dir.join(Path::new(&relative)),
            records,
            compression_level,
        )?;
        shards.push(summary.to_manifest_entry(relative, split.clone()));
    }
    let mut totals = DatasetTotalsV3 {
        shards: shards.len() as u64,
        ..DatasetTotalsV3::default()
    };
    for shard in &shards {
        totals.records += shard.records;
        totals.decisions += shard.decisions;
        totals.policy_eligible_decisions += shard.policy_eligible_decisions;
        totals.value_eligible_decisions += shard.value_eligible_decisions;
    }
    let manifest = DatasetManifestV3 {
        manifest_schema_id: DATASET_MANIFEST_V3_SCHEMA_ID.to_owned(),
        dataset_id: format!("solver-v3:{run_id}"),
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
    validate_manifest(&manifest)?;
    atomic_json(&output_dir.join("manifest.json"), &manifest)?;
    verify_manifest_replays(&output_dir, &manifest, &engine)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "manifest": output_dir.join("manifest.json"),
            "dataset_id": manifest.dataset_id,
            "candidates": candidates.len(),
            "proven": solved,
            "refuted": refuted,
            "budget_exceeded": budget_exceeded,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
        }))?
    );
    Ok(())
}

fn read_records(
    manifest_path: &Path,
    relative: &str,
) -> Result<Vec<DataRecordV3>, Box<dyn std::error::Error>> {
    use augment_ai::data_v3::read_jsonl_zstd_shard;
    let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let path = base.join(relative);
    Ok(read_jsonl_zstd_shard(path)?)
}

fn replay_to_decision(
    engine: &Engine,
    candidate: &CandidatePosition,
) -> Result<Option<augment_core::GameState>, Box<dyn std::error::Error>> {
    let seed = u64::from_str_radix(&candidate.seed_hex, 16)?;
    let mode = match candidate.mode_id.as_str() {
        "standard" => Mode::Standard,
        "chaos" => Mode::Chaos,
        "grand" => Mode::Grand,
        other => Mode::Campaign(other.to_owned()),
    };
    let mut state = engine.new_game(mode, seed)?;
    if candidate.played_action_prefix.len() != candidate.decision_index as usize {
        return Err(format!(
            "candidate {}#{} has a non-contiguous action prefix",
            candidate.game_id, candidate.decision_index
        )
        .into());
    }
    for (prefix_index, played_action_index) in
        candidate.played_action_prefix.iter().copied().enumerate()
    {
        let legal = LegalActionSetV3::from_state(&state, engine)?;
        if legal.is_empty() {
            return Err(format!(
                "candidate {}#{} reached no legal actions at prefix {}",
                candidate.game_id, candidate.decision_index, prefix_index
            )
            .into());
        }
        let transition = legal.apply(played_action_index as usize, &state, engine)?;
        state = transition.state;
        if state.status != GameStatus::Ongoing {
            return Err(format!(
                "candidate {}#{} crossed a terminal state at prefix {}",
                candidate.game_id, candidate.decision_index, prefix_index
            )
            .into());
        }
    }
    let actual = fingerprint_hex(state.fingerprint());
    if actual != candidate.expected_state_fingerprint {
        return Err(format!(
            "candidate {}#{} fingerprint mismatch: expected {}, got {}",
            candidate.game_id,
            candidate.decision_index,
            candidate.expected_state_fingerprint,
            actual
        )
        .into());
    }
    Ok(Some(state))
}

fn solve_candidate(
    engine: &Engine,
    candidate: &CandidatePosition,
    ruleset_hash: &str,
    max_depth: u8,
    max_nodes: u64,
) -> Result<Option<DecisionRecordV3>, ForcedWinErrorV3> {
    let state = match replay_to_decision(engine, candidate) {
        Ok(Some(state)) => state,
        Ok(None) => return Ok(None),
        Err(error) => return Err(ForcedWinErrorV3::Engine(error.to_string())),
    };
    if state.status != GameStatus::Ongoing {
        return Ok(None);
    }
    let solution = solve_forced_win_v3(
        &state,
        engine,
        ForcedWinConfigV3 {
            max_depth,
            max_nodes,
        },
    )?;
    let Some(solution) = solution else {
        return Ok(None);
    };
    let proof_source = format!(
        "solver-m1/depth={}/nodes={}",
        max_depth, solution.searched_nodes
    );
    let legal_actions =
        LegalActionSetV3::from_state(&state, engine).map_err(ForcedWinErrorV3::Action)?;
    let position_key = fingerprint_hex(state.fingerprint());
    let record = DecisionRecordV3 {
        record_schema_id: DECISION_RECORD_V3_SCHEMA_ID.to_owned(),
        decision_id: format!("solver-{}", sha256_bytes(position_key.as_bytes())),
        provenance: DecisionProvenanceV3 {
            source_kind: DecisionSourceKindV3::EngineSearch,
            source_domain: SourceDomainV3::AugmentChess,
            source_id: format!("{}#{}", candidate.game_id, candidate.decision_index),
            split_group_id: format!(
                "game\0{}\0{}\0{}",
                BUNDLED_RULESET_ID, candidate.mode_id, candidate.seed_hex
            ),
            generator_commit: None,
            license: None,
        },
        ruleset: RulesetRefV3 {
            id: BUNDLED_RULESET_ID.to_owned(),
            manifest_sha256: ruleset_hash.to_owned(),
        },
        observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
        action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
        actor: absolute_player(state.turn),
        state_fingerprint: position_key,
        observation: encode_observation_v3(&state),
        legal_actions,
        targets: DecisionTargetsV3 {
            policy: Some(DecisionPolicySupervisionV3 {
                scope: PolicyTrainingScopeV3::AugmentPolicy,
                target: DecisionPolicyTargetV3::OptimalActionSet {
                    action_indices: solution.optimal_action_indices,
                    proof_source: proof_source.clone(),
                },
            }),
            value: Some(DecisionValueTargetV3::Proven {
                value: 1.0,
                proof_source,
            }),
            auxiliary: vec![AuxiliaryTargetV3 {
                name: "forced_win_distance_decisions".to_owned(),
                value: AuxiliaryTargetValueV3::Integer(i64::from(solution.distance_decisions)),
            }],
        },
    };
    record
        .validate()
        .map_err(|error| ForcedWinErrorV3::Engine(error.to_string()))?;
    Ok(Some(record))
}

fn append_refuted(
    log: &Path,
    game_id: &str,
    decision_index: u32,
    fingerprint: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)?;
    writeln!(file, "{game_id}\t{decision_index}\t{fingerprint}")?;
    Ok(())
}

fn absolute_player(player: augment_core::Player) -> AbsolutePlayerV3 {
    match player {
        augment_core::Player::White => AbsolutePlayerV3::White,
        augment_core::Player::Black => AbsolutePlayerV3::Black,
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("json.partial-{}", std::process::id()));
    {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    std::fs::rename(&temporary, path)?;
    Ok(())
}

use std::io::Write as _;

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = std::process::Command::new("git").args(args).output().ok()?;
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

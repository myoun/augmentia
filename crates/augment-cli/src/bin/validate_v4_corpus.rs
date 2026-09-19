use augment_ai::data_v3::{AbsolutePlayerV3, RulesetRefV3, TerminationV3};
use augment_ai::{
    ACTION_V3_SCHEMA_ID, FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID, FACTORIZED_POLICY_V4_SCHEMA_ID,
    GAME_RECORD_V4_SCHEMA_ID, GameRecordV4, LegalActionSetV3, OBSERVATION_V3_SCHEMA_ID,
    encode_observation_v3,
};
use augment_core::{Engine, GameStatus, Mode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

const RULESET_ID: &str = "augment_chess_20260915";
const SOURCE_SEAL_TAG: &str = "ai/v4-training-base-20260919";
const SOURCE_SEAL_COMMIT: &str = "f4e64c35f5adcb7f361ed845fb55eb4d123b370b";
const RUN_MANIFEST_SCHEMA_ID: &str = "augmentia.self-play-v4-run.v1";
const VALIDATION_SCHEMA_ID: &str = "augmentia.self-play-v4-validation.v1";

#[derive(Debug, Deserialize)]
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
    simulations: usize,
    chance_outcome_samples: usize,
    temperature: f32,
    temperature_decisions: usize,
    max_decisions: usize,
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

#[derive(Debug, Deserialize)]
struct EvaluatorProvenanceV4 {
    kind: String,
    agent_id: String,
    model_sha256: Option<String>,
    bundle_metadata_sha256: Option<String>,
    state_encoder_sha256: Option<String>,
    factorized_scorer_sha256: Option<String>,
    bootstrap_seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Serialize)]
struct ValidationReportV4 {
    validation_schema_id: String,
    run_id: String,
    run_kind: String,
    ruleset_id: String,
    ruleset_manifest_sha256: String,
    games: usize,
    decisions: usize,
    terminal_games: usize,
    truncated_games: usize,
    unique_seeds: usize,
    evaluator_kind: String,
    evaluator_id: String,
    model_sha256: Option<String>,
    simulations_per_decision: usize,
    chance_outcome_samples: usize,
    checks: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("V4 corpus validation failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let manifest_path = required_path(&args, "--manifest")?;
    let report_path = optional_path(&args, "--report");
    let base_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));

    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest: RunManifestV4 = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest_identity(&manifest)?;

    let ruleset_path = find_repo_root(&manifest_path)?
        .join("rulesets")
        .join(RULESET_ID)
        .join("manifest.json");
    let actual_ruleset_hash = sha256_file(&ruleset_path)?;
    if actual_ruleset_hash != manifest.ruleset.manifest_sha256 {
        return Err(format!(
            "ruleset manifest SHA-256 mismatch: corpus={} current={actual_ruleset_hash}",
            manifest.ruleset.manifest_sha256
        )
        .into());
    }

    let engine = Engine::bundled_20260915()?;
    let mut seeds = BTreeSet::new();
    let mut game_ids = BTreeSet::new();
    let mut total_decisions = 0usize;
    let mut terminal_games = 0usize;
    let mut truncated_games = 0usize;

    if manifest.games_artifacts.len() != manifest.games {
        return Err("manifest game artifact count does not match games".into());
    }

    for (offset, artifact) in manifest.games_artifacts.iter().enumerate() {
        let expected_index = manifest.game_index_start + offset;
        if artifact.game_index != expected_index {
            return Err(format!(
                "game artifact index {} is not contiguous; expected {expected_index}",
                artifact.game_index
            )
            .into());
        }
        validate_relative_path(&artifact.relative_path)?;
        if !game_ids.insert(artifact.game_id.clone()) {
            return Err(format!("duplicate game id {}", artifact.game_id).into());
        }
        if !seeds.insert(artifact.seed_hex.clone()) {
            return Err(format!("duplicate game seed {}", artifact.seed_hex).into());
        }
        let expected_seed = manifest.base_seed.wrapping_add(artifact.game_index as u64);
        if artifact.seed_hex != format!("{expected_seed:016x}") {
            return Err(format!("game {} seed derivation mismatch", artifact.game_id).into());
        }

        let game_path = base_dir.join(Path::new(&artifact.relative_path));
        let bytes = std::fs::read(&game_path)?;
        let file_hash = sha256_bytes(&bytes);
        if file_hash != artifact.file_sha256 {
            return Err(format!("game {} file SHA-256 mismatch", artifact.game_id).into());
        }
        let game: GameRecordV4 = serde_json::from_slice(&bytes)?;
        game.validate()?;
        validate_game_identity(&manifest, artifact, &game)?;
        validate_replay(&game, &engine)?;

        total_decisions += game.steps.len();
        match game.termination {
            TerminationV3::EngineTerminal { .. } => terminal_games += 1,
            TerminationV3::Limit { .. } => truncated_games += 1,
            TerminationV3::Aborted { .. } | TerminationV3::Error { .. } => {
                return Err(format!("game {} ended abnormally", game.game_id).into());
            }
        }
    }

    let report = ValidationReportV4 {
        validation_schema_id: VALIDATION_SCHEMA_ID.to_owned(),
        run_id: manifest.run_id,
        run_kind: manifest.run_kind,
        ruleset_id: manifest.ruleset.id,
        ruleset_manifest_sha256: manifest.ruleset.manifest_sha256,
        games: manifest.games,
        decisions: total_decisions,
        terminal_games,
        truncated_games,
        unique_seeds: seeds.len(),
        evaluator_kind: manifest.evaluator.kind,
        evaluator_id: manifest.evaluator.agent_id,
        model_sha256: manifest.evaluator.model_sha256,
        simulations_per_decision: manifest.simulations,
        chance_outcome_samples: manifest.chance_outcome_samples,
        checks: vec![
            "manifest_identity".to_owned(),
            "ruleset_manifest_sha256".to_owned(),
            "game_file_sha256".to_owned(),
            "game_seed_uniqueness_and_derivation".to_owned(),
            "game_schema_and_fingerprint_chain".to_owned(),
            "decision_index_contiguity".to_owned(),
            "visit_conservation".to_owned(),
            "factorized_prefix_recalculation".to_owned(),
            "factorized_trie_sha256".to_owned(),
            "played_action_positive_visit".to_owned(),
            "engine_observation_replay".to_owned(),
            "engine_legal_action_replay".to_owned(),
            "engine_transition_replay".to_owned(),
            "terminal_result_actor_perspective".to_owned(),
            "ruleset_and_model_provenance".to_owned(),
        ],
    };

    let pretty = serde_json::to_string_pretty(&report)?;
    if let Some(path) = report_path {
        if path.exists() {
            return Err(
                format!("refusing to overwrite validation report {}", path.display()).into(),
            );
        }
        std::fs::write(&path, format!("{pretty}\n"))?;
    }
    println!("{pretty}");
    Ok(())
}

fn validate_manifest_identity(manifest: &RunManifestV4) -> Result<(), Box<dyn std::error::Error>> {
    if manifest.manifest_schema_id != RUN_MANIFEST_SCHEMA_ID
        || manifest.source_seal_tag != SOURCE_SEAL_TAG
        || manifest.source_seal_commit != SOURCE_SEAL_COMMIT
        || manifest.ruleset.id != RULESET_ID
        || manifest.replay_schema_id != GAME_RECORD_V4_SCHEMA_ID
        || manifest.factorized_policy_schema_id != FACTORIZED_POLICY_V4_SCHEMA_ID
        || manifest.factorized_target_schema_id != FACTORIZED_MCTS_TARGET_V4_SCHEMA_ID
        || manifest.observation_schema_id != OBSERVATION_V3_SCHEMA_ID
        || manifest.action_schema_id != ACTION_V3_SCHEMA_ID
    {
        return Err("V4 corpus manifest identity mismatch".into());
    }
    if manifest.games == 0
        || manifest.simulations == 0
        || !(1..=64).contains(&manifest.chance_outcome_samples)
        || !manifest.temperature.is_finite()
        || manifest.temperature < 0.0
        || manifest.max_decisions == 0
    {
        return Err("V4 corpus manifest search configuration is invalid".into());
    }
    if manifest.mode_id != "standard" {
        return Err("M1 validator currently requires standard mode".into());
    }
    if manifest.run_kind == "production" && manifest.dirty_worktree {
        return Err("production corpus cannot come from a dirty worktree".into());
    }
    if manifest.run_kind == "smoke" && !manifest.run_id.to_ascii_lowercase().contains("smoke") {
        return Err("smoke corpus run_id must contain 'smoke'".into());
    }
    if !is_hex(&manifest.engine_commit, 40)
        || !is_hex(&manifest.ruleset.manifest_sha256, 64)
        || manifest.search_semantics_id.is_empty()
        || manifest.root_selection_id.is_empty()
        || manifest.chance_sampling_id.is_empty()
    {
        return Err("V4 corpus provenance is invalid".into());
    }
    if manifest.evaluator.kind == "onnx" {
        for (name, value) in [
            ("model_sha256", manifest.evaluator.model_sha256.as_deref()),
            (
                "bundle_metadata_sha256",
                manifest.evaluator.bundle_metadata_sha256.as_deref(),
            ),
            (
                "state_encoder_sha256",
                manifest.evaluator.state_encoder_sha256.as_deref(),
            ),
            (
                "factorized_scorer_sha256",
                manifest.evaluator.factorized_scorer_sha256.as_deref(),
            ),
        ] {
            if value.is_none_or(|hash| !is_hex(hash, 64)) {
                return Err(format!("ONNX evaluator {name} is missing or invalid").into());
            }
        }
    } else if manifest.evaluator.kind == "bootstrap" {
        if manifest.evaluator.bootstrap_seed.is_none() || manifest.evaluator.model_sha256.is_some()
        {
            return Err("bootstrap evaluator provenance is invalid".into());
        }
    } else {
        return Err("unknown evaluator kind".into());
    }
    let _ = manifest.temperature_decisions;
    Ok(())
}

fn validate_game_identity(
    manifest: &RunManifestV4,
    artifact: &GameArtifactV4,
    game: &GameRecordV4,
) -> Result<(), Box<dyn std::error::Error>> {
    if game.game_id != artifact.game_id
        || game.seed_hex != artifact.seed_hex
        || game.trajectory_sha256 != artifact.trajectory_sha256
        || game.steps.len() != artifact.decisions
        || game.termination != artifact.termination
        || game.ruleset != manifest.ruleset
        || game.mode_id != manifest.mode_id
        || game.generator.run_id != manifest.run_id
        || game.generator.engine_commit != manifest.engine_commit
        || game.generator.dirty_worktree != manifest.dirty_worktree
    {
        return Err(format!("game {} manifest identity mismatch", game.game_id).into());
    }
    for player in &game.players {
        if player.agent_id != manifest.evaluator.agent_id {
            return Err(format!("game {} evaluator ID mismatch", game.game_id).into());
        }
        match (
            manifest.evaluator.model_sha256.as_deref(),
            player.model.as_ref(),
        ) {
            (Some(expected), Some(model)) if model.sha256 == expected => {}
            (None, None) => {}
            _ => return Err(format!("game {} model provenance mismatch", game.game_id).into()),
        }
    }
    for step in &game.steps {
        if step.policy_target.completed_simulations as usize != manifest.simulations
            || step.policy_target.chance_outcome_samples as usize != manifest.chance_outcome_samples
            || step.policy_target.search_semantics_id != manifest.search_semantics_id
            || step.policy_target.root_selection_id != manifest.root_selection_id
            || step.policy_target.chance_sampling_id != manifest.chance_sampling_id
        {
            return Err(format!(
                "game {} decision {} search provenance mismatch",
                game.game_id, step.decision_index
            )
            .into());
        }
    }
    Ok(())
}

fn validate_replay(game: &GameRecordV4, engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    let seed = u64::from_str_radix(&game.seed_hex, 16)?;
    let mode = match game.mode_id.as_str() {
        "standard" => Mode::Standard,
        "chaos" => Mode::Chaos,
        "grand" => Mode::Grand,
        other => Mode::Campaign(other.to_owned()),
    };
    let mut state = engine.new_game(mode, seed)?;
    if state.ruleset_id != game.ruleset.id
        || fingerprint(state.fingerprint()) != game.initial_state_fingerprint
    {
        return Err(format!("game {} initial engine state drifted", game.game_id).into());
    }

    for step in &game.steps {
        if AbsolutePlayerV3::from_player(state.turn) != step.actor
            || state.ply != step.engine_ply
            || fingerprint(state.fingerprint()) != step.state_fingerprint
            || encode_observation_v3(&state) != step.observation
        {
            return Err(format!(
                "game {} decision {} engine state/observation drifted",
                game.game_id, step.decision_index
            )
            .into());
        }
        let legal = LegalActionSetV3::from_state(&state, engine)?;
        if legal != step.legal_actions {
            return Err(format!(
                "game {} decision {} legal action set drifted",
                game.game_id, step.decision_index
            )
            .into());
        }
        let transition = legal.apply(step.played_action_index as usize, &state, engine)?;
        state = transition.state;
        if fingerprint(state.fingerprint()) != step.next_state_fingerprint {
            return Err(format!(
                "game {} decision {} next-state fingerprint drifted",
                game.game_id, step.decision_index
            )
            .into());
        }
    }

    if game.termination.final_state_fingerprint() != fingerprint(state.fingerprint()) {
        return Err(format!("game {} final fingerprint drifted", game.game_id).into());
    }
    match (&game.termination, &state.status) {
        (TerminationV3::EngineTerminal { .. }, GameStatus::Ongoing) => {
            return Err(format!("game {} falsely claims terminal result", game.game_id).into());
        }
        (TerminationV3::EngineTerminal { .. }, _) => {
            let expected =
                TerminationV3::from_engine_status(&state.status, fingerprint(state.fingerprint()))
                    .ok_or("terminal status conversion failed")?;
            if expected != game.termination {
                return Err(format!("game {} terminal metadata drifted", game.game_id).into());
            }
            for step in &game.steps {
                let value = game
                    .termination
                    .value_for(step.actor)
                    .ok_or("terminal game has no actor-perspective value")?;
                if !matches!(value, -1.0 | 0.0 | 1.0) {
                    return Err(
                        format!("game {} actor-perspective result invalid", game.game_id).into(),
                    );
                }
            }
        }
        (TerminationV3::Limit { .. }, GameStatus::Ongoing) => {}
        (TerminationV3::Aborted { .. } | TerminationV3::Error { .. }, _) => {
            return Err(format!("game {} has abnormal termination", game.game_id).into());
        }
        _ => return Err(format!("game {} hides an engine terminal state", game.game_id).into()),
    }
    Ok(())
}

fn find_repo_root(start: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut current = start
        .canonicalize()?
        .parent()
        .ok_or("manifest has no parent directory")?
        .to_path_buf();
    loop {
        if current.join("Cargo.toml").exists() && current.join("rulesets").exists() {
            return Ok(current);
        }
        if !current.pop() {
            return Err("could not locate repository root".into());
        }
    }
}

fn validate_relative_path(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if path.contains('\\') {
        return Err(format!("artifact path must use forward slashes: {path}").into());
    }
    let candidate = Path::new(path);
    if candidate.is_absolute()
        || candidate.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!("artifact path is not safely relative: {path}").into());
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    Ok(sha256_bytes(&std::fs::read(path)?))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fingerprint(value: u64) -> String {
    format!("{value:016x}")
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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

fn optional_path(args: &[String], name: &str) -> Option<PathBuf> {
    args.iter()
        .position(|item| item == name)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
}

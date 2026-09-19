use augment_ai::data_v3::{
    AbsolutePlayerV3, AuxiliaryTargetV3, AuxiliaryTargetValueV3, CompressionV3,
    DATASET_MANIFEST_V3_SCHEMA_ID, DECISION_RECORD_V3_SCHEMA_ID, DataRecordV3, DatasetManifestV3,
    DatasetSplitV3, DatasetTotalsV3, DecisionPolicySupervisionV3, DecisionPolicyTargetV3,
    DecisionProvenanceV3, DecisionRecordV3, DecisionSourceKindV3, DecisionTargetsV3,
    DecisionValueTargetV3, GeneratorProvenanceV3, PolicyTrainingScopeV3, RulesetRefV3,
    SourceDomainV3, fingerprint_hex,
};
use augment_ai::{
    ACTION_V3_SCHEMA_ID, ForcedWinConfigV3, ForcedWinErrorV3, LegalActionSetV3,
    OBSERVATION_V3_SCHEMA_ID, PhaseV3, encode_observation_v3, solve_forced_win_v3,
    validate_manifest, verify_manifest_replays, write_jsonl_zstd_shard,
};
use augment_core::{
    BUNDLED_RULESET_ID, CardActivation, CardInstance, CardPhase, Engine, GamePhase, GameStatus,
    Mode, implemented_card,
};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const DEFAULT_ATTACKER_CARDS: &str = "disarm,freeze,gale,panic,reposition,vortex";
const DEFAULT_DEFENDER_CARDS: &str =
    "parry,evasion,retreat,emergency-evacuation,sacrifice,suspicious-potion";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ScenarioOwner {
    Actor,
    Opponent,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CardGrant {
    definition_id: String,
    owner: ScenarioOwner,
}

#[derive(Clone, Debug)]
struct CandidatePosition {
    game_id: String,
    seed_hex: String,
    mode_id: String,
    decision_index: u32,
    played_action_prefix: Vec<u32>,
    expected_state_fingerprint: String,
    scenario_id: String,
    card_grants: Vec<CardGrant>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("V3 tactics solver failed: {error}");
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
    let card_scenarios = has_flag(&args, "--card-scenarios");
    let card_scenario_stride = numeric(&args, "--card-scenario-stride", 1usize)?;
    let max_candidates = numeric(&args, "--max-candidates", usize::MAX)?;
    if card_scenario_stride == 0 || max_candidates == 0 {
        return Err("card scenario stride and max candidates must be positive".into());
    }
    let attacker_cards = if card_scenarios {
        card_ids(&args, "--attacker-cards", DEFAULT_ATTACKER_CARDS)?
    } else {
        Vec::new()
    };
    let defender_cards = if card_scenarios {
        card_ids(&args, "--defender-cards", DEFAULT_DEFENDER_CARDS)?
    } else {
        Vec::new()
    };
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
    let generator = GeneratorProvenanceV3 {
        run_id: run_id.clone(),
        engine_commit: engine_commit.clone(),
        engine_api_version: 1,
        collector_id: if card_scenarios {
            "augment-cli/solve_v3_tactics_m3"
        } else {
            "augment-cli/solve_v3_tactics_m2"
        }
        .to_owned(),
        dirty_worktree,
    };

    let engine = Engine::bundled_20260807()?;
    let manifest_bytes = std::fs::read(&selfplay_manifest)?;
    let dataset_manifest: DatasetManifestV3 = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest(&dataset_manifest)?;
    let split_policy = dataset_manifest.split_policy.clone();
    let selfplay_root = selfplay_manifest.parent().unwrap_or_else(|| Path::new("."));
    verify_manifest_replays(selfplay_root, &dataset_manifest, &engine)?;
    let mut base_candidates: Vec<CandidatePosition> = Vec::new();
    for shard in &dataset_manifest.shards {
        let records = read_records(&selfplay_manifest, &shard.path)?;
        for record in records {
            if let DataRecordV3::Game(game) = record {
                let mut played_action_prefix = Vec::with_capacity(game.steps.len());
                for step in &game.steps {
                    if !card_scenarios || matches!(step.observation.phase, PhaseV3::Playing) {
                        base_candidates.push(CandidatePosition {
                            game_id: game.game_id.clone(),
                            seed_hex: game.seed_hex.clone(),
                            mode_id: game.mode_id.clone(),
                            decision_index: step.decision_index,
                            played_action_prefix: played_action_prefix.clone(),
                            expected_state_fingerprint: step.state_fingerprint.clone(),
                            scenario_id: "base".to_owned(),
                            card_grants: Vec::new(),
                        });
                    }
                    played_action_prefix.push(step.played_action_index);
                }
            }
        }
    }
    let candidates = if card_scenarios {
        build_card_scenarios(
            &base_candidates,
            &attacker_cards,
            &defender_cards,
            card_scenario_stride,
            max_candidates,
        )
    } else {
        base_candidates.into_iter().take(max_candidates).collect()
    };
    eprintln!(
        "{}",
        serde_json::json!({
            "event": "candidates_loaded",
            "count": candidates.len(),
            "card_scenarios": card_scenarios,
            "attacker_cards": attacker_cards,
            "defender_cards": defender_cards,
        })
    );

    let mut proven_records: Vec<DataRecordV3> = Vec::new();
    let mut seen_positions: BTreeSet<String> = BTreeSet::new();
    let solved = AtomicU64::new(0);
    let refuted = AtomicU64::new(0);
    let started = Instant::now();
    let mut candidate_scenarios = BTreeMap::<String, u64>::new();
    let mut candidate_cards = BTreeMap::<String, u64>::new();
    let mut proven_scenarios = BTreeMap::<String, u64>::new();
    let mut proven_cards = BTreeMap::<String, u64>::new();
    for candidate in &candidates {
        *candidate_scenarios
            .entry(candidate.scenario_id.clone())
            .or_default() += 1;
        for grant in &candidate.card_grants {
            *candidate_cards
                .entry(grant.definition_id.clone())
                .or_default() += 1;
        }
    }

    let results: Vec<Result<Option<DecisionRecordV3>, ForcedWinErrorV3>> = candidates
        .par_iter()
        .map_init(
            || Engine::bundled_20260807().expect("engine init"),
            |engine, candidate| {
                solve_candidate(
                    engine,
                    candidate,
                    &ruleset_hash,
                    &engine_commit,
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
                    *proven_scenarios
                        .entry(candidate.scenario_id.clone())
                        .or_default() += 1;
                    for grant in &candidate.card_grants {
                        *proven_cards.entry(grant.definition_id.clone()).or_default() += 1;
                    }
                    proven_records.push(DataRecordV3::Decision(Box::new(record)));
                } else {
                    refuted.fetch_add(1, Ordering::Relaxed);
                    append_refuted(&refuted_log, candidate, "duplicate_position")?;
                }
            }
            Err(ForcedWinErrorV3::NodeBudgetExceeded { .. }) => {
                budget_exceeded += 1;
                append_refuted(&refuted_log, candidate, "node_budget_exceeded")?;
            }
            Ok(None) => {
                refuted.fetch_add(1, Ordering::Relaxed);
                append_refuted(&refuted_log, candidate, "not_proven")?;
            }
            Err(error) => {
                return Err(format!(
                    "candidate {}#{}#{} failed: {error}",
                    candidate.game_id, candidate.decision_index, candidate.scenario_id
                )
                .into());
            }
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
        shards.push(summary.to_manifest_entry(relative, *split));
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
    verify_manifest_replays(&output_dir, &manifest, &engine)?;
    atomic_json(
        &output_dir.join("scenario-summary.json"),
        &serde_json::json!({
            "schema_id": "augmentia.solver-scenario-summary.v3",
            "dataset_id": manifest.dataset_id,
            "source_manifest_sha256": sha256_bytes(&manifest_bytes),
            "source_split_policy": &manifest.split_policy,
            "generator": generator,
            "max_depth": max_depth,
            "max_nodes_per_position": max_nodes_per_position,
            "card_scenarios": card_scenarios,
            "card_scenario_stride": card_scenario_stride,
            "candidate_scenarios": candidate_scenarios,
            "candidate_cards": candidate_cards,
            "proven_scenarios": proven_scenarios,
            "proven_cards": proven_cards,
            "candidates": candidates.len(),
            "proven": solved.load(Ordering::Relaxed),
            "refuted": refuted.load(Ordering::Relaxed),
            "budget_exceeded": budget_exceeded,
        }),
    )?;
    atomic_json(&output_dir.join("manifest.json"), &manifest)?;
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

fn build_card_scenarios(
    bases: &[CandidatePosition],
    attacker_cards: &[String],
    defender_cards: &[String],
    stride: usize,
    max_candidates: usize,
) -> Vec<CandidatePosition> {
    let eligible: Vec<_> = bases.iter().step_by(stride).collect();
    let requested_bases = max_candidates / 3 + usize::from(max_candidates % 3 != 0);
    let selected: Vec<_> = if requested_bases < eligible.len() {
        (0..requested_bases)
            .map(|index| eligible[index.saturating_mul(eligible.len()) / requested_bases])
            .collect()
    } else {
        eligible
    };
    let mut scenarios = Vec::with_capacity(max_candidates.min(selected.len().saturating_mul(3)));
    for (index, base) in selected.into_iter().enumerate() {
        let attacker = &attacker_cards[index % attacker_cards.len()];
        let defender = &defender_cards[(index.saturating_mul(7) + 3) % defender_cards.len()];
        let variants = [
            (
                format!("attacker:{attacker}"),
                vec![CardGrant {
                    definition_id: attacker.clone(),
                    owner: ScenarioOwner::Actor,
                }],
            ),
            (
                format!("defender:{defender}"),
                vec![CardGrant {
                    definition_id: defender.clone(),
                    owner: ScenarioOwner::Opponent,
                }],
            ),
            (
                format!("both:{attacker}+{defender}"),
                vec![
                    CardGrant {
                        definition_id: attacker.clone(),
                        owner: ScenarioOwner::Actor,
                    },
                    CardGrant {
                        definition_id: defender.clone(),
                        owner: ScenarioOwner::Opponent,
                    },
                ],
            ),
        ];
        for (scenario_id, card_grants) in variants {
            let mut scenario = base.clone();
            scenario.scenario_id = scenario_id;
            scenario.card_grants = card_grants;
            scenarios.push(scenario);
            if scenarios.len() == max_candidates {
                return scenarios;
            }
        }
    }
    scenarios
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
    state = apply_card_grants(state, engine, &candidate.card_grants)?;
    Ok(Some(state))
}

fn apply_card_grants(
    mut state: augment_core::GameState,
    engine: &Engine,
    grants: &[CardGrant],
) -> Result<augment_core::GameState, Box<dyn std::error::Error>> {
    if !matches!(state.phase, GamePhase::Playing) {
        return Err("synthetic cards require a playing-phase state".into());
    }
    let actor = state.turn;
    let mut next_instance_id = u32::MAX;
    for grant in grants {
        while state
            .cards
            .iter()
            .any(|card| card.instance_id == next_instance_id)
        {
            next_instance_id = next_instance_id
                .checked_sub(1)
                .ok_or("no free synthetic card instance id")?;
        }
        let owner = match grant.owner {
            ScenarioOwner::Actor => actor,
            ScenarioOwner::Opponent => actor.opponent(),
        };
        let definition = implemented_card(&grant.definition_id)
            .ok_or_else(|| format!("unknown synthetic card {:?}", grant.definition_id))?;
        if definition.activation == CardActivation::Passive {
            state = engine
                .grant_card(&state, next_instance_id, &grant.definition_id, owner)?
                .state;
        } else {
            state.add_card(CardInstance {
                instance_id: next_instance_id,
                definition_id: grant.definition_id.clone(),
                owner,
                half_stars: definition.half_stars,
                charges: 1,
                available_after_turn: None,
            });
        }
        next_instance_id = next_instance_id
            .checked_sub(1)
            .ok_or("no free synthetic card instance id")?;
    }
    Ok(state)
}

fn solve_candidate(
    engine: &Engine,
    candidate: &CandidatePosition,
    ruleset_hash: &str,
    engine_commit: &str,
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
    )
    .map_err(|error| contextualize_solver_error(error, &state))?;
    let Some(solution) = solution else {
        return Ok(None);
    };
    let proof_source = format!(
        "solver-m3/scenario={}/depth={}/nodes={}",
        candidate.scenario_id, max_depth, solution.searched_nodes
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
            source_id: format!(
                "{}#{}#{}",
                candidate.game_id, candidate.decision_index, candidate.scenario_id
            ),
            split_group_id: format!(
                "game\0{}\0{}\0{}",
                BUNDLED_RULESET_ID, candidate.mode_id, candidate.seed_hex
            ),
            generator_commit: Some(engine_commit.to_owned()),
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

fn contextualize_solver_error(
    error: ForcedWinErrorV3,
    state: &augment_core::GameState,
) -> ForcedWinErrorV3 {
    match error {
        ForcedWinErrorV3::NodeBudgetExceeded { .. } => error,
        other => ForcedWinErrorV3::Engine(format!(
            "{other}; cards={:?}; opening_auto_card_ids={:?}; turns_taken={:?}",
            state
                .cards
                .iter()
                .map(|card| (
                    card.instance_id,
                    card.definition_id.as_str(),
                    card.owner,
                    card.charges
                ))
                .collect::<Vec<_>>(),
            state.card_runtime.opening_auto_card_ids,
            state.turns_taken,
        )),
    }
}

fn append_refuted(
    log: &Path,
    candidate: &CandidatePosition,
    reason: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)?;
    serde_json::to_writer(
        &mut file,
        &serde_json::json!({
            "source_id": format!(
                "{}#{}#{}",
                candidate.game_id, candidate.decision_index, candidate.scenario_id
            ),
            "game_id": candidate.game_id,
            "decision_index": candidate.decision_index,
            "scenario_id": candidate.scenario_id,
            "card_grants": candidate.card_grants,
            "reason": reason,
        }),
    )?;
    file.write_all(b"\n")?;
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

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|item| item == flag)
}

fn card_ids(args: &[String], flag: &str, default: &str) -> Result<Vec<String>, String> {
    let raw = args
        .iter()
        .position(|item| item == flag)
        .and_then(|index| args.get(index + 1))
        .map_or(default, String::as_str);
    let ids: Vec<_> = raw
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect();
    if ids.is_empty() {
        return Err(format!("{flag} must contain at least one card id"));
    }
    let unique: BTreeSet<_> = ids.iter().collect();
    if unique.len() != ids.len() {
        return Err(format!("{flag} contains duplicate card ids"));
    }
    for id in &ids {
        let definition = implemented_card(id)
            .ok_or_else(|| format!("{flag} contains unknown card id {id:?}"))?;
        if definition.activation == CardActivation::Rule
            || !matches!(definition.phase, CardPhase::Middle | CardPhase::End)
        {
            return Err(format!(
                "{flag} card id {id:?} cannot be injected into a playing-state scenario"
            ));
        }
    }
    Ok(ids)
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

#[cfg(test)]
mod tests {
    use super::*;
    use augment_core::{GamePhase, GameState, Player};

    fn candidate(index: u32) -> CandidatePosition {
        CandidatePosition {
            game_id: "game-1".to_owned(),
            seed_hex: "0000000000000001".to_owned(),
            mode_id: "standard".to_owned(),
            decision_index: index,
            played_action_prefix: vec![0; index as usize],
            expected_state_fingerprint: format!("fingerprint-{index}"),
            scenario_id: "base".to_owned(),
            card_grants: Vec::new(),
        }
    }

    #[test]
    fn card_scenarios_are_deterministic_strided_and_bounded() {
        let bases: Vec<_> = (0..4).map(candidate).collect();
        let attackers = card_ids(&[], "--attacker-cards", DEFAULT_ATTACKER_CARDS).unwrap();
        let defenders = card_ids(&[], "--defender-cards", DEFAULT_DEFENDER_CARDS).unwrap();
        let scenarios = build_card_scenarios(&bases, &attackers, &defenders, 2, 5);

        assert_eq!(scenarios.len(), 5);
        assert_eq!(scenarios[0].scenario_id, "attacker:disarm");
        assert_eq!(scenarios[1].scenario_id, "defender:emergency-evacuation");
        assert_eq!(scenarios[2].scenario_id, "both:disarm+emergency-evacuation");
        assert_eq!(scenarios[3].decision_index, 2);
        assert_eq!(scenarios[3].scenario_id, "attacker:freeze");
        assert_eq!(scenarios[4].scenario_id, "defender:sacrifice");

        let wide_bases: Vec<_> = (0..12).map(candidate).collect();
        let sampled = build_card_scenarios(&wide_bases, &attackers, &defenders, 1, 6);
        assert_eq!(sampled[0].decision_index, 0);
        assert_eq!(sampled[3].decision_index, 6);
    }

    #[test]
    fn synthetic_cards_are_granted_to_actor_relative_owners() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 7);
        state.phase = GamePhase::Playing;
        let grants = vec![
            CardGrant {
                definition_id: "parry".to_owned(),
                owner: ScenarioOwner::Actor,
            },
            CardGrant {
                definition_id: "evasion".to_owned(),
                owner: ScenarioOwner::Opponent,
            },
        ];
        let state = apply_card_grants(state, &engine, &grants).unwrap();

        assert!(
            state
                .cards
                .iter()
                .any(|card| { card.definition_id == "parry" && card.owner == Player::White })
        );
        assert!(
            state
                .cards
                .iter()
                .any(|card| { card.definition_id == "evasion" && card.owner == Player::Black })
        );
        for action in engine.legal_actions(&state).unwrap() {
            engine.apply(&state, action).unwrap();
        }
    }

    #[test]
    fn card_cli_rejects_rule_cards_and_duplicates() {
        let rule = vec!["--cards".to_owned(), "platform".to_owned()];
        assert!(card_ids(&rule, "--cards", "parry").is_err());
        let piece_phase = vec!["--cards".to_owned(), "assassin".to_owned()];
        assert!(card_ids(&piece_phase, "--cards", "parry").is_err());
        let duplicate = vec!["--cards".to_owned(), "parry,parry".to_owned()];
        assert!(card_ids(&duplicate, "--cards", "evasion").is_err());
    }

    #[test]
    fn node_budget_errors_remain_classifiable_after_context_enrichment() {
        let state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 9);
        let error = contextualize_solver_error(
            ForcedWinErrorV3::NodeBudgetExceeded { max_nodes: 17 },
            &state,
        );
        assert!(matches!(
            error,
            ForcedWinErrorV3::NodeBudgetExceeded { max_nodes: 17 }
        ));

        let enriched =
            contextualize_solver_error(ForcedWinErrorV3::Engine("broken".to_owned()), &state);
        assert!(matches!(
            enriched,
            ForcedWinErrorV3::Engine(message) if message.contains("broken") && message.contains("cards=[]")
        ));
    }
}

use augment_ai::data_v3::{
    AbsolutePlayerV3, AuxiliaryTargetV3, AuxiliaryTargetValueV3, CompressionV3,
    DATASET_MANIFEST_V3_SCHEMA_ID, DECISION_RECORD_V3_SCHEMA_ID, DataRecordV3, DatasetManifestV3,
    DatasetSplitV3, DatasetTotalsV3, DecisionPolicySupervisionV3, DecisionPolicyTargetV3,
    DecisionProvenanceV3, DecisionRecordV3, DecisionSourceKindV3, DecisionTargetsV3,
    DecisionValueTargetV3, PolicyTrainingScopeV3, RulesetRefV3, SPLIT_POLICY_V3_ALGORITHM,
    ShardManifestV3, SourceDomainV3, SplitPolicyV3, fingerprint_hex,
};
use augment_ai::{
    ACTION_V3_SCHEMA_ID, ActionDescriptorV3, ForcedWinConfigV3, ForcedWinErrorV3, LegalActionSetV3,
    OBSERVATION_V3_SCHEMA_ID, PIECE_KIND_VOCABULARY, encode_observation_v3, solve_forced_win_v3,
    validate_manifest, verify_manifest_files, write_jsonl_zstd_shard,
};
use augment_core::{
    Action, BUNDLED_RULESET_ID, Board, EnPassantState, Engine, GameState, Mode, Piece, PieceKind,
    Player, Square,
};
use pgn_reader::shakmaty::{
    CastlingMode, Chess, Color, Move as ChessMove, Position, Role, fen::Fen,
};
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Instant;

const COVERAGE_SCHEMA_ID: &str = "augmentia.bootstrap.coverage.v3";
const TACTICAL_INPUT_SCHEMA_ID: &str = "augmentia.bootstrap.tactical-input.v1";

fn main() {
    if let Err(error) = run(std::env::args().skip(1).collect()) {
        eprintln!("V3 bootstrap generation failed: {error}");
        std::process::exit(1);
    }
}

fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let started_at = Instant::now();
    let config = Config::parse(&args)?;
    let engine = Engine::bundled_20260807()?;
    let ruleset = RulesetRefV3 {
        id: BUNDLED_RULESET_ID.to_owned(),
        manifest_sha256: sha256_file(&config.ruleset_manifest)?,
    };
    if config.output_dir.join("manifest.json").exists() {
        return Err(format!(
            "refusing to overwrite existing dataset {}",
            config.output_dir.display()
        )
        .into());
    }

    let mut coverage = BootstrapCoverageV3::new(&config);
    let split_policy = SplitPolicyV3 {
        algorithm: SPLIT_POLICY_V3_ALGORITHM.to_owned(),
        salt: config.split_salt.clone(),
        train_basis_points: 9_000,
        validation_basis_points: 500,
        holdout_basis_points: 500,
    };
    let mut writer = BootstrapShardWriter::new(
        &config.output_dir,
        split_policy.clone(),
        config.shard_records,
        config.compression_level,
    );
    if let Some(pgn_path) = &config.pgn {
        import_pgn(
            pgn_path,
            &config,
            &ruleset,
            &engine,
            &mut writer,
            &mut coverage,
        )?;
    }
    if let Some(tactics_path) = &config.tactics {
        import_tactical_scenarios(
            tactics_path,
            &config,
            &ruleset,
            &engine,
            &mut writer,
            &mut coverage,
        )?;
    }
    if writer.total_records() == 0 {
        return Err("no valid bootstrap records were generated".into());
    }
    let shards = writer.finish()?;
    coverage.output_shards = shards.len() as u64;
    coverage.output_compressed_bytes = shards.iter().map(|shard| shard.compressed_bytes).sum();
    coverage.generation_elapsed_ms = started_at.elapsed().as_millis() as u64;
    coverage.records_per_second = manifest_rate(
        coverage.board_pretraining_records + coverage.proven_tactical_records,
        started_at.elapsed().as_secs_f64(),
    );
    let totals = totals_from_shards(&shards);
    let manifest = DatasetManifestV3 {
        manifest_schema_id: DATASET_MANIFEST_V3_SCHEMA_ID.to_owned(),
        dataset_id: config.dataset_id.clone(),
        created_at_utc: config.created_at_utc.clone(),
        observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
        action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
        split_policy,
        compression: CompressionV3 {
            codec: "zstd".to_owned(),
            level: config.compression_level,
        },
        shards,
        totals,
    };
    validate_manifest(&manifest)?;
    write_new_json(&config.output_dir.join("coverage.json"), &coverage)?;
    verify_manifest_files(&config.output_dir, &manifest)?;
    write_new_json(&config.output_dir.join("manifest.json"), &manifest)?;
    println!(
        "COMPLETE dataset={} records={} board_pretraining={} proven_tactics={} skipped={}",
        manifest.dataset_id,
        manifest.totals.records,
        coverage.board_pretraining_records,
        coverage.proven_tactical_records,
        coverage.skipped_positions,
    );
    Ok(())
}

#[derive(Clone, Debug)]
struct Config {
    pgn: Option<PathBuf>,
    tactics: Option<PathBuf>,
    output_dir: PathBuf,
    dataset_id: String,
    created_at_utc: String,
    engine_commit: String,
    pgn_license: Option<String>,
    expected_pgn_sha256: Option<String>,
    expected_pgn_bytes: Option<u64>,
    tactics_license: Option<String>,
    ruleset_manifest: PathBuf,
    split_salt: String,
    max_games: usize,
    max_records: usize,
    shard_records: usize,
    min_rating: u32,
    allowed_speeds: BTreeSet<String>,
    rated_only: bool,
    normal_termination_only: bool,
    min_game_plies: usize,
    tactical_depth: u8,
    tactical_nodes: u64,
    compression_level: i32,
}

impl Config {
    fn parse(args: &[String]) -> Result<Self, String> {
        let pgn = optional_arg(args, "--pgn").map(PathBuf::from);
        let tactics = optional_arg(args, "--tactics").map(PathBuf::from);
        if pgn.is_none() && tactics.is_none() {
            return Err("at least one of --pgn or --tactics is required".to_owned());
        }
        let dataset_id = required_arg(args, "--dataset-id")?;
        let created_at_utc = required_arg(args, "--created-at-utc")?;
        if !created_at_utc.ends_with('Z') {
            return Err("--created-at-utc must be UTC and end with Z".to_owned());
        }
        let engine_commit = required_arg(args, "--engine-commit")?;
        if engine_commit.len() != 40
            || !engine_commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("--engine-commit must be a 40-character lowercase Git hash".to_owned());
        }
        let pgn_license = match &pgn {
            Some(_) => Some(required_arg(args, "--pgn-license")?),
            None => None,
        };
        let expected_pgn_sha256 = optional_arg(args, "--expected-pgn-sha256");
        if let Some(expected) = &expected_pgn_sha256
            && (expected.len() != 64
                || !expected
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
        {
            return Err("--expected-pgn-sha256 must be a 64-character lowercase hex digest".into());
        }
        let expected_pgn_bytes = optional_arg(args, "--expected-pgn-bytes")
            .map(|value| {
                value
                    .parse::<u64>()
                    .map_err(|error| format!("invalid --expected-pgn-bytes: {error}"))
            })
            .transpose()?;
        if pgn.is_none() && (expected_pgn_sha256.is_some() || expected_pgn_bytes.is_some()) {
            return Err("expected PGN identity arguments require --pgn".into());
        }
        let tactics_license = match &tactics {
            Some(_) => Some(required_arg(args, "--tactics-license")?),
            None => None,
        };
        let max_games = parse_arg(args, "--max-games", 1_000usize)?;
        let max_records = parse_arg(args, "--max-records", 100_000usize)?;
        let shard_records = parse_arg(args, "--shard-records", 10_000usize)?;
        if max_games == 0 || max_records == 0 || shard_records == 0 {
            return Err(
                "--max-games, --max-records and --shard-records must be positive".to_owned(),
            );
        }
        let allowed_speeds = optional_arg(args, "--speeds")
            .map(|value| parse_speeds(&value))
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            pgn,
            tactics,
            output_dir: PathBuf::from(required_arg(args, "--out")?),
            dataset_id,
            created_at_utc,
            engine_commit,
            pgn_license,
            expected_pgn_sha256,
            expected_pgn_bytes,
            tactics_license,
            ruleset_manifest: PathBuf::from(
                optional_arg(args, "--ruleset-manifest")
                    .unwrap_or_else(|| "rulesets/augment_chess_20260818/manifest.json".to_owned()),
            ),
            split_salt: optional_arg(args, "--split-salt")
                .unwrap_or_else(|| "augmentia-bootstrap-v3".to_owned()),
            max_games,
            max_records,
            shard_records,
            min_rating: parse_arg(args, "--min-rating", 0u32)?,
            allowed_speeds,
            rated_only: has_flag(args, "--rated-only"),
            normal_termination_only: has_flag(args, "--normal-termination-only"),
            min_game_plies: parse_arg(args, "--min-game-plies", 1usize)?,
            tactical_depth: parse_arg(args, "--tactical-depth", 3u8)?,
            tactical_nodes: parse_arg(args, "--tactical-nodes", 100_000u64)?,
            compression_level: parse_arg(args, "--compression-level", 3i32)?,
        })
    }
}

fn import_pgn(
    path: &Path,
    config: &Config,
    ruleset: &RulesetRefV3,
    engine: &Engine,
    writer: &mut BootstrapShardWriter<'_>,
    coverage: &mut BootstrapCoverageV3,
) -> Result<(), Box<dyn std::error::Error>> {
    let source_sha256 = sha256_file(path)?;
    if let Some(expected) = &config.expected_pgn_sha256
        && &source_sha256 != expected
    {
        return Err(
            format!("PGN SHA-256 mismatch: expected {expected}, got {source_sha256}").into(),
        );
    }
    let input_bytes = std::fs::metadata(path)?.len();
    if let Some(expected) = config.expected_pgn_bytes
        && input_bytes != expected
    {
        return Err(
            format!("PGN byte count mismatch: expected {expected}, got {input_bytes}").into(),
        );
    }
    coverage.pgn_sha256 = Some(source_sha256.clone());
    coverage.pgn_input_bytes = Some(input_bytes);
    coverage.pgn_compression = Some(if is_zstd_path(path) {
        "zstd".to_owned()
    } else {
        "none".to_owned()
    });
    let mut reader = Reader::new(BufReader::new(open_pgn_input(path)?));
    let mut visitor = MainlineVisitor;
    let mut game_index = 0usize;

    while coverage.games_accepted < config.max_games as u64
        && writer.total_records() < config.max_records as u64
    {
        let Some(game) = reader.read_game(&mut visitor)? else {
            break;
        };
        coverage.games_read += 1;
        match game {
            Ok(game) => {
                if let Some(reason) = pgn_filter_rejection(&game, config) {
                    coverage.observe_filtered_game(reason);
                    game_index += 1;
                    continue;
                }
                coverage.games_accepted += 1;
                // `Site` is a game URL in standard Lichess dumps, but broadcast PGNs may
                // reuse one event URL for many boards. The streaming game index is stable
                // for an immutable, hash-addressed source and makes every game group unique.
                let game_key = game
                    .site
                    .as_deref()
                    .filter(|site| !site.trim().is_empty())
                    .map_or_else(
                        || format!("index:{game_index}"),
                        |site| format!("index:{game_index}:site:{site}"),
                    );
                let split_group_id = format!("standard:{source_sha256}:game:{game_key}");
                for (ply_index, ply) in game.plies.into_iter().enumerate() {
                    if writer.total_records() >= config.max_records as u64 {
                        break;
                    }
                    coverage.positions_seen += 1;
                    let source_id = format!("pgn:{source_sha256}:game:{game_key}:ply:{ply_index}");
                    let state = standard_position_to_augment_state(&ply.position)?;
                    if let Some(record) = board_pretraining_record(
                        &state,
                        &ply.position,
                        &ply.played_move,
                        SourcePositionRef {
                            source_id,
                            split_group_id: split_group_id.clone(),
                        },
                        config,
                        ruleset,
                        engine,
                    )? {
                        emit_record(DataRecordV3::Decision(Box::new(record)), writer, coverage)?;
                    } else {
                        coverage.skipped_positions += 1;
                    }
                }
            }
            Err(_) => coverage.invalid_games += 1,
        }
        game_index += 1;
    }
    Ok(())
}

fn import_tactical_scenarios(
    path: &Path,
    config: &Config,
    ruleset: &RulesetRefV3,
    engine: &Engine,
    writer: &mut BootstrapShardWriter<'_>,
    coverage: &mut BootstrapCoverageV3,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let input: TacticalInputV1 = serde_json::from_slice(&bytes)?;
    if input.schema_id != TACTICAL_INPUT_SCHEMA_ID {
        return Err(format!("unsupported tactical input schema {}", input.schema_id).into());
    }
    let source_sha256 = sha256_bytes(&bytes);
    coverage.tactical_input_sha256 = Some(source_sha256.clone());
    let mut seen_training_content = BTreeSet::new();
    for scenario in input.scenarios {
        if writer.total_records() >= config.max_records as u64 {
            break;
        }
        coverage.positions_seen += 1;
        let state = scenario.to_state()?;
        let source_id = format!("scenario:{source_sha256}:{}", scenario.id);
        match proven_tactical_record(
            &state,
            source_id,
            format!("scenario:{source_sha256}:{}", scenario.group_id),
            config
                .tactics_license
                .clone()
                .expect("required with --tactics"),
            config,
            ruleset,
            engine,
        ) {
            Ok(Some(record)) => {
                if seen_training_content.insert(decision_training_digest(&record)?) {
                    emit_record(DataRecordV3::Decision(Box::new(record)), writer, coverage)?;
                } else {
                    coverage.duplicates_removed += 1;
                }
            }
            Ok(None) => coverage.unproven_tactical_positions += 1,
            Err(ForcedWinErrorV3::NodeBudgetExceeded { .. }) => {
                coverage.tactical_budget_exceeded += 1;
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn board_pretraining_record(
    state: &GameState,
    position: &Chess,
    played_move: &ChessMove,
    source: SourcePositionRef,
    config: &Config,
    ruleset: &RulesetRefV3,
    engine: &Engine,
) -> Result<Option<DecisionRecordV3>, Box<dyn std::error::Error>> {
    let standard_move_keys = position
        .legal_moves()
        .iter()
        .filter_map(chess_move_key)
        .collect::<BTreeSet<_>>();
    let played_key = match chess_move_key(played_move) {
        Some(key) => key,
        None => return Ok(None),
    };
    let engine_legal = LegalActionSetV3::from_state(state, engine)?;
    let mut actions = Vec::new();
    let mut played_index = None;
    for index in 0..engine_legal.len() {
        let action = engine_legal.decode_action(index, state)?;
        let Some(key) = augment_action_key(&action) else {
            continue;
        };
        if !standard_move_keys.contains(&key) {
            continue;
        }
        if key == played_key {
            played_index = Some(actions.len());
        }
        actions.push(engine_legal.actions[index].clone());
    }
    let Some(played_index) = played_index else {
        return Ok(None);
    };
    let mut probabilities = vec![0.0; actions.len()];
    probabilities[played_index] = 1.0;
    let legal_actions = LegalActionSetV3 {
        schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
        actions,
    };
    let decision_id = format!("board-{}", sha256_bytes(source.source_id.as_bytes()));
    let record = DecisionRecordV3 {
        record_schema_id: DECISION_RECORD_V3_SCHEMA_ID.to_owned(),
        decision_id,
        provenance: DecisionProvenanceV3 {
            source_kind: DecisionSourceKindV3::ImportedGame,
            source_domain: SourceDomainV3::StandardChess,
            source_id: source.source_id,
            split_group_id: source.split_group_id,
            generator_commit: Some(config.engine_commit.clone()),
            license: config.pgn_license.clone(),
        },
        ruleset: ruleset.clone(),
        observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
        action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
        actor: absolute_player(state.turn),
        state_fingerprint: fingerprint_hex(state.fingerprint()),
        observation: encode_observation_v3(state),
        legal_actions,
        targets: DecisionTargetsV3 {
            policy: Some(DecisionPolicySupervisionV3 {
                scope: PolicyTrainingScopeV3::BoardPretraining,
                target: DecisionPolicyTargetV3::BehaviorDistribution {
                    probabilities,
                    source: "pgn_played_move".to_owned(),
                },
            }),
            value: None,
            auxiliary: Vec::new(),
        },
    };
    record.validate()?;
    Ok(Some(record))
}

struct SourcePositionRef {
    source_id: String,
    split_group_id: String,
}

fn proven_tactical_record(
    state: &GameState,
    source_id: String,
    split_group_id: String,
    source_license: String,
    config: &Config,
    ruleset: &RulesetRefV3,
    engine: &Engine,
) -> Result<Option<DecisionRecordV3>, ForcedWinErrorV3> {
    let Some(solution) = solve_forced_win_v3(
        state,
        engine,
        ForcedWinConfigV3 {
            max_depth: config.tactical_depth,
            max_nodes: config.tactical_nodes,
        },
    )?
    else {
        return Ok(None);
    };
    let legal_actions = LegalActionSetV3::from_state(state, engine)?;
    let proof_source = format!(
        "augmentia.forced-win.v1/depth={}/nodes={}",
        config.tactical_depth, solution.searched_nodes
    );
    let record = DecisionRecordV3 {
        record_schema_id: DECISION_RECORD_V3_SCHEMA_ID.to_owned(),
        decision_id: format!("tactic-{}", sha256_bytes(source_id.as_bytes())),
        provenance: DecisionProvenanceV3 {
            source_kind: DecisionSourceKindV3::EngineSearch,
            source_domain: SourceDomainV3::AugmentChess,
            source_id,
            split_group_id,
            generator_commit: Some(config.engine_commit.clone()),
            license: Some(source_license),
        },
        ruleset: ruleset.clone(),
        observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
        action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
        actor: absolute_player(state.turn),
        state_fingerprint: fingerprint_hex(state.fingerprint()),
        observation: encode_observation_v3(state),
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

#[derive(Clone, Debug)]
struct PgnPly {
    position: Chess,
    played_move: ChessMove,
}

#[derive(Default)]
struct PgnTags {
    start_position: Option<Chess>,
    event: Option<String>,
    site: Option<String>,
    variant: Option<String>,
    termination: Option<String>,
    white_elo: Option<u32>,
    black_elo: Option<u32>,
}

struct PgnMovetext {
    tags: PgnTags,
    position: Chess,
    plies: Vec<PgnPly>,
}

struct ParsedPgnGame {
    event: Option<String>,
    site: Option<String>,
    variant: Option<String>,
    termination: Option<String>,
    white_elo: Option<u32>,
    black_elo: Option<u32>,
    plies: Vec<PgnPly>,
}

struct MainlineVisitor;

impl Visitor for MainlineVisitor {
    type Tags = PgnTags;
    type Movetext = PgnMovetext;
    type Output = Result<ParsedPgnGame, String>;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue(PgnTags::default())
    }

    fn tag(
        &mut self,
        tags: &mut Self::Tags,
        name: &[u8],
        value: RawTag<'_>,
    ) -> ControlFlow<Self::Output> {
        let bytes = value.as_bytes();
        match name {
            b"FEN" => {
                let fen = match Fen::from_ascii(bytes) {
                    Ok(fen) => fen,
                    Err(error) => return ControlFlow::Break(Err(error.to_string())),
                };
                match fen.into_position(CastlingMode::Standard) {
                    Ok(position) => tags.start_position = Some(position),
                    Err(error) => return ControlFlow::Break(Err(error.to_string())),
                }
            }
            b"Event" => tags.event = Some(String::from_utf8_lossy(bytes).into_owned()),
            b"Site" => tags.site = Some(String::from_utf8_lossy(bytes).into_owned()),
            b"Variant" => tags.variant = Some(String::from_utf8_lossy(bytes).into_owned()),
            b"Termination" => {
                tags.termination = Some(String::from_utf8_lossy(bytes).into_owned());
            }
            b"WhiteElo" => tags.white_elo = parse_ascii_u32(bytes),
            b"BlackElo" => tags.black_elo = parse_ascii_u32(bytes),
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        let position = tags.start_position.clone().unwrap_or_default();
        ControlFlow::Continue(PgnMovetext {
            tags,
            position,
            plies: Vec::new(),
        })
    }

    fn san(
        &mut self,
        movetext: &mut Self::Movetext,
        san_plus: SanPlus,
    ) -> ControlFlow<Self::Output> {
        let played_move = match san_plus.san.to_move(&movetext.position) {
            Ok(played_move) => played_move,
            Err(error) => return ControlFlow::Break(Err(error.to_string())),
        };
        movetext.plies.push(PgnPly {
            position: movetext.position.clone(),
            played_move,
        });
        movetext.position.play_unchecked(played_move);
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, movetext: Self::Movetext) -> Self::Output {
        Ok(ParsedPgnGame {
            event: movetext.tags.event,
            site: movetext.tags.site,
            variant: movetext.tags.variant,
            termination: movetext.tags.termination,
            white_elo: movetext.tags.white_elo,
            black_elo: movetext.tags.black_elo,
            plies: movetext.plies,
        })
    }
}

fn parse_ascii_u32(bytes: &[u8]) -> Option<u32> {
    std::str::from_utf8(bytes).ok()?.parse().ok()
}

fn pgn_filter_rejection(game: &ParsedPgnGame, config: &Config) -> Option<&'static str> {
    if game
        .variant
        .as_deref()
        .is_some_and(|variant| !variant.eq_ignore_ascii_case("standard"))
    {
        return Some("non_standard_variant");
    }
    if config.rated_only
        && !game
            .event
            .as_deref()
            .is_some_and(|event| event.to_ascii_lowercase().contains("rated"))
    {
        return Some("not_rated");
    }
    if config.min_rating > 0
        && (game
            .white_elo
            .is_none_or(|rating| rating < config.min_rating)
            || game
                .black_elo
                .is_none_or(|rating| rating < config.min_rating))
    {
        return Some("rating_below_minimum_or_missing");
    }
    if !config.allowed_speeds.is_empty() {
        let speed = game.event.as_deref().and_then(lichess_speed_from_event);
        if speed.is_none_or(|speed| !config.allowed_speeds.contains(&speed)) {
            return Some("speed_not_allowed_or_missing");
        }
    }
    if config.normal_termination_only
        && !game
            .termination
            .as_deref()
            .is_some_and(|termination| termination.eq_ignore_ascii_case("normal"))
    {
        return Some("termination_not_normal_or_missing");
    }
    if game.plies.len() < config.min_game_plies {
        return Some("game_too_short");
    }
    None
}

fn lichess_speed_from_event(event: &str) -> Option<String> {
    event
        .split(|character: char| !character.is_ascii_alphanumeric())
        .find_map(|word| {
            let word = word.to_ascii_lowercase();
            matches!(
                word.as_str(),
                "ultrabullet" | "bullet" | "blitz" | "rapid" | "classical" | "correspondence"
            )
            .then_some(word)
        })
}

fn standard_position_to_augment_state(
    position: &Chess,
) -> Result<GameState, Box<dyn std::error::Error>> {
    let mut board = Board::empty();
    for (square, piece) in position.board().iter() {
        board.set(
            square.to_string().parse::<Square>()?,
            Some(Piece::new(
                match piece.color {
                    Color::White => Player::White,
                    Color::Black => Player::Black,
                },
                role_to_piece_kind(piece.role),
            )),
        );
    }
    let turn = match position.turn() {
        Color::White => Player::White,
        Color::Black => Player::Black,
    };
    let mut state = GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, turn, 0);
    let completed_fullmoves = position.fullmoves().get().saturating_sub(1);
    state.ply = completed_fullmoves
        .saturating_mul(2)
        .saturating_add(u32::from(turn == Player::Black));
    state.quiet_ply = position.halfmoves();
    state.turns_taken = if turn == Player::White {
        [completed_fullmoves, completed_fullmoves]
    } else {
        [completed_fullmoves.saturating_add(1), completed_fullmoves]
    };
    state.completed_drafts = [true; 3];
    if let Some(target) = position.legal_ep_square() {
        let target = target.to_string().parse::<Square>()?;
        let captured_index = if turn == Player::White {
            target.index().checked_sub(8)
        } else {
            target.index().checked_add(8).filter(|index| *index < 64)
        };
        if let Some(captured_index) = captured_index {
            state.en_passant = Some(EnPassantState {
                target,
                captured: Square::from_index(captured_index as u8)
                    .expect("checked en passant square"),
                pawn_owner: turn.opponent(),
            });
        }
    }
    Ok(state)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct MoveKey {
    from: Square,
    to: Square,
    promotion: Option<PieceKind>,
}

fn chess_move_key(chess_move: &ChessMove) -> Option<MoveKey> {
    match chess_move {
        ChessMove::Normal {
            from,
            to,
            promotion,
            ..
        } => Some(MoveKey {
            from: from.to_string().parse().ok()?,
            to: to.to_string().parse().ok()?,
            promotion: promotion.map(role_to_piece_kind),
        }),
        ChessMove::EnPassant { from, to } => Some(MoveKey {
            from: from.to_string().parse().ok()?,
            to: to.to_string().parse().ok()?,
            promotion: None,
        }),
        ChessMove::Castle { king, rook } => {
            let from: Square = king.to_string().parse().ok()?;
            let rook: Square = rook.to_string().parse().ok()?;
            let rank = from.rank();
            let file = if rook.file() > from.file() { 6 } else { 2 };
            Some(MoveKey {
                from,
                to: Square::new(file, rank)?,
                promotion: None,
            })
        }
        ChessMove::Put { .. } => None,
    }
}

fn augment_action_key(action: &Action) -> Option<MoveKey> {
    match action {
        Action::Move(action) => Some(MoveKey {
            from: action.from,
            to: action.to,
            promotion: action.promotion,
        }),
        _ => None,
    }
}

fn role_to_piece_kind(role: Role) -> PieceKind {
    match role {
        Role::Pawn => PieceKind::Pawn,
        Role::Knight => PieceKind::Knight,
        Role::Bishop => PieceKind::Bishop,
        Role::Rook => PieceKind::Rook,
        Role::Queen => PieceKind::Queen,
        Role::King => PieceKind::King,
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TacticalInputV1 {
    schema_id: String,
    scenarios: Vec<TacticalScenarioV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TacticalScenarioV1 {
    id: String,
    group_id: String,
    seed_hex: String,
    turn: AbsolutePlayerV3,
    pieces: Vec<TacticalPieceV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TacticalPieceV1 {
    square: String,
    owner: AbsolutePlayerV3,
    kind_code: u8,
}

impl TacticalScenarioV1 {
    fn to_state(&self) -> Result<GameState, Box<dyn std::error::Error>> {
        let seed = u64::from_str_radix(&self.seed_hex, 16)?;
        let mut board = Board::empty();
        for piece in &self.pieces {
            let kind = PIECE_KIND_VOCABULARY
                .iter()
                .find_map(|(_, kind)| (kind.stable_code() == piece.kind_code).then_some(*kind))
                .ok_or_else(|| format!("unknown piece kind code {}", piece.kind_code))?;
            let square = piece.square.parse::<Square>()?;
            if board
                .set(
                    square,
                    Some(Piece::new(relative_to_absolute(piece.owner), kind)),
                )
                .is_some()
            {
                return Err(format!("duplicate scenario square {}", piece.square).into());
            }
        }
        Ok(GameState::from_board(
            BUNDLED_RULESET_ID,
            Mode::Standard,
            board,
            relative_to_absolute(self.turn),
            seed,
        ))
    }
}

#[derive(Debug, Serialize)]
struct BootstrapCoverageV3 {
    schema_id: &'static str,
    dataset_id: String,
    pgn_sha256: Option<String>,
    pgn_input_bytes: Option<u64>,
    pgn_compression: Option<String>,
    tactical_input_sha256: Option<String>,
    games_read: u64,
    games_accepted: u64,
    games_filtered: u64,
    game_filter_reasons: BTreeMap<String, u64>,
    invalid_games: u64,
    positions_seen: u64,
    board_pretraining_records: u64,
    proven_tactical_records: u64,
    unproven_tactical_positions: u64,
    tactical_budget_exceeded: u64,
    skipped_positions: u64,
    duplicates_removed: u64,
    output_shards: u64,
    output_compressed_bytes: u64,
    generation_elapsed_ms: u64,
    records_per_second: f64,
    action_kinds: BTreeMap<String, u64>,
    phase_kinds: BTreeMap<String, u64>,
    piece_kind_codes: BTreeMap<u8, u64>,
    card_definition_ids: BTreeMap<String, u64>,
    limitations: Vec<&'static str>,
}

impl BootstrapCoverageV3 {
    fn new(config: &Config) -> Self {
        Self {
            schema_id: COVERAGE_SCHEMA_ID,
            dataset_id: config.dataset_id.clone(),
            pgn_sha256: None,
            pgn_input_bytes: None,
            pgn_compression: None,
            tactical_input_sha256: None,
            games_read: 0,
            games_accepted: 0,
            games_filtered: 0,
            game_filter_reasons: BTreeMap::new(),
            invalid_games: 0,
            positions_seen: 0,
            board_pretraining_records: 0,
            proven_tactical_records: 0,
            unproven_tactical_positions: 0,
            tactical_budget_exceeded: 0,
            skipped_positions: 0,
            duplicates_removed: 0,
            output_shards: 0,
            output_compressed_bytes: 0,
            generation_elapsed_ms: 0,
            records_per_second: 0.0,
            action_kinds: BTreeMap::new(),
            phase_kinds: BTreeMap::new(),
            piece_kind_codes: BTreeMap::new(),
            card_definition_ids: BTreeMap::new(),
            limitations: vec![
                "standard chess records supervise board_pretraining only",
                "standard chess behavior frequency is preserved by repeated one-hot samples",
                "PGN positions are not scanned by the Augmentia forced-win solver",
                "tactical-input.v1 represents board pieces but not cards or runtime state",
                "zero card coverage is reported, never inferred",
            ],
        }
    }

    fn observe_filtered_game(&mut self, reason: &str) {
        self.games_filtered += 1;
        *self
            .game_filter_reasons
            .entry(reason.to_owned())
            .or_default() += 1;
    }

    fn observe_record(&mut self, record: &DecisionRecordV3) {
        let phase = match record.observation.phase {
            augment_ai::PhaseV3::Playing => "playing",
            augment_ai::PhaseV3::ForcedMove { .. } => "forced_move",
            augment_ai::PhaseV3::Draft { .. } => "draft",
            augment_ai::PhaseV3::AwaitingChoice { .. } => "awaiting_choice",
        };
        *self.phase_kinds.entry(phase.to_owned()).or_default() += 1;
        for piece in &record.observation.pieces {
            *self.piece_kind_codes.entry(piece.kind_code).or_default() += 1;
        }
        for card in &record.observation.cards {
            *self
                .card_definition_ids
                .entry(card.definition_id.clone())
                .or_default() += 1;
        }
        for action in &record.legal_actions.actions {
            let kind = match action {
                ActionDescriptorV3::Move { .. } => "move",
                ActionDescriptorV3::PieceAbility { .. } => "piece_ability",
                ActionDescriptorV3::Card { .. } => "card",
                ActionDescriptorV3::DraftPick { .. } => "draft_pick",
                ActionDescriptorV3::ResolveChoice { .. } => "resolve_choice",
            };
            *self.action_kinds.entry(kind.to_owned()).or_default() += 1;
        }
    }
}

struct BootstrapShardWriter<'a> {
    output_dir: &'a Path,
    split_policy: SplitPolicyV3,
    shard_records: usize,
    compression_level: i32,
    buffers: BTreeMap<DatasetSplitV3, Vec<DataRecordV3>>,
    next_shard: BTreeMap<DatasetSplitV3, u32>,
    shards: Vec<ShardManifestV3>,
    total_records: u64,
}

impl<'a> BootstrapShardWriter<'a> {
    fn new(
        output_dir: &'a Path,
        split_policy: SplitPolicyV3,
        shard_records: usize,
        compression_level: i32,
    ) -> Self {
        Self {
            output_dir,
            split_policy,
            shard_records,
            compression_level,
            buffers: BTreeMap::new(),
            next_shard: BTreeMap::new(),
            shards: Vec::new(),
            total_records: 0,
        }
    }

    fn total_records(&self) -> u64 {
        self.total_records
    }

    fn push(&mut self, record: DataRecordV3) -> Result<(), Box<dyn std::error::Error>> {
        let split = self.split_policy.assign(&record.split_group_key())?;
        let should_flush = {
            let buffer = self.buffers.entry(split).or_default();
            buffer.push(record);
            buffer.len() >= self.shard_records
        };
        self.total_records += 1;
        if should_flush {
            self.flush_split(split)?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<ShardManifestV3>, Box<dyn std::error::Error>> {
        for split in [
            DatasetSplitV3::Train,
            DatasetSplitV3::Validation,
            DatasetSplitV3::Holdout,
        ] {
            self.flush_split(split)?;
        }
        self.shards
            .sort_by_key(|shard| (shard.split, shard.path.clone()));
        Ok(self.shards)
    }

    fn flush_split(&mut self, split: DatasetSplitV3) -> Result<(), Box<dyn std::error::Error>> {
        let Some(records) = self.buffers.get_mut(&split) else {
            return Ok(());
        };
        if records.is_empty() {
            return Ok(());
        }
        let shard_index = self.next_shard.entry(split).or_default();
        let relative = format!(
            "shards/{}/decision-{shard_index:05}.jsonl.zst",
            split_name(split)
        );
        let summary = write_jsonl_zstd_shard(
            self.output_dir.join(Path::new(&relative)),
            records,
            self.compression_level,
        )?;
        self.shards.push(summary.to_manifest_entry(relative, split));
        records.clear();
        *shard_index += 1;
        Ok(())
    }
}

fn emit_record(
    record: DataRecordV3,
    writer: &mut BootstrapShardWriter<'_>,
    coverage: &mut BootstrapCoverageV3,
) -> Result<(), Box<dyn std::error::Error>> {
    if let DataRecordV3::Decision(decision) = &record {
        match decision.provenance.source_domain {
            SourceDomainV3::StandardChess => coverage.board_pretraining_records += 1,
            SourceDomainV3::AugmentChess => coverage.proven_tactical_records += 1,
        }
        coverage.observe_record(decision);
    }
    writer.push(record)
}

fn decision_training_digest(record: &DecisionRecordV3) -> Result<String, serde_json::Error> {
    #[derive(Serialize)]
    struct DecisionContent<'a> {
        ruleset: &'a RulesetRefV3,
        observation_schema_id: &'a str,
        action_schema_id: &'a str,
        actor: AbsolutePlayerV3,
        observation: &'a augment_ai::ObservationV3,
        legal_actions: &'a LegalActionSetV3,
        targets: &'a DecisionTargetsV3,
    }

    serde_json::to_vec(&DecisionContent {
        ruleset: &record.ruleset,
        observation_schema_id: &record.observation_schema_id,
        action_schema_id: &record.action_schema_id,
        actor: record.actor,
        observation: &record.observation,
        legal_actions: &record.legal_actions,
        targets: &record.targets,
    })
    .map(|bytes| sha256_bytes(&bytes))
}

fn totals_from_shards(shards: &[ShardManifestV3]) -> DatasetTotalsV3 {
    let mut totals = DatasetTotalsV3 {
        shards: shards.len() as u64,
        ..DatasetTotalsV3::default()
    };
    for shard in shards {
        totals.records += shard.records;
        totals.games += shard.games;
        totals.decisions += shard.decisions;
        totals.game_steps += shard.game_steps;
        totals.policy_eligible_decisions += shard.policy_eligible_decisions;
        totals.value_eligible_decisions += shard.value_eligible_decisions;
    }
    totals
}

fn manifest_rate(records: u64, elapsed_seconds: f64) -> f64 {
    records as f64 / elapsed_seconds.max(f64::EPSILON)
}

fn split_name(split: DatasetSplitV3) -> &'static str {
    match split {
        DatasetSplitV3::Train => "train",
        DatasetSplitV3::Validation => "validation",
        DatasetSplitV3::Holdout => "holdout",
    }
}

fn absolute_player(player: Player) -> AbsolutePlayerV3 {
    match player {
        Player::White => AbsolutePlayerV3::White,
        Player::Black => AbsolutePlayerV3::Black,
    }
}

fn relative_to_absolute(player: AbsolutePlayerV3) -> Player {
    match player {
        AbsolutePlayerV3::White => Player::White,
        AbsolutePlayerV3::Black => Player::Black,
    }
}

fn write_new_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn is_zstd_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("zst"))
}

fn open_pgn_input(path: &Path) -> Result<Box<dyn Read>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    if is_zstd_path(path) {
        Ok(Box::new(zstd::stream::read::Decoder::new(file)?))
    } else {
        Ok(Box::new(file))
    }
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(bytes_to_hex(&hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    bytes_to_hex(&Sha256::digest(bytes))
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn required_arg(args: &[String], name: &str) -> Result<String, String> {
    optional_arg(args, name).ok_or_else(|| format!("missing required argument {name}"))
}

fn optional_arg(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|argument| argument == name)
}

fn parse_speeds(value: &str) -> Result<BTreeSet<String>, String> {
    let mut speeds = BTreeSet::new();
    for speed in value.split(',') {
        let speed = speed.trim().to_ascii_lowercase();
        if !matches!(
            speed.as_str(),
            "ultrabullet" | "bullet" | "blitz" | "rapid" | "classical" | "correspondence"
        ) {
            return Err(format!(
                "invalid --speeds value {speed}; expected comma-separated Lichess speed names"
            ));
        }
        speeds.insert(speed);
    }
    Ok(speeds)
}

fn parse_arg<T>(args: &[String], name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    optional_arg(args, name).map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|error| format!("invalid {name}: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("augmentia-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn imports_pgn_only_as_board_pretraining() {
        let root = temp_dir("bootstrap-pgn");
        std::fs::create_dir_all(&root).unwrap();
        let pgn = root.join("tiny.pgn");
        let fixture = include_str!("../../../../train/v3/fixtures/bootstrap_standard_v1.pgn");
        std::fs::write(
            &pgn,
            format!(
                "{}\n{}",
                fixture,
                fixture.replace("[Site \"local\"]", "[Site \"local-2\"]")
            ),
        )
        .unwrap();
        let ruleset_manifest = root.join("ruleset.json");
        std::fs::write(&ruleset_manifest, "{}\n").unwrap();
        let output = root.join("dataset");
        run(vec![
            "--pgn".to_owned(),
            pgn.display().to_string(),
            "--out".to_owned(),
            output.display().to_string(),
            "--dataset-id".to_owned(),
            "test-pgn".to_owned(),
            "--created-at-utc".to_owned(),
            "2026-08-21T00:00:00Z".to_owned(),
            "--engine-commit".to_owned(),
            "a".repeat(40),
            "--pgn-license".to_owned(),
            "test-only".to_owned(),
            "--ruleset-manifest".to_owned(),
            ruleset_manifest.display().to_string(),
            "--tactical-depth".to_owned(),
            "1".to_owned(),
        ])
        .unwrap();

        let coverage: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output.join("coverage.json")).unwrap()).unwrap();
        assert_eq!(coverage["board_pretraining_records"], 10);
        assert_eq!(coverage["duplicates_removed"], 0);
        let manifest: DatasetManifestV3 =
            serde_json::from_slice(&std::fs::read(output.join("manifest.json")).unwrap()).unwrap();
        for shard in manifest.shards {
            for record in augment_ai::read_jsonl_zstd_shard(output.join(shard.path)).unwrap() {
                let DataRecordV3::Decision(record) = record else {
                    panic!("bootstrap shard must contain decisions");
                };
                assert_eq!(
                    record.provenance.source_domain,
                    SourceDomainV3::StandardChess
                );
                assert!(matches!(
                    record.targets.policy,
                    Some(DecisionPolicySupervisionV3 {
                        scope: PolicyTrainingScopeV3::BoardPretraining,
                        ..
                    })
                ));
                assert!(record.targets.value.is_none());
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn streams_zstd_pgn_and_applies_lichess_quality_filters() {
        let root = temp_dir("bootstrap-lichess-zstd");
        std::fs::create_dir_all(&root).unwrap();
        let pgn = root.join("lichess.pgn.zst");
        let games = r#"[Event "rated rapid game"]
[Site "https://lichess.org/highrated"]
[Result "*"]
[WhiteElo "2250"]
[BlackElo "2100"]
[Variant "Standard"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 *

[Event "rated rapid game"]
[Site "https://lichess.org/lowrated"]
[Result "*"]
[WhiteElo "1900"]
[BlackElo "2200"]
[Variant "Standard"]
[Termination "Normal"]

1. d4 d5 2. c4 e6 3. Nc3 *
"#;
        std::fs::write(&pgn, zstd::stream::encode_all(games.as_bytes(), 1).unwrap()).unwrap();
        let ruleset_manifest = root.join("ruleset.json");
        std::fs::write(&ruleset_manifest, "{}\n").unwrap();
        let output = root.join("dataset");
        run(vec![
            "--pgn".to_owned(),
            pgn.display().to_string(),
            "--out".to_owned(),
            output.display().to_string(),
            "--dataset-id".to_owned(),
            "test-lichess-zstd".to_owned(),
            "--created-at-utc".to_owned(),
            "2026-08-21T00:00:00Z".to_owned(),
            "--engine-commit".to_owned(),
            "c".repeat(40),
            "--pgn-license".to_owned(),
            "CC0-1.0".to_owned(),
            "--ruleset-manifest".to_owned(),
            ruleset_manifest.display().to_string(),
            "--min-rating".to_owned(),
            "2000".to_owned(),
            "--speeds".to_owned(),
            "rapid,classical".to_owned(),
            "--rated-only".to_owned(),
            "--normal-termination-only".to_owned(),
            "--min-game-plies".to_owned(),
            "5".to_owned(),
            "--shard-records".to_owned(),
            "2".to_owned(),
        ])
        .unwrap();

        let coverage: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output.join("coverage.json")).unwrap()).unwrap();
        assert_eq!(coverage["pgn_compression"], "zstd");
        assert_eq!(coverage["games_read"], 2);
        assert_eq!(coverage["games_accepted"], 1);
        assert_eq!(coverage["games_filtered"], 1);
        assert_eq!(
            coverage["game_filter_reasons"]["rating_below_minimum_or_missing"],
            1
        );
        assert_eq!(coverage["board_pretraining_records"], 5);
        assert_eq!(coverage["unproven_tactical_positions"], 0);
        assert_eq!(coverage["output_shards"], 3);
        assert!(coverage["output_compressed_bytes"].as_u64().unwrap() > 0);
        assert!(coverage["records_per_second"].as_f64().unwrap() > 0.0);

        let manifest: DatasetManifestV3 =
            serde_json::from_slice(&std::fs::read(output.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest.totals.records, 5);
        assert_eq!(manifest.shards.len(), 3);
        assert!(manifest.shards.iter().all(|shard| shard.records <= 2));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn keeps_broadcast_games_with_a_shared_site_in_distinct_groups() {
        let root = temp_dir("bootstrap-broadcast-shared-site");
        std::fs::create_dir_all(&root).unwrap();
        let pgn = root.join("broadcast.pgn.zst");
        let games = r#"[Event "Broadcast"]
[Site "https://lichess.org/broadcast/shared"]
[Result "*"]
[WhiteElo "2200"]
[BlackElo "2200"]
[Variant "Standard"]

1. e4 e5 *

[Event "Broadcast"]
[Site "https://lichess.org/broadcast/shared"]
[Result "*"]
[WhiteElo "2200"]
[BlackElo "2200"]
[Variant "Standard"]

1. e4 e5 *
"#;
        std::fs::write(&pgn, zstd::stream::encode_all(games.as_bytes(), 1).unwrap()).unwrap();
        let ruleset_manifest = root.join("ruleset.json");
        std::fs::write(&ruleset_manifest, "{}\n").unwrap();
        let output = root.join("dataset");
        run(vec![
            "--pgn".to_owned(),
            pgn.display().to_string(),
            "--out".to_owned(),
            output.display().to_string(),
            "--dataset-id".to_owned(),
            "test-broadcast-shared-site".to_owned(),
            "--created-at-utc".to_owned(),
            "2026-08-21T00:00:00Z".to_owned(),
            "--engine-commit".to_owned(),
            "d".repeat(40),
            "--pgn-license".to_owned(),
            "test-only".to_owned(),
            "--ruleset-manifest".to_owned(),
            ruleset_manifest.display().to_string(),
        ])
        .unwrap();

        let manifest: DatasetManifestV3 =
            serde_json::from_slice(&std::fs::read(output.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest.totals.records, 4);
        let mut ids = BTreeSet::new();
        let mut groups = BTreeSet::new();
        for shard in manifest.shards {
            for record in augment_ai::read_jsonl_zstd_shard(output.join(shard.path)).unwrap() {
                let DataRecordV3::Decision(record) = record else {
                    panic!("bootstrap shard must contain decisions");
                };
                assert!(ids.insert(record.decision_id.clone()));
                groups.insert(record.provenance.split_group_id.clone());
            }
        }
        assert_eq!(groups.len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_a_pgn_that_does_not_match_its_locked_identity() {
        let root = temp_dir("bootstrap-pgn-identity");
        std::fs::create_dir_all(&root).unwrap();
        let pgn = root.join("tiny.pgn");
        std::fs::write(
            &pgn,
            include_bytes!("../../../../train/v3/fixtures/bootstrap_standard_v1.pgn"),
        )
        .unwrap();
        let ruleset_manifest = root.join("ruleset.json");
        std::fs::write(&ruleset_manifest, "{}\n").unwrap();
        let error = run(vec![
            "--pgn".to_owned(),
            pgn.display().to_string(),
            "--out".to_owned(),
            root.join("dataset").display().to_string(),
            "--dataset-id".to_owned(),
            "test-pgn-identity".to_owned(),
            "--created-at-utc".to_owned(),
            "2026-08-21T00:00:00Z".to_owned(),
            "--engine-commit".to_owned(),
            "e".repeat(40),
            "--pgn-license".to_owned(),
            "test-only".to_owned(),
            "--expected-pgn-sha256".to_owned(),
            "0".repeat(64),
            "--expected-pgn-bytes".to_owned(),
            std::fs::metadata(&pgn).unwrap().len().to_string(),
            "--ruleset-manifest".to_owned(),
            ruleset_manifest.display().to_string(),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("PGN SHA-256 mismatch"));
        assert!(!root.join("dataset/manifest.json").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn emits_only_engine_proven_tactical_scenarios() {
        let root = temp_dir("bootstrap-tactics");
        std::fs::create_dir_all(&root).unwrap();
        let tactics = root.join("tactics.json");
        let mut tactical_input: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../train/v3/fixtures/bootstrap_tactics_v1.json"
        ))
        .unwrap();
        let mut duplicate = tactical_input["scenarios"][0].clone();
        duplicate["id"] = "duplicate-proven-position".into();
        duplicate["group_id"] = "duplicate-group".into();
        tactical_input["scenarios"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        std::fs::write(&tactics, serde_json::to_vec(&tactical_input).unwrap()).unwrap();
        let ruleset_manifest = root.join("ruleset.json");
        std::fs::write(&ruleset_manifest, "{}\n").unwrap();
        let output = root.join("dataset");
        run(vec![
            "--tactics".to_owned(),
            tactics.display().to_string(),
            "--out".to_owned(),
            output.display().to_string(),
            "--dataset-id".to_owned(),
            "test-tactics".to_owned(),
            "--created-at-utc".to_owned(),
            "2026-08-21T00:00:00Z".to_owned(),
            "--engine-commit".to_owned(),
            "b".repeat(40),
            "--tactics-license".to_owned(),
            "test-only".to_owned(),
            "--ruleset-manifest".to_owned(),
            ruleset_manifest.display().to_string(),
            "--tactical-depth".to_owned(),
            "1".to_owned(),
        ])
        .unwrap();
        let coverage: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output.join("coverage.json")).unwrap()).unwrap();
        assert_eq!(coverage["proven_tactical_records"], 1);
        assert_eq!(coverage["board_pretraining_records"], 0);
        assert_eq!(coverage["duplicates_removed"], 1);
        std::fs::remove_dir_all(root).unwrap();
    }
}

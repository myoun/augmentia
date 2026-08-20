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

const COVERAGE_SCHEMA_ID: &str = "augmentia.bootstrap.coverage.v3";
const TACTICAL_INPUT_SCHEMA_ID: &str = "augmentia.bootstrap.tactical-input.v1";

fn main() {
    if let Err(error) = run(std::env::args().skip(1).collect()) {
        eprintln!("V3 bootstrap generation failed: {error}");
        std::process::exit(1);
    }
}

fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
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

    let mut records = Vec::new();
    let mut coverage = BootstrapCoverageV3::new(&config);
    if let Some(pgn_path) = &config.pgn {
        import_pgn(
            pgn_path,
            &config,
            &ruleset,
            &engine,
            &mut records,
            &mut coverage,
        )?;
    }
    if let Some(tactics_path) = &config.tactics {
        import_tactical_scenarios(
            tactics_path,
            &config,
            &ruleset,
            &engine,
            &mut records,
            &mut coverage,
        )?;
    }
    records = deduplicate_records(records, &mut coverage)?;
    if records.is_empty() {
        return Err("no valid bootstrap records were generated".into());
    }

    let split_policy = SplitPolicyV3 {
        algorithm: SPLIT_POLICY_V3_ALGORITHM.to_owned(),
        salt: config.split_salt.clone(),
        train_basis_points: 9_000,
        validation_basis_points: 500,
        holdout_basis_points: 500,
    };
    let mut by_split = BTreeMap::<DatasetSplitV3, Vec<DataRecordV3>>::new();
    for record in records {
        let split = split_policy.assign(&record.split_group_key())?;
        by_split.entry(split).or_default().push(record);
    }

    let mut shards = Vec::new();
    for (split, records) in by_split {
        let relative = format!("shards/{}/decision-00000.jsonl.zst", split_name(split));
        let summary = write_jsonl_zstd_shard(
            config.output_dir.join(Path::new(&relative)),
            &records,
            config.compression_level,
        )?;
        shards.push(summary.to_manifest_entry(relative, split));
    }
    shards.sort_by_key(|shard| shard.split);
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
    write_new_json(&config.output_dir.join("manifest.json"), &manifest)?;
    write_new_json(&config.output_dir.join("coverage.json"), &coverage)?;
    verify_manifest_files(&config.output_dir, &manifest)?;
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
    tactics_license: Option<String>,
    ruleset_manifest: PathBuf,
    split_salt: String,
    max_games: usize,
    max_records: usize,
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
        let tactics_license = match &tactics {
            Some(_) => Some(required_arg(args, "--tactics-license")?),
            None => None,
        };
        Ok(Self {
            pgn,
            tactics,
            output_dir: PathBuf::from(required_arg(args, "--out")?),
            dataset_id,
            created_at_utc,
            engine_commit,
            pgn_license,
            tactics_license,
            ruleset_manifest: PathBuf::from(
                optional_arg(args, "--ruleset-manifest")
                    .unwrap_or_else(|| "rulesets/augment_chess_20260807/manifest.json".to_owned()),
            ),
            split_salt: optional_arg(args, "--split-salt")
                .unwrap_or_else(|| "augmentia-bootstrap-v3".to_owned()),
            max_games: parse_arg(args, "--max-games", 1_000usize)?,
            max_records: parse_arg(args, "--max-records", 100_000usize)?,
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
    records: &mut Vec<DataRecordV3>,
    coverage: &mut BootstrapCoverageV3,
) -> Result<(), Box<dyn std::error::Error>> {
    let source_sha256 = sha256_file(path)?;
    coverage.pgn_sha256 = Some(source_sha256.clone());
    let file = File::open(path)?;
    let mut reader = Reader::new(BufReader::new(file));
    let mut visitor = MainlineVisitor;
    let mut game_index = 0usize;

    while game_index < config.max_games && records.len() < config.max_records {
        let Some(game) = reader.read_game(&mut visitor)? else {
            break;
        };
        coverage.games_read += 1;
        match game {
            Ok(plies) => {
                for (ply_index, ply) in plies.into_iter().enumerate() {
                    if records.len() >= config.max_records {
                        break;
                    }
                    coverage.positions_seen += 1;
                    let source_id =
                        format!("pgn:{source_sha256}:game:{game_index}:ply:{ply_index}");
                    let split_group_id = format!("standard:{source_sha256}:game:{game_index}");
                    let state = standard_position_to_augment_state(&ply.position)?;
                    if let Some(record) = board_pretraining_record(
                        &state,
                        &ply.position,
                        &ply.played_move,
                        SourcePositionRef {
                            source_id: source_id.clone(),
                            split_group_id: split_group_id.clone(),
                        },
                        config,
                        ruleset,
                        engine,
                    )? {
                        records.push(DataRecordV3::Decision(Box::new(record)));
                    } else {
                        coverage.skipped_positions += 1;
                    }
                    if records.len() >= config.max_records {
                        break;
                    }
                    match proven_tactical_record(
                        &state,
                        format!("{source_id}:augment-proof"),
                        split_group_id,
                        config.pgn_license.clone().expect("required with --pgn"),
                        config,
                        ruleset,
                        engine,
                    ) {
                        Ok(Some(record)) => {
                            records.push(DataRecordV3::Decision(Box::new(record)));
                        }
                        Ok(None) => coverage.unproven_tactical_positions += 1,
                        Err(ForcedWinErrorV3::NodeBudgetExceeded { .. }) => {
                            coverage.tactical_budget_exceeded += 1;
                        }
                        Err(error) => return Err(error.into()),
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
    records: &mut Vec<DataRecordV3>,
    coverage: &mut BootstrapCoverageV3,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let input: TacticalInputV1 = serde_json::from_slice(&bytes)?;
    if input.schema_id != TACTICAL_INPUT_SCHEMA_ID {
        return Err(format!("unsupported tactical input schema {}", input.schema_id).into());
    }
    let source_sha256 = sha256_bytes(&bytes);
    coverage.tactical_input_sha256 = Some(source_sha256.clone());
    for scenario in input.scenarios {
        if records.len() >= config.max_records {
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
                records.push(DataRecordV3::Decision(Box::new(record)));
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

struct MainlineVisitor;

impl Visitor for MainlineVisitor {
    type Tags = Option<Chess>;
    type Movetext = (Chess, Vec<PgnPly>);
    type Output = Result<Vec<PgnPly>, String>;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue(None)
    }

    fn tag(
        &mut self,
        tags: &mut Self::Tags,
        name: &[u8],
        value: RawTag<'_>,
    ) -> ControlFlow<Self::Output> {
        if name == b"FEN" {
            let fen = match Fen::from_ascii(value.as_bytes()) {
                Ok(fen) => fen,
                Err(error) => return ControlFlow::Break(Err(error.to_string())),
            };
            match fen.into_position(CastlingMode::Standard) {
                Ok(position) => *tags = Some(position),
                Err(error) => return ControlFlow::Break(Err(error.to_string())),
            }
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        ControlFlow::Continue((tags.unwrap_or_default(), Vec::new()))
    }

    fn san(
        &mut self,
        movetext: &mut Self::Movetext,
        san_plus: SanPlus,
    ) -> ControlFlow<Self::Output> {
        let played_move = match san_plus.san.to_move(&movetext.0) {
            Ok(played_move) => played_move,
            Err(error) => return ControlFlow::Break(Err(error.to_string())),
        };
        movetext.1.push(PgnPly {
            position: movetext.0.clone(),
            played_move,
        });
        movetext.0.play_unchecked(played_move);
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, movetext: Self::Movetext) -> Self::Output {
        Ok(movetext.1)
    }
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
    tactical_input_sha256: Option<String>,
    games_read: u64,
    invalid_games: u64,
    positions_seen: u64,
    board_pretraining_records: u64,
    proven_tactical_records: u64,
    unproven_tactical_positions: u64,
    tactical_budget_exceeded: u64,
    skipped_positions: u64,
    duplicates_removed: u64,
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
            tactical_input_sha256: None,
            games_read: 0,
            invalid_games: 0,
            positions_seen: 0,
            board_pretraining_records: 0,
            proven_tactical_records: 0,
            unproven_tactical_positions: 0,
            tactical_budget_exceeded: 0,
            skipped_positions: 0,
            duplicates_removed: 0,
            action_kinds: BTreeMap::new(),
            phase_kinds: BTreeMap::new(),
            piece_kind_codes: BTreeMap::new(),
            card_definition_ids: BTreeMap::new(),
            limitations: vec![
                "standard chess records supervise board_pretraining only",
                "tactical-input.v1 represents board pieces but not cards or runtime state",
                "zero card coverage is reported, never inferred",
            ],
        }
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

fn deduplicate_records(
    records: Vec<DataRecordV3>,
    coverage: &mut BootstrapCoverageV3,
) -> Result<Vec<DataRecordV3>, serde_json::Error> {
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

    let mut seen = BTreeSet::new();
    let mut unique = Vec::with_capacity(records.len());
    for record in records {
        let content = match &record {
            DataRecordV3::Decision(decision) => serde_json::to_vec(&DecisionContent {
                ruleset: &decision.ruleset,
                observation_schema_id: &decision.observation_schema_id,
                action_schema_id: &decision.action_schema_id,
                actor: decision.actor,
                observation: &decision.observation,
                legal_actions: &decision.legal_actions,
                targets: &decision.targets,
            })?,
            DataRecordV3::Game(game) => serde_json::to_vec(game)?,
        };
        if seen.insert(sha256_bytes(&content)) {
            unique.push(record);
        } else {
            coverage.duplicates_removed += 1;
        }
    }
    coverage.board_pretraining_records = 0;
    coverage.proven_tactical_records = 0;
    coverage.action_kinds.clear();
    coverage.phase_kinds.clear();
    coverage.piece_kind_codes.clear();
    coverage.card_definition_ids.clear();
    for record in &unique {
        if let DataRecordV3::Decision(decision) = record {
            match decision.provenance.source_domain {
                SourceDomainV3::StandardChess => coverage.board_pretraining_records += 1,
                SourceDomainV3::AugmentChess => coverage.proven_tactical_records += 1,
            }
            coverage.observe_record(decision);
        }
    }
    Ok(unique)
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
        std::fs::write(
            &pgn,
            format!(
                "{}\n{}",
                include_str!("../../../../train/v3/fixtures/bootstrap_standard_v1.pgn"),
                include_str!("../../../../train/v3/fixtures/bootstrap_standard_v1.pgn")
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
        assert_eq!(coverage["board_pretraining_records"], 5);
        assert_eq!(coverage["duplicates_removed"], 5);
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
    fn emits_only_engine_proven_tactical_scenarios() {
        let root = temp_dir("bootstrap-tactics");
        std::fs::create_dir_all(&root).unwrap();
        let tactics = root.join("tactics.json");
        std::fs::write(
            &tactics,
            include_str!("../../../../train/v3/fixtures/bootstrap_tactics_v1.json"),
        )
        .unwrap();
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
        std::fs::remove_dir_all(root).unwrap();
    }
}

use augment_core::{
    Action, Board, Engine, GameState, GameStatus, Mode, Piece, PieceKind, Player, Square,
};
use augment_search::arena::{self, ArenaConfig};
use augment_search::tactics::{TacticalConfig, find_tactical_win};
use augment_search::teacher;
use augment_search::{DepthUnit, Evaluation, SearchConfig, search};
use serde_json::{Value, json};
use std::time::Duration;

fn board_state(engine: &Engine, pieces: &[(&str, Player, PieceKind)]) -> GameState {
    let mut board = Board::empty();
    for (square, owner, kind) in pieces {
        board.set(
            square.parse::<Square>().unwrap(),
            Some(Piece::new(*owner, *kind)),
        );
    }
    GameState::from_board(
        engine.ruleset().id.clone(),
        Mode::Standard,
        board,
        Player::White,
        1,
    )
}
fn cannon_fixtures(engine: &Engine) -> Result<Value, Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for (name, screen) in [
        ("empty-rays", None),
        ("pawn-screen", Some(PieceKind::Pawn)),
        ("cannon-screen", Some(PieceKind::Cannon)),
    ] {
        let mut pieces = vec![
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Cannon),
        ];
        if let Some(kind) = screen {
            pieces.push(("d5", Player::White, kind));
            pieces.push(("d7", Player::Black, PieceKind::Rook));
        }
        let state = board_state(engine, &pieces);
        let mut destinations: Vec<String> = engine
            .legal_actions(&state)?
            .into_iter()
            .filter_map(|a| {
                if let Action::Move(m) = a {
                    if m.from == "d4".parse::<Square>().unwrap() {
                        return Some(m.to.to_string());
                    }
                }
                None
            })
            .collect();
        destinations.sort();
        let board: Vec<Value> = pieces.iter().map(|(sq, owner, kind)| json!({"square": sq,
            "color": if *owner == Player::White { "white" } else { "black" },
            "type": match kind { PieceKind::King => "king", PieceKind::Cannon => "cannon", PieceKind::Pawn => "pawn", _ => "rook" }})).collect();
        cases.push(json!({"name": name, "origin": "d4", "board": board, "rust_destinations": destinations}));
    }
    Ok(
        json!({"ruleset": engine.ruleset().id, "scope": "cannon-path/no-rules/no-passives", "cases": cases}),
    )
}
fn tactical_diagnostics(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = TacticalConfig::default();
    let mut flags = args.iter().skip(1);
    while let Some(flag) = flags.next() {
        let value = flags.next().ok_or("Missing tactical flag value")?;
        match flag.as_str() {
            "--depth" => config.max_depth = value.parse()?,
            "--nodes" => config.max_nodes = value.parse()?,
            _ => {
                return Err(
                    format!("Unknown tactical flag {flag}; expected --depth or --nodes").into(),
                );
            }
        }
    }
    if config.max_depth == 0 || config.max_nodes == 0 {
        return Err("Tactical depth and node budgets must be positive".into());
    }
    let engine = Engine::bundled_latest()?;
    let immediate = board_state(
        &engine,
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("h1", Player::White, PieceKind::Rook),
        ],
    );
    let mut continuation = board_state(
        &engine,
        &[
            ("b1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("a2", Player::White, PieceKind::Rook),
            ("a3", Player::Black, PieceKind::Queen),
        ],
    );
    continuation.actions_remaining = 2;
    let mut reports = Vec::new();
    for (scenario, state) in [
        ("immediate-win", immediate),
        ("same-player-combination", continuation),
    ] {
        let before = state.fingerprint();
        let report = match find_tactical_win(&engine, &state, config) {
            Ok(Some(win)) => {
                json!({"scenario": scenario, "status": "win-in-enumerated-tree", "result": win})
            }
            Ok(None) => {
                json!({"scenario": scenario, "status": "unknown", "reason": "no win established within action horizon"})
            }
            Err(error) => {
                json!({"scenario": scenario, "status": "unknown", "reason": error.to_string()})
            }
        };
        assert_eq!(before, state.fingerprint());
        reports.push(report);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "ruleset": engine.ruleset().id, "depth_unit": "actions",
            "max_depth": config.max_depth, "max_nodes": config.max_nodes,
            "information_scope": "full-state-diagnostic", "reports": reports,
        }))?
    );
    Ok(())
}

/// Play self-play games and report every outcome separately.
///
/// The arena exists so playing strength can be measured reproducibly. Colour is exchanged so the
/// first-move advantage cancels, both sides share one configuration so the only difference is the
/// position, and unfinished or errored games are counted apart from decided ones so the score rate
/// stays an estimate over games that actually finished.
fn arena_games(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "arena [--time-ms N] [--depth N] [--nodes N] [--chance-samples N] [--games N] [--max-actions N] [--seed N]\n  [--evaluation material|positional|legacy-material]\n  [--against material|positional|legacy-material | --policy FILE]\n  --trace FILE  Recommended: versioned metadata and flushed new_game/search/apply start/end events, with diagnostic state snapshots.\n  --out FILE    Legacy GameRecord JSONL, one flushed line per finished game, for every arena variant.\n                AUGMENTIA_ARENA_OUT is the fallback when --out is absent.\n  Both outputs create NEW files only; existing paths are refused. No fsync/power-loss guarantee.\n  Games exchange colors but use DIFFERENT seeds. Consecutive base-seed batches can overlap; see trace metadata."
        );
        return Ok(());
    }
    let mut trace_path: Option<std::path::PathBuf> = None;
    let mut out_path: Option<std::path::PathBuf> = None;
    let mut config = SearchConfig::default();
    let mut arena = ArenaConfig::default();
    // When set, the arena becomes a match between this configuration and the opponent.
    let mut opponent: Option<Evaluation> = None;
    // When set, the first side searches with this policy ordering the root.
    let mut policy: Option<augment_search::policy::LinearPolicy> = None;
    let mut policy_path: Option<String> = None;
    let mut it = args.iter().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or("Missing flag value")?;
        match flag.as_str() {
            "--time-ms" => config.time_limit = Duration::from_millis(value.parse()?),
            "--depth" => config.max_depth = value.parse()?,
            "--nodes" => config.max_nodes = value.parse()?,
            "--chance-samples" => config.chance_samples = value.parse()?,
            "--evaluation" => {
                config.evaluation = match value.as_str() {
                    "material" => Evaluation::Material,
                    "positional" => Evaluation::Positional,
                    "legacy-material" => Evaluation::LegacyMaterial,
                    _ => {
                        return Err(
                            "Expected --evaluation material, positional or legacy-material".into(),
                        );
                    }
                }
            }
            "--trace" => trace_path = Some(value.into()),
            "--out" => out_path = Some(value.into()),
            "--games" => arena.games_per_pairing = value.parse()?,
            "--max-actions" => arena.max_actions = value.parse()?,
            "--seed" => arena.seed = value.parse()?,
            // A policy file turns the arena into a policy-vs-no-policy match.
            "--policy" => policy_path = Some(value.clone()),
            // Supplying a second evaluation turns the arena into a match between the two.
            "--against" => {
                opponent = Some(match value.as_str() {
                    "material" => Evaluation::Material,
                    "positional" => Evaluation::Positional,
                    "legacy-material" => Evaluation::LegacyMaterial,
                    _ => {
                        return Err(
                            "Expected --against material, positional or legacy-material".into()
                        );
                    }
                })
            }
            _ => return Err(format!("Unknown flag {flag}").into()),
        }
    }
    if opponent.is_some() && policy_path.is_some() {
        return Err("--against and --policy are mutually exclusive".into());
    }
    if out_path.is_none() {
        out_path = std::env::var_os("AUGMENTIA_ARENA_OUT")
            .filter(|p| !p.is_empty())
            .map(Into::into);
    }
    arena.search = config.clone();
    if let Some(path) = &policy_path {
        let text = std::fs::read_to_string(path)?;
        policy = Some(serde_json::from_str(&text)?);
    }
    let engine = Engine::bundled_latest()?;
    // Snapshot the settings for the report before the arena consumes them.
    let recorded = json!({
        "time_ms": arena.search.time_limit.as_millis(),
        "depth": arena.search.max_depth,
        "evaluation": arena.search.evaluation.label(),
        "against": opponent.map(Evaluation::label),
        "policy": policy.is_some(),
        "games_per_pairing": arena.games_per_pairing,
        "max_actions": arena.max_actions,
        "seed": arena.seed,
        "nodes": arena.search.max_nodes,
        "chance_samples": arena.search.chance_samples,
        "depth_unit": arena.search.depth_unit,
        "policy_path": policy_path,
        "trace_path": trace_path,
        "out_path": out_path,
        "seed_note": "Color-exchanged games use different seeds; consecutive base-seed batches may overlap. See trace run_start.seed_schedule.",
    });
    use arena::recording::JsonlWriter;
    let open = |path: &std::path::PathBuf, flag: &str| {
        JsonlWriter::create_new(path).map_err(|error| {
            std::io::Error::new(error.kind(), format!("{flag} {}: {error}", path.display()))
        })
    };
    let trace_writer = trace_path
        .as_ref()
        .map(|path| open(path, "--trace"))
        .transpose()?;
    let mut out_writer = out_path
        .as_ref()
        .map(|path| open(path, "--out"))
        .transpose()?;
    let comparison = if let Some(policy) = policy {
        arena::ArenaMatch::Policy(std::sync::Arc::new(policy))
    } else if let Some(second) = opponent {
        arena::ArenaMatch::Evaluation {
            first: config.evaluation,
            second,
        }
    } else {
        arena::ArenaMatch::SelfPlay
    };
    let total = u64::from(arena.games_per_pairing) * 2;
    let mut completed = |played: u64, record: &Value| -> std::io::Result<()> {
        if let Some(writer) = &mut out_writer {
            writer.write(record).map_err(|error| {
                std::io::Error::new(error.kind(), format!("--out recording failed: {error}"))
            })?;
        }
        eprintln!(
            "game {played}/{total}: {} {}",
            record["outcome"], record["winner"]
        );
        Ok(())
    };
    let (records, summary) = if let Some(mut writer) = trace_writer {
        arena::run_observed(&engine, arena, comparison, &mut |event| {
            writer.write(event).map_err(|error| {
                std::io::Error::new(error.kind(), format!("--trace recording failed: {error}"))
            })?;
            if event.event == "game_end" {
                completed(event.game_index.unwrap_or(0) + 1, &event.data["record"])?;
            }
            Ok(())
        })?
    } else {
        arena::run_with_progress(&engine, arena, comparison, &mut |played, _, record| {
            let record = serde_json::to_value(record).map_err(std::io::Error::other)?;
            completed(played as u64, &record)
        })?
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "ruleset": engine.ruleset().id,
            "site_parity_verified": false,
            "build": if cfg!(debug_assertions) { "debug" } else { "release" },
            "config": recorded,
            "summary": summary,
            "games": records,
        }))?
    );
    Ok(())
}

/// Emit teacher records: one JSON line per decision, for the learning pipeline.
///
/// The records carry the position, the engine identity and budget, and the full candidate set, so a
/// learner and an independent verifier can both work from the file alone.
fn teacher_records(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = SearchConfig::default();
    let mut seed = 1u64;
    let mut games = 1u32;
    let mut max_actions = 60u32;
    let mut it = args.iter().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or("Missing flag value")?;
        match flag.as_str() {
            "--time-ms" => config.time_limit = Duration::from_millis(value.parse()?),
            "--depth" => config.max_depth = value.parse()?,
            "--nodes" => config.max_nodes = value.parse()?,
            "--evaluation" => {
                config.evaluation = match value.as_str() {
                    "material" => Evaluation::Material,
                    "positional" => Evaluation::Positional,
                    _ => return Err("Expected --evaluation material or positional".into()),
                }
            }
            "--seed" => seed = value.parse()?,
            "--games" => games = value.parse()?,
            "--max-actions" => max_actions = value.parse()?,
            _ => return Err(format!("Unknown flag {flag}").into()),
        }
    }
    let engine = Engine::bundled_latest()?;
    for game in 0..games {
        let (records, _status) = teacher::collect_game(
            &engine,
            seed.wrapping_add(u64::from(game)),
            config.clone(),
            max_actions,
        );
        for record in records {
            println!("{}", serde_json::to_string(&record)?);
        }
    }
    Ok(())
}

/// Fit a policy on teacher records and verify it on records it never saw.
///
/// The split is by record order, and the held-out portion is verified separately, because
/// agreement on the training data says nothing about whether the policy generalised.
fn policy_fit(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use augment_search::policy::{train_policy, verify_imitation};
    let mut epochs = 5usize;
    let mut holdout = 0.3f64;
    let mut k = 3usize;
    let mut it = args.iter().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or("Missing flag value")?;
        match flag.as_str() {
            "--epochs" => epochs = value.parse()?,
            "--holdout" => holdout = value.parse()?,
            "--top-k" => k = value.parse()?,
            _ => return Err(format!("Unknown flag {flag}").into()),
        }
    }
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
    let records: Vec<augment_search::policy::LearnedRecord> = input
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    if records.len() < 2 {
        return Err("need at least two teacher records".into());
    }
    // Split by a fixed stride instead of halving the file order. Records arrive in game order, so a
    // positional split would train on early positions and verify on late ones, which measures the
    // position distribution rather than generalisation. A stride assigns every held-out record to
    // the same mix of plies and phases that the training set sees.
    let stride = ((1.0 / holdout).round() as usize).max(2);
    let mut train = Vec::new();
    let mut held = Vec::new();
    for (index, record) in records.into_iter().enumerate() {
        if index % stride == 0 {
            held.push(record);
        } else {
            train.push(record);
        }
    }
    if train.is_empty() || held.is_empty() {
        return Err("holdout split produced an empty side".into());
    }
    let (policy, training) = train_policy(&train, epochs);
    let held_out = verify_imitation(&held, &policy, k);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "train": training,
            "holdout": held_out,
            "note": "holdout agreement is the quality measure; training agreement is not",
            "policy": policy,
        }))?
    );
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("bench");
    if command == "tactics" {
        return tactical_diagnostics(&args);
    }
    if command == "arena" {
        return arena_games(&args);
    }
    if command == "teacher" {
        return teacher_records(&args);
    }
    if command == "policy-fit" {
        return policy_fit(&args);
    }
    if !["bench", "play", "cannon-fixtures"].contains(&command) {
        return Err(
            "Expected bench, play, tactics, arena, teacher, policy-fit, or cannon-fixtures".into(),
        );
    }
    let mut config = SearchConfig::default();
    let mut seed = 1u64;
    let mut decisions = 24usize;
    let mut it = args.iter().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or("Missing flag value")?;
        match flag.as_str() {
            "--time-ms" => config.time_limit = Duration::from_millis(value.parse()?),
            "--depth" => config.max_depth = value.parse()?,
            "--depth-unit" => {
                config.depth_unit = match value.as_str() {
                    "turns" => DepthUnit::Turns,
                    "actions" => DepthUnit::Actions,
                    _ => return Err("Expected --depth-unit turns or actions".into()),
                }
            }
            "--nodes" => config.max_nodes = value.parse()?,
            "--chance-samples" => config.chance_samples = value.parse()?,
            "--search-seed" => config.search_seed = value.parse()?,
            "--evaluation" => {
                config.evaluation = match value.as_str() {
                    "material" => Evaluation::Material,
                    "positional" => Evaluation::Positional,
                    _ => return Err("Expected --evaluation material or positional".into()),
                }
            }
            "--seed" => seed = value.parse()?,
            "--decisions" => decisions = value.parse()?,
            _ => return Err(format!("Unknown flag {flag}").into()),
        }
    }
    // Only the current snapshot ships, so every subcommand runs on the same engine.
    let engine = Engine::bundled_latest()?;
    if command == "cannon-fixtures" {
        println!(
            "{}",
            serde_json::to_string_pretty(&cannon_fixtures(&engine)?)?
        );
        return Ok(());
    }
    let mut records = Vec::new();
    if command == "bench" {
        let win = board_state(
            &engine,
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("h1", Player::White, PieceKind::Rook),
            ],
        );
        let mut continuation = win.clone();
        continuation.actions_remaining = 2;
        let mut scenarios = vec![
            ("immediate-win".to_owned(), win),
            ("same-player-continuation".to_owned(), continuation),
            (
                "standard-board/no-cards".to_owned(),
                GameState::standard(engine.ruleset().id.clone(), Mode::Standard, seed),
            ),
            (
                "seeded-opening-draft".to_owned(),
                engine.new_game(Mode::Standard, seed)?,
            ),
        ];
        // Recorded arbitrary legal trajectory: a workload fixture, NOT good-play evidence.
        let mut state = engine.new_game(Mode::Standard, seed)?;
        for step in 0..decisions {
            if state.status != GameStatus::Ongoing {
                break;
            }
            let actions = engine.legal_actions(&state)?;
            if actions.is_empty() {
                return Err("Ongoing trajectory has no legal actions".into());
            }
            state = engine
                .apply(
                    &state,
                    actions[(step * 17 + seed as usize % actions.len()) % actions.len()].clone(),
                )?
                .state;
            if [3, 7, 15, 23].contains(&step) {
                scenarios.push((format!("trajectory-decision-{}", step + 1), state.clone()));
            }
        }
        for (name, state) in scenarios {
            let before = state.fingerprint();
            let report = search(&engine, &state, config.clone())?;
            if let Some(ref action) = report.selected_action {
                engine.apply(&state, action.clone())?;
            }
            assert_eq!(before, state.fingerprint());
            records.push(json!({"scenario": name, "phase": format!("{:?}", state.phase), "cards": state.cards.len(), "report": report}));
        }
    } else {
        let mut state = engine.new_game(Mode::Standard, seed)?;
        for step in 0..decisions {
            if state.status != GameStatus::Ongoing {
                break;
            }
            let report = search(&engine, &state, config.clone())?;
            let action = report
                .selected_action
                .clone()
                .ok_or("Ongoing game has no selected action")?;
            records.push(json!({"decision": step, "actor": format!("{:?}", state.turn), "phase": format!("{:?}", state.phase), "report": report}));
            state = engine.apply(&state, action)?.state;
        }
        records.push(json!({"final_status": format!("{:?}", state.status), "final_ply": state.ply, "cards": state.cards.len(), "truncated": state.status == GameStatus::Ongoing}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"ruleset": engine.ruleset().id,
        "semantics": format!("model-free/actor-aware/sampled-chance/{}", config.evaluation.label()),
        "site_parity_verified": false, "build": if cfg!(debug_assertions) {"debug"} else {"release"},
        "seed": seed, "config": {"time_ms": config.time_limit.as_millis(), "depth": config.max_depth,
            "depth_unit": config.depth_unit,
            "nodes": config.max_nodes, "chance_samples": config.chance_samples, "search_seed": config.search_seed},
        "records": records}))?
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

use augment_core::{
    Action, Board, Engine, GameState, GameStatus, Mode, Piece, PieceKind, Player, Square,
};
use augment_search::arena::{self, ArenaConfig};
use augment_search::tactics::{TacticalConfig, find_tactical_win};
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
// Exact local regression trace: full Debug states/events and ordered legal actions.
// This is a before/after engine comparison, not an original-site parity oracle.
fn transition_trace(seed: u64, decisions: usize) -> Result<Value, Box<dyn std::error::Error>> {
    let mut traces = Vec::new();
    for engine in [
        Engine::bundled_20260807()?,
        Engine::bundled_20260818()?,
        Engine::bundled_20260915()?,
    ] {
        let mut state = engine.new_game(Mode::Standard, seed)?;
        let initial = format!("{state:?}");
        let mut steps = Vec::new();
        for step in 0..decisions {
            if state.status != GameStatus::Ongoing {
                break;
            }
            let actions = engine.legal_actions(&state)?;
            if actions.is_empty() {
                return Err("Ongoing trace has no legal actions".into());
            }
            let selected = (step * 17 + seed as usize % actions.len()) % actions.len();
            let transition = engine.apply(&state, actions[selected].clone())?;
            steps.push(
                json!({"decision": step, "legal_actions": format!("{actions:?}"),
                "selected_index": selected, "transition": format!("{transition:?}")}),
            );
            state = transition.state;
        }
        traces.push(json!({"ruleset": engine.ruleset().id, "initial": initial, "steps": steps}));
    }
    Ok(
        json!({"seed": seed, "requested_decisions": decisions, "site_parity_verified": false,
        "traces": traces}),
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
    let engine = Engine::bundled_20260915()?;
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
    let mut config = SearchConfig::default();
    let mut arena = ArenaConfig::default();
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
                    _ => return Err("Expected --evaluation material or positional".into()),
                }
            }
            "--games" => arena.games_per_pairing = value.parse()?,
            "--max-actions" => arena.max_actions = value.parse()?,
            "--seed" => arena.seed = value.parse()?,
            _ => return Err(format!("Unknown flag {flag}").into()),
        }
    }
    arena.search = config;
    let engine = Engine::bundled_latest()?;
    let (records, summary) = arena::run(&engine, arena);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "ruleset": engine.ruleset().id,
            "site_parity_verified": false,
            "build": if cfg!(debug_assertions) { "debug" } else { "release" },
            "config": {
                "time_ms": arena.search.time_limit.as_millis(),
                "depth": arena.search.max_depth,
                "evaluation": arena.search.evaluation.label(),
                "games_per_pairing": arena.games_per_pairing,
                "max_actions": arena.max_actions,
            },
            "summary": summary,
            "games": records,
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
    if !["bench", "play", "cannon-fixtures", "transition-trace"].contains(&command) {
        return Err(
            "Expected bench, play, tactics, arena, cannon-fixtures, or transition-trace".into(),
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
    if command == "transition-trace" {
        println!(
            "{}",
            serde_json::to_string_pretty(&transition_trace(seed, decisions)?)?
        );
        return Ok(());
    }
    let engine = if command == "cannon-fixtures" {
        Engine::bundled_20260915()?
    } else {
        Engine::bundled_latest()?
    };
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
            let report = search(&engine, &state, config)?;
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
            let report = search(&engine, &state, config)?;
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

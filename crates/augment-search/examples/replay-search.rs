//! Replay saved actions, then measure one decision with a process watchdog.
//! Usage: cargo run --release -p augment-search --example replay-search -- [fixture.json] [material|positional]
use augment_core::{Action, Engine, Mode};
use augment_search::{Evaluation, SearchConfig, search};
use serde::Deserialize;
use serde_json::json;
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Deserialize)]
struct Replay {
    seed: u64,
    actions: Vec<Action>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let default = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/arena-stall-seed1000012.json"
    );
    let fixture = args.get(1).map(String::as_str).unwrap_or(default);
    let evaluation = match args.get(2).map(String::as_str).unwrap_or("positional") {
        "material" => Evaluation::Material,
        "positional" => Evaluation::Positional,
        other => return Err(format!("unknown evaluation: {other}").into()),
    };
    let replay: Replay = serde_json::from_str(&std::fs::read_to_string(fixture)?)?;
    let (finished, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        if matches!(
            receiver.recv_timeout(Duration::from_secs(5)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ) {
            eprintln!("watchdog: replay or search exceeded five seconds");
            std::process::exit(124);
        }
    });
    let engine = Engine::bundled_latest()?;
    let start = Instant::now();
    let mut state = engine.new_game(Mode::Standard, replay.seed)?;
    for action in &replay.actions {
        state = engine.apply(&state, action.clone())?.state;
    }
    let replay_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "search_start: seed={}, replay_actions={}, replay_ms={replay_ms:.3}",
        replay.seed,
        replay.actions.len()
    );
    let before = state.fingerprint();
    let report = search(
        &engine,
        &state,
        SearchConfig {
            time_limit: Duration::from_millis(1000),
            max_nodes: 8000,
            max_depth: 4,
            search_seed: replay.seed,
            evaluation,
            ..Default::default()
        },
    )?;
    assert_eq!(state.fingerprint(), before);
    finished.send(())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "seed": replay.seed, "replay_actions": replay.actions.len(), "replay_ms": replay_ms,
            "report": report,
        }))?
    );
    Ok(())
}

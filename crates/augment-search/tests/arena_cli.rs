use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
fn path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "augment-arena-{}-{}-{label}.jsonl",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_augment-search"));
    command.env_remove("AUGMENTIA_ARENA_OUT").args([
        "arena",
        "--games",
        "1",
        "--max-actions",
        "1",
        "--seed",
        "23",
        "--time-ms",
        "1",
        "--nodes",
        "1",
        "--depth",
        "1",
    ]);
    command
}
fn arena(extra: &[&str]) -> Output {
    command().args(extra).output().unwrap()
}
fn lines(path: &PathBuf) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn trace_records_each_call_boundary_and_actual_seeds() {
    let trace = path("trace");
    let out = path("out");
    let result = arena(&[
        "--trace",
        trace.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--against",
        "positional",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let events = lines(&trace);
    assert_eq!(events[0]["event"], "run_start");
    assert_eq!(events[0]["schema_version"], 1);
    assert_eq!(events.last().unwrap()["event"], "run_end");
    let starts: Vec<_> = events
        .iter()
        .filter(|e| e["event"] == "game_start")
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[0]["data"]["seed"], 23);
    assert_eq!(starts[1]["data"]["seed"], 1_000_026);
    for game in 0..2 {
        let calls: Vec<_> = events
            .iter()
            .filter(|e| {
                e["game_index"] == game
                    && e["event"]
                        .as_str()
                        .is_some_and(|s| s.starts_with("search_") || s.starts_with("apply_"))
            })
            .collect();
        assert_eq!(
            calls
                .iter()
                .map(|e| e["event"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["search_start", "search_end", "apply_start", "apply_end"]
        );
        assert_eq!(calls[0]["action_index"], 0);
        assert!(calls[0]["data"]["state"]["board"].as_array().unwrap().len() >= 2);
        assert!(calls[1]["data"]["selected_action"].is_string());
        assert!(calls[3]["data"]["state"]["fingerprint"].is_u64());
    }
    assert_eq!(lines(&out).len(), 2);
    let stdout: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(lines(&out), stdout["games"].as_array().unwrap().clone());
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event["sequence"], index);
        assert_eq!(event["run_id"], events[0]["run_id"]);
        assert!(event["unix_ms"].is_number());
    }
    assert_eq!(
        events[0]["data"]["seed_schedule"]["color_exchange_uses_same_seed"],
        false
    );
    assert_eq!(
        events[0]["data"]["seed_schedule"]["independent_samples"],
        false
    );
    let searches: Vec<_> = events
        .iter()
        .filter(|event| event["event"] == "search_start")
        .collect();
    assert_eq!(searches[0]["data"]["config"]["evaluation"], "material-v1");
    assert_eq!(searches[1]["data"]["config"]["evaluation"], "positional-v1");
    assert!(
        searches[0]["data"]["state"]["full_state_debug"]
            .as_str()
            .unwrap()
            .contains("rng:")
    );
}

#[test]
fn trace_actions_replay_exact_fingerprints_and_preserve_pending_call_after_interruption() {
    use augment_core::{Action, Engine, GameState, Mode};
    use std::collections::BTreeMap;

    fn replay(events: &[Value]) -> (usize, usize) {
        let engine = Engine::bundled_latest().unwrap();
        let mut states = BTreeMap::<u64, GameState>::new();
        let mut selected = BTreeMap::<u64, Action>::new();
        let mut pending = BTreeMap::<u64, GameState>::new();
        let mut completed = 0;
        for event in events {
            let Some(game) = event["game_index"].as_u64() else {
                continue;
            };
            let data = &event["data"];
            match event["event"].as_str().unwrap() {
                "game_start" => {
                    let state = engine
                        .new_game(Mode::Standard, data["seed"].as_u64().unwrap())
                        .unwrap();
                    states.insert(game, state);
                }
                "new_game_end" | "search_start" => {
                    assert_eq!(data["state"]["fingerprint"], states[&game].fingerprint());
                }
                "search_end" => {
                    let action: Action = serde_json::from_value(data["action"].clone())
                        .expect("search_end must carry a deserializable Action");
                    selected.insert(game, action);
                }
                "apply_start" => {
                    let action: Action = serde_json::from_value(data["action"].clone())
                        .expect("apply_start must carry a deserializable Action");
                    assert_eq!(action, selected[&game]);
                    assert_eq!(data["state_fingerprint"], states[&game].fingerprint());
                    let next = engine.apply(&states[&game], action).unwrap().state;
                    assert!(pending.insert(game, next).is_none());
                }
                "apply_end" => {
                    let next = pending.remove(&game).unwrap();
                    assert_eq!(data["state"]["fingerprint"], next.fingerprint());
                    states.insert(game, next);
                    completed += 1;
                }
                _ => {}
            }
        }
        (completed, pending.len())
    }

    let trace = path("replay-trace");
    let result = arena(&[
        "--trace",
        trace.to_str().unwrap(),
        "--max-actions",
        "3",
        "--against",
        "positional",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let events = lines(&trace);
    assert_eq!(replay(&events), (6, 0));
    // A flushed apply_start alone means a pending call, never a confirmed applied action.
    let interrupted = events
        .iter()
        .rposition(|event| event["event"] == "apply_start")
        .unwrap();
    assert_eq!(replay(&events[..=interrupted]), (5, 1));
    let (ruleset, catalog_hash, profile_hash) = augment_core::latest_identity();
    assert_eq!(events[0]["data"]["engine_identity"]["ruleset_id"], ruleset);
    assert_eq!(
        events[0]["data"]["engine_identity"]["catalog_hash"],
        catalog_hash
    );
    assert_eq!(
        events[0]["data"]["engine_identity"]["card_profile_hash"],
        format!("{profile_hash:016x}")
    );
}

#[test]
fn refuses_to_overwrite_results() {
    let out = path("existing");
    std::fs::write(&out, "preserve me\n").unwrap();
    let result = arena(&["--out", out.to_str().unwrap()]);
    assert!(!result.status.success());
    assert_eq!(std::fs::read_to_string(&out).unwrap(), "preserve me\n");
    let result = arena(&["--trace", out.to_str().unwrap()]);
    assert!(!result.status.success());
    assert_eq!(std::fs::read_to_string(&out).unwrap(), "preserve me\n");
}

#[test]
fn self_play_and_policy_variants_record_trace_and_results() {
    let policy = path("policy");
    std::fs::write(
        &policy,
        serde_json::to_vec(&augment_search::policy::LinearPolicy::zeros()).unwrap(),
    )
    .unwrap();
    for is_policy in [false, true] {
        let trace = path("variant-trace");
        let out = path("variant-out");
        let mut extra = vec![
            "--trace",
            trace.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ];
        if is_policy {
            extra.extend(["--policy", policy.to_str().unwrap()]);
        }
        let result = arena(&extra);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let events = lines(&trace);
        assert_eq!(
            events[0]["data"]["variant"],
            if is_policy { "policy" } else { "self_play" }
        );
        assert_eq!(lines(&out).len(), 2);
        assert_eq!(
            events.iter().filter(|e| e["event"] == "apply_end").count(),
            2
        );
        if is_policy {
            let searches: Vec<_> = events
                .iter()
                .filter(|event| event["event"] == "search_start")
                .collect();
            assert!(searches[0]["data"]["config"]["policy"].is_object());
            assert!(searches[1]["data"]["config"]["policy"].is_null());
        }
    }
}

#[test]
fn legacy_environment_output_is_used_unless_explicit_out_overrides_it() {
    let fallback = path("env");
    let result = command()
        .env("AUGMENTIA_ARENA_OUT", &fallback)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(lines(&fallback).len(), 2);
    let explicit = path("explicit");
    let result = command()
        .env("AUGMENTIA_ARENA_OUT", &fallback)
        .args(["--out", explicit.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(lines(&explicit).len(), 2);
    assert_eq!(lines(&fallback).len(), 2);
}

#[test]
fn invalid_output_path_and_ambiguous_opponents_fail_before_play() {
    let missing_parent = path("missing-parent").join("trace.jsonl");
    let result = arena(&["--trace", missing_parent.to_str().unwrap()]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("--trace"));
    assert!(result.stdout.is_empty());
    let result = arena(&["--against", "material", "--policy", "not-read"]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("mutually exclusive"));
}

#[test]
fn help_distinguishes_trace_from_legacy_result_output() {
    let result = arena(&["--help"]);
    assert!(result.status.success());
    let help = String::from_utf8(result.stdout).unwrap();
    assert!(help.contains("--trace FILE"));
    assert!(help.contains("--out FILE"));
    assert!(help.contains("AUGMENTIA_ARENA_OUT"));
    assert!(help.contains("NEW files only"));
}

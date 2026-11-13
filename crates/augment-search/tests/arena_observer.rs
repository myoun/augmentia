use augment_core::Engine;
use augment_search::arena::{ArenaConfig, ArenaMatch, run_match_with_progress, run_observed};
use augment_search::{Evaluation, SearchConfig};
use serde_json::Value;
use std::io;
use std::time::Duration;

fn config() -> ArenaConfig {
    ArenaConfig {
        search: SearchConfig {
            time_limit: Duration::from_millis(1),
            max_nodes: 1,
            max_depth: 1,
            ..Default::default()
        },
        max_actions: 1,
        games_per_pairing: 1,
        seed: 23,
    }
}

#[test]
fn observer_failure_stops_at_exact_boundary_and_is_not_swallowed() {
    let engine = Engine::bundled_latest().unwrap();
    for stop in [
        "run_start",
        "game_start",
        "new_game_start",
        "new_game_end",
        "search_start",
        "search_end",
        "apply_start",
        "apply_end",
        "game_end",
        "run_end",
    ] {
        let mut events = Vec::new();
        let error = run_observed(&engine, config(), ArenaMatch::SelfPlay, &mut |event| {
            events.push(serde_json::to_value(event).unwrap());
            if event.event == stop {
                Err(io::Error::other("recording unavailable"))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "recording unavailable");
        assert_eq!(events.last().unwrap()["event"], stop);
        if stop != "run_end" {
            assert!(!events.iter().any(|event| event["event"] == "run_end"));
        }
        if stop == "search_start" || stop == "apply_start" {
            let last = events.last().unwrap();
            assert_eq!(last["game_index"], 0);
            assert_eq!(last["action_index"], 0);
            let end = if stop == "search_start" {
                "search_end"
            } else {
                "apply_end"
            };
            assert!(!events.iter().any(|event| event["event"] == end));
        }
    }
}

#[test]
fn consecutive_base_seeds_overlap_and_metadata_does_not_claim_independence() {
    let engine = Engine::bundled_latest().unwrap();
    let mut seeds = Vec::new();
    for seed in [23, 24] {
        let mut run = Vec::new();
        let mut config = config();
        config.seed = seed;
        config.max_actions = 0;
        config.games_per_pairing = 2;
        run_observed(&engine, config, ArenaMatch::SelfPlay, &mut |event| {
            if event.event == "run_start" {
                assert_eq!(event.data["seed_schedule"]["independent_samples"], false);
            }
            if event.event == "game_start" {
                run.push(event.data["seed"].as_u64().unwrap());
            }
            Ok(())
        })
        .unwrap();
        seeds.push(run);
    }
    assert_eq!(seeds[0], [23, 1_000_026, 24, 1_000_027]);
    assert_eq!(seeds[1], [24, 1_000_027, 25, 1_000_028]);
}

#[test]
fn legacy_progress_callback_remains_one_call_per_finished_game() {
    let engine = Engine::bundled_latest().unwrap();
    let mut config = config();
    config.max_actions = 0;
    config.games_per_pairing = 2;
    let mut calls = Vec::new();
    let (records, summary) = run_match_with_progress(
        &engine,
        config,
        Evaluation::Material,
        Evaluation::Positional,
        &mut |played, total, record| {
            calls.push((played, total, serde_json::to_value(record).unwrap()));
        },
    );
    assert_eq!(summary.games, 4);
    for (index, (played, total, record)) in calls.iter().enumerate() {
        assert_eq!((*played, *total), (index + 1, 4));
        assert_eq!(*record, serde_json::to_value(&records[index]).unwrap());
    }
}

#[test]
fn jsonl_is_visible_to_a_reader_before_the_next_engine_call() {
    let engine = Engine::bundled_latest().unwrap();
    let path = std::env::temp_dir().join(format!(
        "augment-observer-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut writer = augment_search::arena::recording::JsonlWriter::create_new(&path).unwrap();
    let result = run_observed(&engine, config(), ArenaMatch::SelfPlay, &mut |event| {
        writer.write(event)?;
        let text = std::fs::read_to_string(&path)?;
        assert!(text.ends_with('\n'));
        let last: Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
        assert_eq!(last["event"], event.event);
        assert_eq!(last["sequence"], event.sequence);
        if event.event == "search_start" {
            Err(io::Error::other("simulate interruption after flush"))
        } else {
            Ok(())
        }
    });
    assert!(result.is_err());
    let text = std::fs::read_to_string(path).unwrap();
    let last: Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
    assert_eq!(last["event"], "search_start");
    assert!(last["data"]["state"]["full_state_debug"].is_string());
}

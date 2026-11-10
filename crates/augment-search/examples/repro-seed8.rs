// Verify the fix: the box application must cost a handful of events, and seed 8 must play out.
//
// Before the fix one box use emitted 153 events (149 RandomSample draws) and seed 8 died at ply 9 with
// `effect resolution exceeded 4096 events`.
use augment_core::{Action, Engine, GameStatus, Mode};
use augment_search::{SearchConfig, search};
use std::collections::BTreeMap;

fn main() {
    let engine = Engine::bundled_latest().expect("bundled ruleset");
    let config = SearchConfig {
        time_limit: std::time::Duration::from_millis(250),
        max_depth: 4,
        depth_unit: augment_search::DepthUnit::Turns,
        max_nodes: 100_000,
        chance_samples: 4,
        search_seed: 19,
        evaluation: augment_search::Evaluation::Material,
        policy: None,
    };

    // 1. Replay seed 8 as far as it goes. It used to fail at ply 9.
    let seed = 8;
    let mut state = engine.new_game(Mode::Standard, seed).expect("new game");
    let mut ply = 0;
    loop {
        if state.status != GameStatus::Ongoing {
            println!("seed 8 ended at ply {ply}: {:?}", state.status);
            break;
        }
        let report = match search(&engine, &state, config.clone()) {
            Ok(report) => report,
            Err(error) => {
                println!("seed 8 STILL FAILS at ply {ply}: {error:?}");
                return;
            }
        };
        let Some(action) = report.selected_action else {
            println!("seed 8: no action at ply {ply}");
            break;
        };
        state = engine.apply(&state, action).expect("apply").state;
        ply += 1;
        if ply > 200 {
            println!("seed 8 survived past ply 200");
            break;
        }
    }

    // 2. Event cost of one box use in a state that holds one.
    let mut state = engine.new_game(Mode::Standard, seed).expect("new game");
    for _ in 0..9 {
        let report = search(&engine, &state, config.clone()).expect("search");
        state = engine
            .apply(&state, report.selected_action.expect("an action"))
            .expect("apply")
            .state;
    }
    if let Some(box_action) = engine
        .legal_actions(&state)
        .expect("legal actions")
        .into_iter()
        .find(|action| matches!(action, Action::UseCard { instance_id: 2, .. }))
    {
        let transition = engine.apply(&state, box_action).expect("box apply");
        println!(
            "box application now emits {} events",
            transition.events.len()
        );
        let mut by_kind: BTreeMap<String, usize> = BTreeMap::new();
        for event in &transition.events {
            let kind = format!("{event:?}")
                .split('(')
                .next()
                .unwrap_or("?")
                .to_string();
            *by_kind.entry(kind).or_default() += 1;
        }
        for (kind, count) in by_kind {
            println!("  {count:4} {kind}");
        }
    } else {
        println!("(no box action at ply 9 after the fix; the game path changed)");
    }
}

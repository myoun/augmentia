use augment_core::*;

fn main() {
    let engine = Engine::bundled_latest().unwrap();
    let (seed, transition) = (0..5_000u64)
        .find_map(|seed| {
            let t = engine.new_game_transition(Mode::Standard, seed).ok()?;
            t.state
                .applied_rules
                .iter()
                .any(|r| r == CHESS_344200)
                .then_some((seed, t))
        })
        .expect("seed");
    println!("seed={seed}");
    let mut seen: Vec<String> = transition
        .events
        .iter()
        .filter_map(|e| match e {
            Event::RandomSample { purpose, .. } => Some(purpose.clone()),
            _ => None,
        })
        .collect();
    seen.sort();
    seen.dedup();
    println!("purposes={seen:?}");
    let n344 = transition
        .events
        .iter()
        .filter(|e| matches!(e, Event::RandomSample { purpose, .. } if purpose.contains("344200")))
        .count();
    println!("count_containing_344200={n344}");
}

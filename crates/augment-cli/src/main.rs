use augment_core::{Action, Engine, Mode, MoveAction, Square};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::bundled_20260807()?;
    let state = engine.new_game(Mode::Standard, 20260807)?;
    let legal_actions = engine.legal_actions(&state)?;

    println!("ruleset: {}", engine.ruleset().id);
    println!("coverage: {}", engine.ruleset().coverage.status);
    println!("initial legal actions: {}", legal_actions.len());

    let e2_e4 = Action::Move(MoveAction::new(
        "e2".parse::<Square>()?,
        "e4".parse::<Square>()?,
        None,
    ));
    let transition = engine.apply(&state, e2_e4)?;
    println!(
        "after e2-e4: turn={}, ply={}",
        transition.state.turn, transition.state.ply
    );
    println!("fingerprint: {:016x}", transition.state.fingerprint());
    Ok(())
}

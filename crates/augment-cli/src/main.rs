use augment_core::{Action, Engine, GamePhase, Mode, MoveAction, Square};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::bundled_20260807()?;
    let mut state = engine.new_game(Mode::Standard, 20260807)?;

    println!("ruleset: {}", engine.ruleset().id);
    println!("coverage: {}", engine.ruleset().coverage.status);
    while matches!(state.phase, GamePhase::Draft(_)) {
        let legal_actions = engine.legal_actions(&state)?;
        println!(
            "draft player={} choices={}",
            state.turn,
            legal_actions.len()
        );
        state = engine.apply(&state, legal_actions[0].clone())?.state;
    }
    println!(
        "initial board actions after draft: {}",
        engine.legal_actions(&state)?.len()
    );

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

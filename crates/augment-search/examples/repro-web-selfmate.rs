// Final verification: seed 20260807 self-play with the two-ply safety net.
//
// The reported defect: black's depth-1 search picked a move that let White's archbishop capture the
// king on the spot (three of forty-four moves did; the game ended at ply 33, 1-0). With the net, black
// must avoid all of them, and the game must play out to a decided result without that blunder.
use augment_core::{Engine, GameStatus, Mode, Player};
use augment_search::{DepthUnit, Evaluation, SearchConfig, search};
use std::time::Duration;

fn main() {
    let engine = Engine::bundled_latest().expect("bundled ruleset");
    let config = SearchConfig {
        time_limit: Duration::from_millis(1000),
        max_depth: 4,
        depth_unit: DepthUnit::Turns,
        max_nodes: 100_000,
        chance_samples: 4,
        search_seed: 19,
        evaluation: Evaluation::Material,
        policy: None,
    };

    let seed = 20_260_807;
    let mut state = engine.new_game(Mode::Standard, seed).expect("new game");
    let mut ply = 0;
    loop {
        if state.status != GameStatus::Ongoing {
            let winner = match state.status {
                GameStatus::Won { winner, .. } => {
                    if winner == Player::White {
                        "백"
                    } else {
                        "흑"
                    }
                }
                _ => "무승부",
            };
            println!("게임 종료: ply {ply}, {winner} 승 ({state:?})");
            return;
        }
        let report = search(&engine, &state, config.clone()).expect("search");
        let Some(action) = report.selected_action else {
            println!("ply {ply}: no action");
            return;
        };
        state = engine.apply(&state, action).expect("apply").state;
        ply += 1;
        if ply > 300 {
            println!("300수 초과, 계속 진행 중");
            return;
        }
    }
}

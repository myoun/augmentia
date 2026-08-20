use augment_core::{Engine, GameStatus, Mode, Player};
use std::error::Error;
use std::io::{self, BufRead, Write};

const DEFAULT_SEED: u64 = 20_260_807;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let engine = Engine::bundled_20260807()?;
    let seed = match args.next().as_deref() {
        Some("--seed") => args.next().ok_or("--seed requires value")?.parse()?,
        Some("--help") | Some("-h") => {
            println!("사용법: cargo run -p augment-cli -- [--seed N]");
            return Ok(());
        }
        Some(other) => return Err(format!("알 수 없는 인자: {other}").into()),
        None => DEFAULT_SEED,
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    play_hotseat(&engine, seed, stdin.lock(), stdout.lock())
}

fn play_hotseat<R: BufRead, W: Write>(
    engine: &Engine,
    seed: u64,
    mut input: R,
    mut output: W,
) -> Result<(), Box<dyn Error>> {
    let transition = engine.new_game_transition(Mode::Standard, seed)?;
    let mut state = transition.state;

    writeln!(output, "=== 증강체스 Hotseat 대국 (seed: {seed}) ===")?;
    for event in &transition.events {
        writeln!(output, "  * {event:?}")?;
    }

    while state.status == GameStatus::Ongoing {
        let legal_actions = engine.legal_actions(&state)?;
        if legal_actions.is_empty() {
            writeln!(output, "사용 가능한 행동이 없습니다.")?;
            break;
        }

        writeln!(
            output,
            "\n[Ply {}] {} 차례 | 행동 잔여: {}",
            state.ply,
            match state.turn {
                Player::White => "백",
                Player::Black => "흑",
            },
            state.actions_remaining
        )?;
        writeln!(output, "선택 가능한 행동:")?;
        for (index, action) in legal_actions.iter().enumerate() {
            writeln!(output, "  [{index}] {action:?}")?;
        }

        write!(output, "행동 번호 입력 (0-{}) > ", legal_actions.len() - 1)?;
        output.flush()?;

        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed == "q" || trimmed == "quit" {
            break;
        }

        let Ok(choice) = trimmed.parse::<usize>() else {
            writeln!(output, "올바른 정수를 입력하세요.")?;
            continue;
        };

        if let Some(action) = legal_actions.get(choice) {
            let transition = engine.apply(&state, action.clone())?;
            state = transition.state;
            for event in &transition.events {
                writeln!(output, "  * {event:?}")?;
            }
        } else {
            writeln!(output, "범위를 벗어난 번호입니다.")?;
        }
    }

    writeln!(output, "\n=== 게임 종료 ===")?;
    writeln!(output, "결과: {:?}", state.status)?;
    Ok(())
}

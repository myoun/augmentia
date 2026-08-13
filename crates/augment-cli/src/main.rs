use augment_core::{
    Action, ActionTarget, Engine, Event, GamePhase, GameState, GameStatus, Mode, MoveAction,
    PieceKind, Player, Square, implemented_card,
};
use std::error::Error;
use std::io::{self, BufRead, Write};

const DEFAULT_SEED: u64 = 20_260_807;

fn main() -> Result<(), Box<dyn Error>> {
    let seed = parse_seed(std::env::args().skip(1))?;
    let engine = Engine::bundled_20260807()?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    play_hotseat(&engine, seed, stdin.lock(), stdout.lock())
}

fn parse_seed(mut args: impl Iterator<Item = String>) -> Result<u64, Box<dyn Error>> {
    let mut seed = DEFAULT_SEED;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--seed" => {
                let value = args.next().ok_or("--seed 뒤에 정수가 필요합니다")?;
                seed = value.parse()?;
            }
            "--help" | "-h" => {
                println!("사용법: cargo run -p augment-cli -- [--seed 정수]");
                std::process::exit(0);
            }
            _ => return Err(format!("알 수 없는 인자: {argument}").into()),
        }
    }
    Ok(seed)
}

fn play_hotseat<R: BufRead, W: Write>(
    engine: &Engine,
    seed: u64,
    mut input: R,
    mut output: W,
) -> Result<(), Box<dyn Error>> {
    let initial = engine.new_game_transition(Mode::Standard, seed)?;
    let mut state = initial.state;
    writeln!(output, "증강체스 터미널 핫시트")?;
    writeln!(output, "규칙: {} | seed: {seed}", engine.ruleset().id)?;
    writeln!(output, "백과 흑을 한 터미널에서 번갈아 조종합니다.")?;
    render_help(&mut output)?;
    render_events(&mut output, &initial.events)?;

    loop {
        render_position(&mut output, &state)?;
        if state.status != GameStatus::Ongoing {
            render_result(&mut output, &state.status)?;
            writeln!(output, "최종 fingerprint: {:016x}", state.fingerprint())?;
            return Ok(());
        }

        let legal_actions = engine.legal_actions(&state)?;
        render_actions(&mut output, &state, &legal_actions)?;
        write!(output, "입력> ")?;
        output.flush()?;

        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            writeln!(output, "\n입력이 종료되어 게임을 끝냅니다.")?;
            return Ok(());
        }
        let command = line.trim().trim_start_matches('\u{feff}');
        if command.is_empty() {
            continue;
        }

        match command.to_ascii_lowercase().as_str() {
            "q" | "quit" | "exit" => {
                writeln!(output, "게임을 종료합니다.")?;
                return Ok(());
            }
            "h" | "help" | "?" => {
                render_help(&mut output)?;
                continue;
            }
            "b" | "board" => continue,
            "a" | "actions" | "legal" => {
                render_actions(&mut output, &state, &legal_actions)?;
                continue;
            }
            "c" | "cards" => {
                render_cards(&mut output, &state)?;
                continue;
            }
            _ => {}
        }

        let action = match select_action(command, &legal_actions) {
            Ok(action) => action,
            Err(message) => {
                writeln!(output, "잘못된 입력: {message}")?;
                continue;
            }
        };
        let transition = engine.apply(&state, action)?;
        render_events(&mut output, &transition.events)?;
        state = transition.state;
    }
}

fn select_action(command: &str, legal_actions: &[Action]) -> Result<Action, String> {
    if let Ok(number) = command.parse::<usize>() {
        if number == 0 || number > legal_actions.len() {
            return Err(format!(
                "1부터 {} 사이의 번호를 입력하세요",
                legal_actions.len()
            ));
        }
        return Ok(legal_actions[number - 1].clone());
    }

    let Some(movement) = parse_move(command)? else {
        return Err("행동 번호 또는 e2e4 형식의 이동을 입력하세요".to_owned());
    };
    let candidate = Action::Move(movement);
    if legal_actions.contains(&candidate) {
        return Ok(candidate);
    }

    if movement.promotion.is_none()
        && legal_actions.iter().any(|action| {
            matches!(action, Action::Move(legal)
                if legal.from == movement.from
                    && legal.to == movement.to
                    && legal.promotion.is_some())
        })
    {
        return Err("프로모션 기물을 붙이세요: q, r, b, n (예: e7e8q)".to_owned());
    }
    Err(format!("현재 합법적인 이동이 아닙니다: {command}"))
}

fn parse_move(command: &str) -> Result<Option<MoveAction>, String> {
    let compact: String = command
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect();
    if !matches!(compact.len(), 4 | 5) {
        return Ok(None);
    }
    let from = compact[0..2]
        .parse::<Square>()
        .map_err(|error| error.to_string())?;
    let to = compact[2..4]
        .parse::<Square>()
        .map_err(|error| error.to_string())?;
    let promotion = if compact.len() == 5 {
        Some(match compact.as_bytes()[4] {
            b'q' => PieceKind::Queen,
            b'r' => PieceKind::Rook,
            b'b' => PieceKind::Bishop,
            b'n' => PieceKind::Knight,
            other => {
                return Err(format!(
                    "프로모션 기물 {:?}은 사용할 수 없습니다; q, r, b, n 중 하나를 쓰세요",
                    char::from(other)
                ));
            }
        })
    } else {
        None
    };
    Ok(Some(MoveAction::new(from, to, promotion)))
}

fn render_position(output: &mut impl Write, state: &GameState) -> io::Result<()> {
    writeln!(output)?;
    writeln!(output, "     a  b  c  d  e  f  g  h")?;
    writeln!(output, "  +-------------------------+")?;
    for rank in (0..8).rev() {
        write!(output, "{} |", rank + 1)?;
        for file in 0..8 {
            let square = Square::new(file, rank).expect("board coordinate");
            let symbol = state
                .board
                .get(square)
                .map_or_else(|| ".".to_owned(), piece_symbol);
            write!(output, " {symbol:>2}")?;
        }
        writeln!(output, " | {}", rank + 1)?;
    }
    writeln!(output, "  +-------------------------+")?;
    writeln!(output, "     a  b  c  d  e  f  g  h")?;
    writeln!(
        output,
        "턴: {} | 단계: {} | ply: {} | 남은 이동 행동: {}",
        player_name(state.turn),
        phase_name(&state.phase),
        state.ply,
        state.actions_remaining
    )?;
    writeln!(
        output,
        "별: 백 {} / 흑 {} | 완료 턴: 백 {} / 흑 {}",
        format_half_stars(state.half_star_totals[Player::White.index()]),
        format_half_stars(state.half_star_totals[Player::Black.index()]),
        state.turns_taken[Player::White.index()],
        state.turns_taken[Player::Black.index()]
    )?;
    if let Some(platform) = &state.card_runtime.platform
        && let Some(square) = platform.cell
    {
        writeln!(output, "발판: {square}")?;
    }
    render_statuses(output, state)?;
    render_cards(output, state)
}

fn render_statuses(output: &mut impl Write, state: &GameState) -> io::Result<()> {
    if !state.applied_rules.is_empty() {
        let rules = state
            .applied_rules
            .iter()
            .map(|id| card_name(id))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "적용 RULE: {rules}")?;
    }
    for target in [Player::White, Player::Black] {
        if let Some(status) = state.card_runtime.initiative[target.index()] {
            let elapsed = state.turns_taken[target.index()].saturating_sub(status.start_turn);
            writeln!(
                output,
                "선공권: {} 포획 금지 (남은 {}턴)",
                player_name(target),
                status.limit.saturating_sub(elapsed)
            )?;
        }
        if let Some(status) = state.card_runtime.dice_locks[target.index()] {
            writeln!(
                output,
                "주사위: {} {} 이동 금지 (남은 {}턴)",
                player_name(target),
                kind_name(status.kind),
                status.remaining_turns
            )?;
        }
    }
    render_piece_status_map(
        output,
        state,
        "빙결",
        state
            .card_runtime
            .frozen
            .iter()
            .map(|(piece_id, status)| (*piece_id, status.remaining_turns)),
    )?;
    render_piece_status_map(
        output,
        state,
        "무장해제",
        state
            .card_runtime
            .disarmed
            .iter()
            .map(|(piece_id, remaining)| (*piece_id, *remaining)),
    )?;
    if let Some(status) = state.card_runtime.reposition
        && let Some((square, _)) = state.board.find_by_id(status.piece_id)
    {
        writeln!(
            output,
            "재배치: {square}{}",
            if status.used { " (2차 이동)" } else { "" }
        )?;
    }
    Ok(())
}

fn render_piece_status_map(
    output: &mut impl Write,
    state: &GameState,
    label: &str,
    statuses: impl Iterator<Item = (u32, u32)>,
) -> io::Result<()> {
    let entries = statuses
        .filter_map(|(piece_id, remaining)| {
            state
                .board
                .find_by_id(piece_id)
                .map(|(square, _)| format!("{square}({remaining}턴)"))
        })
        .collect::<Vec<_>>();
    if !entries.is_empty() {
        writeln!(output, "{label}: {}", entries.join(", "))?;
    }
    Ok(())
}

fn render_cards(output: &mut impl Write, state: &GameState) -> io::Result<()> {
    for player in [Player::White, Player::Black] {
        let cards: Vec<_> = state
            .cards
            .iter()
            .filter(|card| card.owner == player)
            .map(|card| {
                let name = implemented_card(&card.definition_id)
                    .map_or(card.definition_id.as_str(), |definition| definition.name_ko);
                let lock = card
                    .available_after_turn
                    .map_or(String::new(), |turn| format!(", {turn}번째 턴 후 활성화"));
                format!("{name}[#{}; 충전 {}{lock}]", card.instance_id, card.charges)
            })
            .collect();
        if !cards.is_empty() {
            writeln!(output, "{} 카드: {}", player_name(player), cards.join(", "))?;
        }
    }
    Ok(())
}

fn render_actions(
    output: &mut impl Write,
    state: &GameState,
    legal_actions: &[Action],
) -> io::Result<()> {
    writeln!(output, "합법 행동 {}개:", legal_actions.len())?;
    for (index, action) in legal_actions.iter().enumerate() {
        writeln!(
            output,
            "  {:>2}. {}",
            index + 1,
            action_description(state, action)
        )?;
    }
    Ok(())
}

fn action_description(state: &GameState, action: &Action) -> String {
    match action {
        Action::Move(movement) => {
            let promotion = movement
                .promotion
                .map_or(String::new(), |kind| format!("={}", kind_code(kind)));
            format!("이동 {}-{}{}", movement.from, movement.to, promotion)
        }
        Action::UseCard {
            instance_id,
            target,
        } => {
            let definition_id = state
                .cards
                .iter()
                .find(|card| card.instance_id == *instance_id)
                .map_or("알 수 없는 카드", |card| card.definition_id.as_str());
            let name = implemented_card(definition_id)
                .map_or(definition_id, |definition| definition.name_ko);
            format!(
                "카드 {name}[#{instance_id}] → {}",
                target_description(target)
            )
        }
        Action::DraftPick { offer_index } => {
            let definition_id = match &state.phase {
                GamePhase::Draft(draft) => draft
                    .offers
                    .get(usize::from(*offer_index))
                    .map(String::as_str)
                    .unwrap_or("알 수 없는 카드"),
                _ => "알 수 없는 카드",
            };
            if let Some(definition) = implemented_card(definition_id) {
                format!(
                    "드래프트 {} ({}, 별 {})",
                    definition.name_ko,
                    definition.id,
                    format_half_stars(definition.half_stars)
                )
            } else {
                format!("드래프트 {definition_id}")
            }
        }
        Action::ResolveChoice {
            choice_id,
            option_index,
        } => {
            let option = match &state.phase {
                GamePhase::AwaitingChoice(choice) if choice.id == *choice_id => choice
                    .options
                    .get(usize::from(*option_index))
                    .map(String::as_str)
                    .unwrap_or("알 수 없는 선택"),
                _ => "알 수 없는 선택",
            };
            format!("선택: {option}")
        }
    }
}

fn render_events(output: &mut impl Write, events: &[Event]) -> io::Result<()> {
    for event in events {
        match event {
            Event::OpeningRuleChecked {
                triggered,
                definition_id,
            } => {
                if *triggered {
                    let name = definition_id
                        .as_deref()
                        .map_or("실행 가능한 RULE 없음", card_name);
                    writeln!(output, "→ RULE CHECK 적중: {name}")?;
                } else {
                    writeln!(output, "→ RULE CHECK 미발동")?;
                }
            }
            Event::RuleApplied { definition_id } => {
                writeln!(output, "→ 전역 RULE 적용: {}", card_name(definition_id))?
            }
            Event::DraftPicked {
                player,
                definition_id,
                ..
            } => writeln!(
                output,
                "→ {}이(가) {}을(를) 선택했습니다.",
                player_name(*player),
                card_name(definition_id)
            )?,
            Event::DraftSkipped { phase, player } => writeln!(
                output,
                "→ {}의 {phase:?} 드래프트를 건너뜁니다 (후보 없음).",
                player_name(*player)
            )?,
            Event::CardUsed {
                owner,
                definition_id,
                ..
            } => writeln!(
                output,
                "→ {}이(가) {}을(를) 사용했습니다.",
                player_name(*owner),
                card_name(definition_id)
            )?,
            Event::PieceCaptured { actor, piece, at } => writeln!(
                output,
                "→ {}이(가) {at}에서 {} {}을(를) 잡았습니다.",
                player_name(*actor),
                player_name(piece.owner),
                kind_name(piece.kind)
            )?,
            Event::PieceRemoved {
                piece, at, cause, ..
            } => writeln!(
                output,
                "→ {at}의 {} {}이(가) 제거되었습니다 ({cause}).",
                player_name(piece.owner),
                kind_name(piece.kind)
            )?,
            Event::ForcedMoveStarted {
                player,
                piece_id,
                source,
            } => writeln!(
                output,
                "→ {} 기물 #{piece_id}의 강제 이동이 시작됩니다 ({source}).",
                player_name(*player)
            )?,
            Event::ChoiceRequested {
                owner, handler_id, ..
            } => writeln!(
                output,
                "→ {}의 추가 선택이 필요합니다 ({handler_id}).",
                player_name(*owner)
            )?,
            Event::PlatformSpawned { square, .. } => match square {
                Some(square) => writeln!(output, "→ 발판이 {square}에 생성되었습니다.")?,
                None => writeln!(output, "→ 발판을 놓을 빈 칸이 없습니다.")?,
            },
            Event::DeathmatchStarted {
                interval_full_turns,
                ..
            } => writeln!(
                output,
                "→ 오버타임 시작: {interval_full_turns}회 연속 무진전 시 별로 판정합니다."
            )?,
            _ => {}
        }
    }
    Ok(())
}

fn render_result(output: &mut impl Write, status: &GameStatus) -> io::Result<()> {
    match status {
        GameStatus::Ongoing => Ok(()),
        GameStatus::Won { winner, reason } => {
            writeln!(
                output,
                "게임 종료: {} 승리 ({reason:?})",
                player_name(*winner)
            )
        }
        GameStatus::Draw { reason } => writeln!(output, "게임 종료: 무승부 ({reason:?})"),
    }
}

fn render_help(output: &mut impl Write) -> io::Result<()> {
    writeln!(output, "명령:")?;
    writeln!(output, "  번호       표시된 드래프트·카드·선택·이동 실행")?;
    writeln!(output, "  e2e4       좌표로 바로 이동 (e2-e4도 가능)")?;
    writeln!(output, "  e7e8q      프로모션: q, r, b, n")?;
    writeln!(output, "  actions    합법 행동 다시 보기")?;
    writeln!(output, "  cards      양쪽 카드 보기")?;
    writeln!(output, "  board      보드 다시 보기")?;
    writeln!(output, "  help/quit  도움말/종료")
}

fn piece_symbol(piece: augment_core::Piece) -> String {
    let symbol = kind_code(piece.kind);
    match piece.owner {
        Player::White => symbol.to_owned(),
        Player::Black => symbol.to_ascii_lowercase(),
    }
}

fn kind_code(kind: PieceKind) -> &'static str {
    match kind {
        PieceKind::Pawn => "P",
        PieceKind::Knight => "N",
        PieceKind::Bishop => "B",
        PieceKind::Rook => "R",
        PieceKind::Queen => "Q",
        PieceKind::King => "K",
        PieceKind::Assassin => "AS",
        PieceKind::Grasshopper => "G",
    }
}

fn kind_name(kind: PieceKind) -> &'static str {
    match kind {
        PieceKind::Pawn => "폰",
        PieceKind::Knight => "나이트",
        PieceKind::Bishop => "비숍",
        PieceKind::Rook => "룩",
        PieceKind::Queen => "퀸",
        PieceKind::King => "킹",
        PieceKind::Assassin => "암살자",
        PieceKind::Grasshopper => "그래스호퍼",
    }
}

fn player_name(player: Player) -> &'static str {
    match player {
        Player::White => "백",
        Player::Black => "흑",
    }
}

fn phase_name(phase: &GamePhase) -> &'static str {
    match phase {
        GamePhase::Playing => "일반 행동",
        GamePhase::ForcedMove { .. } => "강제 이동",
        GamePhase::Draft(_) => "드래프트",
        GamePhase::AwaitingChoice(_) => "카드 선택",
    }
}

fn card_name(definition_id: &str) -> &str {
    implemented_card(definition_id).map_or(definition_id, |definition| definition.name_ko)
}

fn target_description(target: &ActionTarget) -> String {
    match target {
        ActionTarget::None => "대상 없음".to_owned(),
        ActionTarget::Square(square) => square.to_string(),
        ActionTarget::PieceAt(square) => format!("{square}의 기물"),
        ActionTarget::Squares(squares) => squares
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", "),
    }
}

fn format_half_stars(half_stars: u16) -> String {
    if half_stars % 2 == 0 {
        (half_stars / 2).to_string()
    } else {
        format!("{}.5", half_stars / 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_move_variants_and_promotions() {
        assert_eq!(
            parse_move("e2-e4").unwrap(),
            Some(MoveAction::new(
                "e2".parse().unwrap(),
                "e4".parse().unwrap(),
                None
            ))
        );
        assert_eq!(
            parse_move("e7-e8=Q").unwrap(),
            Some(MoveAction::new(
                "e7".parse().unwrap(),
                "e8".parse().unwrap(),
                Some(PieceKind::Queen)
            ))
        );
        assert!(parse_move("e7e8k").is_err());
    }

    #[test]
    fn scripted_hotseat_can_draft_move_and_quit() {
        let engine = Engine::bundled_20260807().unwrap();
        let input = Cursor::new(b"1\n1\ne2e4\nquit\n");
        let mut output = Vec::new();
        play_hotseat(&engine, 20_260_807, input, &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();

        assert!(output.contains("증강체스 터미널 핫시트"));
        assert!(output.contains("턴: 흑 | 단계: 일반 행동 | ply: 1"));
        assert!(output.contains("게임을 종료합니다."));
    }

    #[test]
    fn illegal_move_does_not_end_the_session() {
        let engine = Engine::bundled_20260807().unwrap();
        let input = Cursor::new(b"1\n1\ne2e5\nquit\n");
        let mut output = Vec::new();
        play_hotseat(&engine, 20_260_807, input, &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();

        assert!(output.contains("현재 합법적인 이동이 아닙니다: e2e5"));
        assert!(output.contains("게임을 종료합니다."));
    }
}

use augment_ai::{
    ACTION_V3_SCHEMA_ID, LegalActionSetV3, OBSERVATION_V3_SCHEMA_ID, ObservationV3,
    encode_observation_v3,
};
use augment_core::{
    BUNDLED_RULESET_ID, Board, CardInstance, ChoiceContinuation, DeathmatchState, DelayedHazard,
    DelayedHazardKind, DrawReason, Engine, ForcedMoveContinuation, ForcedMoveSource, FrozenStatus,
    GamePhase, GameState, GameStatus, LargePieceStatus, Mode, PendingChoice, Piece, PieceKind,
    PlatformRuleState, Player,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

const GOLDEN_FIXTURE_SCHEMA_ID: &str = "augmentia.golden.observation-action.v3";
const DEFAULT_FIXTURE_PATH: &str = "train/v3/fixtures/observation_action_v3.json";

#[derive(Debug, Serialize)]
struct GoldenSuiteV3 {
    fixture_schema_id: &'static str,
    observation_schema_id: &'static str,
    action_schema_id: &'static str,
    ruleset_id: &'static str,
    cases: Vec<GoldenCaseV3>,
}

#[derive(Debug, Serialize)]
struct GoldenCaseV3 {
    name: &'static str,
    observation: ObservationV3,
    legal_actions: LegalActionSetV3,
}

fn main() {
    let output = output_path(std::env::args().skip(1).collect());
    let rendered = render_suite();
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("create V3 fixture directory");
    }
    std::fs::write(&output, format!("{rendered}\n")).expect("write V3 golden fixture");
    println!("wrote {}", output.display());
}

fn output_path(args: Vec<String>) -> PathBuf {
    if let Some(index) = args.iter().position(|arg| arg == "--out") {
        return args
            .get(index + 1)
            .map(PathBuf::from)
            .expect("--out requires a path");
    }
    repository_root().join(DEFAULT_FIXTURE_PATH)
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

fn render_suite() -> String {
    serde_json::to_string(&build_suite()).expect("serialize V3 golden fixture")
}

fn build_suite() -> GoldenSuiteV3 {
    let engine = Engine::bundled_20260807().expect("bundled engine");
    let cases = vec![
        golden_case("initial_draft", initial_draft(&engine), &engine),
        golden_case("standard_playing", standard_playing(), &engine),
        golden_case("forced_move", forced_move(), &engine),
        golden_case("awaiting_choice", awaiting_choice(), &engine),
        golden_case("runtime_rich", runtime_rich(), &engine),
        golden_case("promotion", promotion(), &engine),
        golden_case("terminal_draw", terminal_draw(), &engine),
    ];
    GoldenSuiteV3 {
        fixture_schema_id: GOLDEN_FIXTURE_SCHEMA_ID,
        observation_schema_id: OBSERVATION_V3_SCHEMA_ID,
        action_schema_id: ACTION_V3_SCHEMA_ID,
        ruleset_id: BUNDLED_RULESET_ID,
        cases,
    }
}

fn golden_case(name: &'static str, state: GameState, engine: &Engine) -> GoldenCaseV3 {
    GoldenCaseV3 {
        name,
        observation: encode_observation_v3(&state),
        legal_actions: LegalActionSetV3::from_state(&state, engine)
            .expect("golden state must have valid V3 actions"),
    }
}

fn initial_draft(engine: &Engine) -> GameState {
    engine
        .new_game(Mode::Standard, 2026082101)
        .expect("initial draft state")
}

fn standard_playing() -> GameState {
    let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 2026082102);
    state.phase = GamePhase::Playing;
    state
}

fn forced_move() -> GameState {
    let mut state = standard_playing();
    let pawn = state.board.get("e2".parse().unwrap()).unwrap();
    state.phase = GamePhase::ForcedMove {
        piece_id: pawn.id,
        source: ForcedMoveSource::Platform,
        continuation: ForcedMoveContinuation::ResumeTurn,
    };
    state.card_runtime.platform = Some(PlatformRuleState {
        cell: Some("e2".parse().unwrap()),
        spawned_at: 0,
        next_at: 3,
        triggered_piece_ids: [pawn.id].into_iter().collect(),
        nonce: 2,
    });
    state
}

fn awaiting_choice() -> GameState {
    let mut state = standard_playing();
    state.add_card(CardInstance {
        instance_id: 70_001,
        definition_id: "trolley".to_owned(),
        owner: Player::White,
        half_stars: 8,
        charges: 1,
        available_after_turn: None,
    });
    state.phase = GamePhase::AwaitingChoice(PendingChoice {
        id: 91_001,
        owner: Player::White,
        source_card_instance: 70_001,
        handler_id: "trolley".to_owned(),
        options: vec!["bundle:alpha".to_owned(), "bundle:beta".to_owned()],
        continuation: ChoiceContinuation::FinishTurn,
    });
    state
}

fn runtime_rich() -> GameState {
    let mut board = Board::empty();
    board.set(
        "e1".parse().unwrap(),
        Some(Piece::new(Player::White, PieceKind::King).with_id(10_001)),
    );
    board.set(
        "c3".parse().unwrap(),
        Some(Piece::new(Player::White, PieceKind::Wizard).with_id(10_002)),
    );
    board.set(
        "e8".parse().unwrap(),
        Some(Piece::new(Player::Black, PieceKind::King).with_id(20_001)),
    );
    board.set(
        "f6".parse().unwrap(),
        Some(Piece::new(Player::Black, PieceKind::Colossus).with_id(20_002)),
    );
    let mut state = GameState::from_board(
        BUNDLED_RULESET_ID,
        Mode::Standard,
        board,
        Player::White,
        2026082103,
    );
    state.phase = GamePhase::Playing;
    state.ply = 38;
    state.quiet_ply = 4;
    state.actions_remaining = 2;
    state.turns_taken = [19, 19];
    state.completed_drafts = [true, true, false];
    state.deathmatch = Some(DeathmatchState {
        started_at_shared_turn: 18,
        quiet_full_turns: 1,
        interval_full_turns: 10,
        progress_this_round: true,
    });
    state.applied_rules = vec!["black-hole".to_owned(), "portal".to_owned()];
    state.add_card(CardInstance {
        instance_id: 30_001,
        definition_id: "gale".to_owned(),
        owner: Player::White,
        half_stars: 6,
        charges: 1,
        available_after_turn: None,
    });
    state.add_card(CardInstance {
        instance_id: 30_002,
        definition_id: "prophecy".to_owned(),
        owner: Player::Black,
        half_stars: 8,
        charges: 1,
        available_after_turn: Some(20),
    });
    state.card_runtime.hill_king = [true, false];
    state.card_runtime.wizard_mana.insert(10_002, 3);
    state.card_runtime.shielded_piece_ids.insert(10_002);
    state.card_runtime.frozen.insert(
        20_002,
        FrozenStatus {
            source: Player::White,
            remaining_turns: 2,
        },
    );
    state.card_runtime.large_pieces.insert(
        20_002,
        LargePieceStatus {
            anchor: "f6".parse().unwrap(),
            hp: 3,
            max_hp: 4,
        },
    );
    state.card_runtime.black_holes.insert("d4".parse().unwrap());
    state.card_runtime.rule_portals = Some(["b2".parse().unwrap(), "g7".parse().unwrap()]);
    state.card_runtime.delayed_hazards.push(DelayedHazard {
        kind: DelayedHazardKind::Meteor,
        owner: Player::White,
        trigger_after: Player::Black,
        squares: vec!["d5".parse().unwrap(), "e6".parse().unwrap()],
    });
    state
}

fn promotion() -> GameState {
    let mut board = Board::empty();
    board.set(
        "e1".parse().unwrap(),
        Some(Piece::new(Player::White, PieceKind::King).with_id(1)),
    );
    board.set(
        "e8".parse().unwrap(),
        Some(Piece::new(Player::Black, PieceKind::King).with_id(2)),
    );
    board.set(
        "a7".parse().unwrap(),
        Some(Piece::new(Player::White, PieceKind::Pawn).with_id(3)),
    );
    GameState::from_board(
        BUNDLED_RULESET_ID,
        Mode::Standard,
        board,
        Player::White,
        2026082104,
    )
}

fn terminal_draw() -> GameState {
    let mut state = standard_playing();
    state.status = GameStatus::Draw {
        reason: DrawReason::EqualStarsOnRepetition,
    };
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHECKED_IN: &str =
        include_str!("../../../../train/v3/fixtures/observation_action_v3.json");

    #[test]
    fn checked_in_v3_fixture_matches_the_rust_encoder() {
        assert_eq!(CHECKED_IN.trim_end(), render_suite());
    }

    #[test]
    fn suite_covers_every_phase_and_terminal_state() {
        let suite = build_suite();
        let names = suite.cases.iter().map(|case| case.name).collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "initial_draft",
                "standard_playing",
                "forced_move",
                "awaiting_choice",
                "runtime_rich",
                "promotion",
                "terminal_draw",
            ]
        );
        assert!(suite.cases.last().unwrap().legal_actions.is_empty());
    }
}

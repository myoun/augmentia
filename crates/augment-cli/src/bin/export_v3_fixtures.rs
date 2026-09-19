use augment_ai::{
    ACTION_V3_SCHEMA_ID, LegalActionSetV3, OBSERVATION_V3_SCHEMA_ID, ObservationV3,
    PIECE_KIND_VOCABULARY, encode_observation_v3, observation_v3_runtime_text_tokens,
};
use augment_core::{
    BUNDLED_RULESET_ID, Board, CardInstance, ChoiceContinuation, DeathmatchState, DelayedHazard,
    DelayedHazardKind, DrawReason, Engine, ForcedMoveContinuation, ForcedMoveSource, FrozenStatus,
    GamePhase, GameState, GameStatus, LargePieceStatus, Mode, ParrotMemoryStatus, PendingChoice,
    Piece, PieceKind, PlatformRuleState, Player,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

const GOLDEN_FIXTURE_SCHEMA_ID: &str = "augmentia.golden.observation-action.v3";
const RUNTIME_VOCABULARY_SCHEMA_ID: &str = "augmentia.runtime-vocabulary.v3";
const DEFAULT_FIXTURE_PATH: &str = "train/v3/fixtures/observation_action_v3.json";
const DEFAULT_VOCABULARY_PATH: &str = "train/v3/fixtures/runtime_vocabulary_v3.json";

#[derive(Debug, Serialize)]
struct GoldenSuiteV3 {
    fixture_schema_id: &'static str,
    observation_schema_id: &'static str,
    action_schema_id: &'static str,
    ruleset_id: String,
    cases: Vec<GoldenCaseV3>,
}

#[derive(Debug, Serialize)]
struct RuntimeVocabularyV3 {
    schema_id: &'static str,
    ruleset_id: String,
    piece_kind_codes: Vec<u8>,
    tokens: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct GoldenCaseV3 {
    name: &'static str,
    observation: ObservationV3,
    legal_actions: LegalActionSetV3,
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let output = output_path(&args, "--out", DEFAULT_FIXTURE_PATH);
    let vocabulary_output = output_path(&args, "--vocabulary-out", DEFAULT_VOCABULARY_PATH);
    let engine = selected_engine(&args);
    let rendered = render_suite_for_engine(&engine);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("create V3 fixture directory");
    }
    std::fs::write(&output, format!("{rendered}\n")).expect("write V3 golden fixture");
    if let Some(parent) = vocabulary_output.parent() {
        std::fs::create_dir_all(parent).expect("create V3 vocabulary fixture directory");
    }
    std::fs::write(
        &vocabulary_output,
        format!(
            "{}\n",
            render_runtime_vocabulary_for_ruleset(&engine.ruleset().id)
        ),
    )
    .expect("write V3 runtime vocabulary fixture");
    println!("ruleset {}", engine.ruleset().id);
    println!("wrote {}", output.display());
    println!("wrote {}", vocabulary_output.display());
}

fn selected_engine(args: &[String]) -> Engine {
    let requested = args
        .iter()
        .position(|arg| arg == "--ruleset")
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
        .unwrap_or("20260818");
    match requested {
        "20260818" | "augment_chess_20260818" => {
            Engine::bundled_20260818().expect("load 20260818 engine")
        }
        "20260915" | "augment_chess_20260915" => {
            Engine::bundled_20260915().expect("load 20260915 engine")
        }
        other => {
            panic!("--ruleset must be 20260818 or 20260915 (or the full ruleset id), got {other:?}")
        }
    }
}

fn output_path(args: &[String], flag: &str, default: &str) -> PathBuf {
    if let Some(index) = args.iter().position(|arg| arg == flag) {
        return args
            .get(index + 1)
            .map(PathBuf::from)
            .unwrap_or_else(|| panic!("{flag} requires a path"));
    }
    repository_root().join(default)
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

#[allow(dead_code)]
fn render_suite() -> String {
    serde_json::to_string(&build_suite()).expect("serialize V3 golden fixture")
}

fn render_suite_for_engine(engine: &Engine) -> String {
    serde_json::to_string(&build_suite_for_engine(engine)).expect("serialize V3 golden fixture")
}

#[allow(dead_code)]
fn render_runtime_vocabulary() -> String {
    serde_json::to_string(&build_runtime_vocabulary())
        .expect("serialize V3 runtime vocabulary fixture")
}

fn render_runtime_vocabulary_for_ruleset(ruleset_id: &str) -> String {
    serde_json::to_string(&build_runtime_vocabulary_for_ruleset(ruleset_id))
        .expect("serialize V3 runtime vocabulary fixture")
}

#[allow(dead_code)]
fn build_runtime_vocabulary() -> RuntimeVocabularyV3 {
    build_runtime_vocabulary_for_ruleset(BUNDLED_RULESET_ID)
}

fn build_runtime_vocabulary_for_ruleset(ruleset_id: &str) -> RuntimeVocabularyV3 {
    RuntimeVocabularyV3 {
        schema_id: RUNTIME_VOCABULARY_SCHEMA_ID,
        ruleset_id: ruleset_id.to_owned(),
        piece_kind_codes: PIECE_KIND_VOCABULARY
            .iter()
            .map(|(code, _)| *code)
            .collect(),
        tokens: observation_v3_runtime_text_tokens(),
    }
}

#[allow(dead_code)]
fn build_suite() -> GoldenSuiteV3 {
    let engine = Engine::bundled_20260818().expect("bundled engine");
    build_suite_for_engine(&engine)
}

fn build_suite_for_engine(engine: &Engine) -> GoldenSuiteV3 {
    let cases = vec![
        golden_case("initial_draft", initial_draft(engine), engine),
        golden_case("standard_playing", standard_playing(), engine),
        golden_case("forced_move", forced_move(), engine),
        golden_case("forced_effect_move", forced_effect_move(), engine),
        golden_case("quantum_shadow", quantum_shadow(), engine),
        golden_case("awaiting_choice", awaiting_choice(), engine),
        golden_case("runtime_rich", runtime_rich(), engine),
        golden_case("public_pending", public_pending(), engine),
        golden_case("september_six_pieces", september_six_pieces(), engine),
        golden_case("promotion", promotion(), engine),
        golden_case("terminal_draw", terminal_draw(), engine),
    ];
    GoldenSuiteV3 {
        fixture_schema_id: GOLDEN_FIXTURE_SCHEMA_ID,
        observation_schema_id: OBSERVATION_V3_SCHEMA_ID,
        action_schema_id: ACTION_V3_SCHEMA_ID,
        ruleset_id: engine.ruleset().id.clone(),
        cases,
    }
}

fn golden_case(name: &'static str, mut state: GameState, engine: &Engine) -> GoldenCaseV3 {
    state.ruleset_id = engine.ruleset().id.clone();
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

fn forced_effect_move() -> GameState {
    let mut state = standard_playing();
    let pawn = state.board.get("e2".parse().unwrap()).unwrap();
    state.phase = GamePhase::ForcedMove {
        piece_id: pawn.id,
        source: ForcedMoveSource::Effect(augment_core::MIDDLE_EXTRA_MOVE_SOURCE.to_owned()),
        continuation: ForcedMoveContinuation::FinishTurn,
    };
    state
}

fn quantum_shadow() -> GameState {
    let mut state = standard_playing();
    let rook = state.board.get("a1".parse().unwrap()).unwrap();
    state
        .card_runtime
        .quantum_positions
        .insert(rook.id, "d1".parse().unwrap());
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

fn public_pending() -> GameState {
    let mut state = standard_playing();
    state.card_runtime.majesty = [true, false];
    state.card_runtime.proficiency = [false, true];
    state.card_runtime.killer_king = [true, false];
    state.card_runtime.disassembly = [false, true];
    state.card_runtime.overtake = [true, false];
    state.card_runtime.reversal = [false, true];
    state.card_runtime.symmetry = [true, false];
    state.card_runtime.mutation = [false, true];
    state.card_runtime.long_en_passant = [true, false];
    state.card_runtime.vanguard = [false, true];
    // Both owners exercise the append-only Clockwork planes and stable kind 59.
    for square in ["b1", "g8"] {
        let square = square.parse().unwrap();
        let mut piece = state.board.get(square).unwrap();
        piece.kind = PieceKind::Clockwork;
        state.board.set(square, Some(piece));
        state
            .card_runtime
            .outpost_protected_piece_ids
            .insert(piece.id);
        state.card_runtime.nullification_piece_ids.insert(piece.id);
    }
    state.turn = Player::Black;
    state.ply = 23;
    state.turns_taken = [8, 11];
    let white = state.board.get("a2".parse().unwrap()).unwrap();
    let black = state.board.get("h7".parse().unwrap()).unwrap();
    state.card_runtime.capture_the_flag = Some(augment_core::CaptureTheFlagState {
        flags: ["h7".parse().unwrap(), "a2".parse().unwrap()],
        occupations: [
            Some(augment_core::FlagOccupation {
                piece_id: black.id,
                after_owner_turn: 7,
            }),
            Some(augment_core::FlagOccupation {
                piece_id: white.id,
                after_owner_turn: 11,
            }),
        ],
    });
    state.card_runtime.pending_lobsters = vec![
        augment_core::PendingLobster {
            owner: Player::Black,
            square: "c4".parse().unwrap(),
            due_ply: 25,
        },
        augment_core::PendingLobster {
            owner: Player::White,
            square: "e5".parse().unwrap(),
            due_ply: 22,
        },
    ];
    state.card_runtime.pending_portals = vec![
        augment_core::PendingPortal {
            owner: Player::White,
            squares: ["b3".parse().unwrap(), "g4".parse().unwrap()],
            due_turn: 9,
        },
        augment_core::PendingPortal {
            owner: Player::Black,
            squares: ["c3".parse().unwrap(), "f6".parse().unwrap()],
            due_turn: 11,
        },
    ];
    state
}

fn september_six_pieces() -> GameState {
    let mut board = Board::empty();
    board.set(
        "a1".parse().unwrap(),
        Some(Piece::new(Player::White, PieceKind::King).with_id(1)),
    );
    board.set(
        "h8".parse().unwrap(),
        Some(Piece::new(Player::Black, PieceKind::King).with_id(2)),
    );
    let kinds = [
        PieceKind::Parrot,
        PieceKind::Paladin,
        PieceKind::Princess,
        PieceKind::Octopus,
        PieceKind::Hedgehog,
        PieceKind::Campfire,
        PieceKind::Thief,
        PieceKind::Brutus,
        PieceKind::BigBishop,
    ];
    for (index, kind) in kinds.into_iter().enumerate() {
        let file = (index % 6) as u8 + 1;
        let (white_rank, black_rank) = if index < 6 { (2, 5) } else { (3, 4) };
        for (owner, rank, piece_id) in [
            (Player::White, white_rank, 3 + index as u32),
            (Player::Black, black_rank, 103 + index as u32),
        ] {
            let square = augment_core::Square::new(file, rank).unwrap();
            board.set(square, Some(Piece::new(owner, kind).with_id(piece_id)));
        }
    }
    let mut state = GameState::from_board(
        BUNDLED_RULESET_ID,
        Mode::Standard,
        board,
        Player::White,
        2026091706,
    );
    // Asymmetric Parrot memory: White remembered a Clockwork relocation, Black a
    // Trickster sampled Knight; the parity fixture exercises both owners.
    state.card_runtime.nullification_piece_ids.insert(4);
    state.card_runtime.parrot_memory = [
        Some(ParrotMemoryStatus {
            source: PieceKind::Clockwork,
            effective: PieceKind::Clockwork,
        }),
        Some(ParrotMemoryStatus {
            source: PieceKind::Trickster,
            effective: PieceKind::Knight,
        }),
    ];
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
    const CHECKED_IN_RUNTIME_VOCABULARY: &str =
        include_str!("../../../../train/v3/fixtures/runtime_vocabulary_v3.json");

    #[test]
    fn checked_in_v3_fixture_matches_the_rust_encoder() {
        assert_eq!(CHECKED_IN.trim_end(), render_suite());
        assert_eq!(
            CHECKED_IN_RUNTIME_VOCABULARY.trim_end(),
            render_runtime_vocabulary()
        );
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
                "forced_effect_move",
                "quantum_shadow",
                "awaiting_choice",
                "runtime_rich",
                "public_pending",
                "september_six_pieces",
                "promotion",
                "terminal_draw",
            ]
        );
        assert!(suite.cases.last().unwrap().legal_actions.is_empty());
    }
}

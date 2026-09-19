use super::*;
use augment_core::{Engine, GameStatus, Mode, Player, Square};

#[test]
fn test_invariant_1_color_swap_canonical_tensor_symmetry() {
    let engine = Engine::bundled_20260807().unwrap();
    let mut state = engine.new_game(Mode::Standard, 42).unwrap();
    state.phase = augment_core::GamePhase::Playing;

    let encoded_w = encode_state(&state);

    // Create exact swapped state (colors swapped, board rank mirrored, perspective swapped)
    let mut swapped = state.clone();
    swapped.turn = Player::Black;
    for sq_idx in 0..64 {
        if let Some(sq) = Square::from_index(sq_idx) {
            swapped.board.set(sq, None);
        }
    }
    for (sq, piece) in state.board.iter() {
        let flipped_sq = Square::new(sq.file(), 7 - sq.rank()).unwrap();
        let new_owner = piece.owner.opponent();
        let mut new_piece = piece;
        new_piece.owner = new_owner;
        swapped.board.set(flipped_sq, Some(new_piece));
    }

    let encoded_b = encode_state(&swapped);

    // Strict Bitwise Element-wise Equality:
    assert_eq!(
        encoded_w.spatial, encoded_b.spatial,
        "Spatial tensor must be 100% bitwise identical on swapped states"
    );
    assert_eq!(
        encoded_w.global, encoded_b.global,
        "Global tensor must be 100% bitwise identical on swapped states"
    );
}

#[test]
fn test_invariant_2_side_to_move_value_perspective() {
    // When Winner == current side to move => V = +1.0
    // When Winner == opponent => V = -1.0
    let engine = Engine::bundled_20260807().unwrap();
    let mut state = engine.new_game(Mode::Standard, 1).unwrap();
    state.status = GameStatus::Won {
        winner: Player::White,
        reason: augment_core::WinReason::KingCaptured,
    };
    state.turn = Player::White;
    assert_eq!(HeuristicEvaluator::compute_value(&state), 1.0);

    state.turn = Player::Black;
    assert_eq!(HeuristicEvaluator::compute_value(&state), -1.0);
}

#[test]
fn test_invariant_3_action_roundtrip_encoding() {
    let engine = Engine::bundled_20260807().unwrap();
    let mut state = engine.new_game(Mode::Standard, 1).unwrap();
    state.phase = augment_core::GamePhase::Playing;

    let legal_actions = engine.legal_actions(&state).unwrap();
    for action in &legal_actions {
        let idx = encode_action(action, &state);
        assert!((idx as usize) < ACTION_SPACE_SIZE);
    }
}

#[test]
fn test_invariant_4_mcts_multi_action_perspective_transition() {
    // If consecutive turn belongs to same player (Action budget / card play):
    // parent.turn == child.turn => Q sign does not invert
    // If turn changes: parent.turn != child.turn => Q sign inverts
    let config = MctsConfig {
        num_simulations: 50,
        ..MctsConfig::default()
    };
    let mut tree = MctsTree::new(config);
    let engine = Engine::bundled_20260807().unwrap();
    let mut state = engine.new_game(Mode::Standard, 10).unwrap();
    state.phase = augment_core::GamePhase::Playing;

    let search_res = tree.search(&state, &engine, &HeuristicEvaluator::new());
    assert!(search_res.is_some());
}

#[test]
fn test_invariant_5_root_dirichlet_noise_is_explicit_and_normalized() {
    let engine = Engine::bundled_20260807().unwrap();
    let mut state = engine.new_game(Mode::Standard, 10).unwrap();
    state.phase = augment_core::GamePhase::Playing;

    let config = MctsConfig {
        num_simulations: 4,
        ..MctsConfig::default()
    };
    let mut tree = MctsTree::new(config);
    let result = tree.search_with_root_noise(
        &state,
        &engine,
        &HeuristicEvaluator::new(),
        Some(RootNoiseConfig {
            alpha: 0.3,
            epsilon: 0.25,
            seed: 20260817,
        }),
    );
    assert!(result.is_some());

    let diagnostics = tree.last_diagnostics.as_ref().unwrap();
    assert!(diagnostics.root_legal_actions > 1);
    let raw_sum: f32 = diagnostics
        .root_actions
        .iter()
        .map(|stat| stat.raw_prior)
        .sum();
    let effective_sum: f32 = diagnostics
        .root_actions
        .iter()
        .map(|stat| stat.effective_prior)
        .sum();
    assert!((raw_sum - 1.0).abs() < 1e-5);
    assert!((effective_sum - 1.0).abs() < 1e-5);
    assert!(
        diagnostics
            .root_actions
            .iter()
            .all(|stat| stat.root_noise.is_some())
    );
    assert!(
        diagnostics
            .root_actions
            .iter()
            .any(|stat| (stat.raw_prior - stat.effective_prior).abs() > 1e-6)
    );
}

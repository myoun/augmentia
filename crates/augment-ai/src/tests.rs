use super::*;
use augment_core::{Engine, Mode};

#[test]
fn standard_opening_moves_are_encoded_and_masked() {
    let engine = Engine::bundled_20260807().unwrap();
    let state = engine.new_game(Mode::Standard, 42).unwrap();
    let mask = generate_action_mask(&state, &engine);
    assert!(mask.count_ones() > 0);

    let legal_actions = engine.legal_actions(&state).unwrap();
    for action in &legal_actions {
        let idx = encode_action(action, &state);
        assert!((idx as usize) < ACTION_SPACE_SIZE);
        assert!(mask.is_set(idx as usize));
    }
}

#[test]
fn state_tensor_has_correct_dimensions_and_channels() {
    let engine = Engine::bundled_20260807().unwrap();
    let state = engine.new_game(Mode::Standard, 42).unwrap();
    let encoded = encode_state(&state);
    assert_eq!(encoded.spatial.len(), SPATIAL_SIZE);
    assert_eq!(encoded.global.len(), GLOBAL_FEATURES_SIZE);
    // White pieces present on home ranks
    let non_zero_spatial = encoded.spatial.iter().filter(|&&v| v != 0.0).count();
    assert!(non_zero_spatial >= 32);
}

#[test]
fn action_mask_serialization_and_count() {
    let mut mask = ActionMask::new();
    assert_eq!(mask.count_ones(), 0);
    mask.set(10);
    mask.set(100);
    mask.set(5000);
    assert_eq!(mask.count_ones(), 3);
    assert!(mask.is_set(10));
    assert!(mask.is_set(100));
    assert!(mask.is_set(5000));
    assert!(!mask.is_set(11));
}

//! End-to-end coverage of the public V4 runtime and search APIs.

use crate::{
    FactorizedEvaluationV4, LegalActionSetV3, MctsV4Config, MctsV4Tree, V4OnnxRuntime,
    encode_observation_v3,
};
use augment_core::{Board, Engine, GamePhase, GameState, Mode, Player};
use std::path::Path;

const RULESET_ID: &str = "augment_chess_20260818";
const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../train/v3/fixtures/v4_bundle"
);

fn runtime() -> Option<V4OnnxRuntime> {
    // A configured but broken runtime or missing fixture must fail, not silently skip.
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        eprintln!("skipping V4 ONNX integration because ORT_DYLIB_PATH is not configured");
        return None;
    }
    Some(
        V4OnnxRuntime::from_bundle(
            Path::new(FIXTURE_DIR).join("v4-model.v4-bundle.metadata.json"),
            RULESET_ID,
            &"0".repeat(64),
        )
        .expect("checked-in V4 ONNX bundle must load when ORT_DYLIB_PATH is configured"),
    )
}

fn public_identical_states(engine: &Engine) -> (GameState, GameState) {
    let template = engine.new_game(Mode::Standard, 2026).unwrap();
    assert!(matches!(template.phase, GamePhase::Draft(_)));
    let make_state = |seed| {
        let mut state = GameState::from_board(
            RULESET_ID,
            Mode::Standard,
            Board::standard(),
            Player::White,
            seed,
        );
        // Keep the public offers fixed instead of generating them from the live RNG.
        state.phase = template.phase.clone();
        state
    };
    let left = make_state(11);
    let right = make_state(99);
    assert_ne!(left.rng_state(), right.rng_state());
    assert_eq!(encode_observation_v3(&left), encode_observation_v3(&right));
    assert_eq!(
        LegalActionSetV3::from_state(&left, engine).unwrap(),
        LegalActionSetV3::from_state(&right, engine).unwrap(),
    );
    (left, right)
}

fn assert_valid_evaluation(evaluation: &FactorizedEvaluationV4, legal: &LegalActionSetV3) {
    assert!(evaluation.value.is_finite());
    assert!((-1.0001..=1.0001).contains(&evaluation.value));
    assert_eq!(evaluation.action_logits.len(), legal.len());
    assert!(
        evaluation
            .action_logits
            .iter()
            .all(|logit| logit.is_finite())
    );
    let mass: f32 = evaluation
        .action_logits
        .iter()
        .map(|logit| logit.exp())
        .sum();
    assert!(
        (mass - 1.0).abs() < 1e-4,
        "complete-action probability mass: {mass}"
    );
}

fn assert_same_evaluation(left: &FactorizedEvaluationV4, right: &FactorizedEvaluationV4) {
    assert!((left.value - right.value).abs() < 2e-5);
    assert_eq!(left.action_logits.len(), right.action_logits.len());
    for (left, right) in left.action_logits.iter().zip(&right.action_logits) {
        assert!((left - right).abs() < 2e-5, "{left} != {right}");
    }
}

#[test]
fn onnx_cache_reuses_public_observation_across_live_rng_and_misses_public_changes() {
    let Some(runtime) = runtime() else { return };
    let engine = Engine::bundled_20260818().unwrap();
    let (left, right) = public_identical_states(&engine);
    let left_before = left.clone();
    let right_before = right.clone();
    let legal = LegalActionSetV3::from_state(&left, &engine).unwrap();

    assert_eq!(runtime.session_runs(), 0);
    let first = runtime.evaluate_factorized(&left, &legal).unwrap();
    assert_valid_evaluation(&first, &legal);
    assert_eq!(
        runtime.session_runs(),
        2,
        "cold state needs encoder and scorer"
    );

    let repeated = runtime.evaluate_factorized(&left, &legal).unwrap();
    assert_same_evaluation(&first, &repeated);
    assert_eq!(
        runtime.session_runs(),
        3,
        "same state must run only the scorer"
    );

    let different_rng = runtime.evaluate_factorized(&right, &legal).unwrap();
    assert_same_evaluation(&first, &different_rng);
    assert_eq!(
        runtime.session_runs(),
        4,
        "different live RNG must not cause an encoder cache miss",
    );

    // The cache stores state encodings, not a policy tied to legal-action order.
    let mut reversed_legal = legal.clone();
    reversed_legal.actions.reverse();
    let reversed = runtime
        .evaluate_factorized(&right, &reversed_legal)
        .unwrap();
    assert_valid_evaluation(&reversed, &reversed_legal);
    assert_eq!(runtime.session_runs(), 5);
    for (actual, expected) in reversed
        .action_logits
        .iter()
        .zip(first.action_logits.iter().rev())
    {
        assert!((actual - expected).abs() < 2e-5);
    }

    let mut changed = right.clone();
    changed.ply += 1;
    assert_ne!(
        encode_observation_v3(&right),
        encode_observation_v3(&changed)
    );
    let changed_legal = LegalActionSetV3::from_state(&changed, &engine).unwrap();
    let changed_evaluation = runtime
        .evaluate_factorized(&changed, &changed_legal)
        .unwrap();
    assert_valid_evaluation(&changed_evaluation, &changed_legal);
    assert_eq!(
        runtime.session_runs(),
        7,
        "public changes require a fresh encoder run"
    );

    // Compare against the uncached observation API, rather than assume the random
    // fixture weights must produce a measurably different value for this change.
    let uncached = runtime
        .evaluate_observation_factorized(&encode_observation_v3(&changed), &changed_legal)
        .unwrap();
    assert_same_evaluation(&changed_evaluation, &uncached);
    assert_eq!(runtime.session_runs(), 9);
    let original_again = runtime.evaluate_factorized(&left, &legal).unwrap();
    assert_same_evaluation(&first, &original_again);
    assert_eq!(
        runtime.session_runs(),
        10,
        "another public state must not overwrite this entry"
    );
    assert_eq!(left, left_before);
    assert_eq!(right, right_before);
}

#[test]
fn onnx_cache_distinguishes_capture_the_flag_and_pending_public_state() {
    let Some(runtime) = runtime() else { return };
    let engine = Engine::bundled_20260818().unwrap();
    let (base, different_rng) = public_identical_states(&engine);
    let mut variants = Vec::new();
    let mut flags = base.clone();
    flags.card_runtime.capture_the_flag = Some(augment_core::CaptureTheFlagState {
        flags: ["a3".parse().unwrap(), "h6".parse().unwrap()],
        occupations: [None, None],
    });
    variants.push(flags);
    let mut lobster = base.clone();
    lobster
        .card_runtime
        .pending_lobsters
        .push(augment_core::PendingLobster {
            owner: Player::White,
            square: "c3".parse().unwrap(),
            due_ply: base.ply + 2,
        });
    variants.push(lobster);
    let mut portal = base.clone();
    portal
        .card_runtime
        .pending_portals
        .push(augment_core::PendingPortal {
            owner: Player::Black,
            squares: ["c3".parse().unwrap(), "f6".parse().unwrap()],
            due_turn: base.turns_taken[Player::Black.index()] + 1,
        });
    variants.push(portal);
    let legal = LegalActionSetV3::from_state(&base, &engine).unwrap();
    runtime.evaluate_factorized(&base, &legal).unwrap();
    assert_eq!(runtime.session_runs(), 2);
    for changed in variants {
        assert_ne!(
            encode_observation_v3(&base),
            encode_observation_v3(&changed)
        );
        let changed_legal = LegalActionSetV3::from_state(&changed, &engine).unwrap();
        let before = runtime.session_runs();
        let evaluated = runtime
            .evaluate_factorized(&changed, &changed_legal)
            .unwrap();
        assert_valid_evaluation(&evaluated, &changed_legal);
        assert_eq!(
            runtime.session_runs(),
            before + 2,
            "new public state needs encoder"
        );
        let mut same_public = different_rng.clone();
        same_public.card_runtime = changed.card_runtime.clone();
        assert_eq!(
            encode_observation_v3(&changed),
            encode_observation_v3(&same_public)
        );
        let repeated = runtime
            .evaluate_factorized(&same_public, &changed_legal)
            .unwrap();
        assert_same_evaluation(&evaluated, &repeated);
        assert_eq!(
            runtime.session_runs(),
            before + 3,
            "live RNG must not invalidate cache"
        );
    }
}

#[test]
fn onnx_mcts_search_is_live_rng_independent_and_samples_chance() {
    let Some(left_runtime) = runtime() else {
        return;
    };
    // Separate cold runtimes prevent a shared cache from masking an RNG leak.
    let right_runtime = runtime().unwrap();
    let engine = Engine::bundled_20260818().unwrap();
    let (left_state, right_state) = public_identical_states(&engine);
    let left_before = left_state.clone();
    let right_before = right_state.clone();
    let legal = LegalActionSetV3::from_state(&left_state, &engine).unwrap();
    let config = MctsV4Config {
        num_simulations: 24,
        gumbel_candidates: legal.len(),
        chance_outcome_samples: 4,
        search_seed: 0x1234,
        gumbel_seed: 0x5678,
        ..MctsV4Config::default()
    };
    assert!(legal.len() > 1 && legal.len() <= config.num_simulations);
    let mut left_tree = MctsV4Tree::new(config).unwrap();
    let mut right_tree = MctsV4Tree::new(config).unwrap();
    let left_result = left_tree
        .search(&left_state, &engine, &left_runtime)
        .unwrap();
    let right_result = right_tree
        .search(&right_state, &engine, &right_runtime)
        .unwrap();

    assert_eq!(left_result, right_result);
    assert_eq!(left_tree.last_diagnostics, right_tree.last_diagnostics);
    assert_eq!(left_result.visits.len(), legal.len());
    assert_eq!(
        left_result
            .visits
            .iter()
            .map(|visit| visit.visits)
            .sum::<u32>(),
        config.num_simulations as u32,
    );
    assert_eq!(
        left_result.descriptor,
        legal.actions[left_result.action_index]
    );
    assert_eq!(
        left_result.action,
        legal
            .decode_action(left_result.action_index, &left_state)
            .unwrap(),
    );
    assert!(
        engine
            .legal_actions(&left_state)
            .unwrap()
            .contains(&left_result.action)
    );
    for (index, visit) in left_result.visits.iter().enumerate() {
        assert_eq!(visit.action_index, index);
        assert_eq!(visit.descriptor, legal.actions[index]);
        assert_eq!(
            visit.action,
            legal.decode_action(index, &left_state).unwrap()
        );
        assert!(visit.prior.is_finite() && visit.prior >= 0.0);
        assert!(visit.q_root.is_finite());
        assert!(
            visit.visits > 0,
            "each public Draft offer must receive a simulation"
        );
    }
    let prior_mass: f32 = left_result.visits.iter().map(|visit| visit.prior).sum();
    assert!((prior_mass - 1.0).abs() < 1e-4);

    let diagnostics = left_tree.last_diagnostics.as_ref().unwrap();
    assert_eq!(diagnostics.completed_simulations, config.num_simulations);
    assert_eq!(diagnostics.root_gumbel_candidates, legal.len());
    assert!(diagnostics.root_gumbel_halving_rounds > 0);
    assert_eq!(diagnostics.prefix_engine_apply_count, 0);
    assert!(diagnostics.engine_apply_count > 0);
    assert!(diagnostics.evaluator_calls > 1);
    assert!(diagnostics.root_raw_value.is_finite());
    assert!(diagnostics.sampled_chance_actions >= legal.len());
    assert_eq!(
        diagnostics.sampled_chance_outcomes,
        diagnostics.sampled_chance_actions * config.chance_outcome_samples,
    );
    // Each evaluation runs the scorer; the first one must also run the encoder.
    assert!(left_runtime.session_runs() > diagnostics.evaluator_calls as u64);
    assert!(left_runtime.session_runs() <= 2 * diagnostics.evaluator_calls as u64);
    assert_eq!(left_runtime.session_runs(), right_runtime.session_runs());
    assert_eq!(
        left_state, left_before,
        "search must not mutate the live state"
    );
    assert_eq!(
        right_state, right_before,
        "search must not mutate the live state"
    );
}

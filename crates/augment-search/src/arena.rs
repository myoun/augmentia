//! Self-play arena for comparing two search configurations.
//!
//! The arena plays complete games with both sides driven by the same engine, exchanging colours so
//! the first-move advantage cancels out over a batch. Every outcome is classified separately because
//! they mean different things:
//!
//! - a decided game (`win` / `loss` / `draw`) is evidence about playing strength,
//! - an unfinished game (`truncated`) is evidence about the move cap, not about strength,
//! - an engine error (`error`) means the result is unknown, not that one side lost,
//! - a budget overrun (`overrun`) means the clock was exceeded and the move was chosen late.
//!
//! Collapsing those into a win-rate would make a harness that hangs look strong or weak for the
//! wrong reason, so they are counted apart and never folded together.

use augment_core::{Engine, GameStatus, Mode, Player};
use serde::Serialize;

use crate::{Evaluation, SearchConfig, search};

#[path = "arena_recording.rs"]
pub mod recording;

/// A move counts as late only when the search ran this far past its deadline. The search checks the
/// clock between nodes, so the final node can always finish slightly late; that is the mechanism
/// working, not a defect worth relabelling a game over.
const OVERRUN_TOLERANCE_MS: f64 = 25.0;

/// Which side of a pair a game is played from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Side {
    White,
    Black,
}

impl Side {
    pub fn player(self) -> Player {
        match self {
            Self::White => Player::White,
            Self::Black => Player::Black,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArenaConfig {
    /// Search settings shared by both sides. Equal time and equal rules are required for the
    /// comparison to mean anything, so there is no per-side override.
    pub search: SearchConfig,
    /// Maximum player actions before the game is called unfinished.
    pub max_actions: u32,
    /// Games per colour pairing, played both ways so the first-move advantage cancels.
    pub games_per_pairing: u32,
    pub seed: u64,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            search: SearchConfig::default(),
            max_actions: 400,
            games_per_pairing: 1,
            seed: 1,
        }
    }
}

/// How one game ended. These are deliberately distinct outcomes, not a score.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Outcome {
    /// The engine declared a winner.
    Decided,
    /// The action cap was reached while the game was still running.
    Truncated,
    /// The engine or search returned an error, so the result is unknown.
    Error,
    /// A move was chosen but the search exceeded its time budget.
    Overrun,
}

/// Which configuration won a decided game.
///
/// With colour exchanged between games, the configuration is the thing being compared, so a win is
/// recorded by configuration rather than by colour.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Winner {
    First,
    Second,
}

#[derive(Clone, Debug, Serialize)]
pub struct GameRecord {
    /// Which colour the first configuration played.
    pub first_plays: Side,
    pub outcome: Outcome,
    /// Which configuration won, when the outcome is `Decided`.
    pub winner: Option<Winner>,
    pub actions: u32,
    /// Largest single-move budget overrun observed, in milliseconds.
    pub max_overrun_ms: f64,
    /// Set when the outcome is `Error`.
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ArenaSummary {
    pub games: u32,
    /// Decided games where the first configuration won.
    pub first_wins: u32,
    /// Decided games where the first configuration lost.
    pub second_wins: u32,
    pub draws: u32,
    pub truncated: u32,
    pub errors: u32,
    pub overruns: u32,
    /// Decided games only. Unfinished and unknown games are never folded in.
    pub first_score_rate: f64,
    pub notes: Vec<String>,
}

impl ArenaSummary {
    fn from_records(records: &[GameRecord]) -> Self {
        let mut summary = Self {
            games: records.len() as u32,
            ..Default::default()
        };
        for record in records {
            match record.outcome {
                Outcome::Truncated => {
                    summary.truncated += 1;
                    continue;
                }
                Outcome::Error => {
                    summary.errors += 1;
                    continue;
                }
                Outcome::Overrun => {
                    summary.overruns += 1;
                    continue;
                }
                Outcome::Decided => {}
            }
            match record.winner {
                None => summary.draws += 1,
                Some(Winner::First) => summary.first_wins += 1,
                Some(Winner::Second) => summary.second_wins += 1,
            }
        }
        let decided = summary.first_wins + summary.second_wins + summary.draws;
        summary.first_score_rate = if decided == 0 {
            0.0
        } else {
            (f64::from(summary.first_wins) + 0.5 * f64::from(summary.draws)) / f64::from(decided)
        };
        if summary.truncated > 0 {
            summary.notes.push(format!(
                "{} game(s) hit the action cap and are excluded from the score rate",
                summary.truncated
            ));
        }
        if summary.errors > 0 {
            summary.notes.push(format!(
                "{} game(s) errored and are excluded from the score rate",
                summary.errors
            ));
        }
        if summary.overruns > 0 {
            summary.notes.push(format!(
                "{} game(s) exceeded their time budget",
                summary.overruns
            ));
        }
        if decided == 0 {
            summary
                .notes
                .push("no decided games; the score rate is not an estimate".to_owned());
        }
        summary
    }
}

/// The comparison changes only the per-side evaluation or the first side's policy.
#[derive(Clone, Debug)]
pub enum ArenaMatch {
    SelfPlay,
    Evaluation {
        first: Evaluation,
        second: Evaluation,
    },
    Policy(std::sync::Arc<crate::policy::LinearPolicy>),
}

/// Versioned call-boundary event. Start/end pairs share a zero-based action_index.
/// game_end uses the total completed action count instead.
/// If interrupted, the final complete `*_start` without its matching `*_end` identifies the call.
#[derive(Debug, Serialize)]
pub struct ArenaEvent {
    pub schema_version: u32,
    pub run_id: String,
    pub sequence: u64,
    pub elapsed_ms: f64,
    pub unix_ms: u128,
    pub event: &'static str,
    pub game_index: Option<u64>,
    pub action_index: Option<u32>,
    pub data: serde_json::Value,
}

/// Borrowed event sink; an error stops the arena before the next engine call.
pub type ArenaObserver<'a> = &'a mut dyn FnMut(&ArenaEvent) -> std::io::Result<()>;

struct Observer<'a> {
    callback: Option<ArenaObserver<'a>>,
    started: std::time::Instant,
    run_id: String,
    sequence: u64,
}

impl Observer<'_> {
    fn emit(
        &mut self,
        event: &'static str,
        game: Option<u64>,
        action: Option<u32>,
        data: impl FnOnce() -> serde_json::Value,
    ) -> std::io::Result<()> {
        let Some(callback) = self.callback.as_mut() else {
            return Ok(());
        };
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(std::io::Error::other)?
            .as_millis();
        callback(&ArenaEvent {
            schema_version: 1,
            run_id: self.run_id.clone(),
            sequence: self.sequence,
            elapsed_ms: self.started.elapsed().as_secs_f64() * 1000.0,
            unix_ms,
            event,
            game_index: game,
            action_index: action,
            data: data(),
        })?;
        self.sequence += 1;
        Ok(())
    }
}

fn search_settings(config: &SearchConfig) -> serde_json::Value {
    serde_json::json!({
        "time_ms": config.time_limit.as_millis(), "max_depth": config.max_depth,
        "depth_unit": config.depth_unit, "max_nodes": config.max_nodes,
        "chance_samples": config.chance_samples, "search_seed": config.search_seed,
        "evaluation": config.evaluation.label(), "policy": config.policy.as_deref(),
    })
}

fn state_snapshot(state: &augment_core::GameState) -> serde_json::Value {
    // Debug is intentional: GameState has private RNG/runtime fields and no serialization API.
    // Keep a readable board as well, and never pretend this is a restorable state format.
    let board: Vec<_> = state
        .board
        .iter()
        .map(|(square, piece)| {
            serde_json::json!({
                "square": square.to_string(), "owner": format!("{:?}", piece.owner),
                "kind": format!("{:?}", piece.kind), "piece_debug": format!("{piece:?}"),
            })
        })
        .collect();
    serde_json::json!({
        "fingerprint": state.fingerprint(), "turn": format!("{:?}", state.turn),
        "phase": format!("{:?}", state.phase), "status": format!("{:?}", state.status),
        "ply": state.ply, "actions_remaining": state.actions_remaining,
        "board": board, "cards_debug": format!("{:?}", state.cards),
        "full_state_debug": format!("{state:?}"), "restorable": false,
    })
}

/// Run every arena variant with a fallible observer, before and after each potentially slow call.
/// Recording failure stops the run before the next engine call. The legacy APIs below remain usable.
pub fn run_observed(
    engine: &Engine,
    config: ArenaConfig,
    comparison: ArenaMatch,
    observer: ArenaObserver<'_>,
) -> std::io::Result<(Vec<GameRecord>, ArenaSummary)> {
    run_with_callbacks(
        engine,
        config,
        comparison,
        Some(observer),
        &mut |_, _, _| Ok(()),
    )
}

/// Finished-game persistence without per-action snapshots, for callers that do not need a trace.
pub fn run_with_progress(
    engine: &Engine,
    config: ArenaConfig,
    comparison: ArenaMatch,
    progress: &mut dyn FnMut(usize, usize, &GameRecord) -> std::io::Result<()>,
) -> std::io::Result<(Vec<GameRecord>, ArenaSummary)> {
    run_with_callbacks(engine, config, comparison, None, progress)
}

fn run_with_callbacks(
    engine: &Engine,
    config: ArenaConfig,
    comparison: ArenaMatch,
    observer: Option<ArenaObserver<'_>>,
    progress: &mut dyn FnMut(usize, usize, &GameRecord) -> std::io::Result<()>,
) -> std::io::Result<(Vec<GameRecord>, ArenaSummary)> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(std::io::Error::other)?;
    let mut observer = Observer {
        callback: observer,
        started: std::time::Instant::now(),
        run_id: format!("{}-{}", std::process::id(), now.as_nanos()),
        sequence: 0,
    };
    let (variant, first, second, policy) = match &comparison {
        ArenaMatch::SelfPlay => (
            "self_play",
            config.search.evaluation,
            config.search.evaluation,
            config.search.policy.as_deref(),
        ),
        ArenaMatch::Evaluation { first, second } => (
            "evaluation",
            *first,
            *second,
            config.search.policy.as_deref(),
        ),
        ArenaMatch::Policy(policy) => (
            "policy",
            config.search.evaluation,
            config.search.evaluation,
            Some(policy.as_ref()),
        ),
    };
    observer.emit("run_start", None, None, || serde_json::json!({
        "variant": variant, "ruleset": engine.ruleset().id, "ruleset_debug": format!("{:?}", engine.ruleset()),
        "engine_identity": {
            "ruleset_id": augment_core::latest_identity().0,
            "catalog_hash": augment_core::latest_identity().1,
            "card_profile_hash": format!("{:016x}", augment_core::latest_identity().2),
        },
        "package_version": env!("CARGO_PKG_VERSION"),
        "build": if cfg!(debug_assertions) { "debug" } else { "release" },
        "site_parity_verified": false, "search": search_settings(&config.search),
        "first_evaluation": first.label(), "second_evaluation": second.label(), "policy": policy,
        "games_per_pairing": config.games_per_pairing, "total_games": u64::from(config.games_per_pairing) * 2,
        "max_actions": config.max_actions, "base_seed": config.seed,
        "seed_schedule": {"formula": "base_seed.wrapping_add(pairing_index).wrapping_add(first_plays == Black ? 1000003 : 0)",
            "black_offset": 1_000_003, "color_exchange_uses_same_seed": false,
            "independent_samples": false,
            "warning": "Consecutive base-seed batches overlap when games_per_pairing > 1; color-exchanged games use different seeds, not matched-seed pairs. All arithmetic wraps at u64."},
        "indexing": "game_index and pairing_index are zero-based; action start/end share the zero-based action_index; game_end uses total completed actions",
        "snapshot_format": "structured board plus full Rust Debug state; snapshot alone is not deserializable",
        "replay": "Recreate Standard game from game_start.seed under the recorded engine identity, deserialize apply_start.data.action as Action, and apply in action_index order. Only matching successful apply_end confirms completion. If apply_end is missing, preserve the pre-apply state and treat action as pending. Check new_game_end/apply_end fingerprints.",
        "durability": "observer must flush each event; file writer flushes, does not fsync; ignore an incomplete final JSON line after interruption",
    }))?;
    let mut records = Vec::new();
    for pairing in 0..config.games_per_pairing {
        for first_plays in [Side::White, Side::Black] {
            let seed = config.seed.wrapping_add(u64::from(pairing)).wrapping_add(
                if first_plays == Side::White {
                    0
                } else {
                    1_000_003
                },
            );
            let game = records.len() as u64;
            observer.emit("game_start", Some(game), Some(0), || serde_json::json!({
                "pairing_index": pairing, "seed": seed, "search_seed": seed,
                "first_plays": first_plays, "first_evaluation": first.label(), "second_evaluation": second.label(),
                "first_policy": policy.is_some(), "second_policy": !matches!(comparison, ArenaMatch::Policy(_)) && config.search.policy.is_some(),
            }))?;
            let record = play_observed(
                engine,
                &config,
                &comparison,
                first_plays,
                seed,
                game,
                &mut observer,
            )?;
            observer.emit(
                "game_end",
                Some(game),
                Some(record.actions),
                || serde_json::json!({"seed": seed, "record": record}),
            )?;
            progress(
                records.len() + 1,
                (u64::from(config.games_per_pairing) * 2) as usize,
                &record,
            )?;
            records.push(record);
        }
    }
    let summary = ArenaSummary::from_records(&records);
    observer.emit(
        "run_end",
        None,
        None,
        || serde_json::json!({"summary": summary}),
    )?;
    Ok((records, summary))
}

fn play_observed(
    engine: &Engine,
    config: &ArenaConfig,
    comparison: &ArenaMatch,
    first_plays: Side,
    seed: u64,
    game: u64,
    observer: &mut Observer<'_>,
) -> std::io::Result<GameRecord> {
    let mut record = GameRecord {
        first_plays,
        outcome: Outcome::Decided,
        winner: None,
        actions: 0,
        max_overrun_ms: 0.0,
        error: None,
    };
    observer.emit(
        "new_game_start",
        Some(game),
        Some(0),
        || serde_json::json!({"seed": seed, "mode": "Standard"}),
    )?;
    let started = std::time::Instant::now();
    let result = engine.new_game(Mode::Standard, seed);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut state = match result {
        Ok(state) => {
            observer.emit("new_game_end", Some(game), Some(0), || serde_json::json!({"ok": true, "elapsed_ms": elapsed_ms, "state": state_snapshot(&state)}))?;
            state
        }
        Err(error) => {
            observer.emit("new_game_end", Some(game), Some(0), || serde_json::json!({"ok": false, "elapsed_ms": elapsed_ms, "error": error.to_string()}))?;
            record.outcome = Outcome::Error;
            record.error = Some(error.to_string());
            return Ok(record);
        }
    };
    while state.status == GameStatus::Ongoing {
        if record.actions >= config.max_actions {
            record.outcome = Outcome::Truncated;
            return Ok(record);
        }
        let mut search_config = config.search.clone();
        search_config.search_seed = seed;
        let is_first = state.turn == first_plays.player();
        match comparison {
            ArenaMatch::SelfPlay => {}
            ArenaMatch::Evaluation { first, second } => {
                search_config.evaluation = if is_first { *first } else { *second }
            }
            ArenaMatch::Policy(policy) => {
                search_config.policy = if is_first { Some(policy.clone()) } else { None }
            }
        }
        observer.emit("search_start", Some(game), Some(record.actions), || {
            serde_json::json!({
                "seed": seed, "actor": format!("{:?}", state.turn), "is_first": is_first,
                "config": search_settings(&search_config), "state": state_snapshot(&state),
            })
        })?;
        let started = std::time::Instant::now();
        let result = search(engine, &state, search_config);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let report = match result {
            Ok(report) => {
                observer.emit("search_end", Some(game), Some(record.actions), || serde_json::json!({
                    "ok": true, "elapsed_ms": elapsed_ms, "selected_action": report.selected_action.as_ref().map(|a| format!("{a:?}")), "action": report.selected_action.as_ref(), "report": report,
                }))?;
                report
            }
            Err(error) => {
                observer.emit("search_end", Some(game), Some(record.actions), || serde_json::json!({"ok": false, "elapsed_ms": elapsed_ms, "error": error.to_string()}))?;
                record.outcome = Outcome::Truncated;
                record.error = Some(format!("search stopped in {:?}: {error}", state.phase));
                return Ok(record);
            }
        };
        record.max_overrun_ms = record.max_overrun_ms.max(report.budget_overrun_ms);
        let Some(action) = report.selected_action else {
            record.outcome = Outcome::Error;
            record.error = Some("search returned no action".to_owned());
            return Ok(record);
        };
        observer.emit("apply_start", Some(game), Some(record.actions), || serde_json::json!({"selected_action": format!("{action:?}"), "action": action, "state_fingerprint": state.fingerprint()}))?;
        let started = std::time::Instant::now();
        let result = engine.apply(&state, action);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(transition) => {
                observer.emit("apply_end", Some(game), Some(record.actions), || serde_json::json!({
                    "ok": true, "elapsed_ms": elapsed_ms, "state": state_snapshot(&transition.state), "events_debug": format!("{:?}", transition.events),
                }))?;
                state = transition.state;
                record.actions += 1;
            }
            Err(error) => {
                observer.emit("apply_end", Some(game), Some(record.actions), || serde_json::json!({"ok": false, "elapsed_ms": elapsed_ms, "error": error.to_string()}))?;
                record.outcome = Outcome::Error;
                record.error = Some(error.to_string());
                return Ok(record);
            }
        }
    }
    record.winner = match state.status {
        GameStatus::Won { winner, .. } => Some(if winner == first_plays.player() {
            Winner::First
        } else {
            Winner::Second
        }),
        GameStatus::Draw { .. } | GameStatus::Ongoing => None,
    };
    record.outcome = if record.max_overrun_ms > OVERRUN_TOLERANCE_MS {
        Outcome::Overrun
    } else {
        Outcome::Decided
    };
    Ok(record)
}

/// Self-play compatibility API. Use `run_observed` for fallible persistence.
pub fn run(engine: &Engine, config: ArenaConfig) -> (Vec<GameRecord>, ArenaSummary) {
    run_with_callbacks(
        engine,
        config,
        ArenaMatch::SelfPlay,
        None,
        &mut |_, _, _| Ok(()),
    )
    .expect("no-op observer cannot fail")
}

pub fn run_match(
    engine: &Engine,
    config: ArenaConfig,
    first: Evaluation,
    second: Evaluation,
) -> (Vec<GameRecord>, ArenaSummary) {
    run_match_with_progress(engine, config, first, second, &mut |_, _, _| {})
}

/// Legacy finished-game callback, preserved for existing callers. For recording errors, use `run_observed`.
pub fn run_match_with_progress(
    engine: &Engine,
    config: ArenaConfig,
    first: Evaluation,
    second: Evaluation,
    progress: &mut dyn FnMut(usize, usize, &GameRecord),
) -> (Vec<GameRecord>, ArenaSummary) {
    run_with_callbacks(
        engine,
        config,
        ArenaMatch::Evaluation { first, second },
        None,
        &mut |played, total, record| {
            progress(played, total, record);
            Ok(())
        },
    )
    .expect("no-op observer cannot fail")
}

pub fn run_policy_match(
    engine: &Engine,
    config: ArenaConfig,
    policy: std::sync::Arc<crate::policy::LinearPolicy>,
) -> (Vec<GameRecord>, ArenaSummary) {
    run_with_callbacks(
        engine,
        config,
        ArenaMatch::Policy(policy),
        None,
        &mut |_, _, _| Ok(()),
    )
    .expect("no-op observer cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(first_plays: Side, outcome: Outcome, winner: Option<Winner>) -> GameRecord {
        GameRecord {
            first_plays,
            outcome,
            winner,
            actions: 10,
            max_overrun_ms: 0.0,
            error: None,
        }
    }

    /// Unfinished, unknown and late games must never enter the score rate. Folding them in would let
    /// a harness that hangs or crashes look strong.
    #[test]
    fn unfinished_and_unknown_games_are_excluded_from_the_score_rate() {
        let records = vec![
            record(Side::White, Outcome::Decided, Some(Winner::First)),
            record(Side::Black, Outcome::Decided, Some(Winner::First)),
            record(Side::White, Outcome::Truncated, None),
            record(Side::Black, Outcome::Error, None),
            record(Side::White, Outcome::Overrun, Some(Winner::Second)),
        ];
        let summary = ArenaSummary::from_records(&records);
        assert_eq!(summary.games, 5);
        assert_eq!(summary.truncated, 1);
        assert_eq!(summary.errors, 1);
        assert_eq!(summary.overruns, 1);
        assert_eq!(summary.first_wins, 2);
        assert_eq!(summary.second_wins, 0);
        assert!((summary.first_score_rate - 1.0).abs() < 1e-9);
        assert!(
            summary.notes.iter().any(|note| note.contains("action cap")),
            "truncation must be reported: {:?}",
            summary.notes
        );
        assert!(
            summary.notes.iter().any(|note| note.contains("errored")),
            "errors must be reported: {:?}",
            summary.notes
        );
    }

    /// A configuration that wins as White loses as Black, so the batch scores 0.5: colour exchange
    /// is what cancels the first-move advantage.
    #[test]
    fn colour_exchange_cancels_a_first_move_advantage() {
        let records = vec![
            record(Side::White, Outcome::Decided, Some(Winner::First)),
            record(Side::Black, Outcome::Decided, Some(Winner::Second)),
        ];
        let summary = ArenaSummary::from_records(&records);
        assert_eq!(summary.first_wins, 1);
        assert_eq!(summary.second_wins, 1);
        assert!((summary.first_score_rate - 0.5).abs() < 1e-9);
    }

    /// A draw scores half a point and must be counted, not dropped.
    #[test]
    fn draws_score_a_half_point() {
        let records = vec![
            record(Side::White, Outcome::Decided, None),
            record(Side::Black, Outcome::Decided, Some(Winner::First)),
        ];
        let summary = ArenaSummary::from_records(&records);
        assert_eq!(summary.draws, 1);
        assert_eq!(summary.first_wins, 1);
        assert!((summary.first_score_rate - 0.75).abs() < 1e-9);
    }

    /// No decided game means no estimate, and the summary must say so rather than report 0.0 as if
    /// the first configuration had lost everything.
    #[test]
    fn no_decided_games_reports_no_estimate() {
        let records = vec![record(Side::White, Outcome::Truncated, None)];
        let summary = ArenaSummary::from_records(&records);
        assert!(summary.first_score_rate.abs() < 1e-9);
        assert!(
            summary
                .notes
                .iter()
                .any(|note| note.contains("not an estimate")),
            "an empty sample must not read as a score: {:?}",
            summary.notes
        );
    }

    fn quick_config(seed: u64) -> ArenaConfig {
        ArenaConfig {
            search: SearchConfig {
                time_limit: std::time::Duration::from_millis(20),
                max_depth: 1,
                ..Default::default()
            },
            max_actions: 1,
            games_per_pairing: 1,
            seed,
        }
    }

    /// Both sides must be played for every pairing, so a batch cannot be one-sided.
    #[test]
    fn a_pairing_plays_both_colours() {
        let engine = Engine::bundled_latest().unwrap();
        let config = ArenaConfig {
            games_per_pairing: 2,
            ..quick_config(3)
        };
        let (records, summary) = run(&engine, config);
        assert_eq!(records.len(), 4, "two pairings must play four games");
        assert_eq!(summary.games, 4);
        assert_eq!(
            records
                .iter()
                .filter(|r| r.first_plays == Side::White)
                .count(),
            2
        );
        assert_eq!(
            records
                .iter()
                .filter(|r| r.first_plays == Side::Black)
                .count(),
            2
        );
    }

    /// A match must give each configuration both colours, so a result cannot come from the
    /// first-move advantage alone.
    #[test]
    fn a_match_exchanges_colours_between_the_two_evaluations() {
        let engine = Engine::bundled_latest().unwrap();
        let (records, summary) = run_match(
            &engine,
            quick_config(5),
            Evaluation::Positional,
            Evaluation::Material,
        );
        assert_eq!(records.len(), 2, "one pairing plays two games");
        assert_eq!(summary.games, 2);
        assert_eq!(
            records
                .iter()
                .filter(|r| r.first_plays == Side::White)
                .count(),
            1
        );
        assert_eq!(
            records
                .iter()
                .filter(|r| r.first_plays == Side::Black)
                .count(),
            1
        );
    }

    /// A policy match must give the prior to exactly one side and exchange colours, so a result
    /// cannot come from both sides having it or from the first-move advantage.
    #[test]
    fn a_policy_match_exchanges_colours_and_uses_one_prior() {
        let engine = Engine::bundled_latest().unwrap();
        let policy = std::sync::Arc::new(crate::policy::LinearPolicy::zeros());
        let (records, summary) = run_policy_match(&engine, quick_config(11), policy);
        assert_eq!(records.len(), 2, "one pairing plays two games");
        assert_eq!(summary.games, 2);
        assert_eq!(
            records
                .iter()
                .filter(|r| r.first_plays == Side::White)
                .count(),
            1
        );
        assert_eq!(
            records
                .iter()
                .filter(|r| r.first_plays == Side::Black)
                .count(),
            1
        );
    }
}

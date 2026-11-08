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

#[derive(Clone, Copy, Debug)]
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

/// Play `games_per_pairing` games from each colour and return every record.
///
/// Both sides use the same configuration. The point is a reproducible harness that classifies
/// outcomes honestly, not an opponent model.
pub fn run(engine: &Engine, config: ArenaConfig) -> (Vec<GameRecord>, ArenaSummary) {
    let mut records = Vec::new();
    for pairing in 0..config.games_per_pairing {
        for side in [Side::White, Side::Black] {
            let seed = config
                .seed
                .wrapping_add(u64::from(pairing))
                .wrapping_add(if side == Side::White { 0 } else { 1_000_003 });
            records.push(play_one(engine, &config, side, seed));
        }
    }
    let summary = ArenaSummary::from_records(&records);
    (records, summary)
}

/// Pit two configurations against each other.
///
/// This is how an evaluation change is judged: the two search settings differ only in the thing
/// being tested, and colour is exchanged so the first-move advantage cancels. `ArenaConfig` is
/// reused for the shared limits, but each side gets its own evaluation.
pub fn run_match(
    engine: &Engine,
    config: ArenaConfig,
    first: Evaluation,
    second: Evaluation,
) -> (Vec<GameRecord>, ArenaSummary) {
    let mut records = Vec::new();
    for pairing in 0..config.games_per_pairing {
        // Alternate which colour the first configuration holds, so both sides see both colours.
        for (first_plays, _) in [(Side::White, Side::Black), (Side::Black, Side::White)] {
            let seed = config.seed.wrapping_add(u64::from(pairing)).wrapping_add(
                if first_plays == Side::White {
                    0
                } else {
                    1_000_003
                },
            );
            records.push(play_match(
                engine,
                &config,
                first_plays,
                first,
                second,
                seed,
            ));
        }
    }
    let summary = ArenaSummary::from_records(&records);
    (records, summary)
}

/// One game where each side may use a different evaluation.
fn play_match(
    engine: &Engine,
    config: &ArenaConfig,
    first_plays: Side,
    first: Evaluation,
    second: Evaluation,
    seed: u64,
) -> GameRecord {
    let mut record = GameRecord {
        first_plays,
        outcome: Outcome::Decided,
        winner: None,
        actions: 0,
        max_overrun_ms: 0.0,
        error: None,
    };
    let mut state = match engine.new_game(Mode::Standard, seed) {
        Ok(state) => state,
        Err(error) => {
            record.outcome = Outcome::Error;
            record.error = Some(error.to_string());
            return record;
        }
    };

    while state.status == GameStatus::Ongoing {
        if record.actions >= config.max_actions {
            record.outcome = Outcome::Truncated;
            return record;
        }
        // A side's evaluation is chosen by which colour it holds, so colour exchange swaps them.
        let evaluation = if state.turn == first_plays.player() {
            first
        } else {
            second
        };
        let mut search_config = config.search;
        search_config.evaluation = evaluation;
        search_config.search_seed = seed;
        let report = match search(engine, &state, search_config) {
            Ok(report) => report,
            Err(error) => {
                record.outcome = Outcome::Truncated;
                record.error = Some(format!("search stopped in {:?}: {error}", state.phase));
                return record;
            }
        };
        record.max_overrun_ms = record.max_overrun_ms.max(report.budget_overrun_ms);
        let Some(action) = report.selected_action else {
            record.outcome = Outcome::Error;
            record.error = Some("search returned no action".to_owned());
            return record;
        };
        match engine.apply(&state, action) {
            Ok(transition) => {
                state = transition.state;
                record.actions += 1;
            }
            Err(error) => {
                record.outcome = Outcome::Error;
                record.error = Some(error.to_string());
                return record;
            }
        }
    }

    record.winner = match state.status {
        GameStatus::Won { winner, .. } => {
            if winner == first_plays.player() {
                Some(Winner::First)
            } else {
                Some(Winner::Second)
            }
        }
        GameStatus::Draw { .. } | GameStatus::Ongoing => None,
    };
    record.outcome = if record.max_overrun_ms > OVERRUN_TOLERANCE_MS {
        Outcome::Overrun
    } else {
        Outcome::Decided
    };
    record
}

fn play_one(engine: &Engine, config: &ArenaConfig, first_plays: Side, seed: u64) -> GameRecord {
    let mut record = GameRecord {
        first_plays,
        outcome: Outcome::Decided,
        winner: None,
        actions: 0,
        max_overrun_ms: 0.0,
        error: None,
    };
    let mut state = match engine.new_game(Mode::Standard, seed) {
        Ok(state) => state,
        Err(error) => {
            record.outcome = Outcome::Error;
            record.error = Some(error.to_string());
            return record;
        }
    };
    let mut search_config = config.search;
    search_config.search_seed = seed;

    while state.status == GameStatus::Ongoing {
        if record.actions >= config.max_actions {
            record.outcome = Outcome::Truncated;
            return record;
        }
        // Draft and forced-choice phases need their own action selection, which the arena does not
        // implement. It plays whatever `search` returns for the current phase; when the engine
        // offers nothing to search, the game is reported unfinished rather than counted as a loss.
        let report = match search(engine, &state, search_config) {
            Ok(report) => report,
            Err(error) => {
                record.outcome = Outcome::Truncated;
                record.error = Some(format!("search stopped in {:?}: {error}", state.phase));
                return record;
            }
        };
        record.max_overrun_ms = record.max_overrun_ms.max(report.budget_overrun_ms);
        let Some(action) = report.selected_action else {
            record.outcome = Outcome::Error;
            record.error = Some("search returned no action".to_owned());
            return record;
        };
        match engine.apply(&state, action) {
            Ok(transition) => {
                state = transition.state;
                record.actions += 1;
            }
            Err(error) => {
                record.outcome = Outcome::Error;
                record.error = Some(error.to_string());
                return record;
            }
        }
    }

    // `winner` records which CONFIGURATION won, not which colour. With colour exchanged, a win for
    // the configuration that happened to hold White is `First`; the same configuration winning as
    // Black is also `First`. That is what makes a batch comparable across pairings.
    record.winner = match state.status {
        GameStatus::Won { winner, .. } => {
            if winner == first_plays.player() {
                Some(Winner::First)
            } else {
                Some(Winner::Second)
            }
        }
        GameStatus::Draw { .. } | GameStatus::Ongoing => None,
    };
    // An overrun is only worth reporting when it is large enough to have changed the move. The
    // search stops on its own deadline and reports the tiny remainder it spent finishing the
    // current node, so a sub-millisecond figure is normal operation, not a broken clock. Treating
    // any positive value as an overrun would relabel every decided game and hide the result.
    record.outcome = if record.max_overrun_ms > OVERRUN_TOLERANCE_MS {
        Outcome::Overrun
    } else {
        Outcome::Decided
    };
    record
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

    /// Unfinished, unknown and late games must never enter the score rate. Folding them in would
    /// let a harness that hangs or crashes look strong.
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
        // Only the two decided games count, and the first configuration won both.
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
            // Game 1: first configuration held White and won.
            record(Side::White, Outcome::Decided, Some(Winner::First)),
            // Game 2: first configuration held Black and lost, so the second configuration won.
            record(Side::Black, Outcome::Decided, Some(Winner::Second)),
        ];
        let summary = ArenaSummary::from_records(&records);
        assert_eq!(summary.first_wins, 1);
        assert_eq!(summary.second_wins, 1);
        assert!(
            (summary.first_score_rate - 0.5).abs() < 1e-9,
            "mirrored results must score 0.5, got {}",
            summary.first_score_rate
        );
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

    /// Both sides must be played for every pairing, so a batch cannot be one-sided.
    #[test]
    fn a_pairing_plays_both_colours() {
        let engine = Engine::bundled_latest().unwrap();
        let config = ArenaConfig {
            search: SearchConfig {
                time_limit: std::time::Duration::from_millis(20),
                max_depth: 1,
                ..Default::default()
            },
            max_actions: 1,
            games_per_pairing: 2,
            seed: 3,
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
        let config = ArenaConfig {
            search: SearchConfig {
                time_limit: std::time::Duration::from_millis(20),
                max_depth: 1,
                ..Default::default()
            },
            max_actions: 1,
            games_per_pairing: 1,
            seed: 5,
        };
        let (records, summary) = run_match(
            &engine,
            config,
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
            1,
            "the first evaluation must play White once"
        );
        assert_eq!(
            records
                .iter()
                .filter(|r| r.first_plays == Side::Black)
                .count(),
            1,
            "and Black once"
        );
    }
}

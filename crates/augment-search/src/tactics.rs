//! AND-OR tactical diagnostics extracted from the former bootstrap solver.
//! Results cover the engine-enumerated, deterministic full-state tree only.
//! They are not hidden-information strategy proofs or safety certificates.

use augment_core::{
    Action, CLEANUP, Engine, GameState, GameStatus, HYPOCRISY, LegalActionBatch, Player,
    SearchChanceContext,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct TacticalConfig {
    pub max_depth: u8,
    pub max_nodes: u64,
}

impl Default for TacticalConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_nodes: 100_000,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TacticalWin {
    pub actions: Vec<Action>,
    pub distance_actions: u8,
    pub searched_nodes: u64,
    pub scope: &'static str,
}

#[derive(Debug)]
pub enum TacticalError {
    InvalidConfig,
    Engine(String),
    NodeBudgetExceeded,
    ChanceEncountered,
    IncompleteActionSpace,
}

impl fmt::Display for TacticalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "tactical depth and node budgets must be positive"),
            Self::Engine(message) => write!(f, "engine error: {message}"),
            Self::NodeBudgetExceeded => write!(f, "tactical node budget exceeded; result unknown"),
            Self::ChanceEncountered => write!(f, "chance transition encountered; result unknown"),
            Self::IncompleteActionSpace => write!(f, "known omitted card targets; result unknown"),
        }
    }
}
impl std::error::Error for TacticalError {}

/// Find shortest wins against all enumerated deterministic opponent replies.
/// Depth counts actions, including same-player continuations. `None` means no
/// win established within the horizon, never a loss or safety classification.
/// The caller supplies full state; this is a diagnostic, not a player policy.
pub fn find_tactical_win(
    engine: &Engine,
    state: &GameState,
    config: TacticalConfig,
) -> Result<Option<TacticalWin>, TacticalError> {
    if config.max_depth == 0 || config.max_nodes == 0 {
        return Err(TacticalError::InvalidConfig);
    }
    if state.status != GameStatus::Ongoing {
        return Ok(None);
    }
    let legal = actions(engine, state)?;
    let mut context = Context {
        engine,
        root: state.turn,
        max_nodes: config.max_nodes,
        nodes: 0,
        memo: BTreeMap::new(),
    };
    let mut winning = Vec::new();
    let mut distance = u8::MAX;
    for action in legal.iter() {
        context.consume()?;
        let child = apply(&legal, action)?;
        let Some(child_distance) = context.prove(&child, config.max_depth - 1)? else {
            continue;
        };
        let candidate_distance = child_distance.saturating_add(1);
        match candidate_distance.cmp(&distance) {
            std::cmp::Ordering::Less => {
                distance = candidate_distance;
                winning.clear();
                winning.push(action.clone());
            }
            std::cmp::Ordering::Equal => winning.push(action.clone()),
            std::cmp::Ordering::Greater => {}
        }
    }
    Ok((!winning.is_empty()).then_some(TacticalWin {
        actions: winning,
        distance_actions: distance,
        searched_nodes: context.nodes,
        scope: "engine-enumerated/deterministic/full-state-diagnostic",
    }))
}

struct Context<'a> {
    engine: &'a Engine,
    root: Player,
    max_nodes: u64,
    nodes: u64,
    memo: BTreeMap<(u64, u8), Option<u8>>,
}
impl Context<'_> {
    fn consume(&mut self) -> Result<(), TacticalError> {
        if self.nodes >= self.max_nodes {
            return Err(TacticalError::NodeBudgetExceeded);
        }
        self.nodes += 1;
        Ok(())
    }
    fn prove(&mut self, state: &GameState, depth: u8) -> Result<Option<u8>, TacticalError> {
        self.consume()?;
        match state.status {
            GameStatus::Won { winner, .. } => return Ok((winner == self.root).then_some(0)),
            GameStatus::Draw { .. } => return Ok(None),
            GameStatus::Ongoing => {}
        }
        if depth == 0 {
            return Ok(None);
        }
        let key = (state.fingerprint(), depth);
        if let Some(cached) = self.memo.get(&key) {
            return Ok(*cached);
        }
        let legal = actions(self.engine, state)?;
        let result = if state.turn == self.root {
            let mut shortest: Option<u8> = None;
            for action in legal.iter() {
                if let Some(distance) = self.prove(&apply(&legal, action)?, depth - 1)? {
                    shortest = Some(shortest.map_or(distance, |current| current.min(distance)));
                }
            }
            shortest.map(|d| d.saturating_add(1))
        } else {
            let mut longest = 0;
            for action in legal.iter() {
                let Some(distance) = self.prove(&apply(&legal, action)?, depth - 1)? else {
                    self.memo.insert(key, None);
                    return Ok(None);
                };
                longest = longest.max(distance);
            }
            Some(longest.saturating_add(1))
        };
        self.memo.insert(key, result);
        Ok(result)
    }
}

fn actions<'a>(
    engine: &'a Engine,
    state: &'a GameState,
) -> Result<LegalActionBatch<'a>, TacticalError> {
    if state
        .cards
        .iter()
        .any(|card| card.charges > 0 && [CLEANUP, HYPOCRISY].contains(&card.definition_id.as_str()))
    {
        return Err(TacticalError::IncompleteActionSpace);
    }
    let legal = engine
        .legal_action_batch(state)
        .map_err(|e| TacticalError::Engine(e.to_string()))?;
    if legal.is_empty() {
        return Err(TacticalError::Engine(
            "ongoing state has no enumerated actions".into(),
        ));
    }
    Ok(legal)
}

fn apply(legal: &LegalActionBatch<'_>, action: &Action) -> Result<GameState, TacticalError> {
    let mut chance = SearchChanceContext::new(0);
    let child = legal
        .apply_for_search(action, &mut chance)
        .map_err(|e| TacticalError::Engine(e.to_string()))?;
    if child.sampled_chance {
        return Err(TacticalError::ChanceEncountered);
    }
    Ok(child.transition.state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use augment_core::{Board, Mode, Piece, PieceKind, Square};

    fn fixture() -> (Engine, GameState) {
        let engine = Engine::bundled_20260915().unwrap();
        let mut board = Board::empty();
        for (square, owner, kind) in [
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("h1", Player::White, PieceKind::Rook),
        ] {
            board.set(
                square.parse::<Square>().unwrap(),
                Some(Piece::new(owner, kind)),
            );
        }
        let state = GameState::from_board(
            engine.ruleset().id.clone(),
            Mode::Standard,
            board,
            Player::White,
            1,
        );
        (engine, state)
    }
    #[test]
    fn returns_executable_winning_action_without_changing_state() {
        let (engine, state) = fixture();
        let before = state.fingerprint();
        let result = find_tactical_win(
            &engine,
            &state,
            TacticalConfig {
                max_depth: 1,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.distance_actions, 1);
        assert_eq!(result.actions.len(), 1);
        assert!(matches!(
            engine
                .apply(&state, result.actions[0].clone())
                .unwrap()
                .state
                .status,
            GameStatus::Won {
                winner: Player::White,
                ..
            }
        ));
        assert_eq!(before, state.fingerprint());
    }
    #[test]
    fn follows_same_player_second_action() {
        let (engine, mut state) = fixture();
        state.board.set("a1".parse().unwrap(), None);
        state.board.set("h1".parse().unwrap(), None);
        for (sq, owner, kind) in [
            ("b1", Player::White, PieceKind::King),
            ("a2", Player::White, PieceKind::Rook),
            ("a3", Player::Black, PieceKind::Queen),
        ] {
            state
                .board
                .set(sq.parse().unwrap(), Some(Piece::new(owner, kind)));
        }
        state.actions_remaining = 2;
        assert!(
            find_tactical_win(
                &engine,
                &state,
                TacticalConfig {
                    max_depth: 1,
                    ..Default::default()
                }
            )
            .unwrap()
            .is_none()
        );
        let result = find_tactical_win(
            &engine,
            &state,
            TacticalConfig {
                max_depth: 2,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.distance_actions, 2);
    }
    #[test]
    fn budgets_do_not_become_safe_or_losing_results() {
        let (engine, state) = fixture();
        assert!(matches!(
            find_tactical_win(
                &engine,
                &state,
                TacticalConfig {
                    max_nodes: 0,
                    ..Default::default()
                }
            ),
            Err(TacticalError::InvalidConfig)
        ));
        assert!(matches!(
            find_tactical_win(
                &engine,
                &state,
                TacticalConfig {
                    max_nodes: 1,
                    ..Default::default()
                }
            ),
            Err(TacticalError::NodeBudgetExceeded)
        ));
    }
    #[test]
    fn refuses_sampled_draft_chance_and_known_omissions() {
        let (engine, state) = fixture();
        let draft = engine.new_game(Mode::Standard, 1).unwrap();
        assert!(matches!(
            find_tactical_win(&engine, &draft, TacticalConfig::default()),
            Err(TacticalError::ChanceEncountered)
        ));
        let incomplete = engine
            .grant_card(&state, 31, CLEANUP, Player::White)
            .unwrap()
            .state;
        assert!(matches!(
            find_tactical_win(&engine, &incomplete, TacticalConfig::default()),
            Err(TacticalError::IncompleteActionSpace)
        ));
    }
}

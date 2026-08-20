use crate::{ActionV3Error, LegalActionSetV3};
use augment_core::{Engine, GameState, GameStatus, Player};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForcedWinConfigV3 {
    pub max_depth: u8,
    pub max_nodes: u64,
}

impl Default for ForcedWinConfigV3 {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_nodes: 100_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForcedWinSolutionV3 {
    pub optimal_action_indices: Vec<u32>,
    pub distance_decisions: u8,
    pub searched_nodes: u64,
}

#[derive(Debug)]
pub enum ForcedWinErrorV3 {
    InvalidConfig(&'static str),
    Action(ActionV3Error),
    Engine(String),
    NodeBudgetExceeded { max_nodes: u64 },
}

impl fmt::Display for ForcedWinErrorV3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid forced-win config: {message}")
            }
            Self::Action(error) => write!(formatter, "ActionV3 error: {error}"),
            Self::Engine(error) => write!(formatter, "engine error: {error}"),
            Self::NodeBudgetExceeded { max_nodes } => {
                write!(formatter, "forced-win node budget exceeded ({max_nodes})")
            }
        }
    }
}

impl std::error::Error for ForcedWinErrorV3 {}

impl From<ActionV3Error> for ForcedWinErrorV3 {
    fn from(error: ActionV3Error) -> Self {
        Self::Action(error)
    }
}

pub fn solve_forced_win_v3(
    state: &GameState,
    engine: &Engine,
    config: ForcedWinConfigV3,
) -> Result<Option<ForcedWinSolutionV3>, ForcedWinErrorV3> {
    if config.max_depth == 0 {
        return Err(ForcedWinErrorV3::InvalidConfig(
            "max_depth must be positive",
        ));
    }
    if config.max_nodes == 0 {
        return Err(ForcedWinErrorV3::InvalidConfig(
            "max_nodes must be positive",
        ));
    }
    if state.status != GameStatus::Ongoing {
        return Ok(None);
    }

    let root_player = state.turn;
    let legal = LegalActionSetV3::from_state(state, engine)?;
    if legal.is_empty() {
        return Ok(None);
    }
    let mut context = SearchContext {
        engine,
        root_player,
        max_nodes: config.max_nodes,
        searched_nodes: 0,
        memo: BTreeMap::new(),
    };
    let mut winning_actions = Vec::new();
    let mut best_distance = u8::MAX;

    for action_index in 0..legal.len() {
        context.consume_node()?;
        let transition = legal
            .apply(action_index, state, engine)
            .map_err(ForcedWinErrorV3::Action)?;
        let Some(child_distance) =
            context.prove_forced_win(&transition.state, config.max_depth.saturating_sub(1))?
        else {
            continue;
        };
        let distance = child_distance.saturating_add(1);
        match distance.cmp(&best_distance) {
            std::cmp::Ordering::Less => {
                best_distance = distance;
                winning_actions.clear();
                winning_actions.push(action_index as u32);
            }
            std::cmp::Ordering::Equal => winning_actions.push(action_index as u32),
            std::cmp::Ordering::Greater => {}
        }
    }

    if winning_actions.is_empty() {
        Ok(None)
    } else {
        Ok(Some(ForcedWinSolutionV3 {
            optimal_action_indices: winning_actions,
            distance_decisions: best_distance,
            searched_nodes: context.searched_nodes,
        }))
    }
}

struct SearchContext<'a> {
    engine: &'a Engine,
    root_player: Player,
    max_nodes: u64,
    searched_nodes: u64,
    memo: BTreeMap<(u64, u8), Option<u8>>,
}

impl SearchContext<'_> {
    fn consume_node(&mut self) -> Result<(), ForcedWinErrorV3> {
        if self.searched_nodes >= self.max_nodes {
            return Err(ForcedWinErrorV3::NodeBudgetExceeded {
                max_nodes: self.max_nodes,
            });
        }
        self.searched_nodes += 1;
        Ok(())
    }

    fn prove_forced_win(
        &mut self,
        state: &GameState,
        depth_remaining: u8,
    ) -> Result<Option<u8>, ForcedWinErrorV3> {
        self.consume_node()?;
        match state.status {
            GameStatus::Won { winner, .. } => {
                return Ok((winner == self.root_player).then_some(0));
            }
            GameStatus::Draw { .. } => return Ok(None),
            GameStatus::Ongoing => {}
        }
        if depth_remaining == 0 {
            return Ok(None);
        }
        let key = (state.fingerprint(), depth_remaining);
        if let Some(cached) = self.memo.get(&key) {
            return Ok(*cached);
        }

        let legal = LegalActionSetV3::from_state(state, self.engine)?;
        if legal.is_empty() {
            self.memo.insert(key, None);
            return Ok(None);
        }

        let result = if state.turn == self.root_player {
            let mut shortest: Option<u8> = None;
            for action_index in 0..legal.len() {
                let transition = legal
                    .apply(action_index, state, self.engine)
                    .map_err(ForcedWinErrorV3::Action)?;
                if let Some(distance) =
                    self.prove_forced_win(&transition.state, depth_remaining - 1)?
                {
                    shortest = Some(shortest.map_or(distance, |current| current.min(distance)));
                }
            }
            shortest.map(|distance| distance.saturating_add(1))
        } else {
            let mut longest = 0u8;
            for action_index in 0..legal.len() {
                let transition = legal
                    .apply(action_index, state, self.engine)
                    .map_err(ForcedWinErrorV3::Action)?;
                let Some(distance) =
                    self.prove_forced_win(&transition.state, depth_remaining - 1)?
                else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use augment_core::{BUNDLED_RULESET_ID, Board, Mode, Piece, PieceKind, Square};

    fn immediate_capture_state() -> GameState {
        let mut board = Board::empty();
        board.set(
            "a1".parse::<Square>().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King).with_id(1)),
        );
        board.set(
            "h8".parse::<Square>().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King).with_id(2)),
        );
        board.set(
            "h1".parse::<Square>().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook).with_id(3)),
        );
        GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 1)
    }

    #[test]
    fn proves_an_immediate_king_capture_using_action_v3_indices() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = immediate_capture_state();
        let legal = LegalActionSetV3::from_state(&state, &engine).unwrap();
        let solution = solve_forced_win_v3(
            &state,
            &engine,
            ForcedWinConfigV3 {
                max_depth: 1,
                max_nodes: 10_000,
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(solution.distance_decisions, 1);
        assert_eq!(solution.optimal_action_indices.len(), 1);
        let action = legal
            .descriptor(solution.optimal_action_indices[0] as usize)
            .unwrap();
        assert!(matches!(
            action,
            crate::ActionDescriptorV3::Move {
                from: 7,
                to: 63,
                ..
            }
        ));
    }

    #[test]
    fn refuses_zero_search_budgets() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = immediate_capture_state();
        assert!(matches!(
            solve_forced_win_v3(
                &state,
                &engine,
                ForcedWinConfigV3 {
                    max_depth: 1,
                    max_nodes: 0,
                }
            ),
            Err(ForcedWinErrorV3::InvalidConfig(_))
        ));
    }
}

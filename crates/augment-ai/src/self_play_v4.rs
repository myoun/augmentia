//! Reproducible V4 self-play over factorized MCTS targets.

use crate::data_v3::{
    AbsolutePlayerV3, GeneratorProvenanceV3, LimitKindV3, PlayerAgentV3, RulesetRefV3,
    TerminationV3,
};
use crate::{
    ACTION_V3_SCHEMA_ID, CHANCE_SAMPLING_V4_ID, FACTORIZED_POLICY_V4_SCHEMA_ID,
    FactorizedEvaluatorV4, FactorizedMctsTargetV4, GAME_RECORD_V4_SCHEMA_ID, GameRecordV4,
    GameStepV4, LegalActionSetV3, MctsV4Config, MctsV4Tree, OBSERVATION_V3_SCHEMA_ID,
    ROOT_SELECTION_V4_ID, SEARCH_SEMANTICS_V4_ID, encode_observation_v3,
};
use augment_core::{Engine, GameStatus, Mode};
use rand::distributions::WeightedIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct SelfPlayGameConfigV4 {
    pub seed: u64,
    pub max_decisions: usize,
    pub simulations: usize,
    pub chance_outcome_samples: usize,
    pub temperature: f32,
    pub temperature_decisions: usize,
}

impl Default for SelfPlayGameConfigV4 {
    fn default() -> Self {
        Self {
            seed: 2026091401,
            max_decisions: 512,
            simulations: 400,
            chance_outcome_samples: 8,
            temperature: 1.0,
            temperature_decisions: 30,
        }
    }
}

pub struct SelfPlayAgentV4<'a> {
    pub evaluator: &'a dyn FactorizedEvaluatorV4,
    pub provenance: PlayerAgentV3,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelfPlayV4Error(pub String);

impl fmt::Display for SelfPlayV4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "V4 self-play failed: {}", self.0)
    }
}
impl std::error::Error for SelfPlayV4Error {}

#[allow(clippy::too_many_arguments)]
pub fn generate_self_play_game_v4(
    engine: &Engine,
    mode: Mode,
    game_id: String,
    ruleset: RulesetRefV3,
    generator: GeneratorProvenanceV3,
    agents: [&SelfPlayAgentV4<'_>; 2],
    config: SelfPlayGameConfigV4,
) -> Result<GameRecordV4, SelfPlayV4Error> {
    if config.max_decisions == 0
        || config.simulations == 0
        || !(1..=64).contains(&config.chance_outcome_samples)
        || !config.temperature.is_finite()
        || config.temperature < 0.0
    {
        return failure("invalid V4 self-play configuration");
    }
    let mut state = engine
        .new_game(mode.clone(), config.seed)
        .map_err(|error| SelfPlayV4Error(error.to_string()))?;
    let initial_state_fingerprint = fingerprint(state.fingerprint());
    let base_config = MctsV4Config {
        num_simulations: config.simulations,
        chance_outcome_samples: config.chance_outcome_samples,
        search_seed: derive_seed(config.seed, 0, 0x4348_414e_4345_5634),
        gumbel_seed: derive_seed(config.seed, 0, 0x4755_4d42_454c_5634),
        ..MctsV4Config::default()
    };
    let mut trees = [
        MctsV4Tree::new(base_config).map_err(|error| SelfPlayV4Error(error.to_string()))?,
        MctsV4Tree::new(base_config).map_err(|error| SelfPlayV4Error(error.to_string()))?,
    ];
    let mut steps = Vec::new();
    for decision_index in 0..config.max_decisions {
        if state.status != GameStatus::Ongoing {
            break;
        }
        let seat = state.turn.index();
        trees[seat].set_search_seed(derive_seed(
            config.seed,
            decision_index as u64,
            0x4348_414e_4345_5634,
        ));
        trees[seat].set_gumbel_seed(derive_seed(
            config.seed,
            decision_index as u64,
            0x4755_4d42_454c_5634,
        ));
        let legal = LegalActionSetV3::from_state(&state, engine)
            .map_err(|error| SelfPlayV4Error(error.to_string()))?;
        let search = trees[seat]
            .search(&state, engine, agents[seat].evaluator)
            .map_err(|error| SelfPlayV4Error(error.to_string()))?;
        let diagnostics = trees[seat]
            .last_diagnostics
            .as_ref()
            .ok_or_else(|| SelfPlayV4Error("search diagnostics missing".to_owned()))?;
        let mut visits = vec![0u32; legal.len()];
        for item in &search.visits {
            if item.action_index >= visits.len() {
                return failure("search returned an invalid action index");
            }
            visits[item.action_index] = item.visits;
        }
        let temperature = if decision_index < config.temperature_decisions {
            config.temperature
        } else {
            0.0
        };
        let played = sample_visits(
            &visits,
            temperature,
            derive_seed(config.seed, decision_index as u64, 0x5341_4d50_4c45_5634),
        )
        .ok_or_else(|| SelfPlayV4Error("root visits cannot be sampled".to_owned()))?;
        let state_fingerprint = fingerprint(state.fingerprint());
        let observation = encode_observation_v3(&state);
        let transition = legal
            .apply(played, &state, engine)
            .map_err(|error| SelfPlayV4Error(error.to_string()))?;
        let next_state_fingerprint = fingerprint(transition.state.fingerprint());
        let target = FactorizedMctsTargetV4::from_complete_visits(
            &legal.actions,
            visits,
            diagnostics.completed_simulations as u32,
            diagnostics.root_raw_value,
            SEARCH_SEMANTICS_V4_ID,
            ROOT_SELECTION_V4_ID,
            CHANCE_SAMPLING_V4_ID,
            config.chance_outcome_samples as u32,
        )
        .map_err(|error| SelfPlayV4Error(error.to_string()))?;
        steps.push(GameStepV4 {
            decision_index: decision_index as u32,
            engine_ply: state.ply,
            actor: AbsolutePlayerV3::from_player(state.turn),
            state_fingerprint,
            observation,
            legal_actions: legal,
            policy_target: target,
            played_action_index: played as u32,
            next_state_fingerprint,
        });
        state = transition.state;
    }
    let termination =
        TerminationV3::from_engine_status(&state.status, fingerprint(state.fingerprint()))
            .unwrap_or(TerminationV3::Limit {
                limit: LimitKindV3::Decision,
                reason: "max_decisions".to_owned(),
                final_state_fingerprint: fingerprint(state.fingerprint()),
            });
    let mut record = GameRecordV4 {
        record_schema_id: GAME_RECORD_V4_SCHEMA_ID.to_owned(),
        game_id,
        ruleset,
        observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
        action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
        factorized_policy_schema_id: FACTORIZED_POLICY_V4_SCHEMA_ID.to_owned(),
        generator,
        mode_id: mode_id(&mode),
        seed_hex: fingerprint(config.seed),
        players: [agents[0].provenance.clone(), agents[1].provenance.clone()],
        initial_state_fingerprint,
        steps,
        termination,
        trajectory_sha256: String::new(),
    };
    record.trajectory_sha256 = record
        .recompute_trajectory_sha256()
        .map_err(|error| SelfPlayV4Error(error.to_string()))?;
    record
        .validate()
        .map_err(|error| SelfPlayV4Error(error.to_string()))?;
    Ok(record)
}

fn sample_visits(visits: &[u32], temperature: f32, seed: u64) -> Option<usize> {
    if temperature <= f32::EPSILON {
        return visits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.cmp(right.1).then_with(|| right.0.cmp(&left.0)))
            .map(|(index, _)| index);
    }
    let inverse = 1.0 / temperature;
    let weights = visits
        .iter()
        .map(|visit| (*visit as f64).powf(inverse as f64))
        .collect::<Vec<_>>();
    let distribution = WeightedIndex::new(&weights).ok()?;
    let mut rng = StdRng::seed_from_u64(seed);
    Some(rng.sample(distribution))
}

fn derive_seed(base: u64, decision: u64, domain: u64) -> u64 {
    let mut value = base ^ decision.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ domain;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn mode_id(mode: &Mode) -> String {
    match mode {
        Mode::Standard => "standard".to_owned(),
        Mode::Chaos => "chaos".to_owned(),
        Mode::Grand => "grand".to_owned(),
        Mode::Campaign(id) => id.clone(),
    }
}

fn fingerprint(value: u64) -> String {
    format!("{value:016x}")
}

fn failure<T>(message: impl Into<String>) -> Result<T, SelfPlayV4Error> {
    Err(SelfPlayV4Error(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeterministicFactorizedEvaluatorV4;
    use crate::data_v3::{AgentKindV3, ModelArtifactV3};
    use augment_core::Player;

    fn agent<'a>(
        evaluator: &'a DeterministicFactorizedEvaluatorV4,
        seat: Player,
    ) -> SelfPlayAgentV4<'a> {
        SelfPlayAgentV4 {
            evaluator,
            provenance: PlayerAgentV3 {
                seat: AbsolutePlayerV3::from_player(seat),
                agent_kind: AgentKindV3::Mcts,
                agent_id: "v4-test".to_owned(),
                model: Some(ModelArtifactV3 {
                    model_id: "deterministic-v4".to_owned(),
                    sha256: "0".repeat(64),
                }),
                config_id: Some("mcts-v4-test".to_owned()),
            },
        }
    }

    #[test]
    fn generated_limit_game_has_valid_factorized_targets_and_is_reproducible() {
        let engine = Engine::bundled_20260915().unwrap();
        let evaluator = DeterministicFactorizedEvaluatorV4 {
            seed: 17,
            value: 0.0,
        };
        let white = agent(&evaluator, Player::White);
        let black = agent(&evaluator, Player::Black);
        let config = SelfPlayGameConfigV4 {
            seed: 71,
            max_decisions: 2,
            simulations: 16,
            chance_outcome_samples: 4,
            temperature: 1.0,
            temperature_decisions: 2,
        };
        let make = || {
            generate_self_play_game_v4(
                &engine,
                Mode::Standard,
                "v4-test".to_owned(),
                RulesetRefV3 {
                    id: "augment_chess_20260915".to_owned(),
                    manifest_sha256: "0".repeat(64),
                },
                GeneratorProvenanceV3 {
                    run_id: "test".to_owned(),
                    engine_commit: "0".repeat(40),
                    engine_api_version: 1,
                    collector_id: "self-play-v4-test".to_owned(),
                    dirty_worktree: false,
                },
                [&white, &black],
                config,
            )
            .unwrap()
        };
        let first = make();
        let second = make();
        assert_eq!(first.steps, second.steps);
        assert_eq!(first.trajectory_sha256, second.trajectory_sha256);
        assert_eq!(first.steps.len(), 2);
        for step in &first.steps {
            assert_eq!(step.policy_target.completed_simulations, 16);
            assert_eq!(step.policy_target.chance_sampling_id, CHANCE_SAMPLING_V4_ID,);
            assert_eq!(step.policy_target.chance_outcome_samples, 4);
            step.validate().unwrap();
        }
    }
}

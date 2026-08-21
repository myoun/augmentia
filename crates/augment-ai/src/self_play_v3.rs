use crate::data_v3::{
    AbsolutePlayerV3, ErrorStageV3, GAME_RECORD_V3_SCHEMA_ID, GameRecordV3, GameStepV3,
    GeneratorProvenanceV3, LimitKindV3, PlayerAgentV3, PolicyTargetV3, RulesetRefV3, TerminationV3,
    fingerprint_hex,
};
use crate::{
    ACTION_V3_SCHEMA_ID, BatchEvaluatorV3, LegalActionSetV3, MctsV3Config, MctsV3Tree,
    OBSERVATION_V3_SCHEMA_ID, RootNoiseConfigV3, encode_observation_v3,
};
use augment_core::{Engine, GameState, GameStatus, Mode};
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct SelfPlayGameConfigV3 {
    pub seed: u64,
    pub max_decisions: usize,
    pub simulations: usize,
    pub leaf_batch_size: usize,
    pub root_noise_alpha: f32,
    pub root_noise_epsilon: f32,
    pub sampling_temperature: f32,
    pub temperature_decisions: usize,
}

impl Default for SelfPlayGameConfigV3 {
    fn default() -> Self {
        Self {
            seed: 2026082101,
            max_decisions: 512,
            simulations: 400,
            leaf_batch_size: 16,
            root_noise_alpha: 0.3,
            root_noise_epsilon: 0.25,
            sampling_temperature: 1.0,
            temperature_decisions: 30,
        }
    }
}

impl SelfPlayGameConfigV3 {
    fn validate(self) -> Result<Self, SelfPlayV3Error> {
        if self.max_decisions == 0 || self.simulations == 0 || self.leaf_batch_size == 0 {
            return Err(SelfPlayV3Error::Config(
                "decision, simulation, and leaf batch limits must be positive".to_owned(),
            ));
        }
        if !self.root_noise_alpha.is_finite()
            || self.root_noise_alpha <= 0.0
            || !self.root_noise_epsilon.is_finite()
            || !(0.0..=1.0).contains(&self.root_noise_epsilon)
            || !self.sampling_temperature.is_finite()
            || self.sampling_temperature < 0.0
        {
            return Err(SelfPlayV3Error::Config(
                "noise and temperature configuration is invalid".to_owned(),
            ));
        }
        Ok(self)
    }
}

pub struct SelfPlayAgentV3<'a> {
    pub evaluator: &'a dyn BatchEvaluatorV3,
    pub provenance: PlayerAgentV3,
}

#[derive(Debug)]
pub enum SelfPlayV3Error {
    Config(String),
    Initialization(String),
    Record(String),
}

impl fmt::Display for SelfPlayV3Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(formatter, "invalid V3 self-play config: {message}"),
            Self::Initialization(message) => {
                write!(formatter, "V3 self-play initialization failed: {message}")
            }
            Self::Record(message) => write!(formatter, "V3 self-play record failed: {message}"),
        }
    }
}

impl std::error::Error for SelfPlayV3Error {}

#[allow(clippy::too_many_arguments)]
pub fn generate_self_play_game_v3(
    engine: &Engine,
    mode: Mode,
    game_id: String,
    ruleset: RulesetRefV3,
    generator: GeneratorProvenanceV3,
    agents: [&SelfPlayAgentV3<'_>; 2],
    config: SelfPlayGameConfigV3,
) -> Result<GameRecordV3, SelfPlayV3Error> {
    let config = config.validate()?;
    let mut state = engine
        .new_game(mode.clone(), config.seed)
        .map_err(|error| SelfPlayV3Error::Initialization(error.to_string()))?;
    let initial_state_fingerprint = fingerprint_hex(state.fingerprint());
    let mcts_config = MctsV3Config {
        num_simulations: config.simulations,
        leaf_batch_size: config.leaf_batch_size,
        ..MctsV3Config::default()
    };
    let mut trees = [
        MctsV3Tree::new(mcts_config).map_err(|error| SelfPlayV3Error::Config(error.to_string()))?,
        MctsV3Tree::new(mcts_config).map_err(|error| SelfPlayV3Error::Config(error.to_string()))?,
    ];
    let mut steps = Vec::new();
    let mut termination = None;

    for decision_index in 0..config.max_decisions {
        if state.status != GameStatus::Ongoing {
            break;
        }
        let seat = state.turn.index();
        let noise_seed = derive_seed(config.seed, decision_index as u64, 0x4e4f_4953_455f_5633);
        let sample_seed = derive_seed(config.seed, decision_index as u64, 0x5341_4d50_4c45_5633);
        let temperature = if decision_index < config.temperature_decisions {
            config.sampling_temperature
        } else {
            0.0
        };
        let legal_actions = match LegalActionSetV3::from_state(&state, engine) {
            Ok(actions) => actions,
            Err(error) => {
                termination = Some(error_termination(
                    &state,
                    ErrorStageV3::Observation,
                    "legal_action_encoding",
                    error.to_string(),
                ));
                break;
            }
        };
        let search = match trees[seat].search_with_root_noise(
            &state,
            engine,
            agents[seat].evaluator,
            Some(RootNoiseConfigV3 {
                alpha: config.root_noise_alpha,
                epsilon: config.root_noise_epsilon,
                seed: noise_seed,
            }),
        ) {
            Ok(result) => result,
            Err(error) => {
                termination = Some(error_termination(
                    &state,
                    ErrorStageV3::Search,
                    "mcts_v3",
                    error.to_string(),
                ));
                break;
            }
        };
        let diagnostics = trees[seat]
            .last_diagnostics
            .as_ref()
            .ok_or_else(|| SelfPlayV3Error::Record("search diagnostics are missing".to_owned()))?;
        let mut visit_counts = vec![0u32; legal_actions.len()];
        for (_, action_index, visits) in &search.visits {
            if *action_index >= visit_counts.len() {
                return Err(SelfPlayV3Error::Record(
                    "search returned an out-of-range local action index".to_owned(),
                ));
            }
            visit_counts[*action_index] = *visits;
        }
        let sampled =
            MctsV3Tree::sample_action_from_visits(&search.visits, temperature, sample_seed)
                .ok_or_else(|| SelfPlayV3Error::Record("visit sampling failed".to_owned()))?;
        let played_action_index = sampled.1;
        let state_fingerprint = fingerprint_hex(state.fingerprint());
        let observation = encode_observation_v3(&state);
        let transition = match legal_actions.apply(played_action_index, &state, engine) {
            Ok(transition) => transition,
            Err(error) => {
                termination = Some(error_termination(
                    &state,
                    ErrorStageV3::Engine,
                    "apply_selected_action",
                    error.to_string(),
                ));
                break;
            }
        };
        let next_state_fingerprint = fingerprint_hex(transition.state.fingerprint());
        steps.push(GameStepV3 {
            decision_index: decision_index as u32,
            engine_ply: state.ply,
            actor: AbsolutePlayerV3::from_player(state.turn),
            state_fingerprint,
            observation,
            legal_actions,
            policy_target: PolicyTargetV3::MctsVisits {
                visit_counts,
                completed_simulations: config.simulations as u32,
                root_value: diagnostics.root_raw_value,
                noise_seed_hex: Some(fingerprint_hex(noise_seed)),
                sampling_temperature: temperature,
            },
            played_action_index: played_action_index as u32,
            next_state_fingerprint,
        });
        state = transition.state;
    }

    let final_fingerprint = fingerprint_hex(state.fingerprint());
    let termination = termination
        .or_else(|| TerminationV3::from_engine_status(&state.status, final_fingerprint.clone()))
        .unwrap_or(TerminationV3::Limit {
            limit: LimitKindV3::Decision,
            reason: "max_decisions".to_owned(),
            final_state_fingerprint: final_fingerprint,
        });
    let mut record = GameRecordV3 {
        record_schema_id: GAME_RECORD_V3_SCHEMA_ID.to_owned(),
        game_id,
        ruleset,
        observation_schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
        action_schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
        generator,
        mode_id: mode_id(&mode),
        seed_hex: fingerprint_hex(config.seed),
        players: [agents[0].provenance.clone(), agents[1].provenance.clone()],
        initial_state_fingerprint,
        steps,
        termination,
        trajectory_sha256: String::new(),
    };
    record
        .refresh_trajectory_sha256()
        .map_err(|error| SelfPlayV3Error::Record(error.to_string()))?;
    record
        .validate()
        .map_err(|error| SelfPlayV3Error::Record(error.to_string()))?;
    Ok(record)
}

fn error_termination(
    state: &GameState,
    stage: ErrorStageV3,
    code: &str,
    detail: String,
) -> TerminationV3 {
    TerminationV3::Error {
        stage,
        code: code.to_owned(),
        detail: Some(detail),
        final_state_fingerprint: fingerprint_hex(state.fingerprint()),
    }
}

fn derive_seed(seed: u64, decision: u64, domain: u64) -> u64 {
    let mut value = seed ^ decision.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ domain;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EvaluationV3;
    use crate::data_v3::ModelArtifactV3;
    use augment_core::BUNDLED_RULESET_ID;

    struct UniformEvaluator;

    impl BatchEvaluatorV3 for UniformEvaluator {
        fn evaluate_batch_v3(
            &self,
            states: &[GameState],
            engine: &Engine,
        ) -> Result<Vec<EvaluationV3>, crate::MctsV3Error> {
            states
                .iter()
                .map(|state| {
                    let legal = engine.legal_actions(state).unwrap();
                    let prior = 1.0 / legal.len() as f32;
                    Ok(EvaluationV3 {
                        priors: legal
                            .into_iter()
                            .enumerate()
                            .map(|(index, action)| (action, index, prior))
                            .collect(),
                        value: 0.0,
                    })
                })
                .collect()
        }
    }

    fn agent(seat: AbsolutePlayerV3) -> SelfPlayAgentV3<'static> {
        static EVALUATOR: UniformEvaluator = UniformEvaluator;
        SelfPlayAgentV3 {
            evaluator: &EVALUATOR,
            provenance: PlayerAgentV3 {
                seat,
                agent_kind: crate::data_v3::AgentKindV3::Mcts,
                agent_id: "uniform-v3".to_owned(),
                model: Some(ModelArtifactV3 {
                    model_id: "uniform".to_owned(),
                    sha256: "a".repeat(64),
                }),
                config_id: Some("test".to_owned()),
            },
        }
    }

    #[test]
    fn generated_limit_game_replays_and_has_no_value_target() {
        let engine = Engine::bundled_20260807().unwrap();
        let white = agent(AbsolutePlayerV3::White);
        let black = agent(AbsolutePlayerV3::Black);
        let game = generate_self_play_game_v3(
            &engine,
            Mode::Standard,
            "selfplay-test".to_owned(),
            RulesetRefV3 {
                id: BUNDLED_RULESET_ID.to_owned(),
                manifest_sha256: "b".repeat(64),
            },
            GeneratorProvenanceV3 {
                run_id: "selfplay-run".to_owned(),
                engine_commit: "c".repeat(40),
                engine_api_version: 1,
                collector_id: "self-play-v3".to_owned(),
                dirty_worktree: false,
            },
            [&white, &black],
            SelfPlayGameConfigV3 {
                seed: 17,
                max_decisions: 2,
                simulations: 24,
                leaf_batch_size: 8,
                temperature_decisions: 1,
                ..SelfPlayGameConfigV3::default()
            },
        )
        .unwrap();
        assert_eq!(game.steps.len(), 2);
        assert!(matches!(game.termination, TerminationV3::Limit { .. }));
        assert!(
            game.steps
                .iter()
                .all(|step| matches!(step.policy_target, PolicyTargetV3::MctsVisits { .. }))
        );
        assert!(game.value_target(0).is_none());
        game.validate_replay(&engine).unwrap();
    }

    #[test]
    fn seeds_make_the_trajectory_reproducible() {
        let engine = Engine::bundled_20260807().unwrap();
        let white = agent(AbsolutePlayerV3::White);
        let black = agent(AbsolutePlayerV3::Black);
        let generate = || {
            generate_self_play_game_v3(
                &engine,
                Mode::Standard,
                "selfplay-test".to_owned(),
                RulesetRefV3 {
                    id: BUNDLED_RULESET_ID.to_owned(),
                    manifest_sha256: "b".repeat(64),
                },
                GeneratorProvenanceV3 {
                    run_id: "selfplay-run".to_owned(),
                    engine_commit: "c".repeat(40),
                    engine_api_version: 1,
                    collector_id: "self-play-v3".to_owned(),
                    dirty_worktree: false,
                },
                [&white, &black],
                SelfPlayGameConfigV3 {
                    seed: 23,
                    max_decisions: 3,
                    simulations: 24,
                    leaf_batch_size: 8,
                    ..SelfPlayGameConfigV3::default()
                },
            )
            .unwrap()
        };
        assert_eq!(generate().trajectory_sha256, generate().trajectory_sha256);
    }
}

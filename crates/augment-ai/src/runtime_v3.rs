use crate::{
    ACTION_V3_SCHEMA_ID, ActionDescriptorV3, CARD_RUNTIME_SOURCE_FIELDS, LegalActionSetV3,
    OBSERVATION_V3_SCHEMA_ID, OBSERVATION_V3_SPATIAL_PLANES, ObservationV3, PIECE_KIND_VOCABULARY,
    RuntimeValueV3, encode_observation_v3, observation_v3_runtime_text_tokens,
};
use augment_core::{Action, Engine, GameState};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

pub const MODEL_METADATA_V3_SCHEMA_ID: &str = "augmentia.onnx-model-metadata.v3";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum V3ExecutionProvider {
    Cpu,
    Cuda,
}

impl V3ExecutionProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct V3RuntimeOptions {
    pub execution_provider: V3ExecutionProvider,
    pub intra_threads: usize,
}

impl Default for V3RuntimeOptions {
    fn default() -> Self {
        Self {
            execution_provider: V3ExecutionProvider::Cpu,
            intra_threads: 1,
        }
    }
}

#[derive(Debug)]
pub enum V3RuntimeError {
    Io(String),
    Metadata(String),
    Tensor(String),
    Inference(String),
    Engine(String),
}

impl fmt::Display for V3RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "V3 runtime I/O error: {message}"),
            Self::Metadata(message) => write!(formatter, "V3 model metadata error: {message}"),
            Self::Tensor(message) => write!(formatter, "V3 tensor error: {message}"),
            Self::Inference(message) => write!(formatter, "V3 ONNX inference error: {message}"),
            Self::Engine(message) => write!(formatter, "V3 engine error: {message}"),
        }
    }
}

impl std::error::Error for V3RuntimeError {}

#[derive(Clone, Debug)]
pub struct EvaluationV3 {
    pub priors: Vec<(Action, usize, f32)>,
    pub value: f32,
}

#[derive(Clone, Debug)]
pub struct CandidateEvaluationV3 {
    /// Raw policy logits paired with the original local action indices.
    pub logits: Vec<(Action, usize, f32)>,
    pub value: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct VocabularyArtifactV3 {
    pub(crate) schema_id: String,
    pub(crate) tokens: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(crate) struct ModelInputV3 {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) rank: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct SplitGraphContractV3 {
    path: String,
    sha256: String,
    inputs: Vec<ModelInputV3>,
    outputs: Vec<ModelInputV3>,
}

#[derive(Clone, Debug, Deserialize)]
struct SplitPolicyBundleV3 {
    schema_id: String,
    state_encoder: SplitGraphContractV3,
    candidate_scorer: SplitGraphContractV3,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelMetadataV3 {
    metadata_schema_id: String,
    observation_schema_id: String,
    action_schema_id: String,
    ruleset_id: String,
    ruleset_manifest_sha256: String,
    onnx_sha256: String,
    piece_kind_vocabulary_size: usize,
    vocabulary: VocabularyArtifactV3,
    inputs: Vec<ModelInputV3>,
    outputs: Vec<String>,
    #[serde(default)]
    split_policy_bundle: Option<SplitPolicyBundleV3>,
}

#[derive(Clone, Debug)]
pub(crate) struct VocabularyV3 {
    lookup: HashMap<String, i64>,
}

impl VocabularyV3 {
    pub(crate) fn from_artifact(artifact: &VocabularyArtifactV3) -> Result<Self, V3RuntimeError> {
        if artifact.schema_id != "augmentia.model-vocabulary.v3"
            || artifact.tokens.first().map(String::as_str) != Some("<none>")
        {
            return metadata("invalid V3 vocabulary artifact");
        }
        let lookup: HashMap<_, _> = artifact
            .tokens
            .iter()
            .enumerate()
            .map(|(index, token)| (token.clone(), index as i64))
            .collect();
        if lookup.len() != artifact.tokens.len() {
            return metadata("V3 vocabulary contains duplicate tokens");
        }
        Ok(Self { lookup })
    }

    pub(crate) fn index(&self, token: Option<&str>, path: &str) -> Result<i64, V3RuntimeError> {
        let Some(token) = token else { return Ok(0) };
        self.lookup.get(token).copied().ok_or_else(|| {
            V3RuntimeError::Tensor(format!("{path} contains unknown token {token:?}"))
        })
    }
}

struct SplitPolicyRuntimeV3 {
    state_encoder: Mutex<Session>,
    candidate_scorer: Mutex<Session>,
    cache: Mutex<HashMap<u64, Vec<CachedStateEncodingV3>>>,
}

#[derive(Clone)]
struct CachedStateEncodingV3 {
    state: GameState,
    tensors: HashMap<String, TensorDataV3>,
    value: f32,
}

pub struct V3OnnxRuntime {
    session: Mutex<Session>,
    split_policy: Option<SplitPolicyRuntimeV3>,
    metadata: ModelMetadataV3,
    vocabulary: VocabularyV3,
    session_runs: AtomicU64,
    options: V3RuntimeOptions,
}

impl V3OnnxRuntime {
    pub fn from_files(
        model_path: impl AsRef<Path>,
        metadata_path: impl AsRef<Path>,
        expected_ruleset_id: &str,
        expected_ruleset_manifest_sha256: &str,
    ) -> Result<Self, V3RuntimeError> {
        Self::from_files_with_options(
            model_path,
            metadata_path,
            expected_ruleset_id,
            expected_ruleset_manifest_sha256,
            V3RuntimeOptions::default(),
        )
    }

    pub fn from_files_with_options(
        model_path: impl AsRef<Path>,
        metadata_path: impl AsRef<Path>,
        expected_ruleset_id: &str,
        expected_ruleset_manifest_sha256: &str,
        options: V3RuntimeOptions,
    ) -> Result<Self, V3RuntimeError> {
        if options.intra_threads == 0 {
            return Err(V3RuntimeError::Inference(
                "ONNX intra_threads must be positive".to_owned(),
            ));
        }
        let model_path = model_path.as_ref();
        let metadata_path = metadata_path.as_ref();
        let metadata: ModelMetadataV3 = serde_json::from_slice(
            &fs::read(metadata_path).map_err(|error| V3RuntimeError::Io(error.to_string()))?,
        )
        .map_err(|error| V3RuntimeError::Metadata(error.to_string()))?;
        validate_metadata(
            &metadata,
            model_path,
            expected_ruleset_id,
            expected_ruleset_manifest_sha256,
        )?;
        let vocabulary = VocabularyV3::from_artifact(&metadata.vocabulary)?;
        validate_required_runtime_vocabulary(&vocabulary)?;
        let session = build_session(model_path, options)?;
        let split_policy = if let Some(bundle) = &metadata.split_policy_bundle {
            let base = metadata_path.parent().unwrap_or_else(|| Path::new("."));
            let (encoder_path, scorer_path) =
                validate_split_policy_bundle(bundle, &metadata, base)?;
            let encoder = build_session(&encoder_path, options)?;
            let scorer = build_session(&scorer_path, options)?;
            validate_session_contract(
                &encoder,
                &bundle.state_encoder.inputs,
                &bundle.state_encoder.outputs,
                "state encoder",
            )?;
            validate_session_contract(
                &scorer,
                &bundle.candidate_scorer.inputs,
                &bundle.candidate_scorer.outputs,
                "candidate scorer",
            )?;
            Some(SplitPolicyRuntimeV3 {
                state_encoder: Mutex::new(encoder),
                candidate_scorer: Mutex::new(scorer),
                cache: Mutex::new(HashMap::new()),
            })
        } else {
            None
        };
        Ok(Self {
            session: Mutex::new(session),
            split_policy,
            metadata,
            vocabulary,
            session_runs: AtomicU64::new(0),
            options,
        })
    }

    pub fn session_runs(&self) -> u64 {
        self.session_runs.load(Ordering::Relaxed)
    }

    pub fn options(&self) -> V3RuntimeOptions {
        self.options
    }

    pub fn evaluate(
        &self,
        state: &GameState,
        engine: &Engine,
    ) -> Result<EvaluationV3, V3RuntimeError> {
        self.evaluate_batch(std::slice::from_ref(state), engine)
            .map(|mut values| values.remove(0))
    }

    pub fn evaluate_batch(
        &self,
        states: &[GameState],
        engine: &Engine,
    ) -> Result<Vec<EvaluationV3>, V3RuntimeError> {
        if states.is_empty() {
            return tensor_error("cannot evaluate an empty state batch");
        }
        let mut legal_sets = Vec::with_capacity(states.len());
        for state in states {
            let legal = LegalActionSetV3::from_state(state, engine)
                .map_err(|error| V3RuntimeError::Engine(error.to_string()))?;
            legal_sets.push(legal);
        }
        let raw = self.evaluate_batch_with_legal_sets(states, &legal_sets)?;
        let mut evaluations = Vec::with_capacity(states.len());
        for ((state, legal), (logits, value)) in states.iter().zip(&legal_sets).zip(raw.into_iter())
        {
            let probabilities = softmax(&logits)?;
            let mut priors = Vec::with_capacity(legal.len());
            for (index, probability) in probabilities.into_iter().enumerate() {
                let action = legal
                    .decode_action(index, state)
                    .map_err(|error| V3RuntimeError::Engine(error.to_string()))?;
                priors.push((action, index, probability));
            }
            evaluations.push(EvaluationV3 { priors, value });
        }
        Ok(evaluations)
    }

    pub fn evaluate_candidates(
        &self,
        state: &GameState,
        engine: &Engine,
        action_indices: &[usize],
    ) -> Result<CandidateEvaluationV3, V3RuntimeError> {
        if action_indices.is_empty() {
            return tensor_error("candidate evaluation requires at least one action");
        }
        let full = LegalActionSetV3::from_state(state, engine)
            .map_err(|error| V3RuntimeError::Engine(error.to_string()))?;
        let mut seen = std::collections::BTreeSet::new();
        let mut descriptors = Vec::with_capacity(action_indices.len());
        for index in action_indices {
            if !seen.insert(*index) {
                return tensor_error("candidate action indices must be unique");
            }
            let descriptor = full.descriptor(*index).ok_or_else(|| {
                V3RuntimeError::Tensor(format!(
                    "candidate action index {index} exceeds legal action count {}",
                    full.len()
                ))
            })?;
            descriptors.push(descriptor.clone());
        }
        let subset = LegalActionSetV3 {
            schema_id: full.schema_id.clone(),
            actions: descriptors,
        };
        let (logits, value) = if let Some(split) = &self.split_policy {
            self.evaluate_split_candidates(state, &subset, split)?
        } else {
            let mut raw = self.evaluate_batch_with_legal_sets(
                std::slice::from_ref(state),
                std::slice::from_ref(&subset),
            )?;
            raw.remove(0)
        };
        let mut scored = Vec::with_capacity(logits.len());
        for ((subset_index, logit), original_index) in logits
            .into_iter()
            .enumerate()
            .zip(action_indices.iter().copied())
        {
            let action = subset
                .decode_action(subset_index, state)
                .map_err(|error| V3RuntimeError::Engine(error.to_string()))?;
            scored.push((action, original_index, logit));
        }
        Ok(CandidateEvaluationV3 {
            logits: scored,
            value,
        })
    }

    fn evaluate_split_candidates(
        &self,
        state: &GameState,
        legal: &LegalActionSetV3,
        split: &SplitPolicyRuntimeV3,
    ) -> Result<(Vec<f32>, f32), V3RuntimeError> {
        if state.ruleset_id != self.metadata.ruleset_id {
            return metadata(format!(
                "state ruleset {} differs from model ruleset {}",
                state.ruleset_id, self.metadata.ruleset_id
            ));
        }
        let bundle = self.metadata.split_policy_bundle.as_ref().ok_or_else(|| {
            V3RuntimeError::Metadata("split runtime has no bundle contract".to_owned())
        })?;
        let observation = encode_observation_v3(state);
        let mut tensors = tensorize_v3(
            std::slice::from_ref(&observation),
            std::slice::from_ref(legal),
            &self.vocabulary,
        )?;
        let fingerprint = state.fingerprint();
        let cached = {
            let cache = split.cache.lock().map_err(|_| {
                V3RuntimeError::Inference("state encoding cache lock poisoned".to_owned())
            })?;
            cache
                .get(&fingerprint)
                .and_then(|entries| entries.iter().find(|entry| entry.state == *state))
                .cloned()
        };
        let encoding = if let Some(cached) = cached {
            cached
        } else {
            let mut inputs = Vec::with_capacity(bundle.state_encoder.inputs.len());
            for declared in &bundle.state_encoder.inputs {
                let tensor = tensors.get(&declared.name).cloned().ok_or_else(|| {
                    V3RuntimeError::Tensor(format!("missing state encoder input {}", declared.name))
                })?;
                if tensor.rank() != declared.rank || tensor.dtype() != declared.dtype {
                    return tensor_error(format!(
                        "state encoder input {} type/rank drifted",
                        declared.name
                    ));
                }
                inputs.push((declared.name.clone(), tensor.into_ort()?));
            }
            let mut session = split.state_encoder.lock().map_err(|_| {
                V3RuntimeError::Inference("state encoder session lock poisoned".to_owned())
            })?;
            let outputs = session
                .run(inputs)
                .map_err(|error| V3RuntimeError::Inference(error.to_string()))?;
            self.session_runs.fetch_add(1, Ordering::Relaxed);
            let mut encoded = HashMap::new();
            for name in [
                "state_embedding",
                "board_tokens",
                "piece_entities",
                "card_entities",
            ] {
                let output = outputs.get(name).ok_or_else(|| {
                    V3RuntimeError::Inference(format!("state encoder omitted output {name}"))
                })?;
                let (shape, values) = output
                    .try_extract_tensor::<f32>()
                    .map_err(|error| V3RuntimeError::Inference(error.to_string()))?;
                let shape = shape
                    .iter()
                    .map(|dimension| usize::try_from(*dimension))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| V3RuntimeError::Tensor(format!("invalid {name} shape")))?;
                if values.iter().any(|value| !value.is_finite()) {
                    return tensor_error(format!("{name} contains NaN or infinity"));
                }
                encoded.insert(
                    name.to_owned(),
                    TensorDataV3::F32 {
                        shape,
                        values: values.to_vec(),
                    },
                );
            }
            let (_, values) = outputs["value"]
                .try_extract_tensor::<f32>()
                .map_err(|error| V3RuntimeError::Inference(error.to_string()))?;
            if values.len() != 1
                || !values[0].is_finite()
                || !(-1.0001..=1.0001).contains(&values[0])
            {
                return tensor_error("state encoder returned an invalid value");
            }
            let cached = CachedStateEncodingV3 {
                state: state.clone(),
                tensors: encoded,
                value: values[0],
            };
            let mut cache = split.cache.lock().map_err(|_| {
                V3RuntimeError::Inference("state encoding cache lock poisoned".to_owned())
            })?;
            let entries = cache.entry(fingerprint).or_default();
            if !entries.iter().any(|entry| entry.state == *state) {
                entries.push(cached.clone());
            }
            cached
        };

        for (name, tensor) in &encoding.tensors {
            tensors.insert(name.clone(), tensor.clone());
        }
        let mut inputs = Vec::with_capacity(bundle.candidate_scorer.inputs.len());
        for declared in &bundle.candidate_scorer.inputs {
            let tensor = tensors.remove(&declared.name).ok_or_else(|| {
                V3RuntimeError::Tensor(format!("missing candidate scorer input {}", declared.name))
            })?;
            if tensor.rank() != declared.rank || tensor.dtype() != declared.dtype {
                return tensor_error(format!(
                    "candidate scorer input {} type/rank drifted",
                    declared.name
                ));
            }
            inputs.push((declared.name.clone(), tensor.into_ort()?));
        }
        let mut session = split.candidate_scorer.lock().map_err(|_| {
            V3RuntimeError::Inference("candidate scorer session lock poisoned".to_owned())
        })?;
        let outputs = session
            .run(inputs)
            .map_err(|error| V3RuntimeError::Inference(error.to_string()))?;
        self.session_runs.fetch_add(1, Ordering::Relaxed);
        let (_, logits) = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .map_err(|error| V3RuntimeError::Inference(error.to_string()))?;
        if logits.len() != legal.len() || logits.iter().any(|value| !value.is_finite()) {
            return tensor_error("candidate scorer returned invalid policy logits");
        }
        Ok((logits.to_vec(), encoding.value))
    }

    fn evaluate_batch_with_legal_sets(
        &self,
        states: &[GameState],
        legal_sets: &[LegalActionSetV3],
    ) -> Result<Vec<(Vec<f32>, f32)>, V3RuntimeError> {
        if states.is_empty() || states.len() != legal_sets.len() {
            return tensor_error("states and legal action sets must be non-empty and aligned");
        }
        let mut observations = Vec::with_capacity(states.len());
        for (state, legal) in states.iter().zip(legal_sets) {
            if state.ruleset_id != self.metadata.ruleset_id {
                return metadata(format!(
                    "state ruleset {} differs from model ruleset {}",
                    state.ruleset_id, self.metadata.ruleset_id
                ));
            }
            if legal.is_empty() {
                return tensor_error(format!(
                    "states with no legal actions must bypass the model: ply={} phase={:?} status={:?} fingerprint={:016x}",
                    state.ply,
                    state.phase,
                    state.status,
                    state.fingerprint(),
                ));
            }
            observations.push(encode_observation_v3(state));
        }
        let mut tensors = tensorize_v3(&observations, legal_sets, &self.vocabulary)?;
        let mut inputs: Vec<(String, SessionInputValue<'static>)> = Vec::new();
        for declared in &self.metadata.inputs {
            let tensor = tensors.remove(&declared.name).ok_or_else(|| {
                V3RuntimeError::Tensor(format!("missing declared input {}", declared.name))
            })?;
            if tensor.rank() != declared.rank || tensor.dtype() != declared.dtype {
                return tensor_error(format!("input {} type/rank drifted", declared.name));
            }
            inputs.push((declared.name.clone(), tensor.into_ort()?));
        }
        let mut session = self
            .session
            .lock()
            .map_err(|_| V3RuntimeError::Inference("ONNX session lock poisoned".to_owned()))?;
        let outputs = session
            .run(inputs)
            .map_err(|error| V3RuntimeError::Inference(error.to_string()))?;
        self.session_runs.fetch_add(1, Ordering::Relaxed);
        let (_, logits) = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .map_err(|error| V3RuntimeError::Inference(error.to_string()))?;
        let (_, values) = outputs["value"]
            .try_extract_tensor::<f32>()
            .map_err(|error| V3RuntimeError::Inference(error.to_string()))?;
        let max_actions = legal_sets.iter().map(LegalActionSetV3::len).max().unwrap();
        if logits.len() != states.len() * max_actions || values.len() != states.len() {
            return tensor_error("ONNX output shapes do not match the requested batch");
        }
        let mut evaluations = Vec::with_capacity(states.len());
        for (batch_index, (legal, value)) in
            legal_sets.iter().zip(values.iter().copied()).enumerate()
        {
            if !value.is_finite() || !(-1.0001..=1.0001).contains(&value) {
                return tensor_error(format!("invalid value output {value}"));
            }
            let start = batch_index * max_actions;
            let candidate_logits = logits[start..start + legal.len()].to_vec();
            if candidate_logits.iter().any(|logit| !logit.is_finite()) {
                return tensor_error("policy logits contain NaN or infinity");
            }
            evaluations.push((candidate_logits, value));
        }
        Ok(evaluations)
    }
}

pub(crate) fn validate_required_runtime_vocabulary(
    vocabulary: &VocabularyV3,
) -> Result<(), V3RuntimeError> {
    let required = observation_v3_runtime_text_tokens();
    let missing = required
        .iter()
        .filter(|token| !vocabulary.lookup.contains_key(**token))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return metadata(format!(
            "V3 vocabulary is missing required runtime tokens: {}",
            missing.join(", ")
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) enum TensorDataV3 {
    F32 {
        shape: Vec<usize>,
        values: Vec<f32>,
    },
    I64 {
        shape: Vec<usize>,
        values: Vec<i64>,
    },
    Bool {
        shape: Vec<usize>,
        values: Vec<bool>,
    },
}

impl TensorDataV3 {
    pub(crate) fn rank(&self) -> usize {
        match self {
            Self::F32 { shape, .. } | Self::I64 { shape, .. } | Self::Bool { shape, .. } => {
                shape.len()
            }
        }
    }
    pub(crate) fn dtype(&self) -> String {
        match self {
            Self::F32 { .. } => "float32",
            Self::I64 { .. } => "int64",
            Self::Bool { .. } => "bool",
        }
        .to_owned()
    }
    pub(crate) fn into_ort(self) -> Result<SessionInputValue<'static>, V3RuntimeError> {
        match self {
            Self::F32 { shape, values } => {
                Tensor::from_array((shape, values)).map(SessionInputValue::from)
            }
            Self::I64 { shape, values } => {
                Tensor::from_array((shape, values)).map(SessionInputValue::from)
            }
            Self::Bool { shape, values } => {
                Tensor::from_array((shape, values)).map(SessionInputValue::from)
            }
        }
        .map_err(|error| V3RuntimeError::Tensor(error.to_string()))
    }
}

#[derive(Clone, Debug)]
struct RuntimeNodeV3 {
    field: i64,
    kind: i64,
    text: i64,
    player: i64,
    piece_kind: i64,
    square: i64,
    piece_ref: i64,
    card_ref: i64,
    parent: i64,
    numeric: [f32; 7],
}

pub(crate) fn tensorize_v3(
    observations: &[ObservationV3],
    legal_sets: &[LegalActionSetV3],
    vocabulary: &VocabularyV3,
) -> Result<HashMap<String, TensorDataV3>, V3RuntimeError> {
    let batch = observations.len();
    let pieces = observations
        .iter()
        .map(|item| item.pieces.len())
        .max()
        .unwrap_or(0)
        .max(1);
    let cards = observations
        .iter()
        .map(|item| item.cards.len())
        .max()
        .unwrap_or(0)
        .max(1);
    let actions = legal_sets
        .iter()
        .map(LegalActionSetV3::len)
        .max()
        .unwrap_or(0)
        .max(1);
    let runtime_nodes: Vec<_> = observations
        .iter()
        .map(|observation| flatten_runtime(observation, vocabulary))
        .collect::<Result<_, _>>()?;
    let runtime = runtime_nodes.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let state_tokens: Vec<_> = observations
        .iter()
        .map(|observation| {
            std::iter::once(observation.state.mode_id.as_str())
                .chain(observation.state.applied_rules.iter().map(String::as_str))
                .enumerate()
                .map(|(index, token)| {
                    vocabulary.index(Some(token), &format!("state.text[{index}]"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<_, _>>()?;
    let state_text = state_tokens.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let phase_tokens: Vec<_> = observations
        .iter()
        .map(|observation| phase_tokens(&observation.phase, vocabulary))
        .collect::<Result<_, _>>()?;
    let phase_text = phase_tokens.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let target_sequence = legal_sets
        .iter()
        .flat_map(|set| set.actions.iter())
        .map(action_target_squares)
        .map(|squares| squares.len())
        .max()
        .unwrap_or(0)
        .max(1);

    let mut store = TensorStoreV3::new();
    store.f32(
        "spatial",
        &[batch, OBSERVATION_V3_SPATIAL_PLANES, 8, 8],
        0.0,
    );
    store.f32("global_features", &[batch, 32], 0.0);
    store.i64("state_text", &[batch, state_text], 0);
    store.f32("state_text_position", &[batch, state_text, 1], 0.0);
    store.bool("state_text_mask", &[batch, state_text], false);
    store.f32("entity_counts", &[batch, 2], 0.0);
    for name in ["piece_kind", "piece_square", "piece_owner"] {
        store.i64(name, &[batch, pieces], 0);
    }
    store.i64("piece_slot", &[batch, pieces], -1);
    store.f32("piece_numeric", &[batch, pieces, 2], 0.0);
    store.bool("piece_mask", &[batch, pieces], false);
    for name in ["card_definition", "card_owner"] {
        store.i64(name, &[batch, cards], 0);
    }
    store.i64("card_slot", &[batch, cards], -1);
    store.f32("card_numeric", &[batch, cards, 5], 0.0);
    store.bool("card_mask", &[batch, cards], false);
    for name in ["phase_kind", "phase_owner"] {
        store.i64(name, &[batch], 0);
    }
    for name in ["phase_piece_ref", "phase_card_ref"] {
        store.i64(name, &[batch], -1);
    }
    store.f32("phase_numeric", &[batch, 2], 0.0);
    store.i64("phase_text", &[batch, phase_text], 0);
    store.f32("phase_text_position", &[batch, phase_text, 1], 0.0);
    store.bool("phase_text_mask", &[batch, phase_text], false);
    for name in [
        "runtime_field",
        "runtime_type",
        "runtime_text",
        "runtime_player",
        "runtime_piece_kind",
        "runtime_square",
    ] {
        store.i64(name, &[batch, runtime], 0);
    }
    for name in ["runtime_piece_ref", "runtime_card_ref", "runtime_parent"] {
        store.i64(name, &[batch, runtime], -1);
    }
    store.f32("runtime_numeric", &[batch, runtime, 7], 0.0);
    store.bool("runtime_mask", &[batch, runtime], false);
    for name in [
        "action_kind",
        "action_piece_kind",
        "action_from",
        "action_to",
        "action_promotion",
        "action_ability",
        "action_definition",
        "action_handler",
        "action_option",
        "action_target_kind",
    ] {
        store.i64(name, &[batch, actions], 0);
    }
    for name in ["action_piece_ref", "action_card_ref"] {
        store.i64(name, &[batch, actions], -1);
    }
    store.f32("action_target_squares", &[batch, actions, 64], 0.0);
    store.i64(
        "action_target_sequence",
        &[batch, actions, target_sequence],
        0,
    );
    store.f32(
        "action_target_sequence_position",
        &[batch, actions, target_sequence, 1],
        0.0,
    );
    store.bool(
        "action_target_sequence_mask",
        &[batch, actions, target_sequence],
        false,
    );
    store.f32("action_numeric", &[batch, actions, 6], 0.0);
    store.bool("action_mask", &[batch, actions], false);

    for (row, observation) in observations.iter().enumerate() {
        if observation.schema_id != OBSERVATION_V3_SCHEMA_ID
            || observation.spatial.len() != OBSERVATION_V3_SPATIAL_PLANES * 8 * 8
            || observation.global.len() != 32
        {
            return tensor_error("ObservationV3 dense shape/schema mismatch");
        }
        store.f32_slice(
            "spatial",
            row * OBSERVATION_V3_SPATIAL_PLANES * 8 * 8,
            &observation.spatial,
        )?;
        store.f32_slice("global_features", row * 32, &observation.global)?;
        for (index, token) in state_tokens[row].iter().copied().enumerate() {
            store.set_i64("state_text", &[row, index], token)?;
            store.set_f32(
                "state_text_position",
                &[row, index, 0],
                signed_log(index as f32),
            )?;
            store.set_bool("state_text_mask", &[row, index], true)?;
        }
        store.set_f32(
            "entity_counts",
            &[row, 0],
            signed_log(observation.piece_slot_count as f32),
        )?;
        store.set_f32(
            "entity_counts",
            &[row, 1],
            signed_log(observation.card_slot_count as f32),
        )?;
        for (index, piece) in observation.pieces.iter().enumerate() {
            store.set_i64("piece_kind", &[row, index], piece.kind_code.into())?;
            store.set_i64("piece_square", &[row, index], i64::from(piece.square) + 1)?;
            store.set_i64("piece_owner", &[row, index], player_id(piece.owner))?;
            store.set_i64("piece_slot", &[row, index], piece.slot.into())?;
            store.set_f32(
                "piece_numeric",
                &[row, index, 0],
                signed_log(piece.capture_count as f32),
            )?;
            store.set_f32(
                "piece_numeric",
                &[row, index, 1],
                signed_log(piece.slot as f32),
            )?;
            store.set_bool("piece_mask", &[row, index], true)?;
        }
        for (index, card) in observation.cards.iter().enumerate() {
            store.set_i64(
                "card_definition",
                &[row, index],
                vocabulary.index(Some(&card.definition_id), "card.definition_id")?,
            )?;
            store.set_i64("card_owner", &[row, index], player_id(card.owner))?;
            store.set_i64("card_slot", &[row, index], card.slot.into())?;
            let available = card.available_in_owner_turns;
            let numeric = [
                signed_log(card.half_stars as f32),
                signed_log(card.charges as f32),
                if available.is_some() { 1.0 } else { 0.0 },
                signed_log(available.unwrap_or(0) as f32),
                signed_log(card.slot as f32),
            ];
            for (column, value) in numeric.into_iter().enumerate() {
                store.set_f32("card_numeric", &[row, index, column], value)?;
            }
            store.set_bool("card_mask", &[row, index], true)?;
        }
        write_phase(&mut store, row, &observation.phase, &phase_tokens[row])?;
        for (index, node) in runtime_nodes[row].iter().enumerate() {
            for (name, value) in [
                ("runtime_field", node.field),
                ("runtime_type", node.kind),
                ("runtime_text", node.text),
                ("runtime_player", node.player),
                ("runtime_piece_kind", node.piece_kind),
                ("runtime_square", node.square),
                ("runtime_piece_ref", node.piece_ref),
                ("runtime_card_ref", node.card_ref),
                ("runtime_parent", node.parent),
            ] {
                store.set_i64(name, &[row, index], value)?;
            }
            for (column, value) in node.numeric.into_iter().enumerate() {
                store.set_f32("runtime_numeric", &[row, index, column], value)?;
            }
            store.set_bool("runtime_mask", &[row, index], true)?;
        }
        if legal_sets[row].schema_id != ACTION_V3_SCHEMA_ID {
            return tensor_error("ActionV3 schema mismatch");
        }
        for (index, action) in legal_sets[row].actions.iter().enumerate() {
            write_action(&mut store, row, index, action, vocabulary)?;
            store.set_bool("action_mask", &[row, index], true)?;
        }
    }
    Ok(store.values)
}

struct TensorStoreV3 {
    values: HashMap<String, TensorDataV3>,
}

impl TensorStoreV3 {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }
    fn f32(&mut self, name: &str, shape: &[usize], fill: f32) {
        self.values.insert(
            name.to_owned(),
            TensorDataV3::F32 {
                shape: shape.to_vec(),
                values: vec![fill; shape.iter().product()],
            },
        );
    }
    fn i64(&mut self, name: &str, shape: &[usize], fill: i64) {
        self.values.insert(
            name.to_owned(),
            TensorDataV3::I64 {
                shape: shape.to_vec(),
                values: vec![fill; shape.iter().product()],
            },
        );
    }
    fn bool(&mut self, name: &str, shape: &[usize], fill: bool) {
        self.values.insert(
            name.to_owned(),
            TensorDataV3::Bool {
                shape: shape.to_vec(),
                values: vec![fill; shape.iter().product()],
            },
        );
    }
    fn set_f32(&mut self, name: &str, indices: &[usize], value: f32) -> Result<(), V3RuntimeError> {
        match self.values.get_mut(name) {
            Some(TensorDataV3::F32 { shape, values }) => {
                values[flat_index(shape, indices)?] = value;
                Ok(())
            }
            _ => tensor_error(format!("missing f32 tensor {name}")),
        }
    }
    fn set_i64(&mut self, name: &str, indices: &[usize], value: i64) -> Result<(), V3RuntimeError> {
        match self.values.get_mut(name) {
            Some(TensorDataV3::I64 { shape, values }) => {
                values[flat_index(shape, indices)?] = value;
                Ok(())
            }
            _ => tensor_error(format!("missing i64 tensor {name}")),
        }
    }
    fn set_bool(
        &mut self,
        name: &str,
        indices: &[usize],
        value: bool,
    ) -> Result<(), V3RuntimeError> {
        match self.values.get_mut(name) {
            Some(TensorDataV3::Bool { shape, values }) => {
                values[flat_index(shape, indices)?] = value;
                Ok(())
            }
            _ => tensor_error(format!("missing bool tensor {name}")),
        }
    }
    fn f32_slice(
        &mut self,
        name: &str,
        offset: usize,
        slice: &[f32],
    ) -> Result<(), V3RuntimeError> {
        match self.values.get_mut(name) {
            Some(TensorDataV3::F32 { values, .. }) if offset + slice.len() <= values.len() => {
                values[offset..offset + slice.len()].copy_from_slice(slice);
                Ok(())
            }
            _ => tensor_error(format!("cannot write f32 tensor {name}")),
        }
    }
}

fn flat_index(shape: &[usize], indices: &[usize]) -> Result<usize, V3RuntimeError> {
    if shape.len() != indices.len() || indices.iter().zip(shape).any(|(index, size)| index >= size)
    {
        return tensor_error("tensor index is outside shape");
    }
    Ok(indices
        .iter()
        .enumerate()
        .map(|(axis, index)| index * shape[axis + 1..].iter().product::<usize>())
        .sum())
}

fn phase_tokens(
    phase: &crate::PhaseV3,
    vocabulary: &VocabularyV3,
) -> Result<Vec<i64>, V3RuntimeError> {
    let values: Vec<&str> = match phase {
        crate::PhaseV3::Playing => Vec::new(),
        crate::PhaseV3::ForcedMove {
            source_kind,
            source_id,
            continuation,
            ..
        } => {
            let mut values = vec![source_kind.as_str()];
            values.extend(source_id.as_deref());
            values.push(continuation);
            values
        }
        crate::PhaseV3::Draft {
            draft_phase,
            offers,
            ..
        } => std::iter::once(draft_phase.as_str())
            .chain(offers.iter().map(String::as_str))
            .collect(),
        crate::PhaseV3::AwaitingChoice {
            handler_id,
            options,
            continuation,
            ..
        } => std::iter::once(continuation.as_str())
            .chain(std::iter::once(handler_id.as_str()))
            .chain(options.iter().map(String::as_str))
            .collect(),
    };
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| vocabulary.index(Some(value), &format!("phase.{index}")))
        .collect()
}

fn write_phase(
    store: &mut TensorStoreV3,
    row: usize,
    phase: &crate::PhaseV3,
    tokens: &[i64],
) -> Result<(), V3RuntimeError> {
    let (kind, owner, piece, card, choice) = match phase {
        crate::PhaseV3::Playing => (1, 0, -1, -1, None),
        crate::PhaseV3::ForcedMove { piece, .. } => (2, 0, i64::from(*piece), -1, None),
        crate::PhaseV3::Draft { resume_turn, .. } => (3, player_id(*resume_turn), -1, -1, None),
        crate::PhaseV3::AwaitingChoice {
            choice_slot,
            owner,
            source_card,
            ..
        } => (
            4,
            player_id(*owner),
            -1,
            i64::from(*source_card),
            Some(*choice_slot),
        ),
    };
    store.set_i64("phase_kind", &[row], kind)?;
    store.set_i64("phase_owner", &[row], owner)?;
    store.set_i64("phase_piece_ref", &[row], piece)?;
    store.set_i64("phase_card_ref", &[row], card)?;
    store.set_f32(
        "phase_numeric",
        &[row, 0],
        signed_log(choice.unwrap_or(0) as f32),
    )?;
    store.set_f32(
        "phase_numeric",
        &[row, 1],
        if choice.is_some() { 1.0 } else { 0.0 },
    )?;
    for (index, token) in tokens.iter().copied().enumerate() {
        store.set_i64("phase_text", &[row, index], token)?;
        store.set_f32(
            "phase_text_position",
            &[row, index, 0],
            signed_log(index as f32),
        )?;
        store.set_bool("phase_text_mask", &[row, index], true)?;
    }
    Ok(())
}

fn flatten_runtime(
    observation: &ObservationV3,
    vocabulary: &VocabularyV3,
) -> Result<Vec<RuntimeNodeV3>, V3RuntimeError> {
    if observation.runtime.len() != CARD_RUNTIME_SOURCE_FIELDS.len() {
        return tensor_error(format!(
            "runtime has {} fields; expected {}",
            observation.runtime.len(),
            CARD_RUNTIME_SOURCE_FIELDS.len()
        ));
    }
    let mut nodes = Vec::new();
    for feature in &observation.runtime {
        append_runtime_node(
            &mut nodes,
            i64::from(feature.field_id),
            &feature.value,
            vocabulary,
            -1,
            0,
            usize::from(feature.field_id - 1),
            None,
        )?;
    }
    Ok(nodes)
}

#[allow(clippy::too_many_arguments)]
fn append_runtime_node(
    nodes: &mut Vec<RuntimeNodeV3>,
    field: i64,
    value: &RuntimeValueV3,
    vocabulary: &VocabularyV3,
    parent: i64,
    depth: usize,
    sibling: usize,
    name: Option<&str>,
) -> Result<(), V3RuntimeError> {
    let kind = match value {
        RuntimeValueV3::None => 1,
        RuntimeValueV3::Bool(_) => 2,
        RuntimeValueV3::Unsigned(_) => 3,
        RuntimeValueV3::Signed(_) => 4,
        RuntimeValueV3::Player(_) => 5,
        RuntimeValueV3::Piece(_) => 6,
        RuntimeValueV3::Card(_) => 7,
        RuntimeValueV3::PieceKind(_) => 8,
        RuntimeValueV3::Square(_) => 9,
        RuntimeValueV3::Text(_) => 10,
        RuntimeValueV3::List(_) => 11,
        RuntimeValueV3::Record(_) => 12,
    };
    let index = nodes.len() as i64;
    let mut node = RuntimeNodeV3 {
        field,
        kind,
        text: vocabulary.index(name, "runtime.record_name")?,
        player: 0,
        piece_kind: 0,
        square: 0,
        piece_ref: -1,
        card_ref: -1,
        parent,
        numeric: [
            0.0,
            0.0,
            signed_log(depth as f32),
            signed_log(sibling as f32),
            0.0,
            0.0,
            0.0,
        ],
    };
    match value {
        RuntimeValueV3::Bool(value) => {
            node.numeric[0] = if *value { 1.0 } else { 0.0 };
            node.numeric[1] = 1.0;
        }
        RuntimeValueV3::Unsigned(value) => {
            node.numeric[0] = signed_log(*value as f32);
            node.numeric[1] = 1.0;
        }
        RuntimeValueV3::Signed(value) => {
            node.numeric[0] = signed_log(*value as f32);
            node.numeric[1] = 1.0;
        }
        RuntimeValueV3::Player(value) => node.player = player_id(*value),
        RuntimeValueV3::Piece(value) => {
            node.piece_ref = i64::from(*value);
            node.numeric[4] = signed_log(*value as f32);
        }
        RuntimeValueV3::Card(value) => {
            node.card_ref = i64::from(*value);
            node.numeric[5] = signed_log(*value as f32);
        }
        RuntimeValueV3::PieceKind(value) => node.piece_kind = i64::from(*value),
        RuntimeValueV3::Square(value) => node.square = i64::from(*value) + 1,
        RuntimeValueV3::Text(value) => node.text = vocabulary.index(Some(value), "runtime.text")?,
        _ => {}
    }
    nodes.push(node);
    match value {
        RuntimeValueV3::List(children) => {
            for (child_index, child) in children.iter().enumerate() {
                append_runtime_node(
                    nodes,
                    field,
                    child,
                    vocabulary,
                    index,
                    depth + 1,
                    child_index,
                    None,
                )?;
            }
        }
        RuntimeValueV3::Record(children) => {
            for (child_index, child) in children.iter().enumerate() {
                append_runtime_node(
                    nodes,
                    field,
                    &child.value,
                    vocabulary,
                    index,
                    depth + 1,
                    child_index,
                    Some(&child.name),
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn write_action(
    store: &mut TensorStoreV3,
    row: usize,
    index: usize,
    action: &ActionDescriptorV3,
    vocabulary: &VocabularyV3,
) -> Result<(), V3RuntimeError> {
    let mut piece_ref = None;
    let mut card_ref = None;
    let mut offer_index = 0u16;
    let mut choice_slot = 0u32;
    let mut option_index = 0u16;
    match action {
        ActionDescriptorV3::Move {
            piece,
            piece_kind,
            from,
            to,
            promotion,
        } => {
            store.set_i64("action_kind", &[row, index], 1)?;
            piece_ref = Some(*piece);
            store.set_i64("action_piece_kind", &[row, index], i64::from(*piece_kind))?;
            store.set_i64("action_from", &[row, index], i64::from(*from) + 1)?;
            store.set_i64("action_to", &[row, index], i64::from(*to) + 1)?;
            store.set_i64(
                "action_promotion",
                &[row, index],
                i64::from(promotion.unwrap_or(0)),
            )?;
        }
        ActionDescriptorV3::PieceAbility {
            piece,
            piece_kind,
            from,
            ability,
            ..
        } => {
            store.set_i64("action_kind", &[row, index], 2)?;
            piece_ref = Some(*piece);
            store.set_i64("action_piece_kind", &[row, index], i64::from(*piece_kind))?;
            store.set_i64("action_from", &[row, index], i64::from(*from) + 1)?;
            let ability = match ability {
                crate::PieceAbilityV3::WizardLightning => 1,
                crate::PieceAbilityV3::WizardShield => 2,
                crate::PieceAbilityV3::WizardMeteor => 3,
                crate::PieceAbilityV3::WizardTimeStop => 4,
                crate::PieceAbilityV3::ColossusSector => 5,
                crate::PieceAbilityV3::MerchantBuy => 6,
            };
            store.set_i64("action_ability", &[row, index], ability)?;
        }
        ActionDescriptorV3::Card {
            card,
            definition_id,
            ..
        } => {
            store.set_i64("action_kind", &[row, index], 3)?;
            card_ref = Some(*card);
            store.set_i64(
                "action_definition",
                &[row, index],
                vocabulary.index(Some(definition_id), "action.definition_id")?,
            )?;
        }
        ActionDescriptorV3::DraftPick {
            offer_index: offer,
            definition_id,
        } => {
            store.set_i64("action_kind", &[row, index], 4)?;
            offer_index = *offer;
            store.set_i64(
                "action_definition",
                &[row, index],
                vocabulary.index(Some(definition_id), "action.definition_id")?,
            )?;
        }
        ActionDescriptorV3::ResolveChoice {
            choice_slot: choice,
            handler_id,
            option_index: option_idx,
            option,
        } => {
            store.set_i64("action_kind", &[row, index], 5)?;
            choice_slot = *choice;
            option_index = *option_idx;
            store.set_i64(
                "action_handler",
                &[row, index],
                vocabulary.index(Some(handler_id), "action.handler_id")?,
            )?;
            store.set_i64(
                "action_option",
                &[row, index],
                vocabulary.index(Some(option), "action.option")?,
            )?;
        }
    }
    if let Some(piece) = piece_ref {
        store.set_i64("action_piece_ref", &[row, index], i64::from(piece))?;
    }
    if let Some(card) = card_ref {
        store.set_i64("action_card_ref", &[row, index], i64::from(card))?;
    }
    let squares = action_target_squares(action);
    let target_kind = match action_target(action) {
        None | Some(crate::ActionTargetV3::None) => 1,
        Some(crate::ActionTargetV3::Square { .. }) => 2,
        Some(crate::ActionTargetV3::PieceAt { .. }) => 3,
        Some(crate::ActionTargetV3::Squares { .. }) => 4,
    };
    store.set_i64("action_target_kind", &[row, index], target_kind)?;
    for (sequence, square) in squares.iter().copied().enumerate() {
        store.set_f32(
            "action_target_squares",
            &[row, index, usize::from(square)],
            1.0,
        )?;
        store.set_i64(
            "action_target_sequence",
            &[row, index, sequence],
            i64::from(square) + 1,
        )?;
        store.set_f32(
            "action_target_sequence_position",
            &[row, index, sequence, 0],
            signed_log(sequence as f32),
        )?;
        store.set_bool("action_target_sequence_mask", &[row, index, sequence], true)?;
    }
    let numeric = [
        signed_log(offer_index as f32),
        signed_log(choice_slot as f32),
        signed_log(option_index as f32),
        signed_log(squares.len() as f32),
        signed_log(piece_ref.unwrap_or(0) as f32),
        signed_log(card_ref.unwrap_or(0) as f32),
    ];
    for (column, value) in numeric.into_iter().enumerate() {
        store.set_f32("action_numeric", &[row, index, column], value)?;
    }
    Ok(())
}

fn action_target(action: &ActionDescriptorV3) -> Option<&crate::ActionTargetV3> {
    match action {
        ActionDescriptorV3::PieceAbility { target, .. }
        | ActionDescriptorV3::Card { target, .. } => Some(target),
        _ => None,
    }
}

fn action_target_squares(action: &ActionDescriptorV3) -> Vec<u8> {
    match action_target(action) {
        Some(crate::ActionTargetV3::Square { square })
        | Some(crate::ActionTargetV3::PieceAt { square }) => vec![*square],
        Some(crate::ActionTargetV3::Squares { squares }) => squares.clone(),
        _ => Vec::new(),
    }
}

fn player_id(player: crate::RelativePlayerV3) -> i64 {
    match player {
        crate::RelativePlayerV3::SelfPlayer => 1,
        crate::RelativePlayerV3::Opponent => 2,
    }
}

fn signed_log(value: f32) -> f32 {
    value.signum() * value.abs().ln_1p()
}

pub(crate) fn build_session(
    path: &Path,
    options: V3RuntimeOptions,
) -> Result<Session, V3RuntimeError> {
    let builder = Session::builder()
        .map_err(|error| V3RuntimeError::Inference(error.to_string()))?
        .with_intra_threads(options.intra_threads)
        .map_err(|error| V3RuntimeError::Inference(error.to_string()))?;
    let mut builder = match options.execution_provider {
        V3ExecutionProvider::Cpu => builder,
        V3ExecutionProvider::Cuda => builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
            .map_err(|error| V3RuntimeError::Inference(error.to_string()))?,
    };
    builder
        .commit_from_file(path)
        .map_err(|error| V3RuntimeError::Inference(error.to_string()))
}

fn validate_split_policy_bundle(
    bundle: &SplitPolicyBundleV3,
    model: &ModelMetadataV3,
    base: &Path,
) -> Result<(std::path::PathBuf, std::path::PathBuf), V3RuntimeError> {
    if bundle.schema_id != "augmentia.onnx-split-policy-bundle.v1" {
        return metadata("unsupported split policy bundle schema");
    }
    let state_inputs = model
        .inputs
        .iter()
        .filter(|input| !input.name.starts_with("action_"))
        .cloned()
        .collect::<Vec<_>>();
    if bundle.state_encoder.inputs != state_inputs {
        return metadata("state encoder input contract mismatch");
    }
    let state_outputs = vec![
        model_input("state_embedding", "float32", 2),
        model_input("board_tokens", "float32", 3),
        model_input("piece_entities", "float32", 3),
        model_input("card_entities", "float32", 3),
        model_input("value", "float32", 1),
    ];
    if bundle.state_encoder.outputs != state_outputs {
        return metadata("state encoder output contract mismatch");
    }
    let mut scorer_inputs = state_outputs[..4].to_vec();
    for name in ["piece_slot", "piece_mask", "card_slot", "card_mask"] {
        let input = model
            .inputs
            .iter()
            .find(|input| input.name == name)
            .ok_or_else(|| {
                V3RuntimeError::Metadata(format!(
                    "monolithic model omits split scorer lookup input {name}"
                ))
            })?;
        scorer_inputs.push(input.clone());
    }
    scorer_inputs.extend(
        model
            .inputs
            .iter()
            .filter(|input| input.name.starts_with("action_"))
            .cloned(),
    );
    if bundle.candidate_scorer.inputs != scorer_inputs
        || bundle.candidate_scorer.outputs != [model_input("policy_logits", "float32", 2)]
    {
        return metadata("candidate scorer I/O contract mismatch");
    }
    let encoder_path = validate_split_graph_file(&bundle.state_encoder, base, "state encoder")?;
    let scorer_path =
        validate_split_graph_file(&bundle.candidate_scorer, base, "candidate scorer")?;
    if encoder_path == scorer_path {
        return metadata("split policy graphs must use distinct sibling paths");
    }
    Ok((encoder_path, scorer_path))
}

pub(crate) fn model_input(name: &str, dtype: &str, rank: usize) -> ModelInputV3 {
    ModelInputV3 {
        name: name.to_owned(),
        dtype: dtype.to_owned(),
        rank,
    }
}

fn validate_split_graph_file(
    contract: &SplitGraphContractV3,
    base: &Path,
    label: &str,
) -> Result<std::path::PathBuf, V3RuntimeError> {
    let relative = Path::new(&contract.path);
    if contract.path.is_empty()
        || relative.is_absolute()
        || relative.components().count() != 1
        || relative.file_name().is_none()
    {
        return metadata(format!("{label} path must name a sibling file"));
    }
    let path = base.join(relative);
    let bytes = fs::read(&path).map_err(|error| V3RuntimeError::Io(error.to_string()))?;
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != contract.sha256 {
        return metadata(format!("{label} ONNX checksum does not match metadata"));
    }
    Ok(path)
}

pub(crate) fn validate_session_contract(
    session: &Session,
    inputs: &[ModelInputV3],
    outputs: &[ModelInputV3],
    label: &str,
) -> Result<(), V3RuntimeError> {
    let actual_inputs = session
        .inputs()
        .iter()
        .map(session_outlet_contract)
        .collect::<Result<Vec<_>, _>>()?;
    let actual_outputs = session
        .outputs()
        .iter()
        .map(session_outlet_contract)
        .collect::<Result<Vec<_>, _>>()?;
    if actual_inputs != inputs || actual_outputs != outputs {
        return metadata(format!(
            "{label} ONNX graph does not match its I/O contract"
        ));
    }
    Ok(())
}

fn session_outlet_contract(outlet: &ort::value::Outlet) -> Result<ModelInputV3, V3RuntimeError> {
    let ort::value::ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
        return metadata(format!("ONNX outlet {} is not a tensor", outlet.name()));
    };
    let dtype = match ty {
        ort::value::TensorElementType::Float32 => "float32",
        ort::value::TensorElementType::Int64 => "int64",
        ort::value::TensorElementType::Bool => "bool",
        _ => {
            return metadata(format!(
                "ONNX outlet {} has unsupported dtype",
                outlet.name()
            ));
        }
    };
    Ok(model_input(outlet.name(), dtype, shape.len()))
}

fn validate_metadata(
    metadata_value: &ModelMetadataV3,
    model_path: &Path,
    expected_ruleset_id: &str,
    expected_ruleset_hash: &str,
) -> Result<(), V3RuntimeError> {
    if metadata_value.metadata_schema_id != MODEL_METADATA_V3_SCHEMA_ID {
        return metadata("unsupported model metadata schema");
    }
    if metadata_value.observation_schema_id != OBSERVATION_V3_SCHEMA_ID
        || metadata_value.action_schema_id != ACTION_V3_SCHEMA_ID
    {
        return metadata("model ObservationV3/ActionV3 schema mismatch");
    }
    if metadata_value.ruleset_id != expected_ruleset_id
        || metadata_value.ruleset_manifest_sha256 != expected_ruleset_hash
    {
        return metadata("model ruleset artifact does not match the runtime");
    }
    let required_piece_kind_rows = PIECE_KIND_VOCABULARY
        .iter()
        .map(|(code, _)| usize::from(*code))
        .max()
        .unwrap_or(0)
        + 1;
    if metadata_value.piece_kind_vocabulary_size != required_piece_kind_rows {
        return metadata(format!(
            "model piece-kind vocabulary has {} rows, expected {required_piece_kind_rows}",
            metadata_value.piece_kind_vocabulary_size
        ));
    }
    let bytes = fs::read(model_path).map_err(|error| V3RuntimeError::Io(error.to_string()))?;
    let actual_hash = format!("{:x}", Sha256::digest(bytes));
    if actual_hash != metadata_value.onnx_sha256 {
        return metadata("ONNX checksum does not match metadata");
    }
    if metadata_value.outputs != ["policy_logits", "value"] {
        return metadata("model output contract mismatch");
    }
    if metadata_value.inputs.is_empty() {
        return metadata("model declares no inputs");
    }
    let mut names = std::collections::HashSet::new();
    for input in &metadata_value.inputs {
        if !names.insert(input.name.as_str())
            || !matches!(input.dtype.as_str(), "float32" | "int64" | "bool")
            || input.rank == 0
        {
            return metadata(format!("invalid model input declaration {}", input.name));
        }
    }
    Ok(())
}

fn softmax(logits: &[f32]) -> Result<Vec<f32>, V3RuntimeError> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return tensor_error("policy logits are empty, NaN, or infinite");
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<_> = logits
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect();
    let total: f32 = values.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return tensor_error("policy softmax has invalid mass");
    }
    for value in &mut values {
        *value /= total;
    }
    Ok(values)
}

fn metadata<T>(message: impl Into<String>) -> Result<T, V3RuntimeError> {
    Err(V3RuntimeError::Metadata(message.into()))
}

fn tensor_error<T>(message: impl Into<String>) -> Result<T, V3RuntimeError> {
    Err(V3RuntimeError::Tensor(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::BTreeSet;

    #[derive(Deserialize)]
    struct Suite {
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        observation: ObservationV3,
        legal_actions: LegalActionSetV3,
    }

    #[test]
    fn incomplete_runtime_vocabulary_is_rejected_before_inference() {
        let artifact = VocabularyArtifactV3 {
            schema_id: "augmentia.model-vocabulary.v3".to_owned(),
            tokens: vec!["<none>".to_owned(), "opening".to_owned()],
        };
        let vocabulary = VocabularyV3::from_artifact(&artifact).unwrap();
        let error = validate_required_runtime_vocabulary(&vocabulary).unwrap_err();
        assert!(error.to_string().contains("middle"));
        assert!(error.to_string().contains("square"));
    }

    #[test]
    fn tensorizer_pads_heterogeneous_rust_cases_without_losing_actions() {
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../train/v3/fixtures/observation_action_v3.json"
        ));
        let json: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let suite: Suite = serde_json::from_slice(bytes).unwrap();
        let mut tokens = BTreeSet::new();
        collect_strings(&json, &mut tokens);
        let artifact = VocabularyArtifactV3 {
            schema_id: "augmentia.model-vocabulary.v3".to_owned(),
            tokens: std::iter::once("<none>".to_owned())
                .chain(tokens.into_iter().filter(|token| token != "<none>"))
                .collect(),
        };
        let vocabulary = VocabularyV3::from_artifact(&artifact).unwrap();
        let cases: Vec<_> = suite
            .cases
            .into_iter()
            .filter(|case| !case.legal_actions.is_empty())
            .take(3)
            .collect();
        let expected_actions: usize = cases.iter().map(|case| case.legal_actions.len()).sum();
        let observations: Vec<_> = cases.iter().map(|case| case.observation.clone()).collect();
        let actions: Vec<_> = cases
            .iter()
            .map(|case| case.legal_actions.clone())
            .collect();
        let tensors = tensorize_v3(&observations, &actions, &vocabulary).unwrap();
        match &tensors["action_mask"] {
            TensorDataV3::Bool { shape, values } => {
                assert_eq!(shape[0], cases.len());
                assert_eq!(
                    values.iter().filter(|value| **value).count(),
                    expected_actions
                );
            }
            _ => panic!("action_mask must be boolean"),
        }
        match &tensors["spatial"] {
            TensorDataV3::F32 { shape, values } => {
                assert_eq!(
                    shape,
                    &[cases.len(), crate::OBSERVATION_V3_SPATIAL_PLANES, 8, 8]
                );
                assert!(values.iter().all(|value| value.is_finite()));
            }
            _ => panic!("spatial must be float32"),
        }
    }

    fn collect_strings(value: &serde_json::Value, tokens: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::String(value) => {
                tokens.insert(value.clone());
            }
            serde_json::Value::Array(values) => {
                for value in values {
                    collect_strings(value, tokens);
                }
            }
            serde_json::Value::Object(values) => {
                for (name, value) in values {
                    tokens.insert(name.clone());
                    collect_strings(value, tokens);
                }
            }
            _ => {}
        }
    }
}

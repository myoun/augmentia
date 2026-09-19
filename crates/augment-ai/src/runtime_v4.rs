//! Phase 1 V4 factorized ONNX runtime.
//!
//! The V4 bundle (see `train/v4`) splits the network the way search consumes it:
//! a `state_encoder` graph that reuses the authoritative V3 state encoding, and a
//! `factorized_scorer` graph that turns cached encoder outputs plus typed trie
//! edges into conditional prefix logits `[rows, children]`.
//!
//! This runtime loads the split bundle metadata fail-closed (schema ids, ruleset
//! identity, vocabulary, graph checksums, and declared I/O contracts), reuses the
//! `runtime_v3` tensorizer/session helpers, and converts conditional prefix
//! logits into complete-action log probabilities by multiplying the conditional
//! distributions along each canonical factorized path. The resulting
//! `FactorizedEvaluationV4.action_logits` are complete-action log probabilities,
//! so `exp(action_logits)` is a normalized distribution and prefix priors are
//! exact subtree sums.

use crate::mcts_v4::{PrefixScorerV4, PreparedPrefixEvaluationV4};
use crate::runtime_v3::{
    ModelInputV3, TensorDataV3, V3RuntimeError, V3RuntimeOptions, VocabularyArtifactV3,
    VocabularyV3, build_session, model_input, tensorize_v3, validate_required_runtime_vocabulary,
    validate_session_contract,
};
use crate::{
    ACTION_V3_SCHEMA_ID, ActionDescriptorV3, FACTORIZED_POLICY_V4_SCHEMA_ID,
    FactorizedEvaluationV4, FactorizedEvaluatorV4, LegalActionSetV3, MctsV4Error,
    OBSERVATION_V3_SCHEMA_ID, ObservationV3, PIECE_KIND_VOCABULARY, encode_observation_v3,
    factorized_action_path_v4,
};
use augment_core::{Engine, GameState};
use ort::session::Session;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub const V4_BUNDLE_METADATA_SCHEMA_ID: &str = "augmentia.onnx-factorized-policy-bundle.v1";
pub const V4_MINIMUM_OPSET: u32 = 18;

const LOOKUP_INPUT_NAMES_V4: &[&str] = &["piece_slot", "piece_mask", "card_slot", "card_mask"];
const ENCODER_OUTPUT_NAMES_V4: &[&str] = &[
    "state_embedding",
    "board_tokens",
    "piece_entities",
    "card_entities",
];

const EXPECTED_STATE_ENCODER_INPUTS: &[(&str, &str, usize)] = &[
    ("spatial", "float32", 4),
    ("global_features", "float32", 2),
    ("state_text", "int64", 2),
    ("state_text_position", "float32", 3),
    ("state_text_mask", "bool", 2),
    ("entity_counts", "float32", 2),
    ("piece_kind", "int64", 2),
    ("piece_square", "int64", 2),
    ("piece_owner", "int64", 2),
    ("piece_slot", "int64", 2),
    ("piece_numeric", "float32", 3),
    ("piece_mask", "bool", 2),
    ("card_definition", "int64", 2),
    ("card_owner", "int64", 2),
    ("card_slot", "int64", 2),
    ("card_numeric", "float32", 3),
    ("card_mask", "bool", 2),
    ("phase_kind", "int64", 1),
    ("phase_owner", "int64", 1),
    ("phase_piece_ref", "int64", 1),
    ("phase_card_ref", "int64", 1),
    ("phase_numeric", "float32", 2),
    ("phase_text", "int64", 2),
    ("phase_text_position", "float32", 3),
    ("phase_text_mask", "bool", 2),
    ("runtime_field", "int64", 2),
    ("runtime_type", "int64", 2),
    ("runtime_text", "int64", 2),
    ("runtime_player", "int64", 2),
    ("runtime_piece_kind", "int64", 2),
    ("runtime_square", "int64", 2),
    ("runtime_piece_ref", "int64", 2),
    ("runtime_card_ref", "int64", 2),
    ("runtime_parent", "int64", 2),
    ("runtime_numeric", "float32", 3),
    ("runtime_mask", "bool", 2),
];

const EXPECTED_STATE_ENCODER_OUTPUTS: &[(&str, &str, usize)] = &[
    ("state_embedding", "float32", 2),
    ("board_tokens", "float32", 3),
    ("piece_entities", "float32", 3),
    ("card_entities", "float32", 3),
    ("value", "float32", 1),
];

const EXPECTED_FACTORIZED_SCORER_INPUTS: &[(&str, &str, usize)] = &[
    ("state_embedding", "float32", 2),
    ("board_tokens", "float32", 3),
    ("piece_entities", "float32", 3),
    ("card_entities", "float32", 3),
    ("piece_slot", "int64", 2),
    ("piece_mask", "bool", 2),
    ("card_slot", "int64", 2),
    ("card_mask", "bool", 2),
    ("prefix_field", "int64", 2),
    ("prefix_category", "int64", 2),
    ("prefix_index", "int64", 2),
    ("prefix_ordinal", "int64", 2),
    ("prefix_mask", "bool", 2),
    ("child_field", "int64", 2),
    ("child_category", "int64", 2),
    ("child_index", "int64", 2),
    ("child_ordinal", "int64", 2),
    ("child_mask", "bool", 2),
    ("legal_mask", "bool", 2),
    ("state_index", "int64", 1),
];

const EXPECTED_FACTORIZED_SCORER_OUTPUTS: &[(&str, &str, usize)] =
    &[("prefix_logits", "float32", 2)];

/// Stable factor field ids shared with `train/v4/batch.py` (`FACTOR_FIELD_IDS`).
const FACTOR_FIELD_IDS_V4: &[(&str, i64)] = &[
    ("family", 1),
    ("piece", 2),
    ("destination", 3),
    ("promotion", 4),
    ("ability", 5),
    ("target_kind", 6),
    ("target_square", 7),
    ("target_sequence", 8),
    ("card", 9),
    ("offer", 10),
    ("choice", 11),
    ("option", 12),
    ("complete", 13),
    ("source_square", 14),
];

#[derive(Debug)]
pub enum V4RuntimeError {
    Io(String),
    Metadata(String),
    Tensor(String),
    Inference(String),
}

impl fmt::Display for V4RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "V4 runtime I/O error: {message}"),
            Self::Metadata(message) => write!(formatter, "V4 model metadata error: {message}"),
            Self::Tensor(message) => write!(formatter, "V4 tensor error: {message}"),
            Self::Inference(message) => write!(formatter, "V4 ONNX inference error: {message}"),
        }
    }
}

impl std::error::Error for V4RuntimeError {}

impl From<V3RuntimeError> for V4RuntimeError {
    fn from(error: V3RuntimeError) -> Self {
        match error {
            V3RuntimeError::Io(message) => Self::Io(message),
            V3RuntimeError::Metadata(message) => Self::Metadata(message),
            V3RuntimeError::Tensor(message) => Self::Tensor(message),
            V3RuntimeError::Inference(message) => Self::Inference(message),
            V3RuntimeError::Engine(message) => Self::Tensor(message),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct GraphContractV4 {
    path: String,
    sha256: String,
    inputs: Vec<ModelInputV3>,
    outputs: Vec<ModelInputV3>,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelConfigV4 {
    text_vocabulary_size: usize,
    piece_kind_vocabulary_size: usize,
    channels: usize,
    residual_blocks: usize,
    model_dim: usize,
    runtime_message_passes: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct DatasetIdentityV4 {
    ruleset_id: String,
    ruleset_manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct BundleMetadataV4 {
    metadata_schema_id: String,
    observation_schema_id: String,
    action_schema_id: String,
    factorized_policy_schema_id: String,
    ruleset_id: String,
    ruleset_manifest_sha256: String,
    dataset: DatasetIdentityV4,
    vocabulary: VocabularyArtifactV3,
    model_config: ModelConfigV4,
    piece_kind_vocabulary_size: usize,
    opset: u32,
    state_encoder: GraphContractV4,
    factorized_scorer: GraphContractV4,
    checkpoint_sha256: Option<String>,
    code_commit: String,
}

/// One typed factorized-policy trie edge in the wire form shared with Python.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FactorEdgeV4 {
    pub(crate) field: i64,
    pub(crate) category: i64,
    pub(crate) index: i64,
    pub(crate) ordinal: i64,
}

/// One conditional row of the factorized decision: a trie prefix and its children.
#[derive(Clone, Debug)]
pub(crate) struct FactorRowV4 {
    pub(crate) prefix: Vec<String>,
    pub(crate) children: Vec<String>,
    pub(crate) prefix_edges: Vec<FactorEdgeV4>,
    pub(crate) child_edges: Vec<FactorEdgeV4>,
}

/// A single conditional step of one complete action's canonical path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FactorStepV4 {
    pub(crate) row: usize,
    pub(crate) column: usize,
}

/// The factorized form of one decision's legal action set.
#[derive(Clone, Debug)]
pub(crate) struct FactorizedDecisionV4 {
    pub(crate) rows: Vec<FactorRowV4>,
    pub(crate) action_steps: Vec<Vec<FactorStepV4>>,
}

#[derive(Clone, Debug)]
struct EncodedStateV4 {
    encoder_outputs: HashMap<String, TensorDataV3>,
    lookup_tensors: HashMap<String, TensorDataV3>,
    value: f32,
}

// Bound retained encoder tensors across games. Keys contain public observations only.
const MAX_CACHED_STATES_V4: usize = 64;

pub struct V4OnnxRuntime {
    state_encoder: Mutex<Session>,
    factorized_scorer: Arc<Mutex<Session>>,
    metadata: BundleMetadataV4,
    vocabulary: VocabularyV3,
    options: V3RuntimeOptions,
    session_runs: Arc<AtomicU64>,
    cache: Mutex<HashMap<Vec<u8>, EncodedStateV4>>,
}

impl V4OnnxRuntime {
    pub fn from_bundle(
        metadata_path: impl AsRef<Path>,
        expected_ruleset_id: &str,
        expected_ruleset_manifest_sha256: &str,
    ) -> Result<Self, V4RuntimeError> {
        Self::from_bundle_with_options(
            metadata_path,
            expected_ruleset_id,
            expected_ruleset_manifest_sha256,
            V3RuntimeOptions::default(),
        )
    }

    pub fn from_bundle_with_options(
        metadata_path: impl AsRef<Path>,
        expected_ruleset_id: &str,
        expected_ruleset_manifest_sha256: &str,
        options: V3RuntimeOptions,
    ) -> Result<Self, V4RuntimeError> {
        if options.intra_threads == 0 {
            return Err(V4RuntimeError::Inference(
                "ONNX intra_threads must be positive".to_owned(),
            ));
        }
        let metadata_path = metadata_path.as_ref();
        let metadata: BundleMetadataV4 = serde_json::from_slice(
            &fs::read(metadata_path).map_err(|error| V4RuntimeError::Io(error.to_string()))?,
        )
        .map_err(|error| V4RuntimeError::Metadata(error.to_string()))?;
        let vocabulary = validate_bundle_metadata(
            &metadata,
            expected_ruleset_id,
            expected_ruleset_manifest_sha256,
        )?;
        let base = metadata_path.parent().unwrap_or_else(|| Path::new("."));
        let encoder_path = resolve_graph_path(&metadata.state_encoder, base, "state encoder")?;
        let scorer_path =
            resolve_graph_path(&metadata.factorized_scorer, base, "factorized scorer")?;
        if encoder_path == scorer_path {
            return metadata_error("V4 bundle graphs must use distinct sibling paths");
        }
        let state_encoder = build_session(&encoder_path, options)?;
        let factorized_scorer = build_session(&scorer_path, options)?;
        validate_session_contract(
            &state_encoder,
            &metadata.state_encoder.inputs,
            &metadata.state_encoder.outputs,
            "V4 state encoder",
        )?;
        validate_session_contract(
            &factorized_scorer,
            &metadata.factorized_scorer.inputs,
            &metadata.factorized_scorer.outputs,
            "V4 factorized scorer",
        )?;
        Ok(Self {
            state_encoder: Mutex::new(state_encoder),
            factorized_scorer: Arc::new(Mutex::new(factorized_scorer)),
            metadata,
            vocabulary,
            options,
            session_runs: Arc::new(AtomicU64::new(0)),
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn session_runs(&self) -> u64 {
        self.session_runs.load(Ordering::Relaxed)
    }

    pub fn options(&self) -> V3RuntimeOptions {
        self.options
    }

    pub fn ruleset_id(&self) -> &str {
        &self.metadata.ruleset_id
    }

    /// Evaluate one decision from a live `GameState` and its exact legal set.
    pub fn evaluate_factorized(
        &self,
        state: &GameState,
        legal_actions: &LegalActionSetV3,
    ) -> Result<FactorizedEvaluationV4, V4RuntimeError> {
        if state.ruleset_id != self.metadata.ruleset_id {
            return metadata_error(format!(
                "state ruleset {} differs from model ruleset {}",
                state.ruleset_id, self.metadata.ruleset_id
            ));
        }
        let decision = self.prepare_decision(legal_actions)?;
        let encoding = self.cached_state_encoding(state, legal_actions)?;
        let action_logits = self.score_decision(&decision, &encoding)?;
        Ok(FactorizedEvaluationV4 {
            value: encoding.value,
            action_logits,
        })
    }

    /// Evaluate one decision from an already-encoded observation.
    ///
    /// This seam exists for replay/fixture validation: the ONNX graphs only ever
    /// see `ObservationV3` tensors, so recorded observations can be replayed
    /// without reconstructing a `GameState`.
    pub fn evaluate_observation_factorized(
        &self,
        observation: &ObservationV3,
        legal_actions: &LegalActionSetV3,
    ) -> Result<FactorizedEvaluationV4, V4RuntimeError> {
        if observation.ruleset_id != self.metadata.ruleset_id
            || observation.schema_id != OBSERVATION_V3_SCHEMA_ID
            || observation.spatial_shape != [crate::OBSERVATION_V3_SPATIAL_PLANES, 8, 8]
            || observation.spatial.len() != crate::OBSERVATION_V3_SPATIAL_SIZE
            || observation.global_shape != [32]
            || observation.global.len() != 32
        {
            return tensor_error("V4 observation identity or dense shape mismatch");
        }
        let decision = self.prepare_decision(legal_actions)?;
        let encoding = self.encode_observation(observation, legal_actions)?;
        let action_logits = self.score_decision(&decision, &encoding)?;
        Ok(FactorizedEvaluationV4 {
            value: encoding.value,
            action_logits,
        })
    }

    fn prepare_decision(
        &self,
        legal_actions: &LegalActionSetV3,
    ) -> Result<FactorizedDecisionV4, V4RuntimeError> {
        if legal_actions.is_empty() {
            return tensor_error("factorized evaluation requires at least one legal action");
        }
        if legal_actions.schema_id != ACTION_V3_SCHEMA_ID {
            return tensor_error("ActionV3 schema mismatch");
        }
        build_factorized_decision_v4(&legal_actions.actions)
    }

    fn cached_state_encoding(
        &self,
        state: &GameState,
        legal_actions: &LegalActionSetV3,
    ) -> Result<EncodedStateV4, V4RuntimeError> {
        let observation = encode_observation_v3(state);
        let key = serde_json::to_vec(&observation)
            .map_err(|error| V4RuntimeError::Tensor(error.to_string()))?;
        {
            let cache = self.cache.lock().map_err(|_| {
                V4RuntimeError::Inference("V4 state cache lock poisoned".to_owned())
            })?;
            if let Some(encoding) = cache.get(&key) {
                return Ok(encoding.clone());
            }
        }
        let encoding = self.encode_observation(&observation, legal_actions)?;
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| V4RuntimeError::Inference("V4 state cache lock poisoned".to_owned()))?;
        // A simple bounded cache avoids retaining every search state for the runtime's lifetime.
        if cache.len() >= MAX_CACHED_STATES_V4 {
            cache.clear();
        }
        cache.insert(key, encoding.clone());
        Ok(encoding)
    }

    fn encode_observation(
        &self,
        observation: &ObservationV3,
        legal_actions: &LegalActionSetV3,
    ) -> Result<EncodedStateV4, V4RuntimeError> {
        let mut tensors = tensorize_v3(
            std::slice::from_ref(observation),
            std::slice::from_ref(legal_actions),
            &self.vocabulary,
        )?;
        let mut lookup_tensors = HashMap::new();
        for &name in LOOKUP_INPUT_NAMES_V4 {
            let tensor = tensors.get(name).cloned().ok_or_else(|| {
                V4RuntimeError::Tensor(format!("missing scorer lookup tensor {name}"))
            })?;
            lookup_tensors.insert(name.to_owned(), tensor);
        }
        let mut inputs = Vec::with_capacity(self.metadata.state_encoder.inputs.len());
        for declared in &self.metadata.state_encoder.inputs {
            let tensor = tensors.remove(&declared.name).ok_or_else(|| {
                V4RuntimeError::Tensor(format!("missing state encoder input {}", declared.name))
            })?;
            if tensor.rank() != declared.rank || tensor.dtype() != declared.dtype {
                return tensor_error(format!(
                    "state encoder input {} type/rank drifted",
                    declared.name
                ));
            }
            inputs.push((declared.name.clone(), tensor.into_ort()?));
        }
        let mut session = self.state_encoder.lock().map_err(|_| {
            V4RuntimeError::Inference("V4 state encoder session lock poisoned".to_owned())
        })?;
        let outputs = session
            .run(inputs)
            .map_err(|error| V4RuntimeError::Inference(error.to_string()))?;
        self.session_runs.fetch_add(1, Ordering::Relaxed);
        let mut encoder_outputs = HashMap::new();
        for &name in ENCODER_OUTPUT_NAMES_V4 {
            let output = outputs.get(name).ok_or_else(|| {
                V4RuntimeError::Inference(format!("state encoder omitted output {name}"))
            })?;
            let (shape, values) = output
                .try_extract_tensor::<f32>()
                .map_err(|error| V4RuntimeError::Inference(error.to_string()))?;
            let shape = shape
                .iter()
                .map(|dimension| usize::try_from(*dimension))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| V4RuntimeError::Tensor(format!("invalid {name} shape")))?;
            let dim = self.metadata.model_config.model_dim;
            let expected_shape = match name {
                "state_embedding" => vec![1, dim],
                "board_tokens" => vec![1, 64, dim],
                "piece_entities" => vec![1, observation.pieces.len().max(1), dim],
                "card_entities" => vec![1, observation.cards.len().max(1), dim],
                _ => unreachable!("fixed encoder output names"),
            };
            if shape != expected_shape {
                return tensor_error(format!("{name} shape differs from the encoder contract"));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return tensor_error(format!("{name} contains NaN or infinity"));
            }
            encoder_outputs.insert(
                name.to_owned(),
                TensorDataV3::F32 {
                    shape,
                    values: values.to_vec(),
                },
            );
        }
        let (shape, values) = outputs["value"]
            .try_extract_tensor::<f32>()
            .map_err(|error| V4RuntimeError::Inference(error.to_string()))?;
        if shape.as_ref() != [1]
            || values.len() != 1
            || !values[0].is_finite()
            || !(-1.0001..=1.0001).contains(&values[0])
        {
            return tensor_error("state encoder returned an invalid value");
        }
        Ok(EncodedStateV4 {
            encoder_outputs,
            lookup_tensors,
            value: values[0],
        })
    }

    fn score_decision(
        &self,
        decision: &FactorizedDecisionV4,
        encoding: &EncodedStateV4,
    ) -> Result<Vec<f32>, V4RuntimeError> {
        let (logits, max_children) = score_factor_rows_v4(
            &decision.rows,
            encoding,
            &self.factorized_scorer,
            &self.metadata.factorized_scorer.inputs,
            &self.session_runs,
        )?;
        compose_action_log_probabilities(decision, &logits, max_children)
    }
}

fn score_factor_rows_v4(
    decision_rows: &[FactorRowV4],
    encoding: &EncodedStateV4,
    session: &Mutex<Session>,
    declared_inputs: &[ModelInputV3],
    session_runs: &AtomicU64,
) -> Result<(Vec<f32>, usize), V4RuntimeError> {
    let rows = decision_rows.len();
    let max_children = decision_rows
        .iter()
        .map(|row| row.children.len())
        .max()
        .unwrap_or(0);
    let max_prefix = decision_rows
        .iter()
        .map(|row| row.prefix.len())
        .max()
        .unwrap_or(0)
        .max(1);
    if rows == 0 || max_children == 0 {
        return tensor_error("factorized decision must contain rows with legal children");
    }
    let prefix_edges: Vec<_> = decision_rows
        .iter()
        .map(|row| row.prefix_edges.clone())
        .collect();
    let child_edges: Vec<_> = decision_rows
        .iter()
        .map(|row| row.child_edges.clone())
        .collect();
    let mut tensors = HashMap::new();
    insert_edge_tensors(&mut tensors, "prefix", rows, max_prefix, &prefix_edges)?;
    insert_edge_tensors(&mut tensors, "child", rows, max_children, &child_edges)?;
    let mut legal_mask = TensorDataV3::Bool {
        shape: vec![rows, max_children],
        values: vec![false; rows * max_children],
    };
    if let TensorDataV3::Bool { values, .. } = &mut legal_mask {
        for (row, entry) in decision_rows.iter().enumerate() {
            for column in 0..entry.children.len() {
                values[row * max_children + column] = true;
            }
        }
    }
    tensors.insert("legal_mask".to_owned(), legal_mask);
    tensors.insert(
        "state_index".to_owned(),
        TensorDataV3::I64 {
            shape: vec![rows],
            values: vec![0; rows],
        },
    );
    for (name, tensor) in encoding
        .encoder_outputs
        .iter()
        .chain(encoding.lookup_tensors.iter())
    {
        tensors.insert(name.clone(), tensor.clone());
    }
    let mut inputs = Vec::with_capacity(declared_inputs.len());
    for declared in declared_inputs {
        let tensor = tensors.remove(&declared.name).ok_or_else(|| {
            V4RuntimeError::Tensor(format!("missing factorized scorer input {}", declared.name))
        })?;
        if tensor.rank() != declared.rank || tensor.dtype() != declared.dtype {
            return tensor_error(format!(
                "factorized scorer input {} type/rank drifted",
                declared.name
            ));
        }
        inputs.push((declared.name.clone(), tensor.into_ort()?));
    }
    let mut session = session.lock().map_err(|_| {
        V4RuntimeError::Inference("V4 factorized scorer session lock poisoned".to_owned())
    })?;
    let outputs = session
        .run(inputs)
        .map_err(|error| V4RuntimeError::Inference(error.to_string()))?;
    session_runs.fetch_add(1, Ordering::Relaxed);
    let (shape, logits) = outputs["prefix_logits"]
        .try_extract_tensor::<f32>()
        .map_err(|error| V4RuntimeError::Inference(error.to_string()))?;
    if shape.as_ref() != [rows as i64, max_children as i64] || logits.len() != rows * max_children {
        return tensor_error("factorized scorer output shape does not match the prefix rows");
    }
    for (row_index, row) in decision_rows.iter().enumerate() {
        let start = row_index * max_children;
        for column in row.children.len()..max_children {
            if !logits[start + column].is_finite() || logits[start + column] > f32::MIN / 2.0 {
                return tensor_error(
                    "factorized scorer leaked a logit into an illegal child column",
                );
            }
        }
    }
    Ok((logits.to_vec(), max_children))
}

/// Own encoder outputs for a search node, even after the bounded runtime cache
/// evicts that observation. No GameState or live RNG is retained here.
struct OnnxPrefixScorerV4 {
    encoding: EncodedStateV4,
    session: Arc<Mutex<Session>>,
    declared_inputs: Vec<ModelInputV3>,
    session_runs: Arc<AtomicU64>,
}

impl PrefixScorerV4 for OnnxPrefixScorerV4 {
    fn score_prefix_v4(
        &self,
        prefix: &[String],
        children: &[String],
    ) -> Result<Vec<f32>, MctsV4Error> {
        let score = || -> Result<Vec<f32>, V4RuntimeError> {
            let row = FactorRowV4 {
                prefix: prefix.to_vec(),
                children: children.to_vec(),
                prefix_edges: prefix
                    .iter()
                    .map(|token| encode_factor_edge_v4(token))
                    .collect::<Result<Vec<_>, _>>()?,
                child_edges: children
                    .iter()
                    .map(|token| encode_factor_edge_v4(token))
                    .collect::<Result<Vec<_>, _>>()?,
            };
            let (logits, _) = score_factor_rows_v4(
                &[row],
                &self.encoding,
                &self.session,
                &self.declared_inputs,
                &self.session_runs,
            )?;
            if logits.iter().any(|logit| !logit.is_finite()) {
                return tensor_error("prefix logits contain NaN or infinity");
            }
            Ok(logits)
        };
        score().map_err(|error| MctsV4Error::Evaluation(error.to_string()))
    }
}

impl FactorizedEvaluatorV4 for V4OnnxRuntime {
    fn evaluate_v4(
        &self,
        state: &GameState,
        legal_actions: &LegalActionSetV3,
        _engine: &Engine,
    ) -> Result<FactorizedEvaluationV4, MctsV4Error> {
        self.evaluate_factorized(state, legal_actions)
            .map_err(|error| MctsV4Error::Evaluation(error.to_string()))
    }

    fn prepare_prefix_evaluation_v4(
        &self,
        state: &GameState,
        legal_actions: &LegalActionSetV3,
        _engine: &Engine,
    ) -> Result<Option<PreparedPrefixEvaluationV4>, MctsV4Error> {
        let prepare = || -> Result<PreparedPrefixEvaluationV4, V4RuntimeError> {
            if state.ruleset_id != self.metadata.ruleset_id {
                return metadata_error(format!(
                    "state ruleset {} differs from model ruleset {}",
                    state.ruleset_id, self.metadata.ruleset_id,
                ));
            }
            if legal_actions.is_empty() || legal_actions.schema_id != ACTION_V3_SCHEMA_ID {
                return tensor_error("factorized evaluation requires an exact ActionV3 legal set");
            }
            let encoding = self.cached_state_encoding(state, legal_actions)?;
            Ok(PreparedPrefixEvaluationV4 {
                value: encoding.value,
                scorer: Box::new(OnnxPrefixScorerV4 {
                    encoding,
                    session: Arc::clone(&self.factorized_scorer),
                    declared_inputs: self.metadata.factorized_scorer.inputs.clone(),
                    session_runs: Arc::clone(&self.session_runs),
                }),
            })
        };
        prepare()
            .map(Some)
            .map_err(|error| MctsV4Error::Evaluation(error.to_string()))
    }
}

fn expected_contract(contract: &[(&str, &str, usize)]) -> Vec<ModelInputV3> {
    contract
        .iter()
        .map(|(name, dtype, rank)| model_input(name, dtype, *rank))
        .collect()
}

fn validate_bundle_metadata(
    metadata: &BundleMetadataV4,
    expected_ruleset_id: &str,
    expected_ruleset_manifest_sha256: &str,
) -> Result<VocabularyV3, V4RuntimeError> {
    if metadata.metadata_schema_id != V4_BUNDLE_METADATA_SCHEMA_ID {
        return metadata_error("unsupported V4 bundle metadata schema");
    }
    if metadata.observation_schema_id != OBSERVATION_V3_SCHEMA_ID
        || metadata.action_schema_id != ACTION_V3_SCHEMA_ID
        || metadata.factorized_policy_schema_id != FACTORIZED_POLICY_V4_SCHEMA_ID
    {
        return metadata_error("V4 bundle ObservationV3/ActionV3/factorized schema mismatch");
    }
    if metadata.ruleset_id != expected_ruleset_id
        || metadata.ruleset_manifest_sha256 != expected_ruleset_manifest_sha256
    {
        return metadata_error("V4 bundle ruleset artifact does not match the runtime");
    }
    if metadata.dataset.ruleset_id != metadata.ruleset_id
        || metadata.dataset.ruleset_manifest_sha256 != metadata.ruleset_manifest_sha256
    {
        return metadata_error("V4 bundle dataset identity disagrees with the bundle ruleset");
    }
    let required_piece_kind_rows = PIECE_KIND_VOCABULARY
        .iter()
        .map(|(code, _)| usize::from(*code))
        .max()
        .unwrap_or(0)
        + 1;
    if metadata.piece_kind_vocabulary_size != required_piece_kind_rows
        || metadata.model_config.piece_kind_vocabulary_size != required_piece_kind_rows
    {
        return metadata_error(format!(
            "V4 bundle piece-kind vocabulary has {} rows, expected {required_piece_kind_rows}",
            metadata.piece_kind_vocabulary_size
        ));
    }
    if metadata.model_config.text_vocabulary_size != metadata.vocabulary.tokens.len() {
        return metadata_error(
            "V4 bundle text vocabulary size disagrees with the vocabulary artifact",
        );
    }
    if metadata.model_config.channels == 0
        || metadata.model_config.residual_blocks == 0
        || metadata.model_config.model_dim == 0
        || metadata.model_config.runtime_message_passes == 0
        || metadata.code_commit.is_empty()
    {
        return metadata_error("V4 bundle model config and code provenance must be non-empty");
    }
    if metadata.opset < V4_MINIMUM_OPSET {
        return metadata_error(format!(
            "V4 bundle opset {} is below the required minimum {V4_MINIMUM_OPSET}",
            metadata.opset
        ));
    }
    if let Some(checkpoint) = &metadata.checkpoint_sha256 {
        if !is_sha256_hex(checkpoint) {
            return metadata_error("V4 bundle checkpoint hash is not a sha256 hex digest");
        }
    }
    if metadata.state_encoder.inputs != expected_contract(EXPECTED_STATE_ENCODER_INPUTS)
        || metadata.state_encoder.outputs != expected_contract(EXPECTED_STATE_ENCODER_OUTPUTS)
    {
        return metadata_error("V4 state encoder I/O contract mismatch");
    }
    if metadata.factorized_scorer.inputs != expected_contract(EXPECTED_FACTORIZED_SCORER_INPUTS)
        || metadata.factorized_scorer.outputs
            != expected_contract(EXPECTED_FACTORIZED_SCORER_OUTPUTS)
    {
        return metadata_error("V4 factorized scorer I/O contract mismatch");
    }
    let vocabulary = VocabularyV3::from_artifact(&metadata.vocabulary)?;
    validate_required_runtime_vocabulary(&vocabulary)?;
    Ok(vocabulary)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

fn resolve_graph_path(
    contract: &GraphContractV4,
    base: &Path,
    label: &str,
) -> Result<PathBuf, V4RuntimeError> {
    let relative = Path::new(&contract.path);
    if contract.path.is_empty()
        || relative.is_absolute()
        || relative.components().count() != 1
        || relative.file_name().is_none()
    {
        return metadata_error(format!("{label} path must name a sibling file"));
    }
    let path = base.join(relative);
    let bytes = fs::read(&path).map_err(|error| V4RuntimeError::Io(error.to_string()))?;
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != contract.sha256 {
        return metadata_error(format!("{label} ONNX checksum does not match metadata"));
    }
    Ok(path)
}

#[derive(Clone, Debug, Default)]
struct TrieNodeV4 {
    children: BTreeMap<String, TrieNodeV4>,
    action_index: Option<usize>,
}

/// Expand one decision's legal actions into conditional prefix rows.
///
/// Rows follow the Python collate traversal exactly: children are sorted by their
/// `"field:value"` wire token and rows are emitted post-order, so recorded parity
/// fixtures stay comparable across languages.
pub(crate) fn build_factorized_decision_v4(
    actions: &[ActionDescriptorV3],
) -> Result<FactorizedDecisionV4, V4RuntimeError> {
    if actions.is_empty() {
        return tensor_error("factorized decisions require at least one legal action");
    }
    let mut root = TrieNodeV4::default();
    for (action_index, action) in actions.iter().enumerate() {
        let path = factorized_path(action)?;
        if path.is_empty() {
            return tensor_error("canonical action paths must be non-empty");
        }
        let mut node = &mut root;
        for token in &path {
            if node.action_index.is_some() {
                return tensor_error("a complete action is a prefix of another action");
            }
            node = node.children.entry(token.clone()).or_default();
        }
        if let Some(first_index) = node.action_index {
            return tensor_error(format!(
                "complete actions must have unique canonical paths: first_index={first_index} second_index={action_index} path={path:?} first_action={:?} second_action={:?}",
                actions[first_index], action
            ));
        }
        if !node.children.is_empty() {
            return tensor_error(format!(
                "complete action path collides with an existing prefix: second_index={action_index} path={path:?} second_action={action:?}"
            ));
        }
        node.action_index = Some(action_index);
    }
    let mut rows = Vec::new();
    let mut row_of: HashMap<Vec<String>, usize> = HashMap::new();
    flatten_trie(&root, &mut Vec::new(), &mut rows, &mut row_of)?;
    let mut action_steps = Vec::with_capacity(actions.len());
    for action in actions {
        let path = factorized_path(action)?;
        let mut steps = Vec::with_capacity(path.len());
        let mut prefix = Vec::with_capacity(path.len());
        for token in &path {
            let row = row_of.get(&prefix).copied().ok_or_else(|| {
                V4RuntimeError::Tensor("factorized path is missing its parent row".to_owned())
            })?;
            let column = rows[row]
                .children
                .iter()
                .position(|child| child == token)
                .ok_or_else(|| {
                    V4RuntimeError::Tensor(format!("factorized row does not contain child {token}"))
                })?;
            steps.push(FactorStepV4 { row, column });
            prefix.push(token.clone());
        }
        action_steps.push(steps);
    }
    Ok(FactorizedDecisionV4 { rows, action_steps })
}

fn factorized_path(action: &ActionDescriptorV3) -> Result<Vec<String>, V4RuntimeError> {
    factorized_action_path_v4(action).map_err(|error| V4RuntimeError::Tensor(error.to_string()))
}

fn flatten_trie(
    node: &TrieNodeV4,
    prefix: &mut Vec<String>,
    rows: &mut Vec<FactorRowV4>,
    row_of: &mut HashMap<Vec<String>, usize>,
) -> Result<Option<usize>, V4RuntimeError> {
    if node.children.is_empty() {
        return Ok(None);
    }
    let mut children = Vec::with_capacity(node.children.len());
    let mut child_edges = Vec::with_capacity(node.children.len());
    for (token, child) in &node.children {
        prefix.push(token.clone());
        flatten_trie(child, prefix, rows, row_of)?;
        prefix.pop();
        children.push(token.clone());
        child_edges.push(encode_factor_edge_v4(token)?);
    }
    let prefix_edges = prefix
        .iter()
        .map(|token| encode_factor_edge_v4(token))
        .collect::<Result<Vec<_>, _>>()?;
    let index = rows.len();
    rows.push(FactorRowV4 {
        prefix: prefix.clone(),
        children,
        prefix_edges,
        child_edges,
    });
    row_of.insert(prefix.clone(), index);
    Ok(Some(index))
}

/// Encode one typed grammar edge as `field/category/index/ordinal` integers.
///
/// This is the exact wire contract of `v4.batch.encode_factor_edge`: `category`
/// carries the stable V3 enum for family, ability, target kind, and the two
/// sentinel categories (`complete:action`, `promotion:none`); `index` keeps raw
/// stable V3 indexes and stays `-1` for non-index values and padding; `ordinal`
/// is one-based only for ordered `target_N` fields.
pub(crate) fn encode_factor_edge_v4(token: &str) -> Result<FactorEdgeV4, V4RuntimeError> {
    let (mut field_name, value) = token.split_once(':').ok_or_else(|| {
        V4RuntimeError::Tensor(format!("factor token {token:?} is not `field:value`"))
    })?;
    let mut ordinal = 0i64;
    if let Some(rest) = field_name.strip_prefix("target_") {
        if let Ok(position) = rest.parse::<usize>() {
            ordinal = position as i64 + 1;
            field_name = "target_sequence";
        }
    }
    let field = FACTOR_FIELD_IDS_V4
        .iter()
        .find(|(name, _)| *name == field_name)
        .map(|(_, id)| *id)
        .ok_or_else(|| V4RuntimeError::Tensor(format!("unknown factor field {field_name:?}")))?;
    let mut category = 0i64;
    let mut index = -1i64;
    if field_name == "family" {
        category = action_kind_id_v4(value)?;
    } else if field_name == "ability" {
        category = ability_id_v4(value)?;
    } else if field_name == "target_kind" {
        category = target_kind_id_v4(value)?;
    } else if field_name == "complete" {
        if value != "action" {
            return tensor_error("complete edge must be `complete:action`");
        }
        category = 1;
    } else if field_name == "promotion" && value == "none" {
        category = 1;
    } else {
        let parsed: i64 = value.parse().map_err(|_| {
            V4RuntimeError::Tensor(format!("invalid {field_name} factor value {value:?}"))
        })?;
        if parsed < 0 {
            return tensor_error(format!("invalid {field_name} factor value {value:?}"));
        }
        index = parsed;
    }
    Ok(FactorEdgeV4 {
        field,
        category,
        index,
        ordinal,
    })
}

fn action_kind_id_v4(value: &str) -> Result<i64, V4RuntimeError> {
    match value {
        "move" => Ok(1),
        "piece_ability" => Ok(2),
        "card" => Ok(3),
        "draft_pick" => Ok(4),
        "resolve_choice" => Ok(5),
        _ => tensor_error(format!("unknown family factor value {value:?}")),
    }
}

fn ability_id_v4(value: &str) -> Result<i64, V4RuntimeError> {
    match value {
        "wizard_lightning" => Ok(1),
        "wizard_shield" => Ok(2),
        "wizard_meteor" => Ok(3),
        "wizard_time_stop" => Ok(4),
        "colossus_sector" => Ok(5),
        "merchant_buy" => Ok(6),
        _ => tensor_error(format!("unknown ability factor value {value:?}")),
    }
}

fn target_kind_id_v4(value: &str) -> Result<i64, V4RuntimeError> {
    match value {
        "none" => Ok(1),
        "square" => Ok(2),
        "piece_at" => Ok(3),
        "squares" => Ok(4),
        _ => tensor_error(format!("unknown target_kind factor value {value:?}")),
    }
}

fn insert_edge_tensors(
    tensors: &mut HashMap<String, TensorDataV3>,
    label: &str,
    rows: usize,
    width: usize,
    edges: &[Vec<FactorEdgeV4>],
) -> Result<(), V4RuntimeError> {
    if edges.len() != rows {
        return tensor_error("factor edge rows are misaligned");
    }
    let total = rows * width;
    let mut field = vec![0i64; total];
    let mut category = vec![0i64; total];
    let mut index = vec![-1i64; total];
    let mut ordinal = vec![0i64; total];
    let mut mask = vec![false; total];
    for (row, row_edges) in edges.iter().enumerate() {
        if row_edges.len() > width {
            return tensor_error("factor edge row exceeds the padded width");
        }
        for (column, edge) in row_edges.iter().enumerate() {
            let offset = row * width + column;
            field[offset] = edge.field;
            category[offset] = edge.category;
            index[offset] = edge.index;
            ordinal[offset] = edge.ordinal;
            mask[offset] = true;
        }
    }
    let shape = if width == 0 {
        vec![rows]
    } else {
        vec![rows, width]
    };
    tensors.insert(
        format!("{label}_field"),
        TensorDataV3::I64 {
            shape: shape.clone(),
            values: field,
        },
    );
    tensors.insert(
        format!("{label}_category"),
        TensorDataV3::I64 {
            shape: shape.clone(),
            values: category,
        },
    );
    tensors.insert(
        format!("{label}_index"),
        TensorDataV3::I64 {
            shape: shape.clone(),
            values: index,
        },
    );
    tensors.insert(
        format!("{label}_ordinal"),
        TensorDataV3::I64 {
            shape: shape.clone(),
            values: ordinal,
        },
    );
    tensors.insert(
        format!("{label}_mask"),
        TensorDataV3::Bool {
            shape,
            values: mask,
        },
    );
    Ok(())
}

/// Turn flat conditional prefix logits into complete-action log probabilities.
///
/// `flat_logits` is the `[rows, max_children]` scorer output; each path step
/// contributes `log_softmax(row logits over legal children)[column]`.
pub(crate) fn compose_action_log_probabilities(
    decision: &FactorizedDecisionV4,
    flat_logits: &[f32],
    max_children: usize,
) -> Result<Vec<f32>, V4RuntimeError> {
    if decision
        .rows
        .iter()
        .any(|row| row.children.len() > max_children)
        || flat_logits.len() != decision.rows.len() * max_children
    {
        return tensor_error("prefix logits do not cover every row");
    }
    let mut action_logits = Vec::with_capacity(decision.action_steps.len());
    for steps in &decision.action_steps {
        if steps.is_empty() {
            return tensor_error("factorized action has no path steps");
        }
        let mut log_probability = 0.0f32;
        for step in steps {
            let row = decision.rows.get(step.row).ok_or_else(|| {
                V4RuntimeError::Tensor("factorized path row is missing".to_owned())
            })?;
            if step.column >= row.children.len() {
                return tensor_error("factorized path references an illegal child column");
            }
            let start = step.row * max_children;
            let legal = &flat_logits[start..start + row.children.len()];
            if legal.iter().any(|value| !value.is_finite()) {
                return tensor_error("prefix logits contain NaN or infinity");
            }
            log_probability += log_softmax_at(legal, step.column)?;
        }
        action_logits.push(log_probability);
    }
    let probability_mass: f32 = action_logits.iter().map(|value| value.exp()).sum();
    if !probability_mass.is_finite() || (probability_mass - 1.0).abs() > 1e-3 {
        return tensor_error(format!(
            "complete-action probabilities sum to {probability_mass}"
        ));
    }
    Ok(action_logits)
}

fn log_softmax_at(logits: &[f32], column: usize) -> Result<f32, V4RuntimeError> {
    if logits.is_empty() || column >= logits.len() {
        return tensor_error("log-softmax column is outside the legal children");
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = logits.iter().map(|value| (*value - maximum).exp()).sum();
    if !total.is_finite() || total <= 0.0 {
        return tensor_error("prefix log-softmax has invalid mass");
    }
    Ok((logits[column] - maximum) - total.ln())
}

fn metadata_error<T>(message: impl Into<String>) -> Result<T, V4RuntimeError> {
    Err(V4RuntimeError::Metadata(message.into()))
}

fn tensor_error<T>(message: impl Into<String>) -> Result<T, V4RuntimeError> {
    Err(V4RuntimeError::Tensor(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    const FIXTURE_DIR: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../train/v3/fixtures/v4_bundle"
    );

    #[derive(Deserialize)]
    struct ParityFixtureV4 {
        cases: Vec<ParityCaseV4>,
        expected_value: Vec<f32>,
        expected_action_logits: Vec<Vec<f32>>,
    }

    #[derive(Deserialize)]
    struct ParityCaseV4 {
        observation: ObservationV3,
        legal_actions: LegalActionSetV3,
    }

    #[test]
    fn factor_edge_encoding_matches_the_python_contract() {
        assert_eq!(
            encode_factor_edge_v4("family:move").unwrap(),
            FactorEdgeV4 {
                field: 1,
                category: 1,
                index: -1,
                ordinal: 0,
            }
        );
        assert_eq!(
            encode_factor_edge_v4("piece:0").unwrap(),
            FactorEdgeV4 {
                field: 2,
                category: 0,
                index: 0,
                ordinal: 0,
            }
        );
        assert_eq!(
            encode_factor_edge_v4("promotion:none").unwrap(),
            FactorEdgeV4 {
                field: 4,
                category: 1,
                index: -1,
                ordinal: 0,
            }
        );
        assert_eq!(
            encode_factor_edge_v4("ability:wizard_shield").unwrap(),
            FactorEdgeV4 {
                field: 5,
                category: 2,
                index: -1,
                ordinal: 0,
            }
        );
        assert_eq!(
            encode_factor_edge_v4("target_2:5").unwrap(),
            FactorEdgeV4 {
                field: 8,
                category: 0,
                index: 5,
                ordinal: 3,
            }
        );
        assert_eq!(
            encode_factor_edge_v4("complete:action").unwrap(),
            FactorEdgeV4 {
                field: 13,
                category: 1,
                index: -1,
                ordinal: 0,
            }
        );
    }

    #[test]
    fn complete_action_probabilities_are_products_of_prefix_conditionals() {
        let actions = vec![
            ActionDescriptorV3::DraftPick {
                offer_index: 0,
                definition_id: "first".to_owned(),
            },
            ActionDescriptorV3::DraftPick {
                offer_index: 1,
                definition_id: "second".to_owned(),
            },
        ];
        let decision = build_factorized_decision_v4(&actions).unwrap();
        let width = decision
            .rows
            .iter()
            .map(|row| row.children.len())
            .max()
            .unwrap();
        let logits = vec![0.0; decision.rows.len() * width];
        let complete = compose_action_log_probabilities(&decision, &logits, width).unwrap();
        assert_eq!(complete.len(), 2);
        assert!((complete[0].exp() - 0.5).abs() < 1e-6);
        assert!((complete[1].exp() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bundle_metadata_fails_closed_on_schema_and_graph_hash_drift() {
        let metadata_path = Path::new(FIXTURE_DIR).join("v4-model.v4-bundle.metadata.json");
        let metadata: BundleMetadataV4 =
            serde_json::from_slice(&fs::read(&metadata_path).unwrap()).unwrap();
        let mut stale = metadata.clone();
        stale.factorized_policy_schema_id = "stale".to_owned();
        assert!(
            validate_bundle_metadata(
                &stale,
                &metadata.ruleset_id,
                &metadata.ruleset_manifest_sha256
            )
            .is_err()
        );

        validate_bundle_metadata(
            &metadata,
            &metadata.ruleset_id,
            &metadata.ruleset_manifest_sha256,
        )
        .unwrap();
        let mut wrong_rank = metadata.clone();
        wrong_rank.factorized_scorer.inputs[0].rank += 1;
        assert!(
            validate_bundle_metadata(
                &wrong_rank,
                &metadata.ruleset_id,
                &metadata.ruleset_manifest_sha256
            )
            .is_err()
        );
        let mut wrong_dtype = metadata.clone();
        wrong_dtype.state_encoder.inputs[0].dtype = "int64".to_owned();
        assert!(
            validate_bundle_metadata(
                &wrong_dtype,
                &metadata.ruleset_id,
                &metadata.ruleset_manifest_sha256
            )
            .is_err()
        );
        assert!(
            validate_bundle_metadata(
                &metadata,
                "wrong-ruleset",
                &metadata.ruleset_manifest_sha256
            )
            .is_err()
        );
        assert!(
            validate_bundle_metadata(&metadata, &metadata.ruleset_id, &"1".repeat(64)).is_err()
        );
        let mut bad_graph = metadata.factorized_scorer.clone();
        bad_graph.sha256 = "0".repeat(64);
        assert!(
            resolve_graph_path(&bad_graph, Path::new(FIXTURE_DIR), "factorized scorer").is_err()
        );
    }

    #[test]
    fn onnx_accepts_engine_legal_destination_63() {
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            eprintln!("skipping V4 ONNX destination regression: ORT_DYLIB_PATH absent");
            return;
        }
        use augment_core::{BUNDLED_RULESET_ID, Board, GamePhase, Mode, Piece, PieceKind, Player};
        let mut board = Board::empty();
        board.set(
            "a1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King).with_id(1)),
        );
        board.set(
            "h1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook).with_id(2)),
        );
        board.set(
            "a8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King).with_id(3)),
        );
        let mut state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 7);
        state.phase = GamePhase::Playing;
        let engine = Engine::bundled_20260807().unwrap();
        let legal = LegalActionSetV3::from_state(&state, &engine).unwrap();
        assert!(legal.actions.iter().any(|action| matches!(
            action,
            ActionDescriptorV3::Move {
                to: 63,
                promotion: None,
                ..
            }
        )));
        let runtime = V4OnnxRuntime::from_bundle(
            Path::new(FIXTURE_DIR).join("v4-model.v4-bundle.metadata.json"),
            BUNDLED_RULESET_ID,
            &"0".repeat(64),
        )
        .unwrap();
        let result = runtime.evaluate_factorized(&state, &legal).unwrap();
        assert_eq!(result.action_logits.len(), legal.len());
        assert!(result.action_logits.iter().all(|x| x.is_finite()));
        assert!((result.action_logits.iter().map(|x| x.exp()).sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn bounded_encoder_cache_evicts_and_stays_correct() {
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            eprintln!("skipping V4 cache eviction regression: ORT_DYLIB_PATH absent");
            return;
        }
        use augment_core::{BUNDLED_RULESET_ID, Board, GamePhase, Mode, Piece, PieceKind, Player};
        let mut board = Board::empty();
        board.set(
            "a1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King).with_id(1)),
        );
        board.set(
            "h1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook).with_id(2)),
        );
        board.set(
            "a8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King).with_id(3)),
        );
        let mut state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 7);
        state.phase = GamePhase::Playing;
        let engine = Engine::bundled_20260807().unwrap();
        let legal = LegalActionSetV3::from_state(&state, &engine).unwrap();
        let runtime = V4OnnxRuntime::from_bundle(
            Path::new(FIXTURE_DIR).join("v4-model.v4-bundle.metadata.json"),
            BUNDLED_RULESET_ID,
            &"0".repeat(64),
        )
        .unwrap();

        // 70 distinct public observations exceed the 64-entry cache cap.
        let mut first_pass = Vec::new();
        for ply in 7..77u32 {
            let mut variant = state.clone();
            variant.ply = ply;
            let observation = encode_observation_v3(&variant);
            let again = encode_observation_v3(&state);
            if ply != 7 {
                assert_ne!(observation, again, "ply {ply} must change the observation");
            }
            first_pass.push(runtime.evaluate_factorized(&variant, &legal).unwrap());
        }
        assert_eq!(
            runtime.session_runs(),
            140,
            "every distinct public state needs one encoder and one scorer run"
        );
        for evaluation in &first_pass {
            assert!(
                (evaluation
                    .action_logits
                    .iter()
                    .map(|x| x.exp())
                    .sum::<f32>()
                    - 1.0)
                    .abs()
                    < 1e-4
            );
        }

        // The oldest entry was evicted, so replaying it re-runs the encoder.
        let replayed_old = runtime.evaluate_factorized(&state, &legal).unwrap();
        assert_eq!(runtime.session_runs(), 142);
        // ORT CPU kernels select GEMM/conv paths from buffer alignment, so the
        // same input is not bitwise reproducible across an evaluation history:
        // observed drift here is ~6e-4 on logits and ~5e-3 on value. Accuracy
        // against Python is pinned by the parity test at 2e-4; this test pins
        // eviction semantics through the exact session-run counts above, with a
        // documented agreement bound that still fails on a wrong cache hit.
        for (left, right) in replayed_old
            .action_logits
            .iter()
            .zip(&first_pass[0].action_logits)
        {
            assert!((left - right).abs() < 2e-2, "{left} != {right}");
        }
        assert!(
            (replayed_old.value - first_pass[0].value).abs() < 2e-2,
            "replayed value {} vs original {}",
            replayed_old.value,
            first_pass[0].value
        );

        // The most recent entry is still cached and runs the scorer only.
        let mut newest = state.clone();
        newest.ply = 76;
        let replayed_new = runtime.evaluate_factorized(&newest, &legal).unwrap();
        assert_eq!(runtime.session_runs(), 143);
        for (left, right) in replayed_new
            .action_logits
            .iter()
            .zip(&first_pass[69].action_logits)
        {
            assert!((left - right).abs() < 2e-5, "{left} != {right}");
        }
    }

    #[test]
    fn checked_in_onnx_bundle_matches_python_action_logits_and_values() {
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            eprintln!("skipping V4 ONNX parity because ORT_DYLIB_PATH is not configured");
            return;
        }
        let metadata_path = Path::new(FIXTURE_DIR).join("v4-model.v4-bundle.metadata.json");
        let runtime =
            V4OnnxRuntime::from_bundle(&metadata_path, "augment_chess_20260818", &"0".repeat(64))
                .unwrap();
        let parity: ParityFixtureV4 = serde_json::from_slice(
            &fs::read(Path::new(FIXTURE_DIR).join("v4-model.parity.json")).unwrap(),
        )
        .unwrap();
        for (index, case) in parity.cases.iter().enumerate() {
            let actual = runtime
                .evaluate_observation_factorized(&case.observation, &case.legal_actions)
                .unwrap();
            assert!((actual.value - parity.expected_value[index]).abs() < 2e-4);
            assert_eq!(
                actual.action_logits.len(),
                parity.expected_action_logits[index].len()
            );
            for (actual, expected) in actual
                .action_logits
                .iter()
                .zip(&parity.expected_action_logits[index])
            {
                assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
            }
        }
        assert_eq!(runtime.session_runs(), (parity.cases.len() * 2) as u64);
    }
}

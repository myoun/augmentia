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

#[derive(Clone, Debug, Deserialize)]
struct VocabularyArtifactV3 {
    schema_id: String,
    tokens: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelInputV3 {
    name: String,
    dtype: String,
    rank: usize,
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
}

#[derive(Clone, Debug)]
struct VocabularyV3 {
    lookup: HashMap<String, i64>,
}

impl VocabularyV3 {
    fn from_artifact(artifact: &VocabularyArtifactV3) -> Result<Self, V3RuntimeError> {
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

    fn index(&self, token: Option<&str>, path: &str) -> Result<i64, V3RuntimeError> {
        let Some(token) = token else { return Ok(0) };
        self.lookup.get(token).copied().ok_or_else(|| {
            V3RuntimeError::Tensor(format!("{path} contains unknown token {token:?}"))
        })
    }
}

pub struct V3OnnxRuntime {
    session: Mutex<Session>,
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
        let metadata: ModelMetadataV3 = serde_json::from_slice(
            &fs::read(metadata_path.as_ref())
                .map_err(|error| V3RuntimeError::Io(error.to_string()))?,
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
        let session = builder
            .commit_from_file(model_path)
            .map_err(|error| V3RuntimeError::Inference(error.to_string()))?;
        Ok(Self {
            session: Mutex::new(session),
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
        let mut observations = Vec::with_capacity(states.len());
        let mut legal_sets = Vec::with_capacity(states.len());
        for state in states {
            if state.ruleset_id != self.metadata.ruleset_id {
                return metadata(format!(
                    "state ruleset {} differs from model ruleset {}",
                    state.ruleset_id, self.metadata.ruleset_id
                ));
            }
            let legal = LegalActionSetV3::from_state(state, engine)
                .map_err(|error| V3RuntimeError::Engine(error.to_string()))?;
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
            legal_sets.push(legal);
        }
        let mut tensors = tensorize_v3(&observations, &legal_sets, &self.vocabulary)?;
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
        for (batch_index, ((state, legal), value)) in states
            .iter()
            .zip(&legal_sets)
            .zip(values.iter().copied())
            .enumerate()
        {
            if !value.is_finite() || !(-1.0001..=1.0001).contains(&value) {
                return tensor_error(format!("invalid value output {value}"));
            }
            let start = batch_index * max_actions;
            let probabilities = softmax(&logits[start..start + legal.len()])?;
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
}

fn validate_required_runtime_vocabulary(vocabulary: &VocabularyV3) -> Result<(), V3RuntimeError> {
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
enum TensorDataV3 {
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
    fn rank(&self) -> usize {
        match self {
            Self::F32 { shape, .. } | Self::I64 { shape, .. } | Self::Bool { shape, .. } => {
                shape.len()
            }
        }
    }
    fn dtype(&self) -> String {
        match self {
            Self::F32 { .. } => "float32",
            Self::I64 { .. } => "int64",
            Self::Bool { .. } => "bool",
        }
        .to_owned()
    }
    fn into_ort(self) -> Result<SessionInputValue<'static>, V3RuntimeError> {
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

fn tensorize_v3(
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
                assert_eq!(shape, &[cases.len(), 116, 8, 8]);
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

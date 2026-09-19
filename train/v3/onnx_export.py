"""V3 ONNX export, artifact metadata, and numerical equivalence checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from .model import AugmentNetV3, V3ModelConfig, V3StateEncoding
from .mixture import load_dataset_from_identity
from .replay import collate_examples
from .schema import (
    ACTION_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    default_runtime_vocabulary_path,
    load_golden_suite,
    load_runtime_vocabulary,
)
from .tensorize import V3Batch, V3Vocabulary

MODEL_METADATA_SCHEMA_ID = "augmentia.onnx-model-metadata.v3"
SPLIT_POLICY_BUNDLE_SCHEMA_ID = "augmentia.onnx-split-policy-bundle.v1"
INPUT_NAMES = tuple(field.name for field in fields(V3Batch))
ACTION_INPUT_NAMES = tuple(name for name in INPUT_NAMES if name.startswith("action_"))
STATE_INPUT_NAMES = tuple(name for name in INPUT_NAMES if name not in ACTION_INPUT_NAMES)
# Action reference lookup needs these state-side tensors in addition to the cached encoder outputs.
SCORER_LOOKUP_INPUT_NAMES = ("piece_slot", "piece_mask", "card_slot", "card_mask")
STATE_ENCODER_OUTPUT_NAMES = (
    "state_embedding",
    "board_tokens",
    "piece_entities",
    "card_entities",
    "value",
)
CANDIDATE_SCORER_OUTPUT_NAMES = ("policy_logits",)


class V3OnnxWrapper(nn.Module):
    def __init__(self, model: AugmentNetV3) -> None:
        super().__init__()
        self.model = model

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = V3Batch(**dict(zip(INPUT_NAMES, inputs, strict=True)))
        output = self.model._forward(batch)
        return output.policy_logits, output.value


class V3StateEncoderOnnxWrapper(nn.Module):
    """The reusable state half of the V3 policy network."""

    def __init__(self, model: AugmentNetV3) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, *inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = SimpleNamespace(**dict(zip(STATE_INPUT_NAMES, inputs, strict=True)))
        encoding = self.model.encode_state(batch)
        return (
            encoding.state,
            encoding.board_tokens,
            encoding.piece_entities,
            encoding.card_entities,
            self.model.value_head(encoding.state).squeeze(-1),
        )


class V3CandidateScorerOnnxWrapper(nn.Module):
    """Score an action subset from a previously exported state encoding."""

    def __init__(self, model: AugmentNetV3) -> None:
        super().__init__()
        self.model = model

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        cached_count = len(STATE_ENCODER_OUTPUT_NAMES) - 1
        cached = inputs[:cached_count]
        lookup_end = cached_count + len(SCORER_LOOKUP_INPUT_NAMES)
        lookup = inputs[cached_count:lookup_end]
        action = inputs[lookup_end:]
        batch = SimpleNamespace(
            **dict(
                zip(
                    (*SCORER_LOOKUP_INPUT_NAMES, *ACTION_INPUT_NAMES),
                    (*lookup, *action),
                    strict=True,
                )
            )
        )
        encoding = V3StateEncoding(
            state=cached[0],
            board_tokens=cached[1],
            piece_entities=cached[2],
            card_entities=cached[3],
        )
        return self.model.score_actions(batch, encoding)


def export_onnx_v3(
    model: AugmentNetV3,
    sample: V3Batch,
    output_path: str | Path,
    vocabulary: V3Vocabulary,
    dataset: Any,
    checkpoint_path: str | Path | None = None,
    code_commit: str = "unknown",
    opset: int = 18,
    vocabulary_extension: dict[str, Any] | None = None,
    checkpoint_migrations: Sequence[dict[str, Any]] = (),
) -> tuple[Path, Path]:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".onnx.tmp-{uuid.uuid4().hex}")
    model = model.eval().cpu()
    sample = sample.to("cpu")
    tensors = tuple(getattr(sample, name) for name in INPUT_NAMES)
    dynamic_axes = _dynamic_axes(sample)
    try:
        torch.onnx.export(
            V3OnnxWrapper(model),
            tensors,
            temporary,
            input_names=list(INPUT_NAMES),
            output_names=["policy_logits", "value"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
        _fsync(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    checkpoint_hash = _sha256(Path(checkpoint_path)) if checkpoint_path else None
    exported_inputs = tuple(
        item.name
        for item in ort.InferenceSession(
            str(output), providers=["CPUExecutionProvider"]
        ).get_inputs()
    )
    dataset_identity = dataset.to_json()
    ruleset_id, ruleset_manifest_sha256 = _ruleset_identity(dataset_identity)
    metadata = {
        "metadata_schema_id": MODEL_METADATA_SCHEMA_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "ruleset_id": ruleset_id,
        "ruleset_manifest_sha256": ruleset_manifest_sha256,
        "dataset": dataset_identity,
        "vocabulary": vocabulary.to_json(),
        "model_config": asdict(model.config),
        "piece_kind_vocabulary_size": model.config.piece_kind_vocabulary_size,
        "onnx_sha256": _sha256(output),
        "checkpoint_sha256": checkpoint_hash,
        "code_commit": code_commit,
        "opset": opset,
        "inputs": [
            {
                "name": name,
                "dtype": str(getattr(sample, name).dtype).removeprefix("torch."),
                "rank": getattr(sample, name).ndim,
            }
            for name in exported_inputs
        ],
        "outputs": ["policy_logits", "value"],
        "vocabulary_extension": vocabulary_extension,
        "checkpoint_migrations": list(checkpoint_migrations),
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata["split_policy_bundle"] = export_split_policy_bundle(model, sample, output, opset)
    _atomic_json(metadata_path, metadata)
    return output, metadata_path


def export_split_policy_bundle(
    model: AugmentNetV3,
    sample: V3Batch,
    monolithic_output: str | Path,
    opset: int = 18,
) -> dict[str, Any]:
    """Export co-versioned encoder/scorer graphs without changing the legacy model."""
    output = Path(monolithic_output).resolve()
    model = model.eval().cpu()
    sample = sample.to("cpu")
    encoder_path = output.with_suffix(".state-encoder.onnx")
    scorer_path = output.with_suffix(".candidate-scorer.onnx")
    _export_graph(
        V3StateEncoderOnnxWrapper(model),
        tuple(getattr(sample, name) for name in STATE_INPUT_NAMES),
        encoder_path,
        STATE_INPUT_NAMES,
        STATE_ENCODER_OUTPUT_NAMES,
        _state_encoder_dynamic_axes(sample),
        opset,
    )
    scorer_inputs = (
        *(
            getattr(sample, name)
            for name in SCORER_LOOKUP_INPUT_NAMES
        ),
        *(getattr(sample, name) for name in ACTION_INPUT_NAMES),
    )
    # These are tracing examples, not cached inference values. `inference_mode`
    # tensors cannot participate in the scorer export trace.
    with torch.no_grad():
        encoding = model.encode_state(sample)
    _export_graph(
        V3CandidateScorerOnnxWrapper(model),
        (
            encoding.state,
            encoding.board_tokens,
            encoding.piece_entities,
            encoding.card_entities,
            *scorer_inputs,
        ),
        scorer_path,
        (
            *STATE_ENCODER_OUTPUT_NAMES[:-1],
            *SCORER_LOOKUP_INPUT_NAMES,
            *ACTION_INPUT_NAMES,
        ),
        CANDIDATE_SCORER_OUTPUT_NAMES,
        _candidate_scorer_dynamic_axes(sample),
        opset,
    )
    return {
        "schema_id": SPLIT_POLICY_BUNDLE_SCHEMA_ID,
        "state_encoder": _onnx_graph_contract(encoder_path),
        "candidate_scorer": _onnx_graph_contract(scorer_path),
    }


def _export_graph(
    module: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    output: Path,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    temporary = output.with_suffix(f".onnx.tmp-{uuid.uuid4().hex}")
    try:
        torch.onnx.export(
            module,
            tensors,
            temporary,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
        _fsync(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _onnx_graph_contract(path: Path) -> dict[str, Any]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "inputs": [
            {"name": item.name, "rank": len(item.shape), "dtype": _ort_dtype(item.type)}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "rank": len(item.shape), "dtype": _ort_dtype(item.type)}
            for item in session.get_outputs()
        ],
    }


def _ort_dtype(type_name: str) -> str:
    return {
        "tensor(float)": "float32",
        "tensor(int64)": "int64",
        "tensor(bool)": "bool",
    }.get(type_name, type_name)


def _state_encoder_dynamic_axes(sample: V3Batch) -> dict[str, dict[int, str]]:
    result = {name: axes for name, axes in _dynamic_axes(sample).items() if name in STATE_INPUT_NAMES}
    result.update(
        {
            "state_embedding": {0: "batch"},
            "board_tokens": {0: "batch"},
            "piece_entities": {0: "batch", 1: "pieces"},
            "card_entities": {0: "batch", 1: "cards"},
            "value": {0: "batch"},
        }
    )
    return result


def _candidate_scorer_dynamic_axes(sample: V3Batch) -> dict[str, dict[int, str]]:
    result = {name: axes for name, axes in _dynamic_axes(sample).items() if name in ACTION_INPUT_NAMES}
    result.update(
        {
            "state_embedding": {0: "batch"},
            "board_tokens": {0: "batch"},
            "piece_entities": {0: "batch", 1: "pieces"},
            "card_entities": {0: "batch", 1: "cards"},
            "piece_slot": {0: "batch", 1: "pieces"},
            "piece_mask": {0: "batch", 1: "pieces"},
            "card_slot": {0: "batch", 1: "cards"},
            "card_mask": {0: "batch", 1: "cards"},
            "policy_logits": {0: "batch", 1: "legal_actions"},
        }
    )
    return result


def assert_onnx_equivalent(
    model: AugmentNetV3,
    batch: V3Batch,
    onnx_path: str | Path,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    rtol: float = 2e-4,
    atol: float = 2e-5,
) -> dict[str, float]:
    model.eval().cpu()
    batch = batch.to("cpu")
    with torch.inference_mode():
        expected = model(batch)
    session = ort.InferenceSession(str(onnx_path), providers=list(providers))
    input_names = tuple(item.name for item in session.get_inputs())
    inputs = {name: getattr(batch, name).detach().cpu().numpy() for name in input_names}
    actual_policy, actual_value = session.run(["policy_logits", "value"], inputs)
    expected_policy = expected.policy_logits.detach().numpy()
    expected_value = expected.value.detach().numpy()
    np.testing.assert_allclose(actual_policy, expected_policy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_value, expected_value, rtol=rtol, atol=atol)
    return {
        "policy_max_abs_error": float(np.max(np.abs(actual_policy - expected_policy))),
        "value_max_abs_error": float(np.max(np.abs(actual_value - expected_value))),
    }


def assert_split_onnx_equivalent(
    model: AugmentNetV3,
    batch: V3Batch,
    monolithic_path: str | Path,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    rtol: float = 2e-4,
    atol: float = 2e-5,
) -> dict[str, float]:
    """Verify that encoder/scorer composition is numerically identical to eager V3."""
    model.eval().cpu()
    batch = batch.to("cpu")
    with torch.inference_mode():
        expected = model(batch)
    output = Path(monolithic_path)
    encoder = ort.InferenceSession(
        str(output.with_suffix(".state-encoder.onnx")), providers=list(providers)
    )
    scorer = ort.InferenceSession(
        str(output.with_suffix(".candidate-scorer.onnx")), providers=list(providers)
    )
    encoder_inputs = {
        item.name: getattr(batch, item.name).detach().cpu().numpy()
        for item in encoder.get_inputs()
    }
    encoder_values = encoder.run(None, encoder_inputs)
    cached = dict(zip((item.name for item in encoder.get_outputs()), encoder_values, strict=True))
    scorer_inputs: dict[str, np.ndarray] = {}
    for item in scorer.get_inputs():
        if item.name in cached:
            scorer_inputs[item.name] = cached[item.name]
        else:
            scorer_inputs[item.name] = getattr(batch, item.name).detach().cpu().numpy()
    (actual_policy,) = scorer.run(["policy_logits"], scorer_inputs)
    actual_value = cached["value"]
    expected_policy = expected.policy_logits.detach().numpy()
    expected_value = expected.value.detach().numpy()
    np.testing.assert_allclose(actual_policy, expected_policy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_value, expected_value, rtol=rtol, atol=atol)
    return {
        "policy_max_abs_error": float(np.max(np.abs(actual_policy - expected_policy))),
        "value_max_abs_error": float(np.max(np.abs(actual_value - expected_value))),
    }


def load_model_checkpoint(path: str | Path) -> tuple[AugmentNetV3, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    identity = checkpoint["run_identity"]
    config = V3ModelConfig(**identity["model_config"])
    model = AugmentNetV3(config)
    state, migration = migrate_piece_kind_embedding(
        checkpoint["model"], config.piece_kind_vocabulary_size
    )
    model.load_state_dict(state)
    checkpoint["_load_migrations"] = [] if migration is None else [migration]
    return model, checkpoint


def migrate_piece_kind_embedding(
    state: dict[str, torch.Tensor], target_rows: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    """Expand legacy piece-kind rows without changing any learned row."""
    name = "piece_kind_embedding.weight"
    old = state[name]
    if old.ndim != 2:
        raise ValueError("piece_kind_embedding.weight must be a matrix")
    if old.shape[0] == target_rows:
        return state, None
    if old.shape[0] > target_rows:
        raise ValueError(
            f"checkpoint piece-kind embedding has {old.shape[0]} rows, "
            f"but the model declares only {target_rows}"
        )
    expanded = old.new_zeros((target_rows, old.shape[1]))
    expanded[: old.shape[0]].copy_(old)
    migrated = dict(state)
    migrated[name] = expanded
    return migrated, {
        "kind": "expand_zero_initialized_piece_kind_embedding",
        "source_rows": old.shape[0],
        "target_rows": target_rows,
    }


def extend_model_for_runtime_vocabulary(
    model: AugmentNetV3, vocabulary: V3Vocabulary
) -> tuple[AugmentNetV3, V3Vocabulary, tuple[str, ...]]:
    runtime_contract = load_runtime_vocabulary()
    runtime_vocabulary = V3Vocabulary.from_golden_suite(
        load_golden_suite(), extra_tokens=runtime_contract["tokens"]
    )
    missing = tuple(
        token for token in runtime_vocabulary.tokens if token not in vocabulary.tokens
    )
    if not missing:
        return model, vocabulary, ()
    extended_vocabulary = V3Vocabulary((*vocabulary.tokens, *missing))
    extended_config = replace(
        model.config, text_vocabulary_size=extended_vocabulary.size
    )
    extended_model = AugmentNetV3(extended_config)
    state = model.state_dict()
    extended_state = extended_model.state_dict()
    old_embedding = state["text_embedding.weight"]
    extended_embedding = torch.zeros_like(extended_state["text_embedding.weight"])
    extended_embedding[: old_embedding.shape[0]].copy_(old_embedding)
    extended_state.update(state)
    extended_state["text_embedding.weight"] = extended_embedding
    extended_model.load_state_dict(extended_state)
    return extended_model, extended_vocabulary, missing


def _dynamic_axes(sample: V3Batch) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {}
    axis_names = {
        "state_text": "state_text",
        "piece": "pieces",
        "card": "cards",
        "phase_text": "phase_text",
        "runtime": "runtime_nodes",
        "action": "legal_actions",
        "target": "action_targets",
    }
    for name in INPUT_NAMES:
        tensor = getattr(sample, name)
        axes = {0: "batch"}
        for prefix, axis_name in axis_names.items():
            if name.startswith(prefix) and tensor.ndim >= 2:
                axes[1] = axis_name
        if name == "action_target_sequence" or name.startswith(
            "action_target_sequence_"
        ):
            axes[2] = "target_sequence"
        result[name] = axes
    result["policy_logits"] = {0: "batch", 1: "legal_actions"}
    result["value"] = {0: "batch"}
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ruleset_identity(dataset: dict[str, Any]) -> tuple[str, str]:
    source = dataset
    while source.get("identity_kind") == "pretensorized_v3":
        nested = source.get("source")
        if not isinstance(nested, dict):
            raise ValueError("pre-tensor dataset identity has no source provenance")
        source = nested
    ruleset_id = source.get("ruleset_id")
    ruleset_hash = source.get("ruleset_manifest_sha256")
    if not isinstance(ruleset_id, str) or not isinstance(ruleset_hash, str):
        raise ValueError("dataset identity has no ruleset provenance")
    return ruleset_id, ruleset_hash


def _fsync(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f".json.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _fsync(temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a V3 checkpoint to ONNX")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model, checkpoint = load_model_checkpoint(args.checkpoint)
    identity = checkpoint["run_identity"]
    vocabulary = V3Vocabulary.from_json(identity["vocabulary"])
    model, vocabulary, appended_tokens = extend_model_for_runtime_vocabulary(
        model, vocabulary
    )
    runtime_contract = load_runtime_vocabulary()
    replay = load_dataset_from_identity(identity["dataset"])
    dataset = replay.identity
    sample_indices = list(range(min(2, len(replay))))
    if hasattr(replay, "collate_indices"):
        sample = replay.collate_indices(sample_indices).inputs
    else:
        sample = collate_examples(
            [replay[index] for index in sample_indices], vocabulary
        ).inputs
    output, metadata = export_onnx_v3(
        model,
        sample,
        args.output,
        vocabulary,
        dataset,
        args.checkpoint,
        checkpoint.get("code_commit", "unknown"),
        vocabulary_extension={
            "kind": "append_zero_initialized_runtime_tokens",
            "base_size": len(identity["vocabulary"]["tokens"]),
            "appended_tokens": list(appended_tokens),
            "contract_schema_id": runtime_contract["schema_id"],
            "contract_ruleset_id": runtime_contract["ruleset_id"],
            "contract_sha256": _sha256(default_runtime_vocabulary_path()),
        },
        checkpoint_migrations=checkpoint.get("_load_migrations", ()),
    )
    print(
        json.dumps(
            {
                "model": str(output),
                "metadata": str(metadata),
                **assert_onnx_equivalent(model, sample, output),
            }
        )
    )


if __name__ == "__main__":
    main()

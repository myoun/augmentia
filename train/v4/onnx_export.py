"""V4 factorized ONNX bundle export and numerical equivalence checks.

The bundle splits the network the way search consumes it:

* ``state_encoder`` reuses the authoritative V3 state-encoder graph, including
  the value head.
* ``factorized_scorer`` turns cached encoder outputs plus typed factor edges
  into conditional prefix logits ``[rows, children]``.

Complete-action priors are path products of these conditionals, computed by the
consumer; the graphs never see a flat complete-action head.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from v3.factorized import FACTOR_POLICY_SCHEMA_ID
from v3.onnx_export import (
    STATE_ENCODER_OUTPUT_NAMES,
    STATE_INPUT_NAMES,
    SCORER_LOOKUP_INPUT_NAMES,
    V3StateEncoderOnnxWrapper,
    _atomic_json,
    _export_graph,
    _onnx_graph_contract,
    _ruleset_identity,
    _sha256,
    _state_encoder_dynamic_axes,
)
from v3.schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID
from v3.tensorize import V3Batch, V3Vocabulary
from v3.schema import load_runtime_vocabulary

from .batch import FactorEdgeBatch, PrefixBatchV4, V4Batch
from .model import AugmentNetV4, V3StateEncoding

V4_BUNDLE_SCHEMA_ID = "augmentia.onnx-factorized-policy-bundle.v1"
V4_SCORER_CACHED_INPUT_NAMES = STATE_ENCODER_OUTPUT_NAMES[:-1]
V4_FACTOR_EDGE_INPUT_NAMES = ("field", "category", "index", "ordinal", "mask")
V4_PREFIX_INPUT_NAMES = tuple(f"prefix_{name}" for name in V4_FACTOR_EDGE_INPUT_NAMES)
V4_CHILD_INPUT_NAMES = tuple(f"child_{name}" for name in V4_FACTOR_EDGE_INPUT_NAMES)
V4_FACTORIZED_SCORER_INPUT_NAMES = (
    *V4_SCORER_CACHED_INPUT_NAMES,
    *SCORER_LOOKUP_INPUT_NAMES,
    *V4_PREFIX_INPUT_NAMES,
    *V4_CHILD_INPUT_NAMES,
    "legal_mask",
    "state_index",
)
V4_FACTORIZED_SCORER_OUTPUT_NAMES = ("prefix_logits",)


class V4FactorizedScorerOnnxWrapper(nn.Module):
    """Score conditional trie edges from a cached V3 state encoding."""

    def __init__(self, model: AugmentNetV4) -> None:
        super().__init__()
        self.model = model

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        cached = inputs[: len(V4_SCORER_CACHED_INPUT_NAMES)]
        lookup = inputs[
            len(V4_SCORER_CACHED_INPUT_NAMES) : len(V4_SCORER_CACHED_INPUT_NAMES)
            + len(SCORER_LOOKUP_INPUT_NAMES)
        ]
        edge_count = len(V4_FACTOR_EDGE_INPUT_NAMES)
        lookup_end = len(V4_SCORER_CACHED_INPUT_NAMES) + len(SCORER_LOOKUP_INPUT_NAMES)
        prefix = FactorEdgeBatch(
            **dict(zip(V4_FACTOR_EDGE_INPUT_NAMES, inputs[lookup_end : lookup_end + edge_count], strict=True))
        )
        children = FactorEdgeBatch(
            **dict(
                zip(
                    V4_FACTOR_EDGE_INPUT_NAMES,
                    inputs[lookup_end + edge_count : lookup_end + 2 * edge_count],
                    strict=True,
                )
            )
        )
        legal_mask = inputs[lookup_end + 2 * edge_count]
        state_index = inputs[lookup_end + 2 * edge_count + 1]
        # ``score_prefixes`` reads only the tensors below. Visit statistics stay in
        # the training batch and never enter the inference graph.
        prefixes = PrefixBatchV4(
            state_index=state_index,
            prefix=prefix,
            children=children,
            legal_mask=legal_mask,
            opened_mask=torch.ones_like(legal_mask),
            visit_counts=torch.zeros_like(legal_mask, dtype=torch.long),
            prefix_mass=torch.ones(state_index.shape[0], dtype=torch.float32),
            complete_action_visits=torch.zeros((1, 1), dtype=torch.long),
            complete_action_mask=torch.ones((1, 1), dtype=torch.bool),
            complete_action_opened_mask=torch.ones((1, 1), dtype=torch.bool),
        )
        states = SimpleNamespace(**dict(zip(SCORER_LOOKUP_INPUT_NAMES, lookup, strict=True)))
        encoding = V3StateEncoding(
            state=cached[0],
            board_tokens=cached[1],
            piece_entities=cached[2],
            card_entities=cached[3],
        )
        return self.model.score_prefixes(prefixes, states, encoding)


def export_factorized_policy_bundle_v4(
    model: AugmentNetV4,
    sample: V3Batch,
    factorized_sample: V4Batch,
    output_path: str | Path,
    vocabulary: V3Vocabulary,
    dataset: Any,
    checkpoint_path: str | Path | None = None,
    code_commit: str = "unknown",
    opset: int = 18,
    runtime_vocabulary_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Export the split V4 bundle and its fail-closed metadata sidecar."""
    dataset_identity = dataset.to_json()
    ruleset_id, ruleset_manifest_sha256 = _ruleset_identity(dataset_identity)
    runtime_vocabulary = load_runtime_vocabulary(
        runtime_vocabulary_path,
        expected_ruleset_id=ruleset_id,
    )
    missing = set(runtime_vocabulary["tokens"]) - set(vocabulary.tokens)
    if missing:
        raise ValueError(f"V4 vocabulary is missing required runtime tokens: {', '.join(sorted(missing))}")
    if model.config.text_vocabulary_size != vocabulary.size:
        raise ValueError("V4 model text vocabulary size differs from the export vocabulary")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    sample = sample.to("cpu")
    factorized_sample = V4Batch(
        factorized_sample.states.to("cpu"), factorized_sample.prefixes.to("cpu")
    )
    encoder_path = output.with_suffix(".state-encoder.onnx")
    scorer_path = output.with_suffix(".factorized-scorer.onnx")
    _export_graph(
        V3StateEncoderOnnxWrapper(model.state_encoder),
        tuple(getattr(sample, name) for name in STATE_INPUT_NAMES),
        encoder_path,
        STATE_INPUT_NAMES,
        STATE_ENCODER_OUTPUT_NAMES,
        _state_encoder_dynamic_axes(sample),
        opset,
    )
    _export_graph(
        V4FactorizedScorerOnnxWrapper(model),
        _scorer_trace_inputs(model, sample, factorized_sample),
        scorer_path,
        V4_FACTORIZED_SCORER_INPUT_NAMES,
        V4_FACTORIZED_SCORER_OUTPUT_NAMES,
        _factorized_scorer_dynamic_axes(factorized_sample),
        opset,
    )
    metadata = {
        "metadata_schema_id": V4_BUNDLE_SCHEMA_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "factorized_policy_schema_id": FACTOR_POLICY_SCHEMA_ID,
        "ruleset_id": ruleset_id,
        "ruleset_manifest_sha256": ruleset_manifest_sha256,
        "dataset": dataset_identity,
        "vocabulary": vocabulary.to_json(),
        "model_config": asdict(model.config),
        "piece_kind_vocabulary_size": model.config.piece_kind_vocabulary_size,
        "checkpoint_sha256": _sha256(Path(checkpoint_path)) if checkpoint_path else None,
        "code_commit": code_commit,
        "opset": opset,
        "state_encoder": _onnx_graph_contract(encoder_path),
        "factorized_scorer": _onnx_graph_contract(scorer_path),
    }
    metadata_path = output.with_suffix(".v4-bundle.metadata.json")
    _atomic_json(metadata_path, metadata)
    return output, metadata_path


def assert_factorized_onnx_equivalent(
    model: AugmentNetV4,
    states: V3Batch,
    factorized: V4Batch,
    bundle_metadata_path: str | Path,
) -> dict[str, float]:
    """Compare ONNX bundle outputs against eager V4 scoring."""
    metadata = json.loads(Path(bundle_metadata_path).read_text(encoding="utf-8"))
    if metadata.get("metadata_schema_id") != V4_BUNDLE_SCHEMA_ID:
        raise ValueError("metadata is not a V4 factorized policy bundle")
    base = Path(bundle_metadata_path).resolve().parent
    encoder_path = base / metadata["state_encoder"]["path"]
    scorer_path = base / metadata["factorized_scorer"]["path"]
    for entry in ("state_encoder", "factorized_scorer"):
        if _sha256(base / metadata[entry]["path"]) != metadata[entry]["sha256"]:
            raise ValueError(f"{entry} graph hash does not match the bundle metadata")

    model = model.eval().cpu()
    states = states.to("cpu")
    factorized = V4Batch(factorized.states.to("cpu"), factorized.prefixes.to("cpu"))
    prefixes = factorized.prefixes
    encoder_session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    scorer_session = ort.InferenceSession(str(scorer_path), providers=["CPUExecutionProvider"])
    state_inputs = {
        name: getattr(states, name).numpy()
        for name in STATE_INPUT_NAMES
    }
    encoded = dict(
        zip(
            STATE_ENCODER_OUTPUT_NAMES,
            encoder_session.run(
                [name for name in STATE_ENCODER_OUTPUT_NAMES],
                state_inputs,
            ),
            strict=True,
        )
    )
    with torch.no_grad():
        encoding = model.encode_state(states)
        eager_logits = model.score_prefixes(prefixes, states, encoding).numpy()
    scorer_inputs = {
        **{name: encoded[name] for name in V4_SCORER_CACHED_INPUT_NAMES},
        **{name: getattr(states, name).numpy() for name in SCORER_LOOKUP_INPUT_NAMES},
        **{
            f"prefix_{name}": getattr(prefixes.prefix, name).numpy()
            for name in V4_FACTOR_EDGE_INPUT_NAMES
        },
        **{
            f"child_{name}": getattr(prefixes.children, name).numpy()
            for name in V4_FACTOR_EDGE_INPUT_NAMES
        },
        "legal_mask": prefixes.legal_mask.numpy(),
        "state_index": prefixes.state_index.numpy(),
    }
    (onnx_logits,) = scorer_session.run(
        list(V4_FACTORIZED_SCORER_OUTPUT_NAMES),
        scorer_inputs,
    )
    value_error = float(np.abs(encoded["value"] - model.state_encoder.value_head(encoding.state).squeeze(-1).detach().numpy()).max())
    legal = prefixes.legal_mask.numpy()
    masked_error = float(np.abs(onnx_logits[legal] - eager_logits[legal]).max())
    return {
        "value_max_abs_error": value_error,
        "prefix_max_abs_error": masked_error,
        "illegal_is_min": bool(
            np.all(onnx_logits[~legal] <= np.finfo(np.float32).min / 2)
        ),
    }


def _scorer_trace_inputs(
    model: AugmentNetV4, sample: V3Batch, factorized: V4Batch
) -> tuple[torch.Tensor, ...]:
    """Materialize the exact scorer input tuple for tracing or inference."""
    prefixes = factorized.prefixes
    with torch.no_grad():
        encoding = model.encode_state(sample)
    return (
        encoding.state,
        encoding.board_tokens,
        encoding.piece_entities,
        encoding.card_entities,
        *(getattr(sample, name) for name in SCORER_LOOKUP_INPUT_NAMES),
        *(getattr(prefixes.prefix, name) for name in V4_FACTOR_EDGE_INPUT_NAMES),
        *(getattr(prefixes.children, name) for name in V4_FACTOR_EDGE_INPUT_NAMES),
        prefixes.legal_mask,
        prefixes.state_index,
    )


def _factorized_scorer_dynamic_axes(factorized: V4Batch) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {
        "state_embedding": {0: "batch"},
        "board_tokens": {0: "batch"},
        "piece_entities": {0: "batch", 1: "pieces"},
        "card_entities": {0: "batch", 1: "cards"},
        "piece_slot": {0: "batch", 1: "pieces"},
        "piece_mask": {0: "batch", 1: "pieces"},
        "card_slot": {0: "batch", 1: "cards"},
        "card_mask": {0: "batch", 1: "cards"},
        "state_index": {0: "rows"},
        "legal_mask": {0: "rows", 1: "children"},
        "prefix_logits": {0: "rows", 1: "children"},
    }
    for name in V4_PREFIX_INPUT_NAMES:
        axes[name] = {0: "rows", 1: "prefix_length"}
    for name in V4_CHILD_INPUT_NAMES:
        axes[name] = {0: "rows", 1: "children"}
    return axes

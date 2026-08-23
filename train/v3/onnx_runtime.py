"""Reusable, batched V3 ONNX inference runtime with strict artifact checks."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort

from .onnx_export import INPUT_NAMES, MODEL_METADATA_SCHEMA_ID
from .schema import (
    ACTION_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    PIECE_KIND_VOCABULARY_SIZE,
    SchemaError,
)
from .tensorize import V3Batch, V3Vocabulary, tensorize_cases


@dataclass(frozen=True)
class EvaluationV3:
    probabilities: tuple[float, ...]
    value: float


class OnnxRuntimeV3:
    """One long-lived session; callers may send heterogeneous state batches."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path | None = None,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        expected_ruleset_id: str | None = None,
        expected_ruleset_manifest_sha256: str | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        metadata_file = (
            Path(metadata_path).resolve()
            if metadata_path is not None
            else self.model_path.with_suffix(".metadata.json")
        )
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        _validate_metadata(
            metadata,
            self.model_path,
            expected_ruleset_id,
            expected_ruleset_manifest_sha256,
        )
        self.metadata = metadata
        self.vocabulary = V3Vocabulary.from_json(metadata["vocabulary"])
        try:
            self._session = ort.InferenceSession(
                str(self.model_path), providers=list(providers)
            )
        except Exception as error:
            raise RuntimeError(f"failed to load V3 ONNX model: {error}") from error
        declared_inputs = tuple(item["name"] for item in metadata["inputs"])
        actual_inputs = tuple(item.name for item in self._session.get_inputs())
        if actual_inputs != declared_inputs:
            raise SchemaError(
                f"ONNX input contract mismatch: expected {declared_inputs}, got {actual_inputs}"
            )
        if not set(actual_inputs).issubset(INPUT_NAMES):
            raise SchemaError("ONNX declares an unknown V3 tensor input")
        self.input_names = actual_inputs
        if tuple(item.name for item in self._session.get_outputs()) != (
            "policy_logits",
            "value",
        ):
            raise SchemaError("ONNX output contract mismatch")
        self._lock = threading.Lock()
        self.session_runs = 0

    def evaluate_cases(self, cases: Sequence[Mapping[str, Any]]) -> list[EvaluationV3]:
        return self.evaluate_tensor_batch(tensorize_cases(cases, self.vocabulary))

    def evaluate_tensor_batch(self, batch: V3Batch) -> list[EvaluationV3]:
        inputs = {
            name: np.ascontiguousarray(getattr(batch, name).detach().cpu().numpy())
            for name in self.input_names
        }
        with self._lock:
            policy_logits, values = self._session.run(
                ["policy_logits", "value"], inputs
            )
            self.session_runs += 1
        if not np.isfinite(policy_logits).all() or not np.isfinite(values).all():
            raise FloatingPointError("V3 ONNX output contains NaN or infinity")
        results = []
        action_mask = batch.action_mask.detach().cpu().numpy()
        for row in range(batch.batch_size):
            count = int(action_mask[row].sum())
            if count == 0:
                raise ValueError("cannot evaluate a state with no legal actions")
            logits = policy_logits[row, :count].astype(np.float64)
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            value = float(values[row])
            if not math.isfinite(value) or not -1.0001 <= value <= 1.0001:
                raise FloatingPointError(f"V3 value output is invalid: {value}")
            results.append(
                EvaluationV3(tuple(float(item) for item in probabilities), value)
            )
        return results


def benchmark_runtime(
    runtime: OnnxRuntimeV3,
    cases: Sequence[Mapping[str, Any]],
    batch_sizes: Sequence[int] = (1, 8, 32),
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, Any]:
    report: dict[str, Any] = {"session_instances": 1, "batches": {}}
    for batch_size in batch_sizes:
        batch_cases = [cases[index % len(cases)] for index in range(batch_size)]
        batch = tensorize_cases(batch_cases, runtime.vocabulary)
        for _ in range(warmup):
            runtime.evaluate_tensor_batch(batch)
        inference_samples = []
        end_to_end_samples = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            runtime.evaluate_tensor_batch(batch)
            inference_samples.append((time.perf_counter_ns() - started) / 1_000_000)
            started = time.perf_counter_ns()
            runtime.evaluate_cases(batch_cases)
            end_to_end_samples.append((time.perf_counter_ns() - started) / 1_000_000)
        inference = np.asarray(inference_samples)
        end_to_end = np.asarray(end_to_end_samples)
        report["batches"][str(batch_size)] = {
            "inference_p50_ms": float(np.percentile(inference, 50)),
            "inference_p95_ms": float(np.percentile(inference, 95)),
            "end_to_end_p50_ms": float(np.percentile(end_to_end, 50)),
            "end_to_end_p95_ms": float(np.percentile(end_to_end, 95)),
            "positions_per_second": float(batch_size / (inference.mean() / 1000)),
        }
    report["session_runs"] = runtime.session_runs
    return report


def _validate_metadata(
    metadata: Any,
    model_path: Path,
    expected_ruleset_id: str | None,
    expected_ruleset_hash: str | None,
) -> None:
    if (
        not isinstance(metadata, dict)
        or metadata.get("metadata_schema_id") != MODEL_METADATA_SCHEMA_ID
    ):
        raise SchemaError("unsupported V3 model metadata")
    if metadata.get("observation_schema_id") != OBSERVATION_SCHEMA_ID:
        raise SchemaError("model observation schema mismatch")
    if metadata.get("action_schema_id") != ACTION_SCHEMA_ID:
        raise SchemaError("model action schema mismatch")
    if metadata.get("piece_kind_vocabulary_size") != PIECE_KIND_VOCABULARY_SIZE:
        raise SchemaError("model piece-kind vocabulary mismatch")
    if _sha256(model_path) != metadata.get("onnx_sha256"):
        raise SchemaError("ONNX checksum does not match model metadata")
    if (
        expected_ruleset_id is not None
        and metadata.get("ruleset_id") != expected_ruleset_id
    ):
        raise SchemaError("model ruleset ID mismatch")
    if (
        expected_ruleset_hash is not None
        and metadata.get("ruleset_manifest_sha256") != expected_ruleset_hash
    ):
        raise SchemaError("model ruleset manifest mismatch")
    declared_inputs = tuple(item.get("name") for item in metadata.get("inputs", []))
    if not declared_inputs or not set(declared_inputs).issubset(INPUT_NAMES):
        raise SchemaError("model metadata input contract mismatch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

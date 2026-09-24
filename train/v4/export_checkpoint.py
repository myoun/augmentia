"""Export a trained V4 checkpoint and emit a cross-runtime parity sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from v3.factorized import factorized_action_path
from v3.tensorize import V3Vocabulary

from .batch import FACTOR_FIELD_IDS, tensorize_factorized_cases
from .dataset import load_dataset_input
from .model import AugmentNetV4, V4ModelConfig
from .onnx_export import (
    assert_factorized_onnx_equivalent,
    export_factorized_policy_bundle_v4,
)
from .replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    GAME_RECORD_V5_SCHEMA_ID,
    load_game_v4,
)
from .train import (
    RULESET_ID,
    SOURCE_SEAL_COMMIT,
    SOURCE_SEAL_TAG,
    V4_TRAINER_SCHEMA_ID,
    assert_production_policy_target_semantics,
    default_runtime_vocabulary_path,
    load_runtime_vocabulary,
)

PARITY_SCHEMA_ID = "augmentia.v4-trained-bundle-parity.v1"


@dataclass(frozen=True)
class DatasetIdentity:
    ruleset_id: str
    ruleset_manifest_sha256: str
    dataset_manifest_sha256: str
    dataset_run_id: str
    source_seal_tag: str
    source_seal_commit: str

    def to_json(self) -> dict[str, Any]:
        return {
            "ruleset_id": self.ruleset_id,
            "ruleset_manifest_sha256": self.ruleset_manifest_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_run_id": self.dataset_run_id,
            "source_seal_tag": self.source_seal_tag,
            "source_seal_commit": self.source_seal_commit,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--runtime-vocabulary",
        type=Path,
        default=default_runtime_vocabulary_path(),
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    dataset_path = args.manifest.resolve()
    output = args.output.resolve()
    sidecar_path = checkpoint.with_suffix(".v4-checkpoint.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("schema_id") != V4_TRAINER_SCHEMA_ID:
        raise ValueError("checkpoint sidecar is not a production V4 trainer checkpoint")
    assert_production_policy_target_semantics(sidecar)
    if (
        sidecar.get("factorized_target_schema_id")
        != FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID
        or sidecar.get("replay_schema_id") != GAME_RECORD_V5_SCHEMA_ID
    ):
        raise ValueError("checkpoint sidecar replay/target lineage mismatch")
    checkpoint_sha256 = _sha256_file(checkpoint)
    if sidecar.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs from sidecar")

    runtime_vocabulary, runtime_identity = load_runtime_vocabulary(args.runtime_vocabulary)
    checkpoint_vocabulary = V3Vocabulary.from_json(sidecar["vocabulary"])
    if checkpoint_vocabulary != runtime_vocabulary:
        raise ValueError("checkpoint vocabulary is not the exact 0915 runtime vocabulary")

    model_config = V4ModelConfig(**sidecar["model_config"])
    if model_config.text_vocabulary_size != runtime_vocabulary.size:
        raise ValueError("checkpoint model config vocabulary size mismatch")
    model = AugmentNetV4(model_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset_input = load_dataset_input(dataset_path)
    ruleset = dataset_input.ruleset
    if ruleset.get("id") != RULESET_ID:
        raise ValueError("dataset ruleset is not 0915")
    if (
        dataset_input.source_seal_tag != SOURCE_SEAL_TAG
        or dataset_input.source_seal_commit != SOURCE_SEAL_COMMIT
    ):
        raise ValueError("dataset source seal differs from training base")

    provenance = sidecar.get("provenance", {})
    if provenance.get("ruleset_id") != ruleset["id"] or provenance.get(
        "ruleset_manifest_sha256"
    ) != ruleset["manifest_sha256"]:
        raise ValueError("checkpoint ruleset provenance differs from dataset")
    manifest_sha256 = dataset_input.sha256
    if provenance.get("dataset_manifest_sha256") != manifest_sha256:
        raise ValueError("checkpoint dataset provenance differs from export input")

    if not dataset_input.games:
        raise ValueError("dataset contains no games")
    sample_game = dataset_input.games[0]
    examples = load_game_v4(sample_game.path)
    if not examples:
        raise ValueError("sample game contains no training examples")
    example = examples[0]
    case = {
        "observation": example.observation,
        "legal_actions": example.legal_actions,
    }
    factorized = tensorize_factorized_cases(
        [case],
        checkpoint_vocabulary,
        [list(example.complete_visit_counts)],
        opened_action_masks=[list(example.opened_action_mask)],
    )

    dataset = DatasetIdentity(
        ruleset_id=ruleset["id"],
        ruleset_manifest_sha256=ruleset["manifest_sha256"],
        dataset_manifest_sha256=manifest_sha256,
        dataset_run_id=dataset_input.dataset_id,
        source_seal_tag=dataset_input.source_seal_tag or "",
        source_seal_commit=dataset_input.source_seal_commit or "",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _, metadata_path = export_factorized_policy_bundle_v4(
        model,
        factorized.states,
        factorized,
        output,
        checkpoint_vocabulary,
        dataset,
        checkpoint_path=checkpoint,
        code_commit=provenance.get("trainer_git_commit", _git_commit()),
        runtime_vocabulary_path=args.runtime_vocabulary,
    )
    python_onnx = assert_factorized_onnx_equivalent(
        model,
        factorized.states,
        factorized,
        metadata_path,
    )

    with torch.no_grad():
        encoding = model.encode_state(factorized.states)
        prefix_logits = model.score_prefixes(
            factorized.prefixes, factorized.states, encoding
        )
        value = float(
            model.state_encoder.value_head(encoding.state).squeeze(-1)[0]
        )
    action_logits = _complete_action_log_probabilities(
        case["legal_actions"]["actions"],
        factorized.prefixes,
        prefix_logits,
    )
    probability_sum = sum(math_exp(value_) for value_ in action_logits)

    metadata_sha256 = _sha256_file(metadata_path)
    parity_path = output.with_suffix(".parity.json")
    parity = {
        "schema_id": PARITY_SCHEMA_ID,
        "ruleset_id": ruleset["id"],
        "ruleset_manifest_sha256": ruleset["manifest_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "bundle_metadata_path": metadata_path.name,
        "bundle_metadata_sha256": metadata_sha256,
        "sample_game_id": sample_game.game_id,
        "sample_decision_index": 0,
        "observation": case["observation"],
        "legal_actions": case["legal_actions"],
        "expected_value": value,
        "expected_action_logits": action_logits,
        "complete_action_probability_sum": probability_sum,
        "python_onnx": python_onnx,
        "runtime_vocabulary_sha256": runtime_identity["sha256"],
    }
    parity_path.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")

    report_path = output.with_suffix(".export-report.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sidecar_sha256": _sha256_file(sidecar_path),
        "dataset_manifest_sha256": manifest_sha256,
        "policy_target_semantics_id": sidecar["policy_target_semantics_id"],
        "policy_target_semantics_lineage": sidecar[
            "policy_target_semantics_lineage"
        ],
        "policy_target_semantics_production_eligible": True,
        "runtime_vocabulary_sha256": runtime_identity["sha256"],
        "bundle_metadata": str(metadata_path),
        "bundle_metadata_sha256": metadata_sha256,
        "state_encoder_sha256": metadata["state_encoder"]["sha256"],
        "factorized_scorer_sha256": metadata["factorized_scorer"]["sha256"],
        "parity": str(parity_path),
        "parity_sha256": _sha256_file(parity_path),
        "python_onnx": python_onnx,
        "complete_action_probability_sum": probability_sum,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _complete_action_log_probabilities(
    actions: list[Mapping[str, Any]],
    prefixes: Any,
    logits: torch.Tensor,
) -> list[float]:
    row_index = {
        _row_prefix(prefixes, row): row for row in range(prefixes.row_count)
    }
    result: list[float] = []
    for action in actions:
        path = tuple(
            f"{field}:{value}" for field, value in factorized_action_path(action)
        )
        log_probability = 0.0
        for depth, token in enumerate(path):
            row = row_index[path[:depth]]
            children = _row_children(prefixes, row)
            column = children.index(token)
            legal_logits = logits[row][prefixes.legal_mask[row]]
            log_probability += float(torch.log_softmax(legal_logits, dim=-1)[column])
        result.append(log_probability)
    return result


def _row_prefix(prefixes: Any, row: int) -> tuple[str, ...]:
    return tuple(_edge_tokens(prefixes.prefix, row))


def _row_children(prefixes: Any, row: int) -> list[str]:
    return _edge_tokens(prefixes.children, row)


def _edge_tokens(edges: Any, row: int) -> list[str]:
    field_names = {value: key for key, value in FACTOR_FIELD_IDS.items()}
    tokens: list[str] = []
    for column in range(edges.field.shape[1]):
        if not bool(edges.mask[row, column]):
            continue
        field = field_names[int(edges.field[row, column])]
        if field == "target_sequence":
            ordinal = int(edges.ordinal[row, column])
            field = f"target_{ordinal - 1}"
            value = str(int(edges.index[row, column]))
        else:
            category = int(edges.category[row, column])
            index = int(edges.index[row, column])
            if field in ("family", "ability", "target_kind"):
                value = _category_value(field, category)
            elif index < 0:
                value = "none" if field == "promotion" else "action"
            else:
                value = str(index)
        tokens.append(f"{field}:{value}")
    return tokens


def _category_value(field: str, category: int) -> str:
    from v3.tensorize import ABILITY_IDS, ACTION_KIND_IDS, TARGET_KIND_IDS

    tables = {
        "family": ACTION_KIND_IDS,
        "ability": ABILITY_IDS,
        "target_kind": TARGET_KIND_IDS,
    }
    for name, identifier in tables[field].items():
        if identifier == category:
            return name
    raise KeyError((field, category))


def math_exp(value: float) -> float:
    import math

    return math.exp(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    main()

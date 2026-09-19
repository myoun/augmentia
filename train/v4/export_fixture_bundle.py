"""Regenerate the cross-language V4 ONNX bundle fixture.

Writes ``train/v3/fixtures/v4_bundle/`` with:

* the split state-encoder and factorized-scorer ONNX graphs,
* the V4 bundle metadata sidecar produced by ``export_factorized_policy_bundle_v4``,
* a parity JSON with serialized inputs and eager expected outputs.

The Rust runtime tests consume the same files, so the fixture pins the
Python/Rust inference contract. Run from ``train/``:

    uv run python -m v4.export_fixture_bundle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from v3.factorized import factorized_action_path
from v3.tensorize import V3Vocabulary

from v4.batch import tensorize_factorized_cases
from v4.model import AugmentNetV4, V4ModelConfig
from v4.onnx_export import export_factorized_policy_bundle_v4

FIXTURE_DIR = Path(__file__).parents[1] / "v3" / "fixtures"
SOURCE_FIXTURE = FIXTURE_DIR / "observation_action_v3.json"
RUNTIME_VOCABULARY = FIXTURE_DIR / "runtime_vocabulary_v3.json"
BUNDLE_DIR = FIXTURE_DIR / "v4_bundle"
PARITY_SCHEMA_ID = "augmentia.onnx-factorized-bundle-parity.v1"


class _Identity:
    def __init__(self, ruleset_id: str, ruleset_manifest_sha256: str) -> None:
        self.ruleset_id = ruleset_id
        self.ruleset_manifest_sha256 = ruleset_manifest_sha256

    def to_json(self) -> dict[str, Any]:
        return {
            "ruleset_id": self.ruleset_id,
            "ruleset_manifest_sha256": self.ruleset_manifest_sha256,
        }


def wire_path(action: dict[str, Any]) -> tuple[str, ...]:
    """Serialize a canonical factor path into the Rust-compatible wire form."""
    return tuple(f"{field}:{value}" for field, value in factorized_action_path(action))


def _prefix_rows(actions: list[dict[str, Any]], visits: list[int]) -> list[dict[str, Any]]:
    """Reproduce the collate traversal order for the parity record."""
    root: dict[str, Any] = {"children": {}, "count": 0}
    for action, count in zip(actions, visits, strict=True):
        path = wire_path(action)
        node = root
        node["count"] += count
        for token in path:
            node = node["children"].setdefault(token, {"children": {}, "count": 0})
            node["count"] += count
    rows: list[dict[str, Any]] = []

    def descend(prefix: tuple[str, ...], node: dict[str, Any]) -> int:
        if not node["children"]:
            return node["count"]
        children = sorted(node["children"])
        counts = [descend((*prefix, token), node["children"][token]) for token in children]
        rows.append(
            {
                "prefix": list(prefix),
                "children": children,
                "visit_counts": counts,
            }
        )
        return sum(counts)

    descend((), root)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fixture", type=Path, default=SOURCE_FIXTURE)
    parser.add_argument("--runtime-vocabulary", type=Path, default=RUNTIME_VOCABULARY)
    parser.add_argument("--bundle-dir", type=Path, default=BUNDLE_DIR)
    parser.add_argument("--ruleset-manifest-sha256", default="0" * 64)
    args = parser.parse_args()
    if len(args.ruleset_manifest_sha256) != 64:
        raise ValueError("--ruleset-manifest-sha256 must be a 64-character SHA-256")

    torch.manual_seed(11)
    suite = json.loads(args.source_fixture.read_text(encoding="utf-8"))
    runtime_vocabulary = json.loads(args.runtime_vocabulary.read_text(encoding="utf-8"))
    ruleset_id = suite["ruleset_id"]
    if runtime_vocabulary.get("ruleset_id") != ruleset_id:
        raise ValueError("fixture and runtime vocabulary ruleset ids must match")
    cases = suite["cases"][:2]
    catalog_path = Path(__file__).parents[2] / "rulesets" / ruleset_id / "cards" / "catalog.json"
    vocabulary = V3Vocabulary.from_cases(
        cases,
        catalog_path=catalog_path,
        extra_tokens=runtime_vocabulary["tokens"],
    )
    visits = [
        [index + 1 for index in range(len(case["legal_actions"]["actions"]))]
        for case in cases
    ]
    factorized = tensorize_factorized_cases(cases, vocabulary, visits)
    states = factorized.states
    model = AugmentNetV4(
        V4ModelConfig(
            text_vocabulary_size=vocabulary.size,
            channels=8,
            residual_blocks=1,
            model_dim=16,
            runtime_message_passes=1,
        )
    ).eval()
    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    output = args.bundle_dir / "v4-model.onnx"
    _, metadata_path = export_factorized_policy_bundle_v4(
        model,
        states,
        factorized,
        output,
        vocabulary,
        _Identity(ruleset_id, args.ruleset_manifest_sha256),
    )
    with torch.no_grad():
        encoding = model.encode_state(states)
        logits = model.score_prefixes(factorized.prefixes, states, encoding)
        value = model.state_encoder.value_head(encoding.state).squeeze(-1)
    rows = [
        _prefix_rows(case["legal_actions"]["actions"], visit)
        for case, visit in zip(cases, visits, strict=True)
    ]
    prefixes = factorized.prefixes
    row_index: dict[tuple[int, tuple[str, ...]], int] = {}
    for row in range(prefixes.row_count):
        row_index[(int(prefixes.state_index[row]), _row_prefix(prefixes, row))] = row
    for decision, decision_rows in enumerate(rows):
        for entry in decision_rows:
            row = row_index[(decision, tuple(entry["prefix"]))]
            children = _row_children(prefixes, row)
            legal = int(prefixes.legal_mask[row].sum())
            assert children == entry["children"]
            assert len(children) == legal
            entry["logits"] = [
                round(float(logits[row, column]), 6) for column in range(legal)
            ]
    action_logits: list[list[float]] = []
    for decision, case in enumerate(cases):
        decision_logits: list[float] = []
        for action in case["legal_actions"]["actions"]:
            path = list(wire_path(action))
            log_probability = 0.0
            for depth in range(len(path)):
                row = row_index[(decision, tuple(path[:depth]))]
                column = _row_children(prefixes, row).index(path[depth])
                row_logits = logits[row][prefixes.legal_mask[row]]
                log_probability += float(torch.log_softmax(row_logits, dim=-1)[column])
            decision_logits.append(round(log_probability, 6))
        action_logits.append(decision_logits)

    parity = {
        "schema_id": PARITY_SCHEMA_ID,
        "cases": [
            {
                "name": case["name"],
                "observation": case["observation"],
                "legal_actions": case["legal_actions"],
                "visits": visit,
            }
            for case, visit in zip(cases, visits, strict=True)
        ],
        "rows": rows,
        "expected_value": [round(float(item), 6) for item in value],
        "expected_action_logits": action_logits,
    }
    parity_path = args.bundle_dir / "v4-model.parity.json"
    parity_path.write_text(json.dumps(parity, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {metadata_path}")
    print(f"wrote {parity_path}")


def _row_prefix(prefixes, row: int) -> tuple[str, ...]:
    from v4.batch import FACTOR_FIELD_IDS

    tokens: list[str] = []
    field_names = {value: key for key, value in FACTOR_FIELD_IDS.items()}
    edges = prefixes.prefix
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
    return tuple(tokens)


def _row_children(prefixes, row: int) -> list[str]:
    from v4.batch import FACTOR_FIELD_IDS

    tokens: list[str] = []
    field_names = {value: key for key, value in FACTOR_FIELD_IDS.items()}
    edges = prefixes.children
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


if __name__ == "__main__":
    main()

"""Build a hash-sealed V4 multi-shard dataset index from validated run manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .dataset import (
    ACTION_SCHEMA_ID,
    DATASET_INDEX_SCHEMA_ID,
    FACTORIZED_TARGET_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    REPLAY_SCHEMA_ID,
    _load_run_manifest,
    sha256_file,
)


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    game_indices: list[int] = []
    seen_run_ids: set[str] = set()
    lineage: dict[str, str] | None = None
    for value in args.manifests:
        manifest_path = value.resolve()
        manifest = _load_run_manifest(manifest_path)
        current_lineage = {
            "replay_schema_id": manifest["replay_schema_id"],
            "factorized_target_schema_id": manifest["factorized_target_schema_id"],
            "observation_schema_id": manifest["observation_schema_id"],
            "action_schema_id": manifest["action_schema_id"],
        }
        if lineage is None:
            lineage = current_lineage
        elif current_lineage != lineage:
            raise ValueError("source manifests disagree on replay/target lineage")
        if manifest["run_id"] in seen_run_ids:
            raise ValueError(f"duplicate source run_id: {manifest['run_id']}")
        seen_run_ids.add(manifest["run_id"])
        validation_path = manifest_path.parent / "validation.json"
        if not validation_path.exists():
            raise ValueError(f"validated source required; missing {validation_path}")
        entries.append(
            {
                "path": _relative(manifest_path, output.parent),
                "sha256": sha256_file(manifest_path),
                "validation_path": _relative(validation_path, output.parent),
                "validation_sha256": sha256_file(validation_path),
            }
        )
        game_indices.extend(
            int(item["game_index"]) for item in manifest["games_artifacts"]
        )
    if not game_indices:
        raise ValueError("dataset contains no games")
    ordered_indices = sorted(game_indices)
    expected_indices = list(range(ordered_indices[0], ordered_indices[-1] + 1))
    if ordered_indices != expected_indices:
        raise ValueError("source manifests do not form a contiguous game_index range")
    if lineage is None:
        raise ValueError("dataset contains no manifests")
    payload = {
        "dataset_schema_id": DATASET_INDEX_SCHEMA_ID,
        "dataset_id": args.dataset_id,
        "games": len(ordered_indices),
        "game_index_start": ordered_indices[0],
        "game_index_end": ordered_indices[-1],
        "replay_schema_id": REPLAY_SCHEMA_ID,
        "factorized_target_schema_id": FACTORIZED_TARGET_SCHEMA_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "manifests": entries,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_index": str(output),
                "dataset_id": args.dataset_id,
                "manifests": len(entries),
                "sha256": sha256_file(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

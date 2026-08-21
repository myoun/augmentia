"""Bounded retention for reproducible V3 iteration artifact directories."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ITERATION_MARKER_SCHEMA_ID = "augmentia.selfplay-iteration.v3"


@dataclass(frozen=True)
class RetentionResultV3:
    removed: tuple[str, ...]
    retained: tuple[str, ...]
    bytes_after: int


def apply_iteration_retention(
    root: str | Path,
    *,
    keep_iterations: int,
    max_bytes: int,
    protected: frozenset[str] = frozenset(),
) -> RetentionResultV3:
    """Remove complete unpinned iteration units oldest-first within ``root`` only."""
    root_path = Path(root).resolve()
    if keep_iterations <= 0 or max_bytes <= 0:
        raise ValueError("retention limits must be positive")
    root_path.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, str, Path, int]] = []
    for path in root_path.iterdir():
        if not path.is_dir():
            continue
        marker = path / "iteration-manifest.json"
        if not marker.is_file():
            continue
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("iteration_schema_id") != ITERATION_MARKER_SCHEMA_ID:
            continue
        iteration_id = payload.get("iteration_id")
        if not isinstance(iteration_id, str) or iteration_id != path.name:
            continue
        created_at = payload.get("created_at_utc")
        if not isinstance(created_at, str):
            continue
        candidates.append((created_at, iteration_id, path, _directory_size(path)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    total = sum(item[3] for item in candidates)
    retained = list(candidates)
    removed: list[str] = []
    while len(retained) > keep_iterations or total > max_bytes:
        removable = next(
            (
                item
                for item in retained
                if item[1] not in protected and not (item[2] / "PINNED").exists()
            ),
            None,
        )
        if removable is None:
            raise RuntimeError("retention limits cannot be met without deleting protected runs")
        _, iteration_id, path, size = removable
        _require_direct_child(root_path, path)
        shutil.rmtree(path)
        retained.remove(removable)
        total -= size
        removed.append(iteration_id)
    return RetentionResultV3(
        tuple(removed), tuple(item[1] for item in retained), total
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _require_direct_child(root: Path, path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != root or resolved == root:
        raise RuntimeError("retention target escaped the configured iteration root")

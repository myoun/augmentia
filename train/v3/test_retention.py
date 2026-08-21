from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v3.retention import ITERATION_MARKER_SCHEMA_ID, apply_iteration_retention


class RetentionV3Tests(unittest.TestCase):
    def test_oldest_complete_unpinned_iteration_is_removed_as_one_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for iteration_id in ("001", "002", "003"):
                path = root / iteration_id
                path.mkdir()
                (path / "iteration-manifest.json").write_text(
                    json.dumps(
                        {
                            "iteration_schema_id": ITERATION_MARKER_SCHEMA_ID,
                            "iteration_id": iteration_id,
                            "created_at_utc": f"2026-08-21T00:00:0{iteration_id[-1]}Z",
                        }
                    ),
                    encoding="utf-8",
                )
                (path / "artifact.bin").write_bytes(b"x" * 32)
            (root / "001" / "PINNED").touch()
            result = apply_iteration_retention(
                root,
                keep_iterations=2,
                max_bytes=1_000_000,
                protected=frozenset({"003"}),
            )
            self.assertEqual(result.removed, ("002",))
            self.assertTrue((root / "001").exists())
            self.assertTrue((root / "003").exists())

    def test_unmarked_directories_are_never_retention_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unmarked = root / "user-data"
            unmarked.mkdir()
            (unmarked / "file.txt").write_text("keep", encoding="utf-8")
            result = apply_iteration_retention(
                root, keep_iterations=1, max_bytes=1024
            )
            self.assertEqual(result.removed, ())
            self.assertTrue((unmarked / "file.txt").exists())

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v3.factorized import FACTOR_POLICY_SCHEMA_ID
from v3.schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID
from v4.replay import FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID, GAME_RECORD_V5_SCHEMA_ID

from v4.dataset import DATASET_INDEX_SCHEMA_ID, load_dataset_input, sha256_file
from v4.train import _load_examples


class DatasetInputV4Test(unittest.TestCase):
    def _write_run(
        self, root: Path, name: str, game_index: int
    ) -> tuple[Path, Path]:
        run = root / name
        (run / "games").mkdir(parents=True)
        game_id = f"{name}-game-{game_index:08d}"
        game_path = run / "games" / f"{game_id}.json"
        game_path.write_text("{}\n", encoding="utf-8")
        manifest = {
            "manifest_schema_id": "augmentia.self-play-v4-run.v1",
            "run_id": name,
            "run_kind": "production",
            "source_seal_tag": "seal",
            "source_seal_commit": "commit",
            "engine_commit": "engine",
            "ruleset": {
                "id": "augment_chess_20260915",
                "manifest_sha256": "ruleset",
            },
            "evaluator": {"kind": "onnx", "agent_id": "model", "model_sha256": "model"},
            "factorized_policy_schema_id": FACTOR_POLICY_SCHEMA_ID,
            "replay_schema_id": GAME_RECORD_V5_SCHEMA_ID,
            "factorized_target_schema_id": FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "games": 1,
            "games_artifacts": [
                {
                    "game_id": game_id,
                    "game_index": game_index,
                    "relative_path": f"games/{game_path.name}",
                }
            ],
        }
        manifest_path = run / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        validation = {
            "validation_schema_id": "augmentia.self-play-v4-validation.v1",
            "run_id": name,
            "games": 1,
            "terminal_games": 1,
            "truncated_games": 0,
        }
        validation_path = run / "validation.json"
        validation_path.write_text(json.dumps(validation), encoding="utf-8")
        return manifest_path, validation_path

    def _write_index(
        self, root: Path, sources: list[tuple[Path, Path]]
    ) -> Path:
        index = root / "dataset.json"
        indices = [
            int(
                json.loads(manifest.read_text(encoding="utf-8"))["games_artifacts"][0][
                    "game_index"
                ]
            )
            for manifest, _validation in sources
        ]
        payload = {
            "dataset_schema_id": DATASET_INDEX_SCHEMA_ID,
            "dataset_id": "combined",
            "games": len(indices),
            "game_index_start": min(indices),
            "game_index_end": max(indices),
            "replay_schema_id": GAME_RECORD_V5_SCHEMA_ID,
            "factorized_target_schema_id": FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "manifests": [
                {
                    "path": str(manifest.relative_to(root)),
                    "sha256": sha256_file(manifest),
                    "validation_path": str(validation.relative_to(root)),
                    "validation_sha256": sha256_file(validation),
                }
                for manifest, validation in sources
            ],
        }
        index.write_text(json.dumps(payload), encoding="utf-8")
        return index

    def test_single_manifest_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _validation = self._write_run(root, "single", 0)
            dataset = load_dataset_input(manifest)
            self.assertEqual(dataset.dataset_id, "single")
            self.assertEqual([game.game_index for game in dataset.games], [0])

    def test_legacy_target_lineage_is_rejected_by_default_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, _validation = self._write_run(root, "legacy-lineage", 0)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["factorized_target_schema_id"] = "augmentia.factorized-mcts-visits.v2"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lineage"):
                load_dataset_input(manifest)

    def test_multi_shard_index_loads_games_in_global_index_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            second = self._write_run(root, "s02", 65)
            first = self._write_run(root, "s01", 64)
            dataset = load_dataset_input(self._write_index(root, [second, first]))
            self.assertEqual(dataset.dataset_id, "combined")
            self.assertEqual([game.game_index for game in dataset.games], [64, 65])
            self.assertEqual(len(dataset.source_manifests), 2)

    def test_index_rejects_manifest_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._write_run(root, "s01", 64)
            index = self._write_index(root, [source])
            source[0].write_text(source[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                load_dataset_input(index)

    def test_index_rejects_duplicate_game_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self._write_run(root, "s01", 64)
            second = self._write_run(root, "s02", 64)
            index = self._write_index(root, [first, second])
            with self.assertRaisesRegex(ValueError, "duplicate game_index"):
                load_dataset_input(index)

    def test_index_rejects_game_index_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self._write_run(root, "s01", 64)
            second = self._write_run(root, "s02", 66)
            index = self._write_index(root, [first, second])
            with self.assertRaisesRegex(ValueError, "game_index gap"):
                load_dataset_input(index)

    def test_capped_example_loading_stops_early(self) -> None:
        paths = [Path("g1.json"), Path("g2.json"), Path("g3.json")]
        with patch("v4.train.load_game_v4", side_effect=[[1, 2], [3, 4], [5]]) as load:
            result = _load_examples(paths, 3)
        self.assertEqual(result, [1, 2, 3])
        self.assertEqual(load.call_count, 2)


if __name__ == "__main__":
    unittest.main()

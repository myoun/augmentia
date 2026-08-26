from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import zstandard

from .selfplay_parallel import (
    ParallelSelfPlayConfig,
    _commit_root_manifest,
    _validate_part_manifest,
    _validate_part_manifest_payload,
    _worker_command,
)


class ParallelSelfPlayTests(unittest.TestCase):
    def test_worker_uses_absolute_game_index_with_common_seed(self) -> None:
        config = _config(Path("root"))
        command = _worker_command(config, 17, Path("part"))
        self.assertEqual(_value(command, "--games"), "1")
        self.assertEqual(_value(command, "--game-index-start"), "17")
        self.assertEqual(_value(command, "--base-seed"), "1234")
        self.assertEqual(_value(command, "--execution-provider"), "cpu")

    def test_root_manifest_prefixes_part_shards_and_sums_totals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = []
            for game_index, split in enumerate(("train", "validation")):
                part = root / "parts" / f"game-{game_index:08}"
                part.mkdir(parents=True)
                manifest = part / "manifest.json"
                manifest.write_text(
                    json.dumps(_manifest(split, game_index + 2)), encoding="utf-8"
                )
                manifests.append(manifest)
            config = _config(root)
            target = root / "manifest.json"
            with patch("v3.selfplay_parallel._validate_dataset"):
                result = _commit_root_manifest(manifests, target, config)
            self.assertEqual(result, target)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["totals"]["games"], 2)
            self.assertEqual(payload["totals"]["game_steps"], 5)
            self.assertEqual(
                [item["path"] for item in payload["shards"]],
                [
                    "parts/game-00000000/shards/train.games.jsonl.zst",
                    "parts/game-00000001/shards/validation.games.jsonl.zst",
                ],
            )

    def test_part_contract_rejects_wrong_split_salt(self) -> None:
        config = _config(Path("root"))
        payload = _manifest("train", 1)
        payload["split_policy"]["salt"] = "wrong"
        with self.assertRaisesRegex(ValueError, "split policy"):
            _validate_part_manifest_payload(payload, Path("manifest.json"), config)

    def test_part_identity_locks_game_seed_and_engine_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            part = Path(directory)
            config = _config(part)
            manifest = _manifest("train", 1)
            manifest["shards"][0]["path"] = "shards/train.games.jsonl.zst"
            manifest_path = part / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            shard = part / "shards" / "train.games.jsonl.zst"
            shard.parent.mkdir()
            record = {
                "game_id": "run-game-00000000",
                "seed_hex": "00000000000004d2",
                "generator": {
                    "run_id": "run",
                    "engine_commit": "c" * 40,
                    "dirty_worktree": False,
                },
            }
            shard.write_bytes(
                zstandard.ZstdCompressor().compress(
                    (json.dumps(record) + "\n").encode()
                )
            )
            _validate_part_manifest(manifest_path, config, 0, "c" * 40)
            record["game_id"] = "run-game-00000001"
            shard.write_bytes(
                zstandard.ZstdCompressor().compress(
                    (json.dumps(record) + "\n").encode()
                )
            )
            with self.assertRaisesRegex(ValueError, "game index"):
                _validate_part_manifest(manifest_path, config, 0, "c" * 40)


def _config(root: Path) -> ParallelSelfPlayConfig:
    return ParallelSelfPlayConfig(
        repository=root,
        binary=root / "self_play_v3.exe",
        validator=root / "validate_v3_dataset.exe",
        output_dir=root,
        model_a=root / "a.onnx",
        metadata_a=root / "a.json",
        model_b=root / "b.onnx",
        metadata_b=root / "b.json",
        ruleset_hash="a" * 64,
        run_id="run",
        created_at_utc="2026-08-26T00:00:00Z",
        games=2,
        base_seed=1234,
    )


def _manifest(split: str, steps: int) -> dict[str, object]:
    shard = {
        "path": f"shards/{split}.games.jsonl.zst",
        "split": split,
        "record_schema_id": "augmentia.replay.game.v3",
        "sha256": "b" * 64,
        "compressed_bytes": 10,
        "records": 1,
        "games": 1,
        "decisions": 0,
        "game_steps": steps,
        "policy_eligible_decisions": steps,
        "value_eligible_decisions": steps,
    }
    return {
        "manifest_schema_id": "augmentia.dataset.manifest.v3",
        "dataset_id": "selfplay-v3:run",
        "created_at_utc": "2026-08-26T00:00:00Z",
        "observation_schema_id": "augmentia.observation.v3",
        "action_schema_id": "augmentia.action.v3",
        "split_policy": {
            "algorithm": "sha256_split_group_v1",
            "salt": "selfplay-v3:run",
            "train_basis_points": 9000,
            "validation_basis_points": 500,
            "holdout_basis_points": 500,
        },
        "compression": {"codec": "zstd", "level": 3},
        "shards": [shard],
        "totals": {
            "shards": 1,
            "records": 1,
            "games": 1,
            "decisions": 0,
            "game_steps": steps,
            "policy_eligible_decisions": steps,
            "value_eligible_decisions": steps,
        },
    }


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


if __name__ == "__main__":
    unittest.main()

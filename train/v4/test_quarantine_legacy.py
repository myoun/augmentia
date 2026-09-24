from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v4 import quarantine_legacy as quarantine


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LegacyQuarantineTests(unittest.TestCase):
    def _fixture(self, temp: Path) -> tuple[Path, Path, Path, Path]:
        root = temp / "artifact-root"
        corpus = root / "legacy-corpus"
        game_path = corpus / "games" / "game-00000000.json"
        game_path.parent.mkdir(parents=True)
        game_path.write_text('{"record_schema_id":"augmentia.replay.game.v4"}\n')
        manifest_path = corpus / "manifest.json"
        write_json(
            manifest_path,
            {
                "manifest_schema_id": quarantine.RUN_MANIFEST_SCHEMA_ID,
                "run_id": "legacy-corpus",
                "run_kind": "production",
                "ruleset": {"id": quarantine.RULESET_ID},
                "engine_commit": "1" * 40,
                "source_seal_tag": "legacy-seal",
                "source_seal_commit": "2" * 40,
                "evaluator": {
                    "kind": "bootstrap",
                    "agent_id": "legacy-bootstrap",
                    "bootstrap_seed": 7,
                    "bootstrap_value": 0.0,
                },
                "base_seed": 11,
                "seed_derivation": "base+index",
                "simulations": 16,
                "chance_outcome_samples": 4,
                "temperature": 1.0,
                "temperature_decisions": 2,
                "max_decisions": 2,
                "workers": 1,
                "execution_provider": "cpu",
                "intra_threads": 1,
                "search_semantics_id": "legacy-search",
                "root_selection_id": "legacy-root",
                "chance_sampling_id": "legacy-chance",
                "replay_schema_id": quarantine.LEGACY_GAME_RECORD_V4_SCHEMA_ID,
                "factorized_target_schema_id": (
                    quarantine.LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID
                ),
                "games_artifacts": [
                    {
                        "game_index": 0,
                        "relative_path": "games/game-00000000.json",
                        "file_sha256": sha256(game_path),
                    }
                ],
            },
        )
        validation_path = corpus / "validation.json"
        write_json(
            validation_path,
            {
                "validation_schema_id": quarantine.VALIDATION_SCHEMA_ID,
                "run_id": "legacy-corpus",
                "games": 1,
                "decisions": 2,
                "terminal_games": 0,
                "truncated_games": 1,
            },
        )

        dataset_dir = root / "legacy-dataset"
        dataset_path = dataset_dir / "dataset.json"
        write_json(
            dataset_path,
            {
                "dataset_schema_id": quarantine.LEGACY_DATASET_INDEX_SCHEMA_ID,
                "dataset_id": "legacy-dataset",
                "games": 1,
                "game_index_start": 0,
                "game_index_end": 0,
                "manifests": [
                    {
                        "path": "../legacy-corpus/manifest.json",
                        "sha256": sha256(manifest_path),
                    }
                ],
            },
        )

        train_dir = root / "legacy-train"
        checkpoint_path = train_dir / "final.pt"
        optimizer_path = train_dir / "final.optimizer.pt"
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.write_bytes(b"legacy-checkpoint")
        optimizer_path.write_bytes(b"legacy-optimizer")
        checkpoint_hash = sha256(checkpoint_path)
        optimizer_hash = sha256(optimizer_path)
        sidecar_path = train_dir / "final.v4-checkpoint.json"
        provenance = {
            "dataset_run_id": "legacy-dataset",
            "dataset_manifest_sha256": sha256(dataset_path),
            "dataset_input_schema_id": quarantine.LEGACY_DATASET_INDEX_SCHEMA_ID,
            "dataset_source_manifests": [
                {"run_id": "legacy-corpus", "sha256": sha256(manifest_path)}
            ],
            "ruleset_id": quarantine.RULESET_ID,
            "encoder_kind": "baseline",
            "training_config": {"epochs": 1},
        }
        write_json(
            sidecar_path,
            {
                "schema_id": "augmentia.v4-trainer-checkpoint.v2",
                "checkpoint_sha256": checkpoint_hash,
                "optimizer_path": optimizer_path.name,
                "optimizer_sha256": optimizer_hash,
                "completed_epochs": 1,
                "example_count": 1,
                "validation_example_count": 0,
                "provenance": provenance,
            },
        )
        write_json(train_dir / "run.json", {"provenance": provenance})
        write_json(train_dir / "metrics.json", {"schema_id": "legacy-metrics"})
        write_json(train_dir / "history.json", {"epochs": [1]})

        bundle = root / "legacy-bundle"
        state_encoder = bundle / "v4-model.state-encoder.onnx"
        scorer = bundle / "v4-model.factorized-scorer.onnx"
        bundle.mkdir(parents=True)
        state_encoder.write_bytes(b"state-encoder")
        scorer.write_bytes(b"factorized-scorer")
        metadata_path = bundle / "v4-model.v4-bundle.metadata.json"
        write_json(
            metadata_path,
            {
                "metadata_schema_id": quarantine.BUNDLE_METADATA_SCHEMA_ID,
                "checkpoint_sha256": checkpoint_hash,
                "code_commit": "3" * 40,
                "dataset": {
                    "dataset_run_id": "legacy-dataset",
                    "dataset_manifest_sha256": sha256(dataset_path),
                },
                "ruleset_id": quarantine.RULESET_ID,
                "model_config": {"encoder_kind": "baseline"},
                "state_encoder": {
                    "path": state_encoder.name,
                    "sha256": sha256(state_encoder),
                },
                "factorized_scorer": {
                    "path": scorer.name,
                    "sha256": sha256(scorer),
                },
            },
        )
        return root, checkpoint_path, manifest_path, sidecar_path

    def test_non_zero_start_shard_is_contiguous_and_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, _checkpoint, manifest_path, sidecar_path = self._fixture(base)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["game_index_start"] = 64
            manifest["games_artifacts"][0]["game_index"] = 64
            write_json(manifest_path, manifest)
            manifest_hash = sha256(manifest_path)

            dataset_path = root / "legacy-dataset" / "dataset.json"
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            dataset["manifests"][0]["sha256"] = manifest_hash
            write_json(dataset_path, dataset)
            dataset_hash = sha256(dataset_path)

            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["provenance"]["dataset_manifest_sha256"] = dataset_hash
            sidecar["provenance"]["dataset_source_manifests"][0]["sha256"] = manifest_hash
            write_json(sidecar_path, sidecar)

            run_path = sidecar_path.parent / "run.json"
            run_payload = json.loads(run_path.read_text(encoding="utf-8"))
            run_payload["provenance"]["dataset_manifest_sha256"] = dataset_hash
            run_payload["provenance"]["dataset_source_manifests"][0]["sha256"] = manifest_hash
            write_json(run_path, run_payload)

            bundle_path = next(root.rglob("*.v4-bundle.metadata.json"))
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["dataset"]["dataset_manifest_sha256"] = dataset_hash
            write_json(bundle_path, bundle)

            registry = quarantine.build_registry(root)
            corpus = registry["corpora"][0]
            self.assertEqual(corpus["games"], 1)
            self.assertEqual(registry["summary"]["corpus_games"], 1)

    def test_non_destructive_idempotent_registry_and_optional_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, _checkpoint, _manifest, _sidecar = self._fixture(base)
            original = {
                path.relative_to(root).as_posix(): sha256(path)
                for path in root.rglob("*")
                if path.is_file()
            }
            registry = base / "central" / "legacy-registry.json"
            first = quarantine.quarantine(
                root,
                registry,
                write_root_marker=True,
                hash_games=True,
                hash_binaries=True,
            )
            self.assertTrue(first["registry_created"])
            self.assertTrue(first["root_marker_created"])
            after = {
                path.relative_to(root).as_posix(): sha256(path)
                for path in root.rglob("*")
                if path.is_file() and path.name != quarantine.ROOT_MARKER_NAME
            }
            self.assertEqual(after, original)
            marker = json.loads((root / quarantine.ROOT_MARKER_NAME).read_text())
            self.assertEqual(marker["registry_sha256"], sha256(registry))
            registry_mtime = registry.stat().st_mtime_ns
            marker_mtime = (root / quarantine.ROOT_MARKER_NAME).stat().st_mtime_ns

            second = quarantine.quarantine(
                root,
                registry,
                write_root_marker=True,
                hash_games=True,
                hash_binaries=True,
            )
            self.assertFalse(second["registry_created"])
            self.assertFalse(second["root_marker_created"])
            self.assertEqual(registry.stat().st_mtime_ns, registry_mtime)
            self.assertEqual(
                (root / quarantine.ROOT_MARKER_NAME).stat().st_mtime_ns,
                marker_mtime,
            )
            report = quarantine.verify_registry(
                registry,
                hash_games=True,
                hash_binaries=True,
            )
            self.assertEqual(report["corpora"], 1)
            self.assertEqual(report["checkpoints"], 1)
            self.assertEqual(report["bundles"], 1)
            self.assertGreater(report["checked_files"], 0)

    def test_existing_different_registry_or_marker_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, _checkpoint, _manifest, _sidecar = self._fixture(base)
            registry = base / "central" / "legacy-registry.json"
            write_json(registry, {"different": True})
            before = registry.read_bytes()
            with self.assertRaisesRegex(
                quarantine.LegacyQuarantineError, "refusing to overwrite"
            ):
                quarantine.quarantine(root, registry)
            self.assertEqual(registry.read_bytes(), before)

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, _checkpoint, _manifest, _sidecar = self._fixture(base)
            marker = root / quarantine.ROOT_MARKER_NAME
            write_json(marker, {"different": True})
            before = marker.read_bytes()
            registry = base / "central" / "legacy-registry.json"
            with self.assertRaisesRegex(
                quarantine.LegacyQuarantineError, "refusing to overwrite"
            ):
                quarantine.quarantine(root, registry, write_root_marker=True)
            self.assertEqual(marker.read_bytes(), before)
            self.assertFalse(registry.exists())

    def test_legacy_sidecar_schema_is_authoritative_for_missing_dataset_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, _checkpoint, _manifest, sidecar_path = self._fixture(base)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["provenance"]["dataset_run_id"] = "archived-external-smoke"
            sidecar["provenance"]["dataset_source_manifests"] = []
            write_json(sidecar_path, sidecar)

            registry = quarantine.build_registry(root)
            self.assertEqual(registry["summary"]["training_runs"], 1)
            self.assertEqual(registry["summary"]["checkpoints"], 1)
            self.assertEqual(
                registry["training_runs"][0]["legacy_reason"],
                "trainer-checkpoint-schema-v1-or-v2-missing-v3-lineage",
            )

    def test_unknown_lineage_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, _checkpoint, manifest, sidecar = self._fixture(base)
            payload = json.loads(manifest.read_text())
            payload["factorized_target_schema_id"] = "augmentia.factorized-mcts-visits.v9"
            write_json(manifest, payload)
            with self.assertRaisesRegex(
                quarantine.LegacyQuarantineError, "unknown or mixed"
            ):
                quarantine.build_registry(root)

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, _checkpoint, _manifest, sidecar = self._fixture(base)
            payload = json.loads(sidecar.read_text())
            payload["schema_id"] = "augmentia.v4-trainer-checkpoint.v9"
            write_json(sidecar, payload)
            with self.assertRaisesRegex(
                quarantine.LegacyQuarantineError, "unknown checkpoint schema"
            ):
                quarantine.build_registry(root)

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, _checkpoint, _manifest, _sidecar = self._fixture(base)
            metadata_path = next(root.rglob("*.v4-bundle.metadata.json"))
            payload = json.loads(metadata_path.read_text())
            payload["dataset"]["dataset_run_id"] = "unknown-dataset"
            payload["checkpoint_sha256"] = "f" * 64
            write_json(metadata_path, payload)
            with self.assertRaisesRegex(
                quarantine.LegacyQuarantineError,
                "unknown dataset/checkpoint lineage",
            ):
                quarantine.build_registry(root)

    def test_hash_drift_and_registry_location_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, checkpoint, _manifest, _sidecar = self._fixture(base)
            registry = base / "registry.json"
            quarantine.quarantine(
                root,
                registry,
                hash_games=True,
                hash_binaries=True,
            )
            checkpoint.write_bytes(b"changed-checkpoint")
            with self.assertRaisesRegex(
                quarantine.LegacyQuarantineError, "hash changed|size changed"
            ):
                quarantine.verify_registry(registry, hash_binaries=True)

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, _checkpoint, _manifest, _sidecar = self._fixture(base)
            with self.assertRaisesRegex(
                quarantine.LegacyQuarantineError, "outside the artifact root"
            ):
                quarantine.quarantine(root, root / "registry.json")
            with self.assertRaisesRegex(
                quarantine.LegacyQuarantineError, "unsafe registry path"
            ):
                quarantine._safe_path(root, "../outside.pt", "registry")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from v3.schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID

from v4.audit_phase1_corpus import (
    CorpusAuditError,
    audit_manifest_pair,
    audit_phase1_corpus,
    seal_report,
    verify_report_seal,
    write_audit_report,
)
from v4.replay import (
    FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
    GAME_RECORD_V5_SCHEMA_ID,
    LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
    LEGACY_GAME_RECORD_V4_SCHEMA_ID,
    make_factorized_target_v4,
    make_legacy_factorized_target_v2,
)


FIXTURE = Path(__file__).parents[1] / "v3" / "fixtures" / "0915" / "observation_action_v3.json"
RULESET = {
    "id": "augment_chess_20260915",
    "manifest_sha256": "7b2736f8e9f727db818d8c8f4aaf7742a23a717e2f31d749ea293a3dfc051b13",
}
EVALUATOR = {
    "kind": "bootstrap",
    "agent_id": "bootstrap-deterministic-v4:test",
    "model_sha256": None,
    "checkpoint_sha256": None,
    "bundle_metadata_path": None,
    "bundle_metadata_sha256": None,
    "state_encoder_sha256": None,
    "factorized_scorer_sha256": None,
    "bundle_code_commit": None,
    "bootstrap_seed": 17,
    "bootstrap_value": 0.0,
}


class Phase1CorpusAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.case = next(case for case in payload["cases"] if case["name"] == "runtime_rich")
        cls.actions = cls.case["legal_actions"]["actions"]
        cls.visits = [1] * 15 + [0] * (len(cls.actions) - 15)
        cls.opened = [True] * 16 + [False] * (len(cls.actions) - 16)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _make_game(
        self,
        root: Path,
        *,
        run_id: str,
        replay_schema_id: str,
        target_schema_id: str,
        target: dict[str, Any],
    ) -> tuple[Path, str, str]:
        game_id = f"{run_id}-game-00000000"
        state_fingerprint = "state-00000000"
        next_state_fingerprint = "state-00000001"
        record: dict[str, Any] = {
            "record_schema_id": replay_schema_id,
            "game_id": game_id,
            "ruleset": RULESET,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "factorized_policy_schema_id": "augmentia.factorized-policy.v4.1",
            "steps": [
                {
                    "decision_index": 0,
                    "engine_ply": 0,
                    "actor": "white",
                    "state_fingerprint": state_fingerprint,
                    "observation": self.case["observation"],
                    "legal_actions": self.case["legal_actions"],
                    "policy_target": target,
                    "played_action_index": 0,
                    "next_state_fingerprint": next_state_fingerprint,
                }
            ],
            "mode_id": "standard",
            "seed_hex": "000000000000007b",
            "generator": {
                "run_id": run_id,
                "engine_commit": f"commit-{run_id}",
                "engine_api_version": 1,
                "collector_id": "test",
                "dirty_worktree": False,
            },
            "players": [
                {
                    "seat": seat,
                    "agent_kind": "mcts",
                    "agent_id": EVALUATOR["agent_id"],
                    "model": None,
                    "config_id": "mcts-v4:sims=1:chance=1:root=augmentia.gumbel-sequential-halving.v1:search=augmentia.mcts-v4.factorized-gumbel-widening-public-chance.v2",
                }
                for seat in ("white", "black")
            ],
            "initial_state_fingerprint": state_fingerprint,
            "termination": {
                "kind": "limit",
                "limit": "decision",
                "reason": "max_decisions",
                "final_state_fingerprint": next_state_fingerprint,
            },
            "trajectory_sha256": "",
        }
        if target_schema_id != LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID:
            record["factorized_target_schema_id"] = target_schema_id
        steps = [
            {
                "state_fingerprint": step["state_fingerprint"],
                "action": step["legal_actions"]["actions"][step["played_action_index"]],
                "next_state_fingerprint": step["next_state_fingerprint"],
            }
            for step in record["steps"]
        ]
        record["trajectory_sha256"] = hashlib.sha256(
            json.dumps(steps, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        game_path = root / "games" / f"{game_id}.json"
        self._write_json(game_path, record)
        return game_path, record["trajectory_sha256"], record["seed_hex"]

    def _make_manifest(
        self,
        root: Path,
        *,
        run_id: str,
        replay_schema_id: str,
        target_schema_id: str,
        target: dict[str, Any],
    ) -> Path:
        game_path, trajectory_sha256, seed_hex = self._make_game(
            root,
            run_id=run_id,
            replay_schema_id=replay_schema_id,
            target_schema_id=target_schema_id,
            target=target,
        )
        manifest: dict[str, Any] = {
            "manifest_schema_id": "augmentia.self-play-v4-run.v1",
            "run_id": run_id,
            "run_kind": "smoke",
            "source_seal_tag": "seal",
            "source_seal_commit": "commit",
            "engine_commit": f"commit-{run_id}",
            "ruleset": RULESET,
            "evaluator": EVALUATOR,
            "mode_id": "standard",
            "game_index_start": 0,
            "games": 1,
            "base_seed": 123,
            "seed_derivation": "base_seed + game_index (wrapping u64)",
            "simulations": 1,
            "chance_outcome_samples": 1,
            "temperature": 1.0,
            "temperature_decisions": 1,
            "max_decisions": 1,
            "workers": 1,
            "execution_provider": "cpu",
            "intra_threads": 1,
            "search_semantics_id": "augmentia.mcts-v4.factorized-gumbel-widening-public-chance.v2",
            "root_selection_id": "augmentia.gumbel-sequential-halving.v1",
            "chance_sampling_id": "augmentia.public-information-multisample-chance.v1",
            "replay_schema_id": replay_schema_id,
            "factorized_policy_schema_id": "augmentia.factorized-policy.v4.1",
            "factorized_target_schema_id": target_schema_id,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "games_artifacts": [
                {
                    "game_index": 0,
                    "game_id": f"{run_id}-game-00000000",
                    "seed_hex": seed_hex,
                    "relative_path": f"games/{game_path.name}",
                    "file_sha256": hashlib.sha256(game_path.read_bytes()).hexdigest(),
                    "trajectory_sha256": trajectory_sha256,
                    "decisions": 1,
                    "termination": {
                        "kind": "limit",
                        "limit": "decision",
                        "reason": "max_decisions",
                        "final_state_fingerprint": "state-00000001",
                    },
                }
            ],
        }
        self._write_json(root / "manifest.json", manifest)
        self._write_json(
            root / "validation.json",
            {
                "validation_schema_id": "augmentia.self-play-v4-validation.v1",
                "run_id": run_id,
                "run_kind": "smoke",
                "ruleset_id": RULESET["id"],
                "ruleset_manifest_sha256": RULESET["manifest_sha256"],
                "games": 1,
                "decisions": 1,
                "terminal_games": 0,
                "truncated_games": 1,
                "unique_seeds": 1,
                "checks": ["synthetic"],
            },
        )
        return root / "manifest.json"

    def _corpora(self, root: Path) -> tuple[Path, Path]:
        legacy_target = make_legacy_factorized_target_v2(
            self.actions, self.visits, 0.125
        )
        clean_target = make_factorized_target_v4(
            self.actions,
            self.visits,
            0.125,
            self.opened,
            search_semantics_id="augmentia.mcts-v4.factorized-gumbel-widening-public-chance.v2",
            chance_outcome_samples=1,
        )
        legacy = self._make_manifest(
            root / "legacy",
            run_id="legacy-test",
            replay_schema_id=LEGACY_GAME_RECORD_V4_SCHEMA_ID,
            target_schema_id=LEGACY_FACTORIZED_MCTS_TARGET_V2_SCHEMA_ID,
            target=legacy_target,
        )
        clean = self._make_manifest(
            root / "clean",
            run_id="clean-test",
            replay_schema_id=GAME_RECORD_V5_SCHEMA_ID,
            target_schema_id=FACTORIZED_MCTS_TARGET_V3_SCHEMA_ID,
            target=clean_target,
        )
        return legacy, clean

    def test_pairing_metrics_and_hash_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            legacy, clean = self._corpora(Path(temp))
            report = audit_phase1_corpus(legacy, clean)
            verify_report_seal(report)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["trajectory_pairing"]["games_compared"], 1)
            metrics = report["clean_opened_metrics"]
            self.assertEqual(metrics["legal_actions"], len(self.actions))
            self.assertEqual(metrics["opened_actions"], 16)
            self.assertEqual(metrics["closed_actions"], len(self.actions) - 16)
            self.assertGreater(metrics["mixed_prefix_rows"], 0)
            self.assertGreater(metrics["false_zero_closed_child_mass_total"], 0.0)
            self.assertGreater(metrics["weighted_closed_support_fraction_total"], 0.0)
            row = metrics["mixed_rows"][0]
            self.assertIn("closed_child_count", row)
            self.assertIn("weighted_closed_child_count", row)
            self.assertIn("weighted_closed_support_fraction", row)

    def test_writer_is_idempotent_and_seal_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy, clean = self._corpora(root)
            report = audit_phase1_corpus(legacy, clean)
            output = root / "audit.json"
            write_audit_report(output, report)
            first = output.read_bytes()
            write_audit_report(output, report)
            self.assertEqual(output.read_bytes(), first)
            tampered = copy.deepcopy(report)
            tampered["status"] = "tampered"
            with self.assertRaisesRegex(CorpusAuditError, "report hash mismatch"):
                verify_report_seal(tampered)
            resealed = seal_report(tampered)
            with self.assertRaisesRegex(CorpusAuditError, "refusing overwrite"):
                write_audit_report(output, resealed)

    def test_seed_or_trajectory_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy, clean = self._corpora(root)
            payload = json.loads(clean.read_text(encoding="utf-8"))
            payload["games_artifacts"][0]["seed_hex"] = "000000000000007c"
            clean.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CorpusAuditError, "seed"):
                audit_manifest_pair(legacy, clean)

    def test_lineage_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            legacy, clean = self._corpora(Path(temp))
            with self.assertRaisesRegex(CorpusAuditError, "lineage"):
                audit_manifest_pair(clean, legacy)


if __name__ == "__main__":
    unittest.main()

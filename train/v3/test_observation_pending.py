from __future__ import annotations

import copy
import math
import unittest
from dataclasses import fields

import torch

from v3.schema import (
    OBSERVATION_SCHEMA_ID,
    RUNTIME_FIELD_COUNT,
    SchemaError,
    load_golden_suite,
    validate_golden_suite,
)
from v3.tensorize import PLAYER_IDS, RUNTIME_TYPE_IDS, V3Vocabulary, tensorize_cases


class PublicPendingTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_golden_suite()
        cls.case = next(case for case in cls.suite["cases"] if case["name"] == "public_pending")
        cls.vocabulary = V3Vocabulary.from_golden_suite(cls.suite)

    def test_public_pending_reaches_actual_tree_tensors(self) -> None:
        batch = tensorize_cases([self.case], self.vocabulary)
        for feature in self.case["observation"]["runtime"][169:185]:
            field_id = feature["field_id"]
            selected = torch.nonzero(batch.runtime_field[0] == field_id).flatten().tolist()
            self.assertGreater(len(selected), 1)
            cursor = iter(selected)

            def check(value: dict, parent: int, sibling: int, name: str | None = None) -> None:
                index = next(cursor)
                kind = value["type"]
                self.assertTrue(batch.runtime_mask[0, index])
                self.assertEqual(int(batch.runtime_type[0, index]), RUNTIME_TYPE_IDS[kind])
                self.assertEqual(int(batch.runtime_parent[0, index]), parent)
                self.assertEqual(int(batch.runtime_text[0, index]), self.vocabulary.index(name, "test"))
                # Structural sibling position preserves flag/queue/portal endpoint order.
                self.assertAlmostEqual(float(batch.runtime_numeric[0, index, 3]), math.log1p(sibling), places=6)
                if kind == "player":
                    self.assertEqual(int(batch.runtime_player[0, index]), PLAYER_IDS[value["value"]])
                elif kind == "square":
                    self.assertEqual(int(batch.runtime_square[0, index]), value["value"] + 1)
                elif kind == "piece":
                    self.assertEqual(int(batch.runtime_piece_ref[0, index]), value["value"])
                elif kind == "piece_kind":
                    self.assertEqual(int(batch.runtime_piece_kind[0, index]), value["value"])
                    self.assertEqual(int(batch.runtime_piece_ref[0, index]), -1)
                elif kind == "bool":
                    self.assertEqual(float(batch.runtime_numeric[0, index, 0]), float(value["value"]))
                    self.assertEqual(float(batch.runtime_numeric[0, index, 1]), 1.0)
                elif kind == "signed":
                    number = value["value"]
                    expected = math.copysign(math.log1p(abs(number)), number)
                    self.assertAlmostEqual(float(batch.runtime_numeric[0, index, 0]), expected, places=6)
                    self.assertEqual(float(batch.runtime_numeric[0, index, 1]), 1.0)
                elif kind == "list":
                    for position, child in enumerate(value["value"]):
                        check(child, index, position)
                elif kind == "record":
                    for position, child in enumerate(value["value"]):
                        check(child["value"], index, position, child["name"])

            check(feature["value"], -1, field_id - 1)
            self.assertIsNone(next(cursor, None))

    def test_every_public_semantic_leaf_changes_tensors(self) -> None:
        original = tensorize_cases([self.case], self.vocabulary)
        count = 0

        def leaves(value: dict, path: tuple) -> list[tuple]:
            kind = value["type"]
            if kind == "record":
                return [leaf for i, item in enumerate(value["value"])
                        for leaf in leaves(item["value"], (*path, "value", i, "value"))]
            if kind == "list":
                return [leaf for i, item in enumerate(value["value"])
                        for leaf in leaves(item, (*path, "value", i))]
            return [path]

        for field_index in range(169, 172):
            value = self.case["observation"]["runtime"][field_index]["value"]
            for path in leaves(value, ()):
                changed = copy.deepcopy(self.case)
                leaf = changed["observation"]["runtime"][field_index]["value"]
                for step in path:
                    leaf = leaf[step]
                if leaf["type"] == "player":
                    leaf["value"] = "opponent" if leaf["value"] == "self_player" else "self_player"
                elif leaf["type"] == "square":
                    leaf["value"] = (leaf["value"] + 1) % 64
                elif leaf["type"] == "piece":
                    leaf["value"] = (leaf["value"] + 1) % self.case["observation"]["piece_slot_count"]
                elif leaf["type"] == "signed":
                    leaf["value"] += 1
                else:
                    self.fail(f"unreviewed public leaf {leaf}")
                modified = tensorize_cases([changed], self.vocabulary)
                self.assertTrue(any(
                    not torch.equal(getattr(original, item.name), getattr(modified, item.name))
                    for item in fields(original) if item.name.startswith("runtime_")
                ), (field_index, path))
                self.assertTrue(torch.equal(original.spatial, modified.spatial))
                count += 1
        self.assertEqual(count, 22)  # CTF 8, lobsters 6, portals 8.

    def test_presence_and_queue_order_change_tree_tensors(self) -> None:
        original = tensorize_cases([self.case], self.vocabulary)
        for field_index in range(169, 172):
            changed = copy.deepcopy(self.case)
            changed["observation"]["runtime"][field_index]["value"] = (
                {"type": "none"} if field_index == 169 else {"type": "list", "value": []}
            )
            modified = tensorize_cases([changed], self.vocabulary)
            self.assertLess(int(modified.runtime_mask.sum()), int(original.runtime_mask.sum()))
        for field_index in (170, 171):
            changed = copy.deepcopy(self.case)
            changed["observation"]["runtime"][field_index]["value"]["value"].reverse()
            modified = tensorize_cases([changed], self.vocabulary)
            self.assertFalse(torch.equal(original.runtime_player, modified.runtime_player))

    def test_old_schema_and_missing_fields_fail_closed(self) -> None:
        self.assertEqual(RUNTIME_FIELD_COUNT, 194)
        self.assertEqual(OBSERVATION_SCHEMA_ID, "augmentia.observation.v3.20260917.10")
        for old_schema in (
            "augmentia.observation.v3.20260818", "augmentia.observation.v3.20260917",
            "augmentia.observation.v3.20260917.2", "augmentia.observation.v3.20260917.3",
            "augmentia.observation.v3.20260917.4", "augmentia.observation.v3.20260917.5",
            "augmentia.observation.v3.20260917.6",
        ):
            changed = copy.deepcopy(self.case)
            changed["observation"]["schema_id"] = old_schema
            with self.assertRaisesRegex(SchemaError, "schema"):
                tensorize_cases([changed], self.vocabulary)
            suite = copy.deepcopy(self.suite)
            suite["cases"] = [changed]
            with self.assertRaisesRegex(SchemaError, "schema"):
                validate_golden_suite(suite)
        for old_count in (169, 172, 173, 176, 177, 182):
            changed = copy.deepcopy(self.case)
            changed["observation"]["runtime"] = changed["observation"]["runtime"][:old_count]
            with self.assertRaisesRegex(SchemaError, "expected 194"):
                tensorize_cases([changed], self.vocabulary)
            suite = copy.deepcopy(self.suite)
            suite["cases"] = [changed]
            with self.assertRaisesRegex(SchemaError, "expected 194"):
                validate_golden_suite(suite)

    def test_majesty_both_owner_bools_survive_tensorization(self) -> None:
        values = self.case["observation"]["runtime"][172]["value"]
        self.assertEqual(values, {"type": "list", "value": [
            {"type": "bool", "value": False}, {"type": "bool", "value": True},
        ]})  # Rust fixture is Black to move, White owns Majesty.
        signatures = set()
        for own in (False, True):
            for opponent in (False, True):
                changed = copy.deepcopy(self.case)
                changed["observation"]["runtime"][172]["value"]["value"] = [
                    {"type": "bool", "value": own}, {"type": "bool", "value": opponent},
                ]
                batch = tensorize_cases([changed], self.vocabulary)
                mask = (batch.runtime_field[0] == 173) & (batch.runtime_type[0] == RUNTIME_TYPE_IDS["bool"])
                actual = tuple(batch.runtime_numeric[0, mask, 0].tolist())
                self.assertEqual(actual, (float(own), float(opponent)))
                signatures.add(actual)
        self.assertEqual(len(signatures), 4)

    def test_proficiency_both_owner_bools_survive_tensorization(self) -> None:
        signatures = set()
        for own in (False, True):
            for opponent in (False, True):
                changed = copy.deepcopy(self.case)
                changed["observation"]["runtime"][173]["value"]["value"] = [
                    {"type": "bool", "value": own}, {"type": "bool", "value": opponent},
                ]
                batch = tensorize_cases([changed], self.vocabulary)
                mask = (batch.runtime_field[0] == 174) & (batch.runtime_type[0] == RUNTIME_TYPE_IDS["bool"])
                actual = tuple(batch.runtime_numeric[0, mask, 0].tolist())
                self.assertEqual(actual, (float(own), float(opponent)))
                signatures.add(actual)
        self.assertEqual(len(signatures), 4)

    def test_killer_king_both_owner_bools_survive_tensorization(self) -> None:
        signatures = set()
        for own in (False, True):
            for opponent in (False, True):
                changed = copy.deepcopy(self.case)
                changed["observation"]["runtime"][174]["value"]["value"] = [
                    {"type": "bool", "value": own}, {"type": "bool", "value": opponent},
                ]
                batch = tensorize_cases([changed], self.vocabulary)
                mask = (batch.runtime_field[0] == 175) & (batch.runtime_type[0] == RUNTIME_TYPE_IDS["bool"])
                actual = tuple(batch.runtime_numeric[0, mask, 0].tolist())
                self.assertEqual(actual, (float(own), float(opponent)))
                signatures.add(actual)
        self.assertEqual(len(signatures), 4)

    def test_disassembly_both_owner_bools_survive_tensorization(self) -> None:
        signatures = set()
        for own in (False, True):
            for opponent in (False, True):
                changed = copy.deepcopy(self.case)
                changed["observation"]["runtime"][175]["value"]["value"] = [
                    {"type": "bool", "value": own}, {"type": "bool", "value": opponent},
                ]
                batch = tensorize_cases([changed], self.vocabulary)
                mask = (batch.runtime_field[0] == 176) & (batch.runtime_type[0] == RUNTIME_TYPE_IDS["bool"])
                actual = tuple(batch.runtime_numeric[0, mask, 0].tolist())
                self.assertEqual(actual, (float(own), float(opponent)))
                signatures.add(actual)
        self.assertEqual(len(signatures), 4)

    def test_overtake_both_owner_bools_survive_tensorization(self) -> None:
        signatures = set()
        for own in (False, True):
            for opponent in (False, True):
                changed = copy.deepcopy(self.case)
                changed["observation"]["runtime"][176]["value"]["value"] = [
                    {"type": "bool", "value": own}, {"type": "bool", "value": opponent},
                ]
                batch = tensorize_cases([changed], self.vocabulary)
                mask = (batch.runtime_field[0] == 177) & (batch.runtime_type[0] == RUNTIME_TYPE_IDS["bool"])
                actual = tuple(batch.runtime_numeric[0, mask, 0].tolist())
                self.assertEqual(actual, (float(own), float(opponent)))
                signatures.add(actual)
        self.assertEqual(len(signatures), 4)

    def test_latest_public_owner_flags_reach_tensors(self) -> None:
        for field_id in (178, 180, 181, 182):
            signatures = set()
            for own in (False, True):
                for opponent in (False, True):
                    changed = copy.deepcopy(self.case)
                    changed["observation"]["runtime"][field_id - 1]["value"] = {
                        "type": "list", "value": [
                            {"type": "bool", "value": own},
                            {"type": "bool", "value": opponent},
                        ],
                    }
                    batch = tensorize_cases([changed], self.vocabulary)
                    mask = (batch.runtime_field[0] == field_id) & (batch.runtime_type[0] == RUNTIME_TYPE_IDS["bool"])
                    actual = tuple(batch.runtime_numeric[0, mask, 0].tolist())
                    self.assertEqual(actual, (float(own), float(opponent)))
                    signatures.add(actual)
            self.assertEqual(len(signatures), 4, field_id)

    def test_outpost_local_piece_refs_and_presence_reach_tensors(self) -> None:
        for refs in ([], [0], [1], [0, 1]):
            changed = copy.deepcopy(self.case)
            changed["observation"]["runtime"][178]["value"] = {
                "type": "list", "value": [{"type": "piece", "value": ref} for ref in refs],
            }
            suite = copy.deepcopy(self.suite)
            suite["cases"] = [changed]
            validate_golden_suite(suite)
            batch = tensorize_cases([changed], self.vocabulary)
            mask = (batch.runtime_field[0] == 179) & (batch.runtime_type[0] == RUNTIME_TYPE_IDS["piece"])
            self.assertEqual(batch.runtime_piece_ref[0, mask].tolist(), refs)
        changed["observation"]["runtime"][178]["value"]["value"][0]["value"] = 999999
        suite["cases"] = [changed]
        with self.assertRaisesRegex(SchemaError, "local slot"):
            validate_golden_suite(suite)

    def test_clockwork_kind59_and_appended_planes_reach_tensors(self) -> None:
        # Parent exporter supplies real Rust-authored Clockwork positions.
        cases = [case for case in self.suite["cases"]
                 if any(piece["kind_code"] == 59 for piece in case["observation"]["pieces"])]
        self.assertTrue(cases, "golden suite must contain real Rust Clockwork pieces")
        vocabulary = V3Vocabulary.from_cases(cases)
        batch = tensorize_cases(cases, vocabulary)
        self.assertEqual(list(batch.spatial.shape[1:]), [136, 8, 8])
        owners = set()
        for row, case in enumerate(cases):
            for index, piece in enumerate(case["observation"]["pieces"]):
                if piece["kind_code"] != 59:
                    continue
                owners.add(piece["owner"])
                self.assertEqual(int(batch.piece_kind[row, index]), 59)
                self.assertEqual(int(batch.piece_square[row, index]), piece["square"] + 1)
                plane = 116 if piece["owner"] == "self_player" else 117
                rank, file = divmod(piece["square"], 8)
                self.assertEqual(float(batch.spatial[row, plane, rank, file]), 1.0)
        self.assertEqual(owners, {"self_player", "opponent"})

    def test_appended_piece_kinds_and_both_owner_planes_reach_tensors(self) -> None:
        cases = [case for case in self.suite["cases"]
                 if any(60 <= piece["kind_code"] <= 68 for piece in case["observation"]["pieces"])]
        self.assertTrue(cases, "golden suite must contain real Rust kinds60..68")
        batch = tensorize_cases(cases, V3Vocabulary.from_cases(cases))
        self.assertEqual(list(batch.spatial.shape[1:]), [136, 8, 8])
        seen = set()
        for row, case in enumerate(cases):
            for index, piece in enumerate(case["observation"]["pieces"]):
                code = piece["kind_code"]
                if not 60 <= code <= 68:
                    continue
                seen.add((code, piece["owner"]))
                plane = 118 + 2 * (code - 60) + int(piece["owner"] == "opponent")
                rank, file = divmod(piece["square"], 8)
                self.assertEqual(int(batch.piece_kind[row, index]), code)
                self.assertEqual(int(batch.piece_owner[row, index]), PLAYER_IDS[piece["owner"]])
                self.assertEqual(int(batch.piece_square[row, index]), piece["square"] + 1)
                self.assertEqual(float(batch.spatial[row, plane, rank, file]), 1.0)
        self.assertEqual(seen, {(code, owner) for code in range(60, 69)
                                for owner in ("self_player", "opponent")})

    def test_all_piece_planes_preserve_legacy118_and_append_pairs(self) -> None:
        for row, case in enumerate(self.suite["cases"]):
            observation = case["observation"]
            self.assertEqual(observation["spatial_shape"], [136, 8, 8])
            expected = torch.zeros((136, 8, 8))
            for piece in observation["pieces"]:
                code = piece["kind_code"]
                opponent = int(piece["owner"] == "opponent")
                plane = code - 1 + 58 * opponent if code <= 58 else 116 + 2 * (code - 59) + opponent
                rank, file = divmod(piece["square"], 8)
                expected[plane, rank, file] = 1.0
            actual = torch.tensor(observation["spatial"]).reshape(136, 8, 8)
            self.assertTrue(torch.equal(actual, expected), case["name"])

    def test_old118_plane_payload_and_out_of_range_kind_fail_closed(self) -> None:
        changed = copy.deepcopy(self.case)
        changed["observation"]["spatial_shape"] = [118, 8, 8]
        changed["observation"]["spatial"] = changed["observation"]["spatial"][:118 * 64]
        suite = copy.deepcopy(self.suite)
        suite["cases"] = [changed]
        with self.assertRaisesRegex(SchemaError, "spatial_shape"):
            validate_golden_suite(suite)
        changed = copy.deepcopy(self.case)
        changed["observation"]["pieces"][0]["kind_code"] = 69
        suite["cases"] = [changed]
        with self.assertRaisesRegex(SchemaError, "outside 1..68"):
            validate_golden_suite(suite)

    def test_parrot_owner_memory_presence_source_effective_reach_tensors(self) -> None:
        def memory(source: int, effective: int) -> dict:
            return {"type": "record", "value": [
                {"name": "source", "value": {"type": "piece_kind", "value": source}},
                {"name": "effective", "value": {"type": "piece_kind", "value": effective}},
            ]}

        choices = [{"type": "none"}, memory(4, 3), memory(5, 3), memory(4, 2)]
        signatures = set()
        vocabulary = V3Vocabulary.from_cases([self.case], extra_tokens=("source", "effective"))
        for own in choices:
            for opponent in choices:
                changed = copy.deepcopy(self.case)
                changed["observation"]["runtime"][182]["value"] = {
                    "type": "list", "value": [own, opponent],
                }
                suite = copy.deepcopy(self.suite)
                suite["cases"] = [changed]
                validate_golden_suite(suite)
                batch = tensorize_cases([changed], vocabulary)
                mask = batch.runtime_field[0] == 183
                self.assertTrue(torch.all(batch.runtime_piece_ref[0, mask] == -1))
                expected = [field["value"]["value"] for item in (own, opponent)
                            if item["type"] == "record" for field in item["value"]]
                kinds = mask & (batch.runtime_type[0] == RUNTIME_TYPE_IDS["piece_kind"])
                self.assertEqual(batch.runtime_piece_kind[0, kinds].tolist(), expected)
                signatures.add(tuple(
                    tuple(getattr(batch, name)[0, mask].flatten().tolist())
                    for name in ("runtime_type", "runtime_text", "runtime_piece_kind",
                                 "runtime_parent", "runtime_numeric")
                ))
        self.assertEqual(len(signatures), 16)

    def test_real_parrot_memory_fixture_contains_both_kind_fields(self) -> None:
        cases = [case for case in self.suite["cases"]
                 if any(item["type"] == "record"
                        for item in case["observation"]["runtime"][182]["value"]["value"])]
        self.assertTrue(cases, "golden fixture must contain actual Rust Parrot owner memory")
        vocabulary = V3Vocabulary.from_cases(cases)
        batch = tensorize_cases(cases, vocabulary)
        for row, case in enumerate(cases):
            memories = case["observation"]["runtime"][182]["value"]["value"]
            self.assertEqual(len(memories), 2)
            for item in memories:
                if item["type"] == "none":
                    continue
                self.assertEqual([field["name"] for field in item["value"]], ["source", "effective"])
                self.assertTrue(all(field["value"]["type"] == "piece_kind" for field in item["value"]))
            mask = batch.runtime_field[row] == 183
            self.assertTrue(torch.all(batch.runtime_piece_ref[row, mask] == -1))

    def test_noncontiguous_runtime_fields_fail_closed(self) -> None:
        changed = copy.deepcopy(self.case)
        changed["observation"]["runtime"][169]["field_id"] = 171
        with self.assertRaisesRegex(SchemaError, "field_id"):
            tensorize_cases([changed], self.vocabulary)


if __name__ == "__main__":
    unittest.main()

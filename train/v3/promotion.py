"""Fail-closed stage-13 candidate evaluation and promotion eligibility report."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import zstandard

from .onnx_runtime import OnnxRuntimeV3
from .schema import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID

PROMOTION_CONFIG_SCHEMA_ID = "augmentia.promotion-config.v3"
PROMOTION_REPORT_SCHEMA_ID = "augmentia.promotion-report.v3"
ARENA_REPORT_SCHEMA_ID = "augmentia.arena.v3"
REQUIRED_PHASES = {"playing", "forced_move", "draft", "awaiting_choice"}
REQUIRED_ACTION_KINDS = {"move", "piece_ability", "card", "draft_pick", "resolve_choice"}
REQUIRED_RUNTIME_FIELD_IDS = set(range(1, 195))
REQUIRED_PIECE_KIND_COUNT = 68


def evaluate_candidate(config_path: str | Path, report_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    _validate_config(config, config_path)
    report_path = Path(report_path).resolve()
    repository = Path(__file__).parents[2]
    candidate = _artifact(config_path, config["candidate"], metadata_required=True)
    champion = _artifact(
        config_path,
        config["champion"],
        metadata_required=config["champion"]["kind"] == "v3",
    )
    baseline = _artifact(config_path, config["v2_baseline"], metadata_required=False)
    ruleset_hash = config["ruleset_manifest_sha256"]
    ruleset_manifest_path = _resolve(config_path, config["ruleset_manifest"])
    ruleset_manifest = json.loads(ruleset_manifest_path.read_bytes())
    gates: list[dict[str, Any]] = []
    runtime: OnnxRuntimeV3 | None = None

    def schema_gate() -> dict[str, Any]:
        nonlocal runtime
        if _sha256(ruleset_manifest_path) != ruleset_hash:
            raise ValueError("ruleset manifest SHA-256 mismatch")
        if ruleset_manifest.get("id") != config["ruleset_id"]:
            raise ValueError("ruleset manifest ID mismatch")
        runtime = OnnxRuntimeV3(
            candidate["onnx"],
            candidate["metadata"],
            expected_ruleset_id=config["ruleset_id"],
            expected_ruleset_manifest_sha256=ruleset_hash,
        )
        if config["champion"]["kind"] == "v3":
            OnnxRuntimeV3(
                champion["onnx"],
                champion["metadata"],
                expected_ruleset_id=config["ruleset_id"],
                expected_ruleset_manifest_sha256=ruleset_hash,
            )
        expected_v2 = config["v2_baseline"]["sha256"]
        if baseline["onnx_sha256"] != expected_v2:
            raise ValueError("V2 baseline SHA-256 differs from the sealed baseline")
        if config["champion"]["kind"] == "v2" and champion["onnx_sha256"] != expected_v2:
            raise ValueError("V2 champion does not match the sealed baseline")
        return {
            "candidate_sha256": candidate["onnx_sha256"],
            "champion_kind": config["champion"]["kind"],
            "champion_sha256": champion["onnx_sha256"],
            "v2_baseline_sha256": baseline["onnx_sha256"],
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "ruleset_manifest": str(ruleset_manifest_path),
            "implemented_card_count": ruleset_manifest["coverage"][
                "implemented_card_count"
            ],
        }

    gates.append(_run_gate("schema_and_artifacts", schema_gate))
    schema_ok = gates[-1]["passed"]
    if config["evaluation_mode"] == "smoke":
        gates.append(
            {
                "name": "promotion_scale_budget",
                "mandatory": True,
                "passed": False,
                "metrics": {"evaluation_mode": "smoke"},
                "error": "smoke evaluations are never promotion-eligible",
            }
        )

    tactical_spec = config["tactical"]
    calibration_spec = config["calibration"]
    coverage_specs = config["coverage"]["datasets"]
    thresholds = config["thresholds"]

    gates.append(
        _dependent_gate(
            "proven_tactical_policy",
            schema_ok,
            lambda: _threshold_tactical(
                _tactical_metrics(
                    _require_runtime(runtime),
                    _sealed_records(
                        config_path,
                        tactical_spec,
                        repository,
                        config["ruleset_id"],
                        ruleset_hash,
                    ),
                ),
                thresholds,
            ),
        )
    )
    gates.append(
        _dependent_gate(
            "sealed_value_calibration",
            schema_ok,
            lambda: _threshold_calibration(
                _calibration_metrics(
                    _require_runtime(runtime),
                    _sealed_records(
                        config_path,
                        calibration_spec,
                        repository,
                        config["ruleset_id"],
                        ruleset_hash,
                    ),
                    thresholds["calibration_bins"],
                ),
                thresholds,
            ),
        )
    )
    gates.append(
        _dependent_gate(
            "rules_and_card_coverage",
            schema_ok,
            lambda: _threshold_coverage(
                _coverage_metrics(
                    record
                    for spec in coverage_specs
                    for record in _sealed_records(
                        config_path,
                        spec,
                        repository,
                        config["ruleset_id"],
                        ruleset_hash,
                    )
                ),
                thresholds,
            ),
        )
    )

    arena_reports: dict[str, Any] = {}
    if schema_ok:
        arena_reports["champion_equal_simulations"] = _capture_arena(
            repository,
            candidate,
            champion,
            config["champion"]["kind"],
            ruleset_hash,
            config["arena"],
            "simulations",
        )
        arena_reports["champion_equal_wall_clock"] = _capture_arena(
            repository,
            candidate,
            champion,
            config["champion"]["kind"],
            ruleset_hash,
            config["arena"],
            "wall_clock",
        )
        if (
            config["champion"]["kind"] == "v2"
            and champion["onnx_sha256"] == baseline["onnx_sha256"]
        ):
            arena_reports["v2_equal_simulations"] = arena_reports[
                "champion_equal_simulations"
            ]
            arena_reports["v2_equal_wall_clock"] = arena_reports[
                "champion_equal_wall_clock"
            ]
        else:
            arena_reports["v2_equal_simulations"] = _capture_arena(
                repository,
                candidate,
                baseline,
                "v2",
                ruleset_hash,
                config["arena"],
                "simulations",
            )
            arena_reports["v2_equal_wall_clock"] = _capture_arena(
                repository,
                candidate,
                baseline,
                "v2",
                ruleset_hash,
                config["arena"],
                "wall_clock",
            )

    arena_gate_names = (
        ("champion_equal_simulations", "champion_equal_simulations"),
        ("champion_equal_wall_clock", "champion_equal_wall_clock"),
        ("v2_equal_simulations", "v2_baseline_equal_simulations"),
        ("v2_equal_wall_clock", "v2_baseline_equal_wall_clock"),
    )
    for report_key, gate_name in arena_gate_names:
        gates.append(
            _dependent_gate(
                gate_name,
                schema_ok,
                lambda key=report_key: _threshold_arena(
                    _arena_metrics(arena_reports[key]), thresholds, key
                ),
            )
        )

    promotion_eligible = all(gate["passed"] for gate in gates)
    report = {
        "report_schema_id": PROMOTION_REPORT_SCHEMA_ID,
        "evaluation_id": config["evaluation_id"],
        "evaluation_mode": config["evaluation_mode"],
        "created_at_utc": config["created_at_utc"],
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "candidate": candidate,
        "champion": {"kind": config["champion"]["kind"], **champion},
        "v2_baseline": baseline,
        "gates": gates,
        "arena_reports": arena_reports,
        "promotion_eligible": promotion_eligible,
        "registry_mutated": False,
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
    }
    _atomic_json(report_path, report)
    return report


def _tactical_metrics(
    runtime: OnnxRuntimeV3, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    cases = []
    optimal_sets = []
    ids = []
    for record in records:
        if record.get("record_schema_id") != "augmentia.training.decision.v3":
            continue
        targets = record.get("targets", {})
        policy = targets.get("policy") or {}
        target = policy.get("target") or {}
        value = targets.get("value") or {}
        if target.get("kind") != "optimal_action_set" or value.get("kind") != "proven":
            continue
        cases.append(
            {
                "observation": record["observation"],
                "legal_actions": record["legal_actions"],
            }
        )
        optimal_sets.append(set(target["action_indices"]))
        ids.append(record["decision_id"])
    if not cases:
        raise ValueError("sealed tactical set has no solver-proven decisions")
    evaluations = _evaluate_cases(runtime, cases)
    correct = 0
    optimal_mass = 0.0
    failures = []
    for decision_id, evaluation, optimal in zip(ids, evaluations, optimal_sets, strict=True):
        probabilities = evaluation.probabilities
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        correct += int(best in optimal)
        optimal_mass += sum(probabilities[index] for index in optimal)
        if best not in optimal:
            failures.append(decision_id)
    return {
        "positions": len(cases),
        "top1_correct": correct,
        "top1_accuracy": correct / len(cases),
        "mean_optimal_probability_mass": optimal_mass / len(cases),
        "failed_decision_ids": failures[:100],
    }


def _calibration_metrics(
    runtime: OnnxRuntimeV3,
    records: Sequence[Mapping[str, Any]],
    bins: int,
) -> dict[str, Any]:
    cases = []
    targets = []
    for record in records:
        schema = record.get("record_schema_id")
        if schema == "augmentia.training.decision.v3":
            value = (record.get("targets", {}).get("value") or {}).get("value")
            if isinstance(value, (int, float)):
                cases.append(
                    {
                        "observation": record["observation"],
                        "legal_actions": record["legal_actions"],
                    }
                )
                targets.append(float(value))
        elif schema == "augmentia.replay.game.v3":
            termination = record.get("termination", {})
            result = termination.get("result") if termination.get("kind") == "engine_terminal" else None
            if result not in {"white_win", "black_win", "draw"}:
                continue
            for step in record.get("steps", []):
                actor = step.get("actor")
                target = 0.0 if result == "draw" else (1.0 if result == f"{actor}_win" else -1.0)
                cases.append(
                    {
                        "observation": step["observation"],
                        "legal_actions": step["legal_actions"],
                    }
                )
                targets.append(target)
    if not cases:
        raise ValueError("sealed calibration set has no terminal or proven values")
    predictions = np.asarray(
        [item.value for item in _evaluate_cases(runtime, cases)], dtype=np.float64
    )
    expected = np.asarray(targets, dtype=np.float64)
    errors = predictions - expected
    probabilities = (predictions + 1.0) / 2.0
    outcomes = (expected + 1.0) / 2.0
    return {
        "positions": len(cases),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "brier": float(np.mean((probabilities - outcomes) ** 2)),
        "ece": _expected_calibration_error(probabilities, outcomes, bins),
        "prediction_mean": float(np.mean(predictions)),
        "target_mean": float(np.mean(expected)),
    }


def _coverage_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    phases: set[str] = set()
    action_kinds: set[str] = set()
    piece_kinds: set[int] = set()
    cards: set[str] = set()
    runtime_field_ids: set[int] = set()
    positions = 0
    for record in records:
        for observation, legal_actions in _record_cases(record):
            positions += 1
            phase = observation.get("phase", {}).get("kind")
            if isinstance(phase, str):
                phases.add(phase)
            piece_kinds.update(
                piece["kind_code"]
                for piece in observation.get("pieces", [])
                if isinstance(piece.get("kind_code"), int)
            )
            cards.update(
                card["definition_id"]
                for card in observation.get("cards", [])
                if isinstance(card.get("definition_id"), str)
            )
            runtime_field_ids.update(
                feature["field_id"]
                for feature in observation.get("runtime", [])
                if isinstance(feature.get("field_id"), int)
            )
            action_kinds.update(
                action["kind"]
                for action in legal_actions.get("actions", [])
                if isinstance(action.get("kind"), str)
            )
    if positions == 0:
        raise ValueError("coverage datasets contain no evaluable positions")
    return {
        "positions": positions,
        "phases": sorted(phases),
        "action_kinds": sorted(action_kinds),
        "piece_kind_codes": sorted(piece_kinds),
        "card_definition_ids": sorted(cards),
        "runtime_field_ids": sorted(runtime_field_ids),
        "phase_count": len(phases),
        "action_kind_count": len(action_kinds),
        "piece_kind_count": len(piece_kinds),
        "card_definition_count": len(cards),
        "runtime_field_count": len(runtime_field_ids),
    }


def _arena_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("report_schema_id") != ARENA_REPORT_SCHEMA_ID:
        raise ValueError("arena report schema mismatch")
    games = report.get("games")
    if not isinstance(games, list) or len(games) != report["pairs"] * 2:
        raise ValueError("arena did not return exactly two colors per seed")
    pair_colors: dict[tuple[int, int], set[str]] = {}
    scores = []
    color_scores: dict[str, list[float]] = {"white": [], "black": []}
    candidate_cards_by_color: dict[str, Counter[str]] = {
        "white": Counter(),
        "black": Counter(),
    }
    errors = 0
    for game in games:
        key = (game["pair_index"], game["seed"])
        pair_colors.setdefault(key, set()).add(game["candidate_color"])
        if game["outcome"] == "error":
            errors += 1
        if game.get("terminal"):
            score = {
                "candidate_win": 1.0,
                "draw": 0.5,
                "opponent_win": 0.0,
            }.get(game["outcome"])
            if score is not None:
                scores.append(score)
                color_scores[game["candidate_color"]].append(score)
        candidate_cards_by_color[game["candidate_color"]].update(game["candidate_cards"])
    if any(colors != {"white", "black"} for colors in pair_colors.values()):
        raise ValueError("arena seed/color pairing is incomplete")
    score = sum(scores) / len(scores) if scores else 0.0
    lower, upper = _wilson_interval(sum(scores), len(scores)) if scores else (0.0, 0.0)
    white_score = _mean(color_scores["white"])
    black_score = _mean(color_scores["black"])
    simulation_exact = True
    if report["budget_kind"] == "simulations":
        expected = report["simulations"]
        simulation_exact = all(
            game[side]["simulations"] == game[side]["searches"] * expected
            for game in games
            for side in ("candidate", "opponent")
        )
    return {
        "games": len(games),
        "terminal_games": len(scores),
        "terminal_rate": len(scores) / len(games),
        "errors": errors,
        "score": score,
        "score_wilson_95_lower": lower,
        "score_wilson_95_upper": upper,
        "candidate_as_white_score": white_score,
        "candidate_as_black_score": black_score,
        "color_score_gap": abs(white_score - black_score),
        "candidate_card_color_tv": _counter_total_variation(
            candidate_cards_by_color["white"], candidate_cards_by_color["black"]
        ),
        "paired_seed_count": len(pair_colors),
        "simulation_budget_exact": simulation_exact,
        "candidate_mean_search_ms": _mean_search_ms(games, "candidate"),
        "opponent_mean_search_ms": _mean_search_ms(games, "opponent"),
        "candidate_mean_simulations_per_search": _mean_simulations(games, "candidate"),
        "opponent_mean_simulations_per_search": _mean_simulations(games, "opponent"),
    }


def _threshold_tactical(metrics: dict[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    passed = (
        metrics["positions"] >= thresholds["tactical_positions_min"]
        and metrics["top1_accuracy"] >= thresholds["tactical_top1_accuracy_min"]
        and metrics["mean_optimal_probability_mass"]
        >= thresholds["tactical_optimal_mass_min"]
    )
    return {**metrics, "passed": passed}


def _threshold_calibration(metrics: dict[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    passed = (
        metrics["positions"] >= thresholds["calibration_positions_min"]
        and metrics["rmse"] <= thresholds["value_rmse_max"]
        and metrics["ece"] <= thresholds["value_ece_max"]
    )
    return {**metrics, "passed": passed}


def _threshold_coverage(metrics: dict[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    missing_phases = sorted(set(thresholds["required_phases"]) - set(metrics["phases"]))
    missing_actions = sorted(
        set(thresholds["required_action_kinds"]) - set(metrics["action_kinds"])
    )
    missing_runtime = sorted(
        set(thresholds["required_runtime_field_ids"])
        - set(metrics["runtime_field_ids"])
    )
    passed = (
        metrics["piece_kind_count"] >= thresholds["piece_kind_count_min"]
        and metrics["card_definition_count"] >= thresholds["card_definition_count_min"]
        and not missing_phases
        and not missing_actions
        and not missing_runtime
    )
    return {
        **metrics,
        "missing_phases": missing_phases,
        "missing_action_kinds": missing_actions,
        "missing_runtime_field_ids": missing_runtime,
        "passed": passed,
    }


def _threshold_arena(
    metrics: dict[str, Any], thresholds: Mapping[str, Any], key: str
) -> dict[str, Any]:
    baseline = key.startswith("v2_")
    score_min = thresholds["v2_score_min"] if baseline else thresholds["champion_score_min"]
    lower_min = (
        thresholds["v2_score_lcb_min"]
        if baseline
        else thresholds["champion_score_lcb_min"]
    )
    passed = (
        metrics["terminal_games"] >= thresholds["arena_terminal_games_min"]
        and metrics["terminal_rate"] >= thresholds["arena_terminal_rate_min"]
        and metrics["errors"] == 0
        and metrics["score"] >= score_min
        and metrics["score_wilson_95_lower"] >= lower_min
        and metrics["color_score_gap"] <= thresholds["arena_color_score_gap_max"]
        and metrics["candidate_card_color_tv"] <= thresholds["arena_card_color_tv_max"]
        and metrics["simulation_budget_exact"]
    )
    return {**metrics, "passed": passed}


def _sealed_records(
    config_path: Path,
    spec: Mapping[str, Any],
    repository: Path,
    expected_ruleset_id: str,
    expected_ruleset_hash: str,
) -> list[Mapping[str, Any]]:
    manifest_path = _resolve(config_path, spec["manifest"])
    actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual != spec["manifest_sha256"]:
        raise ValueError(f"sealed manifest hash mismatch for {manifest_path}")
    _run_validator(repository, manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    split = spec["split"]
    if split not in {"validation", "holdout"}:
        raise ValueError("promotion datasets must use a sealed validation or holdout split")
    records = []
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        shard_path = manifest_path.parent / Path(shard["path"])
        compressed = shard_path.read_bytes()
        if hashlib.sha256(compressed).hexdigest() != shard["sha256"]:
            raise ValueError(f"shard hash mismatch for {shard_path}")
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
            raw = reader.read()
        records.extend(json.loads(line) for line in raw.splitlines())
    if not records:
        raise ValueError(f"sealed manifest has no {split} records")
    for record in records:
        ruleset = record.get("ruleset", {})
        if (
            ruleset.get("id") != expected_ruleset_id
            or ruleset.get("manifest_sha256") != expected_ruleset_hash
        ):
            raise ValueError("sealed record ruleset artifact differs from the candidate")
    return records


def _run_arena(
    repository: Path,
    candidate: Mapping[str, Any],
    opponent: Mapping[str, Any],
    opponent_kind: str,
    ruleset_hash: str,
    arena: Mapping[str, Any],
    budget_kind: str,
) -> dict[str, Any]:
    command = [
        "cargo",
        "run",
        "-q",
        "-p",
        "augment-cli",
        "--bin",
        "arena_v3",
        "--",
        "--candidate-model",
        str(candidate["onnx"]),
        "--candidate-metadata",
        str(candidate["metadata"]),
        "--opponent-kind",
        opponent_kind,
        "--opponent-model",
        str(opponent["onnx"]),
        "--ruleset-hash",
        ruleset_hash,
        "--budget-kind",
        budget_kind,
    ]
    if opponent_kind == "v3":
        command.extend(["--opponent-metadata", str(opponent["metadata"])])
    for name in (
        "pairs",
        "base_seed",
        "max_decisions",
        "simulations",
        "move_time_ms",
        "max_simulations",
        "leaf_batch_size",
    ):
        command.extend(["--" + name.replace("_", "-"), str(arena[name])])
    environment = os.environ.copy()
    environment["ORT_DYLIB_PATH"] = str(
        Path(sys.prefix)
        / "Lib"
        / "site-packages"
        / "onnxruntime"
        / "capi"
        / "onnxruntime.dll"
    )
    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"arena failed: {completed.stderr}")
    return json.loads(completed.stdout)


def _capture_arena(*args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return _run_arena(*args, **kwargs)
    except Exception as error:
        return {
            "report_schema_id": ARENA_REPORT_SCHEMA_ID,
            "arena_error": f"{type(error).__name__}: {error}",
        }


def _run_validator(repository: Path, manifest: Path) -> None:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "augment-cli",
            "--bin",
            "validate_v3_dataset",
            "--",
            "--manifest",
            str(manifest),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"dataset validation failed: {completed.stderr}")


def _record_cases(
    record: Mapping[str, Any],
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if record.get("record_schema_id") == "augmentia.training.decision.v3":
        yield record["observation"], record["legal_actions"]
    elif record.get("record_schema_id") == "augmentia.replay.game.v3":
        for step in record.get("steps", []):
            yield step["observation"], step["legal_actions"]


def _evaluate_cases(runtime: OnnxRuntimeV3, cases: Sequence[Mapping[str, Any]]) -> list[Any]:
    output = []
    for start in range(0, len(cases), 64):
        output.extend(runtime.evaluate_cases(cases[start : start + 64]))
    return output


def _expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int
) -> float:
    bins = _positive_int(bins, "calibration_bins")
    total = len(probabilities)
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        count = int(selected.sum())
        if count:
            result += count / total * abs(
                float(probabilities[selected].mean() - outcomes[selected].mean())
            )
    return result


def _wilson_interval(successes: float, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires successes inside trial count")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _counter_total_variation(left: Counter[str], right: Counter[str]) -> float:
    left_total, right_total = sum(left.values()), sum(right.values())
    if left_total == 0 and right_total == 0:
        return 0.0
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(
            (left[key] / left_total if left_total else 0.0)
            - (right[key] / right_total if right_total else 0.0)
        )
        for key in keys
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_search_ms(games: Sequence[Mapping[str, Any]], side: str) -> float:
    searches = sum(game[side]["searches"] for game in games)
    return sum(game[side]["elapsed_ms"] for game in games) / searches if searches else 0.0


def _mean_simulations(games: Sequence[Mapping[str, Any]], side: str) -> float:
    searches = sum(game[side]["searches"] for game in games)
    return sum(game[side]["simulations"] for game in games) / searches if searches else 0.0


def _run_gate(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        metrics = function()
        passed = bool(metrics.pop("passed", True))
        return {"name": name, "mandatory": True, "passed": passed, "metrics": metrics}
    except Exception as error:  # gate reports must survive malformed candidates
        return {
            "name": name,
            "mandatory": True,
            "passed": False,
            "metrics": {},
            "error": f"{type(error).__name__}: {error}",
        }


def _dependent_gate(
    name: str, dependency: bool, function: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    if not dependency:
        return {
            "name": name,
            "mandatory": True,
            "passed": False,
            "metrics": {},
            "error": "skipped because schema_and_artifacts failed",
        }
    return _run_gate(name, function)


def _artifact(
    config_path: Path, payload: Mapping[str, Any], *, metadata_required: bool
) -> dict[str, Any]:
    model = _resolve(config_path, payload["onnx"])
    result = {"onnx": str(model), "onnx_sha256": _sha256(model)}
    metadata_value = payload.get("metadata")
    if metadata_required and not metadata_value:
        raise ValueError("V3 artifact requires metadata")
    if metadata_value:
        metadata = _resolve(config_path, metadata_value)
        result.update({"metadata": str(metadata), "metadata_sha256": _sha256(metadata)})
    else:
        result["metadata"] = None
    return result


def _validate_config(config: Any, config_path: Path) -> None:
    required = {
        "promotion_config_schema_id",
        "evaluation_id",
        "created_at_utc",
        "evaluation_mode",
        "ruleset_id",
        "ruleset_manifest",
        "ruleset_manifest_sha256",
        "candidate",
        "champion",
        "v2_baseline",
        "tactical",
        "calibration",
        "coverage",
        "arena",
        "thresholds",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("promotion config fields do not match the V3 contract")
    if config["promotion_config_schema_id"] != PROMOTION_CONFIG_SCHEMA_ID:
        raise ValueError("unsupported promotion config schema")
    if not isinstance(config["evaluation_id"], str) or not config["evaluation_id"]:
        raise ValueError("evaluation_id must be non-empty")
    if not config["created_at_utc"].endswith("Z"):
        raise ValueError("created_at_utc must end with Z")
    if config["evaluation_mode"] not in {"smoke", "promotion"}:
        raise ValueError("evaluation_mode must be smoke or promotion")
    if config["champion"].get("kind") not in {"v2", "v3"}:
        raise ValueError("champion kind must be v2 or v3")
    for spec in (config["tactical"], config["calibration"], *config["coverage"]["datasets"]):
        if set(spec) != {"manifest", "manifest_sha256", "split"}:
            raise ValueError("sealed dataset specs require manifest, hash, and split")
    for name in (
        "pairs",
        "max_decisions",
        "simulations",
        "move_time_ms",
        "max_simulations",
        "leaf_batch_size",
    ):
        _positive_int(config["arena"].get(name), f"arena.{name}")
    ruleset_path = _resolve(config_path, config["ruleset_manifest"])
    if _sha256(ruleset_path) != config["ruleset_manifest_sha256"]:
        raise ValueError("ruleset manifest SHA-256 mismatch")
    ruleset = json.loads(ruleset_path.read_bytes())
    if ruleset.get("id") != config["ruleset_id"]:
        raise ValueError("ruleset manifest ID mismatch")
    thresholds = config["thresholds"]
    implemented_cards = ruleset["coverage"]["implemented_card_count"]
    if thresholds["card_definition_count_min"] < implemented_cards:
        raise ValueError("card coverage threshold is below the implemented ruleset")
    if thresholds["piece_kind_count_min"] < REQUIRED_PIECE_KIND_COUNT:
        raise ValueError("piece coverage threshold is below the V3 vocabulary")
    if set(thresholds["required_phases"]) != REQUIRED_PHASES:
        raise ValueError("required phases must cover the complete V3 phase contract")
    if set(thresholds["required_action_kinds"]) != REQUIRED_ACTION_KINDS:
        raise ValueError("required actions must cover the complete ActionV3 contract")
    if set(thresholds["required_runtime_field_ids"]) != REQUIRED_RUNTIME_FIELD_IDS:
        raise ValueError("required runtime fields must cover all 194 V3 fields")
    if config["evaluation_mode"] == "promotion":
        _validate_promotion_scale(config)


def _validate_promotion_scale(config: Mapping[str, Any]) -> None:
    arena = config["arena"]
    thresholds = config["thresholds"]
    lower_bounds = {
        "tactical_positions_min": 100,
        "tactical_top1_accuracy_min": 0.90,
        "tactical_optimal_mass_min": 0.70,
        "calibration_positions_min": 1000,
        "arena_terminal_games_min": 100,
        "arena_terminal_rate_min": 0.80,
        "champion_score_min": 0.55,
        "champion_score_lcb_min": 0.50,
        "v2_score_min": 0.55,
        "v2_score_lcb_min": 0.50,
    }
    upper_bounds = {
        "value_rmse_max": 0.75,
        "value_ece_max": 0.15,
        "arena_color_score_gap_max": 0.10,
        "arena_card_color_tv_max": 0.20,
    }
    for name, minimum in lower_bounds.items():
        if thresholds[name] < minimum:
            raise ValueError(f"promotion threshold {name} is weaker than {minimum}")
    for name, maximum in upper_bounds.items():
        if thresholds[name] > maximum:
            raise ValueError(f"promotion threshold {name} is weaker than {maximum}")
    if arena["pairs"] < 100 or arena["simulations"] < 150 or arena["move_time_ms"] < 500:
        raise ValueError("promotion arena requires >=100 pairs, 150 simulations, and 500 ms")


def _require_runtime(runtime: OnnxRuntimeV3 | None) -> OnnxRuntimeV3:
    if runtime is None:
        raise RuntimeError("candidate runtime was not initialized")
    return runtime


def _resolve(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one V3 promotion candidate")
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = evaluate_candidate(args.config, args.report)
    print(args.report.resolve())
    if not report["promotion_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

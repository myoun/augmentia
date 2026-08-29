"""Run deterministic V3 self-play games in fail-fast parallel workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import zstandard


@dataclass(frozen=True)
class ParallelSelfPlayConfig:
    repository: Path
    binary: Path
    validator: Path
    output_dir: Path
    model_a: Path
    metadata_a: Path
    model_b: Path
    metadata_b: Path
    ruleset_hash: str
    run_id: str
    created_at_utc: str
    games: int
    base_seed: int
    workers: int = 4
    simulations: int = 400
    leaf_batch_size: int = 16
    max_decisions: int = 512
    temperature_decisions: int = 30
    temperature: float = 1.0
    noise_alpha: float = 0.3
    noise_epsilon: float = 0.25
    train_bp: int = 9000
    validation_bp: int = 500
    holdout_bp: int = 500
    compression_level: int = 3
    execution_provider: str = "cpu"
    intra_threads: int = 1
    ort_dylib: Path | None = None
    require_clean: bool = True


@dataclass
class _Worker:
    game_index: int
    process: subprocess.Popen[str]
    stdout_handle: Any
    stderr_handle: Any
    stderr_path: Path
    started: float


def run_parallel_selfplay(
    config: ParallelSelfPlayConfig, *, resume: bool = False
) -> Path:
    config = _validated(config)
    output = config.output_dir.resolve()
    root_manifest = output / "manifest.json"
    if config.require_clean:
        _require_clean_worktree(config.repository)
    expected_commit = _git_head(config.repository)
    contract = _run_contract(config, expected_commit)
    contract_path = output / "run-config.json"
    if contract_path.exists():
        if _load_json(contract_path) != contract:
            raise RuntimeError("existing self-play run-config differs from this invocation")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("self-play artifacts exist without a matching run-config")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(contract_path, contract)
    if root_manifest.exists():
        if not resume:
            raise FileExistsError(f"self-play dataset already exists: {root_manifest}")
        _validate_root_manifest(root_manifest, config)
        _validate_dataset(config.validator, root_manifest, config.repository)
        for game_index in range(config.games):
            _validate_part_manifest(
                _part_dir(output / "parts", game_index) / "manifest.json",
                config,
                game_index,
                expected_commit,
            )
        return root_manifest
    parts = output / "parts"
    logs = output / "logs"
    parts.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)

    completed: dict[int, Path] = {}
    pending: deque[int] = deque()
    for game_index in range(config.games):
        manifest = _part_dir(parts, game_index) / "manifest.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError(
                    f"partial self-play artifacts exist; use resume: {manifest}"
                )
            _validate_part_manifest(manifest, config, game_index, expected_commit)
            _validate_dataset(config.validator, manifest, config.repository)
            completed[game_index] = manifest
            continue
        part = manifest.parent
        if part.exists() and any(part.iterdir()):
            raise RuntimeError(
                f"incomplete non-empty part requires inspection before retry: {part}"
            )
        pending.append(game_index)

    active: dict[int, _Worker] = {}
    started = time.monotonic()
    initial_completed = len(completed)
    try:
        while pending or active:
            while pending and len(active) < config.workers:
                game_index = pending.popleft()
                worker = _start_worker(config, game_index, parts, logs)
                active[worker.process.pid] = worker
                print(
                    json.dumps(
                        {
                            "event": "selfplay_worker_started",
                            "game_index": game_index,
                            "pid": worker.process.pid,
                            "active": len(active),
                            "completed": len(completed),
                            "games": config.games,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finished = [
                pid for pid, worker in active.items() if worker.process.poll() is not None
            ]
            if not finished:
                time.sleep(0.2)
                continue
            for pid in finished:
                worker = active.pop(pid)
                _close_worker_logs(worker)
                if worker.process.returncode:
                    _stop_workers(active.values())
                    detail = _tail(worker.stderr_path, 40)
                    raise RuntimeError(
                        f"self-play game {worker.game_index} exited "
                        f"{worker.process.returncode}\n{detail}"
                    )
                manifest = _part_dir(parts, worker.game_index) / "manifest.json"
                _validate_part_manifest(
                    manifest, config, worker.game_index, expected_commit
                )
                _validate_dataset(config.validator, manifest, config.repository)
                completed[worker.game_index] = manifest
                elapsed = time.monotonic() - started
                total_completed = len(completed)
                completed_this_run = total_completed - initial_completed
                remaining = config.games - total_completed
                eta = elapsed / max(1, completed_this_run) * remaining
                print(
                    json.dumps(
                        {
                            "event": "selfplay_game_committed",
                            "game_index": worker.game_index,
                            "game_seconds": time.monotonic() - worker.started,
                            "completed": total_completed,
                            "games": config.games,
                            "active": len(active),
                            "elapsed_seconds": elapsed,
                            "eta_seconds": eta,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    except BaseException:
        _stop_workers(active.values())
        raise

    ordered = [completed[index] for index in range(config.games)]
    return _commit_root_manifest(ordered, root_manifest, config)


def _start_worker(
    config: ParallelSelfPlayConfig, game_index: int, parts: Path, logs: Path
) -> _Worker:
    part = _part_dir(parts, game_index)
    part.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"game-{game_index:08}.stdout.log"
    stderr_path = logs / f"game-{game_index:08}.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    if config.ort_dylib is not None:
        environment["ORT_DYLIB_PATH"] = str(config.ort_dylib)
        dll_dirs = [config.ort_dylib.parent]
        torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
        cuda_12 = Path(environment.get("CUDA_PATH_V12_8", "")) / "bin"
        dll_dirs.extend(path for path in (torch_lib, cuda_12) if path.is_dir())
        environment["PATH"] = (
            os.pathsep.join(str(path) for path in dll_dirs)
            + os.pathsep
            + environment.get("PATH", "")
        )
    command = _worker_command(config, game_index, part)
    try:
        process = subprocess.Popen(
            command,
            cwd=config.repository,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    except BaseException:
        stdout_handle.close()
        stderr_handle.close()
        raise
    return _Worker(
        game_index=game_index,
        process=process,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        stderr_path=stderr_path,
        started=time.monotonic(),
    )


def _worker_command(
    config: ParallelSelfPlayConfig, game_index: int, output: Path
) -> list[str]:
    values: Sequence[tuple[str, Any]] = (
        ("--model-a", config.model_a),
        ("--metadata-a", config.metadata_a),
        ("--model-b", config.model_b),
        ("--metadata-b", config.metadata_b),
        ("--output-dir", output),
        ("--ruleset-hash", config.ruleset_hash),
        ("--run-id", config.run_id),
        ("--created-at-utc", config.created_at_utc),
        ("--games", 1),
        ("--game-index-start", game_index),
        ("--base-seed", config.base_seed),
        ("--simulations", config.simulations),
        ("--leaf-batch-size", config.leaf_batch_size),
        ("--max-decisions", config.max_decisions),
        ("--temperature-decisions", config.temperature_decisions),
        ("--temperature", config.temperature),
        ("--noise-alpha", config.noise_alpha),
        ("--noise-epsilon", config.noise_epsilon),
        ("--train-bp", config.train_bp),
        ("--validation-bp", config.validation_bp),
        ("--holdout-bp", config.holdout_bp),
        ("--compression-level", config.compression_level),
        ("--execution-provider", config.execution_provider),
        ("--intra-threads", config.intra_threads),
    )
    command = [str(config.binary)]
    for flag, value in values:
        command.extend((flag, str(value)))
    return command


def _commit_root_manifest(
    part_manifests: Sequence[Path], root_manifest: Path, config: ParallelSelfPlayConfig
) -> Path:
    merged: dict[str, Any] | None = None
    shards: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for manifest_path in part_manifests:
        payload = _load_json(manifest_path)
        _validate_part_manifest_payload(payload, manifest_path, config)
        if merged is None:
            merged = {
                name: payload[name]
                for name in (
                    "manifest_schema_id",
                    "dataset_id",
                    "created_at_utc",
                    "observation_schema_id",
                    "action_schema_id",
                    "split_policy",
                    "compression",
                )
            }
            totals = {name: 0 for name in payload["totals"]}
        else:
            for name in (
                "manifest_schema_id",
                "dataset_id",
                "created_at_utc",
                "observation_schema_id",
                "action_schema_id",
                "split_policy",
                "compression",
            ):
                if payload[name] != merged[name]:
                    raise ValueError(f"part manifest {manifest_path} differs in {name}")
        relative_part = manifest_path.parent.relative_to(root_manifest.parent)
        for shard in payload["shards"]:
            entry = dict(shard)
            entry["path"] = str(
                PurePosixPath(relative_part.as_posix()) / PurePosixPath(shard["path"])
            )
            shards.append(entry)
        for name, value in payload["totals"].items():
            totals[name] += value
    if merged is None:
        raise ValueError("cannot commit an empty self-play dataset")
    merged["shards"] = sorted(shards, key=lambda item: item["path"])
    merged["totals"] = totals
    if totals.get("games") != config.games or totals.get("shards") != len(shards):
        raise ValueError("merged self-play totals do not match completed parts")
    temporary = root_manifest.with_name(f"manifest.json.partial-{os.getpid()}")
    _write_json(temporary, merged)
    try:
        _validate_dataset(config.validator, temporary, config.repository)
        os.replace(temporary, root_manifest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return root_manifest


def consolidate_completed_selfplay(
    output_dir: str | Path,
    validator: str | Path,
    repository: str | Path,
) -> Path:
    """Atomically seal every completed part without requiring a contiguous run."""
    output = Path(output_dir).resolve()
    root_manifest = output / "manifest.json"
    part_manifests = sorted((output / "parts").glob("game-*/manifest.json"))
    if not part_manifests:
        raise ValueError(f"self-play run has no completed parts: {output}")
    merged: dict[str, Any] | None = None
    shards: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    fixed_fields = (
        "manifest_schema_id",
        "dataset_id",
        "observation_schema_id",
        "action_schema_id",
        "split_policy",
        "compression",
    )
    for manifest_path in part_manifests:
        payload = _load_json(manifest_path)
        if merged is None:
            merged = {name: payload[name] for name in fixed_fields}
            merged["created_at_utc"] = payload["created_at_utc"]
            totals = {name: 0 for name in payload["totals"]}
        else:
            for name in fixed_fields:
                if payload[name] != merged[name]:
                    raise ValueError(f"part manifest {manifest_path} differs in {name}")
            if set(payload["totals"]) != set(totals):
                raise ValueError(f"part manifest {manifest_path} totals drifted")
        relative_part = manifest_path.parent.relative_to(output)
        for shard in payload["shards"]:
            entry = dict(shard)
            entry["path"] = str(
                PurePosixPath(relative_part.as_posix()) / PurePosixPath(shard["path"])
            )
            shards.append(entry)
        for name, value in payload["totals"].items():
            totals[name] += value
    assert merged is not None
    merged["shards"] = sorted(shards, key=lambda item: item["path"])
    merged["totals"] = totals
    if totals.get("games") != len(part_manifests) or totals.get("shards") != len(shards):
        raise ValueError("consolidated self-play totals do not match completed parts")
    temporary = root_manifest.with_name(f"manifest.json.partial-{os.getpid()}")
    _write_json(temporary, merged)
    try:
        _validate_dataset(Path(validator).resolve(), temporary, Path(repository).resolve())
        os.replace(temporary, root_manifest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return root_manifest


def _validate_part_manifest(
    path: Path,
    config: ParallelSelfPlayConfig,
    game_index: int,
    expected_commit: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"worker did not commit a manifest: {path}")
    payload = _load_json(path)
    _validate_part_manifest_payload(payload, path, config)
    shards = payload["shards"]
    if len(shards) != 1:
        raise ValueError(f"part {path} must contain exactly one shard")
    shard = path.parent / Path(shards[0]["path"])
    with shard.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            lines = reader.read().splitlines()
    if len(lines) != 1:
        raise ValueError(f"part {path} must contain exactly one JSON record")
    record = json.loads(lines[0])
    expected_game_id = f"{config.run_id}-game-{game_index:08}"
    expected_seed = f"{(config.base_seed + game_index) & ((1 << 64) - 1):016x}"
    generator = record.get("generator", {})
    if record.get("game_id") != expected_game_id or record.get("seed_hex") != expected_seed:
        raise ValueError(f"part {path} does not match game index {game_index}")
    if (
        generator.get("run_id") != config.run_id
        or generator.get("engine_commit") != expected_commit
        or generator.get("dirty_worktree") is not False
    ):
        raise ValueError(f"part {path} has unexpected generator provenance")


def _validate_part_manifest_payload(
    payload: Mapping[str, Any], path: Path, config: ParallelSelfPlayConfig
) -> None:
    if payload.get("dataset_id") != f"selfplay-v3:{config.run_id}":
        raise ValueError(f"unexpected dataset identity in {path}")
    if payload.get("created_at_utc") != config.created_at_utc:
        raise ValueError(f"unexpected creation time in {path}")
    split = payload.get("split_policy", {})
    expected_split = {
        "algorithm": "sha256_split_group_v1",
        "salt": f"selfplay-v3:{config.run_id}",
        "train_basis_points": config.train_bp,
        "validation_basis_points": config.validation_bp,
        "holdout_basis_points": config.holdout_bp,
    }
    if split != expected_split:
        raise ValueError(f"unexpected split policy in {path}")
    compression = payload.get("compression", {})
    if compression != {"codec": "zstd", "level": config.compression_level}:
        raise ValueError(f"unexpected compression contract in {path}")
    totals = payload.get("totals", {})
    if totals.get("games") != 1 or totals.get("records") != 1:
        raise ValueError(f"part {path} must contain exactly one game")
    if totals.get("shards") != len(payload.get("shards", [])):
        raise ValueError(f"part {path} shard totals do not match")


def _validate_root_manifest(path: Path, config: ParallelSelfPlayConfig) -> None:
    payload = _load_json(path)
    if payload.get("dataset_id") != f"selfplay-v3:{config.run_id}":
        raise ValueError("existing root manifest has the wrong dataset identity")
    totals = payload.get("totals", {})
    if totals.get("games") != config.games:
        raise ValueError("existing root manifest has the wrong game count")


def _validate_dataset(validator: Path, manifest: Path, repository: Path) -> None:
    completed = subprocess.run(
        [str(validator), "--manifest", str(manifest)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"dataset validation failed for {manifest}: {completed.stderr.strip()}"
        )


def _stop_workers(workers: Sequence[_Worker] | Any) -> None:
    workers = list(workers)
    for worker in workers:
        if worker.process.poll() is None:
            worker.process.terminate()
    deadline = time.monotonic() + 2.0
    for worker in workers:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            worker.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            worker.process.wait()
        finally:
            _close_worker_logs(worker)


def _close_worker_logs(worker: _Worker) -> None:
    if not worker.stdout_handle.closed:
        worker.stdout_handle.close()
    if not worker.stderr_handle.closed:
        worker.stderr_handle.close()


def _validated(config: ParallelSelfPlayConfig) -> ParallelSelfPlayConfig:
    for name in ("games", "workers", "simulations", "leaf_batch_size", "intra_threads"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.execution_provider not in {"cpu", "cuda"}:
        raise ValueError("execution_provider must be cpu or cuda")
    if len(config.ruleset_hash) != 64:
        raise ValueError("ruleset_hash must be SHA-256")
    if not config.created_at_utc.endswith("Z"):
        raise ValueError("created_at_utc must end with Z")
    for path in (
        config.repository,
        config.binary,
        config.validator,
        config.model_a,
        config.metadata_a,
        config.model_b,
        config.metadata_b,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if config.ort_dylib is not None and not config.ort_dylib.is_file():
        raise FileNotFoundError(config.ort_dylib)
    return config


def _run_contract(
    config: ParallelSelfPlayConfig, engine_commit: str
) -> dict[str, Any]:
    return {
        "schema_id": "augmentia.selfplay-run-config.v3",
        "engine_commit": engine_commit,
        "run_id": config.run_id,
        "created_at_utc": config.created_at_utc,
        "ruleset_hash": config.ruleset_hash,
        "models": [
            {
                "onnx_sha256": _sha256_file(config.model_a),
                "metadata_sha256": _sha256_file(config.metadata_a),
            },
            {
                "onnx_sha256": _sha256_file(config.model_b),
                "metadata_sha256": _sha256_file(config.metadata_b),
            },
        ],
        "generation": {
            "games": config.games,
            "base_seed": config.base_seed,
            "simulations": config.simulations,
            "leaf_batch_size": config.leaf_batch_size,
            "max_decisions": config.max_decisions,
            "temperature_decisions": config.temperature_decisions,
            "temperature": config.temperature,
            "noise_alpha": config.noise_alpha,
            "noise_epsilon": config.noise_epsilon,
            "train_bp": config.train_bp,
            "validation_bp": config.validation_bp,
            "holdout_bp": config.holdout_bp,
            "compression_level": config.compression_level,
        },
        "execution": {
            "workers": config.workers,
            "provider": config.execution_provider,
            "intra_threads": config.intra_threads,
            "ort_dylib_sha256": (
                _sha256_file(config.ort_dylib)
                if config.ort_dylib is not None
                else None
            ),
        },
    }


def _require_clean_worktree(repository: Path) -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"git status failed: {completed.stderr.strip()}")
    if completed.stdout.strip():
        raise RuntimeError("self-play requires a clean worktree")


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if completed.returncode or len(commit) != 40:
        raise RuntimeError(f"cannot resolve engine commit: {completed.stderr.strip()}")
    return commit


def _part_dir(parts: Path, game_index: int) -> Path:
    return parts / f"game-{game_index:08}"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.partial-{os.getpid()}")
    _write_json(temporary, payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail(path: Path, lines: int) -> str:
    if not path.exists():
        return "(stderr log missing)"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fail-fast parallel V3 self-play")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--metadata-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--metadata-b", type=Path, required=True)
    parser.add_argument("--ruleset-hash", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--leaf-batch-size", type=int, default=16)
    parser.add_argument("--max-decisions", type=int, default=512)
    parser.add_argument("--temperature-decisions", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--noise-alpha", type=float, default=0.3)
    parser.add_argument("--noise-epsilon", type=float, default=0.25)
    parser.add_argument("--train-bp", type=int, default=9000)
    parser.add_argument("--validation-bp", type=int, default=500)
    parser.add_argument("--holdout-bp", type=int, default=500)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--execution-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--intra-threads", type=int, default=1)
    parser.add_argument("--ort-dylib", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    values = vars(args).copy()
    resume = values.pop("resume")
    values["require_clean"] = not values.pop("allow_dirty")
    print(run_parallel_selfplay(ParallelSelfPlayConfig(**values), resume=resume))


if __name__ == "__main__":
    main()

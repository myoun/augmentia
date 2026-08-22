"""Measure bounded V3 replay loading on a real manifest."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

from .replay import DeterministicSamplerV3, ReplayDatasetV3, collate_examples


def _memory_bytes() -> tuple[int | None, int | None]:
    if sys.platform == "win32":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not success:
            raise OSError("GetProcessMemoryInfo failed")
        return counters.working_set_size, counters.peak_working_set_size
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak *= 1024
        return None, peak
    except (ImportError, OSError):
        return None, None


def _memory_report() -> dict[str, float | None]:
    current, peak = _memory_bytes()
    return {
        "working_set_gib": None if current is None else current / 1024**3,
        "peak_working_set_gib": None if peak is None else peak / 1024**3,
    }


def benchmark(
    manifest: Path,
    *,
    batch_size: int,
    seed: int,
    max_cached_shards: int,
    verification_workers: int | None,
    progress: bool,
) -> dict[str, Any]:
    started = time.perf_counter()

    def report_progress(completed: int, total: int, shard: str) -> None:
        if progress:
            print(
                json.dumps(
                    {
                        "event": "verification_progress",
                        "completed_shards": completed,
                        "total_shards": total,
                        "shard": shard,
                        "elapsed_seconds": time.perf_counter() - started,
                        **_memory_report(),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    dataset = ReplayDatasetV3(
        manifest,
        max_cached_shards=max_cached_shards,
        verification_workers=verification_workers,
        progress_callback=report_progress,
    )
    initialized = time.perf_counter()
    initialization_memory = _memory_report()

    vocabulary = dataset.build_vocabulary()
    vocabulary_built = time.perf_counter()
    vocabulary_memory = _memory_report()

    sampler = DeterministicSamplerV3(
        len(dataset), seed=seed, group_sizes=dataset.sampler_group_sizes
    )
    indices = sampler.take(min(batch_size, len(dataset)))
    batch = collate_examples([dataset[index] for index in indices], vocabulary)
    batch_built = time.perf_counter()
    batch_memory = _memory_report()
    cache_before_clear = dataset.cache_info()
    dataset.clear_cache()
    del batch
    gc.collect()

    return {
        "manifest": str(manifest.resolve()),
        "examples": len(dataset),
        "shards": len(dataset.shards),
        "index_cache_hit": dataset.index_cache_hit,
        "index_cache_path": str(dataset.index_cache_path),
        "verification_workers": dataset.verification_workers,
        "sampler_groups": len(dataset.sampler_group_sizes),
        "batch_size": len(indices),
        "initialization_seconds": initialized - started,
        "vocabulary_seconds": vocabulary_built - initialized,
        "first_batch_seconds": batch_built - vocabulary_built,
        "total_seconds": batch_built - started,
        "vocabulary_size": vocabulary.size,
        "initialization_memory": initialization_memory,
        "vocabulary_memory": vocabulary_memory,
        "batch_memory": batch_memory,
        "after_clear_memory": _memory_report(),
        "cache_before_clear": cache_before_clear,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-cached-shards", type=int, default=1)
    parser.add_argument(
        "--verification-workers",
        type=int,
        help="cold verification processes (default: all logical CPUs for large corpora)",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.verification_workers is not None and args.verification_workers <= 0:
        parser.error("--verification-workers must be positive")
    result = benchmark(
        args.manifest,
        batch_size=args.batch_size,
        seed=args.seed,
        max_cached_shards=args.max_cached_shards,
        verification_workers=args.verification_workers,
        progress=args.progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

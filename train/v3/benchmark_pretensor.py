"""Measure startup and batch collation of a pre-tensorized V3 artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .benchmark_loader import _memory_report
from .pretensor import PretensorizedReplayDatasetV3
from .replay import DeterministicSamplerV3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-cached-frames", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batches <= 0:
        parser.error("batch size and batch count must be positive")

    started = time.perf_counter()
    dataset = PretensorizedReplayDatasetV3(
        args.manifest, max_cached_frames=args.max_cached_frames
    )
    initialized = time.perf_counter()
    sampler = DeterministicSamplerV3(
        len(dataset), args.seed, dataset.sampler_group_sizes
    )
    batch_seconds = []
    samples = 0
    for _ in range(args.batches):
        before = time.perf_counter()
        indices = sampler.take(min(args.batch_size, len(dataset)))
        sampler.advance_epoch_if_complete()
        batch = dataset.collate_indices(indices)
        batch_seconds.append(time.perf_counter() - before)
        samples += batch.inputs.batch_size
    finished = time.perf_counter()
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "examples": len(dataset),
                "frames": len(dataset.frames),
                "initialization_seconds": initialized - started,
                "first_batch_seconds": batch_seconds[0],
                "mean_batch_seconds": sum(batch_seconds) / len(batch_seconds),
                "batches": len(batch_seconds),
                "samples": samples,
                "samples_per_second": samples / sum(batch_seconds),
                "total_seconds": finished - started,
                "memory": _memory_report(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

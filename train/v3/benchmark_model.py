"""Reproducible batch-one latency probe for the first V3 model."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import fields

import torch

from .model import AugmentNetV3, V3ModelConfig
from .schema import load_golden_suite
from .tensorize import V3Vocabulary, tensorize_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="runtime_rich")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.threads <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error(
            "threads and iterations must be positive; warmup must be non-negative"
        )

    torch.set_num_threads(args.threads)
    suite = load_golden_suite()
    case = next((case for case in suite["cases"] if case["name"] == args.case), None)
    if case is None:
        parser.error(f"unknown fixture case {args.case!r}")
    vocabulary = V3Vocabulary.from_golden_suite(suite)
    device = torch.device(args.device)
    batch = tensorize_cases([case], vocabulary, device=device)
    config = V3ModelConfig(text_vocabulary_size=vocabulary.size)
    torch.manual_seed(20260821)
    model = AugmentNetV3(config).to(device).eval()

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(batch)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            model(batch)
            _synchronize(device)
            samples.append((time.perf_counter_ns() - started) / 1_000_000)

    samples.sort()
    report = {
        "schema_id": "augmentia.v3-model-benchmark.v1",
        "case": args.case,
        "device": str(device),
        "threads": args.threads,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "parameters": model.parameter_count(),
        "parameter_bytes_fp32": model.parameter_count() * 4,
        "batch_tensor_bytes": sum(
            getattr(batch, field.name).numel()
            * getattr(batch, field.name).element_size()
            for field in fields(batch)
        ),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "vocabulary_size": vocabulary.size,
        "pieces": int(batch.piece_mask.sum()),
        "cards": int(batch.card_mask.sum()),
        "runtime_nodes": int(batch.runtime_mask.sum()),
        "legal_actions": int(batch.action_mask.sum()),
        "iterations": args.iterations,
        "latency_ms": {
            "min": samples[0],
            "p50": _percentile(samples, 0.50),
            "p95": _percentile(samples, 0.95),
            "max": samples[-1],
        },
    }
    print(json.dumps(report, indent=2))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    index = round((len(sorted_values) - 1) * percentile)
    return sorted_values[index]


if __name__ == "__main__":
    main()

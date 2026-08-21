"""Export an ephemeral V3 model and benchmark reusable Python/Rust ONNX paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .model import AugmentNetV3, V3ModelConfig
from .onnx_export import export_onnx_v3
from .onnx_runtime import OnnxRuntimeV3, benchmark_runtime
from .replay import DatasetIdentity
from .schema import load_golden_suite
from .tensorize import V3Vocabulary, tensorize_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--soak", type=int, default=1_000)
    parser.add_argument("--rust", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parents[2]
    suite = load_golden_suite()
    vocabulary = V3Vocabulary.from_golden_suite(suite)
    case = next(item for item in suite["cases"] if item["name"] == "standard_playing")
    ruleset_manifest = root / "rulesets" / suite["ruleset_id"] / "manifest.json"
    ruleset_hash = hashlib.sha256(ruleset_manifest.read_bytes()).hexdigest()
    identity = DatasetIdentity(
        manifest_path="benchmark-only",
        dataset_id="benchmark-only",
        manifest_sha256="0" * 64,
        shard_sha256=(),
        ruleset_id=suite["ruleset_id"],
        ruleset_manifest_sha256=ruleset_hash,
    )
    model = AugmentNetV3(V3ModelConfig(text_vocabulary_size=vocabulary.size))
    sample = tensorize_cases([case], vocabulary)
    with tempfile.TemporaryDirectory() as directory:
        model_path, metadata_path = export_onnx_v3(
            model, sample, Path(directory) / "benchmark.onnx", vocabulary, identity
        )
        runtime = OnnxRuntimeV3(
            model_path,
            metadata_path,
            expected_ruleset_id=suite["ruleset_id"],
            expected_ruleset_manifest_sha256=ruleset_hash,
        )
        python_first = runtime.evaluate_cases([case])[0]
        report: dict[str, object] = {
            "python": benchmark_runtime(
                runtime,
                [case],
                iterations=args.iterations,
            )
        }
        if args.rust:
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
                [
                    "cargo",
                    "run",
                    "-q",
                    "-p",
                    "augment-cli",
                    "--bin",
                    "benchmark_v3_runtime",
                    "--",
                    "--model",
                    str(model_path),
                    "--metadata",
                    str(metadata_path),
                    "--ruleset-hash",
                    ruleset_hash,
                    "--iterations",
                    str(args.iterations),
                    "--soak",
                    str(args.soak),
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"Rust runtime exited {completed.returncode}: {completed.stderr}"
                )
            rust = json.loads(completed.stdout)
            np.testing.assert_allclose(
                rust["probabilities"], python_first.probabilities, rtol=2e-4, atol=2e-5
            )
            np.testing.assert_allclose(
                rust["value"], python_first.value, rtol=2e-4, atol=2e-5
            )
            report["rust"] = rust
            report["cross_language_equivalent"] = True
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# V4 training base seal — 2026-09-19

This document seals the source baseline used to begin real V4 model training.

## Identity

- Base commit before seal: `118660f62c072b4c204d0e5c5ea42a7bcbbde61f`
- Seal tag: `ai/v4-training-base-20260919`
- Ruleset: `augment_chess_20260915`
- Ruleset manifest SHA-256: `7b2736f8e9f727db818d8c8f4aaf7742a23a717e2f31d749ea293a3dfc051b13`
- Captured upstream JS SHA-256: `60d5816e78896d4b5371c0d5cc37ef0a302aa0cc814c3b027be3936124878dfc`

The 0915 ruleset manifest still declares `coverage.status = partial`. Therefore this is an AI/training source seal, not a `ruleset/augment_chess_20260915-rN` release tag.

## Included V4 baseline

- Factorized V4 action grammar and replay target
- V4 factorized MCTS with public-information chance sampling
- Split ONNX runtime with lazy prefix scoring
- Baseline and opt-in Hybrid state encoders
- V4 Python replay parser, batch collation, loss, trainer, ONNX export, and tests
- 0915 fixtures and V4 fixture bundles
- 0915 ruleset/source snapshot and engine integration needed by this baseline
- Fixed-budget V4 benchmark and implementation/baseline documentation

## Verification gates

Executed on 2026-09-19 before the seal commit:

- `cargo fmt --all -- --check`: pass
- `cargo clippy --workspace --all-targets -- -D warnings`: pass
- `cargo test --workspace --quiet` with
  `ORT_DYLIB_PATH=D:/augmentia/train/.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll`: pass
- `python -m unittest discover -s . -p "test_*.py" -q` from `train/`: 101 tests, pass
- `git diff --check`: pass

Python ONNX tests still emit the documented legacy exporter/tracing deprecation warnings; they do not fail the gate.

## Training rule

All V4 training datasets, checkpoints, exported bundles, and arena results derived from this baseline must record the seal tag and the exact seal commit. Large training artifacts remain outside Git and are identified by SHA-256 plus provenance metadata.

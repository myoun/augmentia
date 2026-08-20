# Training workspace

This directory intentionally retains only the minimum V2 training architecture
needed to compare the current baseline with the future ObservationV3 pipeline:

- `model.py`: V2 fixed-policy-head network definition;
- `dataset.py`: V2 dataset loader and symmetry behavior;
- `train.py`: V2 training entry point;
- `export_onnx.py`: V2 ONNX export path.

Generated datasets, checkpoints, rejected candidates, calibration reports, and
one-off experiment scripts were removed on 2026-08-21. They are invalid inputs
for ObservationV3 and can be regenerated only for explicit historical work.

New training work should use versioned V3 observation and action descriptors,
terminal-aware outcomes, and engine-native legal actions. Do not extend the V2
JSONL or 32,768-action format.

See `docs/ai-v2-baseline.md` for the preserved baseline and migration policy.

The new `v3/` package contains the Rust-authored ObservationV3/ActionV3 golden fixture and the
strict Python schema validator. It is the only supported starting point for new V3 training code.
